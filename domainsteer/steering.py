"""Norm-relative activation steering at generation time.

Steering strength `alpha` is a fraction of each token position's own
activation magnitude: the hook adds ``alpha * ||h|| * direction`` to the
output of one decoder block. ``alpha = 0.05`` therefore means "push 5% of the
activation's norm toward the direction" and is comparable across models,
layers, and token positions. Typical useful range 0-0.25; values above 1.0
are rejected as almost certainly a unit mistake.

By default the shift is applied at every generated token. Pass
``steer_new_tokens=N`` to steer only the prefill (first-token distribution)
plus the first N new tokens, then drop the intervention — the model often
needs the direction to pick a sense, not to keep pushing every later token
toward the same region.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Optional, Sequence, Union

import numpy as np
import torch

from domainsteer.extract import get_decoder_layers

logger = logging.getLogger(__name__)

# Stop decoding once this fraction of token 3-grams are repeats (T35S loops).
REPEAT_STOP_THRESHOLD = 0.45
REPEAT_STOP_NGRAM = 3
REPEAT_STOP_MIN_NEW = 16

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer in 1-2 short sentences. Do not use lists."
)
# 1-2 sentences. 96 tokens lets a scientific clause finish; 48 clipped mid-word.
DEFAULT_MAX_NEW_TOKENS = 96

_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*\s*$")
_SENTENCE_END_ANY = re.compile(r"[.!?…][\"'”’)\]]*")


def answer_complete(text: str) -> bool:
    """True when the completion ends on sentence punctuation, not a clip."""
    return bool(_SENTENCE_END.search((text or "").strip()))


def trim_hanging_clause(text: str) -> str:
    """Drop a trailing fragment after the last . ? ! so a token cap is not visible."""
    text = (text or "").strip()
    if not text or answer_complete(text):
        return text
    matches = list(_SENTENCE_END_ANY.finditer(text))
    if not matches:
        return text
    return text[:matches[-1].end()].strip()

# Fallback chat template for tokenizers that ship without one
_FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def steered_hidden(hidden_states: torch.Tensor, alpha: float,
                   direction: torch.Tensor) -> torch.Tensor:
    """Pure steering math: shift each position by alpha x its own norm."""
    norms = hidden_states.norm(dim=-1, keepdim=True)
    return hidden_states + alpha * norms * direction


class SteerWindow:
    """Steer the prefill (first-token distribution) plus the first N new tokens.

    ``limit=None`` keeps the residual on for the whole completion (old
    behaviour). ``limit=0`` is prompt-only: the first generated token is
    still biased, then the hook is idle. Continued steering after the
    sense is chosen is what turns "a construct is DNA..." into T35S loops.
    """

    def __init__(self, limit: Optional[int] = None):
        if limit is not None and limit < 0:
            raise ValueError(f"steer_new_tokens must be >= 0, got {limit}.")
        self.limit = limit
        self.prompt_len: Optional[int] = None
        self.generated = 0

    def observe(self, seq_len: int) -> bool:
        """Call once per forward; True if this pass should apply the shift."""
        if seq_len < 1:
            return False
        if self.prompt_len is None:
            self.prompt_len = seq_len
            return True
        if seq_len <= 1:
            self.generated += 1
        else:
            self.generated = max(self.generated, seq_len - self.prompt_len)
        if self.limit is None:
            return True
        return self.generated <= self.limit


class ActivationSteering:
    """Steer a loaded model's generations along one direction at one layer."""

    def __init__(self, model, tokenizer,
                 direction: Union[np.ndarray, torch.Tensor], layer: int,
                 system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.system_prompt = system_prompt
        self.device = next(model.parameters()).device

        decoder_layers = get_decoder_layers(model)
        if not 0 <= layer < len(decoder_layers):
            raise ValueError(f"Layer {layer} out of range (model has {len(decoder_layers)}).")
        self._target_module = decoder_layers[layer]

        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction)
        self.direction = direction.to(device=self.device, dtype=torch.float32)
        self._direction_cache: dict = {}

        if tokenizer.chat_template is None:
            tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

    def _direction_like(self, hidden: torch.Tensor) -> torch.Tensor:
        """The direction on `hidden`'s device and dtype.

        With ``device_map="auto"`` the model is sharded across GPUs, so the
        target block's activations need not be on the model's input device —
        the hook's own tensor is the only reliable placement. Copies are
        cached per (device, dtype) so this costs nothing per token.
        """
        key = (hidden.device, hidden.dtype)
        cached = self._direction_cache.get(key)
        if cached is None:
            cached = self.direction.to(device=hidden.device, dtype=hidden.dtype)
            self._direction_cache[key] = cached
        return cached

    def generate(self, prompt: str, alpha: float = 0.0,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 steer_new_tokens: Optional[int] = None) -> str:
        if abs(alpha) > 1.0:
            raise ValueError(
                f"alpha={alpha} out of range: alpha is norm-relative "
                "(typical 0-0.25), not an absolute magnitude."
            )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        from domainsteer.extract import render_chat
        formatted = render_chat(
            self.tokenizer, messages, add_generation_prompt=True,
        )
        inputs = self.tokenizer([formatted], return_tensors="pt").to(self.device)
        prompt_len = int(inputs.input_ids.shape[1])
        handles = self._register_hooks(alpha, steer_new_tokens) if alpha != 0.0 else []

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        stop = _token_repeat_stop(prompt_len)
        if stop is not None:
            gen_kwargs["stopping_criteria"] = stop

        try:
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, **gen_kwargs)
            new_ids = out_ids[0][prompt_len:]
            decoded = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            return trim_hanging_clause(decoded)
        finally:
            for handle in handles:
                handle.remove()

    def _register_hooks(self, alpha: float, steer_new_tokens: Optional[int]) -> list:
        return [_add_hook(self._target_module, alpha, SteerWindow(steer_new_tokens),
                          self._direction_like)]


class MultiLayerSteering(ActivationSteering):
    """Steer at several layers at once, each along its own direction.

    Every layer gets ``alpha * ||h|| * direction[L]`` on its own output, so the
    same alpha adds up across layers: with directions that agree, n layers at
    alpha push roughly as hard as one layer at n * alpha.
    """

    def __init__(self, model, tokenizer,
                 directions: dict, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        layers = sorted(directions)
        super().__init__(model, tokenizer, directions[layers[0]], layers[0],
                         system_prompt=system_prompt)
        self.layers = layers
        self._per_layer = {
            L: ActivationSteering(model, tokenizer, directions[L], L, system_prompt)
            for L in layers}

    def _register_hooks(self, alpha: float, steer_new_tokens: Optional[int]) -> list:
        return [_add_hook(s._target_module, alpha, SteerWindow(steer_new_tokens),
                          s._direction_like)
                for s in self._per_layer.values()]


def _add_hook(module, alpha: float, window: SteerWindow, direction_like):
    """Forward hook adding alpha x ||h|| x direction to one block's output."""
    def hook(module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if not window.observe(int(hidden.shape[1])):
            return output
        steered = steered_hidden(hidden, alpha, direction_like(hidden))
        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered
    return module.register_forward_hook(hook)


def token_ngram_repeat_ratio(tokens: Sequence, n: int = 3) -> float:
    """Fraction of n-grams that are extra copies of an earlier n-gram (0-1)."""
    if len(tokens) < 2 * n:
        return 0.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


def _token_repeat_stop(prompt_len: int):
    """Halt when generated token 3-grams start looping. None if transformers
    is too old to take StoppingCriteria."""
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except ImportError:
        return None

    class TokenRepeatStop(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            del scores, kwargs
            new = input_ids[0, prompt_len:].tolist()
            if len(new) < REPEAT_STOP_MIN_NEW:
                return False
            return token_ngram_repeat_ratio(
                new, REPEAT_STOP_NGRAM) >= REPEAT_STOP_THRESHOLD

    return StoppingCriteriaList([TokenRepeatStop()])


def complete_unsteered(model, tokenizer, prompt: str,
                       max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    """Same chat wrapper as steered generate, with no residual shift."""
    hidden = int(getattr(model.config, "hidden_size", 0) or 8)
    dummy = torch.zeros(hidden)
    return ActivationSteering(model, tokenizer, dummy, 0).generate(
        prompt, alpha=0.0, max_new_tokens=max_new_tokens)
