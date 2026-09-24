"""Direction extraction: contrastive pairs → per-layer domain directions.

Each pair is probed in the same chat template used at generation time
(system + unnamed stem + assistant answer). Hidden states are mean-pooled
over the assistant span, not taken at the final special token and not
conditioned on a domain dump that will be absent when steering. A
*direction estimator* then turns the training split's two activation clouds
into one unit vector, and the held-out split measures how well that vector
separates in-domain from field-neutral activations.

Two estimators are available, both returning a unit vector so everything
downstream (layer sweep, calibration, the steering hook) is unaffected:

``diff_means`` (default)
    Normalized difference of means. A first-moment statistic: it asks where
    the two clouds' centres sit. Optimal only when both classes share an
    isotropic covariance, and it can only ever return one direction.

``rfm``
    Recursive Feature Machine (Radhakrishnan et al., Science 2024). Fits a
    kernel machine to predict the expert/non-expert label, then takes the top
    eigenvector of its Average Gradient Outer Product — the directions in
    which the learned function actually changes. Captures dependence that is
    nonlinear or spread over a subspace, which a difference of means cannot.

Layer indexing: `hidden_states[L + 1]` is the *output* of decoder block L —
the same tensor a steering hook registered on that block modifies. Extraction
and steering therefore always refer to the same layer L.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

# The estimator math lives in `directions`, which stays free of torch so it
# can be exercised without a model. Re-exported here so existing imports of
# `domainsteer.extract` keep working.
from domainsteer.directions import (  # noqa: F401
    DEFAULT_ESTIMATOR,
    ESTIMATORS,
    difference_of_means,
    direction_cosine,
    rfm_agop,
    separation_margin,
    split_by_concept,
)
from domainsteer.pairs import ContrastivePair

logger = logging.getLogger(__name__)

MIN_PAIRS = 10


def messages_without_system_role(messages: List[dict]) -> List[dict]:
    """Move system text into the first user turn.

    Gemma 2's chat template raises ``System role not supported``. The same
    instructions still have to reach the model, so they are prefixed onto the
    user message. Other roles stay in order.
    """
    system_parts = []
    rest = []
    for message in messages:
        if message.get("role") == "system":
            text = str(message.get("content") or "").strip()
            if text:
                system_parts.append(text)
            continue
        rest.append({"role": message.get("role"), "content": message.get("content", "")})
    if not system_parts:
        return [{"role": m.get("role"), "content": m.get("content", "")} for m in messages]
    prefix = "\n\n".join(system_parts)
    for i, message in enumerate(rest):
        if message["role"] == "user":
            user = str(message["content"] or "").strip()
            rest[i] = {
                "role": "user",
                "content": f"{prefix}\n\n{user}" if user else prefix,
            }
            return rest
    rest.insert(0, {"role": "user", "content": prefix})
    return rest


def render_chat(tokenizer, messages: List[dict],
                add_generation_prompt: bool = False) -> str:
    """Render chat turns. If the template rejects a system role, fold it in."""
    supports = getattr(tokenizer, "_domainsteer_system_role", None)
    prepared = messages if supports is not False else messages_without_system_role(messages)
    try:
        text = tokenizer.apply_chat_template(
            prepared, tokenize=False, add_generation_prompt=add_generation_prompt,
        )
    except Exception as exc:
        if supports is False or "system role" not in str(exc).lower():
            raise
        tokenizer._domainsteer_system_role = False
        text = tokenizer.apply_chat_template(
            messages_without_system_role(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    else:
        if supports is None and any(m.get("role") == "system" for m in messages):
            tokenizer._domainsteer_system_role = True
    return text


def pair_chat_messages(answer: str, stem: Optional[str] = None,
                       system_prompt: Optional[str] = None
                       ) -> List[dict]:
    """Chat turns matching generation: system + unnamed stem + assistant answer.

    Both sides of a pair share the system prompt and stem; only the assistant
    answer differs. That is the contrast that has to transfer to steered
    generation, which uses the same wrapper and never sees a domain dump.
    """
    if system_prompt is None:
        from domainsteer.steering import DEFAULT_SYSTEM_PROMPT
        system_prompt = DEFAULT_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system_prompt}]
    if stem and stem.strip():
        messages.append({"role": "user", "content": stem.strip()})
    messages.append({"role": "assistant", "content": answer})
    return messages


def load_model(model_name: str, device_map: str = "auto"):
    """Load a causal LM + tokenizer ready for probing (bfloat16, padded)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device_map, dtype=torch.bfloat16
    )
    model.eval()
    return model, tokenizer


def get_decoder_layers(model):
    """Return the decoder block list, or raise a clear error for unsupported
    architectures."""
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            f"Unsupported architecture {type(model).__name__}: expected "
            "decoder blocks at model.model.layers (Llama/Mistral/Qwen-style)."
        )
    return layers


def middle_layers(num_hidden_layers: int) -> List[int]:
    """Even layers in the model's middle third — where persona/style steering
    is empirically most effective."""
    start = num_hidden_layers // 3
    end = 2 * num_hidden_layers // 3
    return [layer for layer in range(start, end + 1) if layer % 2 == 0]


@dataclass
class ExtractionResult:
    model_name: str
    directions: Dict[int, np.ndarray]        # layer → unit vector (hidden_dim,)
    holdout_accuracy: Dict[int, float]       # layer → separation accuracy
    n_train_pairs: int
    n_holdout_pairs: int
    # Defaulted so directories written before estimators existed still load.
    estimator: str = DEFAULT_ESTIMATOR
    holdout_margin: Dict[int, float] = field(default_factory=dict)

    def save(self, directory: Union[str, Path]) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for layer, direction in self.directions.items():
            np.save(directory / f"direction_layer_{layer}.npy", direction)
        meta = {
            "model_name": self.model_name,
            "layers": sorted(self.directions),
            "estimator": self.estimator,
            "holdout_accuracy": {str(k): v for k, v in self.holdout_accuracy.items()},
            "holdout_margin": {str(k): v for k, v in self.holdout_margin.items()},
            "n_train_pairs": self.n_train_pairs,
            "n_holdout_pairs": self.n_holdout_pairs,
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Saved {len(self.directions)} directions to {directory}")

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "ExtractionResult":
        directory = Path(directory)
        with open(directory / "meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        directions = {
            layer: np.load(directory / f"direction_layer_{layer}.npy")
            for layer in meta["layers"]
        }
        return cls(
            model_name=meta["model_name"],
            directions=directions,
            holdout_accuracy={int(k): v for k, v in meta["holdout_accuracy"].items()},
            n_train_pairs=meta["n_train_pairs"],
            n_holdout_pairs=meta["n_holdout_pairs"],
            estimator=meta.get("estimator", DEFAULT_ESTIMATOR),
            holdout_margin={int(k): v
                            for k, v in meta.get("holdout_margin", {}).items()},
        )


class DirectionExtractor:
    """Extract expertise directions from a loaded model."""

    def __init__(self, model, tokenizer, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device if device is not None else next(model.parameters()).device
        from domainsteer.steering import _FALLBACK_CHAT_TEMPLATE
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = _FALLBACK_CHAT_TEMPLATE

    def extract(self, pairs: List[ContrastivePair],
                layers: Optional[List[int]] = None,
                holdout_fraction: float = 0.2,
                seed: int = 0,
                model_name: str = "",
                estimator: str = DEFAULT_ESTIMATOR) -> ExtractionResult:
        """One forward pass per pair; directions from the training split,
        separation accuracy and margin from the held-out split.

        `estimator` selects how the two activation clouds become a direction:
        `"diff_means"` (default, unchanged behaviour) or `"rfm"`.
        """
        if len(pairs) < MIN_PAIRS:
            raise ValueError(f"Need at least {MIN_PAIRS} pairs, got {len(pairs)}.")
        if estimator not in ESTIMATORS:
            raise ValueError(
                f"Unknown estimator '{estimator}'; choose from "
                f"{sorted(ESTIMATORS)}."
            )
        estimate = ESTIMATORS[estimator]
        if layers is None:
            layers = middle_layers(self.model.config.num_hidden_layers)

        logger.info(
            "Probing chat-formatted stem+answer (pooling assistant tokens; "
            "no domain prefix)"
        )

        expert_acts: Dict[int, list] = {layer: [] for layer in layers}
        nonexpert_acts: Dict[int, list] = {layer: [] for layer in layers}
        for pair in tqdm(pairs, desc="Probing activations"):
            per_layer = self._pair_last_token_hidden(pair, layers)
            for layer, (expert_vec, nonexpert_vec) in per_layer.items():
                expert_acts[layer].append(expert_vec)
                nonexpert_acts[layer].append(nonexpert_vec)

        holdout_idx, train_idx = split_by_concept(
            [pair.concept for pair in pairs], holdout_fraction, seed
        )

        directions: Dict[int, np.ndarray] = {}
        holdout_accuracy: Dict[int, float] = {}
        holdout_margin: Dict[int, float] = {}
        for layer in layers:
            expert = np.stack(expert_acts[layer])
            nonexpert = np.stack(nonexpert_acts[layer])

            try:
                direction = estimate(expert[train_idx], nonexpert[train_idx])
            except ValueError as e:
                raise ValueError(f"{e} at layer {layer}.") from e

            separated = ((expert[holdout_idx] @ direction)
                         > (nonexpert[holdout_idx] @ direction))
            accuracy = float(separated.mean())
            margin = separation_margin(expert[holdout_idx],
                                       nonexpert[holdout_idx], direction)
            directions[layer] = direction
            holdout_accuracy[layer] = accuracy
            holdout_margin[layer] = margin
            logger.info(f"Layer {layer}: holdout separation {accuracy * 100:.1f}% "
                        f"(margin {margin:.2f})")

        return ExtractionResult(
            model_name=model_name,
            directions=directions,
            holdout_accuracy=holdout_accuracy,
            n_train_pairs=len(train_idx),
            n_holdout_pairs=len(holdout_idx),
            estimator=estimator,
            holdout_margin=holdout_margin,
        )

    def _template_ids(self, messages: List[dict],
                      add_generation_prompt: bool = False) -> List[int]:
        # Format then encode. Some tokenizer builds ignore tokenize=True on
        # apply_chat_template and return a string; torch.tensor(that) raises
        # "too many dimensions 'str'".
        text = render_chat(
            self.tokenizer, messages,
            add_generation_prompt=add_generation_prompt,
        )
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return [int(i) for i in ids]

    def _pair_last_token_hidden(
            self, pair: ContrastivePair, layers: List[int]
            ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Chat-formatted pair in one padded batch; mean-pool the assistant
        span per layer (the tokens that actually carry the answer)."""
        from domainsteer.steering import DEFAULT_SYSTEM_PROMPT

        stem = (pair.concept or pair.framing or "").strip()
        prompt_messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        ]
        if stem:
            prompt_messages.append({"role": "user", "content": stem})
        prompt_len = len(self._template_ids(prompt_messages,
                                            add_generation_prompt=True))

        id_lists = [
            self._template_ids(pair_chat_messages(
                pair.expert_text, stem, DEFAULT_SYSTEM_PROMPT)),
            self._template_ids(pair_chat_messages(
                pair.nonexpert_text, stem, DEFAULT_SYSTEM_PROMPT)),
        ]
        max_len = max(len(ids) for ids in id_lists)
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full((2, max_len), pad_id, dtype=torch.long,
                               device=self.device)
        attention_mask = torch.zeros((2, max_len), dtype=torch.long,
                                     device=self.device)
        for i, ids in enumerate(id_lists):
            input_ids[i, :len(ids)] = torch.tensor(
                ids, dtype=torch.long, device=self.device)
            attention_mask[i, :len(ids)] = 1

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             output_hidden_states=True)

        lengths = attention_mask.sum(dim=1)
        result = {}
        for layer in layers:
            hs = out.hidden_states[layer + 1]
            vecs = []
            for i in range(2):
                end = int(lengths[i].item())
                start = min(prompt_len, max(end - 1, 0))
                vecs.append(hs[i, start:end].mean(dim=0).to(torch.float32).cpu().numpy())
            result[layer] = (vecs[0], vecs[1])
        return result
