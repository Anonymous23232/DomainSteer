"""Tests for direction extraction — synthetic activations, no model download."""

import json

import numpy as np
import pytest
import torch

pytest.importorskip("torch", exc_type=ImportError)

from domainsteer.extract import (DEFAULT_ESTIMATOR, DirectionExtractor,
                                 ExtractionResult, middle_layers,
                                 pair_chat_messages)
from domainsteer.pairs import ContrastivePair

DIM = 32


def test_pair_chat_messages_wraps_stem_and_answer():
    messages = pair_chat_messages(
        "Aquifers store groundwater.",
        "What do outsiders most often get wrong about this field?",
    )
    assert messages[0]["role"] == "system"
    assert "1-2 short sentences" in messages[0]["content"]
    assert "hydrology" not in messages[0]["content"].lower()
    assert messages[1] == {
        "role": "user",
        "content": "What do outsiders most often get wrong about this field?",
    }
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Aquifers store groundwater."


def test_pair_chat_messages_without_stem():
    messages = pair_chat_messages("Just the answer.")
    assert [m["role"] for m in messages] == ["system", "assistant"]


def test_render_chat_folds_system_role_when_template_rejects_it():
    """Gemma 2 raises on role=system. The instructions move into the user turn."""
    from domainsteer.extract import render_chat

    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            del tokenize
            self.calls.append([m["role"] for m in messages])
            if any(m["role"] == "system" for m in messages):
                raise RuntimeError("System role not supported")
            assert messages[0]["role"] == "user"
            assert "1-2 short sentences" in messages[0]["content"]
            assert "What is a strain?" in messages[0]["content"]
            return "folded" + (" gen" if add_generation_prompt else "")

    tokenizer = FakeTokenizer()
    messages = [
        {"role": "system", "content": "Answer in 1-2 short sentences."},
        {"role": "user", "content": "What is a strain?"},
    ]
    assert render_chat(tokenizer, messages, add_generation_prompt=True) == "folded gen"
    assert render_chat(tokenizer, messages, add_generation_prompt=True) == "folded gen"
    assert tokenizer.calls[0] == ["system", "user"]
    assert tokenizer.calls[1:] == [["user"], ["user"]]


def test_template_ids_encodes_chat_string():
    """apply_chat_template may return a string; ids must still be integers."""

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 0

        def apply_chat_template(self, messages, tokenize=False,
                                add_generation_prompt=False):
            del messages, tokenize, add_generation_prompt
            return "<chat>hello</chat>"

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [11, 22, 33] if text else []

    extractor = object.__new__(DirectionExtractor)
    extractor.tokenizer = FakeTokenizer()
    ids = extractor._template_ids([{"role": "user", "content": "hi"}])
    assert ids == [11, 22, 33]
    torch.tensor(ids, dtype=torch.long)


def test_middle_layers_spans_middle_third():
    assert middle_layers(28) == [10, 12, 14, 16, 18]
    assert middle_layers(32) == [10, 12, 14, 16, 18, 20]
    for n in (12, 24, 40):
        layers = middle_layers(n)
        assert all(n // 3 <= layer <= 2 * n // 3 for layer in layers)


def _extractor_with_planted_direction(monkeypatch, planted, noise=0.1, seed=2):
    """Fake activations: positive/negative differ by ±planted plus noise."""
    rng = np.random.default_rng(seed)

    def fake(self, pair, layers, **kwargs):
        base = rng.normal(size=DIM)
        pos = base + planted + rng.normal(scale=noise, size=DIM)
        neg = base - planted + rng.normal(scale=noise, size=DIM)
        return {layer: (pos, neg) for layer in layers}

    monkeypatch.setattr(DirectionExtractor, "_pair_last_token_hidden", fake)
    return object.__new__(DirectionExtractor)


def test_extract_recovers_planted_direction(monkeypatch):
    rng = np.random.default_rng(1)
    planted = rng.normal(size=DIM)
    planted /= np.linalg.norm(planted)

    extractor = _extractor_with_planted_direction(monkeypatch, planted)
    pairs = [ContrastivePair(f"pos {i}", f"neg {i}") for i in range(50)]
    result = extractor.extract(pairs, layers=[3, 5])

    for layer in (3, 5):
        direction = result.directions[layer]
        assert np.linalg.norm(direction) == pytest.approx(1.0, abs=1e-6)
        assert float(direction @ planted) > 0.95          # recovered the signal
        assert result.holdout_accuracy[layer] == 1.0      # clean separation
    assert result.n_train_pairs + result.n_holdout_pairs == 50
    assert result.n_holdout_pairs == 10


def test_extract_rejects_too_few_pairs():
    extractor = object.__new__(DirectionExtractor)
    with pytest.raises(ValueError, match="at least"):
        extractor.extract([ContrastivePair("a", "b")], layers=[0])


def test_noise_only_gives_chance_accuracy(monkeypatch):
    extractor = _extractor_with_planted_direction(
        monkeypatch, planted=np.zeros(DIM), noise=1.0
    )
    pairs = [ContrastivePair(f"pos {i}", f"neg {i}") for i in range(100)]
    result = extractor.extract(pairs, layers=[0])
    assert 0.1 <= result.holdout_accuracy[0] <= 0.9   # no planted signal → ~chance


def test_result_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    result = ExtractionResult(
        model_name="test/model",
        directions={4: rng.normal(size=DIM), 6: rng.normal(size=DIM)},
        holdout_accuracy={4: 0.9, 6: 0.75},
        n_train_pairs=40,
        n_holdout_pairs=10,
    )
    result.save(tmp_path)
    loaded = ExtractionResult.load(tmp_path)

    assert loaded.model_name == "test/model"
    assert loaded.holdout_accuracy == {4: 0.9, 6: 0.75}
    assert loaded.n_train_pairs == 40
    for layer in (4, 6):
        np.testing.assert_allclose(loaded.directions[layer], result.directions[layer])


def test_result_roundtrips_estimator_and_margin(tmp_path):
    rng = np.random.default_rng(5)
    result = ExtractionResult(
        model_name="test/model",
        directions={4: rng.normal(size=DIM)},
        holdout_accuracy={4: 1.0},
        n_train_pairs=40,
        n_holdout_pairs=10,
        estimator="rfm",
        holdout_margin={4: 2.75},
    )
    result.save(tmp_path)
    loaded = ExtractionResult.load(tmp_path)

    assert loaded.estimator == "rfm"
    assert loaded.holdout_margin == {4: 2.75}


def test_meta_without_estimator_still_loads(tmp_path):
    """Directories written before estimators existed must keep loading, and
    must read back as the default rather than failing."""
    rng = np.random.default_rng(6)
    np.save(tmp_path / "direction_layer_4.npy", rng.normal(size=DIM))
    legacy = {
        "model_name": "test/model",
        "layers": [4],
        "holdout_accuracy": {"4": 0.9},
        "n_train_pairs": 40,
        "n_holdout_pairs": 10,
    }
    (tmp_path / "meta.json").write_text(json.dumps(legacy), encoding="utf-8")

    loaded = ExtractionResult.load(tmp_path)
    assert loaded.estimator == DEFAULT_ESTIMATOR
    assert loaded.holdout_margin == {}


def test_extract_rejects_unknown_estimator():
    extractor = object.__new__(DirectionExtractor)
    pairs = [ContrastivePair(f"p{i}", f"n{i}") for i in range(20)]
    with pytest.raises(ValueError, match="Unknown estimator"):
        extractor.extract(pairs, layers=[0], estimator="nope")


def test_extract_records_estimator_and_margin(monkeypatch):
    rng = np.random.default_rng(1)
    planted = rng.normal(size=DIM)
    planted /= np.linalg.norm(planted)

    extractor = _extractor_with_planted_direction(monkeypatch, planted)
    pairs = [ContrastivePair(f"pos {i}", f"neg {i}") for i in range(50)]
    result = extractor.extract(pairs, layers=[3], estimator="rfm")

    assert result.estimator == "rfm"
    assert result.holdout_margin[3] > 1.0
    assert np.linalg.norm(result.directions[3]) == pytest.approx(1.0, abs=1e-8)
