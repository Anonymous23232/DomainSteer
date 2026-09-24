"""Tests for the DomainSteerer API — config handling and dial mapping, no models."""

import json

import numpy as np
import pytest

from domainsteer.clusters import Domain
from domainsteer.steerer import DomainSteerer


class FakeSteering:
    def generate(self, prompt, alpha=0.0, max_new_tokens=256, **kwargs):
        return f"gen(alpha={alpha})"


def make_built_steerer(tmp_path, max_alpha=0.2):
    """A steerer whose cache contains a valid config + direction file."""
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    directions_dir = steerer.model_dir / "directions"
    directions_dir.mkdir(parents=True)
    np.save(directions_dir / "direction_layer_10.npy", np.ones(8))
    config = {
        "domain": "hydrology",
        "model_name": "test/model",
        "layer": 10,
        "max_alpha": max_alpha,
        "direction_file": "directions/direction_layer_10.npy",
        "steering_scale": "hidden_norm_relative",
    }
    steerer.config_path.write_text(json.dumps(config))

    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    steerer._steering = FakeSteering()
    return steerer


def test_unbuilt_steerer_refuses_generation(tmp_path):
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    assert steerer.layer is None
    with pytest.raises(ValueError, match="build"):
        steerer.generate("hi", expertise=1.0)


def test_config_loads_on_construction(tmp_path):
    steerer = make_built_steerer(tmp_path)
    assert steerer.layer == 10
    assert steerer.max_alpha == 0.2
    assert steerer._direction is not None


def test_use_layer_switches_cached_direction(tmp_path):
    steerer = make_built_steerer(tmp_path)
    other = np.full(8, 2.0)
    np.save(steerer.directions_dir / "direction_layer_12.npy", other)
    assert steerer.extracted_layers() == [10, 12]
    steerer.use_layer(12)
    assert steerer.layer == 12
    assert np.allclose(steerer._direction, other)
    assert steerer._steering is None
    with pytest.raises(ValueError, match="No extracted direction"):
        steerer.use_layer(99)


def test_expertise_maps_to_calibrated_alpha(tmp_path):
    steerer = make_built_steerer(tmp_path, max_alpha=0.2)
    assert steerer.generate("q", expertise=0.5) == "gen(alpha=0.1)"
    assert steerer.generate("q", alpha=0.07) == "gen(alpha=0.07)"


def test_exactly_one_strength_argument(tmp_path):
    steerer = make_built_steerer(tmp_path)
    with pytest.raises(ValueError, match="exactly one"):
        steerer.generate("q")
    with pytest.raises(ValueError, match="exactly one"):
        steerer.generate("q", expertise=1.0, alpha=0.1)


def test_expertise_requires_calibration(tmp_path):
    steerer = make_built_steerer(tmp_path, max_alpha=None)
    with pytest.raises(ValueError, match="max_alpha"):
        steerer.generate("q", expertise=1.0)
    # raw alpha still works without calibration
    assert steerer.generate("q", alpha=0.05) == "gen(alpha=0.05)"


def test_compare_defaults_to_full_expertise(tmp_path):
    steerer = make_built_steerer(tmp_path, max_alpha=0.2)
    result = steerer.compare("q")
    assert result["baseline"] == "gen(alpha=0.0)"
    assert result["steered"] == "gen(alpha=0.2)"
    assert result["expertise"] == 1.0
    assert result["alpha"] == pytest.approx(0.2)


def test_unknown_steering_scale_invalidates_config(tmp_path):
    steerer = make_built_steerer(tmp_path)
    config = json.loads(steerer.config_path.read_text())
    config["steering_scale"] = "raw"
    steerer.config_path.write_text(json.dumps(config))

    reloaded = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    assert reloaded.layer is None


# ----------------------------------------------------------- pair resolution

def _fake_completion(prompt, api_key=None, model=None):
    from domainsteer.pairs import NEUTRAL_MARKER
    lay = NEUTRAL_MARKER in prompt
    concepts = [line[2:] for line in prompt.splitlines() if line.startswith("- ")]
    return {"explanations": {c: f"{'lay' if lay else 'expert'} take on {c}"
                             for c in concepts}}


def test_generated_pairs_record_their_concepts(tmp_path, monkeypatch):
    """Concepts passed explicitly must land in the cache, so a later rebuild
    doesn't need the original --concepts file."""
    import domainsteer.pairs as pairs_module
    monkeypatch.setattr(pairs_module, "request_json_completion", _fake_completion)

    from extras.concepts import load_concepts
    from domainsteer.pairs import PairGenerator

    gen = PairGenerator("hydrology", cache_dir=tmp_path)
    gen.generate(concepts=["baseflow", "infiltration"])

    assert gen.concepts_path.exists()
    assert [c.term for c in load_concepts(gen.concepts_path)] == [
        "baseflow", "infiltration"]
    from domainsteer.pairs import GENERIC_STEMS
    assert len(gen.generate(force_regenerate=True)) == len(GENERIC_STEMS)


def test_unbundled_model_answers_shared_stems_locally(tmp_path, monkeypatch):
    """A model with no shipped pairs answers the bundled 30 stems itself."""
    import domainsteer.steering as steering_mod

    steerer = DomainSteerer(
        "google/gemma-2-9b-it", "3106", cache_dir=tmp_path)
    steerer._model = type("M", (), {"config": type("C", (), {"hidden_size": 8})()})()
    steerer._tokenizer = object()
    monkeypatch.setattr(steerer, "_ensure_model", lambda: None)

    calls = []

    class FakeSteeringLocal:
        def __init__(self, model, tokenizer, direction, layer, system_prompt=None):
            self.system_prompt = system_prompt

        def generate(self, prompt, alpha=0.0, max_new_tokens=96, **kwargs):
            calls.append(alpha)
            kind = "expert" if "practitioner" in (self.system_prompt or "") else "default"
            return f"{kind} answer to {prompt}"

    monkeypatch.setattr(steering_mod, "ActivationSteering", FakeSteeringLocal)

    def boom(*args, **kwargs):
        raise AssertionError("API should not be called")

    monkeypatch.setattr("domainsteer.pairs.request_json_completion", boom)

    pairs = steerer._resolve_pairs(force=False)
    assert len(pairs) == 30
    assert pairs[0].expert_text.startswith("expert answer")
    assert pairs[0].nonexpert_text.startswith("default answer")
    assert calls and set(calls) == {0.0}
    assert steerer._model_pairs_path().is_file()
    n_calls = len(calls)
    assert steerer._resolve_pairs(force=False)[0].concept == pairs[0].concept
    assert len(calls) == n_calls


def test_domain_concepts_drive_pair_generation(tmp_path, monkeypatch):
    import domainsteer.pairs as pairs_module
    monkeypatch.setattr(pairs_module, "request_json_completion", _fake_completion)

    steerer = DomainSteerer(
        "test/model",
        Domain("hydrology", concepts=["baseflow", "infiltration"]),
        cache_dir=tmp_path,
    )
    pairs = steerer._resolve_pairs(force=False)
    from domainsteer.pairs import GENERIC_STEMS
    assert [p.concept for p in pairs] == GENERIC_STEMS


# ------------------------------------------------------- estimator isolation

def test_default_estimator_keeps_the_original_paths(tmp_path):
    """The whole point of the suffix scheme: adding estimators must not move
    or invalidate anything already built."""
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)

    assert steerer.estimator == "diff_means"
    assert steerer.directions_dir == steerer.model_dir / "directions"
    assert steerer.config_path == steerer.model_dir / "steering_config.json"


def test_non_default_estimator_gets_its_own_paths(tmp_path):
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                            estimator="rfm")

    assert steerer.directions_dir == steerer.model_dir / "directions-rfm"
    assert steerer.config_path == steerer.model_dir / "steering_config-rfm.json"


def test_estimators_do_not_share_a_config(tmp_path):
    """A built diff_means steerer must not make an rfm steerer look built."""
    make_built_steerer(tmp_path)
    rfm = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                        estimator="rfm")

    assert rfm.layer is None
    with pytest.raises(ValueError, match="build"):
        rfm.generate("hi", expertise=1.0)


def test_rfm_config_loads_independently(tmp_path):
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                            estimator="rfm")
    directions_dir = steerer.model_dir / "directions-rfm"
    directions_dir.mkdir(parents=True)
    np.save(directions_dir / "direction_layer_12.npy", np.ones(8))
    steerer.config_path.write_text(json.dumps({
        "domain": "hydrology", "model_name": "test/model", "layer": 12,
        "max_alpha": 0.1, "estimator": "rfm",
        "direction_file": "directions-rfm/direction_layer_12.npy",
        "steering_scale": "hidden_norm_relative",
    }))

    reloaded = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                             estimator="rfm")
    assert reloaded.layer == 12
    assert reloaded.max_alpha == 0.1
    # ... and the default estimator is still unbuilt, sharing nothing.
    assert DomainSteerer("test/model", "hydrology",
                         cache_dir=tmp_path).layer is None


def test_unknown_estimator_is_rejected_at_construction(tmp_path):
    with pytest.raises(ValueError, match="Unknown estimator"):
        DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                      estimator="nope")


def test_trap_variant_does_not_load_gold_config(tmp_path):
    make_built_steerer(tmp_path)
    trap = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap")
    assert trap.layer is None
    assert trap.directions_dir == trap.model_dir / "directions-trap"
    assert trap.config_path == trap.model_dir / "steering_config-trap.json"
    assert trap._pair_generator.pairs_path.name == "pairs-trap.jsonl"
    with pytest.raises(ValueError, match="build"):
        trap.generate("What is forcing?", expertise=1.0)


def test_stale_trap_config_is_ignored(tmp_path):
    trap = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap")
    trap.config_path.parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    directions = trap.model_dir / "directions-trap"
    directions.mkdir(parents=True)
    np.save(directions / "direction_layer_10.npy", np.ones(8))
    trap.config_path.write_text(json.dumps({
        "domain": "hydrology",
        "model_name": "test/model",
        "layer": 10,
        "max_alpha": 0.02,
        "contrast": "trap_commit_vs_survey_v3",
        "direction_file": "directions-trap/direction_layer_10.npy",
        "steering_scale": "hidden_norm_relative",
    }))
    reloaded = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                             variant="trap")
    assert reloaded.layer is None
    assert reloaded._trap_config_ok is False


def test_trap_rfm_gets_combined_suffix(tmp_path):
    trap = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap", estimator="rfm")
    assert trap.directions_dir == trap.model_dir / "directions-trap-rfm"
    assert trap.config_path == (
        trap.model_dir / "steering_config-trap-rfm.json"
    )


def test_self_gold_gpt_variant_is_isolated(tmp_path):
    gpt = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                        variant="self_gold_gpt")
    gold = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    assert gpt.directions_dir == gpt.model_dir / "directions-self_gold_gpt"
    assert gpt.config_path == (
        gpt.model_dir / "steering_config-self_gold_gpt.json"
    )
    assert gpt.directions_dir != gold.directions_dir
    assert gpt.config_path != gold.config_path


def test_trap_local_variant_is_isolated(tmp_path):
    local = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                          variant="trap_local")
    trap = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap")
    assert local.layer is None
    assert local.directions_dir == local.model_dir / "directions-trap_local"
    assert local.config_path == (
        local.model_dir / "steering_config-trap_local.json"
    )
    assert local._pair_generator.pairs_path == (
        local.model_dir / "pairs-trap-local.jsonl"
    )
    assert local._pair_generator.pairs_path != trap._pair_generator.pairs_path
    with pytest.raises(ValueError, match="build"):
        local.generate("What is forcing?", expertise=1.0)


def _write_trap_config(tmp_path, judge="nli", contrast="trap_commit_vs_survey_v4"):
    from domainsteer.pairs import TRAP_CONTRAST_ID
    trap = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap")
    trap.config_path.parent.mkdir(parents=True, exist_ok=True)
    directions = trap.model_dir / "directions-trap"
    directions.mkdir(parents=True, exist_ok=True)
    np.save(directions / "direction_layer_10.npy", np.ones(8))
    trap.config_path.write_text(json.dumps({
        "domain": "hydrology",
        "model_name": "test/model",
        "layer": 10,
        "max_alpha": 0.12,
        "judge": judge,
        "contrast": contrast or TRAP_CONTRAST_ID,
        "direction_file": "directions-trap/direction_layer_10.npy",
        "steering_scale": "hidden_norm_relative",
    }))
    return DomainSteerer("test/model", "hydrology", cache_dir=tmp_path,
                         variant="trap")


def test_nli_trap_config_is_rebuilt_for_pairwise(tmp_path):
    reloaded = _write_trap_config(tmp_path, judge="nli")
    assert reloaded.layer == 10
    assert reloaded._loaded_judge == "nli"
    assert reloaded._build_is_current(force=False, pairwise=True) is False
    assert reloaded._build_is_current(force=False, pairwise=False) is True


def test_pairwise_trap_config_is_current(tmp_path):
    reloaded = _write_trap_config(tmp_path, judge="pairwise")
    assert reloaded._build_is_current(force=False, pairwise=True) is True
    assert reloaded._build_is_current(force=True, pairwise=True) is False


def test_build_with_layer_skips_extras(tmp_path, monkeypatch):
    """The installed-package path never imports extras/."""
    from domainsteer.extract import ExtractionResult
    import domainsteer.steerer as steerer_mod

    dummy = ExtractionResult(
        model_name="test/model",
        directions={10: np.ones(8), 15: np.full(8, 2.0)},
        holdout_accuracy={10: 0.9, 15: 0.8},
        n_train_pairs=20,
        n_holdout_pairs=10,
    )

    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    dummy.save(steerer.directions_dir)
    steerer._model = object()
    steerer._tokenizer = object()
    monkeypatch.setattr(steerer, "_ensure_model", lambda: None)
    monkeypatch.setattr(steerer, "_resolve_pairs", lambda **k: [])
    monkeypatch.setattr(steerer, "_make_steering", lambda *a, **k: FakeSteering())

    def boom(module):
        raise AssertionError(f"imported extras.{module}")

    monkeypatch.setattr(steerer_mod, "_import_extras", boom)

    steerer.build(layer=15)
    assert steerer.layer == 15
    assert steerer.max_alpha is None
    assert steerer._loaded_judge == "none"
    assert steerer.generate("q", alpha=0.25) == "gen(alpha=0.25)"
    with pytest.raises(ValueError, match="max_alpha"):
        steerer.generate("q", expertise=1.0)
    steerer.build(layer=15)


def test_build_extracts_requested_layer_when_cache_lacks_it(tmp_path, monkeypatch):
    """Llama middle-third defaults are even layers; layer=15 must still build."""
    from domainsteer.extract import ExtractionResult
    import domainsteer.extract as extract_mod

    even = ExtractionResult(
        model_name="test/model",
        directions={10: np.ones(8), 12: np.ones(8), 14: np.ones(8),
                    16: np.ones(8), 18: np.ones(8)},
        holdout_accuracy={10: 0.5, 12: 0.5, 14: 0.5, 16: 0.5, 18: 0.5},
        n_train_pairs=20,
        n_holdout_pairs=10,
    )
    steerer = DomainSteerer("test/model", "hydrology", cache_dir=tmp_path)
    even.save(steerer.directions_dir)
    steerer._model = object()
    steerer._tokenizer = object()
    monkeypatch.setattr(steerer, "_ensure_model", lambda: None)
    monkeypatch.setattr(steerer, "_resolve_pairs", lambda **k: ["pair"])
    monkeypatch.setattr(steerer, "_make_steering", lambda *a, **k: FakeSteering())

    seen = {}

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract(self, pairs, layers=None, **kwargs):
            seen["layers"] = layers
            return ExtractionResult(
                model_name="test/model",
                directions={15: np.full(8, 3.0)},
                holdout_accuracy={15: 0.7},
                n_train_pairs=20,
                n_holdout_pairs=10,
            )

    monkeypatch.setattr(extract_mod, "DirectionExtractor", FakeExtractor)

    steerer.build(layer=15)
    assert seen["layers"] == [15]
    assert steerer.layer == 15
    assert np.allclose(steerer._direction, np.full(8, 3.0))
    assert steerer.generate("q", alpha=0.25) == "gen(alpha=0.25)"
