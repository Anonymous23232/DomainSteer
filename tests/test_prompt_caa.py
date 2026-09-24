"""Tests for the Rimsky-style prompt vs CAA runner — no models."""
from pathlib import Path

from domainsteer.self_gold import eval_system_prompt, gold_system_prompt
from domainsteer.steering import DEFAULT_SYSTEM_PROMPT


def _load_runner():
    import importlib.util

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "run_prompt_caa", root / "extras" / "scripts" / "run_prompt_caa.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def test_eval_prompt_is_a_template_not_gold():
    domain = "Quantum physics"
    eval_p = eval_system_prompt(domain)
    gold_p = gold_system_prompt(domain)
    assert domain in eval_p
    assert eval_p != gold_p
    assert "distractor" not in eval_p.lower()
    assert "trap" not in eval_p.lower()
    assert DEFAULT_SYSTEM_PROMPT.startswith("You are a helpful assistant")
    assert eval_p.startswith(f"You are a {domain} practitioner")
    assert "Answer in 1-2 short sentences" in eval_p
    assert "Do not use lists" in eval_p


def test_steered_at_reads_flat_and_nested():
    runner = _load_runner()
    flat = {
        "steered": {"0.1500": "mid", "0.2000": "lock"},
        "layer": 10,
    }
    alpha, text = runner.steered_at(flat, 0.20)
    assert alpha == 0.2
    assert text == "lock"
    nested = {
        "steered": {
            "10": {"0.2000": "layer10"},
            "12": {"0.2000": "layer12"},
        },
        "layer": 12,
        "chosen_layer": 12,
    }
    alpha, text = runner.steered_at(nested, 0.20)
    assert text == "layer12"


def test_regime_locks_to_gpt_not_llama_gold():
    runner = _load_runner()
    gpt = ("An event is a unique DNA-integration outcome created during "
           "transformation, defined by its insertion site.")
    baseline = ("An event is a concert or a natural disaster that happens "
                "at a particular time and place.")
    caa = ("An event is a unique DNA integration created during plant "
           "transformation, identified by its insertion site.")
    prompt_everyday = ("An event is a happening such as a concert or "
                       "festival at a particular time.")
    scored = runner.score_text(caa, gpt, baseline)
    assert scored["lock"] is True
    assert scored["regime"] == "domain"
    scored_p = runner.score_text(prompt_everyday, gpt, baseline)
    assert scored_p["lock"] is False
    assert scored_p["regime"] == "everyday"


def test_seed_row_reuses_prompt_unless_forced():
    runner = _load_runner()
    src = {
        "question": "What is an event?",
        "term": "event",
        "kind": "direct",
        "gpt_answer": "DNA insertion",
        "gold": "GMO event",
        "baseline": "a concert",
        "steered": {"0.2000": "insertion site"},
        "layer": 12,
    }
    prev = {"prompt": "old prompt", "prompt_caa": "old both"}
    row = runner.seed_row(src, prev, 0.20, force=False, force_all=False)
    assert row["baseline"] == "a concert"
    assert row["caa"] == "insertion site"
    assert row["prompt"] == "old prompt"
    assert row["prompt_caa"] == "old both"
    forced = runner.seed_row(src, prev, 0.20, force=True, force_all=False)
    assert forced["prompt"] == ""
    assert forced["caa"] == "insertion site"


def test_summarize_reports_caa_vs_prompt():
    runner = _load_runner()
    gpt = "A bridge is a ligand that binds two metal centers."
    baseline = "A bridge spans a river and carries traffic."
    items = [
        runner._score_row({
            "kind": "direct",
            "gpt_answer": gpt,
            "baseline": baseline,
            "prompt": baseline,
            "caa": gpt,
            "prompt_caa": gpt,
        }),
        runner._score_row({
            "kind": "tricky",
            "gpt_answer": gpt,
            "baseline": baseline,
            "prompt": gpt,
            "caa": gpt,
            "prompt_caa": gpt,
        }),
    ]
    summary = {row["kind"]: row for row in runner.summarize(items)}
    assert summary["direct"]["n"] == 1
    assert summary["direct"]["prompt_lock"] == 0.0
    assert summary["direct"]["caa_lock"] == 1.0
    assert summary["tricky"]["prompt_lock"] == 1.0
    assert summary["all"]["caa_lock"] == 1.0
    table = runner.format_summary_table(runner.summarize(items), "test")
    assert "Prompt" in table
    assert "CAA" in table
