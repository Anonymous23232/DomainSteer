"""Held-out exact-match trap items — no API, no model."""
from domainsteer.em_traps import (
    clean_em_items, em_hit, em_system_prompt, gold_ok, normalize_em,
    question_leaks, term_in_question,
)


def test_em_system_prompt_names_the_domain():
    from domainsteer.em_traps import EM_CAA_SYSTEM_PROMPT

    prompt = em_system_prompt("Agricultural biotechnology")
    assert prompt.startswith("You are a Agricultural biotechnology practitioner")
    assert "at most 10 words" in prompt
    assert "distractor" not in prompt.lower()
    assert "practitioner" not in EM_CAA_SYSTEM_PROMPT.lower()
    assert "helpful assistant" in EM_CAA_SYSTEM_PROMPT
    assert "at most 10 words" in EM_CAA_SYSTEM_PROMPT


def test_clip_eval_answer_caps_at_ten_words():
    from domainsteer.em_traps import clip_eval_answer

    gpt = "It specifies the amino-acid sequence translated from a coding region."
    long = (
        "The term \"code\" in molecular biology refers to a DNA sequence of "
        "nucleotides that encodes a gene, which is a segment of DNA that "
        "contains the information necessary for the synthesis of a protein. "
        "This genetic code is a set of rules that governs the translation "
        "of DNA into proteins."
    )
    assert clip_eval_answer(gpt) == gpt
    clipped = clip_eval_answer(long)
    assert "This genetic code" not in clipped
    assert len(clipped.split()) <= 10
    assert clipped == "The term \"code\" in molecular biology refers to a DNA"


def test_gold_phrase_in_caa_not_in_prompt():
    from domainsteer.em_traps import gold_mentioned, mentions_key

    item = {
        "term": "code",
        "gold": "amino acid sequence",
        "gold_aliases": [
            "codon meaning", "residue identity", "protein sequence",
            "translation product", "polypeptide order", "translated peptide",
        ],
    }
    prompt = (
        "In agricultural biotechnology, a code typically refers to a set of "
        "genetic instructions or a sequence of nucleotides used to introduce "
        "a specific trait through genetic engineering."
    )
    a20 = (
        "The term code in molecular biology refers to a DNA sequence of "
        "nucleotides that encodes a gene for the synthesis of a protein. "
        "This genetic code governs the translation of DNA into proteins."
    )
    a25 = (
        "The genetic code is the set of three nucleotide codons that are "
        "used to code for amino acids in a protein."
    )
    assert gold_mentioned(prompt, item) == []
    assert gold_mentioned(a20, item) == []
    assert not mentions_key(a20, "amino acid sequence")
    assert mentions_key(a25, "amino acid sequence")
    hits = gold_mentioned(a25, item)
    assert "amino acid sequence" in hits
    assert "codon meaning" in hits


def test_alias_phrase_counts():
    from domainsteer.em_traps import gold_mentioned

    item = {
        "gold": "cultivar identity",
        "gold_aliases": ["taxon ID"],
    }
    assert gold_mentioned("DNA that IDs the taxon.", item) == []
    assert gold_mentioned("It reports the taxon ID from a leaf.", item) == ["taxon ID"]


def test_gold_must_be_one_to_three_words():
    assert gold_ok("any nucleotide")
    assert gold_ok("insertion event")
    assert gold_ok("N")
    assert not gold_ok("")
    assert not gold_ok("unique DNA integration outcome at the insertion site")


def test_question_rejects_domain_preface():
    domain = "Agricultural biotechnology"
    assert question_leaks("In biology, what is an event?", domain)
    assert question_leaks("What is an agricultural biotechnology event?", domain)
    assert not question_leaks("What is an event?", domain)


def test_clean_drops_blocked_leaky_and_long_gold():
    domain = "Bioinformatics"
    raw = [
        {"term": "N", "question": "What does N stand for in a string?",
         "baseline_trap": "newline", "gold": "any nucleotide"},
        {"term": "N", "question": "What does N stand for in a string?",
         "baseline_trap": "null", "gold": "any nucleotide"},
        {"term": "stack", "question": "In bioinformatics, what is a stack?",
         "baseline_trap": "LIFO", "gold": "read pileup"},
        {"term": "event", "question": "What is an event?",
         "baseline_trap": "concert", "gold": "insertion event"},
        {"term": "gap", "question": "What is a gap?",
         "baseline_trap": "pause",
         "gold": "an unaligned stretch of residues in a multiple alignment"},
        {"term": "driver", "question": "Who is the driver?",
         "baseline_trap": "chauffeur", "gold": "lead variant"},
    ]
    out = clean_em_items(raw, domain, blocked=["event"], extra_leak=[domain])
    terms = [row["term"] for row in out]
    assert terms == ["N", "driver"]
    assert out[0]["gold"] == "any nucleotide"


def test_clean_keeps_prompt_hard_and_drops_cs_defs():
    from domainsteer.em_traps import prompt_easy_cs_def

    domain = "Agricultural biotechnology"
    assert prompt_easy_cs_def("vector", "What does a vector store?")
    assert not prompt_easy_cs_def("barcode", "What can a barcode distinguish?")
    raw = [
        {"term": "barcode", "question": "What can a barcode distinguish?",
         "baseline_trap": "a retail product SKU", "gold": "species identity",
         "dominant_prior": "retail"},
        {"term": "vector", "question": "What does a vector store?",
         "baseline_trap": "an ordered collection", "gold": "foreign DNA",
         "dominant_prior": "programming"},
        {"term": "code", "question": "What does code specify?",
         "baseline_trap": "program instructions", "gold": "amino acid sequence",
         "dominant_prior": "programming"},
    ]
    out = clean_em_items(raw, domain)
    assert [row["term"] for row in out] == ["barcode", "code"]
    assert out[0]["dominant_prior"] == "retail"


def test_gold_variants_keep_aliases():
    from domainsteer.em_traps import (
        apply_variants, gold_mentioned, item_gold_keys, item_gold_response,
    )

    thin = {
        "question": "What can a barcode distinguish?",
        "gold": "species identity",
        "baseline_trap": "a retail product SKU",
    }
    assert gold_mentioned("DNA that IDs the taxon.", thin) == []

    item = apply_variants(thin, {
        "gold_aliases": ["taxon ID", "species identity"],
        "gold_response": "A standardized DNA sequence used to identify taxa.",
    })
    assert "taxon ID" in item_gold_keys(item)
    assert item_gold_response(item) == "A standardized DNA sequence used to identify taxa."
    assert item["gold_responses"] == ["A standardized DNA sequence used to identify taxa."]
    assert gold_mentioned("It reports the taxon ID.", item) == ["taxon ID"]
    assert gold_mentioned("The UPC on a grocery package.", item) == []


def test_alias_head_matches_codons_and_herbicide():
    from domainsteer.em_traps import gold_mentioned, mentions_key

    assert mentions_key("three nucleotide codons", "codon meaning")
    assert mentions_key("herbicide resistance in the plant", "herbicide tolerance")
    assert mentions_key("resistance to pests or diseases", "pest resistance")
    assert not mentions_key("the plant is under stress", "stress tolerance")
    assert not mentions_key("a preliminary version of a document", "preliminary assembly")
    assert not mentions_key("DNA that IDs the taxon.", "taxon ID")
    item = {
        "gold": "engineered trait",
        "gold_aliases": ["herbicide tolerance", "pest resistance"],
    }
    assert "herbicide tolerance" in gold_mentioned(
        "including herbicide resistance and drought tolerance", item)
    assert "pest resistance" in gold_mentioned(
        "resistance to pests or diseases", item)


def test_em_hit_any_accepts_aliases():
    from domainsteer.em_traps import em_hit_any

    golds = ["species identity", "taxon ID"]
    assert em_hit_any("taxon ID", golds)["exact"] is True
    assert em_hit_any("It reports taxon ID from a leaf.", golds)["contained"] is True
    assert em_hit_any("a grocery UPC", golds)["contained"] is False


def test_term_must_appear_in_the_question():
    assert term_in_question("event", "What is an event?")
    assert not term_in_question("event", "What happened?")


def test_caa_texts_reads_alpha_grid_and_legacy_string():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_em_traps",
        Path(__file__).resolve().parent.parent / "extras" / "scripts" / "run_em_traps.py",
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    legacy = runner.caa_texts({"caa": "Gene sequences.", "unsteered": "an array"})
    assert legacy["0.2000"] == "Gene sequences."
    assert legacy["0.0000"] == "an array"
    grid = runner.caa_texts({
        "caa": {"0.0000": "books", "0.2000": "cloned DNA", "0.3000": "loop"},
    })
    assert grid["0.2000"] == "cloned DNA"
    scored = runner.score_row({
        "question": "What does a library contain?",
        "gold": "cloned fragments",
        "gold_aliases": ["insert collection"],
        "baseline_trap": "reusable code modules",
        "unsteered": "books",
        "baseline": "cloned DNA fragments",
        "caa": grid,
    }, report_alpha=0.20)
    assert isinstance(scored["caa"], dict)
    assert "0.2000" in scored["caa_gold_hits_by_alpha"]
    assert scored["caa_alpha"] == 0.2
    assert "caa_gold_hits" in scored
    alias_row = runner.score_row({
        "term": "payload",
        "gold": "engineered trait",
        "gold_aliases": ["herbicide tolerance", "pest resistance"],
        "baseline": "resistance to pests or diseases",
        "caa": {"0.2000": "GMO waffle", "0.2500": "herbicide resistance"},
    }, report_alpha=0.20)
    assert "pest resistance" in alias_row["baseline_gold_hits"]
    assert alias_row["caa_gold_hits_by_alpha"]["0.2500"] == ["herbicide tolerance"]
    table = runner.format_sense_table([alias_row], 0.20)
    assert "pest resistance" in table
    assert "engineered trait" not in table.split("gold/alias hit", 1)[-1]


def _load_generate_script():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "generate_em_traps",
        Path(__file__).resolve().parent.parent / "extras" / "scripts" / "generate_em_traps.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_item_wave_uses_parallel_calls():
    import threading
    import time

    gen = _load_generate_script()
    assert gen._n_calls(15, 8, 4, first_wave=True) == 3
    assert gen._n_calls(15, 8, 1, first_wave=True) == 1
    assert [len(c) for c in gen._chunks(list(range(15)), 4)] == [4, 4, 4, 3]

    class Concept:
        def __init__(self, name):
            self.name = name

    class Cluster:
        cluster_id = "3001"
        cluster = "Agricultural biotechnology"
        division = "Agricultural, Veterinary and Food Sciences"
        domain = "Agricultural biotechnology"
        concepts = [Concept("transformation")]

        def context_block(self):
            return "Parent: Agriculture"

    lock = threading.Lock()
    state = {"n": 0, "active": 0, "max_active": 0}

    def _item(i):
        return {
            "term": f"widget{i}",
            "question": f"What can a widget{i} distinguish?",
            "baseline_trap": "a retail tag",
            "gold": "cultivar id",
            "dominant_prior": "retail",
            "gold_aliases": ["taxon id"],
            "gold_responses": [
                "A widget marks which cultivar is which in the collection.",
                "It distinguishes one registered variety from another nearby line.",
                "The profile separates distinct genotypes in breeding material.",
            ],
        }

    def fake(prompt, api_key=None, model=None):
        with lock:
            state["n"] += 1
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            i = state["n"]
        time.sleep(0.05)
        with lock:
            state["active"] -= 1
        return {"items": [_item(i * 10), _item(i * 10 + 1)]}

    gen.request_json_completion = fake
    payload = gen.generate_items(
        Cluster(), n=4, blocked=[], model="test", api_key="x",
        batch_size=2, workers=3,
    )
    assert len(payload["items"]) == 4
    assert state["n"] >= 2
    assert state["max_active"] >= 2


def test_em_hit_ignores_articles_and_punctuation():
    gold = "Any nucleotide"
    assert em_hit("any nucleotide", gold)["exact"] is True
    assert em_hit("Any nucleotide.", gold)["exact"] is True
    assert em_hit("N means any nucleotide in the alignment", gold)["contained"] is True
    assert em_hit("newline", gold)["exact"] is False
    assert normalize_em("The insertion event") == "insertion event"
