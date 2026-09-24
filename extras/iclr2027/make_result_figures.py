"""Result figures and appendix domain tables for the ICLR 2027 paper.

Sources:
- 8B 10-word CAA grid: registry-em-layer-sweep (CAA vs prompt)
- 8B 20-word four-arm: registry-em-layer-sweep-prompt (all three comparisons)
- 3B 10-word four-arm: meta-llama-llama-3-2-3b-instruct
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\likit\Desktop\domainsteer-pkg\results_new\model-sweep")
REG = ROOT / "registry-em-layer-sweep" / "meta-llama-llama-3-1-8b-instruct"
EIGHT = ROOT / "registry-em-layer-sweep-prompt" / "meta-llama-llama-3-1-8b-instruct"
THREE = ROOT / "meta-llama-llama-3-2-3b-instruct"
OUT = Path(__file__).resolve().parent / "figures"
TEX = Path(__file__).resolve().parent / "appendix_domain_tables.tex"

LAYERS = list(range(10, 21))
ALPHAS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
INK = "#1A1A1A"
MUTED = "#5A5A5A"
CAA = "#D55E00"
GOLD = "#009E73"
BLUE = "#0072B2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def hist_counts(wins, n=15):
    c = np.zeros(n + 1, dtype=int)
    for w in wins:
        if 0 <= w <= n:
            c[w] += 1
    return c


def cell_majority_from_scores(path, arm, vs_col):
    """Per (layer, alpha): how many domains have >=8 item wins vs vs_col."""
    wins = defaultdict(lambda: defaultdict(int))
    for r in read_csv(path):
        if r["arm"] != arm or r["layer"] == "" or r["alpha"] == "":
            continue
        key = (int(float(r["layer"])), round(float(r["alpha"]), 2))
        if float(r["sim"]) > float(r[vs_col]):
            wins[key][r["cluster_id"]] += 1
    grid = np.zeros((len(LAYERS), len(ALPHAS)))
    for i, L in enumerate(LAYERS):
        for j, a in enumerate(ALPHAS):
            grid[i, j] = sum(w >= 8 for w in wins[(L, a)].values())
    return grid


def cell_majority_from_summary(rows, arm=None):
    by = defaultdict(list)
    for r in rows:
        if arm is not None and r.get("arm") != arm:
            continue
        by[(int(r["layer"]), round(float(r["alpha"]), 2))].append(
            int(r["questions_won"]))
    grid = np.zeros((len(LAYERS), len(ALPHAS)))
    for i, L in enumerate(LAYERS):
        for j, a in enumerate(ALPHAS):
            grid[i, j] = sum(w >= 8 for w in by[(L, a)])
    return grid


def draw_heatmap(ax, grid, title, cmap):
    im = ax.imshow(grid, origin="lower", cmap=cmap, vmin=0, vmax=144, aspect="auto")
    ax.set_xticks(range(len(ALPHAS)))
    ax.set_xticklabels([f"{a:g}" for a in ALPHAS], fontsize=6.5)
    ax.set_yticks(range(len(LAYERS)))
    ax.set_yticklabels(LAYERS, fontsize=6.5)
    ax.set_xlabel(r"$\alpha$")
    ax.set_title(title, fontsize=7.5, pad=3)
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    ax.plot(j, i, "o", ms=3.0, mfc="none", mec=INK, mew=0.8)
    ax.text(j + 0.38, i, f"{int(grid[i, j])}", fontsize=6, color=INK, va="center")
    peak = (LAYERS[int(i)], ALPHAS[int(j)], int(grid[i, j]))
    return im, peak


def fig_wins():
    eight = read_csv(EIGHT / "domain_best.csv")
    series = [
        ("CAA vs silent",
         [int(r["caa_vs_unsteered_questions_won"]) for r in eight], BLUE),
        ("CAA vs prompt",
         [int(r["caa_vs_prompt_questions_won"]) for r in eight], CAA),
        ("prompt+CAA vs prompt",
         [int(r["prompt_caa_vs_prompt_questions_won"]) for r in eight], GOLD),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.15), sharey=True)
    for ax, (title, wins, color) in zip(axes, series):
        c = hist_counts(wins)
        ax.bar(range(16), c, color=color, width=0.82, edgecolor=INK, linewidth=0.3)
        ax.axvline(7.5, color=MUTED, ls="--", lw=0.7, zorder=0)
        maj = sum(w >= 8 for w in wins)
        ax.set_title(f"{title}\n{maj}/144 domains $\\geq$ 8/15", fontsize=8, pad=4)
        ax.set_xlim(-0.6, 15.6)
        ax.set_xticks([0, 5, 8, 10, 15])
        ax.set_xlabel("items won / 15")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("domains")
    fig.tight_layout(w_pad=0.8)
    fig.savefig(OUT / "fig_wins.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_wins.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_settings():
    eight = read_csv(EIGHT / "domain_best.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.2))

    ax = axes[0]
    bins = ALPHAS
    w = 0.018
    for vals, color, label, dx in (
        ([float(r["caa_vs_prompt_alpha"]) for r in eight],
         CAA, "CAA vs prompt", -w),
        ([float(r["prompt_caa_vs_prompt_alpha"]) for r in eight],
         GOLD, "prompt+CAA vs prompt", w),
    ):
        counts = [sum(abs(v - b) < 1e-9 for v in vals) for b in bins]
        ax.bar([b + dx for b in bins], counts, width=0.032, color=color,
               edgecolor=INK, linewidth=0.3, label=label)
    ax.set_xticks(bins)
    ax.set_xticklabels([f"{b:g}" for b in bins])
    ax.set_xlabel(r"best $\alpha$")
    ax.set_ylabel("domains")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-domain best strength", fontsize=8)

    ax = axes[1]
    for vals, color, label, dx in (
        ([int(r["caa_vs_prompt_layer"]) for r in eight],
         CAA, "CAA vs prompt", -0.18),
        ([int(r["prompt_caa_vs_prompt_layer"]) for r in eight],
         GOLD, "prompt+CAA vs prompt", 0.18),
    ):
        counts = [vals.count(L) for L in LAYERS]
        ax.bar([L + dx for L in LAYERS], counts, width=0.32, color=color,
               edgecolor=INK, linewidth=0.3, label=label)
    ax.set_xticks(LAYERS)
    ax.set_xlabel("best layer")
    ax.set_ylabel("domains")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Per-domain best layer", fontsize=8)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / "fig_settings.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_settings.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_grid():
    """2x3: 8B four-arm (top) and 3B four-arm (bottom), three comparisons."""
    specs = [
        ("caa", "unsteered_sim", "CAA vs silent", "Blues"),
        ("caa", "prompt_sim", "CAA vs prompt", "Oranges"),
        ("prompt_caa", "prompt_sim", "prompt+CAA vs prompt", "Greens"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 4.55))
    peaks = {}
    last_im = None
    for row, (label, scores) in enumerate(
            (("8B 20-word", EIGHT / "scores.csv"),
             ("3B 10-word", THREE / "scores.csv"))):
        for col, (arm, vs, title, cmap) in enumerate(specs):
            ax = axes[row, col]
            grid = cell_majority_from_scores(scores, arm, vs)
            im, peak = draw_heatmap(ax, grid, f"{label}: {title}", cmap)
            last_im = im
            peaks[f"{label} {title}"] = peak
            if col == 0:
                ax.set_ylabel("layer")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    print("grid peaks", peaks)
    fig.tight_layout(w_pad=0.55, h_pad=0.9)
    fig.savefig(OUT / "fig_grid.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_grid.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    del last_im
    return peaks


def fig_grid_10w():
    """Appendix: 8B 10-word CAA vs prompt, no per-domain selection."""
    grid = cell_majority_from_summary(
        read_csv(REG / "layer_summary_all_domains.csv"))
    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    im, peak = draw_heatmap(ax, grid, "8B 10-word: CAA vs prompt", "Oranges")
    ax.set_ylabel("layer")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("domains", fontsize=7)
    print("10-word CAA vs prompt peak", peak)
    fig.tight_layout()
    fig.savefig(OUT / "fig_grid_10w.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_grid_10w.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- appendix LaTeX --------------------------------------------------------

_TEX_ESC = str.maketrans({
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
})


def tex_esc(s):
    return str(s).translate(_TEX_ESC)


def fmt_alpha(x):
    return f"{float(x):g}"


def four_arm_rows(rows):
    lines = []
    for r in rows:
        lines.append(
            f"{r['cluster_id']} & {tex_esc(r['cluster'])} & "
            f"{int(r['caa_vs_unsteered_layer'])} & "
            f"{fmt_alpha(r['caa_vs_unsteered_alpha'])} & "
            f"{int(r['caa_vs_unsteered_questions_won'])} & "
            f"{int(r['caa_vs_prompt_layer'])} & "
            f"{fmt_alpha(r['caa_vs_prompt_alpha'])} & "
            f"{int(r['caa_vs_prompt_questions_won'])} & "
            f"{int(r['prompt_caa_vs_prompt_layer'])} & "
            f"{fmt_alpha(r['prompt_caa_vs_prompt_alpha'])} & "
            f"{int(r['prompt_caa_vs_prompt_questions_won'])} \\\\"
        )
    return "\n".join(lines)


def four_arm_table(caption, label, rows):
    n = len(rows)
    caa_s = sum(int(r["caa_vs_unsteered_questions_won"]) >= 8 for r in rows)
    caa_p = sum(int(r["caa_vs_prompt_questions_won"]) >= 8 for r in rows)
    pcaa = sum(int(r["prompt_caa_vs_prompt_questions_won"]) >= 8 for r in rows)
    head = r"""\begin{longtable}{r p{2.15cm} rrr rrr rrr}
\caption{""" + caption + r"""}
\label{""" + label + r"""} \\
\toprule
ID & Domain & \multicolumn{3}{c}{CAA vs silent} & \multicolumn{3}{c}{CAA vs prompt} & \multicolumn{3}{c}{p+CAA vs prompt} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
 & & $L$ & $\alpha$ & $w$ & $L$ & $\alpha$ & $w$ & $L$ & $\alpha$ & $w$ \\
\midrule
\endfirsthead
\toprule
ID & Domain & \multicolumn{3}{c}{CAA vs silent} & \multicolumn{3}{c}{CAA vs prompt} & \multicolumn{3}{c}{p+CAA vs prompt} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}
 & & $L$ & $\alpha$ & $w$ & $L$ & $\alpha$ & $w$ & $L$ & $\alpha$ & $w$ \\
\midrule
\endhead
\midrule
\multicolumn{11}{r}{\emph{continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
"""
    note = (
        f"\n\\end{{longtable}}\n"
        f"{{\\small $w$: item wins / 15 at that domain's best cell. "
        f"Majority ($\\geq 8$): CAA vs silent {caa_s}/{n}; "
        f"CAA vs prompt {caa_p}/{n}; "
        f"prompt+CAA vs prompt {pcaa}/{n}.}}\n"
    )
    return head + four_arm_rows(rows) + "\n" + note


def first_run_table(rows):
    body = []
    for r in rows:
        body.append(
            f"{r['cluster_id']} & {tex_esc(r['cluster'])} & "
            f"{int(r['best_layer'])} & {fmt_alpha(r['best_alpha'])} & "
            f"{int(r['questions_won'])} & "
            f"{float(r['steered_sim']):.3f} & "
            f"{float(r['prompt_sim']):.3f} & "
            f"{float(r['unsteered_sim']):.3f} \\\\"
        )
    maj = sum(int(r["questions_won"]) >= 8 for r in rows)
    head = r"""\begin{longtable}{r p{3.35cm} rr r rrr}
\caption{Llama-3.1-8B-Instruct, 10-word CAA grid. Per-domain best cell for CAA vs.\ prompt.}
\label{tab:app-8b-10w} \\
\toprule
ID & Domain & $L$ & $\alpha$ & $w$ & CAA sim & prompt sim & silent sim \\
\midrule
\endfirsthead
\toprule
ID & Domain & $L$ & $\alpha$ & $w$ & CAA sim & prompt sim & silent sim \\
\midrule
\endhead
\midrule
\multicolumn{8}{r}{\emph{continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
"""
    note = (
        f"\n\\end{{longtable}}\n"
        f"{{\\small $w$: CAA item wins vs.\\ the domain prompt / 15. "
        f"Majority ($\\geq 8$): {maj}/{len(rows)}.}}\n"
    )
    return head + "\n".join(body) + note


CLUSTERS_JSON = (
    Path(__file__).resolve().parents[2] / "domainsteer" / "data" / "domain_clusters.json"
)
REG_TEX = Path(__file__).resolve().parent / "appendix_registry.tex"


def write_registry():
    payload = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    clusters = sorted(payload["clusters"], key=lambda c: int(c["cluster_id"]))
    n_concepts = sum(len(c.get("concepts") or []) for c in clusters)
    body = []
    for c in clusters:
        names = "; ".join(
            tex_esc(item.get("name") or "")
            for item in (c.get("concepts") or [])
            if item.get("name")
        )
        body.append(
            f"{c['cluster_id']} & {tex_esc(c['cluster'])} & "
            f"{tex_esc(c.get('division') or '')} & {names} \\\\"
        )
    tex = (
        r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\setlength{\LTcapwidth}{\textwidth}

\begin{longtable}{r p{2.45cm} p{2.55cm} p{6.15cm}}
\caption{DomainSteer registry: 144 ANZSRC 4-digit groups and the ten concept names that partition each group.}
\label{tab:app-registry} \\
\toprule
ID & Domain & Division & Concepts \\
\midrule
\endfirsthead
\toprule
ID & Domain & Division & Concepts \\
\midrule
\endhead
\midrule
\multicolumn{4}{r}{\emph{continued on next page}} \\
\endfoot
\bottomrule
\endlastfoot
"""
        + "\n".join(body)
        + (
            f"\n\\end{{longtable}}\n"
            f"{{\\small {len(clusters)} domains, {n_concepts} concept names. "
            f"Names are globally disjoint.}}\n"
            r"\endgroup"
            "\n"
        )
    )
    REG_TEX.write_text(tex, encoding="utf-8")
    print("wrote", REG_TEX, f"({len(clusters)} domains, {n_concepts} concepts)")


def write_appendix():
    eight = read_csv(EIGHT / "domain_best.csv")
    three = read_csv(THREE / "domain_best.csv")
    first = read_csv(REG / "best_per_domain.csv")
    tex = r"""\begingroup
\scriptsize
\setlength{\tabcolsep}{3pt}
\setlength{\LTcapwidth}{\textwidth}

""" + first_run_table(first) + r"""

\vspace{1.2em}
""" + four_arm_table(
        r"Llama-3.1-8B-Instruct, 20-word four-arm grid. Per-domain best cell "
        r"for each comparison (layer $L$, strength $\alpha$, item wins $w$).",
        "tab:app-8b-20w",
        eight,
    ) + r"""

\vspace{1.2em}
""" + four_arm_table(
        r"Llama-3.2-3B-Instruct, 10-word four-arm grid. Per-domain best cell "
        r"for each comparison (layer $L$, strength $\alpha$, item wins $w$).",
        "tab:app-3b",
        three,
    ) + r"""
\endgroup
"""
    TEX.write_text(tex, encoding="utf-8")
    print("wrote", TEX)


EXAMPLES = Path(__file__).resolve().parent / "appendix_prompts_responses.tex"


def clean_ans(s):
    s = " ".join(str(s or "").split())
    s = (s.replace("…", "...")
         .replace("—", "---")
         .replace("–", "--")
         .replace("“", "``").replace("”", "''")
         .replace("‘", "`").replace("’", "'"))
    return tex_esc(s)


def pick_best_item(rows):
    """One item per domain: largest prompt+CAA cosine gain vs the prompt."""
    best = {}
    for r in rows:
        cid = r["cluster_id"]
        gain = float(r["prompt_caa_minus_prompt"])
        sim = float(r["prompt_caa_sim"])
        prev = best.get(cid)
        if prev is None or (gain, sim) > prev[0]:
            best[cid] = ((gain, sim), r)
    return [best[cid][1] for cid in sorted(best, key=lambda x: int(x))]


def write_prompts_and_responses():
    items = pick_best_item(read_csv(EIGHT / "items_at_best.csv"))
    blocks = []
    for r in items:
        cid = r["cluster_id"]
        name = tex_esc(r["cluster"])
        q = clean_ans(r["question"])
        term = clean_ans(r["term"])
        caa_l = int(r["caa_vs_unsteered_layer"])
        caa_a = fmt_alpha(r["caa_vs_unsteered_alpha"])
        pcaa_l = int(r["prompt_caa_layer"])
        pcaa_a = fmt_alpha(r["prompt_caa_alpha"])
        blocks.append(
            f"\\medskip\\noindent\\textbf{{{cid} {name}.}} "
            f"Term: \\emph{{{term}}}.\n"
            f"Q: {q}\n"
            f"\\begin{{itemize}}\\setlength{{\\itemsep}}{{0.15em}}"
            f"\\setlength{{\\parsep}}{{0pt}}\\setlength{{\\topsep}}{{0.2em}}\n"
            f"\\item Gold: {clean_ans(r['gold'])}\n"
            f"\\item Unsteered ({float(r['unsteered_sim']):.2f}): "
            f"{clean_ans(r['unsteered'])}\n"
            f"\\item Prompt ({float(r['prompt_sim']):.2f}): "
            f"{clean_ans(r['prompt'])}\n"
            f"\\item CAA $L{caa_l}$, $\\alpha={caa_a}$ "
            f"({float(r['caa_vs_unsteered_sim']):.2f}): "
            f"{clean_ans(r['caa_vs_unsteered'])}\n"
            f"\\item Prompt+CAA $L{pcaa_l}$, $\\alpha={pcaa_a}$ "
            f"({float(r['prompt_caa_sim']):.2f}): "
            f"{clean_ans(r['prompt_caa'])}\n"
            f"\\end{{itemize}}\n"
        )
    tex = r"""\section{Prompts}
\label{app:prompts}

The user turn is always the bare question. Nothing below is inserted into
the question text.

\subsection{Contrastive pairs}

The same unnamed stem is answered twice by the model under test. Expert
text is the positive class; default text is the negative class.
Extraction replays both answers under the default helper template.

\paragraph{Expert $(+)$.}
\begin{quote}\small\ttfamily\raggedright
You are a \{domain\} practitioner. Every question uses a term that has an
operational meaning in \{domain\}. Answer that meaning even when the wording
sounds like everyday life, media, sports, law, or another field --- those
readings are distractors, not the answer. Answer in 1-2 short sentences.
Do not use lists. Do not explain other fields' meanings of the term.
\end{quote}

\paragraph{Default $(-)$, and extraction wrapper.}
\begin{quote}\small\ttfamily\raggedright
You are a helpful assistant. Answer in 1-2 short sentences. Do not use lists.
\end{quote}

\subsection{Evaluation arms}

Unsteered and CAA use the helper prompt (no domain name). Prompt and
prompt+CAA use the practitioner prompt. The length cap matches the run.

\paragraph{Helper (unsteered, CAA), 10-word runs.}
\begin{quote}\small\ttfamily\raggedright
You are a helpful assistant. Answer in at most 10 words. Do not use lists.
\end{quote}

\paragraph{Practitioner (prompt, prompt+CAA), 10-word runs.}
\begin{quote}\small\ttfamily\raggedright
You are a \{domain\} practitioner. Answer in at most 10 words. Do not use lists.
\end{quote}

\paragraph{Helper, 20-word four-arm run.}
\begin{quote}\small\ttfamily\raggedright
You are a helpful assistant. Answer in at most 20 words. Do not use lists.
\end{quote}

\paragraph{Practitioner, 20-word four-arm run.}
\begin{quote}\small\ttfamily\raggedright
You are a \{domain\} practitioner. Answer in at most 20 words. Do not use lists.
\end{quote}

\section{Best-cell responses (8B, 20-word four-arm)}
\label{app:responses}

One item per domain: the trap with the largest prompt+CAA vs.\ prompt
cosine gain at that domain's best prompt+CAA cell.
Cosines in parentheses are MPNet similarity to the frozen gold.
CAA and prompt+CAA answers are from that arm's own best cell, which need
not be the same $(L,\alpha)$.

\begingroup\small
""" + "".join(blocks) + r"""
\endgroup
"""
    EXAMPLES.write_text(tex, encoding="utf-8")
    print("wrote", EXAMPLES, f"({len(items)} domains)")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_wins()
    fig_settings()
    fig_grid()
    fig_grid_10w()
    write_appendix()
    write_registry()
    write_prompts_and_responses()
    print("wrote", list(OUT.glob("fig_*.pdf")))
