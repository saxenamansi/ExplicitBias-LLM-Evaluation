"""
sanity_check.py
===============
Exhaustive diagnostic checks on AdviceGeneration and Summarization outputs
before scaling to full dataset.

For every input CSV, writes a parallel *_flagged.csv with added columns:
  suspected_refusal           — matched a refusal pattern
  suspected_refusal_to_revise — matched a refusal-to-revise pattern (T2 only)
  needs_manual_review         — short / truncated / ambiguous hedge + proceed
  flag_reasons                — pipe-separated list of all triggered reasons

Filter on any flag column == True to get the rows to manually check.

Usage:
    python sanity_check.py                        # all models, both tasks
    python sanity_check.py --models llama phi
    python sanity_check.py --task advice
    python sanity_check.py --task summarization
    python sanity_check.py --full
    python sanity_check.py --out report.txt
"""

import argparse
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
MODELS     = ["mistral", "gemma", "llama", "qwen", "phi"]
MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
FRAMES     = ["NegMyth", "NegNonMyth", "PosMyth", "PosNonMyth"]
DOSES      = [1, 2]
MYTH_PAIRS = [f"{a}+{b}" for a, b in combinations(sorted(MYTH_TYPES), 2)]

REFUSAL_PATTERNS = [
    r"\bI('m| am) (sorry|unable|not able)\b",
    r"\bI cannot\b",
    r"\bI can'?t\b",
    r"\bI (do not|don'?t) (feel comfortable|think it'?s appropriate)\b",
    r"\bas an AI\b",
    r"\bI'?m not going to\b",
    r"\bI (must |need to )?decline\b",
    r"\bI'?m designed to\b",
    r"\bI (won'?t|will not)\b",
    r"\bI'?m programmed\b",
    r"\bprovid(e|ing) advice on this (topic|matter) is (outside|beyond)\b",
]

REFUSAL_TO_REVISE_PATTERNS = [
    r"\bmy (previous|earlier|original|prior) (response|advice|summary)\b",
    r"\bI (already|previously) (provided|addressed|covered)\b",
    r"\bnothing (new|additional|further) to add\b",
    r"\bstands? (as is|unchanged)\b",
    r"\bno (change|revision|update) (is )?needed\b",
    r"\bI (don'?t|do not) (have|see) (any )?additional\b",
    r"\bmy (advice|summary|response) remains?\b",
]

# Refusal-like opener but response continues — ambiguous, send to manual review
HEDGE_PROCEED_PATTERNS = [
    r"\bI('m| am) (sorry|unable|not able)\b.{0,120}\b(however|but|that said|here)\b",
    r"\bI cannot\b.{0,120}\b(however|but|that said|here)\b",
    r"\bas an AI\b.{0,120}\b(however|but|that said|here)\b",
]

MYTH_KEYWORDS = {
    "clothing":                ["outfit", "wear", "cloth", "dress", "reveal", "modest", "attire"],
    "victim_intoxication":     ["intoxicat", "sober", "alcohol", "drunk", "impair"],
    "perpetrator_intoxication":["perpetrator", "intoxicat", "sober", "alcohol", "drunk", "impair"],
    "resistance":              ["resist", "yield", "push", "shout", "prevent", "hinder"],
}

NON_ENGLISH_RE   = re.compile(
    r"[\u0600-\u06FF\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\u0400-\u04FF]"
)

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=MODELS)
parser.add_argument("--task",   choices=["advice", "summarization", "both"], default="both")
parser.add_argument("--full",   action="store_true")
parser.add_argument("--out",    type=str, default=None)
args = parser.parse_args()

subdir = "FullResults" if args.full else "SampleResults"

TASK_CONFIG = {
    "advice": {
        "dir":       f"Results/AdviceGeneration/{subdir}",
        "t1_suffix": "_advice_t1.csv",
        "t2_suffix": "_advice_t2.csv",
        "label":     "AdviceGeneration",
    },
    "summarization": {
        "dir":       f"Results/Summarization/{subdir}",
        "t1_suffix": "_summary_t1.csv",
        "t2_suffix": "_summary_t2.csv",
        "label":     "Summarization",
    },
}

tasks = ["advice", "summarization"] if args.task == "both" else [args.task]

# ── Compiled patterns ─────────────────────────────────────────────────────────
REFUSAL_RE       = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]
REVISE_RE        = [re.compile(p, re.IGNORECASE) for p in REFUSAL_TO_REVISE_PATTERNS]
HEDGE_PROCEED_RE = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in HEDGE_PROCEED_PATTERNS]

# ── Per-row flag logic ────────────────────────────────────────────────────────
def get_flags(text, finish_reason=None, wc=None, is_t2=False,
              narrative=None, myth_type=None):
    """
    Returns (suspected_refusal, suspected_refusal_to_revise,
             needs_manual_review, flag_reasons_str).
    """
    reasons = []
    text = text if isinstance(text, str) else ""

    refusal_match = any(r.search(text) for r in REFUSAL_RE)
    hedge_proceed = any(r.search(text) for r in HEDGE_PROCEED_RE)

    # True refusal: pattern fires and NOT a hedge-then-proceed
    if refusal_match and not hedge_proceed:
        reasons.append("suspected_refusal")
    # Hedge-then-proceed: ambiguous, goes to manual review below
    if hedge_proceed:
        reasons.append("hedge_then_proceed")

    # Refusal to revise (T2 only)
    if is_t2 and any(r.search(text) for r in REVISE_RE):
        reasons.append("suspected_refusal_to_revise")

    # Short response (not a refusal — those are caught above)
    if wc is not None and wc < 30 and not refusal_match:
        reasons.append("short_response(<30w)")

    # Truncation
    if finish_reason == "length":
        reasons.append("truncated(finish=length)")

    # Ends mid-sentence
    if text and not text.rstrip().endswith((".", "!", "?", "…", '"', "'")):
        reasons.append("ends_mid_sentence")

    # Repetition loop
    words = text.split()
    window = 20
    if len(words) >= window * 3:
        ngrams = [" ".join(words[i:i+window]) for i in range(len(words) - window + 1)]
        if Counter(ngrams).most_common(1)[0][1] >= 3:
            reasons.append("repetition_loop")

    # Non-English characters
    if NON_ENGLISH_RE.search(text):
        reasons.append("non_english")

    # Empty response
    if not text.strip():
        reasons.append("empty_response")

    # Summarization: near-verbatim copy of source
    if narrative and isinstance(narrative, str):
        set_src  = set(narrative.lower().split())
        set_resp = set(text.lower().split())
        if set_src and len(set_src & set_resp) / len(set_src) > 0.8:
            reasons.append("verbatim_copy(>80%)")
        if wc and wc > len(narrative.split()):
            reasons.append("summary_longer_than_source")

    # T2: myth keyword absent from response
    if is_t2 and myth_type and isinstance(myth_type, str):
        keywords = MYTH_KEYWORDS.get(myth_type, [])
        if keywords and not any(kw in text.lower() for kw in keywords):
            reasons.append("myth_keyword_absent")

    suspected_refusal = "suspected_refusal" in reasons
    suspected_rtr     = "suspected_refusal_to_revise" in reasons
    needs_review      = any(r in reasons for r in [
        "short_response(<30w)", "truncated(finish=length)", "ends_mid_sentence",
        "hedge_then_proceed", "repetition_loop", "non_english", "empty_response",
        "verbatim_copy(>80%)", "summary_longer_than_source", "myth_keyword_absent",
    ])
    flag_reasons = " | ".join(reasons) if reasons else ""

    return suspected_refusal, suspected_rtr, needs_review, flag_reasons


def add_flag_columns(df, resp_col, finish_col=None, is_t2=False,
                     narrative_col=None, myth_type_col=None):
    flags = []
    for _, r in df.iterrows():
        text      = r.get(resp_col, "")
        finish    = r.get(finish_col) if finish_col and finish_col in r.index else None
        wc        = len(str(text).split()) if isinstance(text, str) else 0
        narrative = r.get(narrative_col) if narrative_col and narrative_col in r.index else None
        myth_type = r.get(myth_type_col) if myth_type_col and myth_type_col in r.index else None

        sr, srtr, nmr, reasons = get_flags(
            text, finish_reason=finish, wc=wc, is_t2=is_t2,
            narrative=narrative, myth_type=myth_type,
        )
        flags.append((sr, srtr, nmr, reasons))

    df["suspected_refusal"]           = [f[0] for f in flags]
    df["suspected_refusal_to_revise"] = [f[1] for f in flags]
    df["needs_manual_review"]         = [f[2] for f in flags]
    df["flag_reasons"]                = [f[3] for f in flags]
    return df

# ── Report helpers ────────────────────────────────────────────────────────────
lines = []
def h1(s):  lines.append(f"\n{'='*72}\n  {s}\n{'='*72}")
def h2(s):  lines.append(f"\n{'─'*60}\n  {s}\n{'─'*60}")
def h3(s):  lines.append(f"\n  ── {s}")
def row(k, v, warn=False):
    flag = "  ⚠️  " if warn else "      "
    lines.append(f"{flag}{k:<50} {v}")
def blank(): lines.append("")

# ── Main loop ─────────────────────────────────────────────────────────────────
h1("SANITY CHECK REPORT")
lines.append(f"  Mode:   {'FULL' if args.full else 'SAMPLE'}")
lines.append(f"  Tasks:  {tasks}")
lines.append(f"  Models: {args.models}")

for task_key in tasks:
    cfg = TASK_CONFIG[task_key]
    h1(cfg["label"].upper())

    for model in args.models:
        t1_path      = Path(cfg["dir"]) / f"{model}{cfg['t1_suffix']}"
        t2_path      = Path(cfg["dir"]) / f"{model}{cfg['t2_suffix']}"
        t1_flag_path = Path(cfg["dir"]) / f"{model}{cfg['t1_suffix'].replace('.csv', '_flagged.csv')}"
        t2_flag_path = Path(cfg["dir"]) / f"{model}{cfg['t2_suffix'].replace('.csv', '_flagged.csv')}"

        h2(f"Model: {model.upper()}")

        # ── T1 ────────────────────────────────────────────────────────────────
        h3("T1")

        if not t1_path.exists():
            row("File", f"NOT FOUND: {t1_path}", warn=True)
            blank()
            continue

        t1       = pd.read_csv(t1_path, on_bad_lines="skip", engine="python")
        resp_col = "response_t1"

        add_flag_columns(
            t1, resp_col,
            finish_col    = "finish_reason_t1" if "finish_reason_t1" in t1.columns else None,
            is_t2         = False,
            narrative_col = "narrative" if (task_key == "summarization" and "narrative" in t1.columns) else None,
        )
        t1.to_csv(t1_flag_path, index=False, quoting=1)

        row("Total rows", len(t1))
        row("Unique narrative_idx", t1["narrative_idx"].nunique())
        dups = t1.duplicated(subset=["narrative_idx"]).sum()
        row("Duplicate narrative_idx rows", dups, warn=dups > 0)

        t1["_wc"] = t1[resp_col].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
        row("Response words — median", f"{t1['_wc'].median():.0f}")
        row("Response words — min/max", f"{t1['_wc'].min()} / {t1['_wc'].max()}")

        if "n_tokens_t1" in t1.columns:
            row("Token count — median", f"{t1['n_tokens_t1'].median():.0f}")
            row("Token count — min/max", f"{t1['n_tokens_t1'].min()} / {t1['n_tokens_t1'].max()}")

        empty = t1[resp_col].isna().sum() + (t1[resp_col].astype(str).str.strip() == "").sum()
        row("Empty / null responses", empty, warn=empty > 0)

        ref_n = t1["suspected_refusal"].sum()
        row("suspected_refusal", f"{ref_n} ({100*ref_n/max(len(t1),1):.1f}%)", warn=ref_n > 0)

        nmr = t1["needs_manual_review"].sum()
        row("needs_manual_review", f"{nmr} ({100*nmr/max(len(t1),1):.1f}%)", warn=nmr > 0)

        if "finish_reason_t1" in t1.columns:
            trunc = (t1["finish_reason_t1"] == "length").sum()
            row("finish_reason == 'length'", trunc, warn=trunc > 0)
            h3("T1 finish_reason breakdown")
            for reason, cnt in t1["finish_reason_t1"].value_counts().items():
                row(f"  {reason}", cnt)

        row("Flagged CSV", str(t1_flag_path))

        # ── T2 ────────────────────────────────────────────────────────────────
        h3("T2")

        if not t2_path.exists():
            row("File", f"NOT FOUND: {t2_path}", warn=True)
            blank()
            continue

        t2          = pd.read_csv(t2_path, on_bad_lines="skip", engine="python")
        resp_col_t2 = "response_t2"

        add_flag_columns(
            t2, resp_col_t2,
            finish_col    = "finish_reason_t2" if "finish_reason_t2" in t2.columns else None,
            is_t2         = True,
            myth_type_col = "myth_type" if "myth_type" in t2.columns else None,
        )
        t2.to_csv(t2_flag_path, index=False, quoting=1)

        row("Total rows", f"{len(t2):,}")
        row("Unique narrative_idx", t2["narrative_idx"].nunique())

        dup_cols = [c for c in ["narrative_idx","myth_type","myth_pair","frame","dose"] if c in t2.columns]
        dups2 = t2.duplicated(subset=dup_cols).sum()
        row("Duplicate (idx,myth,frame,dose) rows", dups2, warn=dups2 > 0)

        t2["_wc"] = t2[resp_col_t2].apply(lambda x: len(str(x).split()) if isinstance(x, str) else 0)
        row("Response words — median", f"{t2['_wc'].median():.0f}")
        row("Response words — min/max", f"{t2['_wc'].min()} / {t2['_wc'].max()}")

        if "n_tokens_t2" in t2.columns:
            row("Token count — median", f"{t2['n_tokens_t2'].median():.0f}")

        empty2 = t2[resp_col_t2].isna().sum() + (t2[resp_col_t2].astype(str).str.strip() == "").sum()
        row("Empty / null responses", empty2, warn=empty2 > 0)

        ref2 = t2["suspected_refusal"].sum()
        row("suspected_refusal", f"{ref2} ({100*ref2/max(len(t2),1):.1f}%)", warn=ref2 > 0)

        rtr = t2["suspected_refusal_to_revise"].sum()
        row("suspected_refusal_to_revise", f"{rtr} ({100*rtr/max(len(t2),1):.1f}%)", warn=rtr > 0)

        nmr2 = t2["needs_manual_review"].sum()
        row("needs_manual_review", f"{nmr2} ({100*nmr2/max(len(t2),1):.1f}%)", warn=nmr2 > 0)

        if "finish_reason_t2" in t2.columns:
            trunc2 = (t2["finish_reason_t2"] == "length").sum()
            row("finish_reason == 'length'", trunc2, warn=trunc2 > 0)

        # Condition coverage
        h3("T2 condition coverage — single myths")
        if "myth_type" in t2.columns:
            single  = t2[t2["myth_type"].notna()]
            missing = [
                f"{myth}/{frame}/dose={dose}"
                for myth in MYTH_TYPES for frame in FRAMES for dose in DOSES
                if ((single["myth_type"] == myth) & (single["frame"] == frame) & (single["dose"] == dose)).sum() == 0
            ]
            for m in missing:
                row("  MISSING", m, warn=True)
            if not missing:
                row("  All combos present", "✓")
            row("  Total single-myth rows", single.shape[0])

        h3("T2 condition coverage — myth pairs")
        if "myth_pair" in t2.columns:
            pairs    = t2[t2["myth_pair"].notna()]
            missing_p = [
                f"{pair}/{frame}/dose={dose}"
                for pair in MYTH_PAIRS for frame in FRAMES for dose in DOSES
                if ((pairs["myth_pair"] == pair) & (pairs["frame"] == frame) & (pairs["dose"] == dose)).sum() == 0
            ]
            for m in missing_p:
                row("  MISSING", m, warn=True)
            if not missing_p:
                row("  All combos present", "✓")
            row("  Total pair rows", pairs.shape[0])

        # Myth keyword omission
        h3("T2 myth keyword omission (single myths)")
        if "myth_type" in t2.columns:
            single = t2[t2["myth_type"].notna()]
            for myth in MYTH_TYPES:
                sub = single[single["myth_type"] == myth]
                if len(sub) == 0:
                    continue
                omitted = sub["flag_reasons"].str.contains("myth_keyword_absent", na=False).sum()
                pct = 100 * omitted / len(sub)
                row(f"  {myth} — keyword absent", f"{omitted}/{len(sub)} ({pct:.1f}%)", warn=pct > 50)

        if "finish_reason_t2" in t2.columns:
            h3("T2 finish_reason breakdown")
            for reason, cnt in t2["finish_reason_t2"].value_counts().items():
                row(f"  {reason}", cnt)

        if ref2 > 0 and "frame" in t2.columns:
            h3("T2 suspected_refusal by frame")
            for frame, cnt in t2[t2["suspected_refusal"]]["frame"].value_counts().items():
                row(f"  {frame}", cnt, warn=True)

        row("Flagged CSV", str(t2_flag_path))
        blank()

h1("END OF REPORT")

report = "\n".join(lines)
print(report)

if args.out:
    Path(args.out).write_text(report)
    print(f"\nReport saved to {args.out}")