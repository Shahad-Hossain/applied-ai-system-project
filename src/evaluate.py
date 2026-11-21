"""
Evaluation harness for the RAG recommender: runs labeled requests through the real
pipeline and reports pass/fail, confidence, and repeatability.

Run from the project root:
    python -m src.evaluate            # print the results table
    python -m src.evaluate --write    # also save docs/evaluation_results.md

Exit code is 0 if every case marked as expected-to-pass passes, 1 otherwise, so it
can be used as a regression check. Cases marked KNOWN_LIMIT are documented failures.
"""

import argparse
import statistics
import sys
from pathlib import Path

from src.rag import LOW_CONFIDENCE, Retriever, ask, confidence_label, load_knowledge_base, setup_logging
from src.recommender import load_songs

RESULTS_PATH = "docs/evaluation_results.md"

# kind:
#   answer     -> status ok, top retrieved note in `notes`, top song genre in `genres`
#   abstain    -> system should refuse or report low confidence (no good answer exists)
#   invalid    -> guardrail must reject the input
#   conflict   -> system must warn that the request mixes conflicting energy levels
#   min_energy -> top song energy must be at least `value` (known limitation, see README)
CASES = [
    # --- should answer correctly -------------------------------------------------
    dict(kind="answer", query="something calm for studying", notes={"act_study"}, genres={"lofi"}),
    dict(kind="answer", query="focus music for coding", notes={"act_study"}, genres={"lofi"}),
    dict(kind="answer", query="pump-up songs for the gym", notes={"act_workout"}, genres={"edm"}),
    dict(kind="answer", query="sad songs after a breakup", notes={"act_heartbreak", "mood_sad"}, genres={"folk"}),
    dict(kind="answer", query="music for a late night drive", notes={"act_roadtrip"}, genres={"synthwave"}),
    dict(kind="answer", query="I need to fall asleep", notes={"act_sleep"}, genres={"ambient"}),
    dict(kind="answer", query="angry music to vent", notes={"act_anger"}, genres={"rock"}),
    dict(kind="answer", query="upbeat dance party", notes={"act_party"}, genres={"pop", "edm"}),
    dict(kind="answer", query="romantic dinner date", notes={"act_date"}, genres={"jazz", "pop"}),
    dict(kind="answer", query="feeling confident getting ready to go out", notes={"act_confident"}, genres={"hip hop"}),
    dict(kind="answer", query="sunny picnic in the park", notes={"act_nature"}, genres={"folk"}),
    dict(kind="answer", query="relaxing jazz for dinner", notes={"gen_jazz", "act_date"}, genres={"jazz"}),
    dict(kind="answer", query="retro eighties synth vibes", notes={"gen_synthwave"}, genres={"synthwave"}),
    dict(kind="answer", query="cheerful good morning songs", notes={"act_morning"}, genres={"pop", "indie pop"}),
    dict(kind="answer", query="chill beats for a rainy afternoon", notes={"act_rainy", "gen_lofi"}, genres={"lofi", "jazz"}),
    # --- should abstain or flag low confidence -----------------------------------
    dict(kind="abstain", query="classical piano"),
    dict(kind="abstain", query="death metal screaming"),
    dict(kind="abstain", query="asdfgh qwerty"),
    dict(kind="abstain", query="k-pop idol groups"),
    # --- guardrails ---------------------------------------------------------------
    dict(kind="invalid", query=""),
    dict(kind="invalid", query="a"),
    dict(kind="invalid", query="12345"),
    dict(kind="invalid", query="x" * 400),
    dict(kind="invalid", query=None),
    # --- conflicting requests ----------------------------------------------------
    dict(kind="conflict", query="something sad but also really loud"),
    # --- documented limitation ---------------------------------------------------
    dict(kind="min_energy", query="something sad but also really loud", value=0.6, known_limit=True),
]


def run_case(case, songs, retriever):
    """Runs one case; returns (passed, detail, confidence_or_None, result)."""
    result = ask(case["query"], songs, retriever)
    kind = case["kind"]
    conf = result.confidence if result.status == "ok" else None
    top = result.recommendations[0][0] if result.recommendations else None

    if kind == "answer":
        if result.status != "ok":
            return False, f"status {result.status}", conf, result
        note_id = result.sources[0][0].id
        ok = note_id in case["notes"] and top["genre"] in case["genres"]
        return ok, f"top note {note_id}; top song {top['title']} ({top['genre']})", conf, result
    if kind == "abstain":
        if result.status != "ok":
            return True, f"refused ({result.status})", conf, result
        ok = result.confidence < LOW_CONFIDENCE
        return ok, f"answered with confidence {result.confidence:.2f}", conf, result
    if kind == "invalid":
        return result.status == "invalid", f"status {result.status}", conf, result
    if kind == "conflict":
        return bool(result.warnings), ("warned: conflict flagged" if result.warnings else "no warning"), conf, result
    if kind == "min_energy":
        if top is None:
            return False, "no recommendation", conf, result
        return top["energy"] >= case["value"], f"top song {top['title']} energy {top['energy']}", conf, result
    raise ValueError(f"unknown case kind {kind}")


CRITERIA = {
    "answer": "Right note retrieved and matching genre ranked first",
    "abstain": "Refuses or reports low confidence",
    "invalid": "Rejected by guardrail",
    "conflict": "Warns that request mixes energy levels",
    "min_energy": "Top song energy >= 0.6",
}


def evaluate(songs, retriever):
    rows, repeat_ok, repeat_total = [], 0, 0
    for case in CASES:
        passed, detail, conf, result = run_case(case, songs, retriever)
        rows.append((case, passed, detail, conf))
        if case["kind"] == "answer":
            again = ask(case["query"], songs, retriever)
            repeat_total += 1
            repeat_ok += [s["id"] for s, _, _ in again.recommendations] == [s["id"] for s, _, _ in result.recommendations]
    return rows, repeat_ok, repeat_total


def _display_query(q) -> str:
    text = repr(q) if not isinstance(q, str) else q
    return (text[:37] + "...") if len(text) > 40 else (text or "(empty)")


def build_report(rows, repeat_ok, repeat_total) -> str:
    expected = [r for r in rows if not r[0].get("known_limit")]
    passed = sum(1 for _, ok, _, _ in expected if ok)
    lines = ["# Evaluation Results", "",
             "Generated by `python -m src.evaluate` against the real pipeline (no mocks).", "",
             "| # | Test input | Kind | Evaluation criteria | Result | Detail | Confidence |",
             "|---|---|---|---|---|---|---|"]
    for i, (case, ok, detail, conf) in enumerate(rows, 1):
        label = "Pass" if ok else ("Fail (known limitation)" if case.get("known_limit") else "FAIL")
        conf_text = f"{conf:.2f} ({confidence_label(conf)})" if conf is not None else "n/a"
        lines.append(f"| {i} | `{_display_query(case['query'])}` | {case['kind']} | {CRITERIA[case['kind']]} | {label} | {detail} | {conf_text} |")

    lines += ["", "## Summary", "", "| Kind | Passed | Total |", "|---|---|---|"]
    for kind in CRITERIA:
        group = [r for r in expected if r[0]["kind"] == kind]
        if group:
            lines.append(f"| {kind} | {sum(1 for _, ok, _, _ in group if ok)} | {len(group)} |")
    lines.append(f"| **all** | **{passed}** | **{len(expected)}** |")

    answered = [c for case, ok, _, c in rows if case["kind"] == "answer" and c is not None]
    abstain = [c for case, ok, _, c in rows if case["kind"] == "abstain" and c is not None]
    lines += ["", f"- **Overall:** {passed} of {len(expected)} expected-to-pass cases passed "
                  f"({sum(1 for r in rows if r[0].get('known_limit') and not r[1])} documented known-limitation case(s) excluded).",
              f"- **Repeatability:** {repeat_ok} of {repeat_total} answer requests gave identical rankings on a second run."]
    if answered:
        lines.append(f"- **Mean confidence on answerable requests:** {statistics.mean(answered):.2f}.")
    if abstain:
        lines.append(f"- **Mean confidence on out-of-scope requests that were answered:** {statistics.mean(abstain):.2f} "
                     "(refused requests have no confidence).")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"save the report to {RESULTS_PATH}")
    args = parser.parse_args()

    setup_logging()
    songs = load_songs("data/songs.csv")
    retriever = Retriever(load_knowledge_base())
    rows, repeat_ok, repeat_total = evaluate(songs, retriever)
    report = build_report(rows, repeat_ok, repeat_total)
    print(report)
    if args.write:
        Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
        Path(RESULTS_PATH).write_text(report, encoding="utf-8")
        print(f"Saved {RESULTS_PATH}")

    return 0 if all(ok for case, ok, _, _ in rows if not case.get("known_limit")) else 1


if __name__ == "__main__":
    sys.exit(main())
