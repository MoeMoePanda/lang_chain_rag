"""CLI: run the eval set in fast and/or best mode; write reports/eval-report.md."""
from __future__ import annotations

import argparse

from langchain_google_genai import ChatGoogleGenerativeAI

from hdb_rag import config
from hdb_rag.eval.report import render_report
from hdb_rag.eval.runner import load_saved_results, run_mode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["fast", "best", "both"], default="both",
        help="Which retrieval mode(s) to evaluate (default: both).",
    )
    args = parser.parse_args()

    answer_llm = ChatGoogleGenerativeAI(
        model=config.ANSWER_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    fast_llm = ChatGoogleGenerativeAI(
        model=config.FAST_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    judge_llm = fast_llm  # reuse fast model as the LLM judge

    results_by_mode: dict[str, list] = {}
    if args.mode in ("fast", "both"):
        results_by_mode["fast"] = run_mode(
            "fast", answer_llm=answer_llm, fast_llm=fast_llm, judge_llm=judge_llm
        )
    if args.mode in ("best", "both"):
        results_by_mode["best"] = run_mode(
            "best", answer_llm=answer_llm, fast_llm=fast_llm, judge_llm=judge_llm
        )

    # Pull in persisted results for any modes we didn't run this time, so
    # `--mode best` later still produces a combined report.
    for mode in ("fast", "best"):
        if mode not in results_by_mode:
            saved = load_saved_results(mode)
            if saved:
                results_by_mode[mode] = saved

    render_report(results_by_mode, config.EVAL_REPORT)
    print(f"✅ wrote {config.EVAL_REPORT} (modes: {list(results_by_mode.keys())})")


if __name__ == "__main__":
    main()
