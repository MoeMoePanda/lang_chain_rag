"""CLI: run the eval set in fast + best modes; write reports/eval-report.md."""
from __future__ import annotations

from langchain_google_genai import ChatGoogleGenerativeAI

from hdb_rag import config
from hdb_rag.eval.report import render_report
from hdb_rag.eval.runner import run_mode


def main() -> None:
    answer_llm = ChatGoogleGenerativeAI(
        model=config.ANSWER_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    fast_llm = ChatGoogleGenerativeAI(
        model=config.FAST_MODEL, google_api_key=config.GOOGLE_API_KEY
    )
    judge_llm = fast_llm  # reuse fast model as the LLM judge

    results_by_mode = {
        "fast": run_mode("fast", answer_llm=answer_llm, fast_llm=fast_llm, judge_llm=judge_llm),
        "best": run_mode("best", answer_llm=answer_llm, fast_llm=fast_llm, judge_llm=judge_llm),
    }
    render_report(results_by_mode, config.EVAL_REPORT)
    print(f"✅ wrote {config.EVAL_REPORT}")


if __name__ == "__main__":
    main()
