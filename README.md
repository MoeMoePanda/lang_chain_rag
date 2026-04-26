# HDB Rules RAG Chatbot

A LangChain-powered chatbot that answers questions about Singapore HDB housing rules — covering BTO/resale eligibility, grants, MOP, ethnic quotas, subletting, and renovation rules. Hybrid retrieval (BM25 + vector), multi-query expansion, LLM reranking, and history-aware conversation, evaluated against a 30-question test set with committed metrics.

> ⚠️ Informational only. Verify on [hdb.gov.sg](https://www.hdb.gov.sg) before any decisions.

## What this demonstrates

- **Idiomatic LangChain (LCEL)** — the entire chain is composed with `|` and `RunnablePassthrough.assign`, no deprecated `Chain` classes.
- **Hybrid retrieval** — BM25 + vector ensemble, multi-query rewrites, LLM listwise rerank.
- **History-aware RAG** — standalone-question rewriter handles multi-turn conversations.
- **Strict grounding + refusal** — answers only from indexed context; refuses out-of-scope politely.
- **Auditable source allowlist** — `data/sources.yaml` is the curated, committed knowledge base, populated from HDB's sitemap.
- **Two vector backends** — Chroma (local-dev, zero-setup) + pgvector (Supabase, deployed).
- **Committed eval results** — see [`reports/eval-report.md`](reports/eval-report.md). 30 cases, 5 metrics, resumable runner.
- **LangSmith tracing** — set `LANGSMITH_TRACING=true` for full per-step traces.

## Eval results

Latest `make eval` run (fast mode, 30 cases):

| Mode | Retrieval@5 | Faithfulness | Relevance | Citation | Refusal |
|---|---|---|---|---|---|
| fast | 88% | 88% | 96% | 88% | 83% |
| best | — | — | — | — | — |

`best` mode (BM25 + multi-query + rerank) is implemented but not scored in this run; it requires `data/chunks.jsonl`, which is regenerated on `make ingest`. Failure list and per-case detail: [`reports/eval-report.md`](reports/eval-report.md).

## Architecture

```mermaid
flowchart TB
  S[sources.yaml<br/>committed allowlist] -->|make ingest| I[Ingestion<br/>HTML+PDF → chunks → embed]
  I --> V[(Vector store<br/>pgvector | chroma)]
  I --> C[chunks.jsonl<br/>BM25 cache]
  Q[user question + chat history] --> R[Standalone-question rewrite<br/>LCEL]
  R --> H[Hybrid retrieval<br/>BM25 + vector + multi-query + rerank]
  V --> H
  C --> H
  H --> A[Answer LLM<br/>strict grounding]
  A --> U[Streamlit UI<br/>streaming + citations]
  D[discover_sources.py<br/>sitemap-based, robots.txt-respecting] --> S
  E[make eval<br/>30 cases × 5 metrics] --> RP[reports/eval-report.md]
```

## Run it locally

```bash
git clone https://github.com/YOUR_HANDLE/langchain-hdb-rag
cd langchain-hdb-rag
cp .env.example .env       # add OPENAI_API_KEY and GOOGLE_API_KEY
make install
make discover              # crawl HDB sitemap → data/sources.yaml (~2 min)
make ingest                # build vector store + chunks.jsonl (~5 min, ~$0.20 OpenAI)
make run                   # streamlit at localhost:8501
```

To re-run evaluation:

```bash
make eval                  # writes reports/eval-report.md
```

The eval runner is resumable — partial results are appended to `reports/eval_results_<mode>.jsonl` after each case. Delete that file to start fresh.

## Trade-offs / what I'd change for production

- **Wipe-and-reload ingest** is simple and bounded; production would use hash-based incremental updates against `Last-Modified` headers.
- **`gemma-4-31b-it` for everything** keeps it free but adds latency in `best` mode (3 LLM calls per query). Production would split: a small Gemini Flash for query rewriting/reranking, the bigger model only for the final answer.
- **LLM-listwise rerank** is convenient but expensive; a dedicated cross-encoder (e.g., Cohere Rerank) would be faster and cheaper at scale.
- **Sitemap-based discovery** is auditable but stale-prone; production would re-discover on a schedule and diff against the committed YAML.

## Sources

[`data/sources.yaml`](data/sources.yaml) — the full, committed list of indexed HDB pages.

## Stack

Python 3.13 · LangChain (LCEL) · Gemini API (Gemma 4 31B) · OpenAI text-embedding-3-small · Chroma / pgvector · Streamlit · BM25 · LangSmith.

## License

[MIT](LICENSE)
