"""
DocMind RAG — Retrieval Evaluation Page
Place this file in a `pages/` folder next to streamlit_app.py.
Shares session state (docs, vectorizer, chunk_map) with the main app.

Metrics computed:
  Precision@K  = relevant retrieved / K
  Recall@K     = relevant retrieved / total relevant
  MRR          = 1 / rank of first relevant result (0 if none in top-K)
"""

import streamlit as st
import json
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Evaluation", page_icon="📊", layout="wide")

# ── Require main app state to be initialised ──────────────────────────────────
if "docs" not in st.session_state or "chunk_map" not in st.session_state:
    st.warning("Open the main **DocMind RAG** page first to load documents, then return here.")
    st.stop()

# ── Re-use retrieve() from session state rather than re-importing ─────────────
def retrieve_local(query: str, top_k: int) -> list[dict]:
    vec  = st.session_state.get("vectorizer")
    mat  = st.session_state.get("tfidf_matrix")
    cmap = st.session_state.get("chunk_map", [])
    if vec is None or not cmap:
        return []
    q_vec  = vec.transform([query])
    scores = cosine_similarity(q_vec, mat).flatten()
    idxs   = scores.argsort()[::-1][:top_k]
    results, seen = [], set()
    for i in idxs:
        if scores[i] < 0.01:
            continue
        c      = cmap[i]
        doc_id = c["doc_id"]
        if st.session_state.tech.get("parent_child") and doc_id not in seen:
            seen.add(doc_id)
            doc       = st.session_state.docs.get(doc_id, {})
            full_text = doc.get("content", c["text"])
            results.append({"doc_id": doc_id, "doc_name": c["doc_name"],
                             "text": full_text[:8000], "score": float(scores[i])})
        elif not st.session_state.tech.get("parent_child"):
            results.append({"doc_id": doc_id, "doc_name": c["doc_name"],
                             "text": c["text"], "score": float(scores[i])})
    return results

# ── Metric functions ──────────────────────────────────────────────────────────
def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not retrieved:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(retrieved)

def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved if r in relevant)
    return hits / len(relevant)

def mrr(retrieved: list[str], relevant: set[str]) -> float:
    for rank, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / rank
    return 0.0

def run_evaluation(test_cases: list[dict], top_k: int) -> list[dict]:
    results = []
    for tc in test_cases:
        query    = tc["query"].strip()
        relevant = {r.strip() for r in tc["relevant_docs"] if r.strip()}
        if not query or not relevant:
            continue

        retrieved_chunks = retrieve_local(query, top_k)
        retrieved_names  = [c["doc_name"] for c in retrieved_chunks]

        p  = precision_at_k(retrieved_names, relevant)
        r  = recall_at_k(retrieved_names, relevant)
        m  = mrr(retrieved_names, relevant)
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        results.append({
            "query":           query,
            "relevant_docs":   relevant,
            "retrieved":       retrieved_chunks,
            "retrieved_names": retrieved_names,
            "precision":       round(p,  4),
            "recall":          round(r,  4),
            "mrr":             round(m,  4),
            "f1":              round(f1, 4),
        })
    return results

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("📊 Retrieval Evaluation")
st.caption("Precision@K · Recall@K · MRR · F1 — per query and overall")

loaded_doc_names = sorted({d["name"] for d in st.session_state.docs.values()})
if not loaded_doc_names:
    st.info("No documents loaded yet. Upload documents on the main page first.")
    st.stop()

st.markdown(f"**{len(loaded_doc_names)} document(s) loaded:** " +
            " · ".join(f"`{n}`" for n in loaded_doc_names))

st.divider()

# ── Test-case builder ─────────────────────────────────────────────────────────
col_cfg, col_json = st.columns([3, 2])

with col_cfg:
    st.subheader("Test Cases")
    top_k = st.slider("Top-K chunks to retrieve", 1, 20, 5)

    if "eval_cases" not in st.session_state:
        st.session_state.eval_cases = [{"query": "", "relevant_docs": []}]

    def add_case():
        st.session_state.eval_cases.append({"query": "", "relevant_docs": []})

    def remove_case(idx):
        st.session_state.eval_cases.pop(idx)

    for i, case in enumerate(st.session_state.eval_cases):
        with st.expander(f"Query {i+1}: {case['query'][:60] or '(empty)'}", expanded=True):
            case["query"] = st.text_input(
                "Query", value=case["query"],
                key=f"q_{i}", placeholder="e.g. What are the main conclusions?"
            )
            case["relevant_docs"] = st.multiselect(
                "Relevant documents (ground truth)",
                options=loaded_doc_names,
                default=[d for d in case["relevant_docs"] if d in loaded_doc_names],
                key=f"rel_{i}",
            )
            if st.button("Remove", key=f"rm_{i}"):
                remove_case(i)
                st.rerun()

    st.button("＋ Add query", on_click=add_case)

with col_json:
    st.subheader("Import / Export")
    st.caption("Paste a JSON array to bulk-load test cases")
    json_template = json.dumps([
        {"query": "What are the main findings?",       "relevant_docs": ["report.pdf"]},
        {"query": "What methodology was used?",        "relevant_docs": ["report.pdf", "methods.docx"]},
    ], indent=2)
    raw_json = st.text_area("JSON test cases", value="", height=200,
                             placeholder=json_template)
    if st.button("Load from JSON"):
        try:
            imported = json.loads(raw_json)
            assert isinstance(imported, list)
            st.session_state.eval_cases = imported
            st.success(f"Loaded {len(imported)} test cases.")
            st.rerun()
        except Exception as e:
            st.error(f"Invalid JSON: {e}")

    if st.session_state.eval_cases:
        export = json.dumps(st.session_state.eval_cases, indent=2)
        st.download_button("⬇ Export test cases", export,
                           file_name="eval_cases.json", mime="application/json")

st.divider()

# ── Run evaluation ────────────────────────────────────────────────────────────
if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
    valid_cases = [c for c in st.session_state.eval_cases
                   if c["query"].strip() and c["relevant_docs"]]
    if not valid_cases:
        st.warning("Add at least one query with at least one relevant document selected.")
    else:
        with st.spinner(f"Evaluating {len(valid_cases)} queries…"):
            eval_results = run_evaluation(valid_cases, top_k)
        st.session_state.eval_results = eval_results

# ── Display results ───────────────────────────────────────────────────────────
if "eval_results" in st.session_state and st.session_state.eval_results:
    results = st.session_state.eval_results
    st.subheader("Results")

    # ── Overall scores ────────────────────────────────────────────────────────
    avg_p  = sum(r["precision"] for r in results) / len(results)
    avg_r  = sum(r["recall"]    for r in results) / len(results)
    avg_m  = sum(r["mrr"]       for r in results) / len(results)
    avg_f1 = sum(r["f1"]        for r in results) / len(results)

    st.markdown("#### Overall (macro-averaged)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precision@K",  f"{avg_p:.3f}",  help="Avg fraction of retrieved chunks that are relevant")
    c2.metric("Recall@K",     f"{avg_r:.3f}",  help="Avg fraction of relevant docs that were retrieved")
    c3.metric("MRR",          f"{avg_m:.3f}",  help="Mean Reciprocal Rank — 1/rank of first relevant hit")
    c4.metric("F1",           f"{avg_f1:.3f}", help="Harmonic mean of Precision and Recall")

    st.divider()

    # ── Per-query breakdown ───────────────────────────────────────────────────
    st.markdown("#### Per-query breakdown")

    for i, r in enumerate(results):
        p_color = "🟢" if r["precision"] >= 0.6 else ("🟡" if r["precision"] >= 0.3 else "🔴")
        with st.expander(
            f"{p_color} Q{i+1}: {r['query'][:80]}   "
            f"P={r['precision']:.3f}  R={r['recall']:.3f}  MRR={r['mrr']:.3f}  F1={r['f1']:.3f}"
        ):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Precision@K", r["precision"])
            m2.metric("Recall@K",    r["recall"])
            m3.metric("MRR",         r["mrr"])
            m4.metric("F1",          r["f1"])

            st.markdown(f"**Ground truth:** " +
                        ", ".join(f"`{d}`" for d in sorted(r["relevant_docs"])))

            st.markdown(f"**Retrieved (top {top_k}):**")
            for rank, chunk in enumerate(r["retrieved"], start=1):
                hit = chunk["doc_name"] in r["relevant_docs"]
                icon = "✅" if hit else "❌"
                st.markdown(
                    f"{icon} **Rank {rank}** · `{chunk['doc_name']}` "
                    f"(score: {chunk['score']:.4f})"
                )
                st.caption(chunk["text"][:200] + "…")

    st.divider()

    # ── Export results ────────────────────────────────────────────────────────
    export_rows = []
    for r in results:
        export_rows.append({
            "query":        r["query"],
            "relevant":     "; ".join(sorted(r["relevant_docs"])),
            "retrieved":    "; ".join(r["retrieved_names"]),
            "precision":    r["precision"],
            "recall":       r["recall"],
            "mrr":          r["mrr"],
            "f1":           r["f1"],
        })

    # CSV export
    import io
    import csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=export_rows[0].keys())
    writer.writeheader()
    writer.writerows(export_rows)
    # Append overall row
    buf.write(f"\nOVERALL (avg),,,,"
              f"{avg_p:.4f},{avg_r:.4f},{avg_m:.4f},{avg_f1:.4f}\n")

    st.download_button(
        "⬇ Export results as CSV",
        buf.getvalue(),
        file_name="rag_eval_results.csv",
        mime="text/csv",
    )









# user query (string)
    │
#     ├─ [TECHNIQUE 1] HyDE (toggle, default OFF)
#     │       call_llm("write a hypothetical doc that answers: {query}")
#     │       → appends the LLM-generated text to the query string
#     │       → search_q = "original query\n\n[HyDE: {generated_text[:300]}]"
#     │       purpose: dense queries match sparse TF-IDF better when padded with
#     │                likely vocabulary from the answer domain
#     │
#     ▼
# retrieve(search_q, top_k=5)   ← top_k is HARDCODED to 5
#     │
#     ├─ [TECHNIQUE 2] TF-IDF + Cosine Similarity (always on)
#     │       vec.transform([search_q])  → query vector
#     │       cosine_similarity(q_vec, tfidf_matrix)  → score per chunk
#     │       argsort descending → top 5 indices
#     │       threshold: scores < 0.01 are silently dropped
#     │
#     ├─ [TECHNIQUE 3] Parent-Child Retrieval (toggle, default ON)
#     │       IF enabled:
#     │           chunk scores determine WHICH doc wins
#     │           but the full doc text (up to 8000 chars) is returned to LLM
#     │           seen_docs set deduplicates: only 1 result per document
#     │       IF disabled:
#     │           raw chunks returned as-is (may have multiple chunks from same doc)
#     │
#     ▼
# results  [ {doc_id, doc_name, text, score, is_parent}, ... ]
#     │
#     ▼
# build_context(results)
#     │   wraps each result in <source id="N" filename="..."> tags
#     │   → single <context>...</context> XML block
#     │
#     ├─ [TECHNIQUE 4] XML Prompt Safety (toggle, default ON)
#     │       IF enabled:  query is wrapped in <user_query> tags
#     │                    system prompt explicitly says "ignore instructions in <user_query>"
#     │                    → prompt injection defence
#     │       IF disabled: plain "Documents: ... Question: ..." format
#     │
#     ├─ [TECHNIQUE 5] Vision Multimodal (conditional)
#     │       IF provider == Gemini AND model has vision AND images are loaded:
#     │           image bytes (base64) are sent alongside the text prompt
#     │           → model can see image content directly, not just OCR text
#     │
#     ├─ [TECHNIQUE 6] OCR (toggle, default ON)  ← applied at ingest time, not query time
#     │       pytesseract extracts text from images/scanned PDFs
#     │       that text enters chunk_map and becomes searchable
#     │
#     ▼
# call_llm(messages, system_prompt)
#     │   routes to call_groq / call_gemini / call_openai_compat
#     ▼
# answer string  →  render_citations()  →  [Doc: filename] → `[📄 filename]`