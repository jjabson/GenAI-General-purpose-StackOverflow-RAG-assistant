## GenAI StackOverflow RAG Assistant (FAISS + Reranker + Guardrails)

A production-style **Retrieval-Augmented Generation (RAG)** assistant trained on StackOverflow/StackExchange Q&A. The system retrieves 
relevant historical Q&A threads and generates concise answers grounded in retrieved sources, with deterministic citations, refusal 
behavior when evidence is weak, and a full evaluation suite (retrieval quality, reliability, and latency).

### Why this project

Developers spend a lot of time searching, opening multiple tabs, and comparing answers to solve technical issues. This project demonstrates 
how to build a general-purpose technical support assistant that:
* finds relevant Q&A fast,
* produces readable “top bullets” answers,
* avoids hallucinations via grounding + citations
* measures performance like a real ML/GenAI system.

### What this builds

**Architecture (Two-stage retrieval + generation)**
1. **Document construction**
    * For each question, select a single “best” answer:
        * preferred: accepted (selected=True)
        * fallback: highest pm_score
    * Clean HTML and create a single document: doc_text = Question + Top Answer
2. **Dense retrieval (fast)**
    * Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
    * Index: FAISS IndexFlatIP with normalized embeddings (cosine similarity via dot product)
3. **Cross-encoder reranking (accurate)**
    * Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
    * Pattern: FAISS candidates → rerank → top evidence
4. **LLM generation (grounded + controlled)**
    * LLM: microsoft/Phi-3-mini-4k-instruct loaded in 4-bit (BitsAndBytes) for Colab/T4
    * Prompt constraints: bullet-only outputs, no invented fields/tables, token-budget truncation
    * Guardrails:
         * **min_score gate** (refuse if evidence too weak)
         * **token truncation** (prevents context overflow)
         * **deterministic citations** per bullet
         * **grounding check** (each bullet must overlap with at least one source)

### Dataset
* Source: Hugging Face HuggingFaceH4/stack-exchange-preferences
* Sampled subset: 50,000 Q&A items (to keep embedding/indexing feasible in Colab)
  
  Note: the dataset contains diverse programming topics (Python, SQL, web, tooling, etc.) — this is a general StackOverflow assistant.

### Results (Key Metrics)

**Retrieval quality (leakage-reduced evaluation)**

**Evaluation settings:**
* eval_size: 300
* valid question length: question_clean > 30 chars
* query: first_sentence(question_clean)
* rerank candidate pool: faiss_k=20, evaluate top k=(1,3,5,10)
* context passed to LLM: context_k=3
* min_score gate: 0.2

### Hit@k (Baseline vs Reranked)
* Hit@1: 0.774 → 0.8767 (+0.1027)
* Hit@3: 0.838 → 0.8800 (+0.0420)
* Hit@5: 0.846 → 0.8800 (+0.0340)
* Hit@10: 0.864 → 0.8800 (+0.0160)

  Takeaway: reranking significantly improves top-ranked relevance, which matters most because only the top few sources are used during generation.

### Generation reliability (sample n=30, first-sentence queries)
* Answer rate: 0.8333
* Refusal rate: 0.1667
* Citation compliance: 1.0

  Takeaway: the assistant answers most queries while refusing low-evidence cases and maintaining consistent source attribution.

### Latency breakdown (n=30)
* FAISS retrieval: ~18 ms mean
* Reranking: ~160 ms mean
* Generation: ~12–13 s mean (dominant cost)
* End-to-end: ~12–13 s mean

  Takeaway: reranking adds modest overhead relative to generation; the main bottleneck is LLM inference.

### How to run

**Option 1: Google Colab (recommended)
1. Open notebooks/GenAI_StackOverflow_RAG.ipynb in Colab
2. Add a Hugging Face token to Colab secrets as HF_TOKEN (optional but recommended for rate limits)
3. Run top-to-bottom

### Option 2: Local

pip install -r requirements.txt
jupyter notebook

### Requirements

**Code dependencies:**
* datasets
* sentence-transformers
* faiss-cpu (or faiss-gpu if available)
* transformers, accelerate, bitsandbytes
* torch
* pandas, numpy, tqdm

  Tip: pin versions in requirements.txt if you want fully reproducible runs.

### Design choices
* Best-answer selection policy (accepted answer first, else highest score)
* HTML cleaning for better embeddings and cleaner prompt context
* Two-stage retrieval (fast recall + accurate rerank)
* Guardrails (refusal gate, token truncation, deterministic citations, grounding validation)
* Real evaluation (first-sentence queries to reduce leakage; latency + reliability reported)

### Risk & limitations
* StackOverflow answers can be outdated or contradictory (source quality risk).
* Hit@k measures retrieval correctness, not necessarily full answer correctness.
* Token-overlap grounding/citation attachment is heuristic (practical, not perfect semantic alignment).
* Latency is dominated by generation; real deployment would require faster inference or caching/streaming.

### Next steps (optional improvements)
* Hybrid retrieval (BM25 + dense embeddings) for code/URL-heavy questions
* Add a small human-rated eval set (20–50 queries) with correctness rubric
* Caching of embeddings + rerank results for repeated queries
* Deploy a minimal demo UI (Gradio) with streaming responses

**Author**

Jerome Jabson
