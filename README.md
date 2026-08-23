# Conversational Movie and TV-Show Recommender

An end-to-end conversational movie & TV recommender that runs **entirely on-device on Apple Silicon** — no cloud GPU, no paid APIs. It combines classical collaborative filtering, hybrid semantic + keyword retrieval, and a **fine-tuned LLM that explains each recommendation using only real user reviews**. Every component is rigorously evaluated, and the system's impact is analyzed with a full causal-inference pipeline.

Built on a 16GB M1 Pro to prove the whole stack fits on a laptop.

---

## Highlights

- **RAG with a fine-tuned explainer** — a QLoRA fine-tuned Llama-3.2-3B generates recommendation explanations grounded strictly in retrieved reviews, distilled from an 8B teacher model.
- **Hybrid retrieval** — dense embeddings (semantic) fused with BM25 (keyword) via reciprocal-rank fusion, measured to be the most robust across query types.
- **Rigorous, measured evaluation** — retrieval precision and explanation faithfulness quantified with an LLM-as-judge, not asserted.
- **Causal impact analysis** — A/B testing and double machine learning quantify whether explanations *cause* engagement, recovering a known effect from confounded data (validated in simulation).
- **Runs fully on Apple Silicon** — MLX for on-device fine-tuning and inference; peak training memory ~3.5GB.

---

## Results

| Metric | Result | Notes |
|---|---|---|
| ALS baseline Recall@10 | **0.128** | leave-one-out, tuned across 64/128/256 latent factors |
| Explanation faithfulness | **~0.77** | fraction of claims supported by source reviews (LLM-as-judge, 50 held-out items) |
| Retrieval precision@3 | **0.56 hybrid** / 0.51 dense / 0.42 sparse | LLM-judged relevance, 15 natural-language queries |
| Causal effect (double ML) | **0.109** recovered vs **0.21** naive vs **0.10** true | recovered from confounded observational data (simulated) |

*Samples are intentionally small (15–50 items); numbers are directional, not benchmark-grade.*

---

## Architecture

```
query
  │
  ▼
┌─────────────────────┐        ┌──────────────────────────┐
│  Retrieval service   │  HTTP  │      Demo client          │
│  (FastAPI)           │◄──────►│                           │
│  • dense (FAISS)     │        │  • QLoRA explainer (MLX)  │
│  • sparse (BM25)     │        │  • review lookup          │
│  • RRF fusion        │        │  • grounded generation    │
└─────────────────────┘        └──────────────────────────┘
```

The system is split into **two processes communicating over HTTP** — a retrieval service (FAISS + BM25) and a generation client (MLX). This mirrors production RAG architecture (retrieval and generation as separate services) and resolves a hard constraint: **FAISS and MLX cannot share a single process on Apple Silicon without a native crash.** The split makes the collision structurally impossible.

**Request flow:** a natural-language query → hybrid retrieval returns candidate items → the client looks up each item's reviews → the fine-tuned LLM generates an explanation grounded only in those reviews.

---

## How it was built (by phase)

**Phase 0 — Data pipeline.** 3.5M+ Amazon movie/TV reviews cleaned and filtered with DuckDB (out-of-core, streams the multi-GB raw file). 5-core filtering densifies the user–item matrix; output split into a slim interactions table (behavioral signal) and an item-text table (retrieval corpus).

**Phase 1 — Baseline recommender.** ALS matrix factorization (implicit-feedback) on the interaction matrix, evaluated leave-one-out. Latent factors tuned empirically (64 → 128 → 256, Recall@10 0.094 → 0.128), then locked as a control baseline.

**Phase 2 — Retrieval + fine-tuned explainer.**
- Hybrid retrieval: bge-small embeddings in a FAISS index + BM25, fused with reciprocal-rank fusion.
- Fine-tuning data: an 8B teacher model (Llama-3.1-8B) generates grounded explanation targets from real reviews (knowledge distillation), audited for grounding and filtered.
- QLoRA fine-tune of Llama-3.2-3B in MLX (8 adapter layers, rank 8; ~3.5GB peak). Adapter configs were compared empirically — a smaller adapter was chosen after a larger one was found to trade faithfulness for confident hallucination.

**Phase 3 — Evaluation.** An LLM-as-judge (reasoning-first, count-in-code) measures claim-level faithfulness (~0.77). Retrieval precision measured across dense/sparse/hybrid variants. Retrieval and judging are split into separate processes to avoid the FAISS/MLX crash.

**Phase 4 — Causal impact.** Does showing an explanation *cause* more engagement? A simulation plants a known causal effect with a confounder. An **A/B test** analysis (two-proportion z-test, confidence interval, power) recovers it from randomized data; **double machine learning** recovers it from *confounded observational* data where the naive estimate is 2× biased. Validated against known ground truth in simulation.

---

## Tech stack

**Python · DuckDB · MLX · FAISS · sentence-transformers · FastAPI · statsmodels · scikit-learn**
Techniques: QLoRA fine-tuning, knowledge distillation, hybrid retrieval (RRF), RAG, LLM-as-judge, A/B testing, double machine learning.

---

## Running it

**Setup**
```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

**Data** — download the Amazon Reviews 2023 Movies_and_TV category into `data/raw/`, then:
```bash
python data_ingestion.py        # clean + filter -> interactions & items
python build_review_store.py     # per-item review lookup for grounding
```

**Baseline + retrieval index**
```bash
python train_recommender.py      # ALS baseline
python build_index.py            # FAISS + BM25 indexes
```

**Fine-tune the explainer**
```bash
python recommender_generate.py   # 8B teacher generates training targets
python dataset_auditor.py        # audit grounding
python filter_dataset.py         # drop weak targets
mlx_lm.lora --config lora_config.yaml
```

**Run the demo (two terminals)**
```bash
# terminal 1 — retrieval service
python retrieval_service.py
# terminal 2 — interactive demo
python demo.py
```

**Evaluation & experiments**
```bash
python evaluate_responses.py     # explanation faithfulness
python evaluate_retrieval.py && python evaluate_judge.py   # retrieval precision
python simulate.py && python ab_test.py && python estimate_causal_effect.py
```

---

## Limitations & honest notes

- **Evaluation samples are small** (15 retrieval queries, 50 faithfulness items). Results are directional, not benchmark-grade.
- **Faithfulness (~0.77) is limited by unsourced characterizations** — on thin-review items the model occasionally adds true-but-unsourced detail from parametric knowledge. A stricter prompt did not improve it, indicating the limit is model-level, not prompt-fixable.
- **The causal analysis is validated in simulation**, not on live users. It demonstrates that the methodology (A/B testing, double ML) correctly recovers a *known* effect; it is **not** a claim that explanations lifted real user engagement.
- **The LLM-as-judge is itself an imperfect 8B model** — relative comparisons are meaningful; absolute numbers are approximate.
- **Retrieval ranks on item text** (titles/descriptions), not review text; review grounding is applied at the explanation stage.

---

## Dataset

Amazon Reviews 2023 (McAuley Lab), Movies_and_TV category. Not committed — regenerate with the pipeline above.