# Redrob Hackathon Ranker

Ranking pipeline for the Intelligent Candidate Discovery & Ranking Challenge —
scores and ranks candidates from the 100k-candidate pool against the job
description, producing the top 100 with per-candidate reasoning.

## What's here

- `rank.py` — **the single entry point.** Takes a candidates file, produces
  the ranked submission CSV. This is the command Stage 3 reproduction runs.
- `rank_core.py` — the ranking pipeline itself (`run_ranking()`): hard
  filters + soft penalties → T1 coarse scoring (100k → top 5,000) → honeypot
  detection → T2 feature scoring (5,000 → top 200) → T3 cross-encoder
  re-ranking (200 → top 100) → fact-grounded reasoning generation → spec
  self-validation.
- `precompute.py` — script used to generate everything in `precomputed/`
  (embeddings, BM25 index, feature cache, cross-encoder scores). Originally
  developed and run in Google Colab; this is the exported, Colab-independent
  version. Re-run this if the JD or candidate pool changes.
- `precomputed/` — the actual precomputed artifacts (embeddings, BM25 index,
  feature cache, cross-encoder scores) used to produce the submission,
  tracked via Git LFS:
  - `embeddings.npy` / `embeddings_rich.npy`
  - `embed_cids.npy` / `embed_cids_rich.npy`
  - `jd_vector.npy` / `jd_vector_combined.npy`
  - `bm25_index.pkl`
  - `feature_cache.json`
  - `cross_scores_blended_top200.json` (and related `cross_scores_*.json`)
  - `chunk_embeddings.npy`
- `requirements.txt` — dependencies for both `rank.py` and the sandbox app
  (Gradio is only used by the Hugging Face Space sandbox, not by `rank.py`
  itself).

## Setup (one-time, requires network)

```bash
pip install -r requirements.txt
```

## Reproduce the submission (offline, CPU-only, must finish within 5 minutes)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

`candidates.jsonl` is the organizer-provided 100k-candidate pool — it is
**not included in this repo** (not ours to redistribute; supply your own
copy from the hackathon bundle). `rank.py` also accepts a gzipped
`.jsonl.gz` file directly.

Optional flag if `precomputed/` is located somewhere other than the repo
root:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv --precomputed ./precomputed
```

Validate the output against the official spec before submitting:
```bash
python validate_submission.py submission.csv
```

## Regenerating precomputed artifacts

If you need to rebuild `precomputed/` from scratch (e.g. after a JD or
candidate-pool change):
```bash
python precompute.py
```
This performs the embedding generation, BM25 indexing, and cross-encoder
scoring steps. It is not subject to the 5-minute ranking-step time limit —
it's a one-time setup step, not part of the timed run.

## Sandbox demo

A live, browser-based version of this pipeline (upload a small candidate
sample, get back a ranked CSV) is hosted separately on Hugging Face Spaces —
see `submission_metadata.yaml` for the link. That deployment uses the same
`rank_core.py` pipeline via a Gradio front-end (`app.py`), which lives in
that Space's own repo, not here.

## Compute constraints this pipeline is designed to respect

- CPU only, no GPU
- ≤16GB RAM during the ranking step
- No network calls during the ranking step (only during the one-time
  `pip install` / `precompute.py` setup)
- ≤5 minutes wall-clock for `rank.py` against the full 100k pool
