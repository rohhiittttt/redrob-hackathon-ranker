# Redrob Hackathon Ranker — Sandbox Space

## What's here
- `app.py` — Gradio UI. Upload a `.json` / `.jsonl` / `.jsonl.gz` sample of
  candidates, it runs the ranking pipeline, returns a validated CSV.
- `rank_core.py` — the ranking pipeline itself (T1 → T2 → T3 → reasoning →
  self-validation), refactored from the original Colab notebook into a
  single callable function `run_ranking()`. Colab/Drive-specific code has
  been removed; TOP_K cutoffs at each stage now clamp to the size of the
  uploaded sample instead of assuming the full 100k pool.
- `precomputed/` — **you must add this.** Copy in the same artifacts your
  Colab pipeline reads from Google Drive:
  - `embeddings_rich.npy` (or `embeddings.npy`)
  - `embed_cids_rich.npy` (or `embed_cids.npy`)
  - `jd_vector_combined.npy` (or `jd_vector.npy`)
  - `bm25_index.pkl`
  - `feature_cache.json`
  - `cross_scores_blended_top200.json`
  - `chunk_embeddings.npy`

## Local test before deploying
```bash
pip install -r requirements.txt
python app.py
```
Then open the local Gradio URL, upload a small `.jsonl` slice (e.g. the
first 30 lines of `sample_candidates.json` reformatted as jsonl, or any
subset of `candidates.jsonl`), and confirm it returns a CSV with no
validation errors.

## Known limitation
The precomputed artifacts are keyed to the specific JD and the specific
100k candidate pool released for this hackathon. If someone uploads
candidate IDs that aren't in `feature_cache.json` / the embedding index,
those candidates will not be scorable — this is expected for a sandbox
smoke test (per spec Section 10.5, it only needs to prove reproducibility
on a small sample from the released pool, not handle arbitrary data).
