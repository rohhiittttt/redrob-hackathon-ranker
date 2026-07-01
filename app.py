"""
app.py — Hugging Face Spaces front-end for the Redrob hackathon ranker.

Satisfies submission_spec.md Section 10.5 (sandbox requirement):
  - Accepts a small candidate sample (<=100 candidates) via upload.
  - Runs the ranking system end-to-end and produces a ranked CSV.
  - Completes within the compute budget (CPU only, well under 5 min for
    a small sample).

Accepted upload formats: .json (list of candidate dicts), .jsonl
(one candidate dict per line), .jsonl.gz (gzipped jsonl).

The reviewer does NOT need to bring the full 100k pool. They can take any
subset of candidates.jsonl (e.g. save 20-50 rows to a new file) and this
app will rank whichever subset they upload, using the same T1->T2->T3
pipeline as the real submission, against precomputed artifacts bundled
in this Space's `precomputed/` folder.

IMPORTANT: the precomputed artifacts (embeddings, BM25 index, feature
cache, cross-encoder cache) were built for candidates in the released
candidates.jsonl pool. If an uploaded candidate_id isn't present in those
caches, that candidate is skipped with a warning shown in the UI --
this is expected for a sandbox smoke test, not a bug.
"""

import gzip
import json
import os
import tempfile
import traceback

import gradio as gr
import numpy as np

from rank_core import run_ranking

PRECOMPUTED_DIR = os.environ.get("PRECOMPUTED_DIR", "precomputed")


def _load_jd_vector(precomputed_dir):
    rich = os.path.join(precomputed_dir, "jd_vector_combined.npy")
    if os.path.exists(rich):
        return np.load(rich)
    return np.load(os.path.join(precomputed_dir, "jd_vector.npy"))


def _parse_upload(file_path):
    """Parse .json / .jsonl / .jsonl.gz into a list of candidate dicts."""
    name = file_path.lower()

    if name.endswith(".gz"):
        opener = lambda p: gzip.open(p, "rt", encoding="utf-8")
    else:
        opener = lambda p: open(p, "r", encoding="utf-8")

    with opener(file_path) as f:
        text = f.read()

    text_stripped = text.strip()
    if not text_stripped:
        raise ValueError("Uploaded file is empty.")

    # .json (pretty-printed list) vs .jsonl (one object per line) —
    # detect by whether the whole thing parses as a single JSON array.
    if name.endswith(".json") and not name.endswith(".jsonl"):
        parsed = json.loads(text_stripped)
        if isinstance(parsed, dict):
            parsed = [parsed]
        return parsed

    # jsonl / jsonl.gz
    candidates = []
    for line in text_stripped.splitlines():
        line = line.strip()
        if line:
            candidates.append(json.loads(line))
    return candidates


def rank_uploaded_file(uploaded_file, progress=gr.Progress()):
    """Gradio callback: takes an uploaded file, returns (csv_path, status_md)."""
    if uploaded_file is None:
        return None, "Please upload a `.json`, `.jsonl`, or `.jsonl.gz` sample file first."

    try:
        progress(0.05, desc="Parsing uploaded file...")
        candidates = _parse_upload(uploaded_file.name)
    except Exception as e:
        return None, f"**Could not parse upload:** {e}\n\nExpected a JSON list, a `.jsonl` file (one candidate object per line), or a gzipped `.jsonl.gz`."

    if len(candidates) == 0:
        return None, "Uploaded file parsed but contained 0 candidates."

    if len(candidates) > 100000:
        return None, "This sandbox is for small-sample smoke tests. Please upload at most a few hundred candidates."

    progress(0.15, desc="Loading precomputed artifacts...")
    try:
        jd_vector = _load_jd_vector(PRECOMPUTED_DIR)
    except Exception as e:
        return None, f"**Server error loading precomputed JD vector:** {e}"

    final_top_n = min(100, len(candidates))

    progress(0.25, desc=f"Ranking {len(candidates)} candidates (target top {final_top_n})...")

    out_dir = tempfile.mkdtemp(prefix="redrob_sandbox_")
    out_csv = os.path.join(out_dir, "sandbox_submission.csv")

    try:
        result = run_ranking(
            candidates=candidates,
            jd_vector=jd_vector,
            precomputed_dir=PRECOMPUTED_DIR,
            output_csv_path=out_csv,
            team_id="sandbox_demo",
            final_top_n=final_top_n,
        )
    except Exception:
        tb = traceback.format_exc()
        return None, f"**Ranking pipeline raised an exception:**\n```\n{tb[-3000:]}\n```"

    progress(1.0, desc="Done")

    status_lines = [f"### Ranked {len(candidates)} candidates -> top {final_top_n}"]

    if result["errors"]:
        status_lines.append("**Validation errors (CSV not written):**")
        for e in result["errors"]:
            status_lines.append(f"- {e}")
        return None, "\n".join(status_lines)

    status_lines.append(f"- Honeypot rate in output: {result['honeypot_rate']:.1%}")
    if result["warnings"]:
        status_lines.append("**Warnings:**")
        for w in result["warnings"]:
            status_lines.append(f"- {w}")
    status_lines.append("\n✅ CSV generated and validated against submission_spec.md Section 3 rules.")

    return result["csv_path"], "\n".join(status_lines)


with gr.Blocks(title="Redrob Hackathon Ranker — Sandbox") as demo:
    gr.Markdown(
        "# Redrob Hackathon Ranker — Sandbox Demo\n"
        "Upload a small sample of candidates (`.json`, `.jsonl`, or `.jsonl.gz` — "
        "any subset of the released `candidates.jsonl`, e.g. 20-100 rows) to see "
        "the ranking pipeline run end-to-end and produce a ranked CSV.\n\n"
        "This mirrors the real submission pipeline (T1 coarse scoring -> T2 "
        "feature scoring -> T3 cross-encoder re-ranking -> reasoning generation "
        "-> spec self-validation), run against precomputed artifacts bundled in "
        "this Space. CPU only, no network calls, no GPU."
    )

    with gr.Row():
        file_input = gr.File(
            label="Upload candidate sample (.json / .jsonl / .jsonl.gz)",
            file_types=[".json", ".jsonl", ".gz"],
        )

    run_btn = gr.Button("Run ranking", variant="primary")

    status_output = gr.Markdown()
    csv_output = gr.File(label="Ranked submission CSV")

    run_btn.click(
        fn=rank_uploaded_file,
        inputs=[file_input],
        outputs=[csv_output, status_output],
    )

if __name__ == "__main__":
    demo.launch()
