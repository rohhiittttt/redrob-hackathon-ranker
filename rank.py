import argparse, json, gzip
from pathlib import Path
import numpy as np
from rank_core import run_ranking

def load_candidates(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--precomputed', default='precomputed')
    args = ap.parse_args()

    candidates = load_candidates(args.candidates)
    jd_vector = np.load(f'{args.precomputed}/jd_vector_combined.npy')
    run_ranking(candidates, jd_vector, args.precomputed, args.out, final_top_n=100)