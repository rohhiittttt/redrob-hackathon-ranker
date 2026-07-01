"""
rank_core.py — Redrob hackathon ranking pipeline (Colab code stripped, HF-Spaces ready).

This is a refactor of the original Colab notebook logic into a single
callable function, `run_ranking()`, so it can be driven by app.py (Gradio)
on Hugging Face Spaces instead of a mounted Google Drive.

Key changes from the Colab version:
  - No `google.colab.drive.mount()` — precomputed artifacts are read from a
    local `precomputed_dir` bundled in the repo instead of Google Drive.
  - `candidates` is now a function argument (list of dicts already parsed
    from whatever the reviewer uploaded), not read from a fixed Drive path.
  - TOP_K / TOP_K_T2 / TOP_K_T3 are adaptive: they clamp to the size of the
    uploaded sample so a 30- or 50-candidate sandbox test doesn't crash
    trying to select "top 5000" out of 30 candidates.
  - `final_top_n` controls how many rows the final CSV must have. Real
    submissions (100k pool) still use 100 per the spec. The sandbox demo
    should pass `final_top_n=min(100, len(candidates))` so small samples
    still validate cleanly.
"""

import os
from pathlib import Path


def run_ranking(candidates, jd_vector, precomputed_dir, output_csv_path,
                 team_id='sandbox_test', final_top_n=100):
    """
    Run the full T1 -> T2 -> T3 ranking pipeline on `candidates` and write
    a submission-spec-compliant CSV to `output_csv_path`.

    Args:
        candidates: list[dict] — parsed candidate records (already loaded
            from the uploaded .json / .jsonl / .jsonl.gz — see app.py).
        jd_vector: np.ndarray — precomputed JD embedding vector. Loaded by
            the caller from `precomputed_dir` (kept as an explicit arg so
            it's obvious this pipeline is JD-specific, not general-purpose).
        precomputed_dir: str — path to a directory bundled in the HF Space
            repo containing embeddings_rich.npy, embed_cids_rich.npy,
            bm25_index.pkl, feature_cache.json,
            cross_scores_blended_top200.json, chunk_embeddings.npy.
        output_csv_path: str — where to write the final ranked CSV.
        team_id: str — used only for logging / labeling, not the filename
            (the caller decides the actual output filename).
        final_top_n: int — how many ranked rows the output CSV must contain.
            Use 100 for the real submission; for a small sandbox sample use
            min(100, len(candidates)) so validation doesn't fail on an
            intentionally tiny demo input.

    Returns:
        dict with keys: 'rows' (the final ranked rows), 'errors', 'warnings',
        'honeypot_rate', 'csv_path'.
    """
    PRECOMPUTED = precomputed_dir
    OUTPUT_DIR  = str(Path(output_csv_path).parent)
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Single install for all phases ──────────────────────────────────────────
    # sentence-transformers is no longer needed — Phase B removed the live
    # CrossEncoder in favor of precomputed cached scores (cross_score_cache).
    # rank_bm25 is still required (BM25Okapi is used directly + needed to
    # unpickle bm25_index.pkl), but this environment may have no internet
    # access (e.g. Stage 3 offline grading), so only install if it's missing.

    # ── All imports (single block) ─────────────────────────────────────────────
    import csv
    import heapq
    import json
    import re
    import numpy as np
    import pickle
    from datetime import datetime
    from dataclasses import dataclass, field
    from pathlib import Path
    from rank_bm25 import BM25Okapi
    import rank_bm25

    TODAY = datetime(2026, 6, 28)

    # ── Candidates are passed in directly (already parsed dicts) ───────────────
    print(f'Loaded {len(candidates)} candidates')

    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 0 — Load + Build
    # ═══════════════════════════════════════════════════════════════════════════

    print('[Phase 0] Loading precomputed artifacts...')

    # 0.1 — Embeddings
    try:
        embeddings = np.load(f'{PRECOMPUTED}/embeddings_rich.npy')
        embed_cids = np.load(f'{PRECOMPUTED}/embed_cids_rich.npy', allow_pickle=True)
        print(f'  Loaded rich embeddings      : {embeddings.shape}')
    except FileNotFoundError:
        embeddings = np.load(f'{PRECOMPUTED}/embeddings.npy')
        embed_cids = np.load(f'{PRECOMPUTED}/embed_cids.npy', allow_pickle=True)
        print(f'  Loaded original embeddings  : {embeddings.shape}')

    # 0.2 — JD vector
    try:
        jd_vector = np.load(f'{PRECOMPUTED}/jd_vector_combined.npy')
        print(f'  Loaded combined JD vector   : {jd_vector.shape}')
    except FileNotFoundError:
        jd_vector = np.load(f'{PRECOMPUTED}/jd_vector.npy')
        print(f'  Loaded original JD vector   : {jd_vector.shape}')

    # 0.3 — BM25 index
    with open(f'{PRECOMPUTED}/bm25_index.pkl', 'rb') as f:
        bm25_data  = pickle.load(f)
        bm25_index = bm25_data['index']
        bm25_cids  = bm25_data['cids']
    print(f'  Loaded BM25 index           : {len(bm25_cids)} candidates')

    # 0.4 — Feature cache
    with open(f'{PRECOMPUTED}/feature_cache.json') as f:
        feature_cache = json.load(f)
    print(f'  Loaded feature cache        : {len(feature_cache)} candidates')

    # 0.4b — Cross encoder blended cache (replaces live CrossEncoder model)
    with open(f'{PRECOMPUTED}/cross_scores_blended_top200.json') as f:
        cross_score_cache = json.load(f)
    print(f'  Loaded cross score cache    : {len(cross_score_cache)} candidates')

    # 0.5 — Index maps
    embed_index_map = {cid: idx for idx, cid in enumerate(embed_cids)}
    bm25_index_map  = {cid: idx for idx, cid in enumerate(bm25_cids)}
    print(f'  Built index maps            : ready')

    # ═══════════════════════════════════════════════════════════════════════════
    # 0.6 — AI Taxonomy
    # ═══════════════════════════════════════════════════════════════════════════

    print('[Phase 0] Building AI taxonomy...')

    AI_TAXONOMY = {
        'embeddings_retrieval': [
            'sentence-transformers', 'sbert', 'bge', 'e5', 'openai embeddings',
            'embedding', 'dense retrieval', 'semantic search', 'vector search',
            'neural retrieval', 'bi-encoder', 'dense vector'
        ],
        'vector_db': [
            'pinecone', 'weaviate', 'qdrant', 'milvus', 'faiss', 'opensearch',
            'elasticsearch', 'vector database', 'vector store', 'ann index',
            'approximate nearest neighbor', 'hnsw', 'annoy'
        ],
        'ranking_eval': [
            'ndcg', 'mrr', 'map', 'a/b test', 'ranking system', 'learning to rank',
            'ltr', 'xgboost rank', 'lambdamart', 'offline evaluation',
            'online evaluation', 'retrieval quality', 'mean average precision'
        ],
        'production_ml': [
            'production', 'deployed', 'serving', 'model serving', 'inference',
            'real users', 'model drift', 'monitoring', 'mlops', 'pipeline',
            'at scale', 'low latency'
        ],
        'llm_finetuning': [
            'lora', 'qlora', 'peft', 'fine-tuning', 'finetuning',
            'instruction tuning', 'rlhf', 'sft', 'parameter efficient'
        ],
        'recommendation': [
            'recommendation', 'recommender', 'collaborative filtering',
            'content-based', 'hybrid recommendation', 'candidate generation',
            'two-tower', 'matrix factorization'
        ],
        'hybrid_search': [
            'hybrid search', 'bm25', 'sparse dense', 'rrf',
            'reciprocal rank fusion', 'sparse retrieval', 'keyword search'
        ],
        'consulting_only': [
            'tcs', 'infosys', 'wipro', 'accenture', 'cognizant',
            'capgemini', 'hcl', 'tech mahindra', 'mphasis'
        ],
        'wrong_domain': [
            'computer vision', 'object detection', 'image segmentation',
            'speech recognition', 'asr', 'text to speech', 'tts',
            'robotics', 'autonomous driving', 'lidar'
        ],
        'research_only': [
            'phd researcher', 'research scientist', 'lab researcher',
            'postdoc', 'postdoctoral', 'academic research'
        ]
    }

    def compute_taxonomy_hits(cid):
        f    = feature_cache[cid]
        text = (
            f['headline'] + ' ' +
            f['summary']  + ' ' +
            ' '.join(f['skill_names']) + ' ' +
            f['all_desc']
        ).lower()
        return {
            category: sum(1 for term in terms if term in text)
            for category, terms in AI_TAXONOMY.items()
        }

    taxonomy_cache = {cid: compute_taxonomy_hits(cid) for cid in feature_cache}
    print(f'  Taxonomy cache built        : {len(taxonomy_cache)} candidates')

    # ═══════════════════════════════════════════════════════════════════════════
    # 0.7 — Behavioral Multipliers
    # ═══════════════════════════════════════════════════════════════════════════

    print('[Phase 0] Computing behavioral multipliers...')

    def compute_behavioral_multiplier(cid):
        f     = feature_cache[cid]
        score = 1.0

        if f['otw']:                              score *= 1.10
        if   f['inactive_days'] <= 7:             score *= 1.10
        elif f['inactive_days'] <= 30:            score *= 1.00
        elif f['inactive_days'] <= 90:            score *= 0.85
        else:                                     score *= 0.60

        rr = f['resp_rate']
        if   rr >= 0.8:                           score *= 1.08
        elif rr >= 0.6:                           score *= 1.00
        elif rr >= 0.4:                           score *= 0.80
        elif rr >= 0.2:                           score *= 0.60
        else:                                     score *= 0.40
        nd = f['notice_days']
        if   nd <= 15:                            score *= 1.05
        elif nd <= 30:                            score *= 1.00
        elif nd <= 45:                            score *= 0.95
        elif nd <= 60:                            score *= 0.85
        elif nd <= 90:                            score *= 0.70
        else:                                     score *= 0.50

        if   f['github_score'] >= 60:             score *= 1.05
        elif f['github_score'] == -1:             score *= 0.98

        ir = f['interview_rate']
        if   ir >= 0.9:                           score *= 1.04
        elif ir >= 0.7:                           score *= 1.00
        elif ir >= 0.5:                           score *= 0.80
        elif ir >= 0.4:                           score *= 0.70
        else :                                    score *= 0.50

        if   f['location_tier'] == 'preferred':   score *= 1.08
        elif f['location_tier'] == 'tier_1':      score *= 1.03
        elif f['willing_relocate']:               score *= 1.01

        return round(min(score, 1.5), 4)

    behavioral_multipliers = {
        cid: compute_behavioral_multiplier(cid)
        for cid in feature_cache
    }
    print(f'  Behavioral multipliers      : {len(behavioral_multipliers)} candidates')

    print('\n[Phase 0] Complete. In-memory artifacts:')
    print(f'  embeddings        : {embeddings.shape}')
    print(f'  jd_vector         : {jd_vector.shape}')
    print(f'  feature_cache     : {len(feature_cache)} candidates')
    print(f'  taxonomy_cache    : {len(taxonomy_cache)} candidates')
    print(f'  behavioral_mult   : {len(behavioral_multipliers)} candidates')
    print(f'  embed_index_map   : {len(embed_index_map)} entries')
    print(f'  bm25_index_map    : {len(bm25_index_map)} entries')
    print('  Ready for Phase 1.\n')

    # ============================================================
    # PHASE 1 — Hard Filters + Soft Penalties
    # ============================================================

    TITLE_CHASER_THRESHOLD_MONTHS = 18

    FRAMEWORK_TERMS = [
        'langchain', 'llamaindex', 'autogpt', 'crewai', 'flowise',
        'chainlit', 'langflow', 'haystack', 'semantic kernel',
        'auto-gpt', 'agentgpt', 'superagi'
    ]

    CORE_SKILL_TERMS = [
        'embedding', 'vector', 'retrieval', 'faiss', 'pinecone',
        'weaviate', 'qdrant', 'milvus', 'opensearch', 'elasticsearch',
        'bm25', 'ranking', 'reranking', 're-ranking', 'ndcg', 'mrr',
        'sentence-transformers', 'sbert', 'bge', 'dense retrieval',
        'semantic search', 'hybrid search'
    ]

    ARCHITECT_TITLES = [
        'architect', 'tech lead', 'technical lead', 'engineering manager',
        'vp engineering', 'vp of engineering', 'head of engineering',
        'director of engineering', 'principal engineer', 'distinguished engineer'
    ]

    CODING_SIGNALS = [
        'implemented', 'built', 'developed', 'wrote', 'coded', 'shipped',
        'engineered', 'programmed', 'designed and implemented', 'refactored',
        'optimized', 'deployed', 'integrated', 'created', 'authored'
    ]

    RESEARCH_SIGNALS = [
        'phd', 'research scientist', 'research engineer', 'lab researcher',
        'postdoc', 'postdoctoral', 'academic', 'publication', 'published paper',
        'arxiv', 'peer reviewed', 'ieee', 'acm', 'neurips', 'icml', 'iclr'
    ]

    OPEN_SOURCE_SIGNALS = [
        'open source', 'opensource', 'open-source', 'github', 'contributed',
        'contribution', 'published', 'paper', 'arxiv', 'research paper',
        'conference', 'talk', 'speaker', 'blog', 'kaggle', 'huggingface'
    ]

    candidates_dict = {c['candidate_id']: c for c in candidates}

    def is_title_chaser(cid):
        cand = candidates_dict[cid]
        career = sorted(
            cand.get('career_history', []),
            key=lambda x: x.get('start_date', '1970-01-01')
        )
        if len(career) < 2:
            return False

        short_stints  = 0
        total_company_switches = 0

        for i in range(len(career) - 1):
            cur_role, next_role = career[i], career[i + 1]
            cur_company  = (cur_role.get('company', '') or '').strip().lower()
            next_company = (next_role.get('company', '') or '').strip().lower()
            if not cur_company or not next_company or cur_company == next_company:
                continue  # skip internal promotions, only count actual company switches

            total_company_switches += 1
            dur_months = int(cur_role.get('duration_months', 0) or 0)
            if dur_months < 18:
                short_stints += 1

        if total_company_switches == 0:
            return False

        return (short_stints / total_company_switches) >= 0.5
    def get_last_n_roles(cid, n=3):
        cand = candidates_dict[cid]
        career = cand.get('career_history', [])
        return sorted(career, key=lambda x: x.get('start_date', '1970-01-01'), reverse=True)[:n]

    def get_avg_tenure_last_n(cid, n=3):
        roles = get_last_n_roles(cid, n)
        if not roles:
            return 999
        tenures = [int(role.get('duration_months', 0) or 0) for role in roles]
        return sum(tenures) / len(tenures) if tenures else 999

    def get_roles_last_18_months(cid):
        cand   = candidates_dict[cid]
        career = cand.get('career_history', [])
        cutoff = datetime(2024, 12, 28)
        recent = []
        for role in career:
            try:
                if role.get('is_current', False):
                    recent.append(role)
                elif role.get('end_date'):
                    end = datetime.strptime(role['end_date'], '%Y-%m-%d')
                    if end >= cutoff:
                        recent.append(role)
            except:
                pass
        return recent

    def has_coding_signals_in_recent_roles(cid):
        recent = get_roles_last_18_months(cid)
        if not recent:
            return False
        combined = ' '.join(r.get('description', '').lower() for r in recent)
        return any(signal in combined for signal in CODING_SIGNALS)

    def count_terms_in_profile(cid, terms):
        f    = feature_cache[cid]
        text = (
            f['headline'] + ' ' +
            f['summary']  + ' ' +
            ' '.join(f['skill_names']) + ' ' +
            f['all_desc']
        ).lower()
        return sum(1 for term in terms if term in text)
    # ── Phase 1A — Hard Filters ────────────────────────────────────────────────

    print('[Phase 1A] Applying hard filters...')

    hard_filter_stats = {
        'title_chaser': 0, 'framework_enthusiast': 0,
        'consulting_only': 0, 'wrong_domain': 0,
    }

    def apply_hard_filters(cid):
        f   = feature_cache[cid]
        tax = taxonomy_cache[cid]

        if is_title_chaser(cid):
            return False, 'title_chaser'

        framework_hits  = count_terms_in_profile(cid, FRAMEWORK_TERMS)
        core_skill_hits = count_terms_in_profile(cid, CORE_SKILL_TERMS)
        production_hits = tax['production_ml']
        if framework_hits >= 3 and core_skill_hits <= 1 and production_hits == 0:
            return False, 'framework_enthusiast'

        if f['is_consulting_only']:
            return False, 'consulting_only'

        wrong_domain_hits = tax['wrong_domain']
        nlp_ir_hits       = tax['embeddings_retrieval'] + tax['ranking_eval']
        if wrong_domain_hits >= 1 and nlp_ir_hits == 0:
            return False, 'wrong_domain'

        return True, 'passed'

    passed_candidates = []
    filtered_out      = []

    for cid in feature_cache:
        passed, reason = apply_hard_filters(cid)
        if passed:
            passed_candidates.append(cid)
        else:
            filtered_out.append((cid, reason))
            hard_filter_stats[reason] += 1

    print(f'  Total candidates        : {len(feature_cache)}')
    print(f'  Passed hard filters     : {len(passed_candidates)}')
    print(f'  Knocked out             : {len(filtered_out)}')
    for k, v in hard_filter_stats.items():
        print(f'  {k}: {v}')

    # ── Phase 1B — Soft Penalties ─────────────────────────────────────────────

    print('\n[Phase 1B] Computing soft penalties...')

    def compute_soft_penalty_multiplier(cid):
        f    = feature_cache[cid]
        tax  = taxonomy_cache[cid]
        mult = 1.0

        open_source_hits = count_terms_in_profile(cid, OPEN_SOURCE_SIGNALS)
        if f['github_score'] == -1 and f['yoe'] >= 5 and open_source_hits == 0:
            mult *= 0.60

        research_text_hits = count_terms_in_profile(cid, RESEARCH_SIGNALS)
        if research_text_hits >= 2 and tax['production_ml'] == 0:
            mult *= 0.55

        current_title = f['title'].lower()
        is_architect_title = any(t in current_title for t in ARCHITECT_TITLES)
        if is_architect_title and f['yoe'] >= 5:
            if not has_coding_signals_in_recent_roles(cid):
                mult *= 0.65

        return round(mult, 4)

    soft_penalty_multipliers = {}
    for cid in passed_candidates:
        soft_penalty_multipliers[cid] = compute_soft_penalty_multiplier(cid)

    print(f'  Soft penalties computed  : {len(soft_penalty_multipliers)} candidates')
    print('[Phase 1] Complete. Ready for Phase 2.\n')

    # ============================================================
    # PHASE 2 — T1 SCORING (100k → Top 5000)
    # ============================================================

    WEIGHTS = {
        'w1': 0.40,  # career description score (RRF)
        'w2': 0.25,  # title alignment
        'w3': 0.10,  # experience (YOE)
        'w4': 0.20,  # skill match
        'w5': 0.05,  # python signal
    }

    TOP_K  = min(5000, len(passed_candidates))
    RRF_K  = 60
    CHUNK_SIZE = 3

    SKILL_TAXONOMY = {
        'cat1_embeddings': [
            'sentence-transformers', 'sbert', 'bge', 'bge-m3', 'bge-large', 'bge-base',
            'e5', 'e5-mistral', 'multilingual-e5', 'e5-large', 'e5-small',
            'qwen3-embedding', 'nv-embed', 'nv-embed-v2', 'stella', 'stella_en_1.5b_v5',
            'arctic-embed', 'snowflake-arctic-embed', 'jina-embeddings', 'jina',
            'gritlm', 'sfr-embedding', 'instructor', 'instructor-xl',
            'all-mpnet-base', 'all-minilm', 'paraphrase-mpnet',
            'gte', 'gte-large', 'gte-base', 'nomic-embed', 'nomic-embed-text',
            'text-embedding-3', 'text-embedding-ada', 'openai embeddings',
            'cohere embed', 'embed-v4', 'embed-multilingual',
            'voyage', 'voyage-3', 'voyage-large', 'voyage-code',
            'gemini embeddings', 'vertex ai embeddings',
            'amazon titan embeddings', 'bedrock embeddings',
            'matryoshka', 'mrl', 'contrastive learning', 'contrastive loss',
            'triplet loss', 'mnrl', 'siamese', 'bi-encoder',
            'embedding drift', 'embedding quantization', 'int8 quantization',
            'binary quantization', 'fine-tuning embeddings', 'embedding fine-tuning',
            'domain adaptation', 'sentence embedding', 'mean pooling', 'cls pooling',
        ],
        'cat2_vector_db': [
            'pinecone', 'weaviate', 'qdrant', 'milvus', 'zilliz',
            'chroma', 'chromadb', 'marqo', 'vald', 'turbopuffer', 'lancedb', 'vespa',
            'pgvector', 'pg vector', 'postgresql vector',
            'redis', 'redisearch', 'redis vector',
            'mongodb atlas vector', 'atlas vector search',
            'cassandra vector', 'astra db', 'singlestore',
            'elasticsearch', 'elastic search', 'opensearch', 'open search',
            'apache solr', 'solr', 'typesense',
            'faiss', 'hnswlib', 'hnsw', 'scann', 'annoy',
            'nmslib', 'voyager', 'usearch', 'diskann',
        ],
        'cat3_retrieval': [
            'bm25', 'tf-idf', 'tfidf', 'keyword search', 'keyword matching',
            'splade', 'sparse retrieval', 'inverted index', 'lexical search',
            'okapi bm25', 'bm25s', 'rank-bm25', 'pyserini',
            'dense retrieval', 'semantic search', 'vector search',
            'knn', 'k-nn', 'k nearest neighbor', 'nearest neighbor search',
            'ann', 'approximate nearest neighbor',
            'hierarchical navigable small world',
            'dense passage retrieval', 'dpr', 'rag retrieval',
            'bi-encoder retrieval', 'dual encoder',
            'hybrid search', 'sparse dense', 'rrf', 'reciprocal rank fusion',
            'multi-vector retrieval', 'colbert', 'late interaction', 'maxsim', 'plaid',
            'cross-encoder', 'cross encoder', 'reranker', 're-ranker',
            'cohere rerank', 'baai reranker', 'ms-marco reranker',
            'monot5', 'rankgpt', 'llm reranking', 'ltr',
            'learning to rank', 'xgboost ltr', 'lightgbm ltr',
            'lambdamart', 'ranklib', 'pointwise', 'pairwise', 'listwise',
            'retrieve then rerank', 'two-stage retrieval', 'neural reranking',
        ],
        'cat4_eval': [
            'ndcg', 'normalized discounted cumulative gain', 'ndcg@k',
            'mrr', 'mean reciprocal rank', 'mrr@k',
            'map', 'mean average precision',
            'precision@k', 'recall@k', 'f1@k', 'hit rate', 'hit@k',
            'expected reciprocal rank', 'err',
            'rank correlation', 'kendall tau', 'spearman',
            'a/b testing', 'ab testing', 'online evaluation',
            'offline to online correlation', 'ctr', 'click through rate',
            'interleaving', 'multileave', 'counterfactual evaluation',
            'implicit feedback', 'click modeling', 'position bias',
            'propensity scoring', 'ips', 'inverse propensity',
            'beir', 'beir benchmark', 'mteb', 'massive text embedding benchmark',
            'golden dataset', 'evaluation dataset', 'relevance judgment',
            'qrel', 'trec', 'trec eval', 'human evaluation',
            'human in the loop', 'retrieval regression', 'quality regression',
            'offline benchmark', 'online benchmark', 'evaluation pipeline',
            'ragas', 'truera', 'deepeval', 'arize', 'whylogs',
        ],
    }

    DESC_TIERS = {
        1: {'multiplier': 1.00, 'keywords': [
            'embedding', 'embeddings', 'sbert', 'sentence-transformer', 'bge', 'e5',
            'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
            'elasticsearch', 'vector', 'dense retrieval', 'hybrid search',
            'semantic search', 'ndcg', 'mrr', 'map', 'ranking', 're-rank', 'rerank',
            'retrieval', 'bm25', 'index refresh', 'embedding drift',
            'recommendation system', 'recommender', 'collaborative filtering',
            'matrix factorization', 'learning to rank', 'ltr', 'xgboost ltr',
            'cross-encoder', 'cross encoder', 'vector search', 'vector db',
            'vector database', 'colbert', 'splade', 'dpr', 'dense passage',
            'information retrieval', 'search relevance', 'reranking pipeline',
        ]},
        2: {'multiplier': 0.80, 'keywords': [
            'lora', 'qlora', 'peft', 'fine-tun', 'fine tuning', 'finetuning',
            'nlp', 'text classification', 'ner', 'named entity', 'summarization',
            'inference optimization', 'feature store', 'mlflow', 'llm',
            'large language model', 'transformer', 'bert', 'gpt', 'llama',
            'speech recognition', 'conversational ai', 'dialogue', 'chatbot',
            'open source contribution', 'hugging face', 'huggingface',
            'distributed training', 'model serving', 'triton', 'torchserve',
        ]},
        3: {'multiplier': 0.30, 'keywords': [
            'spark', 'airflow', 'kafka', 'dbt', 'snowflake', 'pipeline',
            'data engineering', 'data pipeline', 'etl', 'elt',
            'backend', 'api', 'microservice', 'rest', 'graphql',
            'computer vision', 'image classification', 'object detection',
            'speech', 'robotics', 'mlops', 'devops', 'cloud', 'kubernetes',
            'docker', 'ci/cd', 'analytics', 'business intelligence', 'tableau',
            'power bi', 'sql', 'postgres', 'mysql', 'data warehouse',
        ]},
    }

    TITLE_TIERS = {
        1: {'multiplier': 1.00, 'keywords': [
            'ml engineer', 'machine learning engineer', 'ai engineer',
            'nlp engineer', 'search engineer', 'ranking engineer',
            'retrieval engineer', 'recommendation engineer',
            'research engineer', 'applied scientist', 'machine learning scientist',
            'ai researcher', 'applied ml', 'applied machine learning',
            'senior ml', 'staff ml', 'principal ml', 'lead ml',
            'senior ai', 'staff ai', 'principal ai',
            'senior nlp', 'nlp scientist', 'search scientist',
            'ir engineer', 'information retrieval', 'relevance engineer',
            'recsys engineer', 'ranking scientist',
        ]},
        2: {'multiplier': 0.80, 'keywords': [
            'data scientist', 'computer vision engineer', 'cv engineer',
            'speech engineer', 'conversational ai', 'mlops engineer',
            'ml platform', 'ai product engineer', 'llm engineer',
            'nlp researcher', 'ir researcher', 'ml infrastructure',
            'deep learning engineer', 'deep learning researcher',
        ]},
        3: {'multiplier': 0.50, 'keywords': [
            'data engineer', 'analytics engineer', 'backend engineer',
            'full stack', 'fullstack', 'software engineer', 'platform engineer',
            'cloud engineer', 'devops engineer', 'data analyst', 'bi engineer',
            'sde', 'software developer', 'python developer',
        ]},
    }

    def normalize_text(text):
        return re.sub(r'[-_]', ' ', text.lower().strip())

    def chunk_text(text, chunk_size=CHUNK_SIZE):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        chunks = []
        for i in range(0, len(sentences), chunk_size):
            chunk = ' '.join(sentences[i:i + chunk_size])
            if chunk:
                chunks.append(chunk)
        return chunks if chunks else [text]

    def tokenize(text):
        return re.findall(r'\b\w+\b', normalize_text(text))

    def detect_desc_tier(description):
        norm = normalize_text(description)
        for tier in [1, 2, 3]:
            for kw in DESC_TIERS[tier]['keywords']:
                if kw in norm:
                    return tier
        return 4

    def detect_title_tier(title):
        norm = normalize_text(title)
        for tier in [1, 2, 3]:
            for kw in TITLE_TIERS[tier]['keywords']:
                if kw in norm:
                    return tier
        return 4

    def get_desc_tier_mult(tier):
        return {1: 1.00, 2: 0.80, 3: 0.30, 4: 0.00}.get(tier, 0.0)

    def get_title_tier_mult(tier):
        return {1: 1.00, 2: 0.80, 3: 0.50, 4: 0.30}.get(tier, 0.30)

    def get_best_description(career_history):
        if not career_history:
            return '', 0.0
        best_desc, best_tier, best_len = '', 4, 0
        for role in career_history:
            desc = role.get('description', '') or ''
            if not desc:
                continue
            tier = detect_desc_tier(desc)
            desc_len = len(desc)
            if tier < best_tier or (tier == best_tier and desc_len > best_len):
                best_tier, best_desc, best_len = tier, desc, desc_len
        return best_desc, get_desc_tier_mult(best_tier)

    def get_title_alignment_score(career_history):
        last_3 = career_history[:3]
        best_tier = 4
        for role in last_3:
            title = role.get('title', '') or ''
            tier = detect_title_tier(title)
            if tier < best_tier:
                best_tier = tier
        return get_title_tier_mult(best_tier)

    def get_experience_score(yoe):
        if yoe >= 12: return 0.65
        elif yoe >= 9: return 0.85
        elif yoe >= 5: return 1.00
        elif yoe >= 4: return 0.80
        elif yoe >= 3: return 0.55
        else:          return 0.20

    def get_skill_match_score(skills):
        skill_text = ' '.join(normalize_text(s.get('name', '')) for s in skills)
        hits = 0
        for cat, terms in SKILL_TAXONOMY.items():
            for term in terms:
                if normalize_text(term) in skill_text:
                    hits += 1
                    break
        return min(hits / 4.0, 1.0)

    def get_python_score(skills, career_history):
        if any('python' in normalize_text(s.get('name', '')) for s in skills):
            return 1.0
        for role in career_history:
            if 'python' in normalize_text(role.get('description', '') or ''):
                return 0.6
        return 0.0

    def get_behavioral_multiplier_t1(signals):
        otw = signals.get('open_to_work_flag', False)
        otw_mult = 1.05 if otw else 0.85

        rr = signals.get('recruiter_response_rate', 0.0) or 0.0
        if rr >= 0.80:   response_mult = 1.08
        elif rr >= 0.60: response_mult = 1.03
        elif rr >= 0.40: response_mult = 1.00
        elif rr >= 0.20: response_mult = 0.80
        else:            response_mult = 0.50

        notice = signals.get('notice_period_days', 60) or 60
        if notice <= 15:   notice_mult = 1.08
        elif notice <= 30: notice_mult = 1.05
        elif notice <= 45: notice_mult = 1.00
        elif notice <= 60: notice_mult = 0.95
        elif notice <= 90: notice_mult = 0.92
        else:              notice_mult = 0.88

        return min(otw_mult * response_mult * notice_mult, 1.0)

    # ── Load chunk embeddings ─────────────────────────────────────────────────
    CHUNK_EMB_PATH = f'{PRECOMPUTED}/chunk_embeddings.npy'

    if not os.path.exists(CHUNK_EMB_PATH):
        raise FileNotFoundError(
            f'\n[ERROR] chunk_embeddings.npy not found at {CHUNK_EMB_PATH}\n'
            f'Run Cell 7C in precompute.ipynb first.'
        )

    print('Loading chunk embeddings...')
    chunk_emb_data = np.load(CHUNK_EMB_PATH, allow_pickle=True).item()
    print(f'  Loaded: {len(chunk_emb_data)} candidates')

    passed_cids = set(passed_candidates)
    print(f'Candidates entering Phase 2: {len(passed_cids)}')

    # ── Build BM25 corpus ─────────────────────────────────────────────────────
    print('\nBuilding BM25 corpus...')
    cids_list, tier_mults_list, bm25_corpus, chunk_embeddings_list = [], [], [], []

    for c in candidates:
        cid = c['candidate_id']
        if cid not in passed_cids:
            continue
        career = c.get('career_history', [])
        best_desc, tier_mult = get_best_description(career)
        cids_list.append(cid)
        tier_mults_list.append(tier_mult)
        bm25_corpus.append(tokenize(best_desc) if best_desc else [''])
        emb = chunk_emb_data.get(cid)
        if emb is None or len(emb) == 0:
            emb = np.zeros((1, jd_vector.shape[0]), dtype='float32')
        chunk_embeddings_list.append(emb)

    print(f'  Corpus size: {len(cids_list)}')
    print('  Building BM25 index...')
    bm25 = BM25Okapi(bm25_corpus)

    # ── Compute career description scores ────────────────────────────────────
    print('\nComputing T1 scores...')
    n = len(cids_list)
    jd_vec = jd_vector / (np.linalg.norm(jd_vector) + 1e-9)

    jd_tokens = tokenize(
        'embeddings retrieval vector search ranking evaluation '
        'semantic search hybrid search dense retrieval bm25 '
        'sentence transformers faiss ndcg mrr production deployment'
    )
    bm25_raw = np.array(bm25.get_scores(jd_tokens))

    sbert_raw = np.zeros(n)
    for i, chunk_embs in enumerate(chunk_embeddings_list):
        if chunk_embs is None or len(chunk_embs) == 0:
            sbert_raw[i] = 0.0
        else:
            sbert_raw[i] = float(np.max(chunk_embs @ jd_vec))

    def minmax(arr):
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    bm25_norm  = minmax(bm25_raw)
    sbert_norm = minmax(sbert_raw)
    bm25_ranks  = np.argsort(np.argsort(-bm25_norm)) + 1
    sbert_ranks = np.argsort(np.argsort(-sbert_norm)) + 1
    rrf_scores  = (1.0 / (RRF_K + bm25_ranks)) + (1.0 / (RRF_K + sbert_ranks))
    rrf_norm    = minmax(rrf_scores)
    tier_mults_arr         = np.array(tier_mults_list)
    career_desc_scores_arr = minmax(rrf_norm * tier_mults_arr)
    career_desc_scores     = {cid: float(career_desc_scores_arr[i]) for i, cid in enumerate(cids_list)}

    # ── Per-candidate T1 feature scoring ─────────────────────────────────────
    cand_map  = {c['candidate_id']: c for c in candidates}
    t1_scores = {}

    for cid in passed_cids:
        c = cand_map.get(cid)
        if not c:
            continue
        profile = c.get('profile', {})
        career  = c.get('career_history', [])
        skills  = c.get('skills', [])
        signals = c.get('redrob_signals', {})

        f1 = career_desc_scores.get(cid, 0.0)
        f2 = get_title_alignment_score(career)
        f3 = get_experience_score(profile.get('years_of_experience', 0) or 0)
        f4 = get_skill_match_score(skills)
        f5 = get_python_score(skills, career)

        raw = min(
            WEIGHTS['w1'] * f1 + WEIGHTS['w2'] * f2 + WEIGHTS['w3'] * f3 +
            WEIGHTS['w4'] * f4 + WEIGHTS['w5'] * f5,
            1.0
        )
        b_mult    = get_behavioral_multiplier_t1(signals)
        s_penalty = soft_penalty_multipliers.get(cid, 1.0)
        t1_scores[cid] = min(raw * b_mult * s_penalty, 1.0)

    print(f'  T1 scores computed: {len(t1_scores)}')

    # ── Phase 2B — Top 5000 via min heap ────────────────────────────────────
    print('\nSelecting top 5000 via min heap...')
    heap = []
    for cid, score in t1_scores.items():
        if len(heap) < TOP_K:
            heapq.heappush(heap, (score, cid))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, cid))

    top_5000 = sorted(heap, key=lambda x: -x[0])
    print(f'  Top 5000 selected. Score range: {top_5000[-1][0]:.4f} – {top_5000[0][0]:.4f}')

    # ============================================================
    # PHASE 2C — HONEYPOT DETECTION
    # ============================================================

    @dataclass
    class HoneypotResult:
        cid: str
        is_honeypot: bool = False
        soft_penalty: float = 1.0
        reasons: list = field(default_factory=list)

    def compute_yoe_from_history(career_history):
        if not career_history:
            return 0.0, 0.0
        sum_months = sum(role.get('duration_months', 0) or 0 for role in career_history)
        sum_duration_years = sum_months / 12.0
        intervals = []
        for role in career_history:
            start = role.get('start_date')
            dur   = role.get('duration_months', 0) or 0
            if start and dur:
                try:
                    parts = start.split('-')
                    yr, mo = int(parts[0]), int(parts[1])
                    start_f = yr + mo / 12.0
                    intervals.append((start_f, start_f + dur / 12.0))
                except:
                    continue
        if not intervals:
            return sum_duration_years, sum_duration_years
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        for s, e in intervals[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))
        non_overlap_years = sum(e - s for s, e in merged)
        return sum_duration_years, non_overlap_years

    def check_expert_zero_months(skills):
        flagged = [
            s.get('name', 'unknown') for s in skills
            if (s.get('proficiency') or '').lower().strip() == 'expert'
            and (s.get('duration_months') == 0 or s.get('duration_months') is None)
        ]
        if flagged:
            return True, f"Expert/0mo: {', '.join(flagged)}"
        return False, ''

    def check_overlapping_tenures(career_history, claimed_yoe):
        sum_dur, _ = compute_yoe_from_history(career_history)
        excess = sum_dur - (claimed_yoe + 2.0)
        if excess > 0:
            return True, f"Tenure overlap: sum={sum_dur:.1f}y claimed={claimed_yoe:.1f}y excess={excess:.1f}y"
        return False, ''

    def check_inflated_yoe(career_history, claimed_yoe):
        _, non_overlap = compute_yoe_from_history(career_history)
        inflation = claimed_yoe - non_overlap
        if inflation > 3.0:
            return True, 1.0, f"YOE inflated {inflation:.1f}y (claimed={claimed_yoe:.1f}, computed={non_overlap:.1f})"
        elif inflation > 1.0:
            return False, 0.60, f"Mild YOE inflation {inflation:.1f}y → soft ×0.60"
        return False, 1.0, ''

    def check_candidate(candidate):
        cid     = candidate['candidate_id']
        result  = HoneypotResult(cid=cid)
        profile = candidate.get('profile', {})
        skills  = candidate.get('skills', [])
        career  = candidate.get('career_history', [])
        claimed_yoe = profile.get('years_of_experience', 0) or 0

        flagged, reason = check_expert_zero_months(skills)
        if flagged:
            result.is_honeypot = True
            result.reasons.append(f'[HARD-1] {reason}')

        flagged, reason = check_overlapping_tenures(career, claimed_yoe)
        if flagged:
            result.is_honeypot = True
            result.reasons.append(f'[HARD-2A] {reason}')

        is_hard, penalty, reason = check_inflated_yoe(career, claimed_yoe)
        if is_hard:
            result.is_honeypot = True
            result.reasons.append(f'[HARD-2B] {reason}')
        elif penalty < 1.0:
            result.soft_penalty = penalty
            result.reasons.append(f'[SOFT-2B] {reason}')

        return result

    print('\n[Phase 2C] Running honeypot detection...')
    clean_candidates, knockouts = [], {}

    for score, cid in top_5000:
        c = cand_map.get(cid)
        if not c:
            continue
        result = check_candidate(c)
        if result.is_honeypot:
            knockouts[cid] = result
        else:
            clean_candidates.append((score * result.soft_penalty, cid))

    clean_candidates.sort(key=lambda x: -x[0])
    print(f'  Input: {len(top_5000)} | Knockouts: {len(knockouts)} | Clean: {len(clean_candidates)}')

    # ============================================================
    # PHASE 3 — T2 SCORING (5000 → Top 200)
    # ============================================================

    print('\n' + '='*60)
    print('PHASE 3 — T2 Scoring (5000 → 200)')
    print('='*60)

    TOP_K_T2 = min(200, len(clean_candidates))   # adaptive for small sandbox samples

    T2_WEIGHTS = {
        'f1': 0.20,   # SBERT full profile
        'f2': 0.15,   # Production depth
        'f3': 0.23,   # Retrieval specialization
        'f4': 0.18,   # Skill match
        'f5': 0.08,   # Experience
        'f6': 0.16,   # Evaluation mindset
    }

    BEH_WEIGHTS_T2 = {
        's1': 0.10, 's2': 0.25, 's3': 0.15,
        's4': 0.20, 's5': 0.20, 's6': 0.10,
    }
    BEH_FLOOR, BEH_CAP = 0.45, 1.20

    PRODUCTION_TERMS = {
        'deployed', 'production', 'serving', 'real users', 'scale',
        'latency', 'throughput', 'a/b', 'online', 'inference',
        'pipeline', 'index refresh', 'embedding drift', 'retrieval quality',
        'shipped', 'launched', 'end-to-end', 'built'
    }
    RESEARCH_TERMS = {
        'arxiv', 'research lab', 'academic', 'research internship',
        'published paper', 'under review', 'research scientist'
    }
    RETRIEVAL_DESC_TERMS = {
        'embeddings', 'retrieval', 'ranking', 'semantic search',
        'hybrid search', 'dense retrieval', 'vector search', 'bm25',
        'faiss', 'reranking', 'recommendation system', 'search relevance',
        'index', 'query expansion'
    }
    RETRIEVAL_DEPTH_TERMS = {
        'embedding drift', 'index refresh', 'retrieval quality', 'ndcg',
        'mrr', 'recall@k', 'precision@k', 'relevance feedback',
        'hard negatives', 'contrastive learning', 'bi-encoder', 'cross-encoder'
    }
    MUST_HAVE_SKILLS = {
        'sentence-transformers', 'bge', 'e5', 'sbert', 'openai embeddings',
        'faiss', 'pinecone', 'weaviate', 'qdrant', 'milvus', 'opensearch',
        'elasticsearch', 'vector database', 'hybrid search', 'python',
        'evaluation framework', 'ndcg', 'mrr', 'map', 'a/b testing',
        'dense retrieval', 'embeddings'
    }
    NICE_TO_HAVE_SKILLS = {
        'lora', 'qlora', 'peft', 'fine-tuning', 'ltr', 'xgboost', 'lightgbm',
        'distributed systems', 'large-scale inference', 'open-source',
        'hr-tech', 'recruiting tech', 'marketplace'
    }
    EVAL_OFFLINE_TERMS = {
        'ndcg', 'mrr', 'map', 'precision@k', 'recall@k', 'offline evaluation',
        'benchmark', 'eval framework', 'relevance judgment', 'ground truth',
        'human evaluation', 'annotation', 'labeling pipeline'
    }
    EVAL_ONLINE_TERMS = {
        'a/b test', 'online evaluation', 'click-through rate', 'ctr',
        'engagement metric', 'conversion', 'feedback loop',
        'recruiter feedback', 'user study', 'online experiment'
    }
    RELEVANT_ML_TITLES = {
        'machine learning', 'ml engineer', 'ai engineer', 'nlp engineer',
        'research scientist', 'data scientist', 'nlp', 'retrieval',
        'ranking', 'search', 'recommendations', 'applied scientist'
    }

    def to_lower(text): return (text or '').lower()
    def any_term_in(text, terms): t = to_lower(text); return any(term in t for term in terms)
    def get_all_desc(career):
        return ' '.join(role.get('description', '') or '' for role in career)
    def get_skills_text(skills):
        if not skills: return ''
        return ' '.join(s.get('name', '') if isinstance(s, dict) else str(s) for s in skills)

    def compute_f2(career):
        all_desc = get_all_desc(career)
        has_prod = any_term_in(all_desc, PRODUCTION_TERMS)
        has_res  = any_term_in(all_desc, RESEARCH_TERMS)
        if has_prod and has_res: return 1.20
        return (0.50 if has_prod else 0.0) + (0.50 if not has_res else 0.0)

    def compute_f3(career, skills):
        all_desc = get_all_desc(career)
        full     = all_desc + ' ' + get_skills_text(skills)
        return (0.60 if any_term_in(all_desc, RETRIEVAL_DESC_TERMS) else 0.0) + \
               (0.40 if any_term_in(full, RETRIEVAL_DEPTH_TERMS) else 0.0)

    def compute_f4(career, skills):
        full = to_lower(get_all_desc(career) + ' ' + get_skills_text(skills))
        must_hits = sum(1 for t in MUST_HAVE_SKILLS if t in full)
        nice_hits = sum(1 for t in NICE_TO_HAVE_SKILLS if t in full)
        if must_hits > 3:    base = 1.0
        elif must_hits == 3: base = 0.80
        elif must_hits == 2: base = 0.60 + nice_hits * 0.08
        elif must_hits == 1: base = 0.40 + nice_hits * 0.08
        else:                 base = nice_hits * 0.08
        return min(base, 1.0)

    def get_relevant_yoe(career):
        rel = 0.0
        for role in career:
            title = to_lower(role.get('title', '') or '')
            desc  = to_lower(role.get('description', '') or '')
            if any(t in title + ' ' + desc for t in RELEVANT_ML_TITLES):
                start = role.get('start_date') or role.get('start_year')
                end   = role.get('end_date') or role.get('end_year')
                try:
                    s = int(str(start)[:4]) if start else None
                    e = int(str(end)[:4]) if end else 2026
                    if s: rel += max(0, e - s)
                except: rel += 1.0
        return rel

    def compute_f5(profile, career):
        yoe = float(profile.get('years_of_experience') or 0)
        if 5 <= yoe <= 9:   sa = 0.50
        elif yoe > 9:       sa = 0.40
        elif 3 <= yoe < 5:  sa = 0.35
        else:               sa = 0.20
        rel = get_relevant_yoe(career)
        if rel > 4:         sb = 0.50
        elif rel == 4:      sb = 0.40
        elif rel == 3:      sb = 0.35
        elif rel == 2:      sb = 0.30
        elif 1 <= rel < 2:  sb = 0.20
        else:               sb = 0.05
        return sa + sb

    def compute_f6(career):
        all_desc = get_all_desc(career)
        return (0.50 if any_term_in(all_desc, EVAL_OFFLINE_TERMS) else 0.0) + \
               (0.50 if any_term_in(all_desc, EVAL_ONLINE_TERMS) else 0.0)

    def get_behavioral_multiplier_t2(signals):
        otw = signals.get('open_to_work_flag', False)
        s1  = 1.05 if otw else 0.85

        rr = float(signals.get('recruiter_response_rate') or 0.0)
        if rr >= 0.80:   s2 = 1.08
        elif rr >= 0.60: s2 = 1.03
        elif rr >= 0.40: s2 = 0.90
        elif rr >= 0.20: s2 = 0.70
        else:            s2 = 0.50

        notice = int(signals.get('notice_period_days') or 60)
        if notice <= 15:   s3 = 1.08
        elif notice <= 30: s3 = 1.05
        elif notice <= 45: s3 = 0.95
        elif notice <= 60: s3 = 0.87
        elif notice <= 90: s3 = 0.80
        else:              s3 = 0.73

        work_mode = to_lower(signals.get('preferred_work_mode') or '')
        if 'onsite' in work_mode or 'local' in work_mode or 'office' in work_mode: s4 = 1.10
        elif 'hybrid' in work_mode:  s4 = 1.05
        elif 'remote' in work_mode:  s4 = 0.65
        else:                        s4 = 1.00

        location      = to_lower(signals.get('current_location') or '')
        willing_reloc = signals.get('willing_to_relocate', None)
        if any(c in location for c in ['pune', 'noida']):  s5 = 1.05
        elif willing_reloc is True:  s5 = 1.05
        elif willing_reloc is False: s5 = 0.60
        else:                        s5 = 1.00

        job_status = to_lower(signals.get('job_seeking_status') or '')
        if 'actively' in job_status or 'active' in job_status: s6 = 1.10
        elif 'open' in job_status:        s6 = 1.00
        elif 'not looking' in job_status: s6 = 0.85
        else:                             s6 = 1.00

        raw = (BEH_WEIGHTS_T2['s1']*s1 + BEH_WEIGHTS_T2['s2']*s2 + BEH_WEIGHTS_T2['s3']*s3 +
               BEH_WEIGHTS_T2['s4']*s4 + BEH_WEIGHTS_T2['s5']*s5 + BEH_WEIGHTS_T2['s6']*s6)
        return float(np.clip(raw, BEH_FLOOR, BEH_CAP))

    # ── Load precomputed embeddings for f1 ───────────────────────────────────
    print('Loading precomputed embeddings for T2 f1...')
    rich_path  = f'{PRECOMPUTED}/embeddings_rich.npy'
    cids_path  = f'{PRECOMPUTED}/embed_cids_rich.npy'
    if not os.path.exists(rich_path):
        rich_path = f'{PRECOMPUTED}/embeddings.npy'
        cids_path = f'{PRECOMPUTED}/embed_cids.npy'
    all_embs     = np.load(rich_path).astype('float32')
    all_emb_cids = np.load(cids_path, allow_pickle=True)
    embed_index  = {str(cid): i for i, cid in enumerate(all_emb_cids)}
    jd_vec_norm  = jd_vector / (np.linalg.norm(jd_vector) + 1e-9)

    # ── Score all 5000 candidates with T2 ───────────────────────────────────
    print(f'Computing T2 scores for {len(clean_candidates)} candidates...')
    t2_scores = {}

    for idx, (t1_score, cid) in enumerate(clean_candidates):
        if idx % 500 == 0:
            print(f'  {idx}/{len(clean_candidates)}...')

        c = cand_map.get(cid)
        if not c:
            continue

        profile = c.get('profile', {})
        career  = c.get('career_history', [])
        skills  = c.get('skills', [])
        signals = c.get('redrob_signals', {})

        ei = embed_index.get(str(cid))
        if ei is not None:
            emb = all_embs[ei]
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            f1  = float(np.dot(emb, jd_vec_norm))
        else:
            f1 = 0.0

        f2 = compute_f2(career)
        f3 = compute_f3(career, skills)
        f4 = compute_f4(career, skills)
        f5 = compute_f5(profile, career)
        f6 = compute_f6(career)
        beh = get_behavioral_multiplier_t2(signals)

        t2 = (T2_WEIGHTS['f1']*f1 + T2_WEIGHTS['f2']*f2 + T2_WEIGHTS['f3']*f3 +
              T2_WEIGHTS['f4']*f4 + T2_WEIGHTS['f5']*f5 + T2_WEIGHTS['f6']*f6) * beh

        t2_scores[cid] = round(t2, 6)

    print(f'  T2 scores computed: {len(t2_scores)}')

    # ── Top 200 via min heap ─────────────────────────────────────────────────
    print(f'\nSelecting top {TOP_K_T2} via min heap...')
    heap = []
    for cid, score in t2_scores.items():
        if len(heap) < TOP_K_T2:
            heapq.heappush(heap, (score, cid))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, cid))

    top_200 = sorted(heap, key=lambda x: -x[0])
    print(f'  Top {TOP_K_T2} selected. Score range: {top_200[-1][0]:.4f} – {top_200[0][0]:.4f}')
    t2_lookup = {cid: {'t2_score': score, 't2_rank': rank+1} for rank, (score, cid) in enumerate(top_200)}

    # ============================================================
    # PHASE 4 — T3 CROSS-ENCODER RE-RANKING (200 → 100)
    # ============================================================

    print('\n' + '='*60)
    print('PHASE 4 — T3 Cross-Encoder Re-ranking (200 → 100)')
    print('='*60)

    TOP_K_T3 = min(100, len(top_200))

    # ordered_ids — previously built during the (now-removed) pairs loop
    ordered_ids = [cid for _, cid in top_200]

    # T2 normalization
    t2_values = np.array([t2_lookup[cid]['t2_score'] for cid in ordered_ids], dtype=np.float32)
    t2_min, t2_max = t2_values.min(), t2_values.max()
    t2_norm = (t2_values - t2_min) / (t2_max - t2_min + 1e-9)

    # Load normalized cross scores from cache — fallback 0.7 if missing
    cross_norm = np.array([
        cross_score_cache.get(cid, {}).get('normalized', 0.7)
        for cid in ordered_ids
    ], dtype=np.float32)

    # T3 blend
    T3_W_CROSS, T3_W_T2 = 0.40, 0.60
    t3_scores = T3_W_CROSS * cross_norm + T3_W_T2 * t2_norm
    print(f'  T3 scores — min: {t3_scores.min():.4f}  max: {t3_scores.max():.4f}')

    print(f'[T3] Selecting top {TOP_K_T3} via min heap...')
    heap = []
    for i, cid in enumerate(ordered_ids):
        score = float(t3_scores[i])
        if len(heap) < TOP_K_T3:
            heapq.heappush(heap, (score, i, cid))
        elif score > heap[0][0]:
            heapq.heapreplace(heap, (score, i, cid))

    top_100 = sorted(heap, key=lambda x: -x[0])
    print(f'  Top {TOP_K_T3} selected. T3 score range: {top_100[-1][0]:.4f} – {top_100[0][0]:.4f}')
    print('\n[T3] Phase 4 complete.')

    print('\n[T3] Top 10 candidates:')
    print(f'  {"Rank":<5} {"candidate_id":<18} {"t3_score":<10} {"cross_score":<13} {"t2_score":<10} {"t2_rank"}')
    print('  ' + '-' * 68)
    for rank, (t3, i, cid) in enumerate(top_100[:10], start=1):
        print(f'  {rank:<5} {cid:<18} {t3:<10.4f} {cross_norm[i]:<13.4f} {t2_values[i]:<10.4f} {t2_lookup[cid]["t2_rank"]}')


    # ============================================================
    # PHASE 4 — REASONING GENERATOR (fact-grounded, local, templated)
    # No LLM calls. Pulls only facts that exist in the candidate's
    # actual profile / feature_cache / redrob_signals. Tone and
    # concern surfacing vary by rank tier and by what's actually true.
    # ============================================================

    import random

    random.seed(42)  # deterministic phrasing variation across runs

    # ── Tier definitions (by rank, 1-indexed) ──────────────────────────────────
    def get_tier(rank):
        if rank <= 10:
            return 'top'
        elif rank <= 40:
            return 'mid'
        else:
            return 'lower'

    # ── Sentence-opener banks (rotated by hashing candidate_id, not random per-row,
    #    so re-runs are reproducible) ──────────────────────────────────────────
    OPENERS_TOP = [
        "{title} with {yoe} years of experience",
        "{yoe}-year {title}",
        "Strong fit: {title} with {yoe} years",
        "{title}, {yoe} years in",
    ]
    OPENERS_MID = [
        "{title} with {yoe} years of experience",
        "{yoe}-year {title}",
        "{title} ({yoe} years)",
    ]
    OPENERS_LOWER = [
        "{title} with {yoe} years",
        "{yoe}-year {title}, adjacent fit",
        "{title}, {yoe} years experience",
    ]

    SKILL_CONNECTORS = [
        "with hands-on work in {skills}",
        "having worked with {skills}",
        "with direct experience in {skills}",
        "covering {skills}",
    ]

    PRODUCTION_PHRASES = [
        "shipped to production",
        "deployed at scale",
        "running for real users",
        "in a live production system",
    ]

    CONCERN_PHRASES = {
        'notice': "notice period is {days} days, which may slow down hiring",
        'response': "recruiter response rate is low ({rate}), engagement is uncertain",
        'inactive': "has been inactive on the platform for {days} days",
        'yoe_gap': "experience is on the lower end ({yoe} years) for this role's seniority",
        'no_python': "no explicit Python signal found in skills or descriptions",
        'consulting': "background is primarily consulting/services rather than product",
        'soft_penalty': "profile shows some inconsistency that lowered confidence",
    }

    JD_CONNECTORS = [
        "matches the JD's focus on embeddings-based retrieval and ranking",
        "aligns with the JD's emphasis on production search/recommendation systems",
        "fits the JD's requirement for hands-on retrieval and evaluation work",
        "covers the core JD skill areas (retrieval, ranking, vector search)",
    ]

    WEAK_JD_CONNECTORS = [
        "covers some but not all of the JD's core retrieval/ranking requirements",
        "is an adjacent fit — overlapping skills but not a direct match to the JD's core ask",
        "brings related ML experience without deep retrieval/ranking specialization",
    ]


    def pick(seq, seed_key):
        """Deterministic pseudo-random pick based on candidate_id so phrasing
        varies across the 100 rows without being random on every script run."""
        idx = sum(ord(c) for c in seed_key) % len(seq)
        return seq[idx]


    def format_skills(skill_names, limit=3):
        if not skill_names:
            return ''
        cleaned = [s for s in skill_names if s and len(s) > 1][:limit]
        if not cleaned:
            return ''
        if len(cleaned) == 1:
            return cleaned[0]
        return ', '.join(cleaned[:-1]) + ' and ' + cleaned[-1]


    def get_best_title(career_history):
        if not career_history:
            return 'a professional'
        sorted_roles = sorted(
            career_history,
            key=lambda r: r.get('start_date', '1970-01-01'),
            reverse=True
        )
        return sorted_roles[0].get('title', 'a professional') or 'a professional'


    def get_top_skill_names(skills, taxonomy_terms, limit=3):
        """Pull only skill names that actually appear in the candidate's skills
        list AND match known retrieval/ranking/ML taxonomy terms — avoids
        hallucinating relevance for generic skills."""
        if not skills:
            return []
        matched = []
        for s in skills:
            name = (s.get('name', '') or '').strip()
            if not name:
                continue
            name_lower = name.lower()
            if any(term in name_lower for term in taxonomy_terms):
                matched.append(name)
        return matched[:limit]


    # Flattened taxonomy for skill-name matching in reasoning text
    RELEVANT_TERMS_FLAT = set()
    for _cat, _terms in SKILL_TAXONOMY.items():
        RELEVANT_TERMS_FLAT.update(t.lower() for t in _terms)
    RELEVANT_TERMS_FLAT.update(['python', 'pytorch', 'tensorflow'])


    def build_concerns(cid, candidate, signals, profile, yoe, skills_text):
        """Collect honest, fact-grounded concerns. Only includes a concern if
        the underlying fact is actually present/true in the candidate's data —
        never invented."""
        concerns = []

        notice = signals.get('notice_period_days')
        if notice is not None and notice > 75:
            concerns.append(CONCERN_PHRASES['notice'].format(days=notice))

        rr = signals.get('recruiter_response_rate')
        if rr is not None and rr < 0.3:
            concerns.append(CONCERN_PHRASES['response'].format(rate=f'{rr:.2f}'))

        fc = feature_cache.get(cid, {})
        inactive_days = fc.get('inactive_days')
        if inactive_days is not None and inactive_days > 60:
            concerns.append(CONCERN_PHRASES['inactive'].format(days=inactive_days))

        if yoe and yoe < 3:
            concerns.append(CONCERN_PHRASES['yoe_gap'].format(yoe=yoe))

        if 'python' not in skills_text.lower():
            concerns.append(CONCERN_PHRASES['no_python'])

        if fc.get('is_consulting_only'):
            concerns.append(CONCERN_PHRASES['consulting'])

        s_penalty = soft_penalty_multipliers.get(cid, 1.0)
        if s_penalty < 0.9:
            concerns.append(CONCERN_PHRASES['soft_penalty'])

        return concerns


    def generate_reasoning(rank, cid, candidate, t_score):
        tier    = get_tier(rank)
        profile = candidate.get('profile', {})
        career  = candidate.get('career_history', [])
        skills  = candidate.get('skills', [])
        signals = candidate.get('redrob_signals', {})

        yoe   = profile.get('years_of_experience', 0) or 0
        title = get_best_title(career)

        matched_skills = get_top_skill_names(skills, RELEVANT_TERMS_FLAT, limit=3)
        skills_text    = format_skills(matched_skills)
        full_skills_text = ' '.join((s.get('name', '') or '') for s in skills)

        # ── Opener ──────────────────────────────────────────────────────────
        opener_bank = {'top': OPENERS_TOP, 'mid': OPENERS_MID, 'lower': OPENERS_LOWER}[tier]
        opener_tpl  = pick(opener_bank, cid)
        opener = opener_tpl.format(title=title, yoe=int(yoe) if yoe else 'unspecified')

        # ── Skill clause ────────────────────────────────────────────────────
        skill_clause = ''
        if skills_text:
            connector = pick(SKILL_CONNECTORS, cid + '_sk')
            skill_clause = ' ' + connector.format(skills=skills_text)

        # ── Production signal (only if actually present in description text) ─
        all_desc = ' '.join((r.get('description', '') or '') for r in career).lower()
        production_clause = ''
        if any(p.split()[0] in all_desc for p in ['deployed', 'production', 'shipped', 'scale', 'serving']):
            production_clause = '; ' + pick(PRODUCTION_PHRASES, cid + '_pr')

        # ── JD connection ───────────────────────────────────────────────────
        if len(matched_skills) >= 2:
            jd_clause = pick(JD_CONNECTORS, cid + '_jd')
        else:
            jd_clause = pick(WEAK_JD_CONNECTORS, cid + '_jd')

        # ── Concerns (fact-grounded only) ──────────────────────────────────
        concerns = build_concerns(cid, candidate, signals, profile, yoe, full_skills_text)
        concern_clause = ''
        if concerns:
            # Surface at most 1-2 to keep reasoning to 1-2 sentences as spec'd
            chosen = concerns[:2]
            concern_clause = '; however, ' + ' and '.join(chosen) + '.'
        else:
            concern_clause = '.'

        # ── Assemble: 1-2 sentences ─────────────────────────────────────────
        sentence_1 = f"{opener}{skill_clause}{production_clause}, {jd_clause}{concern_clause}"
        sentence_1 = sentence_1[0].upper() + sentence_1[1:]

        return sentence_1


    # ============================================================
    # Run Phase 4 over the final ranked list
    # (expects `top_100` from Phase 4 cross-encoder step:
    #  list of (t3_score, idx_in_ordered_ids, cid) tuples, already sorted desc)
    # ============================================================

    print('\n' + '='*60)
    print('PHASE 4 — Reasoning Generator')
    print('='*60)

    reasoning_results = []
    for rank, (t3, i, cid) in enumerate(top_100, start=1):
        c = cand_map.get(cid)
        if not c:
            reasoning_results.append((rank, cid, t3, ''))
            continue
        reasoning_text = generate_reasoning(rank, cid, c, t3)
        reasoning_results.append((rank, cid, t3, reasoning_text))

    print(f'  Generated {len(reasoning_results)} reasoning strings.')
    print('\n  Sample (rank 1, 25, 50, 75, 100):')
    for r in [1, 25, 50, 75, 100]:
        if r <= len(reasoning_results):
            rank, cid, t3, text = reasoning_results[r-1]
            print(f'\n  [{rank}] {cid} (score={t3:.4f})')
            print(f'    {text}')



    # ============================================================
    # PHASE 5 — SELF-VALIDATION + FINAL CSV EXPORT
    # Validates against every rule in Section 3 of submission_spec.docx
    # before writing the file the validator will actually check.
    # ============================================================

    import csv

    print('\n' + '='*60)
    print('PHASE 5 — Self-Validation + Final Export')
    print('='*60)

    TEAM_SUBMISSION_ID = team_id
    FINAL_CSV = output_csv_path

    # ── Build final rows from Phase 4 results ──────────────────────────────────
    # reasoning_results: list of (rank, cid, t3_score, reasoning_text)
    final_rows = []
    for rank, cid, t3, reasoning in reasoning_results:
        final_rows.append({
            'candidate_id': cid,
            'rank': rank,
            'score': round(float(t3), 6),
            'reasoning': reasoning,
        })

    # ── Validation checks ──────────────────────────────────────────────────────
    errors, warnings = [], []

    # 1. Exactly 100 rows
    if len(final_rows) != final_top_n:
        errors.append(f'Expected exactly {final_top_n} rows, got {len(final_rows)}')

    # 2. Ranks 1..100 each exactly once
    ranks = sorted(r['rank'] for r in final_rows)
    if ranks != list(range(1, final_top_n + 1)):
        missing = set(range(1, final_top_n + 1)) - set(ranks)
        dupes   = [r for r in ranks if ranks.count(r) > 1]
        errors.append(f'Ranks not exactly 1-100 once each. Missing: {missing}, Duplicated: {set(dupes)}')

    # 3. No duplicate candidate_ids
    cids_seen = [r['candidate_id'] for r in final_rows]
    if len(cids_seen) != len(set(cids_seen)):
        dupes = {c for c in cids_seen if cids_seen.count(c) > 1}
        errors.append(f'Duplicate candidate_ids found: {dupes}')

    # 4. Every candidate_id exists in candidates.jsonl
    unknown = [c for c in cids_seen if c not in cand_map]
    if unknown:
        errors.append(f'{len(unknown)} candidate_ids not found in candidates.jsonl: {unknown[:5]}...')

    # 5. Score is non-increasing as rank increases
    rows_by_rank = sorted(final_rows, key=lambda r: r['rank'])
    for i in range(1, len(rows_by_rank)):
        prev_score = rows_by_rank[i - 1]['score']
        curr_score = rows_by_rank[i]['score']
        if curr_score > prev_score:
            errors.append(
                f'Score increases at rank {rows_by_rank[i]["rank"]}: '
                f'{prev_score} (rank {rows_by_rank[i-1]["rank"]}) -> {curr_score}'
            )

    # 6. Scores not all identical
    unique_scores = {r['score'] for r in final_rows}
    if len(unique_scores) == 1:
        errors.append('All scores are identical — model is not differentiating candidates.')

    # 7. Tie-break check: same score must still have unique ranks (true by construction,
    #    but verify no two rows share both score AND rank — sanity check only)
    score_rank_pairs = [(r['score'], r['rank']) for r in final_rows]
    if len(score_rank_pairs) != len(set(score_rank_pairs)):
        errors.append('Duplicate (score, rank) pairs found — tie-break not applied correctly.')

    # 8. Reasoning column checks (warnings, not hard errors — column is optional
    #    per spec, but empty/identical reasoning is explicitly penalized)
    empty_reasoning = [r['candidate_id'] for r in final_rows if not r['reasoning'].strip()]
    if empty_reasoning:
        warnings.append(f'{len(empty_reasoning)} candidates have empty reasoning: {empty_reasoning[:5]}...')

    reasoning_texts = [r['reasoning'] for r in final_rows]
    if len(set(reasoning_texts)) < len(reasoning_texts) * 0.9:
        warnings.append('Reasoning strings show low variation (<90% unique) — risk of "templated" flag at Stage 4.')

    # 9. Honeypot rate check in top 100 (must be <=10% per spec section 7)
    honeypot_cids_in_top100 = [c for c in cids_seen if c in knockouts]
    honeypot_rate = len(honeypot_cids_in_top100) / len(cids_seen) if cids_seen else 0
    if honeypot_rate > 0.10:
        errors.append(
            f'Honeypot rate in top 100 is {honeypot_rate:.1%} (limit 10%). '
            f'Honeypots leaked through: {honeypot_cids_in_top100}'
        )
    else:
        print(f'  Honeypot check passed: {honeypot_rate:.1%} (limit 10%)')

    # ── Report ───────────────────────────────────────────────────────────────
    print(f'\n  Rows: {len(final_rows)}')
    print(f'  Unique ranks: {len(set(ranks))}')
    print(f'  Unique candidate_ids: {len(set(cids_seen))}')
    print(f'  Unique scores: {len(unique_scores)}')
    print(f'  Score range: {rows_by_rank[-1]["score"]:.4f} (rank 100) -> {rows_by_rank[0]["score"]:.4f} (rank 1)')

    if errors:
        print(f'\n  ✗ {len(errors)} VALIDATION ERROR(S) — DO NOT SUBMIT:')
        for e in errors:
            print(f'    - {e}')
    else:
        print('\n  ✓ All hard validation checks passed.')

    if warnings:
        print(f'\n  ⚠ {len(warnings)} warning(s):')
        for w in warnings:
            print(f'    - {w}')

    # ── Write CSV only if no hard errors ────────────────────────────────────
    if errors:
        print('\n  CSV NOT WRITTEN — fix errors above first.')
    else:
        print(f'\n  Writing final submission CSV to: {FINAL_CSV}')
        with open(FINAL_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['candidate_id', 'rank', 'score', 'reasoning'])
            for r in rows_by_rank:
                writer.writerow([r['candidate_id'], r['rank'], r['score'], r['reasoning']])
        print('  ✓ Saved.')

        # ── Post-write sanity re-read (catches encoding/quoting issues) ──────
        print('\n  Re-reading file to confirm it parses cleanly...')
        with open(FINAL_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            reread_rows = list(reader)
        assert len(reread_rows) == 100, f'Re-read row count mismatch: {len(reread_rows)}'
        assert list(reread_rows[0].keys()) == ['candidate_id', 'rank', 'score', 'reasoning'], \
            f'Column order mismatch: {list(reread_rows[0].keys())}'
        print('  ✓ File re-reads cleanly with correct columns and row count.')
        print(f'\n  Submission ready: {FINAL_CSV}')
    return {
        'rows': rows_by_rank if not errors else [],
        'errors': errors,
        'warnings': warnings,
        'honeypot_rate': honeypot_rate,
        'csv_path': FINAL_CSV if not errors else None,
    }
