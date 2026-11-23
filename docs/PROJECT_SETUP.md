# Project Setup: Rebuilding the Instagram Recommendation Algorithm

## Overview
Rebuild a simplified, Instagram-style feed recommendation system from scratch as a learning ML project: data → candidate generation → ranking → re-ranking → evaluation.

Instagram's real algorithm is proprietary, so this is an independent re-implementation using synthetic or public data. All details below are placeholders and will be updated.

## Goals
- Build an end-to-end recommendation pipeline that runs locally
- Implement a two-stage architecture (candidate generation + ranking)
- Support training, evaluation, and inference with reproducible experiments
- Document design decisions in the repo

## Non-Goals
- Matching Instagram's production scale or exact behavior
- Using real user data or scraping Instagram

## System Design (proposed)
| Stage | Purpose | Initial approach |
|---|---|---|
| Data ingestion | Users, posts, interactions | Synthetic generator or public dataset |
| Feature engineering | User, post, user-post features | Pandas feature module |
| Candidate generation | Narrow millions to hundreds | Two-tower embeddings + ANN search |
| Ranking | Score candidates | LightGBM, then a neural ranker |
| Re-ranking | Diversity, freshness, dedup | Rules + MMR |
| Evaluation | Offline quality | Recall@K, NDCG@K, MAP, AUC |
| Serving | Return a feed | FastAPI endpoint |

Signals to model (placeholders):
- Engagement: likes, comments, shares, saves, watch time
- Relationship: follows, DMs, profile visits
- Content: topic, media type, caption, post age
- Author: popularity, activity level

## Proposed Repository Structure
```
Rebuilding-Instagram-algorithm/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
├── configs/            # YAML configs for data, model, training
├── data/
│   ├── raw/            # gitignored
│   ├── processed/      # gitignored
│   └── README.md       # data sources and schema
├── notebooks/          # EDA and experiments
├── src/
│   ├── data/           # loaders, synthetic data generator
│   ├── features/       # feature engineering
│   ├── candidate_gen/  # two-tower model, ANN index
│   ├── ranking/        # ranking models
│   ├── rerank/         # diversity and business rules
│   ├── evaluation/     # metrics
│   ├── serving/        # FastAPI app
│   └── utils/
├── tests/
├── scripts/            # generate_data.py, train.py, evaluate.py
└── docs/               # design notes, diagrams
```

## Tech Stack (proposed)
- Python 3.10+
- Data: pandas, numpy, pyarrow
- ML: scikit-learn, LightGBM/XGBoost, PyTorch
- Retrieval: FAISS
- Experiment tracking: MLflow or Weights & Biases (TBD)
- API: FastAPI + Uvicorn
- Quality: pytest, ruff, black, pre-commit

## Environment Setup
```bash
git clone https://github.com/UdaySharmaGitHub/Rebuilding-Instagram-algorithm.git
cd Rebuilding-Instagram-algorithm

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
cp .env.example .env

python scripts/generate_data.py --users 1000 --posts 5000
python scripts/train.py --config configs/default.yaml
python scripts/evaluate.py
uvicorn src.serving.app:app --reload
```

## Data Plan (dummy for now)
- `users`: user_id, age_bucket, country, interests, created_at
- `posts`: post_id, author_id, media_type, topic, caption, created_at
- `interactions`: user_id, post_id, event_type, timestamp, dwell_time
- `follows`: follower_id, followee_id

Use a time-based train/validation/test split to avoid leakage.

## Evaluation Plan
- Retrieval: Recall@50, Recall@100
- Ranking: NDCG@10, MAP@10, AUC, log loss
- Beyond accuracy: diversity, novelty, coverage
- Baselines: popularity, recency, random

## Task Checklist
**Phase 1: Foundation**
- [ ] Initialize repo structure, packaging, and linting
- [ ] Write README with project overview
- [ ] Add `.gitignore`, `requirements.txt`, `.env.example`
- [ ] Set up CI (lint + tests)

**Phase 2: Data**
- [ ] Build synthetic data generator
- [ ] Define schemas and data validation
- [ ] EDA notebook

**Phase 3: Modeling**
- [ ] Popularity and recency baselines
- [ ] Feature engineering pipeline
- [ ] Two-tower candidate generation + FAISS index
- [ ] Ranking model (LightGBM, then neural)
- [ ] Re-ranking for diversity and freshness

**Phase 4: Evaluation and Serving**
- [ ] Offline evaluation suite
- [ ] FastAPI feed endpoint
- [ ] Experiment tracking

**Phase 5: Docs**
- [ ] Architecture diagram
- [ ] Design decision log
- [ ] Results and learnings write-up

## Risks and Open Questions
- Which dataset replaces synthetic data, if any?
- How realistic should simulated user behavior be?
- Compute limits for training neural models locally
- Cold-start handling for new users and posts

## Acceptance Criteria
- Fresh clone to running pipeline in under 10 minutes using the setup steps above
- Baseline and model metrics reproducible from a single command
- Tests pass in CI

---
*Dummy details above are placeholders and will be updated as the project evolves.*
