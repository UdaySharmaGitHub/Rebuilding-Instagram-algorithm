# Project: Two-Tower Short-Video Recommender (Instagram Explore-style)

## What we're building
A recommender that picks videos for each user, similar in spirit to the model Meta describes for Instagram Explore. Users and videos are turned into vectors so that users land close to the videos they'd like. Recommendation then becomes a nearest-match search.

Instagram's production system is proprietary, so this is a learning re-implementation trained on real public watch data from a short-video app (KuaiRec, from Kuaishou).

## Goals
- Train a two-tower retrieval model on real watch data
- Beat a popularity baseline on Recall@50
- Correct for popularity bias during training (logQ correction)
- Visualize video embeddings and show category clusters
- Stretch: add a re-ranking step over the top 200 candidates

## Non-goals
- Matching Instagram's production scale or exact behavior
- Scraping Instagram or using private user data

## Dataset
- **KuaiRec**: real watch data from the Kuaishou short-video app, with millions of interactions. A small fully-observed test set has 1,411 users who watched almost every video, which allows fair offline evaluation.
  - Site: https://kuairec.com/ (CC BY-SA 4.0)
  - Download links and docs: https://github.com/chongminggao/KuaiRec
- Paper: https://arxiv.org/abs/2202.10842

Data is stored under `data/raw/` and `data/processed/`, both gitignored.

## Pipeline

### Step 1: Get the data
Download KuaiRec and document the schema in `data/README.md`.

### Step 2: Define "liked"
The dataset records how much of each video was watched. A video counts as **liked** if the user watched it **two or more times**. This threshold lives in `configs/default.yaml` so it can be changed.

### Step 3: Build the two towers
- **User tower:** a small network that maps user features to a 64-dimensional vector.
- **Item tower:** a small network that maps video features (video ID and categories) to a 64-dimensional vector.
- **Score:** similarity between the two vectors (dot product or cosine).

### Step 4: Train
Use in-batch negatives. Each batch contains user-video pairs where the user liked the video. The model learns to score the liked video higher than the other videos in the same batch.

### Step 5: Correct for popularity bias (logQ)
Popular videos appear in almost every batch, so the model can learn to push them down. During training, add a small boost based on each video's sampling probability (logQ correction), following Google's sampling-bias-corrected two-tower method.

### Step 6: Evaluate
For each test user in the fully-observed set:
1. Take the top 50 videos by score.
2. Count how many of the videos the user liked appear in that list. This is **Recall@50**.
3. Compare against a **popularity baseline** that shows the 50 most popular videos.

If the model does not beat the popularity baseline, keep improving it before moving on.

### Step 7: Visualize
Reduce video vectors to 2D (UMAP or t-SNE), color points by category, and check that similar videos form clusters. Save the figure to `docs/`.

### Stretch: Re-ranking
Take the top 200 candidates from retrieval and re-rank them with a second model that uses extra features, such as video length. Report how Recall@50 changes compared to retrieval alone.

## Proposed repository structure
```
Rebuilding-Instagram-algorithm/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   └── default.yaml        # liked threshold, embedding size, training params
├── data/
│   ├── raw/                # gitignored
│   ├── processed/          # gitignored
│   └── README.md           # KuaiRec schema and download steps
├── notebooks/              # EDA and visualization
├── src/
│   ├── data/               # loading, liked-label construction
│   ├── models/             # user tower, item tower, logQ correction
│   ├── train.py
│   ├── evaluate.py         # Recall@50, popularity baseline
│   ├── visualize.py        # 2D embedding plots
│   └── rerank/             # stretch: second-stage re-ranker
├── tests/
├── docs/
│   ├── PROJECT_SETUP.md
│   └── figures/
└── scripts/
```

## Tech stack
- Python 3.10+
- PyTorch (two-tower model), with TorchRec or TensorFlow Recommenders as reference implementations
- pandas, numpy, pyarrow
- FAISS for nearest-neighbor search at retrieval time
- UMAP or scikit-learn for visualization
- pytest, ruff, black

## Setup
```bash
git clone https://github.com/UdaySharmaGitHub/Rebuilding-Instagram-algorithm.git
cd Rebuilding-Instagram-algorithm

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Download KuaiRec into data/raw/ (see data/README.md)
python -m src.train --config configs/default.yaml
python -m src.evaluate --config configs/default.yaml
python -m src.visualize --config configs/default.yaml
```

## Reading list
- Engineering at Meta: Scaling the Instagram Explore recommendations system (https://engineering.fb.com/2023/08/09/ml-applications/scaling-instagram-explore-recommendations-system/)
- Google Research: Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations (https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)
- KuaiRec: A Fully-observed Dataset (https://arxiv.org/abs/2202.10842)
- TensorFlow Recommenders, basic retrieval tutorial (https://www.tensorflow.org/recommenders/examples/basic_retrieval)
- TorchRec retrieval example (https://github.com/pytorch/torchrec/tree/main/examples/retrieval)

## Results (to be filled in)
- Dataset size: [N]M interactions
- Recall@50, two-tower model: [X]%
- Recall@50, popularity baseline: [Y]%

## Resume line
Built a two-tower recommender in PyTorch on [N]M real short-video interactions (KuaiRec), reaching [X]% Recall@50 versus [Y]% for a popularity baseline.

## Task checklist
**Phase 1: Foundation**
- [ ] Repo structure, packaging, linting, `.gitignore`
- [ ] README with project overview
- [ ] Download KuaiRec and document schema

**Phase 2: Data**
- [ ] Build liked labels (watch count ≥ 2)
- [ ] Train/test split using the fully-observed test users
- [ ] EDA notebook

**Phase 3: Model**
- [ ] Popularity baseline and Recall@50 evaluation
- [ ] User and item towers
- [ ] In-batch negative training
- [ ] logQ popularity correction

**Phase 4: Analysis**
- [ ] Recall@50 comparison against baseline
- [ ] 2D embedding visualization by category
- [ ] Stretch: re-ranking top 200 with video length

**Phase 5: Docs**
- [ ] Results section filled in
- [ ] Resume line updated with final numbers

## Open questions
- Which features beyond video ID and category go into the item tower?
- Should the user tower use watch history as well as user IDs?
- How many negatives per batch fit in memory on the available hardware?

---
*This plan is a working draft and will be updated as the project progresses.*
