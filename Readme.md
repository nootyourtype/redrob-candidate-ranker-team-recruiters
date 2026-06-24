# 🚀 Redrob Hackathon – Intelligent Candidate Discovery & Ranking Challenge

This repository contains our submission for the Redrob hackathon. Our ranking system selects the top 100 candidates from **100,000 profiles** for a **Senior AI Engineer (Ranking/Retrieval)** role using a **two-layer scoring architecture** — structured feature extraction, Redrob behavioral signals, TF-IDF job-description similarity, and multiplicative dealbreaker penalties.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point – loads candidates, runs ranking, outputs CSV. |
| `ranker.py` | **Layer 2** – weighted scoring, penalty engine, TF-IDF pre-compute, honeypot filter. |
| `feature_engineering.py` | **Layer 1** – 17 positive feature scorers + 11 dealbreaker detectors. |
| `reasoning.py` | Factual reasoning generator with title, platform, and recruiter signal citations. |
| `config.py` | Pattern sets, tuned weights, penalty factors, `REFERENCE_DATE`. |
| `compute_metrics.py` | Pearson correlation vs. human labels (CLI argument for CSV path). |
| `validate_submission.py` | Format validator (provided in the challenge bundle). |
| `data1/candidates.zip` | **Compressed candidate dataset** (~54 MB, 100K candidates). |
| `data1/sample_candidates.json` | Small sample (50 candidates) for quick testing. |
| `data1/candidate_schema.json` | JSON schema for a single candidate record. |
| `data1/sample_submission.csv` | Reference submission format. |
| `data1/submission_metadata_template.yaml` | Metadata template required for submission. |
| `Team_recruiters_submission.csv` | **Final generated submission** (top 100). |

> **Note on large files**: Raw `candidates.jsonl` (~464 MB) and `data1.zip` (~108 MB) are excluded via `.gitignore`. Use `data1/candidates.zip` (~54 MB), which the pipeline reads natively.

---

## 🚀 Setup and Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd <repo-folder>
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Data is ready** – `data1/candidates.zip` is included. No extra download needed.

---

## 🧠 Approach Overview

Human recruiters weigh *implicit signals* (cultural fit, coding recency, hireability, domain misalignment) as heavily as keyword matches. Our ranker encodes both through **17 positive features** and **11 multiplicative dealbreakers**.

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Layer 1: Feature Extraction  (feature_engineering.py)     │
│  ──────────────────────────────────────────────────────────  │
│  17 Positive Feature Scorers  →  each returns [0, 1]       │
│  11 Dealbreaker Detectors     →  each returns bool         │
│  TF-IDF JD similarity cache   →  built once per run        │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│  Layer 2: Weighted Scoring  (ranker.py)                      │
│  ──────────────────────────────────────────────────────────  │
│  Skip honeypots (score = 0)                                │
│  Base Score  = Σ (weight_i × feature_i)                    │
│  Penalty     = Π (penalty_factor_j)                        │
│  Final Score = Base Score × Penalty Multiplier             │
│  Tie-break   = candidate_id ascending                      │
└──────────────────────────────────────────────────────────────┘
```

### Layer 1: Positive Feature Scorers (17 features)

Weights are **tuned against human-labeled candidates** — emphasizing retrieval/production depth over title wording and recruiter visibility alone.

| Feature | Weight | What It Captures |
|---------|--------|------------------|
| `retrieval_score` | 0.17 | Core retrieval/ranking/embedding keyword depth |
| `production_fit` | 0.13 | Vector DB + hybrid search + eval + shipping composite |
| `evaluation_score` | 0.08 | NDCG, MRR, precision@k, A/B testing evidence |
| `pre_llm_score` | 0.06 | Pre-2022 search/ranking + classic ML tools |
| `ai_yoe_score` | 0.06 | Total years in AI/ML roles |
| `ranking_yoe_score` | 0.08 | Years in ranking/search roles (title + description) |
| `title_fit_score` | 0.06 | Current title alignment, with career-depth override |
| `jd_similarity_score` | 0.04 | TF-IDF cosine similarity to job description |
| `recruiter_demand_score` | 0.03 | Saves, search appearances, profile views (30d) |
| `platform_skill_score` | 0.03 | Redrob `skill_assessment_scores` for relevant skills |
| `product_engineer_fit` | 0.05 | Non-consultancy employers + shipping orientation |
| `experience_fit` | 0.04 | YOE sweet-spot (5–9 years ideal) |
| `recent_coder_score` | 0.04 | Hands-on coding within 18 months |
| `location_score` | 0.03 | Pune/Noida preferred → metro → relocatable |
| `notice_score` | 0.02 | Shorter notice period = better |
| `response_score` | 0.05 | Response rate, activity, interview completion |
| `shipping_score` | 0.03 | Track record of deploying/shipping systems |

#### Key scoring enhancements

- **`title_fit_score` override**: Generic titles (e.g. "AI Specialist") are boosted when retrieval/production evidence is strong — career depth outweighs title wording.
- **`ranking_yoe_score`**: Detects ranking/search keywords in job **descriptions**, not just titles.
- **`jd_similarity_score`**: `TfidfVectorizer` fit on full 100K corpus + JD; cosine similarity per candidate.
- **`response_score`**: Now includes `interview_completion_rate` from Redrob signals.
- **Honeypot pre-filter**: Candidates scoring 0.0 are excluded before top-100 selection.

### Layer 1: Dealbreaker Detectors (11 penalties)

| Dealbreaker | Penalty | Detection Logic |
|-------------|---------|-----------------|
| **Honeypot** | 0.00 | Fictional companies, YoE mismatch, expert skills with 0 months, keyword-stuffed summary |
| **Non-Compete** | 0.15 | Non-compete / restrictive covenant language |
| **CV/Speech Domain** | 0.30 | ≥3 CV/speech terms AND ≤1 ranking term |
| **Manager-Only** | 0.40 | Director/VP/Manager title + no recent coding |
| **Consultancy-Only** | 0.50 | Entire career at service companies |
| **Culture Misfit** | 0.50 | Stability-seeking language vs. high-velocity JD |
| **Job Hopper** | 0.55 | Average tenure < 18 months |
| **Flight Risk** | 0.55 | Offer acceptance rate < 20% |
| **Framework Enthusiast** | 0.60 | LangChain/LlamaIndex without eval frameworks |
| **Ghost** | 0.35 | Low response rate + inactive > 60 days |
| **Research-Only** | 0.60 | Research-heavy + low shipping evidence |

> Penalties are **multiplicative** — stacked concerns compound, mirroring recruiter judgment.

### Honeypot landscape (100K dataset)

| Metric | Count |
|--------|------:|
| Total candidates | 100,000 |
| Honeypots detected | 84,103 (84.1%) |
| Rankable pool | 15,897 |

Primary trigger: fictional company history (~81,500 profiles). The ranker must find the best candidates among the ~16K real profiles.

### Layer 2: Final Score Computation

```python
final_score = base_score × penalty_multiplier
# base_score = Σ (POSITIVE_WEIGHTS[feat] × feature_score[feat])
# penalty_multiplier = Π (penalty_factor for each triggered dealbreaker)
```

All weights and penalties live in `config.py`. Time-based features use a fixed **`REFERENCE_DATE = 2026-06-01`** for fully reproducible scores across runs.

---

## 🔍 Reasoning Generation

For each candidate in the top 100, `reasoning.py` produces a factual justification that:

1. **Zero hallucination** – only cites verified profile data.
2. **Title-aware intros** – distinguishes ranking/retrieval roles from generic ML titles.
3. **Technical depth** – vector DBs, hybrid search, evaluation frameworks, shipping track record.
4. **Redrob platform signals** – recruiter saves, search appearances, skill assessment scores, interview completion rate.
5. **Education tier** – cites tier-1 institutions when present.
6. **Honest concerns** – dealbreakers, title misalignment, evaluation gaps, low platform demand.
7. **Honeypot explanations** – specific reasons for suspicious profiles.
8. **Layout variation** – multiple narrative structures, seeded per `candidate_id` for diversity.

---

## ⚙️ How to Run

### Generate the submission CSV

Default path (`data1/candidates.zip`):
```bash
python main.py --out Team_recruiters_submission.csv --top_n 100
```

Explicit paths:
```bash
# Full dataset (recommended)
python main.py --candidates data1/candidates.zip --out Team_recruiters_submission.csv --top_n 100

# Quick test (50 candidates)
python main.py --candidates data1/sample_candidates.json --out test_output.csv --top_n 10
```

Supported formats: `.json`, `.jsonl`, `.jsonl.gz`, `.zip` (containing a `.jsonl`).

### Validate and evaluate

```bash
python validate_submission.py Team_recruiters_submission.csv
python compute_metrics.py Team_recruiters_submission.csv
```

`compute_metrics.py` accepts an optional CSV path (defaults to `Team_recruiters_submission.csv`).

---

## ⏱️ Performance & Compute Compliance

| Constraint | Limit | Actual |
|------------|-------|--------|
| **Runtime** | < 5 minutes (CPU) | **~71 seconds** |
| **Memory** | < 16 GB | **< 2 GB** |
| **Network** | None (offline) | **No external API calls** |
| **GPU** | Not used | **CPU only** |

### Performance optimizations

- Pre-compiled regex patterns cached in `_PATTERN_CACHE`
- LRU-cached `normalize_text()` (512 entries)
- Shared `_full_text()` / `_career_text()` builders
- Early honeypot exit before positive feature computation
- TF-IDF matrix built once per run, not per candidate
- Honeypot candidates excluded before sort

---

## 📊 Results

### Score distribution (100K → Top 100)

| Metric | Value |
|--------|-------|
| Top score (Rank 1) | **0.9085** |
| 100th score | **0.7341** |
| Score spread | **0.1744** |

**Rank 1**: `CAND_0077337` – Staff ML Engineer @ Paytm  
**Rank 100**: `CAND_0024878` – AI Specialist @ Krutrim

### Ground truth validation (human-labeled spot-check)

| Candidate | Human Label | In Top 100? | Rank | Score |
|-----------|------------|-------------|------|-------|
| CAND_0048558 | **2** (best) | ✅ Yes | 64 | 0.7576 |
| CAND_0096104 | **2** (best) | ✅ Yes | 63 | 0.7578 |
| CAND_0061257 | 1 (good) | ✅ Yes | 69 | 0.7542 |
| CAND_0064904 | 1 (good) | ✅ Yes | 95 | 0.7372 |
| CAND_0007009 | **0** (bad) | ❌ No | — | — |

Label=2 candidates rank above label=1 `CAND_0064904` (rank 95). The label=0 candidate is correctly excluded (caught by `research_only` penalty).

---

## 📝 Reproducing Our Exact Submission

```bash
pip install -r requirements.txt
python main.py --candidates data1/candidates.zip --out Team_recruiters_submission.csv --top_n 100
python validate_submission.py Team_recruiters_submission.csv
python compute_metrics.py Team_recruiters_submission.csv
```

---

## 🤖 AI Tools Declaration

We used AI tools as part of our development workflow:
- **Claude / Cursor** – architecture, code review, feature design, reasoning engine, weight tuning.
- **GitHub Copilot** – autocompletion and boilerplate.

**No candidate data** was sent to any LLM. The ranking logic is entirely deterministic and reproducible offline.

---

## 🧪 Dependencies

See `requirements.txt`:

| Package | Purpose |
|---------|---------|
| Python ≥ 3.8 | Runtime |
| `scikit-learn` ≥ 1.3 | TF-IDF vectorization + cosine similarity for `jd_similarity_score` |

---

## ⚠️ Large File Notice

| File | Size | Reason |
|------|------|--------|
| `data1/candidates.jsonl` | ~464 MB | Exceeds GitHub 100 MB limit |
| `data1.zip` | ~108 MB | Exceeds GitHub 100 MB limit |

**Use instead**: `data1/candidates.zip` (~54 MB), included in the repo.

---

## 📬 Contact

For questions about this repository, please reach out to the team via the hackathon portal.
