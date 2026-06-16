# Redrob Hackathon – Intelligent Candidate Discovery & Ranking Challenge

This repository contains our submission for the Redrob hackathon. Our ranking system selects the top 100 candidates for the given job description, combining skill‑based relevance with behavioural signals and thoughtful penalisation.

---

## 📁 Repository Structure

| File | Description |
|------|-------------|
| `main.py` | Entry point – loads candidates, runs ranking, and outputs the final CSV. |
| `ranker.py` | Core scoring and ranking logic. |
| `feature_engineering.py` | Feature extraction functions (retrieval score, production fit, stability, etc.). |
| `reasoning.py` | Generates 1‑2 sentence reasoning for each candidate using diversified templates. |
| `config.py` | Configuration: JD keywords, penalty lists, city preferences, thresholds. |
| `compute_metrics.py` | Metric computation utilities. |
| `validate_submission.py` | Format validator (provided in the challenge bundle). |
| `hybrid_ranker.py` | Hybrid ranking utilities. |
| `data/candidates.zip` | **Compressed candidate dataset** (~54 MB, tracked in Git). |
| `data/sample_candidates.json` | Small sample for quick testing. |
| `data/candidate_schema.json` | JSON schema for a single candidate record. |
| `data/sample_submission.csv` | Reference submission format. |
| `data/submission_metadata_template.yaml` | Metadata template required for submission. |
| `Team_recruiters_submission.csv` | Our final generated submission. |

> **Note on large files**: The raw `candidates.jsonl` (~464 MB) and `data.zip` (~108 MB) are excluded via `.gitignore` because they exceed GitHub's 100 MB file-size limit. The pipeline reads directly from `data/candidates.zip` (~54 MB), which is fully tracked in this repo.

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

4. **Data is ready to use** – `data/candidates.zip` is already included in the repo.  
   No extra download step is needed. The pipeline extracts it on-the-fly.

---

## 🧠 Approach Overview

Our ranking system is a **multi‑factor scoring model** that balances candidate relevance, fit to the JD, and real‑world availability.

### 1. Relevance Components
- **Retrieval Score** – Measures the presence of JD‑core terms (retrieval, ranking, embedding, RAG, etc.) in the candidate's summary and career history.
- **Production Fit** – Rewards explicit experience with vector databases, hybrid search, and evaluation frameworks. Also considers "shipping" evidence (keywords like *deployed*, *shipped*, *launched*).
- **Evaluation Score** – Counts mentions of evaluation terms (NDCG, MRR, A/B testing, etc.).
- **AI YOE Score** – Favours candidates with ≥5 years of dedicated AI/ML experience.

### 2. Fit Modifiers (Multipliers)
- **Experience Fit** – Adjusts based on total YOE (ideal 5‑9 years) and the ratio of AI‑specific experience.
- **Recent Coder** – Penalises candidates who have not written production code recently.
- **Consultancy Penalty** – Lowers score for candidates who have only worked at pure‑services firms (TCS, Infosys, Accenture, etc.).
- **Stability Score** – Rewards longer average tenure (≥3 years per role).
- **Location Score** – Boosts candidates in preferred cities (Pune, Noida) and those willing to relocate.
- **Research Penalty** – Reduces score for overly academic profiles with low shipping evidence.
- **Title‑Chaser Penalty** – Penalises candidates with frequent short stays and inflated titles.

### 3. Availability Signals
We incorporate behavioural signals from the `redrob_signals` object:
- Open‑to‑work flag
- Recruiter response rate
- Notice period (shorter is better)
- Willingness to relocate

These are combined into an **availability score** that multiplies the relevance×fit product.

### 4. Reasoning Generation
For each candidate in the top 100, we produce a unique reasoning string that:
- References specific facts (YOE, skills, current company).
- Acknowledges both strengths and concerns.
- Uses varied templates to avoid repetition.

---

## ⚙️ How to Run

### Generate the Submission CSV

Using the default path (candidates.zip, already in repo):
```bash
python main.py --out submission.csv --top_n 100
```

Or specify the data file explicitly:
```bash
# From the compressed zip (recommended – already in Git)
python main.py --candidates data/candidates.zip --out submission.csv --top_n 100

# From a raw JSONL (if you have it locally)
python main.py --candidates data/candidates.jsonl --out submission.csv --top_n 100

# From a gzipped JSONL
python main.py --candidates data/candidates.jsonl.gz --out submission.csv --top_n 100
```

Supported formats: `.jsonl`, `.jsonl.gz`, `.zip` (containing a `.jsonl`).

### Validate the CSV
```bash
python validate_submission.py submission.csv
```
This checks header, column order, row count, rank uniqueness, score monotonicity, and candidate ID format.

---

## ⏱️ Compute Compliance

- **Runtime**: < 5 minutes on CPU (tested on a MacBook Pro M2, 16 GB RAM).
- **Memory**: < 16 GB.
- **Network**: Off – no external API calls.
- **GPU**: Not used.

All feature engineering and scoring are implemented with vectorised operations (scikit‑learn, NumPy) and efficient string matching.

---

## 🤖 AI Tools Declaration

We used AI tools as part of our development workflow:
- **Claude** – for architecture discussions, code reviews, and reasoning template design.
- **GitHub Copilot** – for autocompletion and boilerplate generation.

**No candidate data** was sent to any LLM at any point. The final ranking logic is entirely deterministic and reproducible offline.

---

## 📊 Reproducing Our Exact Submission

To reproduce the exact CSV we submitted:

```bash
python main.py --candidates data/candidates.zip --out Team_recruiters_submission.csv --top_n 100
```

You can verify it with the validator:
```bash
python validate_submission.py Team_recruiters_submission.csv
```

---

## 📝 Notes on the Reasoning Column

Our reasoning strings are generated to meet Stage 4 review requirements:
- They reference specific facts (skills, YOE, company, notice period).
- They connect to JD requirements (e.g., production experience, evaluation skills).
- They honestly acknowledge gaps (e.g., missing vector DB experience, long notice periods).
- Each entry is distinct and varies in structure.

---

## 🧪 Dependencies

See `requirements.txt`. Key packages:
- Python ≥ 3.8
- scikit‑learn
- numpy

---

## ⚠️ Large File Notice

GitHub enforces a **100 MB per-file limit**. The following files are excluded from this repository via `.gitignore`:

| File | Size | Reason |
|------|------|--------|
| `data/candidates.jsonl` | ~464 MB | Exceeds GitHub limit |
| `data.zip` | ~108 MB | Exceeds GitHub limit |

**What to use instead**: `data/candidates.zip` (~54 MB) is included in the repo and contains the same candidate data. The pipeline handles it natively – no changes needed.

If you need the original raw files, contact the team via the hackathon portal.

---

## 📬 Contact

For questions about this repository, please reach out to the team via the hackathon portal.
