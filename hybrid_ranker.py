# hybrid_ranker.py
import json
import csv
import gzip
import zipfile
from feature_engineering import extract_facts
from reasoning import generate_reasoning

def load_candidates_dict(filepath: str) -> dict:
    """Loads candidates from .jsonl, .jsonl.gz, or .zip into a dict."""
    content = ""
    
    if filepath.endswith('.zip'):
        with zipfile.ZipFile(filepath, 'r') as zf:
            jsonl_files = [f for f in zf.namelist() if f.endswith('.jsonl')]
            if not jsonl_files:
                raise FileNotFoundError("No .jsonl file found inside the zip.")
            with zf.open(jsonl_files[0], 'r') as f:
                content = f.read().decode('utf-8')
    elif filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            content = f.read().strip()
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    
    if not content:
        return {}
        
    raw_data = [json.loads(line) for line in content.split('\n') if line.strip()]
    
    candidates_dict = {}
    for c in raw_data:
        cid = str(c.get("candidate_id", "")).strip()
        candidates_dict[cid] = c
    return candidates_dict

def main():
    # 1. Configuration Paths
    scores_csv = "final_submission_top100_improved_prepared.csv"
    candidates_file = "data/candidates.zip"  # Uses the zip included in the repo (~54 MB)
    output_csv = "final_hybrid_submission.csv"
    
    print(f"📂 Loading candidate profiles from {candidates_file}...")
    candidates_dict = load_candidates_dict(candidates_file)
    print(f"✅ Loaded {len(candidates_dict)} candidates into memory.")
    
    print(f"📊 Reading ranked scores from {scores_csv}...")
    ranked_rows = []
    with open(scores_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ranked_rows.append(row)
    print(f"✅ Read {len(ranked_rows)} ranked candidates.")
    
    print("🧠 Generating Stage 4-proof reasoning for top 100...")
    final_results = []
    last_template_type = ""
    
    for row in ranked_rows:
        cid = row['candidate_id'].strip()
        score = float(row['score'])
        rank = int(row['rank'])
        
        if cid not in candidates_dict:
            print(f"⚠️ Warning: Candidate {cid} not found in dataset. Skipping.")
            continue
            
        candidate = candidates_dict[cid]
        facts = extract_facts(candidate)
        
        # Generate reasoning using our proven, diversified template engine
        reasoning, last_template_type = generate_reasoning({
            "candidate_id": cid,
            "score": score,
            "facts": facts,
            "raw_candidate": candidate
        }, last_template_type)
        
        final_results.append({
            "candidate_id": cid,
            "rank": rank,
            "score": score,
            "reasoning": reasoning
        })
        
    print(f"✅ Successfully generated reasoning for {len(final_results)} candidates.")
    
    print(f"💾 Writing final hybrid submission to {output_csv}...")
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["candidate_id", "rank", "score", "reasoning"])
        writer.writeheader()
        writer.writerows(final_results)
        
    print("🎉 HYBRID SUBMISSION GENERATED SUCCESSFULLY!")
    print("This file now combines your advanced scoring logic with Stage 4-proof reasoning.")
    print("\n👉 Next Step: Run the validator on this new file:")
    print(f"   python validate_submission.py {output_csv} {candidates_file}")

if __name__ == "__main__":
    main
