import numpy as np
from feature_engineering import (
    extract_facts, calculate_completeness_score, calculate_availability_score,
    is_honeypot, calculate_ai_yoe, has_pre_llm_experience, is_recent_coder,
    calculate_stability_score, is_consultancy_only, calculate_location_score,
    calculate_retrieval_score, calculate_evaluation_score, calculate_ai_yoe_score,
    calculate_experience_fit, calculate_research_penalty, calculate_title_chaser_penalty,
    calculate_production_fit
)


def score_candidate(c: dict) -> float:
    if is_honeypot(c):
        return 0.0

    # extract facts early so we can use shipping and notice signals
    facts = extract_facts(c)
    profile = c["profile"]
    yoe = profile.get("years_of_experience", 0)

    retrieval_score = calculate_retrieval_score(c)
    evaluation_score = calculate_evaluation_score(c)
    production_fit = calculate_production_fit(c)
    ai_yoe_score = calculate_ai_yoe_score(c)
    experience_fit = calculate_experience_fit(c)
    recent_coder = 1.0 if is_recent_coder(c) else 0.45
    consultancy_penalty = 0.5 if is_consultancy_only(c) else 1.0
    stability_score = calculate_stability_score(c)
    location_score = calculate_location_score(c)
    research_penalty = calculate_research_penalty(c)
    title_penalty = calculate_title_chaser_penalty(c)
    completeness_score = calculate_completeness_score(c)
    availability_score = calculate_availability_score(c)

    # Stronger boost for production fit when shipping/deployment evidence exists
    shipped_flag = facts.get("shipping_score", 0.0) > 0.25
    if shipped_flag:
        production_fit = min(production_fit * 1.4, 1.0)

    # Stronger penalty for long notice periods (user feedback driven)
    if facts.get("notice_period", 90) > 60:
        availability_score = availability_score * 0.5

    # Increase emphasis on production fit in relevance based on feedback
    # Relevance: add more weight to evaluation, slightly reduce AI-yoe, and include shipping evidence
    relevance = (
        0.30 * retrieval_score +
        0.30 * production_fit +
        0.20 * evaluation_score +
        0.10 * ai_yoe_score +
        0.10 * completeness_score
    )
    relevance += 0.15 * facts.get("shipping_score", 0.0)

    fit = experience_fit * recent_coder * consultancy_penalty * stability_score * location_score * research_penalty * title_penalty
    final_score = relevance * fit * availability_score
    return round(final_score, 4)


def rank_candidates(candidates: list[dict], top_n: int = 100) -> list[dict]:
    feats = []
    for c in candidates:
        score = score_candidate(c)
        facts = extract_facts(c)
        feats.append({
            "candidate_id": c["candidate_id"],
            "score": score,
            "facts": facts,
            "raw": c,
            "raw_candidate": c
        })

    feats.sort(key=lambda x: (-x["score"], x["candidate_id"]))
    return feats[:top_n]
