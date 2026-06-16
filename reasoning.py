import hashlib
import random


def _hash_seed(cid: str) -> int:
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % (2**32)


def _join_skills(skills: list[str]) -> str:
    if not skills:
        return "relevant AI systems"
    return ", ".join(skills[:4])


def _availability_note(signals: dict) -> str:
    notice = signals.get("notice_period_days", 90)
    open_flag = signals.get("open_to_work_flag", False)
    response_rate = signals.get("recruiter_response_rate", 0.0)

    if open_flag and notice <= 60:
        return f"Available with a manageable {notice}-day notice period."
    if open_flag:
        return f"Open to work, but note the {notice}-day notice period."
    if notice > 75:
        return f"Available, but note the longer {notice}-day notice period."
    if response_rate < 0.25:
        return "Limited recruiter response history may require extra outreach."
    return "Available for consideration."


def _fit_concern(facts: dict) -> str:
    if facts.get("notice_period", 90) > 75:
        return f"Primary concern is the {facts.get('notice_period')}-day notice period."
    if not facts.get("has_product_experience"):
        return "Primary concern is limited product-company shipping experience."
    if not facts.get("has_vector_db_experience") and not facts.get("has_hybrid_search_experience"):
        return "Some gaps in explicit vector DB or hybrid search deployment evidence in recent roles."
    if not facts.get("has_evaluation_experience"):
        return "Some gaps in explicit evaluation framework experience."
    if facts.get("is_researcher") and facts.get("shipping_score", 0.0) <= 0.2:
        return "Background leans more toward research than product delivery."
    if not facts.get("has_pre_llm_experience"):
        return "Minor gap is limited pre-LLM production experience."
    return "Minor gap is limited open-source evidence, though strong private production history is present."


def generate_reasoning(scored_candidate: dict, last_template_type: str = "") -> tuple[str, str]:
    facts = scored_candidate["facts"]
    score = scored_candidate["score"]
    cid = scored_candidate["candidate_id"]

    rng = random.Random(_hash_seed(cid))

    skills_str = _join_skills(facts.get("top_skills", []))
    company = facts.get("best_product_company") or facts.get("current_company", "the current company")

    # Deployed/project evidence
    deployed = False
    deploy_notes = "No clear deployed-project evidence found."
    try:
        ship = float(facts.get("shipping_score", 0.0))
        if ship > 0.25:
            deployed = True
            deploy_notes = f"Shows evidence of deployed projects (shipping_score={ship:.2f})."
    except Exception:
        pass

    # Small pool of varied openers and connectors to reduce repetition
    openers = []
    if score >= 0.70:
        openers = [
            f"Top-tier candidate with {facts.get('yoe', 0):.1f} years shipping AI systems.",
            f"Standout profile: {facts.get('yoe', 0):.1f} years of production ML experience and clear ownership at {company}.",
            f"High-confidence hire — {facts.get('yoe', 0):.1f} years of hands-on work building production systems."
        ]
        template_type = "strong"
    elif score >= 0.45:
        openers = [
            f"Highly relevant profile: {facts.get('yoe', 0):.1f} years in ML/AI roles, including product experience at {company}.",
            f"Good match: {facts.get('yoe', 0):.1f} years with practical systems work and product exposure at {company}.",
            f"Solid candidate with relevant product experience and a practical systems background."
        ]
        template_type = "decent"
    else:
        openers = [
            f"Observed fit: {skills_str} with {facts.get('yoe', 0):.1f} years at {company}.",
            f"Potential fit on specific signals ({skills_str}), though the role may require mentoring and scope alignment.",
            f"Candidate shows adjacent domain strength; core retrieval/ranking signals are weaker."
        ]
        template_type = "weak"

    opener = rng.choice(openers)

    # Variations for details and concerns
    detail_templates = [
        f"Notable skills: {skills_str}.",
        f"Key strengths include {skills_str} and product delivery experience at {company}.",
        f"Relevant toolkit: {skills_str}, with product-role exposure at {company}."
    ]
    detail = rng.choice(detail_templates)

    availability = _availability_note(scored_candidate["raw_candidate"]["redrob_signals"])
    concern = _fit_concern(facts)

    # Randomized closing phrasing that includes deployment evidence
    if deployed:
        deploy_phrase = deploy_notes
        closing_pool = [
            f"{deploy_phrase} {concern}",
            f"{deploy_phrase} One note: {concern.lower()}",
            f"{deploy_phrase} {concern}"
        ]
    else:
        deploy_phrase = deploy_notes
        closing_pool = [
            f"{deploy_phrase} {concern}",
            f"{concern} {deploy_phrase}",
            f"Note: {deploy_phrase} {concern}"
        ]

    closing = rng.choice(closing_pool)

    # Build final reasoning with varied connectors
    connectors = [
        f"{opener} {detail} {availability} {closing}",
        f"{opener} {availability} {detail} {closing}",
        f"{detail} {opener} {availability} {closing}",
        f"{opener} {detail} {closing}"
    ]

    reasoning = rng.choice(connectors)
    return reasoning, template_type
