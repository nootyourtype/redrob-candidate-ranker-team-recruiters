# reasoning.py - Factual Reasoning Sentence Generator
# Version tracked in git repository.
import hashlib
import random


def _hash_seed(cid: str) -> int:
    return int(hashlib.md5(cid.encode()).hexdigest(), 16) % (2**32)


def _top_assessments(assessments: dict, limit: int = 2) -> list[str]:
    if not assessments:
        return []
    ranked = sorted(assessments.items(), key=lambda x: x[1], reverse=True)
    return [f"{name} ({score:.0f})" for name, score in ranked[:limit]]


def generate_reasoning(scored_candidate: dict, last_template_type: str = "") -> tuple[str, str]:
    from config import VECTOR_DBS, EVALUATION_TERMS, HYBRID_SEARCH_TERMS, MUST_HAVE_CONCEPTS
    from feature_engineering import (
        is_honeypot, calculate_ai_yoe, is_consultancy_only,
        is_researcher_profile, is_recent_coder, is_fictional_company_history,
        calculate_shipping_score, has_non_compete, is_cv_speech_domain,
        is_job_hopper, is_manager_only, is_framework_enthusiast, is_ghost,
        has_flight_risk, is_culture_misfit, calculate_title_fit_score,
    )

    facts = scored_candidate["facts"]
    score = scored_candidate["score"]
    cid = scored_candidate["candidate_id"]
    candidate = scored_candidate.get("raw_candidate") or scored_candidate.get("raw")

    rng = random.Random(_hash_seed(cid))

    # 1. Flag Honeypots with highly specific reasons
    if is_honeypot(candidate):
        reasons = []
        if is_fictional_company_history(candidate):
            history = candidate.get("career_history", [])
            companies = [job.get("company", "") for job in history] + [candidate["profile"].get("current_company", "")]
            fictional = [c for c in companies if c.lower() in [
                "dunder mifflin", "stark industries", "wayne enterprises", "acme corp", "hooli",
                "pied piper", "initech", "globex inc", "umbrella corp", "oscorp"
            ]]
            company_str = f" ({fictional[0]})" if fictional else ""
            reasons.append(f"presence of fictional company history{company_str}")

        profile = candidate["profile"]
        yoe = profile.get("years_of_experience", 0)
        total_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
        sum_yoe = total_months / 12.0
        if abs(yoe - sum_yoe) >= 1.0:
            reasons.append(
                f"inconsistency between stated experience ({yoe} YoE) and career history duration ({sum_yoe:.1f} years)"
            )

        skills = candidate.get("skills", [])
        exp_zero_skills = [
            s.get("name", "") for s in skills
            if s.get("proficiency") in {"expert", "advanced"} and s.get("duration_months", 0) == 0
        ]
        if exp_zero_skills:
            reasons.append(f"expert/advanced skills ({', '.join(exp_zero_skills[:2])}) listed with 0 months of usage")

        summary_clean = candidate["profile"].get("summary", "").lower()
        has_concepts = any(concept in summary_clean for concept in MUST_HAVE_CONCEPTS.union(VECTOR_DBS))
        has_titles = any(t in summary_clean for t in ["engineer", "developer", "scientist", "architect", "programmer"])
        if has_concepts and not has_titles:
            reasons.append("keyword-stuffed summary without technical titles or job roles")

        reason_str = ", ".join(reasons) if reasons else "suspicious profile inconsistency"
        reasoning = f"Profile flagged as suspicious/honeypot due to {reason_str}."
        return reasoning, "honeypot"

    # 2. Extract verified facts (zero hallucination)
    yoe = facts.get("yoe", 0)
    ai_yoe = calculate_ai_yoe(candidate)
    current_title = facts.get("current_title", "Engineer")
    current_company = facts.get("current_company", "Current Company")
    best_company = facts.get("best_product_company") or current_company
    title_fit = facts.get("title_fit_score", calculate_title_fit_score(candidate))

    skills_names = [s["name"].lower() for s in candidate.get("skills", [])]
    summary_lower = candidate["profile"].get("summary", "").lower()
    history_desc = " ".join([job.get("description", "").lower() for job in candidate.get("career_history", [])])

    matched_vector_dbs = []
    for db in VECTOR_DBS:
        if db in skills_names or f" {db} " in f" {summary_lower} " or f" {db} " in f" {history_desc} ":
            matched_vector_dbs.append(db.title())

    matched_eval = []
    for term in EVALUATION_TERMS:
        if term in skills_names or f" {term} " in f" {summary_lower} " or f" {term} " in f" {history_desc} ":
            matched_eval.append(term)
    eval_display = []
    if any(t in matched_eval for t in ["ndcg", "mrr", "map"]):
        eval_display.append("search metrics (NDCG/MRR)")
    if any(t in matched_eval for t in ["ab test", "ab testing"]):
        eval_display.append("A/B testing")
    if any(t in matched_eval for t in ["offline evaluation", "online evaluation", "evaluation framework"]):
        eval_display.append("evaluation frameworks")
    if not eval_display and matched_eval:
        eval_display.append(matched_eval[0])

    matched_hybrid = []
    for term in HYBRID_SEARCH_TERMS:
        if term in skills_names or f" {term} " in f" {summary_lower} " or f" {term} " in f" {history_desc} ":
            matched_hybrid.append(term)
    hybrid_display = []
    if "hybrid search" in matched_hybrid or "dense retrieval" in matched_hybrid or "sparse retrieval" in matched_hybrid:
        hybrid_display.append("hybrid retrieval (dense/sparse)")
    elif "vector search" in matched_hybrid or "semantic search" in matched_hybrid:
        hybrid_display.append("semantic vector search")
    elif "bm25" in matched_hybrid:
        hybrid_display.append("BM25 retrieval")
    if not hybrid_display and matched_hybrid:
        hybrid_display.append(matched_hybrid[0])

    top_skills = facts.get("top_skills", [])
    skills_str = ", ".join(top_skills[:3]) if top_skills else "machine learning"

    signals = candidate.get("redrob_signals", {})
    notice = signals.get("notice_period_days", 90)
    open_to_work = signals.get("open_to_work_flag", False)
    saves = facts.get("saved_by_recruiters_30d", 0)
    search_apps = facts.get("search_appearance_30d", 0)
    interview_rate = facts.get("interview_completion_rate", 0.0)
    assessments = facts.get("skill_assessments", {})
    assessment_display = _top_assessments(assessments)

    loc = facts.get("location", "Unknown").lower()
    is_primary_loc = any(city in loc for city in ["pune", "noida"])
    is_secondary_loc = any(city in loc for city in ["delhi", "ncr", "mumbai", "hyderabad", "bangalore", "bengaluru"])
    willing_to_relocate = signals.get("willing_to_relocate", False)

    if is_primary_loc:
        location_note = f"based in preferred location ({facts.get('location')})"
    elif willing_to_relocate:
        location_note = "willing to relocate to Pune/Noida"
    elif is_secondary_loc:
        location_note = f"located in {facts.get('location')}"
    else:
        location_note = f"based in {facts.get('location')}"

    # Intro — tone by score, title-aware
    if score >= 0.75:
        template_type = "strong"
        if facts.get("title_is_ranking_role"):
            intro_options = [
                f"Strong retrieval/ranking fit as {current_title} at {best_company} with {yoe:.1f} YOE ({ai_yoe:.1f} years in AI/ML).",
                f"Excellent JD alignment: {current_title} at {best_company} with {yoe:.1f} years building search and ranking systems.",
            ]
        else:
            intro_options = [
                f"Strong Senior AI Engineer with {yoe:.1f} YOE ({ai_yoe:.1f} years in AI/ML), currently {current_title} at {best_company}.",
                f"High-caliber candidate with {yoe:.1f} years of production ML experience as {current_title} at {best_company}.",
            ]
    elif score >= 0.50:
        template_type = "decent"
        intro_options = [
            f"Qualified ML candidate with {yoe:.1f} YOE ({ai_yoe:.1f} years in AI roles) as {current_title} at {best_company}.",
            f"Solid mid-to-senior profile as {current_title} at {best_company} with {yoe:.1f} years of applied systems work.",
        ]
    else:
        template_type = "weak"
        intro_options = [
            f"Candidate with {yoe:.1f} YOE as {current_title} at {best_company}.",
            f"Adjacent profile with {yoe:.1f} YOE and exposure to {skills_str}.",
        ]
    intro_phrase = rng.choice(intro_options)

    # Technical strengths
    tech_points = []
    if facts.get("title_is_ranking_role"):
        tech_points.append(f"current role ({current_title}) directly targets ranking/retrieval engineering")
    elif title_fit >= 0.75:
        tech_points.append(f"current title ({current_title}) is well aligned with senior ML engineering")
    if matched_vector_dbs:
        tech_points.append(f"hands-on vector database work ({', '.join(matched_vector_dbs[:2])})")
    if hybrid_display:
        tech_points.append(f"practical {hybrid_display[0]} experience")
    if eval_display:
        tech_points.append(f"{eval_display[0]} in production contexts")
    ship_score = calculate_shipping_score(candidate)
    if ship_score > 0.3 or facts.get("product_engineer_fit", 0) >= 0.6:
        tech_points.append("demonstrated product shipping track record at non-consultancy employers")
    if facts.get("jd_similarity_score", 0) >= 0.65:
        tech_points.append("strong overall textual alignment with the job description")

    if len(tech_points) >= 3:
        tech_phrase = f"Technical strengths include {tech_points[0]}, {tech_points[1]}, and {tech_points[2]}."
    elif len(tech_points) == 2:
        tech_phrase = f"Technical strengths include {tech_points[0]} and {tech_points[1]}."
    elif tech_points:
        tech_phrase = f"Demonstrates {tech_points[0]}."
    else:
        tech_phrase = f"Technical toolkit includes {skills_str}."

    # Recruiter / platform signals
    platform_points = []
    if saves >= 20 or search_apps >= 150:
        platform_points.append(
            f"high recruiter demand ({saves} saves, {search_apps} search appearances in last 30 days)"
        )
    elif saves >= 5 or search_apps >= 50:
        platform_points.append(f"solid platform visibility ({saves} recruiter saves, {search_apps} search appearances)")
    if assessment_display:
        platform_points.append(f"verified Redrob assessments: {', '.join(assessment_display)}")
    if interview_rate >= 0.75:
        platform_points.append(f"reliable interview attendance ({interview_rate:.0%} completion rate)")

    if platform_points:
        if len(platform_points) >= 2:
            platform_phrase = f"On Redrob, they show {platform_points[0]} and {platform_points[1]}."
        else:
            platform_phrase = f"On Redrob, they show {platform_points[0]}."
    else:
        platform_phrase = ""

    # Education note
    edu_phrase = ""
    if facts.get("edu_tier") == "tier_1" and facts.get("edu_institution"):
        edu_phrase = f"Education from {facts['edu_institution']} (tier-1 institution)."

    # Availability
    avail_parts = []
    if open_to_work:
        if notice <= 30:
            avail_parts.append(f"immediately available with a {notice}-day notice")
        elif notice <= 60:
            avail_parts.append(f"actively looking with a {notice}-day notice period")
        else:
            avail_parts.append(f"open to work with a longer {notice}-day notice period")
    else:
        if notice <= 60:
            avail_parts.append(f"available within a {notice}-day notice period")
        else:
            avail_parts.append(f"a longer {notice}-day notice period")

    if location_note:
        avail_parts.append(location_note)

    if len(avail_parts) == 2:
        avail_phrase = f"Availability: {avail_parts[0]} and {avail_parts[1]}."
    elif avail_parts:
        avail_phrase = f"Availability: {avail_parts[0]}."
    else:
        avail_phrase = ""

    # Concerns
    concerns = []
    if title_fit < 0.4:
        concerns.append(f"current title ({current_title}) is not closely aligned with search/ranking engineering")
    if is_consultancy_only(candidate):
        concerns.append("consultancy-only career history which may lack product startup speed")
    if not matched_vector_dbs:
        concerns.append("limited explicit vector database production experience")
    if not matched_eval:
        concerns.append("gaps in explicit offline/online evaluation metric design")
    if is_researcher_profile(candidate) and ship_score <= 0.2:
        concerns.append("research-heavy background with sparse shipping evidence")
    if not is_recent_coder(candidate):
        concerns.append("limited evidence of recent hands-on coding activity")
    if notice > 75:
        concerns.append(f"a long notice period ({notice} days)")
    if has_non_compete(candidate):
        concerns.append("potential legal risk from a non-compete clause")
    if is_cv_speech_domain(candidate):
        concerns.append("primary domain in CV/speech/robotics, misaligned with search/ranking")
    if is_job_hopper(candidate):
        concerns.append("short average tenure across roles (job-hopping pattern)")
    if is_manager_only(candidate):
        concerns.append("management-heavy title with limited recent coding evidence")
    if is_framework_enthusiast(candidate):
        concerns.append("LLM framework familiarity without evaluation/ranking foundations")
    if is_ghost(candidate):
        concerns.append("low recruiter response rate and prolonged platform inactivity")
    if has_flight_risk(candidate):
        concerns.append("historically low offer acceptance rate (flight risk)")
    if is_culture_misfit(candidate):
        concerns.append("preference for stable/predictable environments vs. high-velocity shipping")
    if facts.get("recruiter_demand_score", 0) < 0.25 and saves < 3:
        concerns.append("low recruiter demand signals on the platform")

    if concerns:
        filtered = [c for c in concerns if not ("notice" in c and str(notice) in avail_phrase)]
        if filtered:
            concern_phrase = (
                f"Concerns: {filtered[0]} and {filtered[1]}."
                if len(filtered) >= 2
                else f"Concern: {filtered[0]}."
            )
        else:
            concern_phrase = "No major gaps identified relative to core requirements."
    else:
        concern_phrase = "No major gaps identified relative to core requirements."

    # Assemble paragraphs (deterministic order by candidate hash)
    blocks = [intro_phrase, tech_phrase]
    if platform_phrase:
        blocks.append(platform_phrase)
    if edu_phrase:
        blocks.append(edu_phrase)
    if avail_phrase:
        blocks.append(avail_phrase)
    blocks.append(concern_phrase)

    layout = rng.randint(0, 2)
    if layout == 0:
        reasoning = " ".join(blocks)
    elif layout == 1 and len(blocks) >= 4:
        reasoning = " ".join([blocks[0], blocks[2], blocks[1], blocks[-1]])
    else:
        reasoning = " ".join([blocks[0], blocks[1], blocks[-2] if len(blocks) > 3 else blocks[-1], blocks[-1]])

    reasoning = " ".join(reasoning.split())
    return reasoning, template_type
