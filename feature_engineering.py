import re
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    SERVICE_COMPANIES, MUST_HAVE_CONCEPTS, VECTOR_DBS, CORE_LANG, JD_TEXT,
    EVALUATION_TERMS, HYBRID_SEARCH_TERMS, BAD_DOMAINS, RESEARCH_SIGNAL_TERMS,
    CLOSED_SYSTEM_TERMS, FICTIONAL_COMPANIES, PRIMARY_CITY_PREFERENCE,
    SECONDARY_CITY_PREFERENCE
)

_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
_jd_vector = _vectorizer.fit_transform([JD_TEXT])

SHIPPING_KEYWORDS = {"shipped", "deployed", "launched", "production", "scaled", "rolled out", "released", "delivered"}
TECHNICAL_TITLES = {"engineer", "developer", "scientist", "architect", "programmer", "coder", "technical lead", "tech lead"}


def normalize_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def token_contains(text: str, term: str) -> bool:
    regex = r"\b" + re.escape(term) + r"\b"
    return re.search(regex, normalize_text(text)) is not None


def any_term_in_text(text: str, terms: set) -> bool:
    normalized = normalize_text(text)
    return any(re.search(r"\b" + re.escape(term) + r"\b", normalized) for term in terms)


def count_terms_in_text(text: str, terms: set) -> int:
    normalized = normalize_text(text)
    return sum(1 for term in terms if re.search(r"\b" + re.escape(term) + r"\b", normalized))


def extract_facts(candidate: dict) -> dict:
    profile = candidate["profile"]
    history = candidate.get("career_history", [])
    skills = [s["name"].lower() for s in candidate.get("skills", [])]

    matched_skills = [s.title() for s in skills if any(concept in s for concept in MUST_HAVE_CONCEPTS.union(VECTOR_DBS).union(CORE_LANG))]
    top_skills = matched_skills[:4] if matched_skills else [s.title() for s in skills[:4]]

    has_product_experience = any(job["company"].lower() not in SERVICE_COMPANIES for job in history)
    best_product_company = next((job["company"] for job in history if job["company"].lower() not in SERVICE_COMPANIES), profile.get("current_company", "Unknown"))

    return {
        "yoe": profile.get("years_of_experience", 0),
        "current_title": profile.get("current_title", "Unknown"),
        "current_company": profile.get("current_company", "Unknown"),
        "top_skills": top_skills,
        "has_product_experience": has_product_experience,
        "best_product_company": best_product_company,
        "notice_period": candidate["redrob_signals"].get("notice_period_days", 90),
        "willing_to_relocate": candidate["redrob_signals"].get("willing_to_relocate", False),
        "location": profile.get("location", "Unknown"),
        "has_vector_db_experience": has_vector_db_experience(candidate),
        "has_evaluation_experience": has_evaluation_experience(candidate),
        "has_hybrid_search_experience": has_hybrid_search_experience(candidate),
        "has_pre_llm_experience": has_pre_llm_experience(candidate),
        "is_consultancy_only": is_consultancy_only(candidate),
        "is_researcher": is_researcher_profile(candidate),
        "has_closed_system_signal": has_closed_system_signal(candidate),
        "shipping_score": calculate_shipping_score(candidate),
    }


def calculate_shipping_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    score = 0.0
    current_year = datetime.now().year

    for job in history:
        desc = job.get("description", "").lower()
        end_date = job.get("end_date")
        end_year = current_year
        if end_date:
            try:
                end_year = datetime.strptime(end_date, "%Y-%m-%d").year
            except Exception:
                end_year = current_year

        years_ago = current_year - end_year
        weight = 3.0 if years_ago <= 2 else 1.5 if years_ago <= 5 else 0.7
        matches = sum(1 for kw in SHIPPING_KEYWORDS if kw in desc)
        score += matches * weight

    return min(score / 10.0, 1.0)


def calculate_completeness_score(candidate: dict) -> float:
    critical = ["years_of_experience", "current_title", "current_company", "location", "summary"]
    filled = 0
    for k in critical:
        val = candidate["profile"].get(k)
        if val is not None and (not isinstance(val, str) or val.strip() != ""):
            filled += 1
    return filled / len(critical)


def calculate_ai_evidence(candidate: dict) -> float:
    text = candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])])
    cand_vec = _vectorizer.transform([text])
    sim = cosine_similarity(cand_vec, _jd_vector)[0][0]
    return min(sim * 2.5, 1.0)


def calculate_availability_score(candidate: dict) -> float:
    signals = candidate["redrob_signals"]
    open_flag = float(signals.get("open_to_work_flag", False))
    response_rate = signals.get("recruiter_response_rate", 0.0)
    notice = min(signals.get("notice_period_days", 90), 180)
    notice_score = max(0.0, 1.0 - (notice / 180.0))
    return (open_flag * 0.3) + (response_rate * 0.3) + (notice_score * 0.4)


def calculate_location_score(candidate: dict) -> float:
    location = candidate["profile"].get("location", "").lower()
    normalized = normalize_text(location)
    if any(city in normalized for city in PRIMARY_CITY_PREFERENCE):
        return 1.0
    if any(city in normalized for city in SECONDARY_CITY_PREFERENCE):
        return 0.85
    if candidate["redrob_signals"].get("willing_to_relocate", False):
        return 0.75
    return 0.55


def has_vector_db_experience(candidate: dict) -> bool:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]) + " " + " ".join([s.get("name", "") for s in candidate.get("skills", [])]))
    return any_term_in_text(text, VECTOR_DBS)


def has_hybrid_search_experience(candidate: dict) -> bool:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    return any_term_in_text(text, HYBRID_SEARCH_TERMS)


def has_evaluation_experience(candidate: dict) -> bool:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    return any_term_in_text(text, EVALUATION_TERMS)


def is_researcher_profile(candidate: dict) -> bool:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    return any_term_in_text(text, RESEARCH_SIGNAL_TERMS)


def has_closed_system_signal(candidate: dict) -> bool:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    return any_term_in_text(text, CLOSED_SYSTEM_TERMS)


def is_fictional_company_history(candidate: dict) -> bool:
    text = normalize_text(" ".join(job.get("company", "") for job in candidate.get("career_history", [])) + " " + candidate["profile"].get("current_company", ""))
    return any_term_in_text(text, FICTIONAL_COMPANIES)


def calculate_ai_yoe(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    ai_months = 0
    core_kws = set(MUST_HAVE_CONCEPTS).union(VECTOR_DBS).union(HYBRID_SEARCH_TERMS)
    for job in history:
        title = normalize_text(job.get("title", ""))
        desc = normalize_text(job.get("description", ""))
        if any(kw in title or kw in desc for kw in core_kws):
            ai_months += job.get("duration_months", 0)
    return ai_months / 12.0


def has_pre_llm_experience(candidate: dict) -> bool:
    history = candidate.get("career_history", [])
    core_kws = set(MUST_HAVE_CONCEPTS).union(VECTOR_DBS).union(HYBRID_SEARCH_TERMS)
    for job in history:
        s_str = job.get("start_date")
        if not s_str:
            continue
        try:
            s_dt = datetime.strptime(s_str, "%Y-%m-%d")
        except Exception:
            continue
        if s_dt.year < 2022:
            title = normalize_text(job.get("title", ""))
            desc = normalize_text(job.get("description", ""))
            if any(kw in title or kw in desc for kw in core_kws):
                return True
    return False


def is_recent_coder(candidate: dict) -> bool:
    if candidate["redrob_signals"].get("github_activity_score", -1) > 0:
        return True

    history = candidate.get("career_history", [])
    coder_evidence = False
    for job in history:
        end_date = job.get("end_date")
        recent = False
        if not end_date:
            recent = True
        else:
            try:
                e_dt = datetime.strptime(end_date, "%Y-%m-%d")
                recent = e_dt >= datetime.now().replace(year=datetime.now().year - 1)
            except Exception:
                pass
        if not recent:
            continue
        title = normalize_text(job.get("title", ""))
        desc = normalize_text(job.get("description", ""))
        coding_kws = {"python", "code", "develop", "build", "deploy", "shipping", "shipped", "engineer"}
        if any(word in title for word in TECHNICAL_TITLES) or any(kw in desc for kw in coding_kws):
            coder_evidence = True
    return coder_evidence


def calculate_stability_score(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if not history:
        return 1.0
    durations = [job.get("duration_months", 0) for job in history if job.get("duration_months", 0) > 0]
    if not durations:
        return 0.4
    avg_years = (sum(durations) / len(durations)) / 12.0
    if avg_years >= 3.0:
        return 1.0
    if avg_years >= 1.5:
        return 0.7 + 0.3 * (avg_years - 1.5) / 1.5
    return max(0.3, avg_years / 1.5)


def is_consultancy_only(candidate: dict) -> bool:
    history = candidate.get("career_history", [])
    if not history:
        return True
    return all(any(service in job.get("company", "").lower() for service in SERVICE_COMPANIES) for job in history)


def is_honeypot(candidate: dict) -> bool:
    if is_fictional_company_history(candidate):
        return True

    skills = candidate.get("skills", [])
    exp_zero = sum(1 for s in skills if s.get("proficiency") in {"expert", "advanced"} and s.get("duration_months", 0) == 0)
    if exp_zero >= 1:
        return True

    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)
    total_months = sum(job.get("duration_months", 0) for job in candidate.get("career_history", []))
    sum_yoe = total_months / 12.0
    if abs(yoe - sum_yoe) >= 1.0:
        return True

    summary = normalize_text(profile.get("summary", ""))
    if any_term_in_text(summary, MUST_HAVE_CONCEPTS.union(VECTOR_DBS)) and not any_term_in_text(summary, {"engineer", "developer", "scientist", "architect", "programmer"}):
        return True

    return False


def calculate_retrieval_score(candidate: dict) -> float:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    retrieval_hits = count_terms_in_text(text, MUST_HAVE_CONCEPTS.union(HYBRID_SEARCH_TERMS))
    vector_hits = count_terms_in_text(text, VECTOR_DBS)
    shipping = calculate_shipping_score(candidate)
    score = min(0.4 + min(retrieval_hits, 4) * 0.12 + min(vector_hits, 2) * 0.15 + shipping * 0.25, 1.0)
    return score


def calculate_evaluation_score(candidate: dict) -> float:
    text = normalize_text(candidate["profile"].get("summary", "") + " " + " ".join([job.get("description", "") for job in candidate.get("career_history", [])]))
    hits = count_terms_in_text(text, EVALUATION_TERMS)
    return min(0.2 + min(hits, 4) * 0.2, 1.0)


def calculate_experience_fit(candidate: dict) -> float:
    profile = candidate["profile"]
    yoe = profile.get("years_of_experience", 0)
    ai_yoe = calculate_ai_yoe(candidate)
    ratio = ai_yoe / yoe if yoe > 0 else 0.0
    if 5.0 <= yoe <= 9.0:
        base = 1.0
    elif yoe < 5.0:
        base = max(0.3, yoe / 5.0)
    else:
        base = max(0.5, 1.0 - (yoe - 9.0) / 15.0)
    if yoe > 9.0 and ratio < 0.5:
        base *= 0.75
    return base


def calculate_ai_yoe_score(candidate: dict) -> float:
    ai_yoe = calculate_ai_yoe(candidate)
    if ai_yoe >= 5.0:
        return 1.0
    if ai_yoe >= 3.0:
        return 0.7 + 0.3 * (ai_yoe - 3.0) / 2.0
    if ai_yoe >= 1.0:
        return 0.3 + 0.4 * (ai_yoe - 1.0) / 2.0
    return 0.2


def calculate_research_penalty(candidate: dict) -> float:
    if is_researcher_profile(candidate) and not calculate_shipping_score(candidate) > 0.2:
        return 0.6
    return 1.0


def calculate_title_chaser_penalty(candidate: dict) -> float:
    history = candidate.get("career_history", [])
    if len(history) < 3:
        return 1.0
    senior_titles = sum(1 for job in history if any(t in normalize_text(job.get("title", "")) for t in ["director", "vp", "head", "architect", "senior"]))
    average_duration = sum(job.get("duration_months", 0) for job in history) / len(history)
    if senior_titles >= 2 and average_duration < 18:
        return 0.75
    return 1.0


def calculate_production_fit(candidate: dict) -> float:
    score = 0.0
    if has_vector_db_experience(candidate):
        score += 0.3
    if has_hybrid_search_experience(candidate):
        score += 0.25
    if has_evaluation_experience(candidate):
        score += 0.2
    if calculate_shipping_score(candidate) > 0.3:
        score += 0.25
    return min(score, 1.0)
