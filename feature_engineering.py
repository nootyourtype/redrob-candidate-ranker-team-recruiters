# feature_engineering.py - All feature scorers and dealbreaker detectors for V3.
import math
from datetime import datetime
from config import (
    REFERENCE_DATE, CUTOFF_PRE_LLM, TIER1_LOCATIONS, TIER2_LOCATIONS,
    SALARY_BAND_MIN, SALARY_BAND_MAX, SHIPPED_RETRIEVAL, PRE_LLM_RETRIEVAL,
    PRODUCTION_SIGNALS, CODING_SIGNALS, PYTHON_SIGNALS, EVAL_SIGNALS,
    SYSTEM_SIGNALS, AI_TERMS, VIDEO_PRIMARY, CV_SPEECH_ROBOTICS,
    CONSULTING_FIRMS, CONSULTING_INDUSTRIES, JD_VECTOR_DB, JD_RETRIEVAL_EVAL,
    JD_LLM_FT, JD_HR_DOMAIN, CULTURE_NEGATIVE, CULTURE_POSITIVE,
    NON_COMPETE_TERMS, MANAGER_TITLES, SENIOR_IC_TITLES, FICTIONAL_COMPANIES,
    GHOST_INACTIVE_DAYS, GHOST_RRR_THRESHOLD,
)

TODAY = REFERENCE_DATE.date()

def lo(s): return str(s).lower() if s else ""
def hits(text, terms): t = lo(text); return sum(1 for term in terms if term in t)
def any_hit(text, terms): t = lo(text); return any(term in t for term in terms)
def days_ago(d):
    if not d: return 9999
    try: return (TODAY - datetime.strptime(str(d)[:10], "%Y-%m-%d").date()).days
    except: return 9999
def clamp(v, lo_=0.0, hi_=1.0): return max(lo_, min(hi_, float(v)))
def parse_date(s):
    if not s: return None
    try: return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except: return None

def career_yoe(career):
    return sum(j.get("duration_months", 0) for j in career) / 12.0

def career_text(career):
    return lo(" ".join(j.get("description", "") for j in career))

def skill_text(skills):
    return lo(" ".join(s.get("name", "") for s in skills))


# ── POSITIVE FEATURE SCORERS ──────────────────────────────────────────────────

def calculate_jd_alignment(career, skills_list):
    desc = career_text(career)
    sk   = skill_text(skills_list)
    def cat(terms, threshold):
        return clamp((hits(desc, terms) * 3 + hits(sk, terms)) / threshold)
    vector_db = cat(JD_VECTOR_DB, 9.0)
    retrieval  = cat(JD_RETRIEVAL_EVAL, 9.0)
    llm_ft     = cat(JD_LLM_FT, 6.0)
    hr_domain  = cat(JD_HR_DOMAIN, 6.0)
    must_have  = clamp(0.5 * vector_db + 0.5 * retrieval)
    nice_have  = clamp(0.6 * llm_ft + 0.4 * hr_domain)
    score      = clamp(0.70 * must_have + 0.30 * nice_have)
    detail     = {"vector_db": round(vector_db,2), "retrieval": round(retrieval,2),
                  "llm_ft": round(llm_ft,2), "hr_domain": round(hr_domain,2)}
    return score, detail


def calculate_technical(career, skills_list):
    desc = career_text(career)
    sk   = skill_text(skills_list)
    python_sc = clamp(hits(desc, PYTHON_SIGNALS)/5.0)*0.7 + clamp(hits(sk, PYTHON_SIGNALS)/3.0)*0.3
    eval_sc   = clamp(hits(desc, EVAL_SIGNALS)/4.0)
    system_sc = clamp(hits(desc, SYSTEM_SIGNALS)/5.0)
    current = [j for j in career if j.get("is_current")]
    recent  = sorted(career, key=lambda j: j.get("start_date","") or "", reverse=True)[:2]
    check   = current if current else recent
    code_sc = 0.3
    for job in check:
        d    = lo(job.get("description",""))
        t    = lo(job.get("title",""))
        ch   = hits(d, CODING_SIGNALS)
        mgmt = any(x in t for x in ["vp ","vice president","chief","cto","head of","director","manager"])
        rs   = 0.20 if (mgmt and ch < 2) else clamp(ch/3.0)
        end  = job.get("end_date")
        cur  = job.get("is_current", False)
        rec  = 0 if cur else (days_ago(end)/30.44 if end else 36)
        mult = 1.0 if rec<=18 else (0.70 if rec<=36 else 0.40)
        code_sc = max(code_sc, rs*mult)
    opt = 0.0
    if any_hit(desc, ["fine-tuning","fine tuning","lora","qlora","peft","rlhf"]): opt += 0.05
    if any_hit(desc, ["open-source","open source","contributed to","open-sourced"]): opt += 0.03
    return clamp(0.35*eval_sc + 0.30*python_sc + 0.20*system_sc + 0.15*code_sc + opt)


def _ranking_yoe_score(career):
    kw = ["ranking","retrieval","search","recommendation","relevance","discovery","information retrieval"]
    mo = sum(j.get("duration_months",0) for j in career
             if any(k in lo(j.get("title","")) for k in kw)
             or (hits(lo(j.get("description","")), SHIPPED_RETRIEVAL)>=2
                 and hits(lo(j.get("description","")), PRODUCTION_SIGNALS)>=1))
    return clamp(mo/48.0)


def calculate_production(profile, career):
    shipped_sc  = 0.0
    pre_llm_bon = 0.0
    for job in career:
        jd   = lo(job.get("description",""))
        dur  = job.get("duration_months",0) or 0
        start= parse_date(job.get("start_date"))
        rh   = hits(jd, SHIPPED_RETRIEVAL)
        ph   = hits(jd, PRODUCTION_SIGNALS)
        if rh>=2 and ph>=1:
            dw  = clamp(dur/18.0)
            rs  = clamp((rh/6.0)*0.7 + (ph/4.0)*0.3)
            shipped_sc = max(shipped_sc, rs*dw)
            if start and start < CUTOFF_PRE_LLM:
                ph2 = hits(jd, PRE_LLM_RETRIEVAL)
                if ph2>=1: pre_llm_bon = max(pre_llm_bon, clamp(ph2/4.0)*0.25)
        elif rh>=2:
            shipped_sc = max(shipped_sc, clamp(rh/8.0)*0.5*clamp(dur/18.0))
    yoe      = career_yoe(career)
    total_mo = max(sum(j.get("duration_months",0) for j in career), 1)
    ai_mo    = sum(j.get("duration_months",0) for j in career
                   if hits(lo(j.get("description","")), AI_TERMS)>=2
                   or hits(lo(j.get("title","")),
                           ["ml","machine learning","ai","data science","nlp",
                            "recommendation","search","ranking","retrieval"])>=1)
    ai_ratio = ai_mo/total_mo
    if 5<=yoe<=9:    yoe_sc=1.0
    elif 4<=yoe<5:   yoe_sc=0.75
    elif 9<yoe<=12:  yoe_sc=0.80
    elif 3<=yoe<4:   yoe_sc=0.50
    elif yoe>12:     yoe_sc=0.55
    else:            yoe_sc=0.25
    if ai_ratio>=0.55:   ratio_sc=1.0
    elif ai_ratio>=0.40: ratio_sc=0.80
    elif ai_ratio>=0.25: ratio_sc=0.55
    elif ai_ratio>=0.15: ratio_sc=0.35
    else:                ratio_sc=0.10
    antipattern = 0.50 if (yoe>12 and ai_ratio<0.20) else 1.0
    ai_sc = clamp((0.40*yoe_sc + 0.60*ratio_sc + pre_llm_bon)*antipattern)
    return clamp(0.55 * shipped_sc + 0.45 * ai_sc)


def calculate_availability(profile, signals):
    last    = days_ago(signals.get("last_active_date"))
    recency = math.exp(-last/180.0)
    otw     = 1.0 if signals.get("open_to_work_flag") else 0.0
    rrr     = signals.get("recruiter_response_rate",0) or 0
    plat_sc = clamp(0.12*otw + 0.53*recency + 0.35*rrr)
    notice  = signals.get("notice_period_days",60) or 60
    notice_sc = 1.0 if notice<=60 else (0.95 if notice<=90 else 0.90)
    sal     = signals.get("expected_salary_range_inr_lpa",{}) or {}
    sal_min = sal.get("min",0) or 0
    sal_max = sal.get("max",0) or 0
    sal_chk = sal_max if sal_max>0 else sal_min*1.5
    if SALARY_BAND_MIN<=sal_chk<=SALARY_BAND_MAX: sal_sc=1.0
    elif sal_chk<SALARY_BAND_MIN and sal_chk>0:   sal_sc=0.70
    elif sal_chk>SALARY_BAND_MAX*1.4:             sal_sc=0.15
    elif sal_chk>SALARY_BAND_MAX:                 sal_sc=0.45
    else:                                          sal_sc=0.50
    loc     = lo(profile.get("location","") + " " + profile.get("country",""))
    willing = bool(signals.get("willing_to_relocate",False))
    if any(t in loc for t in TIER1_LOCATIONS):    loc_sc=1.0
    elif any(t in loc for t in TIER2_LOCATIONS):  loc_sc=0.90 if willing else 0.70
    elif "india" in loc:                           loc_sc=0.75 if willing else 0.55
    elif willing:                                  loc_sc=0.55
    else:                                          loc_sc=0.10
    if "india" not in loc and not willing:    visa_mult=0.60
    elif "india" not in loc and willing:      visa_mult=0.85
    else:                                     visa_mult=1.0
    logistics = clamp((0.45*notice_sc + 0.35*loc_sc + 0.20*sal_sc)*visa_mult)
    return clamp(0.50*plat_sc + 0.50*logistics)


def calculate_behavior(signals):
    rrr     = signals.get("recruiter_response_rate",0) or 0
    icr     = signals.get("interview_completion_rate",0.5) or 0.5
    oar_raw = signals.get("offer_acceptance_rate",-1)
    oar     = clamp(oar_raw) if oar_raw>=0 else 0.5
    resp_h  = signals.get("avg_response_time_hours",999) or 999
    gh_raw  = signals.get("github_activity_score",-1)
    github  = clamp(gh_raw/100.0) if gh_raw>=0 else 0.25
    apps_30 = min(signals.get("applications_submitted_30d",0) or 0, 10)/10.0
    saved   = min(signals.get("saved_by_recruiters_30d",0) or 0, 10)/10.0
    # Also keep demand composite for facts/reasoning but use original V3 weights
    saves   = signals.get("saved_by_recruiters_30d",0) or 0
    searches= signals.get("search_appearance_30d",0) or 0
    views   = signals.get("profile_views_received_30d",0) or 0
    if resp_h<=12:    resp_sc=1.0
    elif resp_h<=24:  resp_sc=0.85
    elif resp_h<=72:  resp_sc=0.60
    elif resp_h<=168: resp_sc=0.35
    else:             resp_sc=0.10
    raw   = clamp(0.28*rrr + 0.22*resp_sc + 0.20*icr + 0.10*oar + 0.10*github + 0.05*apps_30 + 0.05*saved)
    last  = days_ago(signals.get("last_active_date"))
    stale = last>180
    conf  = 0.85 if stale else 1.0
    ghost_flag = (not signals.get("open_to_work_flag") and last>GHOST_INACTIVE_DAYS and rrr<GHOST_RRR_THRESHOLD)
    ghost_m = 0.45 if ghost_flag else 1.0
    return round(raw*conf*ghost_m, 3), stale


def calculate_env_fit(career):
    total_mo = max(sum(j.get("duration_months",0) for j in career), 1)
    svc_mo   = sum(j.get("duration_months",0) for j in career
                   if any(f in lo(j.get("company","")) for f in CONSULTING_FIRMS)
                   or any(i in lo(j.get("industry","")) for i in CONSULTING_INDUSTRIES))
    product_ratio = (total_mo-svc_mo)/total_mo
    SIZE_MAP = {"1-10":1.0,"11-50":0.90,"51-200":0.75,"201-500":0.60,
                "501-1000":0.45,"1001-5000":0.30,"5001-10000":0.20,"10001+":0.15}
    recent    = sorted(career, key=lambda j: j.get("start_date","") or "", reverse=True)[:2]
    startup_sc= sum(SIZE_MAP.get(j.get("company_size",""),0.40) for j in recent)/max(len(recent),1)
    desc      = career_text(career)
    cv_hits   = hits(desc, CV_SPEECH_ROBOTICS)
    vid_hits  = hits(desc, VIDEO_PRIMARY)
    ret_hits  = hits(desc, SHIPPED_RETRIEVAL)
    if (cv_hits>=4 and ret_hits<=2) or (vid_hits>=2 and ret_hits<=1): domain_p=0.30
    elif cv_hits>=2 and ret_hits<=1: domain_p=0.55
    else: domain_p=1.0
    return clamp((0.55*product_ratio + 0.35*startup_sc + 0.10)*domain_p)


def calculate_stability(career):
    roles  = [j for j in career if (j.get("duration_months") or 0)>3]
    if not roles: return 0.5
    short  = sum(1 for j in roles if (j.get("duration_months") or 0)<18)
    long_  = sum(1 for j in roles if (j.get("duration_months") or 0)>=24)
    total  = len(roles)
    short_r= short/total
    def seniority(t):
        t=lo(t)
        if any(w in t for w in ["vp","director","head of","chief"]): return 5
        if any(w in t for w in ["principal","staff","distinguished"]): return 4
        if "lead" in t: return 3
        if "senior" in t or "sr" in t: return 2
        if "junior" in t or "jr" in t: return 0
        return 1
    levels = [seniority(j.get("title","")) for j in roles]
    rapid  = False
    if len(levels)>=3:
        jumps = sum(1 for i in range(len(levels)-1) if levels[i]<levels[i+1])
        rapid = jumps>=3 and short_r>=0.5
    if rapid:            return 0.30
    elif short_r>=0.70:  return 0.45
    elif short_r>=0.50:  return 0.65
    elif long_>=2:       return 1.0
    elif long_>=1:       return 0.85
    else:                return 0.70


def calculate_bonuses(career, skills_list, signals, profile):
    desc    = career_text(career)
    summary = lo(profile.get("summary",""))
    gh_raw  = signals.get("github_activity_score",-1)
    gh_bon  = clamp(gh_raw/100.0)*0.015 if gh_raw>=0 else 0.0
    oss_m   = hits(desc, ["open-source","open source","contributed to","open-sourced",
                           "published a paper","conference talk"])
    oss_bon = clamp(oss_m/2.0)*0.01
    assess  = signals.get("skill_assessment_scores",{}) or {}
    rel     = [v for k,v in assess.items()
               if any(t in lo(k) for t in ["python","ml","nlp","retrieval","machine learning",
                                            "deep learning","llm","ranking","search","vector"])]
    assess_bon = clamp((sum(rel)/len(rel)-60)/40.0)*0.04 if rel and max(rel)>60 else 0.0
    culture_bon= 0.03 if any_hit(summary, CULTURE_POSITIVE) else 0.0
    return round(gh_bon+oss_bon+assess_bon+culture_bon, 4)


# ── DEALBREAKER DETECTORS ─────────────────────────────────────────────────────

def is_honeypot(c):
    profile = c.get("profile",{})
    career  = c.get("career_history",[])
    skills  = c.get("skills",[])
    summary = lo(profile.get("summary",""))
    companies = [lo(j.get("company","")) for j in career] + [lo(profile.get("current_company",""))]
    if any(fc in companies for fc in FICTIONAL_COMPANIES): return True
    claimed  = profile.get("years_of_experience",0) or 0
    real_yoe = career_yoe(career)
    if claimed>0 and real_yoe>0.1 and abs(claimed-real_yoe)>=5.0: return True
    zero_exp = sum(1 for s in skills
                   if (s.get("duration_months") or 0)==0
                   and s.get("proficiency") in ("advanced","expert"))
    if zero_exp>=4: return True
    all_kw   = set(JD_VECTOR_DB)|set(JD_RETRIEVAL_EVAL)|{"llm","rag","embedding","gpt","transformer"}
    tech_titles = {"engineer","developer","scientist","architect","programmer"}
    if (sum(1 for kw in all_kw if kw in summary)>=6 and
            not any(t in summary for t in tech_titles)): return True
    return False


def calculate_trap_multiplier(profile, career, skills_list):
    desc      = career_text(career)
    skill_txt = skill_text(skills_list)
    headline  = lo(profile.get("headline",""))
    summary   = lo(profile.get("summary",""))
    all_text  = desc+" "+skill_txt+" "+summary
    ai_claimed= sum(1 for kw in [
        "llm","rag","vector database","pinecone","embedding","langchain",
        "gpt","openai","claude","gemini","llama","mistral","huggingface",
        "transformer","fine-tuning","bert","semantic search","faiss","weaviate",
        "qdrant","milvus","chatgpt","generative ai","genai",
    ] if kw in skill_txt+" "+headline+" "+summary)
    ai_evidence = hits(desc, SHIPPED_RETRIEVAL+PRODUCTION_SIGNALS+AI_TERMS)
    penalty = 1.0
    if ai_claimed>=8 and ai_evidence<=2:   penalty*=0.20
    elif ai_claimed>=5 and ai_evidence<=3: penalty*=0.45
    zero_exp = sum(1 for s in skills_list
                   if (s.get("duration_months") or 0)==0
                   and s.get("proficiency") in ("advanced","expert"))
    if zero_exp>=4:   penalty*=0.25
    elif zero_exp>=2: penalty*=0.55
    elif zero_exp>=1: penalty*=0.80
    non_tech = ["marketing","sales","business development","bd ","account manager",
                "product manager","program manager","scrum master","operations","hr ","recruiter"]
    curr_title = lo(profile.get("current_title",""))
    if any(t in curr_title for t in non_tech) and ai_claimed>=4: penalty*=0.10
    claimed = profile.get("years_of_experience",0) or 0
    real    = career_yoe(career)
    if claimed>0 and real>0.1:
        ratio = claimed/real
        if ratio>5.0:   penalty*=0.15
        elif ratio>2.0: penalty*=0.40
        elif ratio>1.3: penalty*=0.70
    if len(career)>=2:
        total_mo   = max(sum(j.get("duration_months",0) for j in career),1)
        seen       = {}
        suspect_mo = 0
        for j in career:
            d   = lo(j.get("description",""))[:200].strip()
            dur = j.get("duration_months",0) or 0
            if not d: continue
            if d in seen: suspect_mo+=min(dur,seen[d])
            else: seen[d]=dur
        dup_frac = min(suspect_mo/total_mo,1.0)
        if dup_frac>=0.50:   penalty*=0.25
        elif dup_frac>=0.30: penalty*=0.55
        elif dup_frac>=0.10: penalty*=0.85
    if any_hit(summary, CULTURE_NEGATIVE):  penalty*=0.15
    if any_hit(all_text, NON_COMPETE_TERMS): penalty*=0.10
    is_mgmt = (any(t in curr_title for t in MANAGER_TITLES) and
               not any(t in curr_title for t in SENIOR_IC_TITLES))
    if is_mgmt:
        recent_coder = False
        for j in sorted(career, key=lambda x: x.get("start_date","") or "", reverse=True)[:2]:
            d   = lo(j.get("description",""))
            cur = j.get("is_current",False)
            end = j.get("end_date")
            mo  = 0 if cur else (days_ago(end)/30.44 if end else 36)
            if hits(d, CODING_SIGNALS)>=2 and mo<=18:
                recent_coder=True; break
        if not recent_coder: penalty*=0.15
    return clamp(penalty,0.05,1.0)


def calculate_must_have(career, skills_list):
    desc = career_text(career)
    sk   = skill_text(skills_list)
    all_text = desc+" "+sk
    has_retrieval = any_hit(all_text, [
        "retrieval","vector search","semantic search","faiss","elasticsearch",
        "opensearch","pinecone","weaviate","qdrant","milvus","ranking system",
        "ranking","information retrieval","embedding","recommendation",
    ])
    has_python = any_hit(all_text, [
        "python","pytorch","tensorflow","pyspark","pandas","numpy",
        "sklearn","scikit-learn","fastapi","flask","pydantic",
    ])
    return 1.0 if (has_retrieval and has_python) else 0.40


def calculate_eval_absence_penalty(career, signals):
    desc = career_text(career)
    gh   = signals.get("github_activity_score",-1)
    if gh>=70 and hits(desc, EVAL_SIGNALS)==0: return 0.75
    return 1.0


def calculate_flight_risk(signals):
    oar  = signals.get("offer_acceptance_rate",-1)
    apps = signals.get("applications_submitted_30d",0) or 0
    if oar==0.0 and apps>=3: return 0.55
    return 1.0


# ── FACTS EXTRACTOR ───────────────────────────────────────────────────────────

def extract_facts(c):
    profile  = c.get("profile",{})
    career   = c.get("career_history",[])
    skills   = c.get("skills",[])
    signals  = c.get("redrob_signals",{})
    all_text = career_text(career)+" "+skill_text(skills)
    top_skills = [s["name"] for s in sorted(skills,key=lambda s:s.get("endorsements",0),reverse=True)[:5]]
    product_cos= [j.get("company","") for j in career
                  if not any(f in lo(j.get("company","")) for f in CONSULTING_FIRMS)
                  and not any(i in lo(j.get("industry","")) for i in CONSULTING_INDUSTRIES)]
    edu_list = c.get("education",[])
    _, jd_detail = calculate_jd_alignment(career, skills)
    return {
        "yoe":                   round(career_yoe(career),1),
        "yoe_claimed":           profile.get("years_of_experience",0) or 0,
        "current_title":         profile.get("current_title","Engineer"),
        "current_company":       profile.get("current_company",""),
        "best_product_company":  product_cos[0] if product_cos else profile.get("current_company",""),
        "location":              profile.get("location",""),
        "country":               profile.get("country",""),
        "top_skills":            top_skills,
        "edu_tier":              edu_list[0].get("tier","") if edu_list else "",
        "edu_institution":       edu_list[0].get("institution","") if edu_list else "",
        "jd_detail":             jd_detail,
        "has_eval":              hits(all_text, EVAL_SIGNALS)>0,
        "has_vector_db":         any_hit(all_text, JD_VECTOR_DB),
        "saved_by_recruiters_30d":   signals.get("saved_by_recruiters_30d",0) or 0,
        "search_appearance_30d":     signals.get("search_appearance_30d",0) or 0,
        "interview_completion_rate": signals.get("interview_completion_rate",0.0) or 0.0,
        "skill_assessments":         signals.get("skill_assessment_scores",{}) or {},
        "notice_period_days":        signals.get("notice_period_days",60) or 60,
        "open_to_work":              bool(signals.get("open_to_work_flag",False)),
        "willing_to_relocate":       bool(signals.get("willing_to_relocate",False)),
    }
