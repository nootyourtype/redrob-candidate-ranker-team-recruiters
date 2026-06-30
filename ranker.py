# ranker.py - V3 Scoring Engine & Pairwise Re-Ranking
from collections import namedtuple
from feature_engineering import (
    career_yoe, lo, days_ago, clamp, extract_facts, is_honeypot,
    calculate_jd_alignment, calculate_technical, calculate_production,
    calculate_availability, calculate_behavior, calculate_env_fit,
    calculate_stability, calculate_bonuses, calculate_trap_multiplier,
    calculate_must_have, calculate_eval_absence_penalty, calculate_flight_risk,
)
from config import (
    DIMENSION_WEIGHTS, UNREACH_CAP, VISA_CAP, INTEGRITY_CAP,
    UNREACH_INACTIVE_DAYS, UNREACH_RRR_THRESHOLD,
)

CLOSE_CALL_THRESHOLD = 0.06
MAX_MOVE = 4
SwapLog = namedtuple("SwapLog", ["rank_a","rank_b","cid_a","cid_b","reason"])


def score_candidate(c):
    profile  = c.get("profile",{})
    career   = c.get("career_history",[])
    skills_l = c.get("skills",[])
    signals  = c.get("redrob_signals",{})

    jd_sc, jd_detail = calculate_jd_alignment(career, skills_l)
    tech_sc           = calculate_technical(career, skills_l)
    prod_sc           = calculate_production(profile, career)
    avail_sc          = calculate_availability(profile, signals)
    behav_sc, is_stale= calculate_behavior(signals)
    env_sc            = calculate_env_fit(career)
    stab_sc           = calculate_stability(career)
    bonuses           = calculate_bonuses(career, skills_l, signals, profile)
    trap_mult         = calculate_trap_multiplier(profile, career, skills_l)
    must_mult         = calculate_must_have(career, skills_l)
    eval_pen          = calculate_eval_absence_penalty(career, signals)
    flight_pen        = calculate_flight_risk(signals)

    base = sum(DIMENSION_WEIGHTS[k]*v for k,v in {
        "jd_alignment":jd_sc,"technical":tech_sc,"production":prod_sc,
        "availability":avail_sc,"behavior":behav_sc,
    }.items())
    env_mult  = 0.80 + 0.20*env_sc
    stab_mult = 0.90 + 0.10*stab_sc
    raw = clamp((base+bonuses)*env_mult*stab_mult*trap_mult*must_mult*eval_pen*flight_pen)

    rrr_chk  = signals.get("recruiter_response_rate",0) or 0
    last_chk = days_ago(signals.get("last_active_date"))
    loc_chk  = lo(profile.get("location","")+" "+profile.get("country",""))
    will_chk = bool(signals.get("willing_to_relocate",False))

    unreach_cap   = UNREACH_CAP if (rrr_chk<UNREACH_RRR_THRESHOLD and last_chk>UNREACH_INACTIVE_DAYS) else 1.0
    visa_cap      = VISA_CAP if ("india" not in loc_chk and not will_chk) else 1.0
    integrity_cap = INTEGRITY_CAP if trap_mult<=0.70 else 1.0
    total = clamp(min(raw, unreach_cap, visa_cap, integrity_cap))

    return {
        "final":total,"sub_jd":round(jd_sc,3),"sub_technical":round(tech_sc,3),
        "sub_production":round(prod_sc,3),"sub_availability":round(avail_sc,3),
        "sub_behavior":round(behav_sc,3),"env_fit":round(env_sc,3),
        "stability":round(stab_sc,3),"trap_mult":round(trap_mult,3),
        "must_mult":must_mult,"eval_pen":round(eval_pen,3),"flight_pen":round(flight_pen,3),
        "bonuses":bonuses,"jd_detail":jd_detail,"is_stale":is_stale,
        "yoe":round(career_yoe(career),1),"yoe_claimed":profile.get("years_of_experience",0) or 0,
        "title":profile.get("current_title",""),"company":profile.get("current_company",""),
        "location":profile.get("location",""),"country":profile.get("country",""),
        "notice":signals.get("notice_period_days",60) or 60,
        "otw":bool(signals.get("open_to_work_flag")),
        "willing_relocate":bool(signals.get("willing_to_relocate",False)),
    }


def _pairwise_compare(feat_a, feat_b):
    a_blocked = feat_a["trap_mult"]<=0.70 or feat_a["must_mult"]<1.0
    b_blocked = feat_b["trap_mult"]<=0.70 or feat_b["must_mult"]<1.0
    if a_blocked and not b_blocked: return -1,"having no integrity flag, unlike the other candidate"
    if b_blocked and not a_blocked: return +1,"having no integrity flag, unlike the other candidate"
    prod_gap = feat_a["sub_production"]-feat_b["sub_production"]
    if abs(prod_gap)>=0.15:
        g=abs(prod_gap)
        return (+1,f"stronger shipped production evidence (+{g:.2f})") if prod_gap>0 else (-1,f"stronger shipped production evidence (+{g:.2f})")
    jd_gap = feat_a["sub_jd"]-feat_b["sub_jd"]
    if abs(jd_gap)>=0.15:
        g=abs(jd_gap)
        return (+1,f"stronger JD tool alignment (+{g:.2f})") if jd_gap>0 else (-1,f"stronger JD tool alignment (+{g:.2f})")
    notice_gap = feat_b["notice"]-feat_a["notice"]
    if abs(notice_gap)>=45:
        g=abs(notice_gap)
        return (+1,f"a {g}-day shorter notice period") if notice_gap>0 else (-1,f"a {g}-day shorter notice period")
    behav_gap = feat_a["sub_behavior"]-feat_b["sub_behavior"]
    if abs(behav_gap)>=0.20:
        g=abs(behav_gap)
        return (+1,f"better reachability (+{g:.2f})") if behav_gap>0 else (-1,f"better reachability (+{g:.2f})")
    stab_gap = feat_a["stability"]-feat_b["stability"]
    if abs(stab_gap)>=0.25:
        g=abs(stab_gap)
        return (+1,f"stronger career stability (+{g:.2f})") if stab_gap>0 else (-1,f"stronger career stability (+{g:.2f})")
    return 0,"no decisive signal — pointwise order retained"


def rerank_shortlist(ranked_results, top_n=100):
    shortlist = ranked_results[:top_n]
    rest      = ranked_results[top_n:]
    swap_log  = []
    arr       = list(shortlist)
    for i in range(1, len(arr)):
        j = i; moves = 0
        while j>0 and moves<MAX_MOVE:
            cid_hi,feat_hi = arr[j-1]
            cid_lo,feat_lo = arr[j]
            if abs(feat_hi["final"]-feat_lo["final"])>CLOSE_CALL_THRESHOLD: break
            verdict,reason = _pairwise_compare(feat_lo,feat_hi)
            if verdict==+1:
                arr[j-1],arr[j] = arr[j],arr[j-1]
                swap_log.append(SwapLog(j,j+1,cid_lo,cid_hi,reason))
                j-=1; moves+=1
            else: break
    return arr+rest, swap_log


def rank_candidates(candidates, top_n=100):
    results = []
    for c in candidates:
        try:
            cid = c.get("candidate_id","")
            if not cid or is_honeypot(c): continue
            feat = score_candidate(c)
            results.append((cid,feat,c))
        except Exception: continue
    results.sort(key=lambda x:(-x[1]["final"],x[0]))
    results_for_rerank = [(cid,feat) for cid,feat,_ in results]
    reranked, swap_log = rerank_shortlist(results_for_rerank, top_n=top_n)
    swap_note = {}
    for s in swap_log:
        swap_note[s.cid_a] = (s.reason,True)
        swap_note[s.cid_b] = (s.reason,False)
    cid_to_raw = {cid:c for cid,_,c in results}
    top_candidates = []
    for cid,feat in reranked[:top_n]:
        raw  = cid_to_raw.get(cid,{})
        facts= extract_facts(raw)
        top_candidates.append({
            "candidate_id": cid,
            "score":        feat["final"],
            "feat":         feat,
            "facts":        facts,
            "raw":          raw,
            "swap_info":    swap_note.get(cid),
        })
    return top_candidates
