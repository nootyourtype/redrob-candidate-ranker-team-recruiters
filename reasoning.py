# reasoning.py - Factual Reasoning Generator for V3
from collections import defaultdict
_tmpl = defaultdict(int)

def _article(n):
    return "an" if round(n) in {8,11,18} else "a"

def _pairwise_note(reason, swap_reason, won):
    if won:
        return f"{reason} In a close call, ranked ahead for {swap_reason}."
    return f"{reason} In a close call, a peer was ranked ahead for {swap_reason}."

def generate_reasoning(scored_candidate):
    f       = scored_candidate["feat"]
    title   = (f["title"] or "engineer").strip()
    company = f["company"] or "a product company"
    yoe, notice, otw = f["yoe"], f["notice"], f["otw"]
    prod, tech, jd   = f["sub_production"], f["sub_technical"], f["sub_jd"]
    jd_d, score      = f["jd_detail"], f["final"]

    # Availability clause
    if otw and notice<=30:    avail=f"open to work, {notice}d notice"
    elif otw:                 avail=f"open to work, {notice}d notice (plan ahead)"
    elif notice<=60:          avail=f"{notice}d notice"
    else:                     avail=f"not open-to-work, {notice}d notice (plan ahead)"

    # JD strengths
    jd_strong=[]
    if jd_d.get("vector_db",0)>=0.5:  jd_strong.append("vector DB/hybrid search")
    if jd_d.get("retrieval",0)>=0.5:  jd_strong.append("retrieval & ranking eval")
    if jd_d.get("llm_ft",0)>=0.5:     jd_strong.append("LLM fine-tuning")
    if jd_d.get("hr_domain",0)>=0.5:  jd_strong.append("HR-tech/marketplace")
    jd_str = ", ".join(jd_strong) if jd_strong else f"partial JD match ({jd:.2f})"

    bh_val = f["sub_behavior"]
    if bh_val>=0.7:   bh=f"high reliability ({bh_val:.2f})"
    elif bh_val>=0.4: bh=f"moderate reliability ({bh_val:.2f})"
    else:             bh=f"low platform reliability ({bh_val:.2f})"

    stale = " Behavioral data >180d old." if f["is_stale"] else ""

    # Cautions
    cautions=[]
    if f["trap_mult"]<=0.70:
        gap=f["yoe_claimed"]-yoe
        if gap>2:
            cautions.append(f"CAUTION: profile claims {f['yoe_claimed']:.0f}y but career "
                            f"history supports only {yoe:.0f}y — verify before proceeding.")
        elif f["trap_mult"]<=0.15:
            cautions.append("CAUTION: non-compete or culture-statement contradiction detected.")
        else:
            cautions.append("CAUTION: data-integrity flag — verify profile claims.")
    if f["eval_pen"]<1.0:
        cautions.append("Note: strong OSS but no NDCG/MRR language — eval frameworks are a major required skill.")
    if f["flight_pen"]<1.0:
        cautions.append("Note: 0% offer-acceptance with active applications — high flight risk.")
    caution_str = " ".join(cautions)

    # Gap diagnosis
    gaps=[]
    if f["trap_mult"]<=0.70: gaps.append("a data-integrity flag on profile claims")
    if f["must_mult"]<1.0:   gaps.append("missing core Python or retrieval evidence")
    if prod<0.30:             gaps.append("no shipped retrieval/production evidence")
    elif prod<0.55:           gaps.append(f"limited shipped evidence (production {prod:.2f})")
    if jd<0.25:               gaps.append(f"weak JD-tool alignment ({jd:.2f})")
    if tech<0.35:             gaps.append(f"limited recent technical depth ({tech:.2f})")
    gap_str = "; ".join(gaps[:2]) if gaps else f"overall fit {score:.2f}"

    # Templates
    if prod>=0.85:
        opts=[
            (f"Shipped retrieval/ranking at {company} ({yoe:.0f}y, production {prod:.2f}). "
             f"{jd_str}. {avail}, {bh}.{stale} {caution_str}".strip(),"S1"),
            (f"{company} — {yoe:.0f}y, strong production-shipping ({prod:.2f}), "
             f"{jd_str}. {avail}.{stale} {caution_str}".strip(),"S2"),
            (f"This {title.lower()} at {company} has {_article(yoe)} {yoe:.0f}-year career "
             f"with shipped retrieval/ranking. {jd_str}, {avail}, {bh}.{stale} {caution_str}".strip(),"S3"),
            (f"{yoe:.0f}y at {company}: shipped ranking/retrieval ({prod:.2f}), "
             f"{jd_str}, {avail}, {bh}.{stale} {caution_str}".strip(),"S4"),
        ]
    elif score>=0.65:
        opts=[
            (f"This {title.lower()} at {company} ({yoe:.0f}y) shows {jd_str} and {avail}. "
             f"Primary gap: {gap_str}.{stale} {caution_str}".strip(),"M1"),
            (f"{company} — {yoe:.0f}y, {jd_str}, {avail}. "
             f"Gap: {gap_str}.{stale} {caution_str}".strip(),"M2"),
            (f"{yoe:.0f}y at {company}: {jd_str}, {avail}. "
             f"Limiting factor: {gap_str}.{stale} {caution_str}".strip(),"M3"),
        ]
    else:
        opts=[
            (f"{company} — {yoe:.0f}y {title.lower()}, {avail}. "
             f"Gap: {gap_str}. {caution_str}".strip(),"W1"),
            (f"Below threshold ({score:.2f}). {title} at {company}, {yoe:.0f}y, "
             f"{avail}. {gap_str}. {caution_str}".strip(),"W2"),
            (f"{avail.capitalize()} — {company}, {yoe:.0f}y. "
             f"{gap_str}. {caution_str}".strip(),"W3"),
            (f"{yoe:.0f}y at {company} as {title.lower()}. {gap_str}. "
             f"{avail}. {caution_str}".strip(),"W4"),
        ]

    best = sorted(opts, key=lambda x: _tmpl[x[1]])[0]
    _tmpl[best[1]] += 1
    text = best[0].strip()
    reason = text[0].upper()+text[1:] if text else text

    swap_info = scored_candidate.get("swap_info")
    if swap_info:
        sr, won = swap_info
        reason = _pairwise_note(reason, sr, won)
    return reason
