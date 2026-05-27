"""
Fan commentary generation functions.
"""
import re
import numpy as np
from typing import List

# These will be injected by inference_engine.py
_state = None
_HAS_NLTK = False
_wordpunct_tokenize = None
_nltk_stemmer = None
_get_current_laptime_fn = None
_get_formatted_team_fn = None
_laptime_str_fn = None

_narr_cooldown: dict = {}

_COMMENTARY_STOPWORDS = {
    "the", "and", "for", "with", "from", "this", "that", "have", "has", "are", "was", "were",
    "into", "after", "before", "over", "under", "next", "lap", "field", "watch", "right",
    "more", "less", "than", "then", "will", "would", "could", "should", "now", "one", "two",
    "three", "four", "five", "driver", "drivers", "team", "teams"
}

_COMMENTARY_VARIANTS = [
    ("leads restart", ["controls restart", "heads restart"]),
    ("pits after", ["dives in after", "boxes after"]),
    ("pulling away", ["stretching the gap", "edging clear"]),
    ("CLOSING", ["HUNTING", "CLOSING IN"]),
    ("Next-lap predictions", ["Projected next-lap pace", "Next-lap pace forecast"]),
    ("undercut threat building", ["undercut window opening", "undercut pressure rising"]),
    ("Pit window", ["Stop window", "Pit phase"]),
    ("in trouble", ["under pressure", "in difficulty"])
]


def set_commentary_context(state, HAS_NLTK, wordpunct_tokenize, nltk_stemmer,
                          get_current_laptime_fn, get_formatted_team_fn, laptime_str_fn):
    """Initialize commentary with state and NLTK helpers."""
    global _state, _HAS_NLTK, _wordpunct_tokenize, _nltk_stemmer
    global _get_current_laptime_fn, _get_formatted_team_fn, _laptime_str_fn
    
    _state = state
    _HAS_NLTK = HAS_NLTK
    _wordpunct_tokenize = wordpunct_tokenize
    _nltk_stemmer = nltk_stemmer
    _get_current_laptime_fn = get_current_laptime_fn
    _get_formatted_team_fn = get_formatted_team_fn
    _laptime_str_fn = laptime_str_fn


def ncool(key: str, lap_no: int, min_laps: int = 4) -> bool:
    """Check if enough laps have passed since last narrative."""
    return lap_no - _narr_cooldown.get(key, -99) >= min_laps


def nfire(key: str, lap_no: int) -> None:
    """Mark narrative as fired at lap."""
    _narr_cooldown[key] = lap_no


def ranked() -> List[str]:
    """Get drivers ranked by speed."""
    return [c for c, _ in sorted(_state["speed_rank"].items(), key=lambda x: x[1])]


def active(min_spd: float = 100.0) -> List[str]:
    """Get active drivers with minimum speed."""
    return [c for c, t in _state["pace_trend"].items() if t and t[-1] > min_spd]


def stem_signature(text: str) -> set:
    """Generate stemmed signature for text to detect duplicates."""
    if not _HAS_NLTK or not _wordpunct_tokenize or _nltk_stemmer is None:
        return {w for w in text.lower().split() if len(w) > 2}
    
    toks = _wordpunct_tokenize(text.lower())
    return {
        _nltk_stemmer.stem(t)
        for t in toks
        if t.isalpha() and len(t) > 2 and t not in _COMMENTARY_STOPWORDS
    }


def apply_commentary_variation(text: str, lap_no: int, idx: int) -> str:
    """Apply narrative variation to avoid repetitive commentary."""
    out = " ".join(str(text).split())
    for base, choices in _COMMENTARY_VARIANTS:
        if base in out and choices:
            pick = choices[(lap_no + idx + len(out)) % len(choices)]
            out = out.replace(base, pick, 1)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out


def nltk_refine_commentary(lines: List[str], lap_no: int) -> List[str]:
    """Refine commentary by removing duplicates and applying variations."""
    if not lines:
        return []
    
    polished: List[str] = []
    signatures: List[set] = []
    
    for idx, raw in enumerate(lines):
        line = apply_commentary_variation(raw, lap_no, idx)
        sig = stem_signature(line)
        
        is_dup = False
        for prev in signatures:
            union = len(sig | prev) or 1
            overlap = len(sig & prev) / union
            if overlap >= 0.78:
                is_dup = True
                break
        
        if is_dup:
            continue
        
        polished.append(line)
        signatures.append(sig)
    
    return polished[:7]


def build_fan_commentary(lap_no: int) -> List[str]:
    """Build fan commentary from race state."""
    stories: List[tuple] = []
    ranked_drivers = ranked()
    active_drivers = active()

    if _state["sc_active"] and ncool("sc_active", lap_no, 1):
        old_tyres = [c for c in active_drivers if _state["tire_age"].get(c, 0) >= 8]
        names = ", ".join(old_tyres[:4])
        tail = f" {names} have a free stop." if names else ""
        stories.append((100, f"🟡 SAFETY CAR — field bunching up.{tail}"))
        nfire("sc_active", lap_no)

    sc_end = _state.get("sc_end_lap", 0)
    if not _state["sc_active"] and sc_end == lap_no and ncool("sc_end", lap_no, 1):
        p1 = ranked_drivers[0] if ranked_drivers else "?"
        p2 = ranked_drivers[1] if len(ranked_drivers) > 1 else "?"
        stories.append((99, f"🟢 GREEN FLAG! {p1} leads restart, {p2} right behind — watch for passes!"))
        nfire("sc_end", lap_no)

    rain_alerts = [x for x in _state["suggestions"] if "RAIN ONSET" in x or "TRACK DRYING" in x]
    if rain_alerts and ncool("weather", lap_no, 2):
        slick_n = sum(1 for c in active_drivers if _state["current_compound"].get(c, "") not in ("INTER", "WET"))
        if "RAIN ONSET" in rain_alerts[0]:
            stories.append((98, f"🌧️ Rain starting! {slick_n} drivers on slicks — the intermediate call is open NOW."))
        else:
            inters = [c for c in active_drivers if _state["current_compound"].get(c, "") in ("INTER", "WET")]
            names = ", ".join(inters[:3]) or "Several drivers"
            stories.append((97, f"☀️ Track drying — {names} must decide: box for slicks or gamble one more lap?"))
        nfire("weather", lap_no)

    for code in list(_state.get("speed_collapsed", set())):
        key = f"col_{code}"
        if not ncool(key, lap_no, 3):
            continue
        lt_self = _get_current_laptime_fn(code)
        all_lts = [_get_current_laptime_fn(c) for c in active_drivers if _get_current_laptime_fn(c) > 0]
        lt_field = float(np.median(all_lts)) if all_lts else 0
        gap_s = lt_self - lt_field
        stories.append((95, f"🚨 {code} ({_get_formatted_team_fn(code)}) in trouble — losing {gap_s:.0f}s/lap to the field."))
        nfire(key, lap_no)

    for alert in _state["suggestions"]:
        if "🔵 PITTED" not in alert:
            continue
        try:
            code = alert.split("]")[1].strip().split()[0]
        except:
            continue
        key = f"pit_{code}_{lap_no}"
        if not ncool(key, lap_no, 1):
            continue
        comp = _state["current_compound"].get(code, "?")
        age = _state["tire_age"].get(code, 0)
        sc_c = " — under Safety Car!" if _state.get("sc_active") else "."
        stories.append((90, f"🔵 {code} ({_get_formatted_team_fn(code)}) pits after {age+1} laps, back on {comp.lower()}{sc_c}"))
        nfire(key, lap_no)

    if len(ranked_drivers) >= 2 and ncool("podium", lap_no, 3):
        p1, p2 = ranked_drivers[0], ranked_drivers[1]
        lt1, lt2 = _get_current_laptime_fn(p1), _get_current_laptime_fn(p2)
        if lt1 > 0 and lt2 > 0:
            gap = lt2 - lt1
            w = _state["win_proba"].get(p1, 0)
            if gap > 0.4:
                stories.append((88, f"🏁 {p1} ({_get_formatted_team_fn(p1)}) pulling away — {gap:.2f}s/lap faster than {p2}. Win: {w:.0%}."))
            elif gap < -0.4:
                stories.append((88, f"🏁 {p2} ({_get_formatted_team_fn(p2)}) CLOSING — {abs(gap):.2f}s/lap quicker than leader {p1}. Lead at risk."))
            else:
                stories.append((85, f"🏁 {p1} ({_get_formatted_team_fn(p1)}) leads, {p2} within {abs(gap):.2f}s/lap. {w:.0%} win probability."))
        nfire("podium", lap_no)

    if not _state["sc_active"] and ncool("lt_table", lap_no, 4):
        rows = []
        for code in ranked_drivers[:5]:
            curr = _state["current_laptime"].get(code, 0)
            pred = _state["predicted_laptime"].get(code, 0)
            if curr > 0 and pred > 0:
                diff = pred - curr
                arrow = f"↑{abs(diff):.2f}s faster" if diff < -0.05 else (f"↓{abs(diff):.2f}s slower" if diff > 0.05 else "~same")
                rows.append(f"{code} {_laptime_str_fn(pred)} ({arrow})")
        if rows:
            stories.append((84, f"⏱️ Next-lap predictions: " + "  |  ".join(rows)))
            nfire("lt_table", lap_no)

    for alert in _state["pattern_insights"]:
        if "UNDERCUT RISK" not in alert:
            continue
        try:
            older = alert.split("]")[1].strip().split("(")[0].strip()
            fresh = alert.split("vs")[1].strip().split("(")[0].strip()
        except:
            continue
        key = f"uc_{older}_{fresh}"
        if not ncool(key, lap_no, 4):
            continue
        age_o, age_f = _state["tire_age"].get(older, 0), _state["tire_age"].get(fresh, 0)
        lt_o, lt_f = _get_current_laptime_fn(older), _get_current_laptime_fn(fresh)
        pace_adv = lt_o - lt_f
        stories.append((78, f"⚡ {_get_formatted_team_fn(older)}: {older} (lap {age_o}) vs {fresh} (lap {age_f}). {fresh} {pace_adv:+.2f}s/lap quicker — undercut threat building."))
        nfire(key, lap_no)

    for alert in _state["pattern_insights"]:
        if "TYRE CLIFF" not in alert:
            continue
        try:
            code = alert.split("]")[1].strip().split()[0]
        except:
            continue
        key = f"cliff_{code}"
        if not ncool(key, lap_no, 5):
            continue
        age = _state["tire_age"].get(code, 0)
        comp = _state["current_compound"].get(code, "?")
        pred = _state["predicted_laptime"].get(code, 0)
        curr = _state["current_laptime"].get(code, 0)
        loss = f" — {pred-curr:+.2f}s forecast next lap." if pred > 0 and curr > 0 and abs(pred-curr) > 0.1 else "."
        stories.append((75, f"📉 {code} ({_get_formatted_team_fn(code)}) hitting the cliff — {age} laps on {comp.lower()}{loss} Pit window urgent."))
        nfire(key, lap_no)

    pit_flagged = [c for c in active_drivers if _state["pit_alert"].get(c)]
    if pit_flagged and ncool("pit_grp", lap_no, 3):
        names = ", ".join(pit_flagged[:4])
        tail = " 🟡 FREE stop under SC!" if _state["sc_active"] else "."
        stories.append((72, f"🔧 Pit window: {names} flagged by model{tail}"))
        nfire("pit_grp", lap_no)

    fade = [a for a in _state["suggestions"] if "PACE FADE" in a]
    if fade and ncool("fade_grp", lap_no, 5):
        try:
            code = fade[0].split("]")[1].strip().split()[0]
            curr, pred = _state["current_laptime"].get(code, 0), _state["predicted_laptime"].get(code, 0)
            trend_str = f" — next lap forecast {_laptime_str_fn(pred)}" if pred > 0 else ""
            stories.append((65, f"🐢 {code} ({_get_formatted_team_fn(code)}) fading, lap times getting slower each stint lap{trend_str}."))
        except:
            pass
        nfire("fade_grp", lap_no)

    stories.sort(key=lambda x: -x[0])
    seen, result = set(), []
    for _, text in stories:
        k = text[:40]
        if k not in seen:
            seen.add(k)
            result.append(text)
        if len(result) >= 7:
            break
    
    return result
