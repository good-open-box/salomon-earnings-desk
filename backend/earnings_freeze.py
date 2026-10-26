#!/usr/bin/env python3
"""
earnings_freeze.py  (copy of the Salomon AI backend module, built during the hackathon) — frozen, timestamped earnings calls + private grading.

Oct 26 2026 (Patrick). earnings_predictions.json is regenerated every 4 hours, so a
call quoted in a post could later read differently. This file makes "frozen" true:

  freeze  Append the beat/miss call (prediction + confidence) for every company that
          reports in the next 7 days and is not frozen yet. Only reports AFTER today
          (ET) are frozen, so a call is always locked before the report. The first
          freeze of a report wins; entries are never edited or removed.
  grade   After a report, compare actual EPS with the estimate (Finnhub calendar) and
          append the result. Same definition as beat_rate: BEAT if actual >= estimate,
          MISS if below; an IN-LINE call is correct within +/-2%.
  verify  Recompute the hash chain — any edit to an earlier line breaks it.
  show    Print the frozen calls and the private scorecard.

Files (append-only JSONL): earnings_freeze.jsonl (one batch per line, sha256-chained)
and earnings_freeze_grades.jsonl. Only the CALL is frozen — no option contracts
(posts never carry strikes). Grades stay private; posting them needs Patrick's OK.

Cron: freeze 08:15 + 20:15 daily, grade 07:30 + 19:30 (ET, local time).
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

BASE_DIR    = os.environ.get("SALOMON_DATA_DIR", ".")   # Salomon backend data folder
PRED_FILE   = os.path.join(BASE_DIR, "earnings_predictions.json")
FREEZE_FILE = os.path.join(BASE_DIR, "earnings_freeze.jsonl")
GRADE_FILE  = os.path.join(BASE_DIR, "earnings_freeze_grades.jsonl")
ET          = ZoneInfo("America/New_York")
HORIZON_DAYS = 7
INLINE_PCT   = 2.0
CALLS = ("BEAT", "MISS", "IN-LINE")
CONFS = ("HIGH", "MEDIUM", "LOW")


def _lines(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return [json.loads(x) for x in f if x.strip()]
    except FileNotFoundError:
        return []


def _append(path: str, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _batch_hash(prev_hash: str, frozen_at: str, calls: list[dict]) -> str:
    body = json.dumps({"prev": prev_hash, "frozen_at": frozen_at, "calls": calls},
                      separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def frozen_calls() -> list[dict]:
    """Every frozen call, oldest first, each carrying its batch time and hash."""
    out = []
    for b in _lines(FREEZE_FILE):
        for c in b["calls"]:
            out.append({**c, "frozen_at": b["frozen_at"], "batch_hash": b["hash"]})
    return out


def freeze(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(ET)
    today = now.date()
    try:
        with open(PRED_FILE) as f:
            preds = json.load(f)
    except (OSError, ValueError):
        return []
    done = {(c["symbol"], c["earnings_date"]) for c in frozen_calls()}
    calls = []
    for sym, p in sorted(preds.items()):
        if not isinstance(p, dict):
            continue
        try:
            ed = datetime.strptime(str(p.get("earnings_date") or "")[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        call, conf = p.get("prediction"), p.get("confidence")
        if not (today < ed <= today + timedelta(days=HORIZON_DAYS)):
            continue
        if call not in CALLS or conf not in CONFS or (sym, ed.isoformat()) in done:
            continue
        calls.append({"symbol": sym, "earnings_date": ed.isoformat(),
                      "prediction": call, "confidence": conf,
                      "eps_estimate": p.get("eps_estimate")})
    if not calls:
        return []
    batches = _lines(FREEZE_FILE)
    prev = batches[-1]["hash"] if batches else "genesis"
    frozen_at = now.astimezone(ET).isoformat(timespec="seconds")
    _append(FREEZE_FILE, {"batch": len(batches) + 1, "frozen_at": frozen_at,
                          "calls": calls, "prev_hash": prev,
                          "hash": _batch_hash(prev, frozen_at, calls)})
    return calls


def verify() -> tuple[bool, str]:
    prev = "genesis"
    for i, b in enumerate(_lines(FREEZE_FILE), 1):
        if b.get("prev_hash") != prev or b.get("hash") != _batch_hash(prev, b["frozen_at"], b["calls"]):
            return False, f"chain broken at batch {i}"
        prev = b["hash"]
    return True, f"ok ({prev[:12]})"


def _actual(symbol: str, earnings_date: str) -> dict | None:
    import requests
    from dotenv import load_dotenv
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    key = os.environ.get("FINNHUB_API_KEY", "")
    if not key:
        return None
    d = datetime.strptime(earnings_date, "%Y-%m-%d").date()
    r = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                     params={"from": (d - timedelta(days=3)).isoformat(),
                             "to": (d + timedelta(days=3)).isoformat(),
                             "symbol": symbol, "token": key}, timeout=10)
    if r.status_code != 200:
        return None
    for row in r.json().get("earningsCalendar") or []:
        if row.get("epsActual") is not None and row.get("epsEstimate") not in (None, 0):
            return row
    return None


def grade(now: datetime | None = None) -> list[dict]:
    today = (now or datetime.now(ET)).date()
    graded = {(g["symbol"], g["earnings_date"]) for g in _lines(GRADE_FILE)}
    out = []
    for c in frozen_calls():
        key = (c["symbol"], c["earnings_date"])
        if key in graded or c["earnings_date"] > today.isoformat():
            continue
        try:
            row = _actual(*key)
        except Exception:
            row = None
        if not row:
            continue  # not reported / not published yet: try next run
        actual, est = float(row["epsActual"]), float(row["epsEstimate"])
        surprise = (actual - est) / abs(est) * 100
        outcome = "BEAT" if actual >= est else "MISS"
        correct = (abs(surprise) <= INLINE_PCT) if c["prediction"] == "IN-LINE" \
            else (c["prediction"] == outcome)
        g = {"symbol": c["symbol"], "earnings_date": c["earnings_date"],
             "prediction": c["prediction"], "confidence": c["confidence"],
             "frozen_at": c["frozen_at"], "eps_actual": actual, "eps_estimate": est,
             "surprise_pct": round(surprise, 2), "outcome": outcome, "correct": correct,
             "graded_at": datetime.now(ET).isoformat(timespec="seconds")}
        _append(GRADE_FILE, g)
        graded.add(key)
        out.append(g)
    return out


def public_view(days_back: int = 14) -> dict:
    """Frozen calls (recent + upcoming), their grades, the scorecard and the chain
    head — what /api/intel/frozen serves to any key (beat/miss calls only)."""
    cutoff = (datetime.now(ET) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    grades = {(g["symbol"], g["earnings_date"]): g for g in _lines(GRADE_FILE)}
    calls = []
    for c in frozen_calls():
        if c["earnings_date"] < cutoff:
            continue
        g = grades.get((c["symbol"], c["earnings_date"]))
        calls.append({k: c[k] for k in ("symbol", "earnings_date", "prediction",
                                        "confidence", "frozen_at", "batch_hash")}
                     | ({"result": g["outcome"], "correct": g["correct"],
                         "surprise_pct": g["surprise_pct"]} if g else {}))
    calls.sort(key=lambda c: (c["earnings_date"], c["symbol"]))
    ok, chain = verify()
    return {"calls": calls, "scorecard": scorecard(), "chain_ok": ok, "chain": chain,
            "method": "Each call is appended once, before its report, to a sha256-chained "
                      "log; graded after the report: BEAT if actual EPS >= estimate.",
            "generated_at": datetime.now(ET).isoformat(timespec="seconds")}


def scorecard() -> dict:
    gs = _lines(GRADE_FILE)
    right = sum(g["correct"] for g in gs)
    return {"graded": len(gs), "correct": right,
            "pct": round(right / len(gs) * 100, 1) if gs else None}


# "$410c", "$410 calls", "strike", "DTE" — but not "8 calls this week" (calls = predictions).
_STRIKE_RE = re.compile(r"\$\s?\d+(\.\d+)?\s?([cp]\b|calls?\b|puts?\b)|"
                        r"\bstrikes?\b|\b(DTE|expiry|expiration)\b", re.I)
_SCORELINE_RE = re.compile(r"correct|right|wrong|scorecard|graded|calls", re.I)
_TRADE_RE  = re.compile(r"\b(BUY|SELL|SHORT|LONG)\b")            # capitals = trade labels
_CLAIM_RE  = re.compile(r"frozen|freeze|locked|timestamp|graded?|grading|scorecard|scored", re.I)
_CALL_RE   = re.compile(r"\b(beat|miss|in-line|inline)\b", re.I)
_SCORE_RE  = re.compile(r"\b(\d+)\s*(?:of|/|out of)\s*(\d+)\b", re.I)
_TICK_RE   = re.compile(r"\$([A-Z]{1,5})\b")


def check_post(text: str) -> str | None:
    """Reason a public post misstates Salomon's earnings calls, or None.

    Patrick, Oct 26 2026: frozen calls and graded beat/miss results may be posted.
    A post that claims frozen/graded calls must quote them exactly: one $TICKER per
    call line, matching the freeze file, and any "N of M" score must match the grades.
    """
    t = text or ""
    if _STRIKE_RE.search(t):
        return "contains option contracts or strikes — posts give the beat/miss outlook only"
    for line in t.splitlines():
        if _TRADE_RE.search(line) and _TICK_RE.search(line):
            return "uses trade labels (BUY/SELL) — say 'expects a beat / miss' instead"
    if not _CLAIM_RE.search(t):
        return None
    frozen = {}
    for c in frozen_calls():
        frozen.setdefault(c["symbol"], []).append(c)
    grades = {g["symbol"]: g for g in _lines(GRADE_FILE)}
    for line in t.splitlines():
        if not _CALL_RE.search(line):
            continue
        ticks = _TICK_RE.findall(line)
        if len(ticks) != 1:
            return (f"each call line needs exactly one $TICKER so it can be checked: '{line.strip()[:60]}'")
        sym = ticks[0]
        if sym not in frozen:
            return f"${sym} has no frozen call — only frozen calls can be posted as frozen or graded"
        said = _CALL_RE.findall(line)[0].upper().replace("INLINE", "IN-LINE")
        calls = {c["prediction"] for c in frozen[sym]}
        g = grades.get(sym)
        if said not in calls and not (g and said == g["outcome"]):
            return f"${sym}: the frozen call is {'/'.join(sorted(calls))}, not {said}"
    sc = scorecard()
    for line in t.splitlines():
        if not _SCORELINE_RE.search(line) or "quarter" in line.lower():
            continue  # "beat 4 of the last 4 quarters" is history, not our score
        for m in _SCORE_RE.finditer(line):
            n, d = int(m.group(1)), int(m.group(2))
            if (n, d) != (sc["correct"], sc["graded"]):
                return (f"score says {n} of {d}, the graded record is "
                        f"{sc['correct']} of {sc['graded']}")
    return None


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "freeze":
        new = freeze()
        print(f"{datetime.now(ET):%Y-%m-%d %H:%M} froze {len(new)} calls: "
              + ", ".join(f"{c['symbol']} {c['prediction']}" for c in new))
    elif cmd == "grade":
        new = grade()
        print(f"{datetime.now(ET):%Y-%m-%d %H:%M} graded {len(new)}: "
              + ", ".join(f"{g['symbol']} {'OK' if g['correct'] else 'X'}" for g in new)
              + f" | scorecard {scorecard()}")
    elif cmd == "verify":
        print(verify())
    else:
        for c in frozen_calls():
            print(f"{c['earnings_date']} {c['symbol']:6} {c['prediction']:8} {c['confidence']:6} frozen {c['frozen_at']}")
        print(verify(), scorecard())
