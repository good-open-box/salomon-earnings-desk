#!/usr/bin/env python3
"""
earnings_freeze.py — frozen, timestamped earnings calls + private grading.

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

BASE_DIR    = os.environ.get("SALOMON_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
PRED_FILE   = os.path.join(BASE_DIR, "earnings_predictions.json")
FREEZE_FILE = os.path.join(BASE_DIR, "earnings_freeze.jsonl")
GRADE_FILE  = os.path.join(BASE_DIR, "earnings_freeze_grades.jsonl")
ET          = ZoneInfo("America/New_York")
HORIZON_DAYS = 7
INLINE_PCT   = 2.0
CALLS = ("BEAT", "MISS", "IN-LINE")
# Oct 27 2026: AMT "beat" by +73% (2.87 vs 1.66): the calendar compared AFFO with a
# GAAP EPS estimate. A surprise this large on an estimate of $1+ almost always means
# two different earnings measures, so the grade is shown but not counted. Same rule
# for every company, right or wrong; small estimates (BA -0.21) are exempt.
MISMATCH_MIN_EST = 1.0
MISMATCH_PCT     = 50.0
MISMATCH_REASON  = "not counted: the reported and expected figures look like different earnings measures"
CONFS = ("HIGH", "MEDIUM", "LOW")
# Oct 27 2026 (Patrick: "yes archive"): outside proof. Each batch line's sha256 is
# stamped by public OpenTimestamps calendars (anchored in Bitcoin), so a batch can be
# shown to have existed at that time without trusting Salomon. Only the 32-byte
# hash leaves this machine. Receipts: earnings_freeze_ots/batch-N.<calendar>.ots
OTS_DIR       = os.path.join(BASE_DIR, "earnings_freeze_ots")
ANCHOR_FILE   = os.path.join(BASE_DIR, "earnings_freeze_anchors.jsonl")
OTS_CALENDARS = {"alice": "https://alice.btc.calendar.opentimestamps.org",
                 "bob":   "https://bob.btc.calendar.opentimestamps.org"}
# .ots file = magic, version 1, "sha256 of the file" op, the digest, then the
# calendar's pending timestamp.
_OTS_HEADER = (b"\x00OpenTimestamps\x00\x00Proof\x00\xbf\x89\xe2\xe8\x84\xe8\x92\x94"
               b"\x01\x08")


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


def batch_lines() -> list[bytes]:
    """The freeze file's lines exactly as stored (what gets stamped)."""
    try:
        with open(FREEZE_FILE, "rb") as f:
            return [x.rstrip(b"\n") for x in f if x.strip()]
    except FileNotFoundError:
        return []


def anchor(batch_no: int | None = None) -> list[dict]:
    """Stamp every batch that has no receipt yet (or just batch_no). Never raises."""
    import requests
    done = {a["batch"] for a in _lines(ANCHOR_FILE)}
    out = []
    for n, line in enumerate(batch_lines(), 1):
        if n in done or (batch_no and n != batch_no):
            continue
        digest = hashlib.sha256(line).digest()
        got = []
        for name, url in OTS_CALENDARS.items():
            try:
                r = requests.post(url + "/digest", data=digest, timeout=20,
                                  headers={"Accept": "application/vnd.opentimestamps.v1",
                                           "User-Agent": "SalomonEarningsDesk/1.0"})
                if r.status_code == 200 and r.content:
                    os.makedirs(OTS_DIR, exist_ok=True)
                    with open(os.path.join(OTS_DIR, f"batch-{n}.{name}.ots"), "wb") as f:
                        f.write(_OTS_HEADER + digest + r.content)
                    got.append(name)
            except Exception:
                pass
        if got:
            row = {"batch": n, "sha256": digest.hex(), "calendars": got,
                   "stamped_at": datetime.now(ET).isoformat(timespec="seconds")}
            _append(ANCHOR_FILE, row)
            out.append(row)
    return out


def proof(batch_no: int) -> dict | None:
    """One batch line + its receipts, for 'verify it yourself'."""
    lines = batch_lines()
    if not 1 <= batch_no <= len(lines):
        return None
    import base64
    rec = {}
    for name in OTS_CALENDARS:
        path = os.path.join(OTS_DIR, f"batch-{batch_no}.{name}.ots")
        if os.path.exists(path):
            with open(path, "rb") as f:
                rec[name] = base64.b64encode(f.read()).decode()
    a = next((x for x in _lines(ANCHOR_FILE) if x["batch"] == batch_no), None)
    return {"batch": batch_no, "line": lines[batch_no - 1].decode(),
            "sha256": hashlib.sha256(lines[batch_no - 1]).hexdigest(),
            "stamped_at": a and a["stamped_at"], "ots_b64": rec}


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
        row = {k: c[k] for k in ("symbol", "earnings_date", "prediction",
                                 "confidence", "frozen_at", "batch_hash")}
        eps = {"eps_actual": g.get("eps_actual"), "eps_estimate": g.get("eps_estimate")} if g else {}
        if g and excluded(g):
            row |= {"result": g["outcome"], "surprise_pct": g["surprise_pct"],
                    "excluded": True, "excluded_reason": MISMATCH_REASON} | eps
        elif g:
            row |= {"result": g["outcome"], "correct": g["correct"],
                    "surprise_pct": g["surprise_pct"]} | eps
        calls.append(row)
    calls.sort(key=lambda c: (c["earnings_date"], c["symbol"]))
    ok, chain = verify()
    anchors = [{k: a[k] for k in ("batch", "sha256", "stamped_at")} for a in _lines(ANCHOR_FILE)]
    return {"calls": calls, "scorecard": scorecard(), "chain_ok": ok, "chain": chain,
            "anchors": anchors,
            "method": "Each call is appended once, before its report, to a sha256-chained "
                      "log; graded after the report: BEAT if actual EPS >= estimate.",
            "generated_at": datetime.now(ET).isoformat(timespec="seconds")}


def excluded(g: dict) -> bool:
    """A grade kept in the file but left out of the score (see MISMATCH_*)."""
    try:
        return (abs(float(g["eps_estimate"])) >= MISMATCH_MIN_EST
                and abs(float(g["surprise_pct"])) >= MISMATCH_PCT)
    except (KeyError, TypeError, ValueError):
        return False


def scorecard() -> dict:
    all_gs = _lines(GRADE_FILE)
    gs = [g for g in all_gs if not excluded(g)]
    right = sum(g["correct"] for g in gs)
    return {"graded": len(gs), "correct": right, "excluded": len(all_gs) - len(gs),
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
        stamped = anchor() if new else []
        print(f"{datetime.now(ET):%Y-%m-%d %H:%M} froze {len(new)} calls: "
              + ", ".join(f"{c['symbol']} {c['prediction']}" for c in new)
              + (f" | stamped batch {', '.join(str(a['batch']) for a in stamped)}" if stamped else ""))
    elif cmd == "anchor":
        print(anchor())
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
