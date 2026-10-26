# Salomon Earnings Desk

**Live: https://desk.slmn.trade** · Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026) · Startup track · Theme: *Build for Your World*

An earnings desk for everyday investors. Ask, by typing or by voice, "Who reports
earnings this week and what's expected?", "Will WELL beat?" or "Was Salomon right
about Boeing?" and get a plain-language answer built from live market data, with
the numbers and sources shown.

What makes it more than a chatbot over a data feed is the **frozen-calls ledger**.
Every beat/miss call is locked *before* the company reports, timestamped and
sha256-chained so it can't be edited afterwards, then graded automatically against
the real result. The public scorecard can only show what was actually called.

## Meta models used

| Model | How it is used |
|---|---|
| **Muse Spark Open** (open weights) | Main model. Tool-calling over the live data feed for every answer |
| **Muse Spark 1.3** (Meta Model API) | Automatic backup when the main model errors or times out (daily-capped) |
| **Muse Voice Transcribe 1.0** (Meta Model API) | "Ask by voice": the browser records 16 kHz WAV, `/v1/asr/transcribe` turns it into the question |
| **Muse Wonder** | Coding model for the build session that wrote the first version of this app |

## Built during the hackathon vs. what existed before

The judges asked for this split, so here it is plainly.

**Built during the hackathon (from Oct 26, 2026):**
- **This whole app**: `server.py` and `index.html`. That covers the Muse Spark tool-calling loop over 9 data tools, the backup model, voice input, the frozen-calls panel, and the public deploy (`Dockerfile`, `fly.toml`).
- **Safety rules enforced in code, not just in the prompt:**
  - BUY/SELL trade labels never reach users and are never mapped to beat/miss.
  - Option contracts are kept off the public build.
  - The disclaimer is appended server-side.
  - Jailbreak and "which model are you" prompts are refused.
- **Production guards:**
  - per-visitor limits (Fly-Client-IP; spoofed `X-Forwarded-For` is ignored)
  - global daily caps, a concurrency cap and a model-backup cap
  - a 10-minute data cache, a kill switch (`DEMO_ENABLED=0`) and key-gated stats
- **The frozen-calls ledger, `backend/earnings_freeze.py`:** freeze, grade, verify and post-check, plus its read-only route `/api/intel/frozen`. It runs inside Salomon AI's private backend; it's published here as a copy.
- **Fixes to Salomon's data feed** found while building, in the private backend:
  - Report dates were frozen at prediction time, so companies that had already reported still showed "reporting today".
  - Added the real beat/miss call (`prediction`, `confidence`), which had been confused with the *trade* action.
  - Added the sample size behind beat rates (`quarters_analyzed`, `avg_surprise_pct`).

**Existed before the hackathon (Salomon AI, not part of this entry):**
- Salomon AI's market data API (`api.slmn.trade/api/intel/*`) and its earnings prediction engine.
- The slmn.trade website and accounts.
- The Salomon AI connector for Muse, submitted Oct 20. The page links to it as a way to keep using Salomon.

## How it works

```
browser ──question / voice──▶ server.py ──▶ Muse Spark Open (tool calls) ──▶ answer + sources
                                  │  └─ backup: Muse Spark 1.3 (Meta Model API)
                                  ├─ voice WAV ─▶ Muse Voice Transcribe
                                  └─ tools (cached 10 min) ─▶ Salomon AI read-only API
                                                               (earnings, macro, signals,
                                                                frozen calls, …)
```

- **Read-only:** no trading, no orders, no money movement, no access to anyone's account.
- **Not financial advice:** it describes Salomon's outlook ("expects a miss, high confidence") and never tells anyone what to buy.
- **Grading rule:** a BEAT call is correct when actual EPS is at or above the estimate; an IN-LINE call is correct within ±2%.

## Run it locally

Python 3.10+, standard library only:

```bash
export SALOMON_API_KEY=...     # free key: sign up at slmn.trade → Settings → API key
export BASETEN_API_KEY=...     # Muse Spark Open
export META_API_KEY=...        # Meta Model API (backup model + voice)
python3 server.py              # http://localhost:8000
```

Deploy: `fly deploy --remote-only`, with keys set as Fly secrets. Every limit is an env var; see the top of `server.py`.

## License

MIT, © 2026 Salomon AI Inc.
