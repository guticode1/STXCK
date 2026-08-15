# Spread Scout — Project Recap & Working Context

## What this is
A private, password-gated Streamlit web app that screens for **deep out-of-the-money
put credit spreads** on single stocks (no ETFs), using the owner's Massive.com
(Polygon-compatible) market data API. Built August 2026.

Live deployment: Streamlit Community Cloud, deployed from this repo
(`spread-scout/app.py` on branch `main`). Pushing to `main` auto-redeploys the
live app within ~1 minute.

## The strategy it screens for
Bull put spreads (put credit spreads): sell a put ~18–22% below spot, buy a
further-OTM put to cap risk.
- Max loss per spread = (strike width × 100) − credit, capped at **$2,200** by default
- Two expiration regimes, each toggleable: **30–45 DTE** and **90–190 DTE** (far-dated)
- Far-dated spreads carry vega risk (vol spikes hurt before price does);
  short-dated 20% OTM only pays on high-IV names — both are known trade-offs,
  not bugs.

## Architecture (single file: spread-scout/app.py)
1. **Auth gate** — shared password from `st.secrets["APP_PASSWORD"]`
2. **Data layer** — REST calls to `https://api.massive.com`:
   - `/v2/snapshot/locale/us/markets/stocks/tickers` → spot prices (batched)
   - `/v3/snapshot/options/{ticker}` → put chains (filtered by expiry window +
     strike band, paginated via `next_url`)
   - `/v2/reference/news` → per-ticker articles with sentiment `insights`
3. **Pricing rule** — bid/ask **midpoint preferred**, falls back to last trade;
   every result row is labeled `mid` or `last`. This exists because stale
   last-trade prints on illiquid strikes produce phantom "great" spreads.
   Never remove this labeling.
4. **Screener** — `build_spreads()`: pairs short strikes in the OTM band with
   lower long legs on the same expiry, applies max-loss cap, min return-on-risk,
   min OI per leg, max bid/ask width; ranks by return on risk.
5. **News overlay** — sentiment chips per surfaced ticker + heuristic earnings
   flag (keyword scan; the Benzinga earnings-calendar feed is NOT on the
   current Massive plan). Optional Claude API news brief if
   `ANTHROPIC_API_KEY` is present in secrets (currently not set — app runs
   fine without it).

## Secrets (in Streamlit Cloud app settings, and local .streamlit/secrets.toml)
- `MASSIVE_API_KEY` — required
- `APP_PASSWORD` — required, letters/numbers only (quotes or symbols in the
  password once broke TOML parsing; keep it simple)
- `ANTHROPIC_API_KEY` — optional, enables AI news briefs
Secrets are NEVER committed to this repo. `.streamlit/secrets.toml` must stay
out of version control.

## Known constraints & history
- Massive plan does not include Benzinga partner feeds (earnings calendar) —
  earnings detection is news-keyword-based; users must verify report dates.
- FMP calendar tier also unavailable on owner's plan.
- Default universe: ~36 liquid single stocks across sectors (mega-cap tech,
  semis/high-beta, financials, energy, staples, healthcare). ETFs deliberately
  excluded per owner's preference.
- Chain pulls cached 10 min, news 15 min (st.cache_data) for rate limits.
- Repo is public; the app's privacy comes from the password gate + secret keys,
  not repo visibility.

## Owner's typical asks / roadmap ideas
- Adjust universe, OTM band, DTE windows, liquidity floors
- Better earnings detection if plan upgrades ever add a calendar feed
- Possible: per-user logins, saved scan history, alerting on new candidates,
  IV-rank filter, CSV/email export of scans

## House rules for edits
- Keep it a single-file Streamlit app unless complexity truly demands otherwise
- Preserve the mid/last pricing labels and liquidity filters
- This is a screening tool, not investment advice — keep the disclaimer
- After any edit: verify `python3 -m py_compile spread-scout/app.py` passes,
  then commit and push to `main` to redeploy
