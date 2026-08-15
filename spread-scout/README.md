# Spread Scout

A private screener for deep out-of-the-money **put credit spreads** on single
stocks (no ETFs), powered by your Massive.com market data.

What it does on every scan:

1. Pulls live put chains for your ticker universe in two expiration windows —
   30–45 DTE and 90–190 DTE (each can be toggled off).
2. Builds every valid spread whose short strike sits 18–22% below spot
   (band adjustable) and whose max loss fits your collateral cap
   (default $2,200).
3. Prices from **bid/ask midpoints** when quotes are available, falls back to
   last trade otherwise, and labels each row so you know which — spreads that
   look too good from stale last-trade prints are exactly the ones to distrust.
4. Filters for liquidity (open interest per leg, max bid/ask width).
5. Ranks by return on risk and shows annualized figures per DTE regime.
6. Overlays recent news with per-ticker sentiment tags, flags any
   earnings-related coverage, and (optionally) has Claude write a short risk
   read of the headlines for each surfaced name.

## Run locally

```bash
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# edit secrets.toml: add your MASSIVE_API_KEY and pick an APP_PASSWORD
streamlit run app.py
```

Open http://localhost:8501, enter the password, set parameters, run the scan.

## Share it privately

Two good options, in order of simplicity:

**Streamlit Community Cloud (free):**
1. Push this folder to a **private** GitHub repo (never commit
   `secrets.toml` — it's your API key).
2. Go to share.streamlit.io → New app → pick the repo.
3. In the app's Settings → Secrets, paste the contents of your
   `secrets.toml`.
4. In Settings → Sharing, keep the app private and invite viewers by email,
   or leave it link-accessible and rely on the in-app password gate.

Everyone you share with sees the app; nobody sees your API key, and every
scan runs through *your* Massive data.

**Your own box (VPS / home server):**
`streamlit run app.py --server.port 8501` behind any reverse proxy. The
in-app password gate handles access; add HTTPS via Caddy/nginx if exposing
it to the internet.

## Notes on the data

- API base is `https://api.massive.com` (Polygon-compatible; `api.polygon.io`
  works identically if you ever migrate keys).
- Earnings flagging is heuristic (keyword scan of the news feed) because the
  Benzinga earnings-calendar feed isn't included in the current Massive plan.
  Always verify exact report dates before opening an expiration that crosses
  one.
- Chain pulls are cached for 10 minutes and news for 15 to stay well inside
  rate limits; use the sidebar's Run scan to refresh.
- Far-dated (90–190 DTE) spreads carry meaningful **vega** risk: a volatility
  spike marks the position against you long before price nears your strike.
  The screener measures reward; the vega, and the crash-shaped loss profile,
  are the price of it.

## Notes on the interface

- All styling lives in `assets/styles.css`, injected once per run. It is a
  token system: change the palette, spacing scale, or type roles at the top of
  that file and the whole app follows.
- Rules marked `[ST]` in the stylesheet target Streamlit's internal DOM through
  `[data-testid="..."]` selectors and the `.st-key-<widget key>` class. Neither
  is a public API, so `requirements.txt` pins `streamlit==1.57.*`. If you
  unpin and the layout breaks, those rules are where to look — the stylesheet
  header lists every fragile selector.
- Two Streamlit quirks the code works around, both verified on 1.57:
  `st.html` sanitizes with DOMPurify and drops `<style>` and inline `<svg>`
  (so CSS and the payoff diagram go through `st.markdown`), and inline scripts
  must not contain a `<` character or the HTML parser mangles them.
- The payoff diagram overlays a ±1σ expected-move cone on the profit/loss
  profile. In the sidebar it is schematic (assumed 35% IV, $100 stock, drawn at
  your minimum return on risk); in a row's detail panel it uses that contract's
  real implied vol. The labels say which.

This is a screening tool, not investment advice.
