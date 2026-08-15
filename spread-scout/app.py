"""
Spread Scout — deep-OTM put credit spread screener
Runs on your Massive.com (Polygon-compatible) market data.

Screens single-stock underlyings (no ETFs) for put credit spreads:
  - short strike 18-22% below spot (configurable)
  - two DTE regimes: 30-45 days and 90-190 days (each toggleable)
  - max loss (collateral) capped per spread
  - quote-midpoint pricing with liquidity + bid/ask width filters
  - news sentiment + earnings-mention overlay per candidate
  - optional Claude summarization of the news picture

Secrets required (see .streamlit/secrets.toml.example):
  MASSIVE_API_KEY   your Massive/Polygon API key
  APP_PASSWORD      shared password for private access
  ANTHROPIC_API_KEY optional, enables AI news briefs
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st

API_BASE = "https://api.massive.com"  # Polygon-compatible; api.polygon.io also works

DEFAULT_UNIVERSE = [
    # mega cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    # semis / high beta
    "AMD", "MU", "PLTR", "COIN", "MSTR", "SHOP", "UBER", "ABNB",
    # financials
    "JPM", "BAC", "GS", "V", "MA",
    # energy / industrials / staples
    "XOM", "CVX", "CAT", "GE", "BA", "COST", "WMT", "MCD", "HD", "NKE", "DIS",
    # healthcare
    "UNH", "LLY", "JNJ", "MRK",
]

# ---------------------------------------------------------------- auth gate

st.set_page_config(page_title="Spread Scout", page_icon="📈", layout="wide")

st.markdown("""
<style>
  :root {
    --bg:#0e1218; --panel:#161c27; --line:#242c3a;
    --gold:#c9a227; --text:#e6e9ef; --muted:#8b95a7;
    --good:#5fd39a; --bad:#f0806c;
  }
  .stApp { background: var(--bg); }
  .block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1500px; }
  h1, h2, h3 { color:#e8e2d4 !important; letter-spacing:-.01em; font-weight:600; }
  h1 { font-size:2.1rem !important; margin-bottom:.1rem; }

  /* masthead */
  .scout-tag { color:var(--gold); font-size:.72rem; text-transform:uppercase;
               letter-spacing:.22em; margin:0 0 .35rem 0; font-weight:600; }
  .scout-sub { color:var(--muted); font-size:.95rem; margin:.15rem 0 0 0; }
  .scout-rule { height:1px; background:linear-gradient(90deg,var(--gold),
                rgba(201,162,39,0) 55%); margin:1.1rem 0 1.4rem 0; border:0; }

  /* metric cards */
  div[data-testid="stMetric"] {
    background:var(--panel); border:1px solid var(--line);
    border-radius:10px; padding:.85rem 1rem;
  }
  div[data-testid="stMetricLabel"] p { color:var(--muted) !important;
    font-size:.72rem !important; text-transform:uppercase; letter-spacing:.1em; }
  div[data-testid="stMetricValue"] { color:var(--gold); font-size:1.6rem; }

  /* table */
  div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }

  /* sidebar */
  section[data-testid="stSidebar"] { background:#121721; border-right:1px solid var(--line); }
  section[data-testid="stSidebar"] .block-container { padding-top:1.5rem; }
  .side-label { color:var(--gold); font-size:.7rem; text-transform:uppercase;
                letter-spacing:.16em; font-weight:600;
                margin:1.1rem 0 .2rem 0; padding-bottom:.3rem;
                border-bottom:1px solid var(--line); }

  /* buttons */
  .stButton > button { border-radius:8px; font-weight:600; letter-spacing:.02em; }
  .stDownloadButton > button { border-radius:8px; border:1px solid var(--line);
                               background:var(--panel); color:var(--text); }

  /* expanders */
  details, div[data-testid="stExpander"] {
    background:var(--panel); border:1px solid var(--line) !important;
    border-radius:10px; }

  /* chips */
  .chip { display:inline-block; padding:3px 11px; border-radius:999px;
          font-size:.75rem; font-weight:600; margin:0 6px 6px 0;
          letter-spacing:.02em; }
  .chip-pos { background:rgba(95,211,154,.12); color:var(--good);
              border:1px solid rgba(95,211,154,.28); }
  .chip-neg { background:rgba(240,128,108,.12); color:var(--bad);
              border:1px solid rgba(240,128,108,.28); }
  .chip-neu { background:rgba(139,149,167,.12); color:var(--muted);
              border:1px solid rgba(139,149,167,.25); }
  .chip-warn{ background:rgba(201,162,39,.14); color:#e8c766;
              border:1px solid rgba(201,162,39,.35); }

  /* news list */
  .news-row { padding:.45rem 0; border-bottom:1px solid rgba(36,44,58,.7);
              font-size:.9rem; }
  .news-row:last-child { border-bottom:0; }
  .news-date { color:var(--muted); font-variant-numeric:tabular-nums;
               margin-right:.6rem; font-size:.82rem; }
  .news-row a { color:var(--text); text-decoration:none; }
  .news-row a:hover { color:var(--gold); text-decoration:underline; }

  footer, #MainMenu { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


def masthead(subtitle: str) -> None:
    st.markdown(
        '<p class="scout-tag">put credit spreads · single stocks</p>'
        '<h1>Spread Scout</h1>'
        f'<p class="scout-sub">{subtitle}</p>'
        '<hr class="scout-rule">', unsafe_allow_html=True)


def gate() -> bool:
    if st.session_state.get("authed"):
        return True
    left, mid, right = st.columns([1, 1.1, 1])
    with mid:
        st.markdown('<div style="height:12vh"></div>', unsafe_allow_html=True)
        masthead("Private screener — enter the access password to continue.")
        pw = st.text_input("Access password", type="password",
                           placeholder="••••••••", label_visibility="collapsed")
        if pw:
            if pw == st.secrets.get("APP_PASSWORD", ""):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False


if not gate():
    st.stop()

API_KEY = st.secrets.get("MASSIVE_API_KEY", "")
if not API_KEY:
    st.error("MASSIVE_API_KEY missing from secrets. Add it and reload.")
    st.stop()

# ---------------------------------------------------------------- data layer


def _get(path: str, params: dict | None = None) -> dict:
    params = dict(params or {})
    params["apiKey"] = API_KEY
    r = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=600, show_spinner=False)
def spot_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    data = _get("/v2/snapshot/locale/us/markets/stocks/tickers",
                {"tickers": ",".join(tickers)})
    for t in data.get("tickers", []):
        px = (t.get("day") or {}).get("c") or (t.get("prevDay") or {}).get("c")
        if px:
            out[t["ticker"]] = float(px)
    return out


@st.cache_data(ttl=600, show_spinner=False)
def put_chain(ticker: str, exp_gte: str, exp_lte: str,
              k_lo: float, k_hi: float) -> pd.DataFrame:
    """All puts for one underlying inside an expiry window and strike band."""
    rows, url_params = [], {
        "contract_type": "put",
        "expiration_date.gte": exp_gte, "expiration_date.lte": exp_lte,
        "strike_price.gte": k_lo, "strike_price.lte": k_hi, "limit": 250,
    }
    data = _get(f"/v3/snapshot/options/{ticker}", url_params)
    while True:
        for c in data.get("results", []):
            det, day = c.get("details", {}), c.get("day", {})
            quote, greeks = c.get("last_quote", {}) or {}, c.get("greeks", {}) or {}
            bid, ask = quote.get("bid"), quote.get("ask")
            mid = (bid + ask) / 2 if bid and ask else None
            rows.append({
                "ticker": ticker,
                "exp": det.get("expiration_date"),
                "strike": det.get("strike_price"),
                "mid": mid,                       # preferred price
                "last": day.get("close"),         # fallback price
                "bid": bid, "ask": ask,
                "delta": greeks.get("delta"),
                "iv": c.get("implied_volatility"),
                "oi": c.get("open_interest") or 0,
                "vol": day.get("volume") or 0,
            })
        nxt = data.get("next_url")
        if not nxt:
            break
        data = requests.get(nxt, params={"apiKey": API_KEY}, timeout=30).json()
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["px"] = df["mid"].fillna(df["last"])
    df["px_src"] = df["mid"].notna().map({True: "mid", False: "last"})
    return df.dropna(subset=["px", "strike", "exp"])


@st.cache_data(ttl=900, show_spinner=False)
def ticker_news(ticker: str, limit: int = 10) -> list[dict]:
    data = _get("/v2/reference/news",
                {"ticker": ticker, "limit": limit,
                 "order": "desc", "sort": "published_utc"})
    arts = []
    for a in data.get("results", []):
        senti = "neutral"
        for ins in a.get("insights") or []:
            if ins.get("ticker") == ticker:
                senti = ins.get("sentiment", "neutral")
        arts.append({
            "title": a.get("title", ""),
            "published": (a.get("published_utc") or "")[:10],
            "url": a.get("article_url", ""),
            "sentiment": senti,
            "description": a.get("description", ""),
            "keywords": a.get("keywords") or [],
        })
    return arts


def earnings_mentioned(articles: list[dict]) -> bool:
    hot = ("earnings", "eps", "quarterly results", "guidance", "reports q")
    for a in articles:
        blob = (a["title"] + " " + a["description"] + " "
                + " ".join(a["keywords"])).lower()
        if any(h in blob for h in hot):
            return True
    return False


def claude_brief(ticker: str, articles: list[dict]) -> str | None:
    key = st.secrets.get("ANTHROPIC_API_KEY")
    if not key or not articles:
        return None
    digest = "\n".join(
        f"- [{a['sentiment']}] {a['published']} {a['title']}: {a['description'][:200]}"
        for a in articles[:8]
    )
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={
                "model": "claude-sonnet-4-6", "max_tokens": 400,
                "messages": [{"role": "user", "content":
                    f"You are helping evaluate a bullish put credit spread on "
                    f"{ticker} (profits if the stock stays above a strike ~20% "
                    f"below today's price until expiration). Based only on these "
                    f"recent headlines, give a 3-4 sentence risk read: overall "
                    f"sentiment, any looming events, and whether anything here "
                    f"threatens a 20%+ decline scenario. Be direct, no preamble, "
                    f"no financial advice disclaimer.\n\n{digest}"}],
            }, timeout=45)
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json().get("content", []))
    except Exception:
        return None

# ---------------------------------------------------------------- screener


@dataclass
class Params:
    otm_lo: float
    otm_hi: float
    max_loss: float
    min_ror: float
    min_oi_short: int
    min_oi_long: int
    max_spread_pct: float
    min_width: float


def build_spreads(chain: pd.DataFrame, spot: float, dte_of: dict[str, int],
                  p: Params) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame()
    shorts = chain[(chain["strike"] >= (1 - p.otm_hi) * spot)
                   & (chain["strike"] <= (1 - p.otm_lo) * spot)
                   & (chain["oi"] >= p.min_oi_short)]
    out = []
    for _, s in shorts.iterrows():
        # penalize wide markets on the short leg when quotes exist
        if s["bid"] and s["ask"] and s["mid"]:
            if s["mid"] > 0 and (s["ask"] - s["bid"]) / s["mid"] > p.max_spread_pct:
                continue
        longs = chain[(chain["exp"] == s["exp"])
                      & (chain["strike"] < s["strike"] - p.min_width + 1e-9)
                      & (chain["oi"] >= p.min_oi_long)]
        for _, l in longs.iterrows():
            width = s["strike"] - l["strike"]
            credit = s["px"] - l["px"]
            if credit <= 0.05:
                continue
            max_loss = (width - credit) * 100
            if max_loss <= 0 or max_loss > p.max_loss:
                continue
            ror = 100 * credit / (width - credit)
            if ror < p.min_ror:
                continue
            dte = dte_of.get(s["exp"], 0) or 1
            out.append({
                "ticker": s["ticker"], "exp": s["exp"], "dte": dte,
                "short_k": s["strike"], "long_k": l["strike"],
                "pct_otm": round(100 * (1 - s["strike"] / spot), 1),
                "width": round(width, 1), "credit": round(credit, 2),
                "max_loss": round(max_loss, 0),
                "ror_pct": round(ror, 1),
                "annualized_pct": round(ror * 365 / dte, 1),
                "short_delta": round(s["delta"], 3) if s["delta"] else None,
                "short_iv": round(s["iv"], 2) if s["iv"] else None,
                "short_oi": int(s["oi"]), "long_oi": int(l["oi"]),
                "pricing": s["px_src"],
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # keep the best spread per (ticker, exp, short strike) to reduce noise
    return (df.sort_values("ror_pct", ascending=False)
              .groupby(["ticker", "exp", "short_k"], as_index=False).head(2)
              .sort_values("ror_pct", ascending=False).reset_index(drop=True))

# ---------------------------------------------------------------- UI

today = dt.date.today()


def side_label(text: str) -> None:
    st.markdown(f'<p class="side-label">{text}</p>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown('<p class="scout-tag">scan setup</p>', unsafe_allow_html=True)

    side_label("Universe")
    tickers_txt = st.text_area("Tickers", ", ".join(DEFAULT_UNIVERSE), height=120,
                               label_visibility="collapsed",
                               help="Comma separated. Single stocks only — no ETFs.")
    universe = [t.strip().upper() for t in tickers_txt.split(",") if t.strip()]
    st.caption(f"{len(universe)} underlyings")

    side_label("Expirations")
    use_short = st.checkbox("30–45 DTE", value=True)
    use_long = st.checkbox("90–190 DTE (far-dated)", value=True,
                           help="Longer dated spreads carry vega risk: a vol spike "
                                "hurts the position before price does.")

    side_label("Strategy")
    otm_band = st.slider("Short strike, % below spot", 10, 35, (18, 22))
    max_loss = st.number_input("Max loss per spread ($)", 100, 25000, 2200, 100)
    min_ror = st.slider("Min return on risk (%)", 0, 40, 5)
    min_width = st.number_input("Min spread width ($)", 1.0, 50.0, 2.5, 0.5)

    side_label("Liquidity")
    min_oi_s = st.number_input("Min OI, short leg", 0, 50000, 300, 50)
    min_oi_l = st.number_input("Min OI, long leg", 0, 50000, 100, 50)
    max_ba = st.slider("Max bid/ask width (% of mid)", 5, 100, 40)

    side_label("Events")
    flag_earn = st.checkbox("Flag earnings mentions in news", value=True)
    has_claude_key = bool(st.secrets.get("ANTHROPIC_API_KEY"))
    use_claude = st.checkbox("AI news brief per candidate", value=has_claude_key,
                             disabled=not has_claude_key,
                             help=None if has_claude_key else
                             "Add ANTHROPIC_API_KEY to secrets to enable.")

    st.markdown('<div style="height:.6rem"></div>', unsafe_allow_html=True)
    run = st.button("Run scan", type="primary", use_container_width=True)

params = Params(otm_lo=otm_band[0] / 100, otm_hi=otm_band[1] / 100,
                max_loss=max_loss, min_ror=min_ror,
                min_oi_short=min_oi_s, min_oi_long=min_oi_l,
                max_spread_pct=max_ba / 100, min_width=min_width)

windows = []
if use_short:
    windows.append(("30–45 DTE", today + dt.timedelta(days=28),
                    today + dt.timedelta(days=46)))
if use_long:
    windows.append(("90–190 DTE", today + dt.timedelta(days=90),
                    today + dt.timedelta(days=190)))

win_txt = " + ".join(w[0] for w in windows) or "no expiration window selected"
masthead(f"{len(universe)} names · {otm_band[0]}–{otm_band[1]}% OTM · "
         f"max loss ${max_loss:,} · {win_txt}")

# ------------------------------------------------------------------- scan


def run_scan() -> dict:
    """Fetch chains and build the candidate table. Returns a scan record."""
    spots = spot_prices(tuple(universe))
    notes = []
    missing = [t for t in universe if t not in spots]
    if missing:
        notes.append(f"No spot price for {', '.join(missing)} — skipped.")

    all_rows = []
    work = [(t, w) for t in universe if t in spots for w in windows]
    prog = st.progress(0.0, text="Scanning chains…")
    for i, (t, (label, d0, d1)) in enumerate(work):
        prog.progress((i + 1) / len(work), text=f"Scanning {t} · {label}")
        spot = spots[t]
        k_lo = (1 - params.otm_hi) * spot - max_loss / 100 - 5   # room for long legs
        k_hi = (1 - params.otm_lo) * spot + 1
        try:
            chain = put_chain(t, d0.isoformat(), d1.isoformat(), k_lo, k_hi)
        except requests.HTTPError as e:
            notes.append(f"{t}: chain fetch failed ({e.response.status_code}).")
            continue
        if chain.empty:
            continue
        dte_of = {e: (dt.date.fromisoformat(e) - today).days
                  for e in chain["exp"].unique()}
        spreads = build_spreads(chain, spot, dte_of, params)
        if not spreads.empty:
            spreads.insert(1, "window", label)
            all_rows.append(spreads)
    prog.empty()

    df = (pd.concat(all_rows, ignore_index=True)
          .sort_values("ror_pct", ascending=False).reset_index(drop=True)
          if all_rows else pd.DataFrame())
    return {"results": df, "notes": notes,
            "at": dt.datetime.now().strftime("%b %d, %Y · %H:%M")}


if run:
    if not windows:
        st.warning("Enable at least one DTE window in the sidebar.")
        st.stop()
    st.session_state["scan"] = run_scan()

scan = st.session_state.get("scan")

if scan is None:
    st.info("**Ready.** Set your parameters in the sidebar, then run the scan.\n\n"
            "Candidates price off live bid/ask midpoints where available; rows "
            "marked `last` fell back to a last trade and deserve skepticism.")
    st.stop()

results = scan["results"]
for note in scan["notes"]:
    st.caption(note)

if results.empty:
    st.warning("**No spreads matched.** Loosen the return-on-risk floor, widen "
               "the OTM band, or add higher-IV names — 20% OTM pays near zero "
               "on calm mega caps at short DTE.")
    st.stop()

# ------------------------------------------------------------------ summary

c1, c2, c3, c4 = st.columns(4)
c1.metric("Spreads found", f"{len(results):,}")
c2.metric("Underlyings", results["ticker"].nunique())
c3.metric("Best return on risk", f"{results['ror_pct'].max():.1f}%")
c4.metric("Priced from quotes", f"{(results['pricing'] == 'mid').mean():.0%}")
st.caption(f"Scan completed {scan['at']}")

tab_spreads, tab_news = st.tabs(["Candidates", "News & events"])

with tab_spreads:
    f1, f2, f3 = st.columns([1.2, 2.2, 1])
    win_pick = f1.selectbox("Window", ["All"] + sorted(results["window"].unique()))
    tick_pick = f2.multiselect("Tickers", sorted(results["ticker"].unique()),
                               placeholder="All tickers")
    quotes_only = f3.toggle("Quote-priced only", value=False,
                            help="Hide rows that fell back to a stale last trade.")

    view = results
    if win_pick != "All":
        view = view[view["window"] == win_pick]
    if tick_pick:
        view = view[view["ticker"].isin(tick_pick)]
    if quotes_only:
        view = view[view["pricing"] == "mid"]

    if view.empty:
        st.info("No candidates match these filters.")
    else:
        disp = view.copy()
        disp["legs"] = (disp["short_k"].map("{:g}".format) + " / "
                        + disp["long_k"].map("{:g}".format))
        disp = disp[["ticker", "window", "exp", "dte", "legs", "pct_otm", "width",
                     "credit", "max_loss", "ror_pct", "annualized_pct",
                     "short_delta", "short_iv", "short_oi", "long_oi", "pricing"]]
        st.dataframe(
            disp, use_container_width=True, hide_index=True, height=560,
            column_config={
                "ticker": st.column_config.TextColumn("Ticker", width="small"),
                "window": st.column_config.TextColumn("Window", width="small"),
                "exp": st.column_config.TextColumn("Expiry", width="small"),
                "dte": st.column_config.NumberColumn("DTE", format="%d", width="small"),
                "legs": st.column_config.TextColumn(
                    "Short / long", help="Short strike / long strike"),
                "pct_otm": st.column_config.NumberColumn("% OTM", format="%.1f%%"),
                "width": st.column_config.NumberColumn("Width", format="$%.1f"),
                "credit": st.column_config.NumberColumn("Credit", format="$%.2f"),
                "max_loss": st.column_config.NumberColumn("Max loss", format="$%d"),
                "ror_pct": st.column_config.ProgressColumn(
                    "Return on risk", format="%.1f%%", min_value=0.0,
                    max_value=float(max(disp["ror_pct"].max(), 1))),
                "annualized_pct": st.column_config.NumberColumn(
                    "Annualized", format="%.0f%%"),
                "short_delta": st.column_config.NumberColumn("Δ short", format="%.3f"),
                "short_iv": st.column_config.NumberColumn("IV short", format="%.2f"),
                "short_oi": st.column_config.NumberColumn("OI short", format="%d"),
                "long_oi": st.column_config.NumberColumn("OI long", format="%d"),
                "pricing": st.column_config.TextColumn(
                    "Priced", width="small",
                    help="mid = bid/ask midpoint · last = stale last trade"),
            })
        st.caption(f"Showing {len(view):,} of {len(results):,} candidates.")

    st.download_button("Download CSV", results.to_csv(index=False),
                       f"spread_scout_{today}.csv", "text/csv", icon=":material/download:")

# ------------------------------------------------------------ news overlay

CHIP_CLASS = {"positive": "chip-pos", "negative": "chip-neg", "neutral": "chip-neu"}

with tab_news:
    top_names = list(results["ticker"].drop_duplicates().head(8))
    st.caption(f"Recent coverage and sentiment for the top {len(top_names)} "
               "underlyings by return on risk.")
    for t in top_names:
        try:
            arts = ticker_news(t)
        except requests.RequestException as e:
            st.caption(f"{t}: news fetch failed ({e}).")
            continue
        counts = pd.Series([a["sentiment"] for a in arts]).value_counts().to_dict()
        earn = flag_earn and earnings_mentioned(arts)
        title = f"{t} — {len(arts)} recent articles"
        with st.expander(title + ("  ·  ⚠ earnings in news flow" if earn else "")):
            chips = "".join(
                f'<span class="chip {CHIP_CLASS.get(k, "chip-neu")}">{k} · {v}</span>'
                for k, v in counts.items())
            if earn:
                chips += '<span class="chip chip-warn">earnings mentioned</span>'
            st.markdown(chips or "<span class='chip chip-neu'>no tagged sentiment</span>",
                        unsafe_allow_html=True)
            if earn:
                st.warning("Earnings-related coverage detected. Verify the exact "
                           "report date before opening any expiration that "
                           "crosses it.")
            if use_claude:
                brief = claude_brief(t, arts)
                if brief:
                    st.markdown(f"**AI risk read** — {brief}")
            st.markdown("".join(
                f'<div class="news-row"><span class="news-date">{a["published"]}'
                f'</span><span class="chip {CHIP_CLASS.get(a["sentiment"], "chip-neu")}">'
                f'{a["sentiment"]}</span> <a href="{a["url"]}" target="_blank">'
                f'{a["title"]}</a></div>' for a in arts[:6]),
                unsafe_allow_html=True)

st.markdown('<hr class="scout-rule">', unsafe_allow_html=True)
st.caption("Screening tool, not investment advice. Rows marked `last` used stale "
           "last-trade prices; verify against a live quote before trading.")
