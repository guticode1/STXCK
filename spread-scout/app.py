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

Layout of the code:
  strategy.py          pure strategy layer — formulas, filters, ranking.
                       No Streamlit, no network. Unit tested in tests/.
  app.py (this file)   data fetching, caching, and the interface.
  assets/styles.css    all presentation.

Load-bearing rules that must not be quietly removed: spreads are priced from
bid/ask midpoints and rows that fell back to a stale last trade are tagged and
excluded by default; open interest is required on both legs before a spread is
constructed; the two DTE regimes are ranked separately.

This is a screening tool, not investment advice.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

from strategy import (
    BLOCK_REASON, ETF_BLOCKLIST, Params, RANK_METRICS, alternates_for,
    build_spreads, dedupe, earnings_in_window, is_blocked, market_session,
    prob_otm, rank_cross_regime, rank_within_regime, rv_from_closes,
    valid_quote,
)

API_BASE = "https://api.massive.com"  # Polygon-compatible; api.polygon.io also works
CSS_PATH = Path(__file__).parent / "assets" / "styles.css"

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

# Presentation-only metadata: drives chip dots and the sector legend.
SECTORS = {
    **{t: "tech" for t in ("AAPL", "MSFT", "AMZN", "GOOGL", "META", "ORCL", "NFLX",
                           "PLTR", "SHOP", "UBER", "ABNB", "TSLA", "DIS")},
    **{t: "semis" for t in ("NVDA", "AVGO", "AMD", "MU")},
    **{t: "fin" for t in ("JPM", "BAC", "GS", "V", "MA", "COIN", "MSTR")},
    **{t: "energy" for t in ("XOM", "CVX", "CAT", "GE", "BA")},
    **{t: "consumer" for t in ("COST", "WMT", "MCD", "HD", "NKE")},
    **{t: "health" for t in ("UNH", "LLY", "JNJ", "MRK")},
}
SECTOR_META = {
    "tech":     ("Tech",        "#6EA8FF"),
    "semis":    ("Semis",       "#C77DFF"),
    "fin":      ("Financials",  "#4FD18B"),
    "energy":   ("Energy/Ind",  "#F2994A"),
    "consumer": ("Consumer",    "#E0A72C"),
    "health":   ("Healthcare",  "#43C6DB"),
    "other":    ("Other",       "#8A94A6"),
}
PRESETS = {
    "Mega tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                      "AVGO", "ORCL", "NFLX"],
    "Semis":         ["NVDA", "AVGO", "AMD", "MU"],
    "Financials":    ["JPM", "BAC", "GS", "V", "MA"],
    "Energy":        ["XOM", "CVX", "CAT", "GE", "BA"],
    "Consumer":      ["COST", "WMT", "MCD", "HD", "NKE", "DIS"],
    "High IV":       ["TSLA", "COIN", "MSTR", "PLTR", "AMD", "MU", "SHOP"],
}
ALL_TICKERS = sorted(set(DEFAULT_UNIVERSE) | {t for v in PRESETS.values() for t in v})

WIN_SHORT = "30–45 DTE"
WIN_LONG = "90–190 DTE"


def sector_of(ticker: str) -> str:
    return SECTORS.get(ticker, "other")


def sector_color(ticker: str) -> str:
    return SECTOR_META[sector_of(ticker)][1]


# ---------------------------------------------------------------- chrome

st.set_page_config(page_title="Spread Scout", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")


@st.cache_data(show_spinner=False)
def _css(mtime: float) -> str:
    return CSS_PATH.read_text()


def inject_css(extra: str = "") -> None:
    """Single stylesheet injection per run.

    st.markdown, not st.html: st.html sanitizes with DOMPurify, which strips
    <style> blocks entirely — verified on 1.57.0. st.html stays in use for
    content markup, where sanitizing is exactly what we want.
    """
    css = _css(CSS_PATH.stat().st_mtime) + extra
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


inject_css()


def secret(name: str, default: str = "") -> str:
    """st.secrets.get() raises StreamlitSecretNotFoundError when no secrets file
    exists at all, which turned a missing-config case into a raw traceback."""
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def raw_html(body: str) -> None:
    """Escape hatch for markup st.html would sanitize away.

    st.html runs DOMPurify, which strips <style> AND inline <svg> wholesale
    (verified on 1.57.0). The payoff diagram is an inline SVG, so it has to go
    through st.markdown. Everything passed here is built by this module — no
    user or API text reaches it unescaped.
    """
    st.markdown(body, unsafe_allow_html=True)


# ---------------------------------------------------------------- auth gate


def gate() -> bool:
    if st.session_state.get("authed"):
        return True
    _, mid, _ = st.columns([1, 1.15, 1])
    with mid:
        st.html('<div style="height:14vh"></div>'
                '<p class="eyebrow rise">put credit spreads · single stocks</p>'
                '<p class="lede rise d1" style="margin-top:6px">Spread Scout</p>'
                '<p class="sub rise d2">Private screener. Enter the access '
                'password to continue.</p><div style="height:18px"></div>')
        pw = st.text_input("Access password", type="password", key="pw",
                           placeholder="Access password", label_visibility="collapsed")
        if pw:
            if pw == secret("APP_PASSWORD"):
                st.session_state["authed"] = True
                st.rerun()
            else:
                st.error("Wrong password.")
    return False


if not gate():
    st.stop()

API_KEY = secret("MASSIVE_API_KEY")
if not API_KEY:
    st.html('<p class="eyebrow">configuration</p>'
            '<p class="lede">Missing market-data key</p>'
            '<p class="sub">Add <code>MASSIVE_API_KEY</code> to your Streamlit '
            'secrets and reload. On Community Cloud that is Settings → Secrets; '
            'locally it is <code>.streamlit/secrets.toml</code>.</p>')
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


@st.cache_data(show_spinner=False)
def put_chain(ticker: str, exp_gte: str, exp_lte: str, k_lo: float, k_hi: float,
              is_open: bool, session_day: str, _ttl_bucket: int) -> pd.DataFrame:
    """All puts for one underlying inside an expiry window and strike band.

    Emits ONE price per contract (`px`) chosen by market session, plus the
    provenance needed to refuse incoherent pairs:

      px_src   "mid"   live two-sided quote
               "close" that session's mark, used only when the market is shut
      mark_day the session the mark belongs to

    day.close is ZERO-FILLED by the feed for a contract that did not trade, so
    a 0 there means "no mark", not "worth nothing". Treating it as a price is
    what let a 15-wide spread appear to pay 11.85: the liquid leg carried a
    real mark and the illiquid one carried 0.
    """
    rows, url_params = [], {
        "contract_type": "put",
        "expiration_date.gte": exp_gte, "expiration_date.lte": exp_lte,
        "strike_price.gte": k_lo, "strike_price.lte": k_hi, "limit": 250,
    }
    data = _get(f"/v3/snapshot/options/{ticker}", url_params)
    while True:
        for c in data.get("results", []):
            det, day = c.get("details", {}) or {}, c.get("day", {}) or {}
            quote_ = c.get("last_quote", {}) or {}
            greeks = c.get("greeks", {}) or {}
            trade = c.get("last_trade", {}) or {}
            bid, ask = quote_.get("bid"), quote_.get("ask")
            ok = valid_quote(bid, ask)
            mid = (bid + ask) / 2 if ok else None

            # a zero close is an absent mark, not a zero price
            close_px = trade.get("price") or day.get("close")
            if not close_px or close_px <= 0:
                close_px = None
            mark_day = _mark_day(trade, day, session_day)

            if is_open:
                px, src = (mid, "mid") if ok else (None, None)
            elif ok:
                px, src = mid, "mid"
            elif close_px:
                px, src = close_px, "close"
            else:
                px, src = None, None

            rows.append({
                "ticker": ticker,
                "exp": det.get("expiration_date"),
                "strike": det.get("strike_price"),
                "px": px, "px_src": src, "mark_day": mark_day,
                "bid": bid, "ask": ask,
                "ba_pct": ((ask - bid) / mid) if ok and mid > 0 else None,
                "delta": greeks.get("delta"),
                "iv": c.get("implied_volatility"),
                "vega": greeks.get("vega"),
                "oi": c.get("open_interest") or 0,
                "vol": day.get("volume") or 0,
                "spc": det.get("shares_per_contract"),
            })
        nxt = data.get("next_url")
        if not nxt:
            break
        data = requests.get(nxt, params={"apiKey": API_KEY}, timeout=30).json()
        time.sleep(0.15)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["px", "strike", "exp"])


def _mark_day(trade: dict, day: dict, session_day: str) -> str | None:
    """Which session a mark belongs to, from the last trade's timestamp."""
    ts = trade.get("sip_timestamp") or trade.get("t")
    if ts:
        try:
            secs = ts / 1e9 if ts > 1e15 else ts / 1e3
            return dt.datetime.fromtimestamp(secs).date().isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return session_day if (day.get("close") or 0) > 0 else None


@st.cache_data(ttl=3600, show_spinner=False)
def realized_vol(ticker: str, window: int = 30) -> float | None:
    """Annualized close-to-close realized volatility from daily bars.

    Uses the aggregates endpoint on the same key — no new data source. This is
    the denominator for IV/RV: it answers whether a fat premium is fat because
    the market is scared, or because the stock actually moves.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=window * 2 + 20)
    try:
        data = _get(f"/v2/aggs/ticker/{ticker}/range/1/day/"
                    f"{start.isoformat()}/{end.isoformat()}",
                    {"adjusted": "true", "sort": "asc", "limit": 400})
    except requests.RequestException:
        return None
    closes = [r.get("c") for r in (data.get("results") or []) if r.get("c")]
    return rv_from_closes(closes, window)


# Earnings: the Massive plan of record does not include a calendar feed
# (Benzinga partner data is not entitled). These probes are tried in order and
# fail quietly; when none answer, the UI says earnings data is unavailable
# rather than implying a candidate is clear of an event.
#
# NOT probed on purpose: /vX/reference/tickers/{t}/events. That is the ticker
# LIFECYCLE feed (ticker_change, name_change, IPO/delisting), not earnings.
# Reading dates from it would answer "yes, we have earnings data" using events
# that are not earnings, which flips the honest "earnings unknown" badge into a
# green "no earnings in window" all-clear — the exact failure this overlay
# exists to prevent. A ticker change dated inside the window would also render
# as a report that does not exist.
EARNINGS_PROBES = (
    ("/v3/reference/earnings", ("results",)),
    ("/benzinga/v1/earnings", ("results",)),
    ("/benzinga/v1/calendar/earnings", ("results",)),
)


def _is_earnings_item(item: dict) -> bool:
    """Defense in depth: if a payload labels its rows, only earnings count."""
    typ = str(item.get("type") or item.get("event_type") or "").lower()
    return ("earning" in typ) if typ else True


@st.cache_data(ttl=86400, show_spinner=False)
def earnings_dates(ticker: str) -> tuple[list[str], str]:
    """(ISO dates, source label). Empty list + "" when no feed is entitled."""
    for path, keys in EARNINGS_PROBES:
        try:
            data = _get(path.format(t=ticker), {"ticker": ticker, "limit": 12})
        except Exception:
            continue
        node = data
        for k in keys:
            node = (node or {}).get(k) if isinstance(node, dict) else node
        if not node:
            continue
        found = []
        for item in (node if isinstance(node, list) else []):
            if not isinstance(item, dict) or not _is_earnings_item(item):
                continue
            d = (item.get("date") or item.get("report_date")
                 or item.get("execution_date") or "")
            if isinstance(d, str) and len(d) >= 10:
                found.append(d[:10])
        if found:
            return sorted(set(found)), path
    return [], ""


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
    key = secret("ANTHROPIC_API_KEY")
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


# ------------------------------------------------------- signature element


def payoff_svg(spot: float, short_k: float, long_k: float, credit: float,
               dte: int, iv: float, live: bool, w: int = 328, h: int = 156,
               otm_lo: float = 0.18, otm_hi: float = 0.22) -> str:
    """Payoff profile with a ±1σ expected-move cone overlaid.

    The cone is the point: a payoff diagram alone can't tell you whether 20%
    OTM is far away — that depends entirely on the name's vol. sigma is the
    lognormal 1-sd move over the holding period, so when the short strike sits
    outside the cone you can see it at a glance.
    """
    width = max(short_k - long_k, 0.01)
    max_profit = credit * 100
    max_loss = max((width - credit) * 100, 1.0)
    be = short_k - credit

    sigma = max(iv, 0.01) * math.sqrt(max(dte, 1) / 365.0)
    cone_lo, cone_hi = spot * math.exp(-sigma), spot * math.exp(sigma)

    pad_l, pad_r, pad_t, pad_b = 8, 8, 24, 20
    lo = max(min(long_k - width * 0.8, cone_lo * 0.97), 0.01)
    hi = max(spot * 1.06, cone_hi * 1.02)
    # headroom above the profit plateau, or its label clips the top edge
    y_hi, y_lo = max(max_profit * 1.7, max_loss * 0.25), -max_loss * 1.12

    def X(p: float) -> float:
        return pad_l + (p - lo) / (hi - lo) * (w - pad_l - pad_r)

    def Y(v: float) -> float:
        return h - pad_b - (v - y_lo) / (y_hi - y_lo) * (h - pad_t - pad_b)

    x0, x1 = X(lo), X(hi)
    y_zero = Y(0)
    # loss leg (flat at max loss, then rising to breakeven), then profit leg
    loss_path = (f"M{x0:.1f},{Y(-max_loss):.1f} L{X(long_k):.1f},{Y(-max_loss):.1f} "
                 f"L{X(be):.1f},{y_zero:.1f}")
    prof_path = (f"M{X(be):.1f},{y_zero:.1f} L{X(short_k):.1f},{Y(max_profit):.1f} "
                 f"L{x1:.1f},{Y(max_profit):.1f}")

    cone_x, cone_w = X(cone_lo), X(cone_hi) - X(cone_lo)
    # The 18-22%-below-spot band the short strike is supposed to live in.
    # Named by PRICE, not by percentage: the larger OTM percentage is the LOWER
    # strike. Getting these backwards makes the rect's width negative, which
    # clamps to 1px and renders the band as an invisible hairline.
    band_lo, band_hi = spot * (1 - otm_hi), spot * (1 - otm_lo)
    strike_in_cone = short_k > cone_lo

    def vline(p: float, cls: str) -> str:
        return (f'<line class="{cls}" x1="{X(p):.1f}" y1="{pad_t - 6:.1f}" '
                f'x2="{X(p):.1f}" y2="{h - pad_b:.1f}" stroke-width="1"/>')

    return f"""
<svg class="payoff" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet"
     style="width:100%;height:auto;display:block;font-size:{11 if w > 400 else 8.5}px"
     role="img" aria-label="Payoff profile with expected move cone">
  <rect class="otm-band fade-in" x="{X(band_lo):.1f}" y="{pad_t - 6:.1f}"
        width="{max(X(band_hi) - X(band_lo), 1):.1f}"
        height="{h - pad_b - pad_t + 6:.1f}" rx="3"/>
  <rect class="cone fade-in" x="{cone_x:.1f}" y="{pad_t - 6:.1f}"
        width="{max(cone_w, 1):.1f}" height="{h - pad_b - pad_t + 6:.1f}" rx="3"/>
  <line class="zero" x1="{x0:.1f}" y1="{y_zero:.1f}" x2="{x1:.1f}"
        y2="{y_zero:.1f}" stroke-width="1"/>
  {vline(spot, 'spot')}
  {vline(short_k, 'kmark')}
  {vline(long_k, 'kmark')}
  <path class="line-loss draw" d="{loss_path}" fill="none" stroke-width="2"
        stroke-linejoin="round"/>
  <path class="line-profit draw" d="{prof_path}" fill="none" stroke-width="2"
        stroke-linejoin="round"/>
  <text class="val" x="{x1:.1f}" y="{Y(max_profit) - 5:.1f}" text-anchor="end"
        fill="#4FD18B">+${max_profit:,.0f}</text>
  <text class="val" x="{x0 + 2:.1f}" y="{Y(-max_loss) - 5:.1f}"
        fill="#FF6B5A">−${max_loss:,.0f}</text>
  <text x="{X(long_k):.1f}" y="{h - 7:.1f}" text-anchor="middle">{long_k:g}</text>
  <text x="{X(short_k):.1f}" y="{h - 7:.1f}" text-anchor="middle">{short_k:g}</text>
  <text x="{X(spot):.1f}" y="{pad_t - 12:.1f}" text-anchor="middle"
        fill="#E8EBF2">spot {spot:,.2f}</text>
  <text x="{cone_x + max(cone_w, 1) / 2:.1f}" y="{h - 7:.1f}" text-anchor="middle"
        fill="{'#FF6B5A' if strike_in_cone else '#6EA8FF'}">±1σ</text>
  <text x="{(X(band_hi) + X(band_lo)) / 2:.1f}" y="{pad_t - 12:.1f}"
        text-anchor="middle" fill="#E0A72C">{otm_lo * 100:.0f}–{otm_hi * 100:.0f}% OTM</text>
</svg>
<div class="payoff-note" style="display:flex;justify-content:space-between;
     padding:2px 8px 2px 8px">
  <span>breakeven <span style="color:#E8EBF2">{be:,.2f}</span></span>
  <span>{'live IV ' + format(iv * 100, '.0f') + '%'
         if live else 'schematic · assumed 35% IV'}</span>
</div>"""


def cone_svg(spot: float, short_k: float, long_k: float, credit: float,
             dte: int, iv: float, rv: float | None, r: float = 0.04,
             q: float = 0.0, w: int = 1200, h: int = 300) -> str:
    """The probability cone: where the stock can actually finish.

    A lognormal density of terminal price at expiration, with the outcome
    regions shaded and their probabilities labelled — so the EV column stops
    being an assertion and becomes something you can see.

    Two curves are drawn: implied vol (what the market charges) and realized
    vol (how the stock has actually moved). The gap between them IS the
    variance risk premium this strategy harvests. When the amber curve sits
    inside the cyan one, the market is paying for more movement than the stock
    has been delivering.
    """
    T = max(dte, 1) / 365.0
    iv = max(float(iv or 0.35), 0.01)
    be = short_k - credit
    pad_l, pad_r, pad_t, pad_b = 12, 12, 40, 42
    lo = min(long_k * 0.86, spot * math.exp((r - q - iv * iv / 2) * T
                                            - 3.2 * iv * math.sqrt(T)))
    hi = spot * math.exp((r - q - iv * iv / 2) * T + 3.2 * iv * math.sqrt(T))
    lo, hi = max(lo, 0.01), max(hi, spot * 1.05)

    def X(p_):
        return pad_l + (p_ - lo) / (hi - lo) * (w - pad_l - pad_r)

    def pdf(x, sigma):
        """Lognormal density of S_T."""
        if x <= 0:
            return 0.0
        mu = math.log(spot) + (r - q - sigma * sigma / 2) * T
        sd = sigma * math.sqrt(T)
        z = (math.log(x) - mu) / sd
        return math.exp(-z * z / 2) / (x * sd * math.sqrt(2 * math.pi))

    N = 180
    xs = [lo + (hi - lo) * i / N for i in range(N + 1)]
    dens_iv = [pdf(x, iv) for x in xs]
    dens_rv = [pdf(x, rv) for x in xs] if (rv and rv > 0) else []
    peak = max(dens_iv + (dens_rv or [0])) or 1.0
    plot_h = h - pad_t - pad_b

    def Y(d):
        return pad_t + plot_h - (d / peak) * plot_h

    def path(dens):
        return " ".join(("M" if i == 0 else "L")
                        + f"{X(x):.1f},{Y(d):.1f}" for i, (x, d) in
                        enumerate(zip(xs, dens)))

    def area(x0, x1, cls):
        pts = [(x, d) for x, d in zip(xs, dens_iv) if x0 <= x <= x1]
        if len(pts) < 2:
            return ""
        seg = " ".join(f"L{X(x):.1f},{Y(d):.1f}" for x, d in pts)
        return (f'<path class="{cls}" d="M{X(pts[0][0]):.1f},{Y(0):.1f} {seg} '
                f'L{X(pts[-1][0]):.1f},{Y(0):.1f} Z"/>')

    # outcome regions, by the same maths the EV column uses
    p_win = prob_otm(spot, short_k, T, r, q, iv)
    p_full = 1 - prob_otm(spot, long_k, T, r, q, iv)
    p_part = max(0.0, 1 - p_win - p_full)

    def vline(x, cls, label, lab_cls="", row=0):
        """`row` staggers the label so adjacent strikes do not collide — the
        long strike, short strike and breakeven can sit within a few dollars
        of each other and their labels overlapped when all shared one line."""
        y_lab = pad_t - 26 if row else pad_t - 12
        return (f'<line class="{cls}" x1="{X(x):.1f}" y1="{pad_t - 8:.1f}" '
                f'x2="{X(x):.1f}" y2="{h - pad_b:.1f}" stroke-width="1"/>'
                f'<text class="{lab_cls}" x="{X(x):.1f}" y="{y_lab:.1f}" '
                f'text-anchor="middle">{label}</text>')

    rv_curve = (f'<path class="curve-rv springy" d="{path(dens_rv)}"/>'
                if dens_rv else "")
    rv_note = (f"realized {rv * 100:.0f}%" if rv else "realized vol unavailable")
    return f"""
<svg class="cone-svg" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet"
     role="img" aria-label="Probability of the stock finishing in each outcome
     region at expiration">
  {area(lo, long_k, 'area-loss')}
  {area(long_k, short_k, 'area-part')}
  {area(short_k, hi, 'area-win')}
  <path class="curve-iv springy" d="{path(dens_iv)}"/>
  {rv_curve}
  {vline(long_k, 'kline', f'long {long_k:g}', '', 1)}
  {vline(short_k, 'kline', f'short {short_k:g}', '', 0)}
  {vline(be, 'kline', f'BE {be:,.2f}', 'belabel', 1)}
  {vline(spot, 'spotline', f'spot {spot:,.2f}', '', 0)}
  <text x="{X((short_k + hi) / 2):.1f}" y="{h - pad_b + 14:.1f}"
        text-anchor="middle" fill="#FFB000">max profit {p_win * 100:.1f}%</text>
  <text x="{X(max(lo, long_k * 0.94)):.1f}" y="{h - pad_b + 14:.1f}"
        text-anchor="start" fill="#E5484D">full loss {p_full * 100:.1f}%</text>
  <!-- the partial band is narrow by construction, so its label drops a row
       rather than colliding with the full-loss label beside it -->
  <text x="{X((long_k + short_k) / 2):.1f}" y="{h - pad_b + 25:.1f}"
        text-anchor="middle" fill="#E5484D">partial {p_part * 100:.1f}%</text>
</svg>
<div class="payoff-note" style="display:flex;justify-content:space-between;
     padding:4px 4px 0 4px">
  <span><span style="color:#4FC3D9">— implied {iv * 100:.0f}%</span>
    &nbsp; <span style="color:#FFB000">-- {esc(rv_note)}</span></span>
  <span>the gap is the variance risk premium</span>
</div>"""


def payoff_panel(title: str, note: str, svg: str) -> str:
    return (f'<div class="payoff-wrap"><div class="payoff-title">'
            f'<span>{esc(title)}</span><span class="payoff-note">{esc(note)}</span>'
            f'</div>{svg}</div>')


# ---------------------------------------------------------------- sidebar

today = dt.date.today()

# Universe lives in a plain session key; the multiselect gets a nonce-suffixed
# key so preset buttons can rewrite the selection without hitting Streamlit's
# "cannot modify a widget key after instantiation" rule.
if "universe" not in st.session_state:
    saved = st.query_params.get("u")
    st.session_state["universe"] = ([t for t in saved.split(",") if t] if saved
                                    else list(DEFAULT_UNIVERSE))
st.session_state.setdefault("uni_nonce", 0)


def set_universe(tickers: list[str]) -> None:
    st.session_state["universe"] = sorted(set(tickers), key=ALL_TICKERS.index
                                          if all(t in ALL_TICKERS for t in tickers)
                                          else str)
    st.session_state["uni_nonce"] += 1


with st.sidebar:
    st.html('<div class="side-brand">Scan setup</div>')

    universe = st.session_state["universe"]
    with st.expander(f"◈  Universe · {len(universe)} names", expanded=True):
        st.html('<p class="helper">Single stocks only — no ETFs. Type to search, '
                'or add a ticker that is not listed.</p>')
        picked = st.multiselect(
            "Tickers", options=ALL_TICKERS, default=universe,
            key=f"uni_ms_{st.session_state['uni_nonce']}",
            accept_new_options=True, label_visibility="collapsed",
            placeholder="Search tickers…",
            help="Removable chips. Anything you type is accepted, so tickers "
                 "outside the presets still work.")
        picked = [t.strip().upper() for t in picked if t.strip()]
        # ETFs are rejected at the point of entry, with the reason, rather than
        # disappearing silently somewhere downstream.
        rejected = [t for t in picked if is_blocked(t)]
        picked = [t for t in picked if not is_blocked(t)]
        if rejected:
            # Rewriting session_state alone does not clear the chip: after first
            # interaction the widget's own key is the source of truth and
            # `default=` is ignored. set_universe() bumps the nonce so the next
            # render builds a fresh widget, and the notice is stashed so it
            # fires exactly once instead of on every later rerun.
            st.session_state["etf_notice"] = rejected
            set_universe(picked)
            st.rerun()
        if picked != universe:
            st.session_state["universe"] = picked
            universe = picked
        notice = st.session_state.pop("etf_notice", None)
        if notice:
            st.html('<p class="helper" style="color:#FF6B5A;margin:6px 0 0 0">'
                    f'<b>{esc(", ".join(notice))}</b> removed — single stocks '
                    'only. Index and leveraged products do not pay for a 20% '
                    'buffer: a diversified index falling 20% is a rarer, far '
                    'more correlated event, and the premium reflects that.</p>')

        # preset baskets — additive/subtractive, active when fully contained
        cols = st.columns(2)
        for i, (name, basket) in enumerate(PRESETS.items()):
            active = bool(basket) and set(basket) <= set(universe)
            slug = name.lower().replace(" ", "_").replace("-", "")
            if cols[i % 2].button(name.upper(),
                                  key=f"preset_{slug}", width="stretch",
                                  help=f"{'Remove' if active else 'Add'} "
                                       f"{len(basket)} tickers"):
                if active:
                    set_universe([t for t in universe if t not in basket])
                else:
                    set_universe(universe + basket)
                st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("CLEAR", key="uni_clear", width="stretch"):
            set_universe([])
            st.rerun()
        if c2.button("RESET", key="uni_default", width="stretch"):
            set_universe(list(DEFAULT_UNIVERSE))
            st.rerun()
        if c3.button("SAVE", key="uni_save", width="stretch",
                     help="Stores the selection in the page URL so it survives "
                          "a refresh and can be bookmarked or shared."):
            st.query_params["u"] = ",".join(universe)
            st.toast("Universe saved to the URL.")

        # sector legend — counts by sector, with the dot colors used everywhere
        by_sec: dict[str, int] = {}
        for t in universe:
            by_sec[sector_of(t)] = by_sec.get(sector_of(t), 0) + 1
        if by_sec:
            st.html('<div class="legend" style="margin-top:10px">' + "".join(
                f'<span class="legend-item"><i class="chip-dot" style="background:'
                f'{SECTOR_META[k][1]}"></i>{SECTOR_META[k][0]} '
                f'<b class="mono" style="color:#E8EBF2">{v}</b></span>'
                for k, v in sorted(by_sec.items(), key=lambda kv: -kv[1])) + '</div>')
        else:
            st.html('<p class="helper" style="color:#FF6B5A;margin:8px 0 0 0">'
                    'No tickers selected — the scan has nothing to do.</p>')

    # ---- expirations
    exp_default = st.session_state.get("exp_mode", "Both")
    with st.expander(f"◷  Expirations · {exp_default}", expanded=True):
        st.html('<p class="helper">Which expiry windows to pull chains for.</p>')
        exp_mode = st.segmented_control(
            "Windows", [WIN_SHORT, WIN_LONG, "Both"], default=exp_default,
            key="exp_mode", label_visibility="collapsed",
            help="Far-dated spreads carry vega risk: a vol spike hurts the "
                 "position before price does.")
        exp_mode = exp_mode or "Both"
        use_short = exp_mode in (WIN_SHORT, "Both")
        use_long = exp_mode in (WIN_LONG, "Both")

        # where those windows fall on a 200-day rail
        span = 200
        bands = ([(28, 46)] if use_short else []) + ([(90, 190)] if use_long else [])
        st.html('<div class="calstrip"><div class="rail"></div>'
                + '<div class="now"></div>'
                + "".join(f'<div class="band" style="left:{a / span * 100:.1f}%;'
                          f'width:{(b - a) / span * 100:.1f}%"></div>'
                          for a, b in bands)
                + '<span class="tick" style="left:0">today</span>'
                + '<span class="tick" style="left:50%">+100d</span>'
                + '<span class="tick" style="left:98%">+200d</span></div>')

        # The trade-off belongs where the choice is made, not only on the
        # results tab you reach afterwards.
        st.html(
            '<div class="regime-note">'
            + ('<span><b>30–45</b> fastest theta per day of risk, so annualized '
               'runs higher — but small credits, twelve redeployments a year, '
               'and only high-IV names pay at 20% OTM.</span>'
               if use_short else '')
            + ('<span><b>90–190</b> bigger premium and one decision — paid for '
               'with months of tied-up capital, slow early theta, and real '
               'vega: a vol spike marks you against long before the stock '
               'nears your strike.</span>' if use_long else '')
            + '</div>')

    # ---- strategy
    otm_state = st.session_state.get("otm_band", (18, 22))
    ml_state = st.session_state.get("risk_cap", 2200)
    with st.expander(f"◆  Strategy · {otm_state[0]}–{otm_state[1]}% OTM · "
                     f"risk ${ml_state:,}", expanded=True):
        st.html('<p class="helper">Where the short strike sits, and how much '
                'collateral a position may risk.</p>')
        otm_band = st.slider("Short strike, % below spot", 10, 35, (18, 22),
                             key="otm_band")
        risk_cap = st.number_input(
            "Risk cap per position", 100, 25000, 2200, 100, key="risk_cap",
            help="Two jobs: a spread whose single contract risks more than this "
                 "is rejected outright, and everything that survives is sized "
                 "to contracts = floor(cap / max loss per contract).")
        min_ror = st.slider("Min return on risk (%)", 0, 40, 5, key="min_ror")
        min_width = st.number_input("Min spread width", 1.0, 50.0, 2.5, 0.5,
                                    key="min_width")
        min_prem = st.number_input(
            "Min net premium", 0, 5000, 200, 25, key="min_prem",
            help="Applied to the POSITION total: credit x 100 x contracts. "
                 "At 20% OTM this thins the 30-45 day sleeve hard, which is "
                 "the informative outcome — only high-IV names clear it.")
        # POP and short delta are two expressions of the same constraint, so
        # moving one shows the other's implied value.
        min_pop = st.slider("Min probability of profit (%)", 50, 99, 80,
                            key="min_pop",
                            help="N(d2) of the short strike.")
        max_delta = st.slider("Max short-leg |delta|", 5, 50, 15, key="max_delta",
                              help="The market's rough proxy for P(ITM). "
                                   "Roughly equivalent to a "
                                   f"{100 - 15}% probability of profit.")
        st.html('<p class="helper" style="margin-top:-4px">'
                f'POP {min_pop}% implies |&#916;| near {(100 - min_pop) / 100:.2f}; '
                f'the delta cap is set to {max_delta / 100:.2f}. '
                'Whichever binds first wins.</p>')

        # live payoff preview — the shape of the worst trade you'd accept
        prev_spot = 100.0
        prev_short = prev_spot * (1 - (otm_band[0] + otm_band[1]) / 200)
        prev_long = max(prev_short - max(min_width, 1.0), 1.0)
        prev_width = prev_short - prev_long
        r = min_ror / 100.0
        prev_credit = max(prev_width * r / (1 + r), 0.01)
        prev_dte = int(((28 + 46) / 2 if use_short else 0) or 0) or 140
        if use_short and use_long:
            prev_dte = 37
        elif use_long and not use_short:
            prev_dte = 140
        raw_html(payoff_panel(
            "Payoff preview", f"{prev_dte}d · spot 100",
            payoff_svg(prev_spot, round(prev_short, 1), round(prev_long, 1),
                       round(prev_credit, 2), prev_dte, 0.35, live=False,
                       w=292, h=150, otm_lo=otm_band[0] / 100,
                       otm_hi=otm_band[1] / 100)))
        st.html('<p class="helper" style="margin-top:8px">Drawn at your minimum '
                'return on risk — the least you would accept, on a $100 stock.</p>')

    # ---- liquidity
    oi_s_state = st.session_state.get("min_oi_s", 300)
    oi_l_state = st.session_state.get("min_oi_l", 100)
    ba_state = st.session_state.get("max_ba", 40)
    with st.expander(f"≋  Liquidity · OI {oi_s_state}/{oi_l_state} · "
                     f"≤{ba_state}% width", expanded=False):
        st.html('<p class="helper">Open-interest floors per leg, and how wide a '
                'bid/ask either leg may quote.</p>')
        min_oi_s = st.number_input("Min OI, short leg", 0, 50000, 300, 50,
                                   key="min_oi_s")
        min_oi_l = st.number_input("Min OI, long leg", 0, 50000, 100, 50,
                                   key="min_oi_l")
        max_ba = st.slider("Max bid/ask width (% of mid)", 5, 100, 40, key="max_ba")
        max_cred_pct = st.slider(
            "Max credit as % of width", 10, 60, 35, key="max_cred_pct",
            help="The sanity gate. A credit worth most of the width at 20% "
                 "OTM is a broken quote, not a market — this is what rejects "
                 "the 15-wide spread that appeared to pay 11.85.")
        max_per_name = st.number_input(
            "Max candidates per underlying", 1, 10, 2, 1, key="max_per_name",
            help="Concentration control, so one high-IV name cannot own the "
                 "table.")
        use_min_ev = st.toggle("Require positive expected value", value=False,
                               key="use_min_ev",
                               help="Opt-in. Uses the realized-vol basis.")

        strict = (min(min_oi_s / 1500, 1) + min(min_oi_l / 800, 1)
                  + min((100 - max_ba) / 80, 1)
                  + min((60 - max_cred_pct) / 50, 1)) / 4
        lit = max(1, round(strict * 5))
        st.html('<div class="meter"><span>loose</span><span class="meter-track">'
                + "".join(f'<i class="meter-seg{" on" if i < lit else ""}"></i>'
                          for i in range(5))
                + '</span><span>strict</span></div>')

    # ---- events
    with st.expander("⚑  Events · news overlay", expanded=False):
        st.html('<p class="helper">Applied to the top names after a scan.</p>')
        flag_earn = st.checkbox("Flag earnings mentions in news", value=True,
                                key="flag_earn")
        has_claude_key = bool(secret("ANTHROPIC_API_KEY"))
        use_claude = st.checkbox("AI news brief per candidate", value=has_claude_key,
                                 disabled=not has_claude_key, key="use_claude",
                                 help=None if has_claude_key else
                                 "Add ANTHROPIC_API_KEY to secrets to enable.")

    # ---- run (sticky: button and its cost line travel together)
    n_win = int(use_short) + int(use_long)
    n_chains = len(universe) * n_win
    with st.container(key="runbox"):
        run = st.button("Run scan", type="primary", width="stretch",
                        key="run_scan", disabled=not universe or n_win == 0)
        st.html(f'<div class="run-cost">{len(universe)} names &nbsp; '
                f'{n_chains} chains &nbsp; est '
                f'{max(1, round(n_chains * 0.45 / 60)):d}–'
                f'{max(2, round(n_chains * 1.1 / 60)):d} min</div>')

SESSION = market_session()
MARKET_OPEN = SESSION["state"] == "open"

params = Params(otm_lo=otm_band[0] / 100, otm_hi=otm_band[1] / 100,
                risk_cap=risk_cap, min_ror=min_ror,
                min_oi_short=min_oi_s, min_oi_long=min_oi_l,
                max_spread_pct=max_ba / 100, min_width=min_width,
                max_credit_pct=max_cred_pct / 100,
                min_net_premium=float(min_prem),
                min_pop=min_pop / 100, max_short_delta=max_delta / 100,
                min_ev=0.0 if use_min_ev else None,
                max_per_underlying=int(max_per_name),
                market_open=MARKET_OPEN)

windows = []
if use_short:
    windows.append((WIN_SHORT, today + dt.timedelta(days=28),
                    today + dt.timedelta(days=46)))
if use_long:
    windows.append((WIN_LONG, today + dt.timedelta(days=90),
                    today + dt.timedelta(days=190)))

# Active preset buttons get an amber outline. Generated per run because "active"
# is derived from membership, not from widget state.
active_rules = [
    f'.st-key-preset_{n.lower().replace(" ", "_").replace("-", "")} button'
    for n, b in PRESETS.items() if b and set(b) <= set(universe)]
if active_rules:
    st.markdown("<style>" + ", ".join(active_rules) +
                "{background:rgba(224,167,44,.12)!important;color:#E0A72C"
                "!important;border-color:#E0A72C!important}</style>",
                unsafe_allow_html=True)

# ---------------------------------------------------------------- top bar

st.html(
    '<div class="topbar">'
    '<span class="wordmark"><i class="lozenge"></i>SPREAD SCOUT</span>'
    '<span class="topbar-chips">'
    f'<span class="chip">names <b>{len(universe)}</b></span>'
    f'<span class="chip">OTM <b>{otm_band[0]}–{otm_band[1]}%</b></span>'
    f'<span class="chip">risk cap <b>${risk_cap:,}</b></span>'
    f'<span class="chip">min RoR <b>{min_ror}%</b></span>'
    f'<span class="chip{"" if use_short else " is-off"}">{WIN_SHORT}</span>'
    f'<span class="chip{"" if use_long else " is-off"}">{WIN_LONG}</span>'
    '</span></div>')

# ---------------------------------------------------------------- scan


def tape_html(entries: list[tuple[str, int, str, str]]) -> str:
    rows = []
    for tk, n, kind, win in entries[-9:]:
        cls, txt = ("hit", f"{n} candidates") if kind == "hit" else (
            ("none", "no match") if kind == "none" else ("err", "fetch failed"))
        mark = "✓" if kind == "hit" else ("·" if kind == "none" else "✕")
        rows.append(f'<div class="tape-row tape-new"><span class="{cls}">{mark}</span>'
                    f'<span class="tk">{esc(tk)}</span>'
                    f'<span style="width:78px;opacity:.6">{esc(win)}</span>'
                    f'<span class="{cls}">{txt}</span></div>')
    return '<div class="tape">' + "".join(rows) + "</div>"


def skeleton_html(n: int = 4, label: str | None = None) -> str:
    """Ghost of a real result row — same columns, so the preview does not
    promise a different table than the one that arrives."""
    head = ('<div class="ghost-head"><span>ticker</span><span>short / long</span>'
            '<span>credit</span><span>qty</span><span>max loss</span>'
            '<span>return on risk</span><span>iv/rv</span>'
            '<span>earnings</span></div>')
    rows = "".join(
        '<div class="ghost-row">'
        '<div class="bone w70"></div><div class="bone w85"></div>'
        '<div class="bone w50"></div><div class="bone w50"></div>'
        '<div class="bone w70"></div>'
        f'<div class="bone-bar"><i style="width:{w}%"></i></div>'
        '<div class="bone w50"></div><div class="bone w50"></div></div>'
        for w in (78, 62, 48, 36, 30)[:n])
    cap = (f'<div class="payoff-note" style="padding:8px 16px 2px">{esc(label)}</div>'
           if label else "")
    return f'<div class="panel" style="padding:0">{head}{rows}{cap}</div>'


def run_scan() -> dict:
    """Fetch chains and build the candidate table. Returns a scan record.

    The data layer is a blocking loop over (ticker × window), so results cannot
    stream into a live table without restructuring it. What streams instead is
    the tape: each name reports as it resolves.
    """
    spots = spot_prices(tuple(universe))
    notes, tape, rejects = [], [], {}
    missing = [t for t in universe if t not in spots]
    if missing:
        notes.append(f"No spot price for {', '.join(missing)} — skipped.")

    all_rows = []
    work = [(t, w) for t in universe if t in spots for w in windows]
    if not work:
        return {"results": pd.DataFrame(), "notes": notes + [
            "No spot prices returned for any ticker — check the data feed."],
            "at": dt.datetime.now().strftime("%b %d · %H:%M"), "params": {},
            "rejects": {}, "session": SESSION}

    session_day = SESSION["last_session"].isoformat()
    ttl_bucket = int(time.time() // (600 if MARKET_OPEN else 3600))

    bar = st.progress(0.0, text="Starting scan…")
    tape_slot, skel_slot = st.empty(), st.empty()
    skel_slot.html(skeleton_html(4, "waiting for the first candidates…"))

    for i, (t, (label, d0, d1)) in enumerate(work):
        bar.progress(i / len(work), text=f"Fetching {t} chains · {label} "
                                         f"· {i + 1} of {len(work)}")
        spot = spots[t]
        k_lo = (1 - params.otm_hi) * spot - risk_cap / 100 - 5
        k_hi = (1 - params.otm_lo) * spot + 1
        try:
            chain = put_chain(t, d0.isoformat(), d1.isoformat(), k_lo, k_hi,
                              MARKET_OPEN, session_day, ttl_bucket)
        except requests.HTTPError as e:
            notes.append(f"{t}: chain fetch failed ({e.response.status_code}).")
            tape.append((t, 0, "err", label))
            tape_slot.html(tape_html(tape))
            continue
        if chain.empty:
            tape.append((t, 0, "none", label))
            tape_slot.html(tape_html(tape))
            continue
        dte_of = {e: (dt.date.fromisoformat(e) - today).days
                  for e in chain["exp"].unique()}
        rv = realized_vol(t)          # needed before pricing: EV uses it
        spreads = build_spreads(chain, spot, dte_of, params, rv=rv,
                                rejects=rejects)
        if not spreads.empty:
            spreads.insert(1, "window", label)
            all_rows.append(spreads)
            tape.append((t, len(spreads), "hit", label))
        else:
            tape.append((t, 0, "none", label))
        tape_slot.html(tape_html(tape))

    df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    earn_source = ""
    if not df.empty:
        names = sorted(df["ticker"].unique())
        bar.progress(1.0, text=f"Events · {len(names)} names")
        earn_map = {}
        for t in names:
            dates, src = earnings_dates(t)
            earn_map[t] = dates
            earn_source = earn_source or src
        df["earnings_in_window"] = [
            earnings_in_window(earn_map.get(t, []), today,
                               dt.date.fromisoformat(e))
            for t, e in zip(df["ticker"], df["exp"])]
        df["earnings_known"] = bool(earn_source)
        if not earn_source:
            notes.append("No earnings calendar is entitled on this data plan — "
                         "event risk is unverified, not absent.")

    bar.empty(); tape_slot.empty(); skel_slot.empty()
    return {"results": df, "notes": notes, "earn_source": earn_source,
            "at": dt.datetime.now().strftime("%b %d · %H:%M"),
            "rejects": rejects, "session": SESSION,
            "params": {"min_ror": min_ror, "otm": tuple(otm_band),
                       "risk_cap": risk_cap, "min_oi_s": min_oi_s,
                       "max_ba": max_ba, "min_width": min_width,
                       "min_prem": min_prem, "min_pop": min_pop,
                       "max_cred_pct": max_cred_pct,
                       "max_per_name": int(max_per_name)}}


if run:
    st.session_state["scan"] = run_scan()
    st.session_state["metrics_seen"] = None   # re-arm the count-up animation

scan = st.session_state.get("scan")

# ------------------------------------------------------------ pre-scan state

if scan is None:
    st.html(
        '<div class="hero">'
        '<p class="eyebrow rise">ready</p>'
        '<p class="lede rise d1">Deep OTM premium, screened.</p>'
        '<p class="sub rise d2">Sell puts far below spot, cap the risk with a '
        'long leg, and keep only what clears your floor. Set the box on the '
        'left, then run the scan.</p></div>'
        '<div class="rise d3"><p class="section-title">What a candidate looks '
        'like</p></div>')
    st.html('<div class="rise d4">' + skeleton_html(5) + '</div>')
    st.html(
        '<div class="legend rise d5" style="row-gap:10px">'
        '<span class="legend-item"><i class="legend-swatch" style="background:'
        '#FFB000"></i>return on risk, ranked within a sleeve</span>'
        '<span class="legend-item"><span class="badge badge-win">POP</span>'
        'N(d2) — the chance the short strike expires worthless</span>'
        '<span class="legend-item"><span class="badge badge-mid">mid</span>'
        'both legs quoted, priced off the midpoint</span>'
        '<span class="legend-item"><span class="badge badge-last">rejected</span>'
        'a leg had no market, or the credit was implausible for the width — '
        'refused before ranking, never shown as a candidate</span>'
        '<span class="legend-item"><span class="badge badge-last">⚑ 1</span>'
        'earnings reports scheduled inside the window</span>'
        '<span class="legend-item"><span class="badge badge-win">b/a 38%</span>'
        'widest leg as a share of mid — wide markets cost you the edge</span>'
        '</div>')
    st.stop()

results, sparams = scan["results"], scan.get("params", {})
for note in scan["notes"]:
    st.html(f'<p class="helper" style="margin:0 0 6px 0">{esc(note)}</p>')

# --------------------------------------------------------- empty-results state

if results.empty:
    p_ror = sparams.get("min_ror", min_ror)
    p_otm = sparams.get("otm", tuple(otm_band))
    p_oi = sparams.get("min_oi_s", min_oi_s)
    # Name the missing high-IV names specifically — the reason a 20% buffer
    # pays nothing is almost always that the universe is too calm, not that
    # the filters are too tight.
    calm = [t for t in ("TSLA", "COIN", "MSTR", "PLTR", "AMD", "MU")
            if t not in universe][:3] or ["TSLA", "COIN", "MSTR"]
    st.html(
        '<div class="hero"><p class="eyebrow">no matches</p>'
        f'<p class="lede">Nothing cleared a {p_ror}% return on risk at '
        f'{p_otm[0]}–{p_otm[1]}% OTM.</p>'
        f'<p class="sub">High-IV names pay at this distance; calm mega caps '
        f"don't. You screened {len(universe)} names with a {p_oi}-contract "
        'open-interest floor — at short DTE a 20% one-month drop is priced at '
        'close to nothing unless the market already expects the stock to move '
        'that far.</p></div>'
        '<p class="section-title">Try one of these</p>'
        '<div class="legend">'
        f'<span class="chip">widen OTM to <b>{max(10, p_otm[0] - 4)}–'
        f'{p_otm[1]}%</b></span>'
        f'<span class="chip">lower min RoR to <b>{max(0, p_ror - 3)}%</b></span>'
        f'<span class="chip">drop OI floor to <b>{min(max(p_oi // 2, 0), 250)}</b>'
        f'</span>'
        f'<span class="chip">add <b>{esc(" · ".join(calm))}</b></span>'
        '</div>')
    st.stop()

# ---------------------------------------------------------------- metrics


def metrics_html(df: pd.DataFrame, rejected: int, animate: bool) -> str:
    """Strip above the table. `rejected` is how many candidate pairs the sanity
    gate refused to construct — surfaced, never silently dropped."""
    med = df["ror_pct"].median() if len(df) else 0
    best = df["ror_pct"].max() if len(df) else 0
    best_ann = df["annualized_pct"].max() if len(df) else 0
    pop_med = 100 * df["pop"].median() if df.get("pop") is not None \
        and df["pop"].notna().any() else 0
    ev_best = df["ev_adj"].max() if df.get("ev_adj") is not None \
        and df["ev_adj"].notna().any() else 0
    with_earn = int((df["earnings_in_window"] > 0).sum()) \
        if "earnings_in_window" in df.columns else 0
    tiles = [("candidates", len(df), 0, "", ""),
             ("median POP", pop_med, 1, "", "%"),
             ("median RoR", med, 1, "", "%"),
             ("best RoR", best, 1, "", "%"),
             ("best EV (RV basis)", ev_best, 0, "$", ""),
             ("earnings in window", with_earn, 0, "", ""),
             ("rejected by gate", rejected, 0, "", "")]
    cards = ""
    for label, val, dec, pre, suf in tiles:
        shown = f"{pre}{val:,.{dec}f}{suf}"
        inner = (f'<span class="cu" data-to="{val:.4f}" data-dec="{dec}" '
                 f'data-pre="{pre}" data-suf="{suf}">{pre}0{suf}</span>'
                 if animate else shown)
        cards += (f'<div class="panel panel-tight hover-lift">'
                  f'<div class="kv-k">{label}</div>'
                  f'<div class="kv-v" style="font-size:22px">{inner}</div></div>')

    # return-on-risk distribution — one number hides the shape of the set
    bins, lo, hi = 26, float(df["ror_pct"].min()), float(df["ror_pct"].max())
    span = max(hi - lo, 1e-9)
    counts = [0] * bins
    for v in df["ror_pct"]:
        counts[min(int((v - lo) / span * bins), bins - 1)] += 1
    peak = max(counts) or 1
    med_ror = float(df["ror_pct"].median())
    med_pct = 100 * (med_ror - lo) / span
    bars = "".join(
        f'<i style="flex:1 1 0;min-width:2px;'
        f'height:{max(3, c / peak * 100):.0f}%;background:'
        f'{"#FFB000" if i > bins * 0.6 else "#6E4D00"}"></i>'
        for i, c in enumerate(counts))
    spark = (
        f'<div class="panel panel-tight" style="grid-column:1/-1;border-right:0">'
        f'<div class="kv-k">return on risk · distribution · '
        f'{len(df):,} candidates</div>'
        f'<div style="position:relative;display:flex;align-items:flex-end;'
        f'gap:2px;height:38px;margin-top:6px">{bars}'
        f'<span style="position:absolute;left:{med_pct:.1f}%;top:-2px;bottom:0;'
        f'width:1px;background:#E8E6E1;opacity:.75"></span>'
        f'<span style="position:absolute;left:{med_pct:.1f}%;top:-12px;'
        f'transform:translateX(-50%);font-family:var(--mono);font-size:8.5px;'
        f'color:#E8E6E1">median {med_ror:.1f}%</span></div>'
        f'<div class="payoff-note" style="display:flex;'
        f'justify-content:space-between;margin-top:3px">'
        f'<span>{lo:.1f}% &nbsp;lowest</span>'
        f'<span>count per bin</span>'
        f'<span>highest&nbsp; {hi:.1f}%</span></div></div>')

    # The script is extracted and run by Streamlit, which can fire it before the
    # markup lands in the DOM — hence the retry rather than a bare query.
    js = """
<script>(function(){
  var reduce=window.matchMedia('(prefers-reduced-motion: reduce)').matches, tries=0;
  function go(){
    var els=document.querySelectorAll('.cu:not([data-done])');
    if(els.length===0){ tries=tries+1; if(tries>40){return;} setTimeout(go,40); return; }
    els.forEach(function(el){
      el.setAttribute('data-done','1');
      var to=parseFloat(el.dataset.to), dec=+el.dataset.dec,
          pre=el.dataset.pre||'', suf=el.dataset.suf||'';
      var fmt=function(v){return pre+v.toLocaleString(undefined,
        {minimumFractionDigits:dec,maximumFractionDigits:dec})+suf;};
      if(reduce || document.visibilityState==='hidden'){ el.textContent=fmt(to); return; }
      var t0=performance.now(), D=620;
      /* rAF is suspended in a hidden/throttled tab; this guarantees the final
         value lands even if the animation frames never arrive. */
      setTimeout(function(){ el.textContent=fmt(to); }, D+140);
      (function step(t){
        var p=Math.min(1,(t-t0)/D), e=1-Math.pow(1-p,3);
        el.textContent=fmt(to*e);
        if(p===1){ el.textContent=fmt(to); } else { requestAnimationFrame(step); }
      })(t0);
    });
  }
  go();})();</script>""" if animate else ""
    return f'<div class="metric-strip">{cards}{spark}</div>{js}'


# Stale-priced rows are excluded by default; the count is reported, never
# silently dropped.
# Everything that survives the sanity gate is a real candidate now — stale
# marks are rejected at construction rather than filtered afterwards.
shown = results
sess = scan.get("session") or {}
rejects = scan.get("rejects") or {}
n_rejected = sum(rejects.values())

if sess.get("state") == "closed":
    nxt = sess.get("next_open")
    nxt_txt = nxt.strftime("%a %-d %b, %-I:%M %p ET") if nxt else "the next session"
    last = sess.get("last_session")
    st.html(
        '<div class="banner banner-info">'
        '<span class="banner-k">Markets closed</span>'
        f'<span>Quotes are {last.strftime("%A %-d %b") if last else "the last session"}'
        "'s close. Results are for planning, not execution — verify against a "
        'live quote before trading.</span>'
        f'<span class="banner-x">Next open {esc(nxt_txt)}</span></div>')
else:
    st.html('<div class="banner banner-live"><span class="banner-k">Live</span>'
            '<span>Priced from two-sided quotes. Strikes without a live market '
            'are rejected, not ranked.</span></div>')

animate = st.session_state.get("metrics_seen") != scan["at"]
st.html(metrics_html(shown, n_rejected, animate), unsafe_allow_javascript=animate)
st.session_state["metrics_seen"] = scan["at"]

if rejects:
    with st.expander(f"Sanity gate rejected {n_rejected:,} candidate pairs"):
        st.html('<p class="helper">Every pair the gate refused to construct, '
                'by reason. These are not hidden results — they are pairs '
                'whose quotes did not describe a tradable spread.</p>'
                '<div class="ledger">' + "".join(
                    f'<span>{esc(k)}<b>{v:,}</b></span>'
                    for k, v in sorted(rejects.items(), key=lambda kv: -kv[1]))
                + '</div>')

# ---------------------------------------------------------------- results


def dot_uri(color: str) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14">'
           f'<circle cx="7" cy="7" r="4.5" fill="{color}"/></svg>')
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def order_ticket(row: pd.Series) -> str:
    n = int(row.get("contracts") or 1)
    return (f"SELL -{n} {row['ticker']} {row['exp']} {row['short_k']:g}P / "
            f"BUY +{n} {row['long_k']:g}P  @ {row['credit']:.2f} CR   "
            f"[{n} × ${row['max_loss']:,.0f} risk = ${row['total_risk']:,.0f} · "
            f"credit ${row['total_credit']:,.0f} · {int(row['dte'])}d]")


QUICK = {
    "POP ≥ 85%": lambda d: d["pop"].fillna(0) >= 0.85,
    "No earnings in window": lambda d: d.get("earnings_in_window",
                                             pd.Series(0, index=d.index)) == 0,
    "Tight markets only": lambda d: (d["short_ba_pct"].fillna(999) <= 20)
                                    & (d["long_ba_pct"].fillna(999) <= 20),
    "Width ≥ $5": lambda d: d["width"] >= 5,
    "Positive EV (RV)": lambda d: d["ev_adj"].fillna(-1e9) > 0,
}

# Eight columns drive the decision; the rest are opt-in via the column control.
CORE_COLS = ["ticker", "legs", "dte", "credit", "max_loss", "pop",
             "ev_per_collateral", "ror_pct"]
EXTRA_COLS = ["spot", "exp", "pct_otm", "width", "credit_pct_width",
              "contracts", "total_credit", "annualized_pct", "short_delta",
              "p_touch", "ev_adj", "ev_rn", "short_iv", "rv30", "iv_minus_rv",
              "earn", "ba_worst", "short_oi", "long_oi", "priced"]


def table_config(disp: pd.DataFrame, cross: bool) -> dict:
    ror_cap = float(disp["ror_pct"].quantile(0.95)) if len(disp) else 1.0
    return {
        "pop": st.column_config.NumberColumn(
            "POP", format="%.1f%%",
            help="N(d2): probability the short strike expires out of the "
                 "money. Not the same as 1 - |delta|."),
        "ev_per_collateral": st.column_config.NumberColumn(
            "EV / $ risk", format="%.3f",
            help="Expected value per dollar of collateral, on the REALIZED-vol "
                 "basis. Priced with chain IV, EV is ~0 by construction; the "
                 "gap is the variance risk premium this strategy harvests."),
        "ev_adj": st.column_config.NumberColumn(
            "EV (RV basis)", format="$%.0f",
            help="Assumes the terminal distribution follows realized vol, not "
                 "implied. That assumption is the edge — and it is an "
                 "assumption."),
        "ev_rn": st.column_config.NumberColumn(
            "EV (IV basis)", format="$%.0f",
            help="Risk-neutral baseline. Expect ~0 minus costs."),
        "p_touch": st.column_config.NumberColumn(
            "P(touch)", format="%.1f%%",
            help="Approximate: 2 x N(-d2), driftless."),
        "short_delta": st.column_config.NumberColumn("Δ short", format="%.3f"),
        "credit_pct_width": st.column_config.NumberColumn(
            "Cr/width", format="%.0f%%",
            help="Credit as a share of width. The sanity gate lives here."),
        "iv_minus_rv": st.column_config.NumberColumn(
            "IV−RV", format="%.2f",
            help="Positive means implied exceeds realized — the premium is "
                 "richer than how the stock has actually moved."),
        "ticker": st.column_config.TextColumn("Ticker", width="small"),
        "spot": st.column_config.NumberColumn("Spot", format="$%.2f"),
        "exp": st.column_config.TextColumn("Expiry", width="small"),
        "dte": st.column_config.NumberColumn("DTE", format="%d", width="small"),
        "legs": st.column_config.TextColumn("Short / long",
                                            help="Short strike / long strike"),
        "pct_otm": st.column_config.NumberColumn("% OTM", format="%.1f%%"),
        "width": st.column_config.NumberColumn("Width", format="$%.2f"),
        "credit": st.column_config.NumberColumn("Credit", format="$%.2f"),
        "max_loss": st.column_config.NumberColumn(
            "Max loss", format="$%.2f",
            help="Per contract — the collateral at risk. Qty × this = total risk."),
        "contracts": st.column_config.NumberColumn(
            "Qty", format="%d", help="floor(risk cap / max loss per contract)"),
        "total_credit": st.column_config.NumberColumn(
            "Total credit", format="$%d", help="Credit collected at that size"),
        "ror_pct": st.column_config.ProgressColumn(
            "Return on risk", format="%.1f%%", min_value=0.0,
            max_value=float(max(disp["ror_pct"].max(), 1))),
        "annualized_pct": (
            st.column_config.ProgressColumn(
                "Annualized", format="%.0f%%", min_value=0.0,
                max_value=float(max(disp["annualized_pct"].max(), 1)))
            if cross else
            st.column_config.NumberColumn("Annualized", format="%.0f%%")),
        "earn": st.column_config.TextColumn(
            "Earnings", width="small",
            help="Reports scheduled between today and expiration"),
        "iv_rv": st.column_config.NumberColumn(
            "IV/RV", format="%.2f",
            help="Implied vs realized vol. Near 1.0 means the premium is "
                 "priced for how much this name actually moves."),
        "short_iv": st.column_config.NumberColumn("IV", format="%.2f"),
        "rv30": st.column_config.NumberColumn("RV30", format="%.2f",
                                              help="30-day realized volatility"),
        "ba_worst": st.column_config.NumberColumn(
            "B/A width", format="%.0f%%",
            help="Widest of the two legs, as a share of mid"),
        "short_oi": st.column_config.NumberColumn("OI short", format="%d"),
        "long_oi": st.column_config.NumberColumn("OI long", format="%d"),
        "priced": st.column_config.TextColumn(
            "Priced", width="small",
            help="mid = bid/ask midpoint · LAST = stale last trade"),
    }


def exposure_tray(sel: pd.DataFrame) -> str:
    """Correlated-loss profile of a basket, made visible.

    Losses in this strategy arrive in crashes: clustered, and correlated across
    positions. Concentration is the thing that turns a bad month into a bad
    year, so it is shown next to the total.
    """
    total_risk = float(sel["total_risk"].sum())
    total_credit = float(sel["total_credit"].sum())
    by_sec: dict[str, float] = {}
    for t, risk in zip(sel["ticker"], sel["total_risk"]):
        k = SECTOR_META[sector_of(t)][0]
        by_sec[k] = by_sec.get(k, 0) + float(risk)
    top = max(by_sec.items(), key=lambda kv: kv[1]) if by_sec else ("—", 0)
    conc = 100 * top[1] / total_risk if total_risk else 0
    bars = "".join(
        f'<span style="display:flex;justify-content:space-between;gap:12px">'
        f'<span>{esc(k)}</span><span class="mono" style="color:#E8EBF2">'
        f'${v:,.0f}</span></span>'
        for k, v in sorted(by_sec.items(), key=lambda kv: -kv[1]))
    warn = ('<span class="badge badge-last" style="margin-left:8px">'
            'concentrated</span>' if conc >= 60 and len(sel) > 1 else "")
    return (
        '<div class="panel" style="margin-top:14px"><div class="kv">'
        f'<div class="kv-item"><div class="kv-k">spreads selected</div>'
        f'<div class="kv-v">{len(sel)}</div></div>'
        f'<div class="kv-item"><div class="kv-k">total risk</div>'
        f'<div class="kv-v risk">${total_risk:,.0f}</div></div>'
        f'<div class="kv-item"><div class="kv-k">total credit</div>'
        f'<div class="kv-v credit">${total_credit:,.0f}</div></div>'
        f'<div class="kv-item"><div class="kv-k">largest sector</div>'
        f'<div class="kv-v">{conc:.0f}%{warn}</div></div>'
        f'</div><div class="payoff-note" style="margin-top:12px;display:grid;'
        f'gap:4px;max-width:340px">{bars}</div></div>')


def render_results(df: pd.DataFrame, slot: str, cross: bool,
                   scan_otm: tuple[int, int]) -> None:
    """One regime sleeve (or the cross-regime comparison).

    Ranking is the strategy's, not the table's: raw return on risk inside a
    regime, simple annualized across them. Sorting one mixed list by raw RoR
    would bury every 30-45 DTE candidate, because far-dated spreads collect a
    bigger credit for the same 20% buffer.
    """
    if df.empty:
        st.html('<div class="panel"><p class="sub">Nothing in this sleeve.</p></div>')
        return
    metric = st.session_state.get("rank_metric", "EV per $ collateral")
    if cross:
        ranked = rank_cross_regime(df)
    else:
        col, asc = RANK_METRICS.get(metric, ("ror_pct", False))
        if col in df.columns and df[col].notna().any():
            ranked = df.sort_values(col, ascending=asc,
                                    na_position="last").reset_index(drop=True)
        else:
            ranked = rank_within_regime(df)
    alternates = ranked
    ranked = dedupe(ranked, "Annualized RoR" if cross else metric,
                    scan.get("params", {}).get("max_per_name", 2))

    st.html(
        '<p class="payoff-note" style="margin:2px 0 10px 2px">'
        + ('Ranked by <b>simple annualized</b> return on risk, not compounded.'
           if cross else
           f'Ranked by <b>{esc(metric.lower())}</b> within this sleeve.') + '</p>')

    picks = st.pills("Quick filters", list(QUICK), selection_mode="multi",
                     key=f"qf_{slot}", label_visibility="collapsed")
    c1, c2 = st.columns([1, 1])
    tick_pick = c1.multiselect("Tickers", sorted(ranked["ticker"].unique()),
                               key=f"tk_{slot}", placeholder="All tickers",
                               label_visibility="collapsed")
    show_all = c2.toggle("All columns", value=False, key=f"cols_{slot}",
                         help="Default shows the eight columns that drive a "
                              "decision.")
    view = ranked
    for q in picks or []:
        try:
            view = view[QUICK[q](view)]
        except KeyError:
            pass
    if tick_pick:
        view = view[view["ticker"].isin(tick_pick)]

    if view.empty:
        st.html('<div class="panel"><p class="sub">No candidates match these '
                'filters. Clear one to widen the set.</p></div>')
        return

    disp = view.copy()
    disp["sec"] = [SECTOR_META[sector_of(t)][0][:2].upper()
                   for t in disp["ticker"]]
    disp["legs"] = (disp["short_k"].map("{:g}".format) + " / "
                    + disp["long_k"].map("{:g}".format))
    disp["priced"] = disp["pricing"].map({"mid": "mid", "close": "close"})
    # fixed decimals within a column — "percent" auto-format gave 81.2% next
    # to 95.01% in the same column
    for c, mult in (("pop", 100), ("p_touch", 100)):
        if c in disp.columns:
            disp[c] = disp[c] * mult
    disp["ba_worst"] = disp[["short_ba_pct", "long_ba_pct"]].max(axis=1)
    if "earnings_in_window" in disp.columns:
        known = bool(disp.get("earnings_known", pd.Series([False])).any())
        disp["earn"] = [("—" if not known else ("none" if n == 0 else f"⚑ {int(n)}"))
                        for n in disp["earnings_in_window"]]
    else:
        disp["earn"] = "—"
    for c in ("iv_rv", "rv30"):
        if c not in disp.columns:
            disp[c] = None

    cols = [c for c in (CORE_COLS + EXTRA_COLS if show_all else CORE_COLS)
            if c in disp.columns]
    event = st.dataframe(
        disp[cols], width="stretch", hide_index=True, height=460,
        key=f"tbl_{slot}", on_select="rerun", selection_mode="multi-row",
        column_config=table_config(disp, cross))

    st.html(f'<p class="payoff-note" style="margin:6px 0 0 2px">'
            f'{len(view):,} of {len(ranked):,} shown &nbsp;&nbsp; '
            f'best per underlying and expiry &nbsp;&nbsp; '
            f'select rows for the ticket and aggregate exposure</p>')

    rows = list(event.selection.rows) if event and event.selection else []
    if rows:
        sel = view.iloc[rows]
        if len(sel) > 1:
            st.html('<p class="section-title" style="margin-top:18px">'
                    'Selected exposure</p>')
            st.html(exposure_tray(sel))
        row_detail(sel.iloc[0], slot, scan_otm)

    st.download_button("Download CSV", ranked.to_csv(index=False),
                       f"spread_scout_{slot}_{today}.csv", "text/csv",
                       key=f"dl_{slot}", icon=":material/download:")


def row_detail(r: pd.Series, slot: str, scan_otm: tuple[int, int]) -> None:
    """`scan_otm` is the band the scan ran with. Reading the live slider here
    would redraw the band under a strike that was chosen at a different one."""
    iv = float(r["short_iv"]) if pd.notna(r.get("short_iv")) else 0.35
    spot = float(r["spot"]) if pd.notna(r.get("spot")) else \
        float(r["short_k"]) / (1 - r["pct_otm"] / 100)
    st.html('<p class="section-title" style="margin-top:20px">'
            f'{esc(r["ticker"])} · {esc(r["exp"])} · {r["short_k"]:g}/'
            f'{r["long_k"]:g} put spread</p>')
    rv = float(r["rv30"]) if pd.notna(r.get("rv30")) else None
    raw_html(payoff_panel(
        "Probability cone · terminal price at expiration",
        f"{int(r['dte'])}d · P(max profit) {100 * float(r['pop'] or 0):.1f}%",
        cone_svg(spot, float(r["short_k"]), float(r["long_k"]),
                 float(r["credit"]), int(r["dte"]), iv, rv)))
    left, right = st.columns([1.05, 1])
    with left:
        raw_html(payoff_panel(
            "Payoff at expiration",
            f"{int(r['dte'])}d · {SECTOR_META[sector_of(r['ticker'])][0]}",
            payoff_svg(spot, float(r["short_k"]), float(r["long_k"]),
                       float(r["credit"]), int(r["dte"]), iv,
                       live=pd.notna(r.get("short_iv")), w=600, h=232,
                       otm_lo=scan_otm[0] / 100, otm_hi=scan_otm[1] / 100)))
    with right:
        badge = ('<span class="badge badge-last">last-trade priced</span>'
                 if r["pricing"] == "last"
                 else '<span class="badge badge-mid">midpoint priced</span>')
        n_earn = int(r.get("earnings_in_window") or 0)
        known = bool(r.get("earnings_known"))
        if not known:
            ebadge = ('<span class="badge badge-win">earnings unknown</span>')
        elif n_earn:
            ebadge = (f'<span class="badge badge-last">⚑ {n_earn} report'
                      f'{"s" if n_earn > 1 else ""} in window</span>')
        else:
            ebadge = '<span class="badge badge-mid">no earnings in window</span>'
        st.html(
            '<div class="panel"><div class="kv">'
            f'<div class="kv-item"><div class="kv-k">credit</div>'
            f'<div class="kv-v credit">${r["credit"]:.2f}</div></div>'
            f'<div class="kv-item"><div class="kv-k">max loss / contract</div>'
            f'<div class="kv-v risk">${r["max_loss"]:,.0f}</div></div>'
            f'<div class="kv-item"><div class="kv-k">breakeven</div>'
            f'<div class="kv-v">{r["breakeven"]:,.2f}</div></div>'
            f'<div class="kv-item"><div class="kv-k">return on risk</div>'
            f'<div class="kv-v">{r["ror_pct"]:.1f}%</div></div>'
            f'<div class="kv-item"><div class="kv-k">credit / width</div>'
            f'<div class="kv-v">{100 * float(r["credit"]) / float(r["width"]):.0f}%'
            f'</div></div>'
            f'<div class="kv-item"><div class="kv-k">at the cap</div>'
            f'<div class="kv-v">{int(r["contracts"])} × '
            f'${r["total_credit"]:,.0f}</div></div>'
            '</div>'
            '<div class="payoff-note" style="margin-top:14px;display:grid;gap:3px">'
            f'<span>short {r["short_k"]:g}P &nbsp; '
            f'{_q(r.get("short_bid"))} / {_q(r.get("short_ask"))} &nbsp; '
            f'OI {int(r["short_oi"]):,}</span>'
            f'<span>long &nbsp;{r["long_k"]:g}P &nbsp; '
            f'{_q(r.get("long_bid"))} / {_q(r.get("long_ask"))} &nbsp; '
            f'OI {int(r["long_oi"]):,}</span></div>'
            f'<div style="margin-top:12px">{badge} {ebadge}</div></div>')
    st.html('<p class="kv-k" style="margin:14px 0 4px 2px">order ticket</p>')
    st.code(order_ticket(r), language=None, wrap_lines=True)


def _q(v) -> str:
    return f"{float(v):.2f}" if v is not None and pd.notna(v) else "—"


# The two sleeves are separate products with different risks, so they get
# separate tables. The cross-regime tab is the only place they are compared,
# and only on annualized return.
scan_otm = tuple(sparams.get("otm", tuple(otm_band)))

rc1, rc2 = st.columns([1.2, 3])
rc1.selectbox("Rank by", list(RANK_METRICS), key="rank_metric",
              help="EV per $ collateral is the default because it is the only "
                   "metric that does not systematically favour one sleeve.")
present = [w for w in (WIN_SHORT, WIN_LONG) if w in set(shown["window"])]
labels = list(present)
if len(present) > 1:
    labels.append("Cross-regime")
labels.append("News & events")
tabs = st.tabs(labels)

REGIME_NOTE = {
    WIN_SHORT: ("Theta decays fastest here, so annualized returns run higher — "
                "but credits are small, you redeploy twelve times a year, and "
                "at 20% OTM only high-IV names pay anything at all."),
    WIN_LONG:  ("Bigger absolute premium and one decision instead of twelve — "
                "paid for with months of tied-up capital, slow early theta, and "
                "real vega risk: a volatility spike marks the position against "
                "you long before the stock approaches your strike."),
}

for i, w in enumerate(present):
    with tabs[i]:
        st.html(f'<p class="section-title">{esc(w)}</p>'
                f'<p class="sub">{REGIME_NOTE[w]}</p>')
        render_results(shown[shown["window"] == w], slot=w[:6], cross=False,
                       scan_otm=scan_otm)

if len(present) > 1:
    with tabs[len(present)]:
        st.html('<p class="section-title">Cross-regime</p>'
                '<p class="sub">Both sleeves on one annualized axis.</p>')
        with st.popover("What simple annualization assumes"):
            st.html('<p class="sub">RoR × 365 ÷ DTE, not compounded. It '
                    'assumes you redeploy at the same terms all year — which '
                    'the 30–45 sleeve has to earn twelve times over, and the '
                    'far-dated sleeve does not have to earn at all. It is the '
                    'only way to line the two up, and it flatters the short '
                    'sleeve. Read it as a comparison, not a forecast.</p>')
        render_results(shown, slot="cross", cross=True, scan_otm=scan_otm)

tab_news = tabs[-1]

# ------------------------------------------------------------ news overlay

CHIP_CLASS = {"positive": "badge-mid", "negative": "badge-last",
              "neutral": "badge-win"}

with tab_news:
    # run_scan no longer returns a globally sorted frame (ranking moved into
    # the per-sleeve tables), so rank here explicitly rather than inheriting an
    # order that no longer exists — otherwise this is just universe order.
    top_names = list(rank_cross_regime(shown)["ticker"].drop_duplicates().head(8))
    st.html('<p class="section-title">Coverage on the top names</p>'
            f'<p class="sub" style="margin-bottom:10px">Recent articles and '
            f'sentiment for the {len(top_names)} highest-ranked underlyings.</p>')
    for t in top_names:
        try:
            arts = ticker_news(t)
        except requests.RequestException as e:
            st.html(f'<p class="helper">{esc(t)}: news fetch failed ({esc(e)}).</p>')
            continue
        counts = pd.Series([a["sentiment"] for a in arts]).value_counts().to_dict()
        earn = flag_earn and earnings_mentioned(arts)
        with st.expander(f"{t} — {len(arts)} recent articles"
                         + ("  ·  ⚑ earnings in news flow" if earn else "")):
            chips = "".join(
                f'<span class="badge {CHIP_CLASS.get(k, "badge-win")}" '
                f'style="margin-right:6px">{esc(k)} {v}</span>'
                for k, v in counts.items())
            st.html(chips or '<span class="badge badge-win">no tagged sentiment</span>')
            if earn:
                st.warning("Earnings-related coverage detected. Verify the exact "
                           "report date before opening any expiration that "
                           "crosses it.")
            if use_claude:
                brief = claude_brief(t, arts)
                if brief:
                    st.markdown(f"**AI risk read** — {brief}")
            st.html("".join(
                f'<div class="news-row"><span class="news-date">'
                f'{esc(a["published"])}</span>'
                f'<span class="badge {CHIP_CLASS.get(a["sentiment"], "badge-win")}">'
                f'{esc(a["sentiment"])}</span>'
                f'<a href="{esc(a["url"])}" target="_blank" rel="noopener">'
                f'{esc(a["title"])}</a></div>' for a in arts[:6]))

st.html('<p class="payoff-note" style="margin-top:18px">Screening tool, not '
        'investment advice. Nothing here is a recommendation — a candidate is '
        'a spread that survived a filter, and quotes move.</p>')

# Keyboard layer. `/` focuses search, j/k walk rows, enter expands, r runs.
# Streamlit owns the grid, so j/k drive its own scroller rather than a custom
# selection model — the row click remains the source of truth for selection.
st.html("""
<div class="keybar">
  <span><kbd>/</kbd>search</span><span><kbd>j</kbd><kbd>k</kbd>row</span>
  <span><kbd>&#8629;</kbd>expand</span><span><kbd>r</kbd>run scan</span>
  <span class="spacer">SPREAD SCOUT</span>
</div>
<script>(function(){
  if (window.__ss_keys) { return; }
  window.__ss_keys = true;
  var typing = function(e){
    var t = e.target || {};
    var n = (t.tagName || '').toLowerCase();
    return n === 'input' || n === 'textarea' || t.isContentEditable;
  };
  document.addEventListener('keydown', function(e){
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    var grid = document.querySelector('[data-testid="stDataFrame"] [role="grid"]')
            || document.querySelector('[data-testid="stDataFrame"]');
    if (e.key === '/' && !typing(e)) {
      var box = document.querySelector('[data-testid="stMultiSelect"] input');
      if (box) { e.preventDefault(); box.focus(); }
      return;
    }
    if (typing(e)) { return; }
    if (e.key === 'j' || e.key === 'k') {
      if (grid) { e.preventDefault(); grid.scrollTop += (e.key === 'j' ? 34 : -34); }
    } else if (e.key === 'r') {
      var run = document.querySelector('.st-key-run_scan button');
      if (run) { e.preventDefault(); run.click(); }
    }
  });
})();</script>""", unsafe_allow_javascript=True)
