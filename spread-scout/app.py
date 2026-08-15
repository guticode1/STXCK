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

Presentation lives in assets/styles.css. The data layer and screener below
are load-bearing: pricing labels (mid/last) and liquidity filters must not be
removed. This is a screening tool, not investment advice.
"""

from __future__ import annotations

import datetime as dt
import math
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

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
    "Mega-cap tech": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
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
            quote_, greeks = c.get("last_quote", {}) or {}, c.get("greeks", {}) or {}
            bid, ask = quote_.get("bid"), quote_.get("ask")
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

# ------------------------------------------------------- signature element


def payoff_svg(spot: float, short_k: float, long_k: float, credit: float,
               dte: int, iv: float, live: bool, w: int = 328, h: int = 156) -> str:
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
    strike_in_cone = short_k > cone_lo

    def vline(p: float, cls: str) -> str:
        return (f'<line class="{cls}" x1="{X(p):.1f}" y1="{pad_t - 6:.1f}" '
                f'x2="{X(p):.1f}" y2="{h - pad_b:.1f}" stroke-width="1"/>')

    return f"""
<svg class="payoff" viewBox="0 0 {w} {h}" preserveAspectRatio="xMidYMid meet"
     style="width:100%;height:auto;display:block;font-size:{11 if w > 400 else 8.5}px"
     role="img" aria-label="Payoff profile with expected move cone">
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
</svg>
<div class="payoff-note" style="display:flex;justify-content:space-between;
     padding:2px 8px 2px 8px">
  <span>breakeven <span style="color:#E8EBF2">{be:,.2f}</span></span>
  <span>{'live IV ' + format(iv * 100, '.0f') + '%'
         if live else 'schematic · assumed 35% IV'}</span>
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
        if picked != universe:
            st.session_state["universe"] = picked
            universe = picked

        # preset baskets — additive/subtractive, active when fully contained
        cols = st.columns(2)
        for i, (name, basket) in enumerate(PRESETS.items()):
            active = bool(basket) and set(basket) <= set(universe)
            slug = name.lower().replace(" ", "_").replace("-", "")
            if cols[i % 2].button(("● " if active else "＋ ") + name,
                                  key=f"preset_{slug}", width="stretch",
                                  help=f"{'Remove' if active else 'Add'} "
                                       f"{len(basket)} tickers"):
                if active:
                    set_universe([t for t in universe if t not in basket])
                else:
                    set_universe(universe + basket)
                st.rerun()

        c1, c2, c3 = st.columns(3)
        if c1.button("Clear", key="uni_clear", width="stretch"):
            set_universe([])
            st.rerun()
        if c2.button("Defaults", key="uni_default", width="stretch"):
            set_universe(list(DEFAULT_UNIVERSE))
            st.rerun()
        if c3.button("Save list", key="uni_save", width="stretch",
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

    # ---- strategy
    otm_state = st.session_state.get("otm_band", (18, 22))
    ml_state = st.session_state.get("max_loss", 2200)
    with st.expander(f"◆  Strategy · {otm_state[0]}–{otm_state[1]}% OTM · "
                     f"max ${ml_state:,}", expanded=True):
        st.html('<p class="helper">Where the short strike sits, and how much '
                'collateral each spread may risk.</p>')
        otm_band = st.slider("Short strike, % below spot", 10, 35, (18, 22),
                             key="otm_band")
        max_loss = st.number_input("Max loss per spread", 100, 25000, 2200, 100,
                                   key="max_loss")
        min_ror = st.slider("Min return on risk (%)", 0, 40, 5, key="min_ror")
        min_width = st.number_input("Min spread width", 1.0, 50.0, 2.5, 0.5,
                                    key="min_width")

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
                       w=292, h=150)))
        st.html('<p class="helper" style="margin-top:8px">Drawn at your minimum '
                'return on risk — the least you would accept, on a $100 stock.</p>')

    # ---- liquidity
    oi_s_state = st.session_state.get("min_oi_s", 300)
    oi_l_state = st.session_state.get("min_oi_l", 100)
    ba_state = st.session_state.get("max_ba", 40)
    with st.expander(f"≋  Liquidity · OI {oi_s_state}/{oi_l_state} · "
                     f"≤{ba_state}% width", expanded=False):
        st.html('<p class="helper">Open-interest floors per leg, and how wide a '
                'bid/ask the short leg may quote.</p>')
        min_oi_s = st.number_input("Min OI, short leg", 0, 50000, 300, 50,
                                   key="min_oi_s")
        min_oi_l = st.number_input("Min OI, long leg", 0, 50000, 100, 50,
                                   key="min_oi_l")
        max_ba = st.slider("Max bid/ask width (% of mid)", 5, 100, 40, key="max_ba")

        strict = (min(min_oi_s / 1500, 1) + min(min_oi_l / 800, 1)
                  + min((100 - max_ba) / 80, 1)) / 3
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
        st.html(f'<div class="run-cost">{len(universe)} names · {n_chains} chains'
                f' · est {max(1, round(n_chains * 0.45 / 60)):d}–'
                f'{max(2, round(n_chains * 1.1 / 60)):d} min</div>')

params = Params(otm_lo=otm_band[0] / 100, otm_hi=otm_band[1] / 100,
                max_loss=max_loss, min_ror=min_ror,
                min_oi_short=min_oi_s, min_oi_long=min_oi_l,
                max_spread_pct=max_ba / 100, min_width=min_width)

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
    f'<span class="chip">max loss <b>${max_loss:,}</b></span>'
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
    head = ('<div class="ghost-head"><span>ticker</span><span>short / long</span>'
            '<span>width</span><span>credit</span><span>max loss</span>'
            '<span>return on risk</span><span>dte</span></div>')
    rows = "".join(
        '<div class="ghost-row">'
        '<div class="bone w70"></div><div class="bone w85"></div>'
        '<div class="bone w50"></div><div class="bone w50"></div>'
        '<div class="bone w70"></div>'
        f'<div class="bone-bar"><i style="width:{w}%"></i></div>'
        '<div class="bone w50"></div></div>'
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
    notes, tape = [], []
    missing = [t for t in universe if t not in spots]
    if missing:
        notes.append(f"No spot price for {', '.join(missing)} — skipped.")

    all_rows = []
    work = [(t, w) for t in universe if t in spots for w in windows]
    if not work:
        return {"results": pd.DataFrame(), "notes": notes + [
            "No spot prices returned for any ticker — check the data feed."],
            "at": dt.datetime.now().strftime("%b %d · %H:%M"), "params": {}}

    bar = st.progress(0.0, text="Starting scan…")
    tape_slot, skel_slot = st.empty(), st.empty()
    skel_slot.html(skeleton_html(4, "waiting for the first candidates…"))

    for i, (t, (label, d0, d1)) in enumerate(work):
        bar.progress(i / len(work), text=f"Fetching {t} chains · {label} "
                                         f"· {i + 1} of {len(work)}")
        spot = spots[t]
        k_lo = (1 - params.otm_hi) * spot - max_loss / 100 - 5   # room for long legs
        k_hi = (1 - params.otm_lo) * spot + 1
        try:
            chain = put_chain(t, d0.isoformat(), d1.isoformat(), k_lo, k_hi)
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
        spreads = build_spreads(chain, spot, dte_of, params)
        if not spreads.empty:
            spreads.insert(1, "window", label)
            all_rows.append(spreads)
            tape.append((t, len(spreads), "hit", label))
        else:
            tape.append((t, 0, "none", label))
        tape_slot.html(tape_html(tape))

    bar.progress(1.0, text="Ranking candidates…")
    df = (pd.concat(all_rows, ignore_index=True)
          .sort_values("ror_pct", ascending=False).reset_index(drop=True)
          if all_rows else pd.DataFrame())
    bar.empty(); tape_slot.empty(); skel_slot.empty()
    return {"results": df, "notes": notes,
            "at": dt.datetime.now().strftime("%b %d · %H:%M"),
            "params": {"min_ror": min_ror, "otm": tuple(otm_band),
                       "max_loss": max_loss, "min_oi_s": min_oi_s,
                       "max_ba": max_ba, "min_width": min_width}}


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
        '<div class="legend rise d5">'
        '<span class="legend-item"><i class="legend-swatch" style="background:'
        'linear-gradient(90deg,#8A6A1C,#4FD18B)"></i>return on risk, ranked</span>'
        '<span class="legend-item"><span class="badge badge-mid">mid</span>'
        'priced off the bid/ask midpoint</span>'
        '<span class="legend-item"><span class="badge badge-last">last</span>'
        'fell back to a stale last trade — verify before trading</span>'
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
    st.html(
        '<div class="hero"><p class="eyebrow">no matches</p>'
        f'<p class="lede">Nothing cleared a {p_ror}% return on risk.</p>'
        f'<p class="sub">You screened {p_otm[0]}–{p_otm[1]}% below spot with a '
        f'{p_oi}-contract open-interest floor. At short DTE, 20% OTM pays close '
        'to nothing on calm mega caps — the premium is only there when vol is.'
        '</p></div>'
        '<p class="section-title">Try one of these</p>'
        '<div class="legend">'
        f'<span class="chip">widen OTM to <b>{max(10, p_otm[0] - 4)}–'
        f'{p_otm[1]}%</b></span>'
        f'<span class="chip">lower min RoR to <b>{max(0, p_ror - 3)}%</b></span>'
        f'<span class="chip">drop OI floor to <b>{min(max(p_oi // 2, 0), 250)}</b>'
        f'</span>'
        '<span class="chip">add high-IV names <b>TSLA · COIN · MSTR</b></span>'
        '</div>')
    st.stop()

# ---------------------------------------------------------------- metrics


def metrics_html(df: pd.DataFrame, animate: bool) -> str:
    med, best = df["ror_pct"].median(), df["ror_pct"].max()
    avg_dte, last_pct = df["dte"].mean(), (df["pricing"] == "last").mean() * 100
    tiles = [("candidates", len(df), 0, "", ""), ("median RoR", med, 1, "", "%"),
             ("best RoR", best, 1, "", "%"), ("avg DTE", avg_dte, 0, "", "d"),
             ("priced off last", last_pct, 0, "", "%")]
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
    bars = "".join(
        f'<i style="flex:1 1 0;min-width:2px;border-radius:2px 2px 0 0;'
        f'height:{max(3, c / peak * 100):.0f}%;background:'
        f'{"#4FD18B" if i > bins * 0.6 else "#8A6A1C"}"></i>'
        for i, c in enumerate(counts))
    spark = (f'<div class="panel panel-tight" style="grid-column:1/-1">'
             f'<div class="kv-k">return on risk · distribution</div>'
             f'<div style="display:flex;align-items:flex-end;gap:2px;height:34px;'
             f'margin-top:6px">{bars}</div>'
             f'<div class="payoff-note" style="display:flex;'
             f'justify-content:space-between;margin-top:4px">'
             f'<span>{lo:.1f}%</span><span>{hi:.1f}%</span></div></div>')

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


animate = st.session_state.get("metrics_seen") != scan["at"]
st.html(metrics_html(results, animate), unsafe_allow_javascript=animate)
st.session_state["metrics_seen"] = scan["at"]
st.html(f'<p class="payoff-note" style="margin:6px 0 18px 2px">scan completed '
        f'{esc(scan["at"])} · {len(results):,} candidates ranked by return on '
        f'risk</p>')

# ---------------------------------------------------------------- results


def dot_uri(color: str) -> str:
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14">'
           f'<circle cx="7" cy="7" r="4.5" fill="{color}"/></svg>')
    return "data:image/svg+xml;charset=utf-8," + quote(svg)


def order_ticket(row: pd.Series) -> str:
    return (f"SELL -1 {row['ticker']} {row['exp']} {row['short_k']:g}P / "
            f"BUY +1 {row['long_k']:g}P  @ {row['credit']:.2f} CR   "
            f"[width {row['width']:g} · max loss ${row['max_loss']:,.0f} · "
            f"{int(row['dte'])}d]")


QUICK = {"RoR > 10%": lambda d: d["ror_pct"] > 10,
         "30–45 DTE only": lambda d: d["window"] == WIN_SHORT,
         "Exclude last-trade": lambda d: d["pricing"] == "mid",
         "Width ≥ $5": lambda d: d["width"] >= 5}


@st.fragment
def results_panel(df: pd.DataFrame) -> None:
    """Filters + table + detail live in a fragment so filtering re-renders only
    this block instead of the whole page."""
    st.html('<p class="section-title">Candidates</p>')
    picks = st.pills("Quick filters", list(QUICK), selection_mode="multi",
                     key="quick_filters", label_visibility="collapsed")
    f1, f2 = st.columns([1, 2])
    win_pick = f1.selectbox("Window", ["All windows"] + sorted(df["window"].unique()),
                            key="win_pick", label_visibility="collapsed")
    tick_pick = f2.multiselect("Tickers", sorted(df["ticker"].unique()),
                               key="tick_pick", placeholder="All tickers",
                               label_visibility="collapsed")

    view = df
    for p in picks or []:
        view = view[QUICK[p](view)]
    if win_pick != "All windows":
        view = view[view["window"] == win_pick]
    if tick_pick:
        view = view[view["ticker"].isin(tick_pick)]

    if view.empty:
        st.html('<div class="panel"><p class="sub">No candidates match these '
                'filters. Clear one to widen the set.</p></div>')
        return

    disp = view.copy()
    disp["sector"] = [dot_uri(sector_color(t)) for t in disp["ticker"]]
    disp["legs"] = (disp["short_k"].map("{:g}".format) + " / "
                    + disp["long_k"].map("{:g}".format))
    disp["priced"] = disp["pricing"].map({"mid": "mid", "last": "⚠ LAST"})
    disp = disp[["sector", "ticker", "window", "exp", "dte", "legs", "pct_otm",
                 "width", "credit", "max_loss", "ror_pct", "annualized_pct",
                 "short_delta", "short_iv", "short_oi", "long_oi", "priced"]]

    event = st.dataframe(
        disp, width="stretch", hide_index=True, height=520,
        key="results_table", on_select="rerun", selection_mode="single-row",
        column_config={
            "sector": st.column_config.ImageColumn("", width=36,
                                                   help="Sector"),
            "ticker": st.column_config.TextColumn("Ticker", width="small"),
            "window": st.column_config.TextColumn("Window", width="small"),
            "exp": st.column_config.TextColumn("Expiry", width="small"),
            "dte": st.column_config.NumberColumn("DTE", format="%d", width="small"),
            "legs": st.column_config.TextColumn("Short / long",
                                                help="Short strike / long strike"),
            "pct_otm": st.column_config.NumberColumn("% OTM", format="%.1f%%"),
            "width": st.column_config.NumberColumn("Width", format="$%.1f"),
            "credit": st.column_config.NumberColumn("Credit", format="$%.2f"),
            "max_loss": st.column_config.NumberColumn("Max loss", format="$%d"),
            "ror_pct": st.column_config.ProgressColumn(
                "Return on risk", format="%.1f%%", min_value=0.0,
                max_value=float(max(disp["ror_pct"].max(), 1))),
            "annualized_pct": st.column_config.NumberColumn("Annualized",
                                                            format="%.0f%%"),
            "short_delta": st.column_config.NumberColumn("Δ short", format="%.3f"),
            "short_iv": st.column_config.NumberColumn("IV short", format="%.2f"),
            "short_oi": st.column_config.NumberColumn("OI short", format="%d"),
            "long_oi": st.column_config.NumberColumn("OI long", format="%d"),
            "priced": st.column_config.TextColumn(
                "Priced", width="small",
                help="mid = bid/ask midpoint · LAST = stale last trade"),
        })

    st.html(f'<p class="payoff-note" style="margin:6px 0 0 2px">showing '
            f'{len(view):,} of {len(df):,} · select a row for the full ticket</p>')

    rows = list(event.selection.rows) if event and event.selection else []
    if rows:
        r = view.iloc[rows[0]]
        iv = float(r["short_iv"]) if pd.notna(r["short_iv"]) else 0.35
        spot_est = float(r["short_k"]) / (1 - r["pct_otm"] / 100)
        st.html('<p class="section-title" style="margin-top:20px">'
                f'{esc(r["ticker"])} · {esc(r["exp"])} · {r["short_k"]:g}/'
                f'{r["long_k"]:g} put spread</p>')
        left, right = st.columns([1.05, 1])
        with left:
            raw_html(payoff_panel(
                "Payoff at expiration",
                f"{int(r['dte'])}d · {SECTOR_META[sector_of(r['ticker'])][0]}",
                payoff_svg(spot_est, float(r["short_k"]), float(r["long_k"]),
                           float(r["credit"]), int(r["dte"]), iv,
                           live=pd.notna(r["short_iv"]), w=600, h=232)))
        with right:
            badge = ('<span class="badge badge-last">last-trade priced</span>'
                     if r["pricing"] == "last"
                     else '<span class="badge badge-mid">midpoint priced</span>')
            st.html(
                '<div class="panel"><div class="kv">'
                f'<div class="kv-item"><div class="kv-k">credit</div>'
                f'<div class="kv-v credit">${r["credit"]:.2f}</div></div>'
                f'<div class="kv-item"><div class="kv-k">max loss</div>'
                f'<div class="kv-v risk">${r["max_loss"]:,.0f}</div></div>'
                f'<div class="kv-item"><div class="kv-k">breakeven</div>'
                f'<div class="kv-v">{float(r["short_k"]) - float(r["credit"]):,.2f}'
                f'</div></div>'
                f'<div class="kv-item"><div class="kv-k">return on risk</div>'
                f'<div class="kv-v">{r["ror_pct"]:.1f}%</div></div>'
                f'<div class="kv-item"><div class="kv-k">credit / width</div>'
                f'<div class="kv-v">{100 * float(r["credit"]) / float(r["width"]):.0f}%'
                f'</div></div>'
                f'<div class="kv-item"><div class="kv-k">open interest</div>'
                f'<div class="kv-v">{int(r["short_oi"]):,} / {int(r["long_oi"]):,}'
                f'</div></div>'
                f'</div><div style="margin-top:14px">{badge}</div></div>')
        st.html('<p class="kv-k" style="margin:14px 0 4px 2px">order ticket</p>')
        st.code(order_ticket(r), language=None, wrap_lines=True)

    st.download_button("Download CSV", df.to_csv(index=False),
                       f"spread_scout_{today}.csv", "text/csv",
                       key="dl_csv", icon=":material/download:")


tab_spreads, tab_news = st.tabs(["Candidates", "News & events"])
with tab_spreads:
    results_panel(results)

# ------------------------------------------------------------ news overlay

CHIP_CLASS = {"positive": "badge-mid", "negative": "badge-last",
              "neutral": "badge-win"}

with tab_news:
    top_names = list(results["ticker"].drop_duplicates().head(8))
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

st.html('<div style="height:26px"></div>'
        '<p class="payoff-note">Screening tool, not investment advice. Rows '
        'marked LAST used stale last-trade prices; verify against a live quote '
        'before trading.</p>')
