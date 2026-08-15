"""Spread Scout — strategy layer.

Pure functions: no Streamlit, no network, no I/O. Everything here is the
specification of record for bull put spreads, and everything here is unit
tested in tests/test_strategy.py.

Split out of app.py so the formulas can be tested without booting a Streamlit
session. (CLAUDE.md prefers a single-file app; testable strategy code is the
exception that earns a second module.)

The trade: sell a put ~18-22% below spot, buy a lower-strike put on the same
expiration to cap risk, collect the net credit. The stock must fall more than
~20% before the position is threatened.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd

# --------------------------------------------------------------- universe

# Index and leveraged products are rejected at input. The thesis is that a
# single company has to fall 20% for the trade to be threatened; a diversified
# index falling 20% is a different, rarer, far more correlated event, and it is
# priced accordingly — index puts do not pay for this buffer.
ETF_BLOCKLIST = {
    # broad market
    "SPY", "VOO", "IVV", "QQQ", "QQQM", "IWM", "DIA", "VTI", "RSP", "MDY",
    "EFA", "EEM", "VEA", "VWO", "ACWI", "SCHD", "VIG", "VYM",
    # sector SPDRs and friends
    "XLF", "XLK", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "XLC", "SMH", "SOXX", "XBI", "IBB", "XOP", "XME", "XRT", "KRE", "ITB",
    # thematic
    "ARKK", "ARKG", "ARKW", "ICLN", "TAN", "JETS", "IPO", "MOON",
    # leveraged / inverse — never appropriate here
    "TQQQ", "SQQQ", "SPXL", "SPXS", "SOXL", "SOXS", "TNA", "TZA", "UPRO",
    "SPXU", "UDOW", "SDOW", "LABU", "LABD", "NUGT", "DUST", "BOIL", "KOLD",
    "TMF", "TMV", "FAS", "FAZ", "YINN", "YANG", "NVDL", "TSLL", "TSLQ",
    # vol, commodities, rates, credit
    "VXX", "UVXY", "SVXY", "VIXY", "GLD", "IAU", "SLV", "USO", "UNG", "DBC",
    "TLT", "IEF", "SHY", "HYG", "LQD", "JNK", "AGG", "BND", "TIP",
    # crypto trusts / funds
    "GBTC", "IBIT", "FBTC", "ETHE", "BITO", "BITX",
}

BLOCK_REASON = ("Single stocks only. Index and leveraged products don't pay "
                "for a 20% buffer — a diversified index falling 20% is a "
                "rarer, far more correlated event, and the premium reflects "
                "that.")


def is_blocked(ticker: str) -> bool:
    return str(ticker).strip().upper() in ETF_BLOCKLIST


# ---------------------------------------------------------------- pricing


def valid_quote(bid, ask) -> bool:
    """A quote we are willing to price from.

    Rejects missing sides, zero/negative bids, and crossed or locked markets.
    The zero bid is the case this strategy cares about most: far-OTM strikes
    with no bid still carry a stale last-trade print, and pricing off that
    print manufactures spreads that look excellent and cannot be filled.
    """
    if bid is None or ask is None:
        return False
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return False
    if bid != bid or ask != ask:            # NaN
        return False
    return bid > 0 and ask > 0 and ask > bid


def quote_mid(bid, ask) -> float | None:
    return (float(bid) + float(ask)) / 2 if valid_quote(bid, ask) else None


def ba_fraction(bid, ask) -> float | None:
    """Bid/ask width as a fraction of mid. None when there is no market."""
    mid = quote_mid(bid, ask)
    if not mid or mid <= 0:
        return None
    return (float(ask) - float(bid)) / mid


# ------------------------------------------------------------- volatility


def rv_from_closes(closes: list[float], window: int = 30) -> float | None:
    """Annualized close-to-close realized volatility."""
    closes = [c for c in closes if c and c > 0]
    if len(closes) < 12:
        return None
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes))][-window:]
    if len(rets) < 10:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var * 252)


# ------------------------------------------------------------------ events


def earnings_in_window(dates, start: dt.date, end: dt.date) -> int:
    """How many reports fall between today and expiration (inclusive)."""
    n = 0
    for d in dates or []:
        day = d
        if isinstance(d, str):
            try:
                day = dt.date.fromisoformat(d[:10])
            except ValueError:
                continue
        if isinstance(day, dt.datetime):
            day = day.date()
        if isinstance(day, dt.date) and start <= day <= end:
            n += 1
    return n


# ----------------------------------------------------------------- screener


@dataclass
class Params:
    otm_lo: float          # 0.18  -> short strike at least 18% below spot
    otm_hi: float          # 0.22  -> and no more than 22% below
    risk_cap: float        # $ ceiling: rejects a spread, then sizes contracts
    min_ror: float         # percent
    min_oi_short: int
    min_oi_long: int
    max_spread_pct: float  # 0.40 -> bid/ask width <= 40% of mid
    min_width: float


def build_spreads(chain: pd.DataFrame, spot: float, dte_of: dict,
                  p: Params) -> pd.DataFrame:
    """Every spread that clears the strategy's filters, one row each.

    Formulas, verbatim from the specification:
        credit         = short_mid - long_mid
        width          = short_strike - long_strike
        max_loss       = (width - credit) * 100          per contract
        return_on_risk = credit * 100 / (width * 100 - credit * 100)
        breakeven      = short_strike - credit
        otm_pct        = (spot - short_strike) / spot
        annualized     = return_on_risk * 365 / DTE      simple, not compounded
        contracts      = floor(risk_cap / max_loss)

    Open interest is required on BOTH legs before a spread is constructed, not
    filtered afterwards. Rows priced off a stale last trade are kept but tagged
    `pricing == "last"`, so the caller can exclude them (the default) and still
    report how many were dropped.

    `chain` must carry: strike, exp, oi, px_src ("mid"|"last"), px_fallback,
    ba_pct (None when there is no two-sided market), delta, iv.
    """
    if chain is None or len(chain) == 0:
        return pd.DataFrame()

    shorts = chain[(chain["strike"] >= (1 - p.otm_hi) * spot)
                   & (chain["strike"] <= (1 - p.otm_lo) * spot)
                   & (chain["oi"] >= p.min_oi_short)]
    out = []
    for _, s in shorts.iterrows():
        # Market quality, short leg. A leg with no two-sided market cannot be
        # measured, so it is not rejected here — it is tagged `last` below and
        # excluded by default downstream.
        if _has(s["ba_pct"]) and s["ba_pct"] > p.max_spread_pct:
            continue
        longs = chain[(chain["exp"] == s["exp"])
                      & (chain["strike"] < s["strike"] - p.min_width + 1e-9)
                      & (chain["oi"] >= p.min_oi_long)]
        for _, l in longs.iterrows():
            if _has(l["ba_pct"]) and l["ba_pct"] > p.max_spread_pct:
                continue
            # Money is rounded to cents BEFORE any comparison or division.
            # (width - credit) * 100 in raw floats gives 440.00000000000006 for
            # a 5-wide at 0.60, and 2200 // that is 4, not 5 — which would
            # silently under-size every position.
            width = round(s["strike"] - l["strike"], 4)
            credit = round(s["px_fallback"] - l["px_fallback"], 4)
            if credit <= 0:                       # zero or negative credit
                continue
            max_loss = round((width - credit) * 100, 2)   # per contract
            if max_loss <= 0 or max_loss > p.risk_cap:
                continue
            ror = 100 * credit / (width - credit)
            if ror < p.min_ror:
                continue
            dte = dte_of.get(s["exp"])
            if not dte or dte <= 0:               # never annualize on a guess
                continue
            contracts = int(p.risk_cap // max_loss)
            # midpoint-priced only if BOTH legs had a real market
            priced = ("mid" if (s["px_src"] == "mid" and l["px_src"] == "mid")
                      else "last")
            out.append({
                "ticker": s["ticker"], "exp": s["exp"], "dte": int(dte),
                "spot": round(spot, 2),
                "short_k": s["strike"], "long_k": l["strike"],
                "pct_otm": round(100 * (spot - s["strike"]) / spot, 1),
                "width": round(width, 2), "credit": round(credit, 2),
                "breakeven": round(s["strike"] - credit, 2),
                "max_loss": round(max_loss, 0),
                "contracts": contracts,
                "total_credit": round(credit * 100 * contracts, 0),
                "total_risk": round(max_loss * contracts, 0),
                "ror_pct": round(ror, 1),
                "annualized_pct": round(ror * 365 / dte, 1),
                "short_delta": round(s["delta"], 3) if _has(s["delta"]) else None,
                "short_iv": round(s["iv"], 3) if _has(s["iv"]) else None,
                "short_oi": int(s["oi"]), "long_oi": int(l["oi"]),
                "short_bid": s.get("bid"), "short_ask": s.get("ask"),
                "long_bid": l.get("bid"), "long_ask": l.get("ask"),
                "short_ba_pct": round(100 * s["ba_pct"], 1) if _has(s["ba_pct"]) else None,
                "long_ba_pct": round(100 * l["ba_pct"], 1) if _has(l["ba_pct"]) else None,
                "pricing": priced,
            })
    df = pd.DataFrame(out)
    if df.empty:
        return df
    # De-noiser (not from the spec): keep the two best long legs per short
    # strike so a single strike cannot flood the table. Disclosed in the UI.
    return (df.sort_values("ror_pct", ascending=False)
              .groupby(["ticker", "exp", "short_k"], as_index=False).head(2)
              .sort_values("ror_pct", ascending=False).reset_index(drop=True))


def _has(v) -> bool:
    """True when a numeric field is really present (0.0 counts, NaN doesn't)."""
    if v is None:
        return False
    try:
        return not pd.isna(v)
    except (TypeError, ValueError):
        return True


def rank_within_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Rank by return on risk, with stale-priced rows deprioritized.

    Never call this on a mixed-regime frame. Far-dated spreads carry larger raw
    credits for the same buffer, so one blended sort structurally buries the
    entire 30-45 DTE sleeve — that is a ranking bug, not a preference.
    """
    if df is None or df.empty:
        return df
    d = df.copy()
    d["_mid_first"] = (d["pricing"] == "mid").astype(int)
    return (d.sort_values(["_mid_first", "ror_pct"], ascending=[False, False])
             .drop(columns="_mid_first").reset_index(drop=True))


def rank_cross_regime(df: pd.DataFrame) -> pd.DataFrame:
    """The only legitimate way to compare sleeves: simple annualized return."""
    if df is None or df.empty:
        return df
    d = df.copy()
    d["_mid_first"] = (d["pricing"] == "mid").astype(int)
    return (d.sort_values(["_mid_first", "annualized_pct"],
                          ascending=[False, False])
             .drop(columns="_mid_first").reset_index(drop=True))
