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
    # --- sanity gate + probability filters
    max_credit_pct: float = 0.35    # credit may not exceed this share of width
    min_net_premium: float = 200.0  # on the POSITION total, credit x 100 x qty
    min_pop: float = 0.80           # N(d2) of the short strike
    max_short_delta: float = 0.15   # |delta|, linked to min_pop in the UI
    min_ev: float | None = None     # opt-in, on the realized-vol basis
    max_per_underlying: int = 2
    r: float = 0.04                 # risk-free
    q: float = 0.0                  # dividend yield
    market_open: bool = True        # drives the pricing policy


def build_spreads(chain: pd.DataFrame, spot: float, dte_of: dict, p: Params,
                  rv: float | None = None,
                  rejects: dict | None = None) -> pd.DataFrame:
    """Every spread that survives the sanity gate and the filters.

    Pricing policy branches on the session (p.market_open):

      open   - both legs must have a live two-sided quote. A leg with no
               market is unfillable now, so it is rejected outright.
      closed - both legs may use the session close mark, but they must come
               from the SAME session. Differencing two marks from different
               days does not produce a spread price; it produces the artifact
               that made a 15-wide GS spread appear to pay 11.85.

    The sanity gate runs before a candidate is ever constructed. Every
    rejection is counted by reason in `rejects`.

    `chain` needs: strike, exp, oi, px, px_src ("mid"|"close"), mark_day,
    ba_pct, bid, ask, delta, iv, spc (shares per contract).
    """
    rej = rejects if rejects is not None else {}

    def drop(reason: str, n: int = 1) -> None:
        rej[reason] = rej.get(reason, 0) + n

    if chain is None or len(chain) == 0:
        return pd.DataFrame()

    shorts = chain[(chain["strike"] >= (1 - p.otm_hi) * spot)
                   & (chain["strike"] <= (1 - p.otm_lo) * spot)]
    out = []
    for _, s in shorts.iterrows():
        if s["oi"] < p.min_oi_short:
            drop("open interest below floor")
            continue
        if not _leg_ok(s, p, drop):
            continue
        longs = chain[(chain["exp"] == s["exp"])
                      & (chain["strike"] < s["strike"] - p.min_width + 1e-9)]
        for _, l in longs.iterrows():
            if l["oi"] < p.min_oi_long:
                drop("open interest below floor")
                continue
            if not _leg_ok(l, p, drop):
                continue
            # both legs must describe the same deliverable
            if _has(s.get("spc")) and _has(l.get("spc")) and s["spc"] != l["spc"]:
                drop("non-standard deliverable")
                continue
            # and, when closed, the same session
            if not p.market_open:
                if s.get("mark_day") and l.get("mark_day") \
                        and s["mark_day"] != l["mark_day"]:
                    drop("marks from different sessions")
                    continue

            width = round(s["strike"] - l["strike"], 4)
            credit = round(s["px"] - l["px"], 4)
            if credit <= 0:
                drop("zero or negative credit")
                continue
            # THE gate: a credit worth most of the width at 20% OTM is a
            # broken quote, not a market.
            if not sane_credit(credit, width, p.max_credit_pct):
                drop("credit above ceiling vs width")
                continue
            max_loss = round((width - credit) * 100, 2)
            if max_loss <= 0 or max_loss > p.risk_cap:
                drop("max loss above risk cap")
                continue
            ror = 100 * credit / (width - credit)
            if ror < p.min_ror:
                drop("below min return on risk")
                continue
            dte = dte_of.get(s["exp"])
            if not dte or dte <= 0:
                drop("unknown expiry")
                continue
            contracts = int(p.risk_cap // max_loss)
            net_premium = round(credit * 100 * contracts, 2)
            if net_premium < p.min_net_premium:
                drop("below min net premium")
                continue

            # ---- probability, from each leg's implied vol
            T = dte / 365.0
            sig_s = _leg_iv(s, spot, T, p)
            sig_l = _leg_iv(l, spot, T, p)
            pop = d_short = p_full = p_tch = ev_rn = ev_adj = None
            if sig_s and sig_s > 0:
                pop = prob_otm(spot, s["strike"], T, p.r, p.q, sig_s)
                d_short = put_delta(spot, s["strike"], T, p.r, p.q, sig_s)
                p_tch = prob_touch(spot, s["strike"], T, p.r, p.q, sig_s)
                if pop < p.min_pop:
                    drop("below min probability of profit")
                    continue
                if abs(d_short) > p.max_short_delta:
                    drop("short delta above limit")
                    continue
            if sig_l and sig_l > 0:
                p_full = 1.0 - prob_otm(spot, l["strike"], T, p.r, p.q, sig_l)
            if sig_s and sig_l:
                ev_rn = spread_ev(spot, s["strike"], l["strike"], credit, T,
                                  p.r, p.q, sig_s, sig_l)
                if rv and rv > 0:
                    ev_adj = spread_ev(spot, s["strike"], l["strike"], credit,
                                       T, p.r, p.q, rv, rv)
            if p.min_ev is not None:
                if ev_adj is None or ev_adj < p.min_ev:
                    drop("below min expected value")
                    continue

            priced = "mid" if (s["px_src"] == "mid" and l["px_src"] == "mid") \
                else "close"
            out.append({
                "ticker": s["ticker"], "exp": s["exp"], "dte": int(dte),
                "spot": round(spot, 2),
                "short_k": s["strike"], "long_k": l["strike"],
                "pct_otm": round(100 * (spot - s["strike"]) / spot, 1),
                "width": round(width, 2), "credit": round(credit, 2),
                "credit_pct_width": round(100 * credit / width, 1),
                "breakeven": round(s["strike"] - credit, 2),
                "max_loss": max_loss,
                "contracts": contracts,
                "total_credit": net_premium,
                "total_risk": round(max_loss * contracts, 2),
                "ror_pct": round(ror, 1),
                "annualized_pct": round(ror * 365 / dte, 1),
                "pop": round(pop, 4) if pop is not None else None,
                "p_itm": round(1 - pop, 4) if pop is not None else None,
                "p_full_loss": round(p_full, 4) if p_full is not None else None,
                "p_touch": round(p_tch, 4) if p_tch is not None else None,
                "short_delta": round(d_short, 3) if d_short is not None else None,
                "ev_rn": round(ev_rn, 2) if ev_rn is not None else None,
                "ev_adj": round(ev_adj, 2) if ev_adj is not None else None,
                "ev_per_collateral": (round(ev_adj / max_loss, 4)
                                      if (ev_adj is not None and max_loss) else None),
                "ev_crude": (round(spread_ev_crude(credit, width, pop, p_full), 2)
                             if (pop is not None and p_full is not None) else None),
                "short_iv": round(sig_s, 4) if sig_s else None,
                "long_iv": round(sig_l, 4) if sig_l else None,
                "rv30": round(rv, 4) if rv else None,
                "iv_minus_rv": (round(sig_s - rv, 4) if (sig_s and rv) else None),
                "short_oi": int(s["oi"]), "long_oi": int(l["oi"]),
                "short_bid": s.get("bid"), "short_ask": s.get("ask"),
                "long_bid": l.get("bid"), "long_ask": l.get("ask"),
                "short_ba_pct": round(100 * s["ba_pct"], 1) if _has(s["ba_pct"]) else None,
                "long_ba_pct": round(100 * l["ba_pct"], 1) if _has(l["ba_pct"]) else None,
                "mark_day": s.get("mark_day"),
                "pricing": priced,
            })
    return pd.DataFrame(out)


def _leg_ok(leg, p: Params, drop) -> bool:
    """Quote-quality gate applied to a single leg before pairing."""
    px, src = leg.get("px"), leg.get("px_src")
    if not _has(px) or px <= 0:
        drop("leg priced at zero")
        return False
    if p.market_open and src != "mid":
        drop("no live two-sided quote")
        return False
    bid, ask = leg.get("bid"), leg.get("ask")
    if _has(bid) and float(bid) <= 0 and src == "mid":
        drop("zero bid")
        return False
    if _has(bid) and _has(ask) and float(ask) <= float(bid):
        drop("crossed or locked market")
        return False
    if _has(leg.get("ba_pct")) and leg["ba_pct"] > p.max_spread_pct:
        drop("bid/ask wider than limit")
        return False
    return True


def _leg_iv(leg, spot: float, T: float, p: Params) -> float | None:
    """Chain IV when supplied, otherwise solved from the leg's own mark."""
    iv = leg.get("iv")
    if _has(iv) and iv and iv > 0:
        return float(iv)
    return implied_vol_put(leg.get("px"), spot, leg["strike"], T, p.r, p.q)


RANK_METRICS = {
    "EV per $ collateral": ("ev_per_collateral", False),
    "Return on risk": ("ror_pct", False),
    "Annualized RoR": ("annualized_pct", False),
    "Probability of profit": ("pop", False),
    "Premium collected": ("total_credit", False),
}


def dedupe(df: pd.DataFrame, metric: str, max_per_underlying: int = 2
           ) -> pd.DataFrame:
    """Best candidate per (underlying, expiration), then cap per underlying.

    One high-IV name produced five of the first nine rows before this existed.
    Alternates are not discarded — they are returned separately for row-expand.
    """
    if df is None or df.empty:
        return df
    col, asc = RANK_METRICS.get(metric, ("ror_pct", False))
    d = df.copy()
    if col not in d.columns:
        col = "ror_pct"
    d["_k"] = d[col].fillna(-1e18)
    d["_mid_first"] = (d["pricing"] == "mid").astype(int)
    d = d.sort_values(["_mid_first", "_k"], ascending=[False, asc])
    best = d.groupby(["ticker", "exp"], as_index=False, sort=False).head(1)
    best = best.groupby("ticker", as_index=False, sort=False).head(
        max(1, max_per_underlying))
    return best.drop(columns=["_k", "_mid_first"]).reset_index(drop=True)


def alternates_for(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """The other spreads on the same underlying+expiry, for row-expand."""
    if df is None or df.empty:
        return df
    m = ((df["ticker"] == row["ticker"]) & (df["exp"] == row["exp"])
         & ~((df["short_k"] == row["short_k"]) & (df["long_k"] == row["long_k"])))
    return df[m]


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

# ============================================================ market session

# US equity-options holidays. Hardcoded rather than pulling a dependency; the
# list needs a yearly top-up, which is cheaper than a calendar package.
MARKET_HOLIDAYS = {
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}
# 13:00 ET closes
EARLY_CLOSES = {"2026-11-27", "2026-12-24", "2027-11-26"}
ET = "America/New_York"


def _et_now(now: dt.datetime | None = None) -> dt.datetime:
    from zoneinfo import ZoneInfo
    if now is None:
        return dt.datetime.now(ZoneInfo(ET))
    if now.tzinfo is None:
        return now.replace(tzinfo=ZoneInfo(ET))
    return now.astimezone(ZoneInfo(ET))


def _is_trading_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d.isoformat() not in MARKET_HOLIDAYS


def market_session(now: dt.datetime | None = None) -> dict:
    """Where we are in the US equity-options session.

    Returns state in {"open", "closed"}, the reason it is closed, the next
    open, and whether today is a half day. The whole pricing-quality treatment
    branches on this: a quote that is a last-session mark at 04:57 on a
    Saturday is expected and usable for planning; the same mark mid-session on
    a Tuesday means the strike has no market and is unfillable.
    """
    n = _et_now(now)
    today = n.date()
    early = today.isoformat() in EARLY_CLOSES
    close_h, close_m = (13, 0) if early else (16, 0)
    open_t = n.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = n.replace(hour=close_h, minute=close_m, second=0, microsecond=0)

    if not _is_trading_day(today):
        reason = ("holiday" if today.isoformat() in MARKET_HOLIDAYS else "weekend")
        state = "closed"
    elif n < open_t:
        reason, state = "pre-market", "closed"
    elif n > close_t:
        reason, state = "after-hours", "closed"
    else:
        reason, state = "", "open"

    nxt = n
    if state == "open":
        next_open = None
    else:
        if _is_trading_day(today) and n < open_t:
            next_open = open_t
        else:
            d = today + dt.timedelta(days=1)
            while not _is_trading_day(d):
                d += dt.timedelta(days=1)
            next_open = dt.datetime.combine(d, dt.time(9, 30), tzinfo=n.tzinfo)
    return {"state": state, "reason": reason, "now": n, "early_close": early,
            "next_open": next_open,
            "last_session": _prev_trading_day(today, n, open_t)}


def _prev_trading_day(today: dt.date, n: dt.datetime, open_t: dt.datetime) -> dt.date:
    """The session whose marks we would be looking at right now."""
    if _is_trading_day(today) and n >= open_t:
        return today
    d = today - dt.timedelta(days=1)
    while not _is_trading_day(d):
        d -= dt.timedelta(days=1)
    return d


# ============================================================== probability

SQRT2 = math.sqrt(2.0)


def norm_cdf(x: float) -> float:
    """N(x). math.erf is double precision — no scipy dependency needed."""
    return 0.5 * (1.0 + math.erf(x / SQRT2))


def bs_d1_d2(S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> tuple[float, float]:
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("bs_d1_d2 requires positive S, K, T, sigma")
    vt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + sigma * sigma / 2.0) * T) / vt
    return d1, d1 - vt


def bs_put(S: float, K: float, T: float, r: float, q: float,
           sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    return (K * math.exp(-r * T) * norm_cdf(-d2)
            - S * math.exp(-q * T) * norm_cdf(-d1))


def put_delta(S: float, K: float, T: float, r: float, q: float,
              sigma: float) -> float:
    """Negative. |delta| is the market's rough proxy for P(ITM)."""
    if T <= 0 or sigma <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = bs_d1_d2(S, K, T, r, q, sigma)
    return -math.exp(-q * T) * norm_cdf(-d1)


def prob_otm(S: float, K: float, T: float, r: float, q: float,
             sigma: float) -> float:
    """P(S_T > K) = N(d2) — probability the short put expires worthless."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    _, d2 = bs_d1_d2(S, K, T, r, q, sigma)
    return norm_cdf(d2)


def prob_touch(S: float, K: float, T: float, r: float, q: float,
               sigma: float) -> float:
    """Driftless approximation: 2 x N(-d2). Always label it approximate."""
    return min(1.0, 2.0 * (1.0 - prob_otm(S, K, T, r, q, sigma)))


def implied_vol_put(price: float, S: float, K: float, T: float, r: float,
                    q: float) -> float | None:
    """Solve IV from a put mid by bisection. None when the price is outside
    the no-arbitrage band (which is itself a sign of a broken quote)."""
    if price is None or price <= 0 or T <= 0 or S <= 0 or K <= 0:
        return None
    intrinsic = max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)
    if price < intrinsic - 1e-6 or price > K * math.exp(-r * T):
        return None
    lo, hi = 1e-4, 5.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if bs_put(S, K, T, r, q, mid) > price:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-6:
            break
    return (lo + hi) / 2


def spread_ev(S: float, Ks: float, Kl: float, credit: float, T: float,
              r: float, q: float, sig_s: float, sig_l: float) -> float:
    """Expected P/L per contract at expiration, in dollars.

    Full integral, not POP x credit. E[payoff] decomposes exactly into the two
    undiscounted put expectations, so partial losses between the strikes are
    included rather than ignored:

        E[P/L] = 100 x ( credit - ( E[max(Ks-S_T,0)] - E[max(Kl-S_T,0)] ) )

    and E[max(K-S_T,0)] is the undiscounted Black-Scholes put. Priced with
    chain IV this is ~0 by construction — that is the point of showing it.
    Priced with realized vol the gap is the variance risk premium.
    """
    # Present value, so the credit (received today) and the expected payout
    # (at expiry) are compared in the same dollars. The Black-Scholes put IS
    # the PV of E[max(K-S_T,0)], so no growth factor belongs here — adding one
    # to the payout but not the credit makes EV at fair value come out
    # negative by credit x (e^rT - 1) instead of zero.
    pv_short = bs_put(S, Ks, T, r, q, sig_s)
    pv_long = bs_put(S, Kl, T, r, q, sig_l)
    return (credit - (pv_short - pv_long)) * 100.0


def spread_ev_crude(credit: float, width: float, pop: float,
                    p_full_loss: float) -> float:
    """POP x credit - P(full loss) x max loss. Ignores partial losses between
    the strikes, so it flatters every candidate. Shown only as `approx`."""
    max_loss = (width - credit) * 100.0
    return pop * credit * 100.0 - p_full_loss * max_loss


# ============================================================== sanity gate

REJECT_REASONS = (
    "zero or negative credit",
    "credit above ceiling vs width",
    "leg priced at zero",
    "zero bid",
    "crossed or locked market",
    "bid/ask wider than limit",
    "marks from different sessions",
    "non-standard deliverable",
    "max loss above risk cap",
    "below min return on risk",
    "below min net premium",
    "below min probability of profit",
    "short delta above limit",
    "below min expected value",
    "open interest below floor",
    "width below minimum",
    "unknown expiry",
)


def sane_credit(credit: float, width: float, max_pct: float) -> bool:
    """A credit worth more than ~a third of the width at 20% OTM is not a
    market, it is a broken quote. This is the gate that would have caught the
    GS 815/800 row at 79% of width."""
    return 0 < credit <= max_pct * width

