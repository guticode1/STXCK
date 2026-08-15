"""Unit tests for the Spread Scout strategy layer.

Dependency-free on purpose: run with `python3 spread-scout/tests/test_strategy.py`
(no pytest needed, nothing added to requirements.txt).

Every expected value below is hand-computed from the specification, not copied
from program output.
"""

import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy import (  # noqa: E402
    Params, build_spreads, earnings_in_window, is_blocked, quote_mid,
    rank_cross_regime, rank_within_regime, rv_from_closes, valid_quote,
)

EXP = "2026-09-18"
DTE = {EXP: 35}
FAILURES = []


def check(name, got, want, tol=1e-9):
    ok = (abs(got - want) <= tol) if isinstance(want, (int, float)) \
        and isinstance(got, (int, float)) else got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(name)


def leg(strike, bid, ask, last=None, oi=1000, iv=0.45, delta=-0.05,
        day="2026-08-14", spc=100):
    """One chain row shaped the way put_chain emits them.

    `last` stands in for a session close mark used when the market is shut;
    a zero there means "no mark", which the gate must refuse to price.
    """
    ok = valid_quote(bid, ask)
    mid = quote_mid(bid, ask)
    px = mid if ok else last
    return {
        "ticker": "TEST", "exp": EXP, "strike": float(strike),
        "px": px, "px_src": "mid" if ok else "close", "mark_day": day,
        "bid": bid, "ask": ask,
        "ba_pct": ((ask - bid) / mid) if ok and mid else None,
        "delta": delta, "iv": iv, "oi": oi, "vol": 100, "spc": spc,
    }


def params(**kw):
    """Permissive by default so each test isolates the one gate it targets."""
    base = dict(otm_lo=0.18, otm_hi=0.22, risk_cap=2200.0, min_ror=5.0,
                min_oi_short=100, min_oi_long=100, max_spread_pct=0.40,
                min_width=2.5, max_credit_pct=0.35, min_net_premium=0.0,
                min_pop=0.0, max_short_delta=1.0, market_open=True)
    base.update(kw)
    return Params(**base)


# ---------------------------------------------------------------- formulas

print("\n1. Formulas — spot 100, short 80, long 75, credit 0.60")
# credit    = 1.00 - 0.40                       = 0.60
# width     = 80 - 75                           = 5.00
# max_loss  = (5.00 - 0.60) * 100               = 440.00
# ror       = 0.60*100 / (5.00*100 - 0.60*100)  = 60/440 = 13.636%
# breakeven = 80 - 0.60                         = 79.40
# otm_pct   = (100 - 80) / 100                  = 20.0%
# annual    = 13.636 * 365/35                   = 142.2%
# contracts = floor(2200 / 440)                 = 5
chain = pd.DataFrame([
    leg(80, 0.95, 1.05),      # mid 1.00
    leg(75, 0.35, 0.45),      # mid 0.40
])
r = build_spreads(chain, 100.0, DTE, params()).iloc[0]
check("credit", r["credit"], 0.60, 1e-9)
check("width", r["width"], 5.00, 1e-9)
check("max_loss", r["max_loss"], 440.0, 1e-9)
check("return_on_risk %", r["ror_pct"], 13.6, 0.05)
check("breakeven", r["breakeven"], 79.40, 1e-9)
check("otm_pct", r["pct_otm"], 20.0, 1e-9)
check("annualized %", r["annualized_pct"], 142.2, 0.2)
check("contracts", r["contracts"], 5)
check("total_credit", r["total_credit"], 300.0)     # 0.60*100*5
check("total_risk", r["total_risk"], 2200.0)        # 440*5, exactly at cap
check("pricing tag", r["pricing"], "mid")

# ------------------------------------------------------- phantom candidate

print("\n2. Phantom candidate — long leg has NO bid but a stale last print")
# Short 80 quoted 0.95/1.05 -> mid 1.00. Long 75 has bid 0, ask 0.10 and a
# stale last of 0.02. Pricing off the stale print gives credit 0.98 and a
# 24.4% RoR that cannot be filled. The gate must refuse to build it at all.
rej = {}
chain = pd.DataFrame([
    leg(80, 0.95, 1.05),
    leg(75, 0.0, 0.10, last=0.02),
])
res = build_spreads(chain, 100.0, DTE, params(), rejects=rej)
check("phantom is never constructed", len(res), 0)
check("and the reason is logged", bool(rej), True)
print(f"        reject reasons: {rej}")
check("zero bid rejected by valid_quote", valid_quote(0.0, 0.10), False)
check("crossed market rejected", valid_quote(1.20, 1.00), False)
check("locked market rejected", valid_quote(1.00, 1.00), False)
check("normal quote accepted", valid_quote(0.95, 1.05), True)

print("\n2b. A stale SHORT leg is refused too (the old build tagged spreads "
      "by the short leg only, so a stale long leg read as fully quoted)")
rej = {}
chain = pd.DataFrame([
    leg(80, 0.0, 2.00, last=1.00),     # no bid on the short leg
    leg(75, 0.35, 0.45),
])
res = build_spreads(chain, 100.0, DTE, params(), rejects=rej)
check("stale short leg blocks construction", len(res), 0)

print("\n2c. Credit above the width ceiling is rejected (the GS artifact)")
rej = {}
chain = pd.DataFrame([leg(80, 3.95, 4.05), leg(75, 0.08, 0.12)])
# credit 3.90 on a 5 wide = 78% of width; both markets are tight, so the
# only gate that can fire is the credit ceiling
res = build_spreads(chain, 100.0, DTE, params(), rejects=rej)
check("78%-of-width credit rejected", len(res), 0)
check("logged as a width-ceiling reject",
      "credit above ceiling vs width" in rej, True)

# -------------------------------------------------------------- risk cap

print("\n3. Risk cap — exactly at $2,200, and just over")
# width 25, credit 3.00 -> max_loss = (25-3)*100 = 2200 exactly -> accepted,
# contracts = floor(2200/2200) = 1
chain = pd.DataFrame([
    leg(80, 3.95, 4.05),      # mid 4.00
    leg(55, 0.95, 1.05),      # mid 1.00 -> credit 3.00, width 25
])
res = build_spreads(chain, 100.0, DTE, params(min_width=2.5))
check("spread at exactly the cap is accepted", len(res), 1)
check("max_loss at cap", res.iloc[0]["max_loss"], 2200.0)
check("contracts at cap", res.iloc[0]["contracts"], 1)
check("total_risk at cap", res.iloc[0]["total_risk"], 2200.0)

# width 25, credit 2.90 -> max_loss = 2210 -> rejected
chain = pd.DataFrame([leg(80, 3.85, 3.95), leg(55, 0.95, 1.05)])
res = build_spreads(chain, 100.0, DTE, params())
check("spread $10 over the cap is rejected", len(res), 0)

print("\n3b. Contract stacking")
# width 5, credit 0.90 -> max_loss 410 -> floor(2200/410) = 5 contracts,
# total credit 0.90*100*5 = 450, total risk 2050
chain = pd.DataFrame([leg(80, 1.45, 1.55), leg(75, 0.55, 0.65)])
r = build_spreads(chain, 100.0, DTE, params()).iloc[0]
check("per-contract max_loss", r["max_loss"], 410.0)
check("contracts", r["contracts"], 5)
check("total_credit", r["total_credit"], 450.0)
check("total_risk under cap", r["total_risk"], 2050.0)
check("total_risk <= cap", r["total_risk"] <= 2200, True)

# --------------------------------------------------------------- filters

print("\n4. Filters")
# both-leg OI is enforced before construction
chain = pd.DataFrame([leg(80, 0.95, 1.05, oi=1000),
                      leg(75, 0.35, 0.45, oi=5)])       # long leg illiquid
check("long-leg OI floor blocks construction",
      len(build_spreads(chain, 100.0, DTE, params(min_oi_long=100))), 0)
chain = pd.DataFrame([leg(80, 0.95, 1.05, oi=5),
                      leg(75, 0.35, 0.45, oi=1000)])    # short leg illiquid
check("short-leg OI floor blocks construction",
      len(build_spreads(chain, 100.0, DTE, params(min_oi_short=100))), 0)

# bid/ask width applies to BOTH legs (old build checked the short leg only)
chain = pd.DataFrame([leg(80, 0.95, 1.05),
                      leg(75, 0.20, 0.60)])   # mid 0.40, width 0.40 = 100%
check("wide long leg rejected",
      len(build_spreads(chain, 100.0, DTE, params(max_spread_pct=0.40))), 0)
chain = pd.DataFrame([leg(80, 0.60, 1.40),    # mid 1.00, width 0.80 = 80%
                      leg(75, 0.35, 0.45)])
check("wide short leg rejected",
      len(build_spreads(chain, 100.0, DTE, params(max_spread_pct=0.40))), 0)

# min width
chain = pd.DataFrame([leg(80, 0.95, 1.05), leg(78, 0.55, 0.65)])   # width 2.0
check("width below $2.50 rejected",
      len(build_spreads(chain, 100.0, DTE, params(min_width=2.5))), 0)

# OTM band: a strike 10% below spot is outside 18-22%
chain = pd.DataFrame([leg(90, 0.95, 1.05), leg(85, 0.35, 0.45)])
check("strike outside the OTM band rejected",
      len(build_spreads(chain, 100.0, DTE, params())), 0)

# zero / negative credit
chain = pd.DataFrame([leg(80, 0.35, 0.45), leg(75, 0.95, 1.05)])   # inverted
check("negative credit rejected",
      len(build_spreads(chain, 100.0, DTE, params())), 0)

# min return on risk
chain = pd.DataFrame([leg(80, 0.95, 1.05), leg(75, 0.85, 0.95)])
# credit 0.10, max_loss 490, ror = 10/490 = 2.04% -> under a 5% floor
check("below min return on risk rejected",
      len(build_spreads(chain, 100.0, DTE, params(min_ror=5.0))), 0)

# missing DTE must not be annualized on a guess (old build defaulted to 1 day)
chain = pd.DataFrame([leg(80, 0.95, 1.05), leg(75, 0.35, 0.45)])
check("unknown expiry is dropped, not annualized at DTE=1",
      len(build_spreads(chain, 100.0, {}, params())), 0)

print("\n5. Universe")
for t in ("SPY", "QQQ", "IWM", "XLF", "TQQQ", "SOXL", "GLD", "TLT", "spy"):
    if not is_blocked(t):
        FAILURES.append(f"ETF not blocked: {t}")
print(f"  {'PASS' if not [f for f in FAILURES if 'ETF' in f] else 'FAIL'}  "
      "index / sector / leveraged / commodity ETFs blocked")
for t in ("NVDA", "AAPL", "MSTR", "COIN"):
    check(f"single stock {t} allowed", is_blocked(t), False)

# ---------------------------------------------------------------- ranking

print("\n6. Ranking — regimes must not be blended")
mixed = pd.DataFrame([
    {"ticker": "A", "window": "30–45 DTE", "dte": 35, "ror_pct": 9.0,
     "annualized_pct": 93.9, "pricing": "mid"},
    {"ticker": "B", "window": "90–190 DTE", "dte": 140, "ror_pct": 22.0,
     "annualized_pct": 57.4, "pricing": "mid"},
    {"ticker": "C", "window": "30–45 DTE", "dte": 35, "ror_pct": 7.0,
     "annualized_pct": 73.0, "pricing": "mid"},
])
short_sleeve = rank_within_regime(mixed[mixed["window"] == "30–45 DTE"])
check("within-regime ranks by raw ROR", list(short_sleeve["ticker"]), ["A", "C"])
check("short sleeve is not buried by far-dated ROR",
      short_sleeve.iloc[0]["ticker"], "A")
cross = rank_cross_regime(mixed)
check("cross-regime ranks by annualized", list(cross["ticker"]), ["A", "C", "B"])

print("\n6b. Stale-priced rows rank below quoted rows")
tainted = pd.DataFrame([
    {"ticker": "STALE", "ror_pct": 99.0, "pricing": "last",
     "annualized_pct": 999.0},
    {"ticker": "REAL", "ror_pct": 8.0, "pricing": "mid",
     "annualized_pct": 83.0},
])
check("quoted row outranks a better stale row",
      rank_within_regime(tainted).iloc[0]["ticker"], "REAL")

# ------------------------------------------------------- vol and events

print("\n7. Volatility and events")
flat = [100.0] * 40
_rv_flat = rv_from_closes(flat)
check("zero-variance series -> ~0 vol",
      round(_rv_flat, 6) if _rv_flat is not None else None, 0.0)
check("too few points -> None", rv_from_closes([100, 101, 102]), None)
# alternating +1%/-1% daily: |r| = ln(1.01) ~= 0.00995, sd ~= 0.00995 (about
# the mean of ~0), annualized ~= 0.00995 * sqrt(252) ~= 15.8%
alt = [100.0]
for i in range(40):
    alt.append(alt[-1] * (1.01 if i % 2 == 0 else 1 / 1.01))
rv = rv_from_closes(alt)
check("alternating 1% moves -> ~16% annualized", round(rv, 2), 0.16, 0.02)

today = dt.date(2026, 8, 15)
dates = ["2026-08-20", "2026-11-19", "2027-02-18"]
check("earnings inside a 35d window", earnings_in_window(dates, today,
                                                         dt.date(2026, 9, 18)), 1)
check("earnings inside a 140d window", earnings_in_window(dates, today,
                                                          dt.date(2027, 1, 1)), 2)
check("no calendar -> 0, not a crash", earnings_in_window([], today,
                                                          dt.date(2027, 1, 1)), 0)
check("garbage dates ignored", earnings_in_window(["not-a-date"], today,
                                                  dt.date(2027, 1, 1)), 0)

# ------------------------------------------------------------------ result

print()
if FAILURES:
    print(f"FAILED ({len(FAILURES)}): {FAILURES}")
    sys.exit(1)
print("ALL STRATEGY TESTS PASS")
