"""Unit tests for the probability engine, sanity gate and market session.

Dependency-free: `python3 spread-scout/tests/test_probability.py`

Reference values are computed from the closed-form Black-Scholes identities or
taken from textbook cases — not from this program's own output.
"""

import datetime as dt
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategy import (  # noqa: E402
    Params, bs_d1_d2, bs_put, build_spreads, dedupe, implied_vol_put,
    market_session, norm_cdf, prob_otm, prob_touch, put_delta, sane_credit,
    spread_ev, spread_ev_crude,
)

ET = ZoneInfo("America/New_York")
FAIL = []


def check(name, got, want, tol=1e-9):
    if isinstance(want, (int, float)) and isinstance(got, (int, float)):
        ok = abs(got - want) <= tol
    else:
        ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got!r}, want {want!r}")
    if not ok:
        FAIL.append(name)


# ------------------------------------------------------------ normal CDF

print("\n1. Normal CDF against known values")
check("N(0)", norm_cdf(0), 0.5, 1e-12)
check("N(1.96)", norm_cdf(1.96), 0.9750021049, 1e-9)
check("N(-1.96)", norm_cdf(-1.96), 0.0249978951, 1e-9)
check("N(1.645)", norm_cdf(1.645), 0.9500150944, 1e-9)
check("N(x)+N(-x)=1", norm_cdf(0.83) + norm_cdf(-0.83), 1.0, 1e-12)

# ------------------------------------------------- Black-Scholes reference

print("\n2. Black-Scholes put — textbook case S=100 K=100 T=1 r=5% q=0 sig=20%")
# d1 = (ln(1) + (0.05 + 0.02)*1) / 0.2 = 0.35 ; d2 = 0.15
# put = 100*e^-0.05*N(-0.15) - 100*N(-0.35) = 95.1229*0.440382 - 0.363169*100
#     = 41.8904 - 36.3169 = 5.5735
d1, d2 = bs_d1_d2(100, 100, 1.0, 0.05, 0.0, 0.20)
check("d1", d1, 0.35, 1e-12)
check("d2", d2, 0.15, 1e-12)
check("put price", bs_put(100, 100, 1.0, 0.05, 0.0, 0.20), 5.5735, 1e-3)
# put-call parity: C - P = S*e^-qT - K*e^-rT
P = bs_put(100, 100, 1.0, 0.05, 0.0, 0.20)
C = P + 100 - 100 * math.exp(-0.05)
check("parity implies call", C, 10.4506, 1e-3)

print("\n2b. Probabilities are complementary and correctly signed")
S, K, T, r, q, sig = 100.0, 80.0, 0.25, 0.04, 0.0, 0.35
pop = prob_otm(S, K, T, r, q, sig)
check("P(OTM) + P(ITM) = 1", pop + (1 - pop), 1.0, 1e-12)
check("P(OTM) in (0,1)", 0 < pop < 1, True)
check("deep OTM short put has high P(OTM)", pop > 0.85, True)
check("delta is negative", put_delta(S, K, T, r, q, sig) < 0, True)
check("|delta| < 1", abs(put_delta(S, K, T, r, q, sig)) < 1, True)
check("P(touch) >= P(ITM)", prob_touch(S, K, T, r, q, sig) >= (1 - pop), True)
check("P(touch) capped at 1", prob_touch(100, 99, 2.0, r, q, 1.5) <= 1.0, True)

print("\n2c. Edge cases: deep OTM, near expiry, high IV")
check("deep OTM, 5d, low IV -> P(OTM) ~ 1",
      round(prob_otm(100, 60, 5 / 365, r, q, 0.20), 6), 1.0, 1e-6)
check("near expiry ATM -> P(OTM) ~ 0.5",
      abs(prob_otm(100, 100, 1 / 365, r, q, 0.30) - 0.5) < 0.02, True)
check("high IV widens the distribution (P(OTM) falls)",
      prob_otm(100, 80, 0.5, r, q, 1.20) < prob_otm(100, 80, 0.5, r, q, 0.25),
      True)
check("T=0 degenerates safely", prob_otm(100, 80, 0, r, q, 0.3), 1.0)

print("\n2d. N(d2) and 1-|delta| diverge (they are different questions)")
# N(d2) is P(finishing OTM); |delta| ~ P(ITM) only under zero drift and no
# vol skew. At high IV and long T the gap is large and must not be conflated.
S, K, T, sig = 100.0, 70.0, 1.0, 0.90
pop = prob_otm(S, K, T, r, q, sig)
dl = abs(put_delta(S, K, T, r, q, sig))
print(f"        N(d2)={pop:.4f}  |delta|={dl:.4f}  gap={abs(pop-(1-dl)):.4f}")
check("gap is material at high IV/long T", abs(pop - (1 - dl)) > 0.05, True)

print("\n2e. Implied vol solve round-trips")
px = bs_put(100, 85, 0.5, 0.04, 0.0, 0.42)
check("solved IV recovers input", round(implied_vol_put(px, 100, 85, 0.5, 0.04, 0.0), 4),
      0.42, 5e-4)
check("nonsense price -> None", implied_vol_put(999, 100, 85, 0.5, 0.04, 0.0), None)
check("zero price -> None", implied_vol_put(0, 100, 85, 0.5, 0.04, 0.0), None)

# ---------------------------------------------------------------- EV

print("\n3. Expected value")
# Priced AT theoretical value, risk-neutral EV must be ~0 by construction.
S, Ks, Kl, T, sig = 100.0, 80.0, 75.0, 0.5, 0.40
fair = bs_put(S, Ks, T, r, q, sig) - bs_put(S, Kl, T, r, q, sig)
ev0 = spread_ev(S, Ks, Kl, fair, T, r, q, sig, sig)
check("EV at fair value ~ 0 (before costs)", abs(ev0) < 0.02, True)
# Collect more than fair -> positive EV of exactly the excess, grown at r.
ev1 = spread_ev(S, Ks, Kl, fair + 0.50, T, r, q, sig, sig)
check("EV rises by the excess credit", round(ev1 - ev0, 2), 50.0, 0.01)
# Same credit but realized vol below implied -> positive EV (variance premium)
ev_rv = spread_ev(S, Ks, Kl, fair, T, r, q, 0.25, 0.25)
check("EV positive when RV < IV (variance risk premium)", ev_rv > 0, True)
print(f"        EV(IV=40%)={ev0:+.2f}   EV(RV=25%)={ev_rv:+.2f}"
      f"   variance premium={ev_rv-ev0:+.2f}")
# The crude version ignores partial losses, so it overstates
pop = prob_otm(S, Ks, T, r, q, sig)
pfl = 1 - prob_otm(S, Kl, T, r, q, sig)
crude = spread_ev_crude(fair, Ks - Kl, pop, pfl)
check("crude EV overstates the full integral", crude > ev0, True)
print(f"        crude={crude:+.2f} vs full integral={ev0:+.2f}"
      f"  (difference is the partial-loss region)")

# --------------------------------------------------------- sanity gate

print("\n4. Sanity gate — the exact GS row from the reported build")
check("GS 815/800 credit 11.85 on a 15 wide is rejected",
      sane_credit(11.85, 15.0, 0.35), False)
check("a plausible 1.92 credit on the same width passes",
      sane_credit(1.92, 15.0, 0.35), True)
check("exactly at the ceiling passes", sane_credit(5.25, 15.0, 0.35), True)
check("a cent over the ceiling fails", sane_credit(5.26, 15.0, 0.35), False)

EXP = "2027-01-15"


def leg(k, bid, ask, px, src="mid", day="2026-08-14", oi=900, iv=0.35, spc=100):
    return dict(ticker="GS", exp=EXP, strike=float(k), px=px, px_src=src,
                mark_day=day, bid=bid, ask=ask,
                ba_pct=((ask - bid) / ((ask + bid) / 2))
                if (bid and ask and ask > bid) else None,
                delta=-0.06, iv=iv, oi=oi, spc=spc)


def params(**kw):
    base = dict(otm_lo=0.18, otm_hi=0.22, risk_cap=2200.0, min_ror=5.0,
                min_oi_short=100, min_oi_long=100, max_spread_pct=0.40,
                min_width=2.5, min_net_premium=0.0, min_pop=0.0,
                max_short_delta=1.0, market_open=True)
    base.update(kw)
    return Params(**base)


print("\n4b. The GS row cannot be constructed end to end")
rej = {}
chain = pd.DataFrame([leg(815, 11.80, 11.90, 11.85), leg(800, 0.0, 0.10, 0.0)])
got = build_spreads(chain, 1039.42, {EXP: 153}, params(), rejects=rej)
check("no candidate produced", len(got), 0)
check("rejected with a logged reason", bool(rej), True)
print(f"        reject reasons: {rej}")

print("\n4c. A leg priced at zero is rejected even when the other is fine")
rej = {}
chain = pd.DataFrame([leg(815, None, None, 11.85, src="close"),
                      leg(800, None, None, 0.0, src="close")])
build_spreads(chain, 1039.42, {EXP: 153},
              params(market_open=False), rejects=rej)
check("zero-priced leg logged", "leg priced at zero" in rej, True)

print("\n4d. Marks from different sessions are refused when closed")
rej = {}
chain = pd.DataFrame([leg(815, None, None, 11.85, src="close", day="2026-08-14"),
                      leg(800, None, None, 9.90, src="close", day="2026-07-02")])
build_spreads(chain, 1039.42, {EXP: 153},
              params(market_open=False), rejects=rej)
check("cross-session pairing logged", "marks from different sessions" in rej, True)

print("\n4e. When the market is open, a leg with no live quote is rejected")
rej = {}
chain = pd.DataFrame([leg(815, None, None, 11.85, src="close"),
                      leg(800, None, None, 9.90, src="close")])
build_spreads(chain, 1039.42, {EXP: 153},
              params(market_open=True), rejects=rej)
check("no-live-quote logged", "no live two-sided quote" in rej, True)

print("\n4f. Non-standard deliverables never pair with standard ones")
rej = {}
chain = pd.DataFrame([leg(815, 11.0, 11.2, 11.1, spc=100),
                      leg(800, 9.0, 9.2, 9.1, spc=10)])
build_spreads(chain, 1039.42, {EXP: 153}, params(), rejects=rej)
check("deliverable mismatch logged", "non-standard deliverable" in rej, True)

# --------------------------------------------------- premium floor

print("\n5. Minimum net premium, at the boundary")
# width 5, credit 0.41 -> max_loss 459, qty = floor(2200/459) = 4,
# net premium = 0.41 * 100 * 4 = $164  (below)
# credit 0.50 -> max_loss 450, qty = 4, premium = $200 exactly (at)
base = dict(otm_lo=0.18, otm_hi=0.22, risk_cap=2200.0, min_ror=1.0,
            min_oi_short=100, min_oi_long=100, max_spread_pct=0.99,
            min_width=2.5, min_pop=0.0, max_short_delta=1.0, market_open=True)
for short_px, want in ((0.55, 1), (0.54, 0)):
    ch = pd.DataFrame([leg(80, short_px - 0.02, short_px + 0.02, short_px),
                       leg(75, 0.04, 0.06, 0.05)])
    rej = {}
    got = build_spreads(ch, 100.0, {EXP: 153},
                        Params(min_net_premium=200.0, **base), rejects=rej)
    qty = int(got.iloc[0]["contracts"]) if len(got) else 0
    prem = float(got.iloc[0]["total_credit"]) if len(got) else 0
    label = "at $200" if want else "a cent below"
    check(f"premium floor {label}", len(got), want)
    if len(got):
        print(f"        qty={qty} premium=${prem:.0f}")

print("\n5b. The floor is applied to the POSITION total, not per contract")
print("        (credit x 100 x contracts). A per-contract reading would let a")
print("        $0.50 single-contract spread through at $50.")

# ------------------------------------------------------ market session

print("\n6. Market session detection")
cases = [
    (dt.datetime(2026, 8, 15, 4, 57, tzinfo=ET), "closed", "weekend",
     "the reported scan time"),
    (dt.datetime(2026, 8, 14, 11, 0, tzinfo=ET), "open", "", "mid-session Friday"),
    (dt.datetime(2026, 8, 14, 9, 0, tzinfo=ET), "closed", "pre-market", "before the bell"),
    (dt.datetime(2026, 8, 14, 16, 30, tzinfo=ET), "closed", "after-hours", "after the bell"),
    (dt.datetime(2026, 12, 25, 12, 0, tzinfo=ET), "closed", "holiday", "Christmas"),
    (dt.datetime(2026, 7, 3, 12, 0, tzinfo=ET), "closed", "holiday", "July 3 observed"),
]
for when, state, reason, label in cases:
    m = market_session(when)
    check(f"{label} -> {state}", (m["state"], m["reason"]), (state, reason))

print("\n6b. Early close: 13:00 ET on the day after Thanksgiving")
m_before = market_session(dt.datetime(2026, 11, 27, 12, 30, tzinfo=ET))
m_after = market_session(dt.datetime(2026, 11, 27, 13, 30, tzinfo=ET))
check("12:30 on a half day is open", m_before["state"], "open")
check("13:30 on a half day is closed", m_after["state"], "closed")
check("flagged as an early close", m_before["early_close"], True)

print("\n6c. Next open and last session")
m = market_session(dt.datetime(2026, 8, 15, 4, 57, tzinfo=ET))
check("next open is Monday 09:30", m["next_open"].strftime("%a %H:%M"), "Mon 09:30")
check("marks are Friday's", m["last_session"].isoformat(), "2026-08-14")

# ------------------------------------------------------------- dedupe

print("\n7. Dedupe and concentration cap")
rows = []
for i, (tk, exp, ror) in enumerate([
        ("MU", "2027-01-15", 30), ("MU", "2027-01-15", 25), ("MU", "2027-01-15", 20),
        ("MU", "2027-02-19", 28), ("MU", "2027-03-19", 26),
        ("GS", "2027-01-15", 22), ("GS", "2027-01-15", 18)]):
    rows.append({"ticker": tk, "exp": exp, "ror_pct": ror, "pricing": "mid",
                 "short_k": 100 + i, "long_k": 95 + i, "ev_per_collateral": ror / 100})
df = pd.DataFrame(rows)
out = dedupe(df, "Return on risk", max_per_underlying=2)
check("one row per (ticker, expiry) then capped at 2 per name", len(out), 3)
check("MU appears twice, not five times",
      int((out["ticker"] == "MU").sum()), 2)
check("the best MU row survives", float(out[out["ticker"] == "MU"]["ror_pct"].max()), 30.0)

print()
if FAIL:
    print(f"FAILED ({len(FAIL)}): {FAIL}")
    sys.exit(1)
print("ALL PROBABILITY / GATE / SESSION TESTS PASS")
