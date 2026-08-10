"""
Scoring budgets.

Each strategy allocates exactly 100 points across its confirmations, with the
heaviest weights on the links that actually decide whether the setup works. A
perfect setup scores 100, so the default threshold of 90 means "almost
everything must be present".

Keeping the budgets in one table has a practical benefit: the test suite
asserts each column sums to 100, so an edit that quietly inflates a weight
fails loudly instead of silently making every signal look stronger.
"""
from __future__ import annotations

from typing import Dict, Tuple

# --------------------------------------------------------------------------- #
# STRATEGY 1 — Liquidity Sweep + MSS  (reversal)
# --------------------------------------------------------------------------- #
SWEEP_MSS_WEIGHTS: Dict[str, float] = {
    # --- the mandatory chain ---
    "liquidity_sweep":     16.0,   # the stop hunt itself
    "mss":                 15.0,   # displacement break confirming intent flipped
    "poi_entry":           12.0,   # the FVG/OB the displacement left behind
    "htf_context":         10.0,   # sweeping INTO higher-timeframe interest
    "sniper_trigger":       9.0,   # the entry event on the trigger TF
    # --- extra confirmations ---
    "target_liquidity":     7.0,   # somewhere real for price to go
    "premium_discount":     6.0,   # buying discount / selling premium
    "sweep_volume":         6.0,   # size actually traded on the raid
    "inducement":           5.0,   # minor pool taken first — the fuel
    "divergence":           5.0,   # RSI/CVD absorption against price
    "htf_poi_confluence":   4.0,   # the sweep happened at an HTF zone
    "session":              2.0,
    "btc_regime":           2.0,
    "volatility_regime":    1.0,
}

# --------------------------------------------------------------------------- #
# STRATEGY 2 — Order Block + FVG  (continuation)
# --------------------------------------------------------------------------- #
OB_FVG_WEIGHTS: Dict[str, float] = {
    # --- the mandatory chain ---
    "order_block":         15.0,   # unmitigated origin of the impulse
    "fvg":                 13.0,   # the imbalance that must be filled
    "bos_continuation":    13.0,   # trend confirmed, not hoped for
    "htf_trend":           11.0,
    "sniper_trigger":       9.0,
    # --- extra confirmations ---
    "zone_confluence":      7.0,   # OB and FVG overlapping = the A+ pocket
    "target_liquidity":     7.0,
    "fib_confluence":       6.0,   # 0.618-0.79 of the impulse
    "pullback_quality":     6.0,   # retracement on falling volume, not selling
    "inducement":           5.0,   # minor liquidity swept into the zone
    "not_overextended":     4.0,
    "session":              2.0,
    "btc_regime":           1.0,
    "volatility_regime":    1.0,
}


def scaled(base: float, quality: float) -> float:
    """
    Weight a confirmation by how good it is, without letting quality gut it.

    A textbook example earns the full allocation; a marginal one still earns
    75%. Scaling linearly from zero would put the 90-point threshold out of
    reach, since nothing in markets is ever a clean 1.0.
    """
    q = max(0.0, min(1.0, quality))
    return round(base * (0.75 + 0.25 * q), 2)


def budget_total(weights: Dict[str, float]) -> float:
    return round(sum(weights.values()), 2)


def session_now(ts: float) -> Tuple[str, float]:
    """
    Which institutional session is open, and how much that is worth.

    Liquidity and follow-through concentrate in the London and New York
    sessions. The overlap is where the largest moves originate; the Asian
    afternoon is where breakouts go to die.
    """
    import time as _t

    hour = _t.gmtime(ts).tm_hour
    london = 7 <= hour < 16
    newyork = 12 <= hour < 21

    if london and newyork:
        return "London/NY overlap", 1.0
    if newyork:
        return "New York", 0.8
    if london:
        return "London", 0.8
    if 0 <= hour < 7:
        return "Asia", 0.0
    return "Late US", 0.0
