"""
Vectorised indicators built on pandas/numpy only (no TA-Lib / C build steps).
Every function is NaN-safe and returns a Series aligned to the input index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# moving averages
# --------------------------------------------------------------------------- #
def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False, min_periods=1).mean()


def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean()


def wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True)


def hma(series: pd.Series, length: int) -> pd.Series:
    half = max(1, length // 2)
    sqrt_len = max(1, int(np.sqrt(length)))
    return wma(2 * wma(series, half) - wma(series, length), sqrt_len)


# --------------------------------------------------------------------------- #
# volatility
# --------------------------------------------------------------------------- #
def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.fillna(df["high"] - df["low"])


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / length, adjust=False, min_periods=1).mean()


def bollinger(series: pd.Series, length: int = 20, mult: float = 2.0):
    mid = sma(series, length)
    std = series.rolling(length, min_periods=1).std(ddof=0)
    return mid + mult * std, mid, mid - mult * std


def bb_width(series: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    up, mid, low = bollinger(series, length, mult)
    return ((up - low) / mid.replace(0, np.nan)).fillna(0) * 100


def keltner(df: pd.DataFrame, length: int = 20, mult: float = 1.5):
    mid = ema(df["close"], length)
    rng = atr(df, length) * mult
    return mid + rng, mid, mid - rng


# --------------------------------------------------------------------------- #
# momentum
# --------------------------------------------------------------------------- #
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False, min_periods=1).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def stoch_rsi(series: pd.Series, length: int = 14, k: int = 3, d: int = 3):
    r = rsi(series, length)
    lo = r.rolling(length, min_periods=1).min()
    hi = r.rolling(length, min_periods=1).max()
    raw = ((r - lo) / (hi - lo).replace(0, np.nan) * 100).fillna(50.0)
    k_line = raw.rolling(k, min_periods=1).mean()
    d_line = k_line.rolling(d, min_periods=1).mean()
    return k_line, d_line


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    line = ema(series, fast) - ema(series, slow)
    sig = ema(line, signal)
    return line, sig, line - sig


def momentum(series: pd.Series, length: int = 10) -> pd.Series:
    return series.diff(length).fillna(0)


def cci(df: pd.DataFrame, length: int = 20) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    ma = tp.rolling(length, min_periods=1).mean()
    md = (tp - ma).abs().rolling(length, min_periods=1).mean()
    return ((tp - ma) / (0.015 * md.replace(0, np.nan))).fillna(0)


# --------------------------------------------------------------------------- #
# trend strength
# --------------------------------------------------------------------------- #
def adx(df: pd.DataFrame, length: int = 14):
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr = true_range(df)
    atr_ = tr.ewm(alpha=1 / length, adjust=False, min_periods=1).mean()
    atr_safe = atr_.replace(0, np.nan)

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / length, adjust=False, min_periods=1).mean() / atr_safe
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / length, adjust=False, min_periods=1).mean() / atr_safe

    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_ = dx.ewm(alpha=1 / length, adjust=False, min_periods=1).mean()
    return adx_.fillna(0), plus_di.fillna(0), minus_di.fillna(0)


def supertrend(df: pd.DataFrame, length: int = 10, mult: float = 3.0):
    """Returns (trend_direction[+1/-1], supertrend_line)."""
    hl2 = (df["high"] + df["low"]) / 2
    a = atr(df, length)
    upper = (hl2 + mult * a).to_numpy()
    lower = (hl2 - mult * a).to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    final_up = np.zeros(n)
    final_lo = np.zeros(n)
    direction = np.ones(n)

    final_up[0], final_lo[0] = upper[0], lower[0]
    for i in range(1, n):
        final_up[i] = (min(upper[i], final_up[i - 1])
                       if close[i - 1] <= final_up[i - 1] else upper[i])
        final_lo[i] = (max(lower[i], final_lo[i - 1])
                       if close[i - 1] >= final_lo[i - 1] else lower[i])
        if close[i] > final_up[i - 1]:
            direction[i] = 1
        elif close[i] < final_lo[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    line = np.where(direction == 1, final_lo, final_up)
    return (pd.Series(direction, index=df.index),
            pd.Series(line, index=df.index))


# --------------------------------------------------------------------------- #
# volume / price-volume
# --------------------------------------------------------------------------- #
def vwap_rolling(df: pd.DataFrame, length: int = 60) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(length, min_periods=1).sum()
    vol = df["volume"].rolling(length, min_periods=1).sum().replace(0, np.nan)
    return (pv / vol).ffill().fillna(df["close"])


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP anchored to each UTC day."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    day = pd.Series(df.index.date, index=df.index) if isinstance(
        df.index, pd.DatetimeIndex) else pd.Series(0, index=df.index)
    pv = (tp * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return (pv / vol).ffill().fillna(df["close"])


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["close"].diff().fillna(0))
    return (sign * df["volume"]).cumsum()


def volume_zscore(df: pd.DataFrame, length: int = 50) -> pd.Series:
    v = df["volume"]
    mean = v.rolling(length, min_periods=5).mean()
    std = v.rolling(length, min_periods=5).std(ddof=0).replace(0, np.nan)
    return ((v - mean) / std).fillna(0)


def money_flow_index(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    rmf = tp * df["volume"]
    delta = tp.diff()
    pos = rmf.where(delta > 0, 0.0).rolling(length, min_periods=1).sum()
    neg = rmf.where(delta < 0, 0.0).rolling(length, min_periods=1).sum()
    ratio = pos / neg.replace(0, np.nan)
    return (100 - 100 / (1 + ratio)).fillna(50.0)


def cvd_proxy(df: pd.DataFrame) -> pd.Series:
    """
    Cumulative volume delta proxy from OHLCV: splits each candle's volume by
    where the close sits inside the range. Directionally reliable enough for
    divergence detection without needing tick data.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = ((df["close"] - df["low"]) / rng).clip(0, 1).fillna(0.5)
    delta = df["volume"] * (2 * ratio - 1)
    return delta.cumsum()


# --------------------------------------------------------------------------- #
# candles
# --------------------------------------------------------------------------- #
def body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


def body_ratio(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    return (body(df) / rng).fillna(0)


def upper_wick(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df[["open", "close"]].max(axis=1)


def lower_wick(df: pd.DataFrame) -> pd.Series:
    return df[["open", "close"]].min(axis=1) - df["low"]


def is_bullish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    return ((df["close"] > df["open"]) & (prev_c < prev_o) &
            (df["close"] >= prev_o) & (df["open"] <= prev_c)).fillna(False)


def is_bearish_engulfing(df: pd.DataFrame) -> pd.Series:
    prev_o, prev_c = df["open"].shift(1), df["close"].shift(1)
    return ((df["close"] < df["open"]) & (prev_c > prev_o) &
            (df["close"] <= prev_o) & (df["open"] >= prev_c)).fillna(False)


def is_pin_bar(df: pd.DataFrame, direction: str = "bull", ratio: float = 2.0) -> pd.Series:
    b = body(df).replace(0, np.nan)
    if direction == "bull":
        return ((lower_wick(df) / b >= ratio) &
                (upper_wick(df) < body(df))).fillna(False)
    return ((upper_wick(df) / b >= ratio) &
            (lower_wick(df) < body(df))).fillna(False)


# --------------------------------------------------------------------------- #
# bundle
# --------------------------------------------------------------------------- #
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the standard indicator set used across all strategies."""
    if df is None or len(df) < 25:
        return df

    out = df.copy()
    c = out["close"]

    out["ema9"] = ema(c, 9)
    out["ema21"] = ema(c, 21)
    out["ema50"] = ema(c, 50)
    out["ema100"] = ema(c, 100)
    out["ema200"] = ema(c, 200)

    out["atr"] = atr(out, 14)
    out["atr_pct"] = (out["atr"] / c.replace(0, np.nan) * 100).fillna(0)

    out["rsi"] = rsi(c, 14)
    out["stoch_k"], out["stoch_d"] = stoch_rsi(c, 14)
    out["macd"], out["macd_sig"], out["macd_hist"] = macd(c)
    out["adx"], out["di_plus"], out["di_minus"] = adx(out, 14)

    out["bb_up"], out["bb_mid"], out["bb_low"] = bollinger(c, 20, 2.0)
    out["bb_width"] = bb_width(c, 20, 2.0)

    out["vwap"] = vwap_rolling(out, 60)
    out["vol_ma"] = sma(out["volume"], 20)
    out["vol_z"] = volume_zscore(out, 50)
    out["mfi"] = money_flow_index(out, 14)
    out["cvd"] = cvd_proxy(out)
    out["obv"] = obv(out)

    out["body_ratio"] = body_ratio(out)
    out["upper_wick"] = upper_wick(out)
    out["lower_wick"] = lower_wick(out)
    out["st_dir"], out["st_line"] = supertrend(out, 10, 3.0)

    return out
