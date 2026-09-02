"""
Stock Move Probability Analyzer
--------------------------------
Given an NSE-listed stock, a target percentage move, and a number of
trading days, this tool checks how "possible" that move is by combining:

  1. Historical frequency  - how often has a move of this size (or bigger)
                              actually happened over N-day windows, historically?
  2. Binomial CI (Wilson)  - how much should we trust that frequency, given
                              how many independent-ish samples we have?
  3. Chi-square test       - does this stock's return distribution actually
                              behave like a normal distribution? If not, any
                              probability calculated from a normal-curve
                              assumption (the usual "z-score" approach) is
                              unreliable, and the historical frequency should
                              be trusted more than the theoretical one.

Data source: Yahoo Finance via yfinance (ticker.NS for NSE equities).

Usage:
    python stock_move_analysis.py

Then follow the interactive prompts.
"""

import sys
import math
import csv
import argparse
import difflib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    print("This script needs yfinance. Install it with:\n    pip install yfinance")
    sys.exit(1)

# readline gives tab-completion in the interactive prompt. It's built into
# Python on macOS/Linux. On Windows it's not available by default -- install
# `pyreadline3` (pip install pyreadline3) to get the same behaviour there.
try:
    import readline
    HAVE_READLINE = True
except ImportError:
    HAVE_READLINE = False


# --------------------------------------------------------------------------
# Ticker list handling (optional — only used to validate / look up names)
# --------------------------------------------------------------------------

def load_ticker_map(csv_path: str) -> dict:
    """Load SYMBOL -> NAME OF COMPANY from the NSE equity list CSV.
    Returns an empty dict (and prints a warning) if the file isn't found —
    the script still works, it just can't show you the company name."""
    mapping = {}
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sym = row.get("SYMBOL", "").strip()
                name = row.get("NAME OF COMPANY", "").strip()
                if sym:
                    mapping[sym.upper()] = name
    except FileNotFoundError:
        print(f"(Note: ticker list '{csv_path}' not found — skipping name lookup.)")
    return mapping


class SymbolCompleter:
    """Tab-completion source for readline: matches on ticker symbol prefix,
    and also lets you tab-complete by typing the start of the company name."""

    def __init__(self, ticker_map: dict):
        self.symbols = sorted(ticker_map.keys())
        # allow completing by company name too, mapped back to its symbol
        self.name_to_symbol = {
            name.upper(): sym for sym, name in ticker_map.items() if name
        }
        self._matches = []

    def complete(self, text, state):
        text_u = text.upper()
        if state == 0:
            by_symbol = [s for s in self.symbols if s.startswith(text_u)]
            by_name = [
                f"{sym} ({name})"
                for name, sym in self.name_to_symbol.items()
                if name.startswith(text_u)
            ]
            self._matches = by_symbol + by_name
        try:
            return self._matches[state]
        except IndexError:
            return None


def enable_autofill(ticker_map: dict):
    """Wire up readline so pressing Tab while typing a symbol autocompletes
    it from the loaded ticker CSV. No-op if readline isn't available
    (e.g. Windows without pyreadline3) or the ticker map is empty."""
    if not HAVE_READLINE or not ticker_map:
        return
    completer = SymbolCompleter(ticker_map)
    readline.set_completer(completer.complete)
    # treat these as word boundaries so it completes the whole symbol/name
    readline.set_completer_delims(" \t\n")
    readline.parse_and_bind("tab: complete")


def resolve_symbol(raw_input: str, ticker_map: dict) -> str:
    """Cleans up user input and, if it doesn't match the ticker list exactly,
    suggests close matches (fuzzy match on symbol and on company name)."""
    symbol = raw_input.strip().upper()

    # allow "SYMBOL (Company Name)" pasted straight from a tab-completion
    if "(" in symbol:
        symbol = symbol.split("(")[0].strip()

    if not ticker_map or symbol in ticker_map:
        return symbol

    # try matching against company names typed in full or partially
    name_matches = difflib.get_close_matches(
        symbol, [n.upper() for n in ticker_map.values() if n], n=3, cutoff=0.6
    )
    symbol_matches = difflib.get_close_matches(symbol, ticker_map.keys(), n=3, cutoff=0.6)

    if symbol_matches or name_matches:
        print(f"'{symbol}' not found in the ticker list exactly. Did you mean:")
        seen = set()
        options = []
        for s in symbol_matches:
            if s not in seen:
                seen.add(s)
                options.append(s)
        for n in name_matches:
            for sym, name in ticker_map.items():
                if name.upper() == n and sym not in seen:
                    seen.add(sym)
                    options.append(sym)
        for i, opt in enumerate(options, 1):
            print(f"    {i}. {opt} — {ticker_map.get(opt, '')}")
        choice = input(f"Pick a number to use it, or press Enter to keep '{symbol}': ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
    else:
        print(f"'{symbol}' not found in the local NSE ticker list — proceeding anyway "
              f"(it may still be a valid Yahoo Finance symbol).")

    return symbol


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------

@dataclass
class MoveAnalysisResult:
    symbol: str
    n_days: int
    pct_move: float
    direction: str
    mode: str
    total_windows: int
    hit_count: int
    p_hat: float
    wilson_lo: float
    wilson_hi: float
    hist_mean: float
    hist_std: float
    z_score: float
    normal_prob: float
    chi2_stat: float
    chi2_pvalue: float
    chi2_dof: int
    normal_fits: bool
    last_price: float
    high_pct_percentiles: dict    # percentiles of the BEST intra-window % swing, historically
    low_pct_percentiles: dict     # percentiles of the WORST intra-window % swing, historically
    high_price_percentiles: dict  # same, converted to a price using today's last close
    low_price_percentiles: dict


def wilson_interval(count: int, total: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion.
    More reliable than the plain normal-approximation interval,
    especially when the hit count is small (rare-event tails)."""
    if total == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = count / total
    denom = 1 + z**2 / total
    centre = p + z**2 / (2 * total)
    adj = z * math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total)
    lo = (centre - adj) / denom
    hi = (centre + adj) / denom
    return max(0.0, lo), min(1.0, hi)


PERCENTILE_QS = (5, 25, 50, 75, 95)


def percentile_dict(values: np.ndarray, qs=PERCENTILE_QS) -> dict:
    """{q: percentile value} for each q in qs. Returns NaN for every q if
    `values` is empty, so downstream formatting doesn't have to special-case
    an empty array."""
    values = np.asarray(values)
    if values.size == 0:
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(values, q)) for q in qs}


def compute_window_paths(close: pd.Series, n_days: int) -> np.ndarray:
    """
    Builds every overlapping N-trading-day window and returns the full
    day-by-day cumulative % return path within each window, as a 2D array
    of shape (num_windows, n_days).

    Row i is the window starting at trading day i; column o (0-indexed) is
    the % return from the window's starting price to o+1 trading days
    later. Column -1 (the last column) is therefore the "endpoint" return
    -- equivalent to the old close.pct_change(periods=n_days) approach.
    The full row is what 'touch' mode needs, since it has to know whether
    the threshold was crossed on *any* day in the window, not just the last.
    """
    prices = close.to_numpy(dtype=float)
    if len(prices) <= n_days:
        return np.empty((0, n_days))
    windows = sliding_window_view(prices, window_shape=n_days + 1)  # (num_windows, n_days+1)
    base = windows[:, :1]
    future = windows[:, 1:]
    return (future / base - 1.0) * 100.0


def compute_rolling_returns(close: pd.Series, n_days: int) -> pd.Series:
    """Overlapping N-trading-day ENDPOINT percentage returns (price N days
    later vs. today) across the whole history. Kept for backwards
    compatibility / use in the normal-model & chi-square sections, which
    are about the endpoint return distribution regardless of which mode
    is used to count hits."""
    returns = close.pct_change(periods=n_days) * 100.0
    return returns.dropna()


def direction_hits(returns: pd.Series, pct_move: float, direction: str) -> int:
    """Endpoint-mode hit count: was the price N days later past the target?"""
    if direction == "up":
        return int((returns >= pct_move).sum())
    elif direction == "down":
        return int((returns <= -pct_move).sum())
    else:  # either direction
        return int((np.abs(returns) >= pct_move).sum())


def direction_hits_touch(paths: np.ndarray, pct_move: float, direction: str) -> int:
    """
    Touch-mode hit count: was the target move reached on ANY day within
    the window, even if the price settled back down by the window's end?
    `paths` is the (num_windows, n_days) array from compute_window_paths.
    """
    if paths.size == 0:
        return 0
    max_ret = paths.max(axis=1)
    min_ret = paths.min(axis=1)
    if direction == "up":
        return int((max_ret >= pct_move).sum())
    elif direction == "down":
        return int((min_ret <= -pct_move).sum())
    else:  # either direction: touched +X% OR -X% at any point
        return int(((max_ret >= pct_move) | (min_ret <= -pct_move)).sum())


def chi_square_normality(returns: pd.Series, pct_move: float, direction: str):
    """
    Bins the observed N-day returns into 3 buckets split at the move
    thresholds relevant to the question being asked, and compares the
    observed counts to the counts a normal distribution (fit to this
    stock's own historical mean/std) would predict.

    A significant chi-square result (small p-value) means the stock's
    actual return distribution does NOT behave like a normal curve near
    this threshold (e.g. fat tails) -- so a plain z-score/normal-CDF
    probability for this move should not be trusted as much as the
    historical (empirical) frequency.
    """
    mean = returns.mean()
    std = returns.std(ddof=1)

    if direction == "up":
        edges = [-np.inf, pct_move, np.inf]
    elif direction == "down":
        edges = [-np.inf, -pct_move, np.inf]
    else:
        edges = [-np.inf, -pct_move, pct_move, np.inf]

    observed, _ = np.histogram(returns, bins=edges)

    # expected counts under Normal(mean, std) fit to this stock's history
    cdf_vals = [stats.norm.cdf(e, loc=mean, scale=std) if np.isfinite(e) else (0.0 if e < 0 else 1.0)
                for e in edges]
    expected_props = np.diff(cdf_vals)
    expected = expected_props * len(returns)

    # Guard against expected-count-too-small warnings by merging tiny bins
    # (rare in practice here since we typically only have 2-3 bins)
    chi2_stat, p_value = stats.chisquare(f_obs=observed, f_exp=expected)
    dof = len(observed) - 1
    return chi2_stat, p_value, dof, mean, std


# Index/ticker symbols that yfinance already recognizes as-is (or via a "^"
# prefix rather than an NSE-equity ".NS" suffix) -- appending ".NS" to these
# would break them. Add more here if you hit the same issue with another
# index symbol.
NON_NS_SYMBOLS = {
    "NSEI",       # Nifty 50
    "NSEBANK",    # Bank Nifty
    "NSMIDCP",    # Nifty Midcap Select
    "CNX100",     # Nifty 100
    "CNXIT",      # Nifty IT
    "INDIAVIX",   # India VIX
}


def fetch_price_series(symbol: str, period: str = "10y") -> pd.Series:
    """Downloads daily close prices for a symbol via yfinance.
    Plain equity symbols get NSE's '.NS' suffix appended (e.g. RELIANCE ->
    RELIANCE.NS). Symbols that are already index/full tickers -- starting
    with '^', already ending in '.NS', or in NON_NS_SYMBOLS (e.g. NSEI,
    which yfinance expects bare, not NSEI.NS) -- are passed through as-is."""
    sym_u = symbol.strip().upper()
    if sym_u.startswith("^") or sym_u.endswith(".NS") or sym_u in NON_NS_SYMBOLS:
        yf_symbol = sym_u
    else:
        yf_symbol = f"{sym_u}.NS"
    df = yf.download(yf_symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for '{yf_symbol}'. Check the symbol.")
    close = df["Close"].dropna()
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col DataFrame
        close = close.iloc[:, 0]
    return close


def analyze(symbol: str, pct_move: float, n_days: int, direction: str,
            period: str = "10y", close: pd.Series = None,
            mode: str = "touch") -> MoveAnalysisResult:
    """
    mode='endpoint' (default): a window counts as a "hit" only if the price
        exactly n_days later is past the target -- ignores anything that
        happened in between.
    mode='touch': a window counts as a "hit" if the target was reached on
        ANY day within the window, even if the price came back down (or up)
        by the end. This directly answers "could this move have happened at
        some point?" rather than "did it end up there?".
    """
    if mode not in ("endpoint", "touch"):
        raise ValueError("mode must be 'endpoint' or 'touch'")

    if close is None:
        close = fetch_price_series(symbol, period)

    last_price = float(close.iloc[-1])

    # Endpoint returns are always computed -- they're what the normal-model
    # and chi-square sections analyze, regardless of which mode is used to
    # count hits.
    endpoint_returns = compute_rolling_returns(close, n_days)
    total = len(endpoint_returns)
    if total < 30:
        print(f"Warning: only {total} overlapping {n_days}-day windows available. "
              f"Results will be statistically weak.")

    # Full within-window paths are always computed too, both for touch-mode
    # hit counting AND for the "possible high/low" price levels below --
    # those are useful regardless of which mode is selected.
    paths = compute_window_paths(close, n_days)

    if mode == "endpoint":
        hits = direction_hits(endpoint_returns, pct_move, direction)
    else:
        hits = direction_hits_touch(paths, pct_move, direction)

    p_hat = hits / total if total else 0.0
    lo, hi = wilson_interval(hits, total)

    chi2_stat, chi2_p, dof, mean, std = chi_square_normality(endpoint_returns, pct_move, direction)
    normal_fits = chi2_p >= 0.05  # fail to reject normality at 5% level

    z = (pct_move - mean) / std if std > 0 else float("nan")
    if direction == "up":
        normal_prob = 1 - stats.norm.cdf(pct_move, loc=mean, scale=std)
    elif direction == "down":
        normal_prob = stats.norm.cdf(-pct_move, loc=mean, scale=std)
    else:
        normal_prob = (1 - stats.norm.cdf(pct_move, loc=mean, scale=std)) + \
                       stats.norm.cdf(-pct_move, loc=mean, scale=std)

    # "Possible high/low" -- across every historical N-day window, what was
    # the best (highest) and worst (lowest) point reached at any time within
    # the window? Percentiles of those per-window extremes, applied to
    # TODAY's actual price, give a historically-grounded price range for
    # what the stock could plausibly swing to over the next n_days.
    window_high_pct = paths.max(axis=1) if paths.size else np.array([])
    window_low_pct = paths.min(axis=1) if paths.size else np.array([])
    high_pct_percentiles = percentile_dict(window_high_pct)
    low_pct_percentiles = percentile_dict(window_low_pct)
    high_price_percentiles = {q: last_price * (1 + v / 100.0) for q, v in high_pct_percentiles.items()}
    low_price_percentiles = {q: last_price * (1 + v / 100.0) for q, v in low_pct_percentiles.items()}

    return MoveAnalysisResult(
        symbol=symbol.upper(), n_days=n_days, pct_move=pct_move, direction=direction,
        mode=mode, total_windows=total, hit_count=hits, p_hat=p_hat, wilson_lo=lo, wilson_hi=hi,
        hist_mean=mean, hist_std=std, z_score=z, normal_prob=normal_prob,
        chi2_stat=chi2_stat, chi2_pvalue=chi2_p, chi2_dof=dof, normal_fits=normal_fits,
        last_price=last_price, high_pct_percentiles=high_pct_percentiles,
        low_pct_percentiles=low_pct_percentiles, high_price_percentiles=high_price_percentiles,
        low_price_percentiles=low_price_percentiles,
    )


# --------------------------------------------------------------------------
# Monte Carlo simulation — forward-looking forecast
# --------------------------------------------------------------------------

@dataclass
class SimulationResult:
    symbol: str
    method: str
    mode: str
    n_sims: int
    n_days: int
    pct_move: float
    direction: str
    last_price: float
    hit_count: int
    p_sim: float
    wilson_lo: float
    wilson_hi: float
    price_percentiles: dict     # {5: .., 25: .., 50: .., 75: .., 95: ..}  -- day-n_days ENDpoint price
    pct_move_percentiles: dict  # same keys, in % terms
    most_probable_price: float  # mode of the simulated final-price distribution
    most_probable_pct_move: float
    period_high_price_percentiles: dict  # highest price reached at ANY point in the simulated period
    period_low_price_percentiles: dict   # lowest price reached at ANY point in the simulated period
    period_high_pct_percentiles: dict    # same, in % terms
    period_low_pct_percentiles: dict


def estimate_mode(values: np.ndarray) -> float:
    """
    Finds the most probable (highest-density) value in a distribution of
    simulated outcomes, using a Gaussian kernel density estimate.

    This is NOT the same as the median: compounded returns produce a
    right-skewed price distribution (a stock can only fall to 0 but can
    rise indefinitely), so the single most likely outcome typically sits
    a bit below the median/mean.
    """
    try:
        kde = stats.gaussian_kde(values)
        grid = np.linspace(values.min(), values.max(), 2000)
        density = kde(grid)
        return float(grid[np.argmax(density)])
    except Exception:
        # fallback: histogram-based mode if KDE fails (e.g. near-zero variance)
        counts, edges = np.histogram(values, bins=min(100, max(10, len(values) // 50)))
        peak_bin = np.argmax(counts)
        return float((edges[peak_bin] + edges[peak_bin + 1]) / 2)


def run_simulation(close: pd.Series, n_days: int, pct_move: float, direction: str,
                    n_sims: int = 10000, method: str = "bootstrap",
                    seed: int = None, mode: str = "touch") -> SimulationResult:
    """
    Simulates n_sims independent N-trading-day price paths starting from
    TODAY's actual last close, and measures what fraction of those
    simulated futures hit the target move.

    method='bootstrap' (recommended): each simulated day's return is drawn
        (with replacement) from the stock's own actual historical daily
        returns. This preserves whatever fat-tail / skew behaviour the
        chi-square test found, unlike an assumed normal curve.
    method='normal': each simulated day's return is drawn from
        Normal(historical mean, historical std) — the classic textbook
        Monte Carlo / GBM approach. Included for side-by-side comparison.

    mode='endpoint' (default): a simulated path counts as a "hit" only if
        its price on day n_days is past the target.
    mode='touch': a simulated path counts as a "hit" if the target was
        crossed on ANY simulated day within the path, not just the last.
        Note this only affects the hit-rate (p_sim); the forecast price
        percentiles below are always the day-n_days (endpoint) distribution,
        since "what will the price probably be" is inherently an endpoint
        question.
    """
    if mode not in ("endpoint", "touch"):
        raise ValueError("mode must be 'endpoint' or 'touch'")

    daily_returns = close.pct_change().dropna().values  # decimal, e.g. 0.012 = 1.2%
    last_price = float(close.iloc[-1])
    rng = np.random.default_rng(seed)

    if method == "bootstrap":
        sampled = rng.choice(daily_returns, size=(n_sims, n_days), replace=True)
    elif method == "normal":
        mu, sigma = daily_returns.mean(), daily_returns.std(ddof=1)
        sampled = rng.normal(mu, sigma, size=(n_sims, n_days))
    else:
        raise ValueError("method must be 'bootstrap' or 'normal'")

    cum_growth = np.cumprod(1 + sampled, axis=1)          # (n_sims, n_days)
    path_pct_moves = (cum_growth - 1.0) * 100.0            # full within-path % move, every day
    final_prices = last_price * cum_growth[:, -1]
    pct_moves = path_pct_moves[:, -1]                      # endpoint % move (day n_days only)

    # Always compute the per-path high/low (highest and lowest cumulative
    # % move reached at any point within the simulated period) -- needed
    # for touch-mode hit counting, and also reported directly as the
    # "possible high/low" price range regardless of which mode is active.
    max_path = path_pct_moves.max(axis=1)
    min_path = path_pct_moves.min(axis=1)

    if mode == "endpoint":
        target = pct_moves
        if direction == "up":
            hits = int((target >= pct_move).sum())
        elif direction == "down":
            hits = int((target <= -pct_move).sum())
        else:
            hits = int((np.abs(target) >= pct_move).sum())
    else:  # touch
        if direction == "up":
            hits = int((max_path >= pct_move).sum())
        elif direction == "down":
            hits = int((min_path <= -pct_move).sum())
        else:
            hits = int(((max_path >= pct_move) | (min_path <= -pct_move)).sum())

    p_sim = hits / n_sims
    lo, hi = wilson_interval(hits, n_sims)

    price_pcts = percentile_dict(final_prices)
    move_pcts = percentile_dict(pct_moves)
    mode_price = estimate_mode(final_prices)
    mode_pct_move = (mode_price / last_price - 1) * 100.0

    # "Possible high/low" over the whole period: percentiles of the running
    # max/min price reached by each simulated path, not just where it ends
    # up on day n_days.
    period_high_pct_pcts = percentile_dict(max_path)
    period_low_pct_pcts = percentile_dict(min_path)
    period_high_price_pcts = {q: last_price * (1 + v / 100.0) for q, v in period_high_pct_pcts.items()}
    period_low_price_pcts = {q: last_price * (1 + v / 100.0) for q, v in period_low_pct_pcts.items()}

    return SimulationResult(
        symbol="", method=method, mode=mode, n_sims=n_sims, n_days=n_days, pct_move=pct_move,
        direction=direction, last_price=last_price, hit_count=hits, p_sim=p_sim,
        wilson_lo=lo, wilson_hi=hi, price_percentiles=price_pcts,
        pct_move_percentiles=move_pcts, most_probable_price=mode_price,
        most_probable_pct_move=mode_pct_move,
        period_high_price_percentiles=period_high_price_pcts,
        period_low_price_percentiles=period_low_price_pcts,
        period_high_pct_percentiles=period_high_pct_pcts,
        period_low_pct_percentiles=period_low_pct_pcts,
    )


def print_simulation_report(sim: SimulationResult):
    method_label = "Bootstrap (resampled from actual historical daily returns)" \
        if sim.method == "bootstrap" else "Parametric Normal (assumes normal daily returns)"

    mode_note = "reached target on ANY simulated day (touch)" if sim.mode == "touch" \
        else "price on day N only (endpoint)"

    print(f"\n[6] MONTE CARLO FORWARD SIMULATION — {sim.n_sims:,} simulated {sim.n_days}-day futures")
    print(f"    Method                                     : {method_label}")
    print(f"    Hit counted as                             : {mode_note}")
    print(f"    Starting price (last close)                : {sim.last_price:.2f}")
    print(f"    Simulated probability of the target move   : {sim.p_sim*100:.2f}%")
    print(f"    95% Wilson CI on that simulated probability: {sim.wilson_lo*100:.2f}% - {sim.wilson_hi*100:.2f}%")
    print(f"\n    >>> MOST PROBABLE PRICE in {sim.n_days} trading days : {sim.most_probable_price:.2f}"
          f"  ({sim.most_probable_pct_move:+.2f}%)")
    print(f"        (peak of the simulated outcome distribution — the single most")
    print(f"         likely price, not the average or midpoint)")
    print(f"\n    Forecast price ON DAY {sim.n_days} (endpoint), by percentile:")
    for q in PERCENTILE_QS:
        pct_change = sim.pct_move_percentiles[q]
        marker = "  <- median" if q == 50 else ""
        print(f"        {q:>2}th percentile : {sim.price_percentiles[q]:>10.2f}  ({pct_change:+.2f}%){marker}")
    print(f"    (5th/95th percentiles give a rough 90% forecast range for where the price")
    print(f"     ends up exactly {sim.n_days} days from now.)")

    print(f"\n    Possible HIGH / LOW at ANY point during the next {sim.n_days} days (not just day {sim.n_days}):")
    print(f"        {'Percentile':>12}   {'Possible HIGH':>16}   {'Possible LOW':>16}")
    for q in PERCENTILE_QS:
        hi_p = sim.period_high_price_percentiles[q]
        hi_pct = sim.period_high_pct_percentiles[q]
        lo_p = sim.period_low_price_percentiles[q]
        lo_pct = sim.period_low_pct_percentiles[q]
        marker = "  <- median" if q == 50 else ""
        print(f"        {q:>10}th   {hi_p:>10.2f} ({hi_pct:+.2f}%)   {lo_p:>10.2f} ({lo_pct:+.2f}%){marker}")
    print(f"    e.g. the 95th-percentile HIGH is the level only the best 5% of simulated")
    print(f"    futures reached; the 5th-percentile LOW is the level only the worst 5% of")
    print(f"    simulated futures fell to (or below).")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(res: MoveAnalysisResult, company_name: str = ""):
    dir_word = {"up": "gain of at least", "down": "drop of at least", "either": "move (up or down) of at least"}[res.direction]
    title = f"{res.symbol}" + (f" ({company_name})" if company_name else "")

    mode_label = {
        "endpoint": "ENDPOINT mode — only the price exactly N days later counts",
        "touch": "TOUCH mode — counts if the target was reached on ANY day within the window",
    }[res.mode]

    print("\n" + "=" * 70)
    print(f" MOVE PROBABILITY ANALYSIS — {title}")
    print("=" * 70)
    print(f"Question: over any {res.n_days}-trading-day window, what's the chance of a")
    print(f"          {dir_word} {res.pct_move:.2f}%?")
    print(f"Mode:     {mode_label}")
    print("-" * 70)

    print(f"\n[1] HISTORICAL FREQUENCY ({res.mode} mode)")
    print(f"    Overlapping {res.n_days}-day windows examined : {res.total_windows}")
    print(f"    Windows meeting/exceeding the move        : {res.hit_count}")
    print(f"    Empirical probability                     : {res.p_hat*100:.2f}%")

    print(f"\n[2] 95% BINOMIAL (WILSON) CONFIDENCE INTERVAL")
    print(f"    True probability likely lies between      : {res.wilson_lo*100:.2f}% and {res.wilson_hi*100:.2f}%")

    print(f"\n[3] POSSIBLE HIGH / LOW LEVELS (historical, applied to today's price of {res.last_price:.2f})")
    print(f"    Across all {res.total_windows} overlapping {res.n_days}-day windows, historically, this is")
    print(f"    how high/low the stock swung at ANY point within the window (not just at the")
    print(f"    end) -- percentiles of those per-window extremes, scaled to today's price:")
    print(f"        {'Percentile':>12}   {'Possible HIGH':>16}   {'Possible LOW':>16}")
    for q in PERCENTILE_QS:
        hi_p = res.high_price_percentiles[q]
        hi_pct = res.high_pct_percentiles[q]
        lo_p = res.low_price_percentiles[q]
        lo_pct = res.low_pct_percentiles[q]
        marker = "  <- median" if q == 50 else ""
        print(f"        {q:>10}th   {hi_p:>10.2f} ({hi_pct:+.2f}%)   {lo_p:>10.2f} ({lo_pct:+.2f}%){marker}")
    print(f"    e.g. the 95th-percentile HIGH is the level only the best 5% of historical")
    print(f"    windows reached or exceeded; the 5th-percentile LOW is the level only the")
    print(f"    worst 5% of historical windows fell to or below.")

    print(f"\n[4] NORMAL-DISTRIBUTION MODEL (for comparison, always based on endpoint returns)")
    print(f"    Historical mean {res.n_days}-day return        : {res.hist_mean:.2f}%")
    print(f"    Historical std-dev {res.n_days}-day return     : {res.hist_std:.2f}%")
    print(f"    Z-score of the target move                : {res.z_score:.2f}")
    print(f"    Normal-model implied probability          : {res.normal_prob*100:.2f}%")

    print(f"\n[5] CHI-SQUARE GOODNESS-OF-FIT TEST (is 'Normal' a fair model here? endpoint returns)")
    print(f"    Chi-square statistic                      : {res.chi2_stat:.3f}  (dof={res.chi2_dof})")
    print(f"    p-value                                   : {res.chi2_pvalue:.4f}")
    if res.normal_fits:
        print("    -> p >= 0.05: no strong evidence against normality here.")
        print("       The normal-model probability above is reasonably trustworthy.")
    else:
        print("    -> p < 0.05: the stock's real return distribution deviates")
        print("       significantly from normal near this threshold (fat tails/skew).")
        print("       Trust the HISTORICAL FREQUENCY + Wilson interval over the")
        print("       normal-model number.")


def print_verdict(res: MoveAnalysisResult, sim: SimulationResult = None):
    print("\n" + "-" * 70)
    print("VERDICT")
    if res.hit_count == 0:
        print(f"    This exact move has NOT occurred historically in {res.total_windows}")
        print(f"    windows. That doesn't make it impossible, but the data gives no")
        print(f"    direct empirical support --- treat the normal-model estimate")
        print(f"    ({res.normal_prob*100:.2f}%) as an upper-bound guess, tempered by the")
        print(f"    chi-square result above.")
    else:
        print(f"    Historically this has happened ~{res.p_hat*100:.1f}% of the time")
        print(f"    (95% CI: {res.wilson_lo*100:.1f}%-{res.wilson_hi*100:.1f}%), so it IS possible")
        print(f"    and has real historical precedent for {res.symbol}.")

    if sim is not None:
        print(f"\n    Looking FORWARD from today's price ({sim.n_sims:,} simulated {sim.n_days}-day")
        print(f"    futures, {sim.method} method): estimated probability ~{sim.p_sim*100:.1f}%")
        print(f"    (95% CI: {sim.wilson_lo*100:.1f}%-{sim.wilson_hi*100:.1f}%).")
        print(f"    Most probable price in {sim.n_days} days: {sim.most_probable_price:.2f} "
              f"({sim.most_probable_pct_move:+.2f}% from {sim.last_price:.2f}).")
        gap = abs(sim.p_sim - res.p_hat) * 100
        if gap > 5:
            print(f"    Note: this differs from the pure historical frequency by {gap:.1f} points --")
            print(f"    that's expected since the simulation starts from TODAY's price/volatility")
            print(f"    regime rather than averaging over the stock's whole history.")
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------
# Interactive CLI
# --------------------------------------------------------------------------

class UserExit(Exception):
    """Raised when the user asks to stop the interactive loop."""
    pass


def ask_choice(prompt: str, options: dict, default: str) -> str:
    """
    Prompts for a single-letter choice, e.g. options={"u": "up", "d": "down",
    "e": "either"}. Accepts the letter (preferred) or the full word, either
    case-insensitively; blank input takes the default. Returns the full word
    (e.g. "up"), never the letter, so callers don't need to translate.
    """
    display = "/".join(f"[{letter}]{word[1:]}" for letter, word in options.items())
    words_to_letters = {word: letter for letter, word in options.items()}
    while True:
        raw = input(f"{prompt} — {display} [{default}]: ").strip().lower()
        if not raw:
            raw = default
        if raw in options:
            return options[raw]
        if raw in words_to_letters:
            return raw
        print(f"Please enter one of: {', '.join(options.keys())}")


def run_one_analysis(ticker_map: dict) -> None:
    """Runs the full prompt -> fetch -> analyze -> report flow for a single
    stock. Raises no exceptions to the caller for analysis errors (bad
    symbol, no data, etc.) — those are caught and printed so a loop can
    move on to the next stock instead of crashing out. Raises UserExit if
    the user types 'exit'/'quit' at the symbol prompt."""
    raw = input("Enter stock symbol (e.g. RELIANCE, TCS, INFY): ")
    if raw.strip().lower() in ("exit", "quit", "q"):
        raise UserExit
    symbol = resolve_symbol(raw, ticker_map)

    while True:
        try:
            pct_move = float(input("Enter target % move (e.g. 5 for 5%): ").strip())
            break
        except ValueError:
            print("Please enter a number, e.g. 5 or 7.5")

    while True:
        try:
            n_days = int(input("Enter number of trading days (e.g. 10): ").strip())
            if n_days < 1:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive whole number of days.")

    direction = ask_choice("Direction", {"u": "up", "d": "down", "e": "either"}, "e")

    print("Mode — 't' only counts it if the move happened on ANY day within the window,")
    print("       even if the price came back down (or up) by the end (recommended);")
    print("       'e' only checks the price exactly N days later.")
    mode = ask_choice("Mode", {"t": "touch", "e": "endpoint"}, "t")

    do_sim = ask_choice("Also run a Monte Carlo forward simulation?", {"y": "yes", "n": "no"}, "y") == "yes"
    sim_method = "bootstrap"
    n_sims = 10000
    if do_sim:
        sim_method = ask_choice(
            "Simulation method", {"b": "bootstrap", "n": "normal"}, "b"
        )
        raw_n = input("Number of simulations [10000]: ").strip()
        if raw_n.isdigit():
            n_sims = int(raw_n)

    try:
        close = fetch_price_series(symbol)
        result = analyze(symbol, pct_move, n_days, direction, close=close, mode=mode)
        sim_result = None
        if do_sim:
            sim_result = run_simulation(close, n_days, pct_move, direction,
                                         n_sims=n_sims, method=sim_method, mode=mode)
    except Exception as e:
        print(f"\nError: {e}")
        return

    print_report(result, ticker_map.get(symbol, ""))
    if sim_result is not None:
        print_simulation_report(sim_result)
    print_verdict(result, sim_result)


def interactive_main(ticker_csv: str = "tickers.csv"):
    ticker_map = load_ticker_map(ticker_csv)
    enable_autofill(ticker_map)

    print("Stock Move Probability Analyzer (NSE)")
    print("--------------------------------------")
    if HAVE_READLINE and ticker_map:
        print("(Tip: press Tab while typing the symbol or company name to autocomplete.)")
    print("(Tip: type 'exit' or 'quit' at the symbol prompt any time to stop.)\n")

    while True:
        try:
            run_one_analysis(ticker_map)
        except UserExit:
            print("Exiting.")
            return
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        again = ask_choice("Analyze another stock?", {"y": "yes", "n": "no"}, "y")
        if again == "no":
            print("Exiting.")
            return
        print()  # blank line to visually separate the next run


def cli_main():
    parser = argparse.ArgumentParser(description="Stock move probability analyzer (NSE)")
    parser.add_argument("--symbol", help="NSE ticker symbol, e.g. RELIANCE")
    parser.add_argument("--pct", type=float, help="Target percentage move, e.g. 5")
    parser.add_argument("--days", type=int, help="Number of trading days, e.g. 10")
    parser.add_argument("--direction", choices=["up", "down", "either"], default="either")
    parser.add_argument("--mode", choices=["endpoint", "touch"], default="touch",
                         help="touch (default) = counts if the target was reached on ANY day "
                              "within the window, even if the price settled back down by the "
                              "end; endpoint = only the price exactly N days later counts")
    parser.add_argument("--period", default="10y", help="History window to download, e.g. 5y, 10y, max")
    parser.add_argument("--tickers-csv", default="tickers.csv")
    parser.add_argument("--simulate", action="store_true",
                         help="Also run a Monte Carlo forward simulation from today's price")
    parser.add_argument("--sim-method", choices=["bootstrap", "normal"], default="bootstrap",
                         help="bootstrap = resample real historical daily returns (default); "
                              "normal = assume normally distributed daily returns")
    parser.add_argument("--n-sims", type=int, default=10000, help="Number of simulated paths")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible simulations")
    args = parser.parse_args()

    if args.symbol and args.pct is not None and args.days is not None:
        ticker_map = load_ticker_map(args.tickers_csv)
        close = fetch_price_series(args.symbol, period=args.period)
        result = analyze(args.symbol, args.pct, args.days, args.direction, close=close, mode=args.mode)
        sim_result = None
        if args.simulate:
            sim_result = run_simulation(close, args.days, args.pct, args.direction,
                                         n_sims=args.n_sims, method=args.sim_method, seed=args.seed,
                                         mode=args.mode)
        print_report(result, ticker_map.get(args.symbol.upper(), ""))
        if sim_result is not None:
            print_simulation_report(sim_result)
        print_verdict(result, sim_result)
    else:
        interactive_main(args.tickers_csv)


if __name__ == "__main__":
    cli_main()
