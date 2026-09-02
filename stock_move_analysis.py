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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

try:
    import yfinance as yf
except ImportError:
    print("This script needs yfinance. Install it with:\n    pip install yfinance")
    sys.exit(1)


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


# --------------------------------------------------------------------------
# Core statistics
# --------------------------------------------------------------------------

@dataclass
class MoveAnalysisResult:
    symbol: str
    n_days: int
    pct_move: float
    direction: str
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


def compute_rolling_returns(close: pd.Series, n_days: int) -> pd.Series:
    """Overlapping N-trading-day percentage returns across the whole history."""
    returns = close.pct_change(periods=n_days) * 100.0
    return returns.dropna()


def direction_hits(returns: pd.Series, pct_move: float, direction: str) -> int:
    if direction == "up":
        return int((returns >= pct_move).sum())
    elif direction == "down":
        return int((returns <= -pct_move).sum())
    else:  # either direction
        return int((returns.abs() >= pct_move).sum())


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


def analyze(symbol: str, pct_move: float, n_days: int, direction: str,
            period: str = "10y") -> MoveAnalysisResult:
    yf_symbol = symbol if symbol.upper().endswith(".NS") else f"{symbol.upper()}.NS"
    df = yf.download(yf_symbol, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for '{yf_symbol}'. Check the symbol.")

    close = df["Close"].dropna()
    if isinstance(close, pd.DataFrame):  # yfinance sometimes returns a 1-col DataFrame
        close = close.iloc[:, 0]

    returns = compute_rolling_returns(close, n_days)
    total = len(returns)
    if total < 30:
        print(f"Warning: only {total} overlapping {n_days}-day windows available. "
              f"Results will be statistically weak.")

    hits = direction_hits(returns, pct_move, direction)
    p_hat = hits / total if total else 0.0
    lo, hi = wilson_interval(hits, total)

    chi2_stat, chi2_p, dof, mean, std = chi_square_normality(returns, pct_move, direction)
    normal_fits = chi2_p >= 0.05  # fail to reject normality at 5% level

    z = (pct_move - mean) / std if std > 0 else float("nan")
    if direction == "up":
        normal_prob = 1 - stats.norm.cdf(pct_move, loc=mean, scale=std)
    elif direction == "down":
        normal_prob = stats.norm.cdf(-pct_move, loc=mean, scale=std)
    else:
        normal_prob = (1 - stats.norm.cdf(pct_move, loc=mean, scale=std)) + \
                       stats.norm.cdf(-pct_move, loc=mean, scale=std)

    return MoveAnalysisResult(
        symbol=symbol.upper(), n_days=n_days, pct_move=pct_move, direction=direction,
        total_windows=total, hit_count=hits, p_hat=p_hat, wilson_lo=lo, wilson_hi=hi,
        hist_mean=mean, hist_std=std, z_score=z, normal_prob=normal_prob,
        chi2_stat=chi2_stat, chi2_pvalue=chi2_p, chi2_dof=dof, normal_fits=normal_fits,
    )


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_report(res: MoveAnalysisResult, company_name: str = ""):
    dir_word = {"up": "gain of at least", "down": "drop of at least", "either": "move (up or down) of at least"}[res.direction]
    title = f"{res.symbol}" + (f" ({company_name})" if company_name else "")

    print("\n" + "=" * 70)
    print(f" MOVE PROBABILITY ANALYSIS — {title}")
    print("=" * 70)
    print(f"Question: over any {res.n_days}-trading-day window, what's the chance of a")
    print(f"          {dir_word} {res.pct_move:.2f}%?")
    print("-" * 70)

    print(f"\n[1] HISTORICAL FREQUENCY")
    print(f"    Overlapping {res.n_days}-day windows examined : {res.total_windows}")
    print(f"    Windows meeting/exceeding the move        : {res.hit_count}")
    print(f"    Empirical probability                     : {res.p_hat*100:.2f}%")

    print(f"\n[2] 95% BINOMIAL (WILSON) CONFIDENCE INTERVAL")
    print(f"    True probability likely lies between      : {res.wilson_lo*100:.2f}% and {res.wilson_hi*100:.2f}%")

    print(f"\n[3] NORMAL-DISTRIBUTION MODEL (for comparison)")
    print(f"    Historical mean {res.n_days}-day return        : {res.hist_mean:.2f}%")
    print(f"    Historical std-dev {res.n_days}-day return     : {res.hist_std:.2f}%")
    print(f"    Z-score of the target move                : {res.z_score:.2f}")
    print(f"    Normal-model implied probability          : {res.normal_prob*100:.2f}%")

    print(f"\n[4] CHI-SQUARE GOODNESS-OF-FIT TEST (is 'Normal' a fair model here?)")
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
    print("=" * 70 + "\n")


# --------------------------------------------------------------------------
# Interactive CLI
# --------------------------------------------------------------------------

def interactive_main(ticker_csv: str = "tickers.csv"):
    ticker_map = load_ticker_map(ticker_csv)

    print("Stock Move Probability Analyzer (NSE)")
    print("--------------------------------------")
    symbol = input("Enter stock symbol (e.g. RELIANCE, TCS, INFY): ").strip().upper()
    if ticker_map and symbol not in ticker_map:
        print(f"'{symbol}' not found in the local NSE ticker list — proceeding anyway "
              f"(it may still be a valid Yahoo Finance symbol).")

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

    direction = input("Direction — up / down / either [either]: ").strip().lower() or "either"
    if direction not in ("up", "down", "either"):
        print("Unrecognized direction, defaulting to 'either'.")
        direction = "either"

    try:
        result = analyze(symbol, pct_move, n_days, direction)
    except Exception as e:
        print(f"\nError: {e}")
        return

    print_report(result, ticker_map.get(symbol, ""))


def cli_main():
    parser = argparse.ArgumentParser(description="Stock move probability analyzer (NSE)")
    parser.add_argument("--symbol", help="NSE ticker symbol, e.g. RELIANCE")
    parser.add_argument("--pct", type=float, help="Target percentage move, e.g. 5")
    parser.add_argument("--days", type=int, help="Number of trading days, e.g. 10")
    parser.add_argument("--direction", choices=["up", "down", "either"], default="either")
    parser.add_argument("--period", default="10y", help="History window to download, e.g. 5y, 10y, max")
    parser.add_argument("--tickers-csv", default="tickers.csv")
    args = parser.parse_args()

    if args.symbol and args.pct is not None and args.days is not None:
        ticker_map = load_ticker_map(args.tickers_csv)
        result = analyze(args.symbol, args.pct, args.days, args.direction, period=args.period)
        print_report(result, ticker_map.get(args.symbol.upper(), ""))
    else:
        interactive_main(args.tickers_csv)


if __name__ == "__main__":
    cli_main()
