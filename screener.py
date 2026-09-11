"""
Screener — Monte Carlo movement screener across a whole stock list
--------------------------------------------------------------------
Runs the SAME Monte Carlo forward simulation engine as
stock_move_analysis.py, but across every symbol in tickers.csv (or another
file you point to), and ranks stocks by how much movement the simulation
says is most likely over the next N trading days.

Usage:
    python screener.py
        -> asks a short series of questions (same style as
           stock_move_analysis.py), then runs.

    python screener.py --days 10 --symbols RELIANCE,TCS,INFY
        -> runs immediately for specific manual symbols.

Run `python screener.py --help` for the full flag list.
"""

import sys
import time
import argparse

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    print("This script needs yfinance. Install it with:\n    pip install yfinance")
    sys.exit(1)

try:
    from tqdm import tqdm
    HAVE_TQDM = True
except ImportError:
    HAVE_TQDM = False

# Enable tab completion for interactive prompts on Unix/Linux/macOS
try:
    import readline
    HAVE_READLINE = True
except ImportError:
    HAVE_READLINE = False

# Reuse the exact same simulation engine + prompt helper as the single-stock
# tool, so numbers and interaction style stay consistent between the two.
from stock_move_analysis import run_simulation, wilson_interval, ask_choice, PERCENTILE_QS  # noqa: E402


# --------------------------------------------------------------------------
# Loading the ticker list
# --------------------------------------------------------------------------

def load_ticker_list(path: str, symbol_col: str = "SYMBOL",
                      name_col: str = "NAME OF COMPANY",
                      series_col: str = "SERIES",
                      series_filter: str = "EQ") -> pd.DataFrame:
    """
    Reads symbols (+ optional company name) from an .xlsx or .csv file.
    Expects columns matching the standard NSE equity-list layout:
        SYMBOL | NAME OF COMPANY | SERIES | ...
    If series_filter is given (default 'EQ') and a SERIES column exists,
    only rows matching it are kept. Pass series_filter=None to keep everything.
    """
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if symbol_col not in df.columns:
        raise ValueError(
            f"Column '{symbol_col}' not found in {path}. "
            f"Available columns: {list(df.columns)}. "
            f"Pass --symbol-col to match your file."
        )

    if series_filter and series_col in df.columns:
        df = df[df[series_col].astype(str).str.strip().str.upper() == series_filter.upper()]

    df = df[[symbol_col] + ([name_col] if name_col in df.columns else [])].copy()
    df.columns = ["SYMBOL"] + (["NAME"] if name_col in df.columns else [])
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df = df.drop_duplicates(subset="SYMBOL").reset_index(drop=True)
    return df


def setup_ticker_completer(csv_path: str, symbol_col: str, series_col: str, series_filter: str):
    """Sets up tab-completion for symbol input based on fnotickers.csv."""
    if not HAVE_READLINE:
        return
    try:
        series_f = None if series_filter.upper() in ("ALL", "") else series_filter
        df = load_ticker_list(csv_path, symbol_col=symbol_col, series_col=series_col, series_filter=series_f)
        symbols = df["SYMBOL"].tolist()
    except Exception:
        symbols = []

    def completer(text, state):
        matches = [s for s in symbols if s.startswith(text.upper())]
        if state < len(matches):
            return matches[state]
        else:
            return None

    readline.set_completer(completer)
    readline.set_completer_delims(' \t\n,')
    readline.parse_and_bind("tab: complete")


# --------------------------------------------------------------------------
# Batched price downloads
# --------------------------------------------------------------------------

def batch_download(symbols: list, period: str = "5y", batch_size: int = 10,
                    sleep_between: float = 1.0, min_history_days: int = 250) -> dict:
    """
    Downloads Close price history for many NSE symbols efficiently by batching
    them into multi-ticker yfinance calls instead of one request per symbol.
    Returns {symbol: pandas Series of close prices}, skipping symbols with
    too little history or that fail to download.
    """
    results = {}
    yf_symbols = [f"{s}.NS" for s in symbols]
    batches = [yf_symbols[i:i + batch_size] for i in range(0, len(yf_symbols), batch_size)]

    iterator = tqdm(batches, desc="Downloading", unit="batch") if HAVE_TQDM else batches
    for batch in iterator:
        try:
            data = yf.download(batch, period=period, group_by="ticker",
                                threads=False, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  Warning: batch download failed ({e}); skipping this batch.")
            continue

        for yf_sym in batch:
            symbol = yf_sym[:-3]  # strip ".NS"
            try:
                if len(batch) == 1:
                    close = data["Close"].dropna()
                else:
                    close = data[yf_sym]["Close"].dropna()
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                if len(close) >= min_history_days:
                    results[symbol] = close
            except (KeyError, TypeError):
                continue  # symbol had no data in this batch (delisted / wrong ticker / etc.)

        if sleep_between > 0:
            time.sleep(sleep_between)

    return results


# --------------------------------------------------------------------------
# Screening
# --------------------------------------------------------------------------

def screen_stocks(price_data: dict, n_days: int, direction: str,
                   rank_by: str = "most_probable", pct_threshold: float = None,
                   method: str = "bootstrap", n_sims: int = 2000,
                   mode: str = "touch", seed: int = None) -> pd.DataFrame:
    """
    Runs the Monte Carlo simulation on every {symbol: close_series} pair and
    builds a ranked results table.

    rank_by='most_probable' (default): ranks by the size of the most-probable
        (mode) simulated move -- no threshold needed.
    rank_by='probability': ranks by the chance of hitting pct_threshold
        (requires pct_threshold to be set).
    """
    if rank_by == "probability" and pct_threshold is None:
        raise ValueError("rank_by='probability' requires pct_threshold to also be set.")

    rows = []
    items = price_data.items()
    iterator = tqdm(items, total=len(price_data), desc="Simulating", unit="stock") if HAVE_TQDM else items

    threshold_for_sim = pct_threshold if pct_threshold is not None else 0.0

    for symbol, close in iterator:
        try:
            sim = run_simulation(close, n_days, threshold_for_sim, direction,
                                  n_sims=n_sims, method=method, seed=seed, mode=mode)
        except Exception as e:
            print(f"  Warning: simulation failed for {symbol} ({e}); skipping.")
            continue

        rows.append({
            "SYMBOL": symbol,
            "LastPrice": sim.last_price,
            "MostProbablePrice": sim.most_probable_price,
            "MostProbablePctMove": sim.most_probable_pct_move,
            "MedianPctMove": sim.pct_move_percentiles[50],
            "P5PctMove": sim.pct_move_percentiles[5],
            "P95PctMove": sim.pct_move_percentiles[95],
            "PeriodHighPctMove": sim.period_high_pct_percentiles[50],   # typical best point reached
            "PeriodLowPctMove": sim.period_low_pct_percentiles[50],     # typical worst point reached
            "ProbOfThreshold": sim.p_sim if pct_threshold is not None else None,
            "ProbCI_Lo": sim.wilson_lo if pct_threshold is not None else None,
            "ProbCI_Hi": sim.wilson_hi if pct_threshold is not None else None,
        })

    if not rows:
        raise RuntimeError("No stocks were successfully simulated -- check your ticker list and network.")

    results = pd.DataFrame(rows)

    if rank_by == "probability":
        results["RankMetric"] = results["ProbOfThreshold"]
    else:  # most_probable (default)
        results["RankMetric"] = results["MostProbablePctMove"].abs() if direction == "either" \
            else results["MostProbablePctMove"]

    results = results.sort_values("RankMetric", ascending=False).reset_index(drop=True)
    return results


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_top_results(results: pd.DataFrame, top_n: int, rank_by: str,
                       direction: str, n_days: int, mode: str, name_map: dict = None):
    print("\n" + "=" * 110)
    print(f" TOP {top_n} STOCKS — Monte Carlo {n_days}-day screen "
          f"(direction={direction}, mode={mode}, rank-by={rank_by})")
    print("=" * 110)

    top = results.head(top_n).copy()
    if name_map:
        top.insert(1, "NAME", top["SYMBOL"].map(name_map).fillna(""))

    with pd.option_context("display.max_rows", None, "display.width", 180,
                            "display.float_format", "{:.2f}".format):
        cols = ["SYMBOL"] + (["NAME"] if name_map else []) + [
            "LastPrice", "MostProbablePrice", "MostProbablePctMove",
            "MedianPctMove", "PeriodHighPctMove", "PeriodLowPctMove"
        ]
        if "ProbOfThreshold" in top.columns and top["ProbOfThreshold"].notna().any():
            cols += ["ProbOfThreshold", "ProbCI_Lo", "ProbCI_Hi"]
        print(top[cols].to_string(index=False))
    print("=" * 110)
    print("(PeriodHigh/Low = median best/worst point reached at any time within the window,")
    print(" per the Monte Carlo simulation -- not just the day-N endpoint.)\n")


# --------------------------------------------------------------------------
# Interactive prompt (used when --days isn't given on the CLI)
# --------------------------------------------------------------------------

def interactive_prompt(defaults: argparse.Namespace) -> argparse.Namespace:
    """Short, letter-choice style Q&A -- matches stock_move_analysis.py."""
    print("Monte Carlo Stock Screener (NSE)")
    print("---------------------------------")
    print(f"Using ticker file: {defaults.excel_file}\n")

    args = argparse.Namespace(**vars(defaults))

    while True:
        try:
            args.days = int(input("Enter number of trading days to scan forward (e.g. 10): ").strip())
            if args.days < 1:
                raise ValueError
            break
        except ValueError:
            print("Please enter a positive whole number.")

    args.direction = ask_choice("Direction", {"u": "up", "d": "down", "e": "either"}, "e")

    print("Mode — 't' counts it if the move happened on ANY day within the window")
    print("       (recommended); 'e' only checks the price exactly N days later.")
    args.mode = ask_choice("Mode", {"t": "touch", "e": "endpoint"}, "t")

    print("\nRank stocks by:")
    print("  'm' magnitude    - size of the most-probable simulated move (no threshold needed)")
    print("  'p' probability  - chance of hitting a specific % target you choose")
    rank_choice = ask_choice("Rank by", {"m": "magnitude", "p": "probability"}, "m")
    args.rank_by = "probability" if rank_choice == "probability" else "most_probable"

    if args.rank_by == "probability":
        while True:
            try:
                args.pct_threshold = float(input("Target % move to rank probability against (e.g. 5): ").strip())
                break
            except ValueError:
                print("Please enter a number.")

    args.method = ask_choice("Simulation method", {"b": "bootstrap", "n": "normal"}, "b")

    raw = input(f"Simulations per stock [{args.n_sims}]: ").strip()
    if raw.isdigit():
        args.n_sims = int(raw)

    # Setup tab-completion for manual symbols input from CSV
    setup_ticker_completer(args.excel_file, args.symbol_col, args.series_col, args.series_filter)
    if HAVE_READLINE:
        print("(Tab-completion enabled: press Tab to auto-complete stock symbols)")

    raw = input("Enter specific symbols separated by commas manually (or Enter to use whole file): ").strip()
    if raw:
        args.symbols = raw

    raw = input(f"How many top results to show [{args.top_n}]: ").strip()
    if raw.isdigit():
        args.top_n = int(raw)

    return args


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monte Carlo movement screener across a stock list")
    parser.add_argument("--excel-file", default="fnotickers.csv",
                         help="Path to ticker list (fixed default: fnotickers.csv)")
    parser.add_argument("--symbol-col", default="SYMBOL")
    parser.add_argument("--name-col", default="NAME OF COMPANY")
    parser.add_argument("--series-col", default="SERIES")
    parser.add_argument("--series-filter", default="EQ",
                         help="Keep only rows matching this SERIES value; 'ALL' or '' to disable filtering")
    parser.add_argument("--symbols", default=None,
                         help="Comma-separated list of specific symbols to process manually (e.g., RELIANCE,TCS,INFY)")
    parser.add_argument("--days", type=int, default=None, help="Number of trading days to forecast forward")
    parser.add_argument("--direction", choices=["up", "down", "either"], default="either")
    parser.add_argument("--mode", choices=["endpoint", "touch"], default="touch",
                         help="touch (default) = counts if the target was reached on ANY day within "
                              "the window; endpoint = only the price exactly N days later counts")
    parser.add_argument("--rank-by", choices=["most_probable", "probability"], default="most_probable")
    parser.add_argument("--pct-threshold", type=float, default=None,
                         help="Required if --rank-by probability; also used as the simulation's target move")
    parser.add_argument("--method", choices=["bootstrap", "normal"], default="bootstrap")
    parser.add_argument("--n-sims", type=int, default=2000,
                         help="Simulations per stock (lower = faster screening; default 2000)")
    parser.add_argument("--period", default="5y", help="History window to download per stock")
    parser.add_argument("--min-history-days", type=int, default=250,
                         help="Skip stocks with fewer trading days of history than this")
    parser.add_argument("--batch-size", type=int, default=10, help="Symbols per yfinance batch download")
    parser.add_argument("--sleep", type=float, default=1.0, help="Seconds to pause between batches")
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N symbols (for a quick test run)")
    parser.add_argument("--top-n", type=int, default=20, help="How many top results to print")
    parser.add_argument("--output", default="screener_results.csv", help="CSV path for the full ranked results")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def run_screener(args: argparse.Namespace):
    series_filter = None if args.series_filter.upper() in ("ALL", "") else args.series_filter
    name_col = args.name_col if args.name_col else "__no_name_col__"

    print(f"Loading ticker list from {args.excel_file} ...")
    tickers_df = load_ticker_list(args.excel_file, args.symbol_col, name_col,
                                   args.series_col, series_filter)

    # Filter manually by specified symbols if provided via CLI flag or prompt
    if args.symbols:
        target_symbols = [s.strip().upper() for s in args.symbols.split(",")]
        tickers_df = tickers_df[tickers_df["SYMBOL"].isin(target_symbols)]
        if tickers_df.empty:
            print(f"Warning: None of the specified symbols ({args.symbols}) matched the loaded file/filter.")

    symbols = tickers_df["SYMBOL"].tolist()
    if args.limit and not args.symbols:
        symbols = symbols[:args.limit]
    name_map = dict(zip(tickers_df["SYMBOL"], tickers_df["NAME"])) if "NAME" in tickers_df.columns else None

    print(f"{len(symbols)} symbols to process.")
    if len(symbols) > 500:
        est_minutes = (len(symbols) / args.batch_size) * args.sleep / 60
        print(f"Note: this is a large list -- downloading alone could take roughly "
              f"{est_minutes:.1f}+ minutes depending on your connection and rate limits.")

    print("Downloading price history (batched)...")
    price_data = batch_download(symbols, period=args.period, batch_size=args.batch_size,
                                 sleep_between=args.sleep, min_history_days=args.min_history_days)
    print(f"{len(price_data)}/{len(symbols)} symbols had enough history to simulate.")

    print(f"Running Monte Carlo simulation ({args.n_sims} paths/stock, {args.method} method, "
          f"{args.mode} mode)...")
    results = screen_stocks(price_data, args.days, args.direction, rank_by=args.rank_by,
                             pct_threshold=args.pct_threshold, method=args.method,
                             n_sims=args.n_sims, mode=args.mode, seed=args.seed)

    print_top_results(results, args.top_n, args.rank_by, args.direction, args.days, args.mode, name_map)

    if name_map:
        results.insert(1, "NAME", results["SYMBOL"].map(name_map).fillna(""))
    results.to_csv(args.output, index=False)
    print(f"Full ranked results ({len(results)} stocks) saved to: {args.output}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # If --days wasn't given on the command line, ask instead of erroring out
    # (mirrors stock_move_analysis.py's behaviour).
    if not args.days:
        args = interactive_prompt(args)

    try:
        run_screener(args)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
