# screener.py — Monte Carlo movement screener

Runs the same Monte Carlo simulation engine as `stock_move_analysis.py`
across every stock in an Excel/CSV ticker list, and ranks them by expected
movement over the next N trading days.

## Setup

Same dependencies as the single-stock tool, plus `openpyxl` (for reading
.xlsx) and optionally `tqdm` (progress bars — script works fine without it):

```bash
pip install yfinance scipy numpy pandas openpyxl tqdm
```

## Basic usage

Ticker file is fixed to `tickers.csv` in the same folder by default —
override anytime with `--excel-file`.

### Interactive mode (no flags)

```bash
python screener.py
```

If you don't pass `--days`, the script drops into the same kind of
interactive Q&A as `stock_move_analysis.py` — it'll ask for days, direction,
ranking method, etc., showing the default for each in brackets (press
Enter to accept it). The ticker file is fixed to `tickers.csv` in the
current folder unless you override it with `--excel-file`.

### Command-line mode

```bash
python screener.py --days 10 --top-n 20
```

This ranks every EQ-series stock in `tickers.csv` (fixed filename — put your
NSE ticker list at that path in the same folder) by the size of its
most-probable simulated move (either direction) over the next 10 trading
days, prints the top 20, and saves the full ranked list to
`screener_results.csv`.

## Key decisions made for you (all changeable via flags)

| What | Default | Change with |
|---|---|---|
| Ranking metric | Size of most-probable (mode) % move | `--rank-by median` / `p95_upside` / `probability` |
| Direction | Either way (biggest swing) | `--direction up` / `down` |
| Simulations per stock | 2,000 (fast enough for large lists) | `--n-sims` |
| Simulation method | Bootstrap (real historical returns) | `--method normal` |
| History pulled per stock | 5 years | `--period 10y` |
| Only EQ series | Yes | `--series-filter ALL` |

### Rank by "probability of hitting X%" instead of magnitude

```bash
python screener.py --days 5 \
    --rank-by probability --pct-threshold 5 --direction up
```

This ranks stocks by "highest chance of a ≥5% gain within 5 trading days" —
note this can surface *different* stocks than ranking by pure magnitude,
since a highly volatile stock can have a lower most-probable move but a
higher chance of *at some point* crossing a given threshold (especially
combined with `--check-mode touch`).

### Quick test run before doing the full list

```bash
python screener.py --days 10 --limit 30
```

`--limit` caps how many symbols are processed — use this to sanity check
timing and column names before running against 1000+ symbols.

## Performance notes

- Downloads are **batched** (default 50 symbols/request via yfinance's
  multi-ticker mode) rather than one-by-one, which is far faster and less
  likely to hit rate limits than looping per symbol.
- `--sleep` (default 1s) pauses between batches — raise it if you get
  rate-limited, lower it if you want to push speed.
- `--n-sims` is the main lever for total runtime: 2,000 sims/stock is
  usually enough for ranking purposes; use 10,000+ only for a final
  deep-dive on your shortlisted candidates (or better, run those through
  `stock_move_analysis.py` directly for the full report).
- Stocks with less than `--min-history-days` (default 250 trading days,
  about 1 year) of price history are skipped automatically — new listings
  won't have enough data for a meaningful simulation.
- Failures (delisted symbols, bad tickers, network hiccups) are logged and
  skipped rather than stopping the whole run.

## Ticker list format expected

Matches the standard NSE equity list layout:

```
SYMBOL,NAME OF COMPANY,SERIES,...
RELIANCE,Reliance Industries Limited,EQ,...
```

If your columns are named differently, or your ticker file lives elsewhere
or has a different name, point to it explicitly:

```bash
python screener.py --excel-file mylist.xlsx \
    --symbol-col Ticker --name-col CompanyName --series-col Type
```

## Output columns (screener_results.csv)

| Column | Meaning |
|---|---|
| `LastPrice` | Most recent close used as the simulation's starting point |
| `MostProbablePrice` / `MostProbablePctMove` | Peak of the simulated outcome distribution (mode) |
| `MedianPctMove` | 50th percentile forecast move |
| `P5PctMove` / `P95PctMove` | Rough 90% forecast range |
| `ProbOfThreshold` + CI | Only populated when `--rank-by probability` is used |

## Suggested workflow

1. Run `screener.py` on the full list with a moderate `--n-sims` (2,000-5,000)
   to get a fast shortlist of the top 10-20 movers.
2. Run `stock_move_analysis.py` individually on those shortlisted symbols
   with a higher `--n-sims` (10,000+) for the full historical-frequency +
   chi-square + detailed simulation report before acting on anything.
