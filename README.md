# Stock Move Probability Analyzer

Checks whether a hypothetical % move over N trading days is "possible" for
an NSE stock, using three complementary statistical lenses.

## Setup

```bash
pip install yfinance scipy numpy pandas
```

Put your NSE ticker list (the SYMBOL / NAME OF COMPANY CSV you have) in the
same folder as `tickers.csv`, or pass `--tickers-csv path/to/file.csv`.
This is optional — it's only used to print the company name; the analysis
works without it as long as you type a valid symbol.

## Run — interactive mode

```bash
python stock_move_analysis.py
```

You'll be prompted for:
- Symbol (e.g. `RELIANCE`, `TCS`, `INFY`)
- Target % move (e.g. `5`)
- Number of trading days (e.g. `10`)
- Direction: `up`, `down`, or `either`

## Run — command-line mode (for scripting)

```bash
python stock_move_analysis.py --symbol RELIANCE --pct 5 --days 10 --direction either
```

Optional: `--period 5y` (default `10y`) controls how much history is pulled.

## What it reports

1. **Historical frequency** — of all overlapping N-day windows in the
   stock's price history, what fraction actually moved by ≥X%?
2. **95% Wilson confidence interval** — how much to trust that frequency,
   given the sample size (rare events with few hits get wide intervals).
3. **Normal-distribution model** — the "textbook" z-score/probability you'd
   get by assuming returns are normally distributed, using this stock's own
   historical mean/std.
4. **Chi-square goodness-of-fit test** — checks whether this stock's actual
   N-day return distribution really does look normal near the threshold in
   question. If it doesn't (p < 0.05), the historical frequency (steps 1-2)
   should be trusted more than the normal-model number (step 3), since
   real stock returns often have fatter tails than a normal curve predicts.

## Notes / limitations

- Uses **overlapping** N-day windows (day 1-10, day 2-11, day 3-12, ...),
  which is standard for this kind of study but means samples aren't fully
  independent — the Wilson interval is still a reasonable practical guide,
  just don't over-interpret precision to the decimal point.
- History length matters: very short histories or very large N (few
  non-overlapping windows) will give wide, low-confidence intervals — the
  script warns you when the sample size is thin.
- Yahoo Finance data via `yfinance`; NSE symbols are queried as `SYMBOL.NS`.
