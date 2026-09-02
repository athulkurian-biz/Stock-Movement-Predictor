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
- Mode: `endpoint` or `touch` (see "Endpoint vs. touch mode" below)

### Autofill from the ticker CSV

If `tickers.csv` is present, the symbol prompt supports:
- **Tab-completion** — start typing a symbol or company name and press `Tab`
  to autocomplete or cycle through matches (uses Python's built-in
  `readline`; on Windows run `pip install pyreadline3` first).
- **Typo suggestions** — if what you type doesn't match the list exactly,
  it shows the closest symbol/company-name matches and lets you pick one
  by number.

## Run — command-line mode (for scripting)

```bash
python stock_move_analysis.py --symbol RELIANCE --pct 5 --days 10 --direction either --mode touch
```

Optional: `--period 5y` (default `10y`) controls how much history is pulled.
Optional: `--mode endpoint` (default) or `--mode touch` — see below.

## Endpoint vs. touch mode

This controls what counts as a "hit" when scanning historical windows (and
simulated futures):

- **`endpoint` (default)** — a window only counts if the price *exactly*
  N trading days later is past the target. A stock that spikes +8% on day
  4 and settles back to +2% by day 10 counts as a **miss** for a 5%/10-day
  target, since it doesn't compare anything in between.
- **`touch`** — a window counts if the target was reached on **any** day
  within the window, even if the price came back down (or up) by the end.
  The same +8%-then-+2% example above counts as a **hit**.

Which one you want depends on the question you're actually asking:
- "Where will the price likely *be* in N days?" → `endpoint`
- "Could the stock realistically *move* that much at some point within
  N days?" (e.g. for setting a stop-loss, a target-sell alert, or an
  options strike) → `touch`

`touch` mode's hit rate is always **greater than or equal to** `endpoint`
mode's for the same inputs, since every endpoint hit is also a touch (the
last day is one of the days being checked) — the gap between the two tells
you how much of the "possibility" is coming from intra-window swings that
don't survive to the end of the window.

The mode applies to both the historical-frequency count (steps 1-2) and
the Monte Carlo simulation (step 5). The normal-distribution model and
chi-square test (steps 3-4) always analyze the endpoint return
distribution regardless of mode, since they're about the shape of the
N-day return distribution itself, not about hit-counting.

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
5. **Monte Carlo forward simulation** (optional) — simulates thousands of
   possible N-day futures *starting from today's actual last price*, and
   reports:
   - the fraction of simulated futures that hit your target move
   - a **most probable price** — the single highest-density outcome (the
     peak of the simulated distribution, estimated via kernel density
     estimation), which is *not* the same as the median. Compounded returns
     produce a right-skewed price distribution (a stock can fall at most to
     zero but can rise indefinitely), so the most likely single outcome
     typically sits a little below the median/average forecast.
   - a full forecast price range (5th/25th/50th/75th/95th percentiles)

   Two simulation methods:
   - `bootstrap` (default, recommended) — each simulated day's return is
     resampled from the stock's own real historical daily returns, so any
     fat tails / skew flagged by the chi-square test carry through into
     the forecast.
   - `normal` — the classic textbook approach: each simulated day's return
     is drawn from Normal(historical mean, historical std). Useful as a
     side-by-side comparison, but understates tail risk if the chi-square
     test rejected normality.

   The historical-frequency number (steps 1-2) and the simulation number
   (step 5) can differ — that's expected. The historical number averages
   over the stock's *entire* history; the simulation starts from *today's*
   price and (via bootstrap) today's recent volatility character, so it's
   the more genuinely forward-looking estimate of the two.

### Running the simulation

Interactive mode will ask if you want to run it, which method, and how
many simulated paths (default 10,000).

Command-line mode:

```bash
python stock_move_analysis.py --symbol RELIANCE --pct 5 --days 10 \
    --simulate --sim-method bootstrap --n-sims 10000 --seed 42 --mode touch
```

`--seed` is optional — set it to get reproducible simulation results run to run.

## Notes / limitations

- Uses **overlapping** N-day windows (day 1-10, day 2-11, day 3-12, ...),
  which is standard for this kind of study but means samples aren't fully
  independent — the Wilson interval is still a reasonable practical guide,
  just don't over-interpret precision to the decimal point.
- History length matters: very short histories or very large N (few
  non-overlapping windows) will give wide, low-confidence intervals — the
  script warns you when the sample size is thin.
- Yahoo Finance data via `yfinance`; NSE symbols are queried as `SYMBOL.NS`.
