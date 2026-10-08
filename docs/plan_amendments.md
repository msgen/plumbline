# Plan amendments

Changes to the implementation plan (the PDF in the repo root) after review. Where this file and
the PDF disagree, this file wins. Gate values are proposals until you confirm them.

## Done in code

| # | Issue | Change |
|---|---|---|
| 1 | Split-adjusted history rewrote past price levels | Bars are stored raw (`bars_raw_1d`, `bars_raw_1m`). Splits are stored as corporate actions. `engines/adjust.py` applies only splits effective by `as_of`. Test: a reverse split after `as_of` cannot change the universe at `as_of` (`tests/leakage/test_split_adjustment.py`). Replaces point-in-time rule 4 on page 9. |
| 3 | Catalyst survivorship unaudited | EDGAR symbols with no CIK are recorded (`EdgarFilings.unmapped`) and printed. E0 now reports filings coverage for current vs delisted names, and news coverage. No gate yet. |

Not fixed yet: EDGAR still maps tickers through the current ticker file, so delisted issuers have
no filings. Closing that needs a CIK source that covers delisted names, which E0's coverage
numbers will size before E9.

| 8 | Minute bars incomplete for expensive stocks | E0 on the 200-name pilot showed missing regular-session minutes and a minute-volume shortfall growing steadily with price (0.39% missing under $50, 24.5% above $1,000). The pattern fits minute bars skipping odd-lot trades; not confirmed. A price cap on the universe was tried and rejected: expensive stocks stay in the universe. The completeness gate (0.5%, unchanged) is measured on universe days priced below `complete_bars_max_price` (50, config/universe.yaml); the whole-universe and per-price-tier numbers are reported alongside. E3 and E7 results must be broken down by price tier, and features for expensive names must not rely on minute volume (use daily volume, forward-fill empty minutes). XTB fractional-share support decides whether very expensive names are tradable at $500.

| 9 | E1 failed: free IEX volume is too noisy for RVOL | On 17,112 universe symbol-days, scaled-IEX RVOL has a median error of 36% at 09:35, 23.5% at 10:00 and 20.4% at 10:30 (gate: under 15% by 10:00). The RVOL >= 2.5 flag agrees with SIP on 41% of days either feed flags (gate 90%). Leaving out the opening-auction minute did not help (23.7% at 10:00). IEX VWAP is close to SIP VWAP (about 5 to 8 bps median). Decision pending: see below. |

## Consequence of E1 for all research

Features must be computed from the data that will exist live. Backtests that use true SIP volume
for a feature (RVOL, volume-weighted anything) while live trading uses IEX estimates would
produce live signals that differ from the backtest on roughly half the flagged days, and E13
replay parity would fail. Either live data is upgraded to the full feed, or research features
use the same IEX-scaled estimate (so the research sees the same noise) or avoid live volume.

## Built after E1 (research code, not yet run on real data)

- Triple-barrier labeler (`engines/labeling.py`): target, stop, time-out and no-trade outcomes;
  entry delay; spread and slippage in the fills; time-outs keep their realised R (amendment 2).
- Setup detectors (`engines/setups.py`): A (opening-range breakout), B (VWAP reclaim) and C
  (gap and go, price part only; the catalyst check waits for E9/E10). A and C run with three
  volume-trigger variants: `sip` (true RVOL), `iex_scaled` (what free live data gives) and
  `none`. B carries variant `n/a`. First signal per symbol, day, strategy and variant.
- `engines/features.py` computes RVOL by minute from both feeds; `engines/validation.py` has
  the day-clustered bootstrap and the E3 gate lines (amendment 4).
- Each strategy, variant, target and delay is one trial; the count is saved with the run.
- Rule definitions (opening range 15 minutes, 1x ATR stop, 2R target, 60-minute limit, signals
  between 09:35 and 11:00) are starting values in config/strategies.yaml, not tuned.

## XTB fee table read (30.09.2026): what it confirms and what it does not

Confirmed by the table: real shares and ETFs, 0% commission (minimum 0 EUR) up to EUR 100,000 of
monthly turnover across all accounts, then 0.2% with a EUR 10 minimum per transaction; 0.5%
currency conversion (0.8% on weekends and holidays) when the instrument currency differs from the
account currency; US SEC fee 0.00206% of value sold; custody fee only above EUR 250,000; fractional
rights are listed with the same commission. Transaction taxes on foreign purchases: France 0.4%
(companies above EUR 1bn), Spain 0.2%, Italy 0.1%, UK stamp duty 0.5% to 1%. CFDs carry a 0.30%
mark-up plus financing: not used. Dividends are withheld at the highest local rate unless a treaty
applies; US-listed shares allow a W-8BEN form.

Not confirmed, to check in xStation: (1) which ETFs are offered as real shares. The table lists ETFs
on US markets only as CFDs, and EU retail rules often block US-domiciled ETFs, so the plan to use
US-listed ETFs in dollars may not be possible; European-listed ETFs may need a currency conversion
unless they have a dollar line. (2) Whether fractional rights cover US shares.

Not in the table: income tax on trading profit. A separate document supplied by the user (an
earlier research report, not checked here) says an intermediary registered in Romania withholds
6% on each winning trade held under a year (3% above a year) with no offset for losses, while a
foreign broker taxes 16% of the net annual gain with losses offset. If that applies to XTB's Romanian
branch, tax would take about 0.05R per trade at a 40% win rate and 2R average win, as large as the
edges tested so far. Verify with XTB (its Romanian tax help page) or an accountant before relying on
any short-term result.

## E3 first run and what followed

First E3 run (7 combinations, one geometry: 1x daily ATR stop, 2R target, 60 minutes): every
strategy and volume variant had a negative mean R after costs (-0.014 to -0.074), positive in
0 to 50% of years, and 89 to 100% of trades timed out. The volume variants did not differ.
The geometry was a poor fit for the horizon (a 2R target is about 5% away), so the run was
low-powered, not conclusive. The runner now reports gross R and average stop distance, and runs
the plan's E5 grid once, fixed in advance: stops (1x ATR, 0.5x ATR, structure), targets
(1.5R, 2R, 3R), time limits (30, 60, 90 minutes). Every combination is a trial and the gate's
confidence is raised for the trial count (Bonferroni). Do not add combinations after seeing
results without recounting.

## E3 second run: short-term rule-based path stopped

189 trials (3 strategies, 3 volume variants where they apply, 3 stop rules, 3 targets,
3 time limits), 516,213 labelled signals, 2023-10 to 2026-09. No combination passes the
trial-adjusted gate. One of 189 has a positive net mean R (+0.003, lower bound -0.256). Gross R
(before costs) is at most about +0.07R in the best rows, which are the best of 189 by net result and
rest on 126 to 374 trades. The two large-sample strategies (A without a volume filter, B) are
near -0.03R net. Under the decision rule agreed before the run (no pass and gross R near zero
means stop), the simple short-term setups A, B and C have no edge worth trading in this data.
Not tested: catalyst-filtered setups (news and filings classifier, E9/E10), because they were
not built. Reusable: data layer, point-in-time universe, labeler, validation, experiment store.

## To build into the engines (not written yet)

**2. Expectancy must include time-outs (labeling, ranking, E12).** Every label has one of three
outcomes: target, stop or time-out, and stores its realised R at exit. Replace the page-10
formula with

    EV_R = p_target * R_target - p_stop * 1 + p_timeout * E[R | timeout] - c

`p_target` before `p_stop` stays as a separate reported metric. E12 ranks on `EV_R` with the
time-out term, and the calibration check in E7 is on the three-way outcome.

**4. E3 needs a clustered precision criterion (validation).** Signals on the same day or event
are not independent. Proposed gate: at least 300 signals on at least 100 distinct days, and the
lower bound of a day-clustered bootstrap (resample trading days, 10,000 draws, fixed seed) of
mean R after costs is above 0 at 90%, pooled out of sample. The 60%-of-years rule stays as an
additional check. Applies equally to E5, E6 and E12.

**5. Fill basis (costs, E2, E6).** A signal fills at the modelled XTB ask:
`reference_mid(t + delay) + half_spread(bucket) + slippage`, where `reference_mid` is the Alpaca
mid at the delayed time and `half_spread` is calibrated per time-of-day bucket from the E2 quote
log, concentrating on the setup windows (09:35 to 10:30 New York). E6 evaluates the same basis at
each delay, so delay and spread effects are not mixed.

**6. Purge and embargo (validation).** Splits are by whole trading days. A training label is
dropped if its window `[entry, exit]` overlaps any test day. Embargo is one trading day after each
test block, which exceeds the longest horizon (90 minutes, labels never run past the close).
Test: for every split, `max(train label exit) < min(test start)`.

**7. Replay parity on live-captured bars (scan, E13).** The scan manager persists the bars it
fetched live (IEX, polled) under `live/captured_bars`. Replay parity runs those captured bars
through the same engines and compares signal lists. A separate comparison of SIP research bars to
IEX live bars is E1, so feed differences are measured there, not hidden in parity.
