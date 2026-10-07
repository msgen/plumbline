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
