# Parts Usage Forecast Model — Version 5

## Overview

Version 5 builds on the Version 3 rolling baseline and introduces **Avg Commits/Month** as a business-demand feature.

Commitments represent real demand already tied to a future job, sales order, or other requirement for the part. A commitment may remain open across multiple months until the part is eventually used.

## Purpose

The purpose of Version 5 is to test whether historical commitment levels improve forecast accuracy beyond historical usage patterns alone.

Version 5 keeps the Version 3 rolling architecture and adds commitment features without relying on MPL features.

## Required Input Columns

- `fpartno` — part number
- `Date` — month associated with the data
- `Monthly Inventory Issues` — actual monthly usage
- `Avg Commits/Month` — average commitment quantity recorded during the month

## Model

- Algorithm: `LGBMRegressor`
- Objective: Poisson
- Model type: supervised machine learning / predictive AI
- Validation style: rolling one-month-ahead backtest

## Rolling Validation Approach

The model is retrained separately for each forecast month using all actual information available before the prediction month.

Example:

- January 2026 uses data through December 2025
- February 2026 uses data through January 2026
- March 2026 uses data through February 2026
- This continues through the validation period

This is intended to reflect how a future automated monthly retraining process could work.

## Existing Usage Features

Version 5 retains the Version 3 usage feature set:

- Month, year, quarter, and time index
- Usage lags: 1, 2, 3, 6, and 12 months
- Rolling means: 3, 6, and 12 months
- Rolling medians: 3, 6, and 12 months
- Rolling 12-month total
- Rolling standard deviations: 3, 6, and 12 months
- Rolling 12-month minimum and maximum
- Zero-month percentage over 12 months
- 12-month coefficient of variation
- Recent 3-month average versus 12-month average
- 3-month and 6-month trends

## Commitment Features

Version 5 adds:

- `commits_lag_1`
- `commits_lag_2`
- `commits_lag_3`
- `commits_lag_6`
- `commits_rolling_mean_3`
- `commits_rolling_mean_6`
- `commits_rolling_mean_12`
- `commits_rolling_sum_3`
- `commits_rolling_sum_6`
- `commits_recent_vs_annual`

## Why Commitments Were Added

Commitments represent confirmed demand that historical usage alone cannot see.

A part may be committed to a job months before the actual inventory issue occurs. Because of that, commitments can act as a leading indicator of future usage.

The model does not assume:

> commitments this month = usage this month

Instead, it uses lagged and rolling commitment features so LightGBM can learn the timing relationship from historical data.

## Initial Data Findings

Commitment data showed a substantially stronger historical relationship with actual usage than MPL data.

The relationship also remained meaningful across multiple lag periods, supporting the idea that commitments behave as a forward-looking demand signal rather than only a same-month indicator.

Because the strength of the relationship varies by part, LightGBM is allowed to determine how much weight commitment features receive.

## Outputs

Version 5 can produce:

- Part number
- Prediction month
- Training-through month
- Decimal predicted usage
- Rounded predicted usage
- Actual usage
- Error
- Absolute error
- Overall MAE
- Commitment diagnostic summaries
- S3 backtest output

## Evaluation

Version 5 should be compared directly against the Version 3 rolling baseline using the same parts, historical range, validation period, model parameters, and MAE calculation.

The main question is:

> Does historical commitment information reduce Average Absolute Error compared with the rolling usage-only baseline?

## Important Data Consideration

Missing commitment values should only be converted to zero when a blank truly means that no commitments existed.

If a blank instead means the value was unavailable or not captured, it should remain missing rather than be treated as zero.

## Key Takeaway

Version 5 introduces a true business-demand signal into the forecasting process. Unlike features derived only from historical usage, commitments provide information about demand that is already known to exist before the final inventory issue occurs.

This makes commitments one of the most promising feature categories tested for improving the forecasting model.
