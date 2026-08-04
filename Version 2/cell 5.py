# GENERATIVE THE RECURSIVE FORECAST
def forecast_future_months(
    model,
    historical_data,
    feature_columns,
    forecast_months,
    part_number,
):

    forecast_history = (
        historical_data[
            [
                DATE_COLUMN,
                TARGET_COLUMN,
            ]
        ]
        .copy()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    predictions = []

    for _ in range(forecast_months):

        next_date = (
            forecast_history[
                DATE_COLUMN
            ].max()
            + pd.DateOffset(months=1)
        )

        recent_usage = (
            forecast_history[TARGET_COLUMN]
        )

        future_row = {
            "month": next_date.month,
            "year": next_date.year,
            "quarter": next_date.quarter,
            "time_idx": len(
                forecast_history
            ),
        }

        # Lag features
        for lag in LAGS:
            future_row[f"lag_{lag}"] = (
                recent_usage.iloc[-lag]
            )

        last_3 = recent_usage.iloc[-3:]
        last_6 = recent_usage.iloc[-6:]
        last_12 = recent_usage.iloc[-12:]

        # Rolling averages
        future_row["rolling_mean_3"] = (
            last_3.mean()
        )

        future_row["rolling_mean_6"] = (
            last_6.mean()
        )

        future_row["rolling_mean_12"] = (
            last_12.mean()
        )

        # Rolling medians
        future_row["rolling_median_3"] = (
            last_3.median()
        )

        future_row["rolling_median_6"] = (
            last_6.median()
        )

        future_row["rolling_median_12"] = (
            last_12.median()
        )

        # Annual usage
        future_row["rolling_total_12"] = (
            last_12.sum()
        )

        # Standard deviations
        future_row["rolling_std_3"] = (
            last_3.std()
        )

        future_row["rolling_std_6"] = (
            last_6.std()
        )

        future_row["rolling_std_12"] = (
            last_12.std()
        )

        # Annual range
        future_row["rolling_min_12"] = (
            last_12.min()
        )

        future_row["rolling_max_12"] = (
            last_12.max()
        )

        # Zero-usage percentage
        future_row[
            "zero_month_percentage_12"
        ] = last_12.eq(0).mean()

        rolling_mean_12 = (
            future_row["rolling_mean_12"]
        )

        rolling_std_12 = (
            future_row["rolling_std_12"]
        )

        # Coefficient of variation
        if rolling_mean_12 != 0:
            future_row[
                "coefficient_variation_12"
            ] = (
                rolling_std_12
                / rolling_mean_12
            )
        else:
            future_row[
                "coefficient_variation_12"
            ] = 0.0

        # Recent versus annual demand
        future_row["recent_vs_annual"] = (
            future_row["rolling_mean_3"]
            -
            future_row["rolling_mean_12"]
        )

        # Trend features
        future_row["trend_3"] = (
            recent_usage.iloc[-1]
            -
            recent_usage.iloc[-3]
        )

        future_row["trend_6"] = (
            recent_usage.iloc[-1]
            -
            recent_usage.iloc[-6]
        )

        future_features = pd.DataFrame(
            [future_row],
            columns=feature_columns,
        )

        predicted_usage = float(
            model.predict(
                future_features
            )[0]
        )

        # Usage cannot be negative.
        predicted_usage = max(
            0.0,
            predicted_usage,
        )

        predictions.append(
            {
                PART_COLUMN: part_number,
                DATE_COLUMN: next_date,
                "Predicted Usage": (
                    predicted_usage
                ),
            }
        )

        # Use the decimal prediction internally
        # for the next forecast month.
        new_history_row = pd.DataFrame(
            {
                DATE_COLUMN: [next_date],
                TARGET_COLUMN: [
                    predicted_usage
                ],
            }
        )

        forecast_history = pd.concat(
            [
                forecast_history,
                new_history_row,
            ],
            ignore_index=True,
        )

    return pd.DataFrame(predictions)


print(
    "Version 2 forecast function created."
)
