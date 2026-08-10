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
                MPL_COLUMN,
            ]
        ]
        .copy()
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    predictions = []


    for _ in range(forecast_months):

        next_date = (
            forecast_history[DATE_COLUMN].max()
            + pd.DateOffset(months=1)
        )

        recent_usage = (
            forecast_history[TARGET_COLUMN]
        )

        recent_mpl = (
            forecast_history[MPL_COLUMN]
        )


        # =====================================================
        # DATE / TIME
        # =====================================================

        future_row = {
            "month": next_date.month,
            "year": next_date.year,
            "quarter": next_date.quarter,
            "time_idx": len(forecast_history),
        }


        # =====================================================
        # USAGE FEATURES
        # =====================================================

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


        # Usage range
        future_row["rolling_min_12"] = (
            last_12.min()
        )

        future_row["rolling_max_12"] = (
            last_12.max()
        )


        # Zero-month percentage
        future_row[
            "zero_month_percentage_12"
        ] = (
            last_12.eq(0).mean()
        )


        # Coefficient of variation
        rolling_mean_12 = (
            future_row["rolling_mean_12"]
        )

        rolling_std_12 = (
            future_row["rolling_std_12"]
        )

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


        # Recent versus annual usage
        future_row["recent_vs_annual"] = (
            future_row["rolling_mean_3"]
            -
            future_row["rolling_mean_12"]
        )


        # Trends
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


        # =====================================================
        # MPL FEATURES
        # =====================================================

        last_mpl_6 = (
            recent_mpl.iloc[-6:]
        )

        last_mpl_12 = (
            recent_mpl.iloc[-12:]
        )


        # MPL lags
        future_row["mpl_lag_1"] = (
            recent_mpl.iloc[-1]
        )

        future_row["mpl_lag_3"] = (
            recent_mpl.iloc[-3]
        )

        future_row["mpl_lag_6"] = (
            recent_mpl.iloc[-6]
        )


        # MPL totals
        future_row["mpl_rolling_sum_6"] = (
            last_mpl_6.sum()
        )

        future_row["mpl_rolling_sum_12"] = (
            last_mpl_12.sum()
        )


        # Number of months with MPL activity
        future_row["mpl_months_12"] = (
            (last_mpl_12 > 0).sum()
        )


        # Percentage of months with MPL activity
        future_row["mpl_percentage_12"] = (
            (last_mpl_12 > 0).mean()
        )


        # Potential demand
        future_row["potential_demand_12"] = (
            future_row["rolling_total_12"]
            +
            future_row["mpl_rolling_sum_12"]
        )


        # =====================================================
        # MODEL INPUT
        # =====================================================

        future_features = pd.DataFrame(
            [future_row],
            columns=feature_columns,
        )


        # =====================================================
        # PREDICTION
        # =====================================================

        predicted_usage = float(
            model.predict(
                future_features
            )[0]
        )

        predicted_usage = max(
            0.0,
            predicted_usage,
        )


        predictions.append(
            {
                PART_COLUMN: part_number,
                DATE_COLUMN: next_date,
                "Predicted Usage": predicted_usage,
            }
        )


        # =====================================================
        # UPDATE HISTORY
        # =====================================================

        # Cell 6 only forecasts one month at a time
        # during the rolling backtest.
        #
        # Therefore this future MPL placeholder does
        # not affect the rolling validation results.

        new_history_row = pd.DataFrame(
            {
                DATE_COLUMN: [next_date],
                TARGET_COLUMN: [
                    predicted_usage
                ],
                MPL_COLUMN: [0],
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


print("Version 4 MPL forecast function created.")
