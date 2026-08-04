# TRAIN THROUGH 2025 AND PREDICT JAN-JUNE 2026
cutoff_date = pd.Timestamp(
    TRAINING_CUTOFF
)

forecast_end = (
    cutoff_date
    + pd.DateOffset(
        months=FORECAST_MONTHS
    )
)

all_results = []
errors = []

for part_number in sorted(
    data[PART_COLUMN]
    .dropna()
    .unique()
):

    print(
        f"\nProcessing: {part_number}"
    )

    try:
        part_data = (
            data[
                data[PART_COLUMN]
                == part_number
            ]
            .copy()
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )

        # Training data ends with
        # December 2025.
        historical_data = (
            part_data[
                part_data[DATE_COLUMN]
                < cutoff_date
            ]
            .copy()
        )

        # Actual January-June 2026 values
        actual_data = (
            part_data[
                (
                    part_data[DATE_COLUMN]
                    >= cutoff_date
                )
                &
                (
                    part_data[DATE_COLUMN]
                    < forecast_end
                )
            ][
                [
                    DATE_COLUMN,
                    TARGET_COLUMN,
                ]
            ]
            .copy()
        )

        if len(actual_data) != FORECAST_MONTHS:
            raise ValueError(
                f"Expected "
                f"{FORECAST_MONTHS} "
                f"actual validation months, "
                f"but found "
                f"{len(actual_data)}."
            )

        training_rows, feature_columns = (
            create_training_features(
                historical_data
            )
        )

        if training_rows.empty:
            raise ValueError(
                "No usable training rows "
                "after feature creation."
            )

        model = train_model(
            training_rows,
            feature_columns,
        )

        forecast = forecast_future_months(
            model=model,
            historical_data=historical_data,
            feature_columns=feature_columns,
            forecast_months=FORECAST_MONTHS,
            part_number=part_number,
        )

        comparison = forecast.merge(
            actual_data,
            on=DATE_COLUMN,
            how="left",
        )

        comparison = comparison.rename(
            columns={
                TARGET_COLUMN: (
                    "Actual Usage"
                )
            }
        )

        comparison["Error"] = (
            comparison["Predicted Usage"]
            -
            comparison["Actual Usage"]
        )

        comparison["Absolute Error"] = (
            comparison["Error"].abs()
        )

        comparison["Training Through"] = (
            historical_data[
                DATE_COLUMN
            ].max()
        )

        all_results.append(comparison)

        print(
            f"Finished: {part_number}"
        )

    except Exception as error:

        errors.append(
            {
                PART_COLUMN: (
                    str(part_number)
                ),
                "Error": str(error),
            }
        )

        print(
            f"Skipped {part_number}: "
            f"{error}"
        )


if not all_results:
    raise RuntimeError(
        "No Version 2 forecasts "
        "completed successfully."
    )

print(
    "\nVersion 2 backtest complete."
)
