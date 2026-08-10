all_results = []
errors = []

validation_start = pd.Timestamp(
    VALIDATION_START
)

validation_months = FORECAST_MONTHS


for part_number in sorted(
    data[PART_COLUMN].dropna().unique()
):

    print(f"\nProcessing: {part_number}")

    try:

        part_data = (
            data[
                data[PART_COLUMN] == part_number
            ]
            .copy()
            .sort_values(DATE_COLUMN)
            .reset_index(drop=True)
        )

        part_results = []


        for month_number in range(
            validation_months
        ):

            prediction_date = (
                validation_start
                + pd.DateOffset(
                    months=month_number
                )
            )


            # Use all actual usage and MPL
            # information available before
            # the prediction month.
            historical_data = part_data[
                part_data[DATE_COLUMN]
                < prediction_date
            ].copy()


            actual_row = part_data[
                part_data[DATE_COLUMN]
                == prediction_date
            ][
                [
                    DATE_COLUMN,
                    TARGET_COLUMN,
                ]
            ].copy()


            if actual_row.empty:

                raise ValueError(
                    f"No actual usage found for "
                    f"{prediction_date:%Y-%m}"
                )


            training_rows, feature_columns = (
                create_training_features(
                    historical_data
                )
            )


            if training_rows.empty:

                raise ValueError(
                    f"No usable training rows for "
                    f"{prediction_date:%Y-%m}"
                )


            # Retrain using everything available
            # before this prediction month.
            model = train_model(
                training_rows,
                feature_columns,
            )


            # Forecast one month ahead.
            one_month_forecast = (
                forecast_future_months(
                    model=model,
                    historical_data=historical_data,
                    feature_columns=feature_columns,
                    forecast_months=1,
                    part_number=part_number,
                )
            )


            predicted_usage = (
                one_month_forecast[
                    "Predicted Usage"
                ].iloc[0]
            )


            actual_usage = (
                actual_row[
                    TARGET_COLUMN
                ].iloc[0]
            )


            error = (
                predicted_usage
                - actual_usage
            )


            part_results.append(
                {
                    PART_COLUMN: part_number,
                    DATE_COLUMN: prediction_date,
                    "Predicted Usage": (
                        predicted_usage
                    ),
                    "Actual Usage": (
                        actual_usage
                    ),
                    "Error": error,
                    "Absolute Error": abs(
                        error
                    ),
                    "Training Through": (
                        historical_data[
                            DATE_COLUMN
                        ].max()
                    ),
                }
            )


        all_results.append(
            pd.DataFrame(
                part_results
            )
        )

        print(
            f"Finished: {part_number}"
        )


    except Exception as error:

        errors.append(
            {
                PART_COLUMN: str(
                    part_number
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
        "No rolling forecasts "
        "completed successfully."
    )


print(
    "\nVersion 4 rolling backtest complete."
)
