results = (
    pd.concat(
        all_results,
        ignore_index=True,
    )
    .sort_values(
        [
            PART_COLUMN,
            DATE_COLUMN,
        ]
    )
    .reset_index(drop=True)
)


# Keep decimal predictions for accuracy calculations.
# Create rounded values only for presentation.
results["Predicted Usage Rounded"] = (
    results["Predicted Usage"]
    .round()
    .astype(int)
)

results["Actual Usage Rounded"] = (
    results["Actual Usage"]
    .round()
    .astype(int)
)


display(
    results[
        [
            PART_COLUMN,
            DATE_COLUMN,
            "Training Through",
            "Predicted Usage Rounded",
            "Actual Usage Rounded",
            "Error",
            "Absolute Error",
        ]
    ]
)
