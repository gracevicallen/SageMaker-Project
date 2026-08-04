# DISPLAY PREDICTIONS WITH ACTUAL USAGE
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
            "Predicted Usage",
            "Actual Usage",
            "Error",
            "Absolute Error",
        ]
    ]
)
