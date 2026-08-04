# ACCURACY BY PART
part_accuracy = (
    results
    .groupby(PART_COLUMN)
    .agg(
        Average_Actual_Usage=(
            "Actual Usage",
            "mean",
        ),
        Average_Predicted_Usage=(
            "Predicted Usage",
            "mean",
        ),
        Actual_Total=(
            "Actual Usage",
            "sum",
        ),
        Predicted_Total=(
            "Predicted Usage",
            "sum",
        ),
        MAE=(
            "Absolute Error",
            "mean",
        ),
    )
    .reset_index()
)

part_accuracy[
    "Six_Month_Total_Error"
] = (
    part_accuracy["Predicted_Total"]
    -
    part_accuracy["Actual_Total"]
)

display(part_accuracy)
