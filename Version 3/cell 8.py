# OPTIONAL CALCULATE MAE
average_absolute_error = (
    results["Absolute Error"].mean()
)

print(
    "Version 3 Rolling Average Absolute Error: "
    f"{average_absolute_error:.2f} units"
)
