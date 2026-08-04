# OPTIONAL: CALCULATES OVERALL MAE
average_absolute_error = (
    results["Absolute Error"].mean()
)

print(
    "Version 2 Average Absolute Error: "
    f"{average_absolute_error:.2f} units"
)
