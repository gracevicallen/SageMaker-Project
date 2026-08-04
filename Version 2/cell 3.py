# CREATE THE FEATURES
def create_training_features(
    historical_data,
):

    feature_data = historical_data.copy()

    # Date and time features
    feature_data["month"] = (
        feature_data[DATE_COLUMN].dt.month
    )

    feature_data["year"] = (
        feature_data[DATE_COLUMN].dt.year
    )

    feature_data["quarter"] = (
        feature_data[DATE_COLUMN].dt.quarter
    )

    feature_data["time_idx"] = np.arange(
        len(feature_data)
    )

    # Lag features
    for lag in LAGS:
        feature_data[f"lag_{lag}"] = (
            feature_data[TARGET_COLUMN].shift(lag)
        )

    # Shift usage so the current month's actual usage
    # is never used to predict itself.
    prior_usage = (
        feature_data[TARGET_COLUMN].shift(1)
    )

    # Rolling averages
    feature_data["rolling_mean_3"] = (
        prior_usage.rolling(3).mean()
    )

    feature_data["rolling_mean_6"] = (
        prior_usage.rolling(6).mean()
    )

    feature_data["rolling_mean_12"] = (
        prior_usage.rolling(12).mean()
    )

    # Rolling medians
    feature_data["rolling_median_3"] = (
        prior_usage.rolling(3).median()
    )

    feature_data["rolling_median_6"] = (
        prior_usage.rolling(6).median()
    )

    feature_data["rolling_median_12"] = (
        prior_usage.rolling(12).median()
    )

    # Total usage over the prior 12 months
    feature_data["rolling_total_12"] = (
        prior_usage.rolling(12).sum()
    )

    # Demand variability
    feature_data["rolling_std_3"] = (
        prior_usage.rolling(3).std()
    )

    feature_data["rolling_std_6"] = (
        prior_usage.rolling(6).std()
    )

    feature_data["rolling_std_12"] = (
        prior_usage.rolling(12).std()
    )

    # Annual demand range
    feature_data["rolling_min_12"] = (
        prior_usage.rolling(12).min()
    )

    feature_data["rolling_max_12"] = (
        prior_usage.rolling(12).max()
    )

    # Percentage of prior 12 months
    # with zero usage
    feature_data[
        "zero_month_percentage_12"
    ] = (
        prior_usage
        .rolling(12)
        .apply(
            lambda values: (
                values == 0
            ).mean(),
            raw=True,
        )
    )

    # Demand variability relative to
    # average demand
    feature_data[
        "coefficient_variation_12"
    ] = (
        feature_data["rolling_std_12"]
        /
        feature_data[
            "rolling_mean_12"
        ].replace(0, np.nan)
    )

    # Recent usage compared with
    # the annual monthly average
    feature_data["recent_vs_annual"] = (
        feature_data["rolling_mean_3"]
        -
        feature_data["rolling_mean_12"]
    )

    # Demand trends
    feature_data["trend_3"] = (
        feature_data["lag_1"]
        -
        feature_data["lag_3"]
    )

    feature_data["trend_6"] = (
        feature_data["lag_1"]
        -
        feature_data["lag_6"]
    )

    feature_columns = [
        "month",
        "year",
        "quarter",
        "time_idx",

        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",

        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",

        "rolling_median_3",
        "rolling_median_6",
        "rolling_median_12",

        "rolling_total_12",

        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",

        "rolling_min_12",
        "rolling_max_12",

        "zero_month_percentage_12",
        "coefficient_variation_12",
        "recent_vs_annual",

        "trend_3",
        "trend_6",
    ]

    training_rows = (
        feature_data
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=feature_columns
        )
        .reset_index(drop=True)
    )

    return (
        training_rows,
        feature_columns,
    )


print(
    "Version 2 feature function created."
)
