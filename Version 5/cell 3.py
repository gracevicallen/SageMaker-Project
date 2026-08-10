def create_training_features(historical_data):

    feature_data = historical_data.copy()

    # =====================================================
    # DATE / TIME FEATURES
    # =====================================================

    feature_data["month"] = (
        feature_data[DATE_COLUMN].dt.month
    )

    feature_data["year"] = (
        feature_data[DATE_COLUMN].dt.year
    )

    feature_data["quarter"] = (
        feature_data[DATE_COLUMN].dt.quarter
    )

    feature_data["time_idx"] = (
        np.arange(len(feature_data))
    )


    # =====================================================
    # USAGE LAG FEATURES
    # =====================================================

    for lag in LAGS:

        feature_data[f"lag_{lag}"] = (
            feature_data[TARGET_COLUMN].shift(lag)
        )


    # Shift usage so current usage can never
    # be used to predict itself.
    prior_usage = (
        feature_data[TARGET_COLUMN].shift(1)
    )


    # =====================================================
    # ROLLING USAGE AVERAGES
    # =====================================================

    feature_data["rolling_mean_3"] = (
        prior_usage.rolling(3).mean()
    )

    feature_data["rolling_mean_6"] = (
        prior_usage.rolling(6).mean()
    )

    feature_data["rolling_mean_12"] = (
        prior_usage.rolling(12).mean()
    )


    # =====================================================
    # ROLLING USAGE MEDIANS
    # =====================================================

    feature_data["rolling_median_3"] = (
        prior_usage.rolling(3).median()
    )

    feature_data["rolling_median_6"] = (
        prior_usage.rolling(6).median()
    )

    feature_data["rolling_median_12"] = (
        prior_usage.rolling(12).median()
    )


    # =====================================================
    # ANNUAL USAGE
    # =====================================================

    feature_data["rolling_total_12"] = (
        prior_usage.rolling(12).sum()
    )


    # =====================================================
    # USAGE VARIABILITY
    # =====================================================

    feature_data["rolling_std_3"] = (
        prior_usage.rolling(3).std()
    )

    feature_data["rolling_std_6"] = (
        prior_usage.rolling(6).std()
    )

    feature_data["rolling_std_12"] = (
        prior_usage.rolling(12).std()
    )


    # =====================================================
    # USAGE RANGE
    # =====================================================

    feature_data["rolling_min_12"] = (
        prior_usage.rolling(12).min()
    )

    feature_data["rolling_max_12"] = (
        prior_usage.rolling(12).max()
    )


    # =====================================================
    # USAGE TRENDS
    # =====================================================

    feature_data["trend_3"] = (
        feature_data["lag_1"]
        - feature_data["lag_3"]
    )

    feature_data["trend_6"] = (
        feature_data["lag_1"]
        - feature_data["lag_6"]
    )


    # =====================================================
    # ZERO-USAGE BEHAVIOR
    # =====================================================

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


    # =====================================================
    # COEFFICIENT OF VARIATION
    # =====================================================

    feature_data[
        "coefficient_variation_12"
    ] = (
        feature_data["rolling_std_12"]
        /
        feature_data[
            "rolling_mean_12"
        ].replace(
            0,
            np.nan,
        )
    )


    # =====================================================
    # RECENT VS ANNUAL USAGE
    # =====================================================

    feature_data["recent_vs_annual"] = (
        feature_data["rolling_mean_3"]
        -
        feature_data["rolling_mean_12"]
    )


    # =====================================================
    # COMMITMENT FEATURES
    # =====================================================

    # Shift commitments so the model does not see
    # the commitment value from the month being predicted.
    prior_commits = (
        feature_data[COMMITS_COLUMN].shift(1)
    )


    # Commitment lags
    feature_data["commits_lag_1"] = (
        feature_data[COMMITS_COLUMN].shift(1)
    )

    feature_data["commits_lag_2"] = (
        feature_data[COMMITS_COLUMN].shift(2)
    )

    feature_data["commits_lag_3"] = (
        feature_data[COMMITS_COLUMN].shift(3)
    )

    feature_data["commits_lag_6"] = (
        feature_data[COMMITS_COLUMN].shift(6)
    )


    # Rolling commitment averages
    feature_data["commits_rolling_mean_3"] = (
        prior_commits.rolling(3).mean()
    )

    feature_data["commits_rolling_mean_6"] = (
        prior_commits.rolling(6).mean()
    )

    feature_data["commits_rolling_mean_12"] = (
        prior_commits.rolling(12).mean()
    )


    # Rolling commitment totals
    feature_data["commits_rolling_sum_3"] = (
        prior_commits.rolling(3).sum()
    )

    feature_data["commits_rolling_sum_6"] = (
        prior_commits.rolling(6).sum()
    )


    # Compare recent commitment pressure with
    # the longer-term commitment level.
    feature_data[
        "commits_recent_vs_annual"
    ] = (
        feature_data[
            "commits_rolling_mean_3"
        ]
        -
        feature_data[
            "commits_rolling_mean_12"
        ]
    )


    # =====================================================
    # FEATURE LIST
    # =====================================================

    feature_columns = [

        # Date / time
        "month",
        "year",
        "quarter",
        "time_idx",

        # Usage lags
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_6",
        "lag_12",

        # Usage averages
        "rolling_mean_3",
        "rolling_mean_6",
        "rolling_mean_12",

        # Usage medians
        "rolling_median_3",
        "rolling_median_6",
        "rolling_median_12",

        # Annual usage
        "rolling_total_12",

        # Usage variability
        "rolling_std_3",
        "rolling_std_6",
        "rolling_std_12",

        # Usage range
        "rolling_min_12",
        "rolling_max_12",

        # Demand behavior
        "zero_month_percentage_12",
        "coefficient_variation_12",
        "recent_vs_annual",

        # Usage trends
        "trend_3",
        "trend_6",

        # Commitment features
        "commits_lag_1",
        "commits_lag_2",
        "commits_lag_3",
        "commits_lag_6",

        "commits_rolling_mean_3",
        "commits_rolling_mean_6",
        "commits_rolling_mean_12",

        "commits_rolling_sum_3",
        "commits_rolling_sum_6",

        "commits_recent_vs_annual",
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
    "Version 5 commitment feature function created."
)
