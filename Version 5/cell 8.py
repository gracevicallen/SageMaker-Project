profile_data = data[
    data[DATE_COLUMN]
    < pd.Timestamp(VALIDATION_START)
].copy()


demand_profile = (
    profile_data
    .groupby(PART_COLUMN)
    .agg(

        Average_Monthly_Usage=(
            TARGET_COLUMN,
            "mean",
        ),

        Median_Monthly_Usage=(
            TARGET_COLUMN,
            "median",
        ),

        Zero_Month_Percentage=(
            TARGET_COLUMN,
            lambda x: (
                x == 0
            ).mean(),
        ),

        Nonzero_Months=(
            TARGET_COLUMN,
            lambda x: (
                x > 0
            ).sum(),
        ),

        Average_Commits=(
            COMMITS_COLUMN,
            "mean",
        ),

        Median_Commits=(
            COMMITS_COLUMN,
            "median",
        ),

        Maximum_Commits=(
            COMMITS_COLUMN,
            "max",
        ),

        Months_With_Commits=(
            COMMITS_COLUMN,
            lambda x: (
                x > 0
            ).sum(),
        ),

        Commit_Month_Percentage=(
            COMMITS_COLUMN,
            lambda x: (
                x > 0
            ).mean(),
        ),
    )
    .reset_index()
)


demand_profile[
    "Zero_Month_Percentage"
] *= 100

demand_profile[
    "Commit_Month_Percentage"
] *= 100


# Round for presentation
demand_profile[
    "Average_Monthly_Usage"
] = (
    demand_profile[
        "Average_Monthly_Usage"
    ].round(2)
)

demand_profile[
    "Median_Monthly_Usage"
] = (
    demand_profile[
        "Median_Monthly_Usage"
    ].round(2)
)

demand_profile[
    "Average_Commits"
] = (
    demand_profile[
        "Average_Commits"
    ].round(2)
)

demand_profile[
    "Median_Commits"
] = (
    demand_profile[
        "Median_Commits"
    ].round(2)
)

demand_profile[
    "Zero_Month_Percentage"
] = (
    demand_profile[
        "Zero_Month_Percentage"
    ].round(1)
)

demand_profile[
    "Commit_Month_Percentage"
] = (
    demand_profile[
        "Commit_Month_Percentage"
    ].round(1)
)


display(demand_profile)
