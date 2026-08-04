# LOAD AND PREPARE DATA
data = pd.read_csv(INPUT_FILE)

required_columns = {
    PART_COLUMN,
    DATE_COLUMN,
    TARGET_COLUMN,
}

missing_columns = required_columns.difference(
    data.columns
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{sorted(missing_columns)}"
    )

data[DATE_COLUMN] = pd.to_datetime(
    data[DATE_COLUMN],
    errors="raise",
)

data[TARGET_COLUMN] = pd.to_numeric(
    data[TARGET_COLUMN],
    errors="raise",
)

# Keep January 2023 and later.
data = data[
    data[DATE_COLUMN] >= pd.Timestamp("2023-01-01")
].copy()

data = (
    data
    .dropna(
        subset=[
            PART_COLUMN,
            DATE_COLUMN,
            TARGET_COLUMN,
        ]
    )
    .sort_values(
        [
            PART_COLUMN,
            DATE_COLUMN,
        ]
    )
    .reset_index(drop=True)
)

duplicates = data.duplicated(
    [
        PART_COLUMN,
        DATE_COLUMN,
    ],
    keep=False,
)

if duplicates.any():
    duplicate_rows = data.loc[
        duplicates,
        [
            PART_COLUMN,
            DATE_COLUMN,
        ],
    ]

    raise ValueError(
        "Duplicate part/month rows were found:\n"
        f"{duplicate_rows.head(20)}"
    )

print(f"Rows loaded: {len(data):,}")
print(
    f"Unique parts: "
    f"{data[PART_COLUMN].nunique():,}"
)
print(
    f"First date: "
    f"{data[DATE_COLUMN].min()}"
)
print(
    f"Last date: "
    f"{data[DATE_COLUMN].max()}"
)
