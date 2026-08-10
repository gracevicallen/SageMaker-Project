from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


INPUT_FILE = "V4 Usage Training.csv"
OUTPUT_FOLDER = Path("forecast_results")

# Rolling backtest settings
VALIDATION_START = "2026-01-01"
FORECAST_MONTHS = 6

TARGET_COLUMN = "Monthly Inventory Issues"
PART_COLUMN = "fpartno"
DATE_COLUMN = "Date"

# Version 5 commitment feature
COMMITS_COLUMN = "Avg Commits/Month"

LAGS = [1, 2, 3, 6, 12]

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

print("Version 5 settings loaded.")
