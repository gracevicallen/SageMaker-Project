# IMPORTS AND SETTINGS
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


INPUT_FILE = "Parts Training Usage.csv"
OUTPUT_FOLDER = Path("forecast_results")

# Fixed historical backtest:
# Train through December 2025 and predict January-June 2026.
TRAINING_CUTOFF = "2026-01-01"
FORECAST_MONTHS = 6

TARGET_COLUMN = "Monthly Inventory Issues"
PART_COLUMN = "fpartno"
DATE_COLUMN = "Date"

LAGS = [1, 2, 3, 6, 12]

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True,
)

print("Settings loaded.")
