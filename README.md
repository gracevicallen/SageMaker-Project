[README.md](https://github.com/user-attachments/files/29900422/README.md)
# SageMaker-Project

# Monthly Parts Usage Forecast Validation

This project trains one LightGBM model per part using historical monthly inventory usage.

It simulates what would have happened if the model had been run on December 31, 2025:

- Training data: all months before January 1, 2026
- Forecast period: January through June 2026
- Validation: predicted usage compared with actual usage

## Input

Default file:

```text
Usage Training.csv
```

Required columns:

```text
fpartno
Date
Monthly Inventory Issues
```

Each part should have one row per month.

## Features used

- Month, year, quarter, and time index
- Usage lags of 1, 2, 3, 6, and 12 months
- Rolling averages over 3, 6, and 12 months
- Rolling standard deviations over 3 and 6 months

All lag and rolling features use prior months only.

## Outputs

The script creates a `forecast_results` folder containing:

- One monthly validation CSV per part
- One actual-versus-predicted PNG chart per part
- `All_Parts_Monthly_Forecast_Validation.csv`
- `All_Parts_Forecast_Summary.csv`
- `Forecast_Errors.csv` when any part fails

## Run

```bash
python forecast_validation.py
```

## Optional S3 upload

Update these settings in `forecast_validation.py`:

```python
UPLOAD_TO_S3 = True
S3_BUCKET = "your-bucket-name"
S3_PREFIX = "forecast-results/all-parts"
```

Do not store AWS access keys in GitHub. Use the SageMaker execution role or another approved IAM role.
