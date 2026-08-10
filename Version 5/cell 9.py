import boto3
from sagemaker_studio import Project
import io


# Get project S3 path
proj = Project()
project_s3_root = proj.s3.root

s3_parts = (
    project_s3_root
    .replace("s3://", "")
    .split("/", 1)
)

bucket = s3_parts[0]

prefix = (
    s3_parts[1]
    if len(s3_parts) > 1
    else ""
)

s3 = boto3.client("s3")


# Select results to upload
results_to_upload = results[
    [
        PART_COLUMN,
        DATE_COLUMN,
        "Training Through",
        "Predicted Usage",
        "Predicted Usage Rounded",
        "Actual Usage",
        "Error",
        "Absolute Error",
    ]
].copy()


# Create CSV in memory
csv_buffer = io.StringIO()

results_to_upload.to_csv(
    csv_buffer,
    index=False,
)


# Separate Version 5 output file
s3_key = (
    f"{prefix}/results/"
    "version_5_commitments_rolling_backtest_results.csv"
)


# Upload to S3
s3.put_object(
    Bucket=bucket,
    Key=s3_key,
    Body=csv_buffer.getvalue().encode(
        "utf-8"
    ),
    ContentType="text/csv",
)


s3_path = (
    f"s3://{bucket}/{s3_key}"
)


print(
    "Successfully uploaded Version 5 "
    "commitment rolling backtest results to S3!"
)

print(
    f"S3 path: {s3_path}"
)

print(
    f"Rows uploaded: "
    f"{len(results_to_upload)}"
)
