import pandas as pd
import glob
import os

def load_monthly_csvs(config):
    base_path = config["data"]["base_path"]
    dfs = []

    csv_files = glob.glob(os.path.join(base_path, "*.csv"))

    for file in csv_files:
        df = pd.read_csv(file)

        # Extract date from filename (MM_YYYY.csv)
        name = os.path.basename(file).replace(".csv", "")
        month, year = name.split("_")
        df["date"] = pd.to_datetime(f"{year}-{month}")

        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=True)

    # Create point_id from geometry
    geo_col = config["data"]["geo_col"]
    full_df["point_id"] = full_df[geo_col].astype("category").cat.codes

    # Sort for time-series consistency
    full_df.sort_values(["point_id", "date"], inplace=True)

    return full_df
