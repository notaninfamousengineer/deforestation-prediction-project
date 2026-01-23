import numpy as np
import pandas as pd
import torch

def fill_missing_months(group, features, max_gap):
    group = group.set_index("date").asfreq("MS")

    # Drop long gaps
    if group[features].isna().all(axis=1).rolling(max_gap + 1).sum().max() > max_gap:
        return None

    group[features] = group[features].interpolate(limit=max_gap)
    group = group.dropna(subset=features)

    return group.reset_index()

def create_sequences(df, config):
    features = config["data"]["features"]
    window = config["data"]["window_size"]
    max_gap = config["data"]["max_gap"]

    X, y = [], []

    for _, group in df.groupby("point_id"):
        group = fill_missing_months(group, features, max_gap)
        if group is None or len(group) <= window:
            continue

        values = group[features].values

        for i in range(len(values) - window):
            X.append(values[i:i + window])
            y.append(values[i + window])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    return X, y
