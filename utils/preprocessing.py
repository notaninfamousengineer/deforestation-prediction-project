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
    
    # Add time embeddings
    # We use the index (datetime) to extract month
    months = group.index.month
    group["month_sin"] = np.sin(2 * np.pi * months / 12)
    group["month_cos"] = np.cos(2 * np.pi * months / 12)

    return group.reset_index()

def create_sequences(df, config):
    base_features = config["data"]["features"]
    # We explicitly add the time embeddings to the feature list for usage
    features = base_features + ["month_sin", "month_cos"]
    
    window = config["data"]["window_size"]
    max_gap = config["data"]["max_gap"]

    X, y = [], []

    for _, group in df.groupby("point_id"):
        group = fill_missing_months(group, base_features, max_gap)
        if group is None or len(group) <= window:
            continue

        # Select all features (base + time embeddings)
        values = group[features].values
        # Targets: We usually predict just the base features (NDVI etc), or maybe just NDVI?
        # The original code predicted 'values[i+window]', which implies ALL features.
        # Predicting the DATE features for the next step is trivial (we know what month it is).
        # But if the model training expects target to have same shape as input, let's keep it consistent for now.
        # IMPROVEMENT: We actually only care about predicting NDVI (and maybe others).
        # However, keeping it auto-regressive on all features is standard for multivariate.
        # But predicting month_sin/cos is useless loss.
        # Let's check train.py -> it predicts 'output_size=y.shape[1]'.
        # If we include time features in y, the model will learn to predict time. That's easy but waste of capacity.
        # Optimally, we should only predict the vegetation indices.
        
        # Original code: y.append(values[i + window])
        # Let's stick to predicting everything for minimal friction, 
        # OR better: only predict base features.
        
        for i in range(len(values) - window):
            X.append(values[i:i + window])
            # Only predict the base vegetation indices, not time
            y.append(group[base_features].values[i + window])

    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)

    return X, y
