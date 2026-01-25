import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from data.loader import load_monthly_csvs
from utils.preprocessing import create_sequences
from models.lstm_model import LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load config
with open("config.yaml") as f:
    config = yaml.safe_load(f)

# Load & prepare data
df = load_monthly_csvs(config)
X, y = create_sequences(df, config)

print("X:", X.shape, "y:", y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=config["data"]["test_split"],
    random_state=42,
    shuffle=True
)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=config["training"]["batch_size"],
    shuffle=True
)

test_loader = DataLoader(
    TensorDataset(X_test, y_test),
    batch_size=config["training"]["batch_size"]
)

# Model
model = LSTMModel(
    input_size=X.shape[2],
    hidden_size=config["model"]["lstm_units"],
    dropout=config["model"]["dropout"],
    output_size=y.shape[1]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config["training"]["learning_rate"]
)

# Training loop
best_val_loss = float('inf')

# Training loop
for epoch in range(config["training"]["epochs"]):
    model.train()
    train_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            val_loss += criterion(model(xb), yb).item()

    val_loss /= len(test_loader)

    print(f"Epoch {epoch+1}/{config['training']['epochs']} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Checkpoint: Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_lstm_ndvi_model.pt")
        print(f"  >>> Saved Best Model (Val Loss: {best_val_loss:.6f})")

print(f"✅ Training complete. Best Validation Loss: {best_val_loss:.6f}")
