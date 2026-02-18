"""
V2 Training Script — ConvLSTM with all improvements.
Features:
  - LR Scheduler (ReduceLROnPlateau)
  - Gradient Clipping
  - Combined Loss (MSE + SSIM)
  - TensorBoard Logging
  - Empty Dataset Guard
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml

# Import our custom modules
from models.convlstm import ConvLSTM
from models.losses import CombinedLoss
from data.tiff_loader import GeoTIFFDataset

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


def load_config(config_path="config.yaml"):
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_loss(config):
    """Build loss function from config."""
    loss_cfg = config.get('loss', {})
    loss_type = loss_cfg.get('type', 'mse')
    
    if loss_type == 'mse':
        return nn.MSELoss()
    elif loss_type == 'mse+ssim':
        ssim_weight = loss_cfg.get('ssim_weight', 0.3)
        channels = config['model'].get('output_dim', config['model']['input_dim'])
        return CombinedLoss(ssim_weight=ssim_weight, channels=channels)
    else:
        return nn.MSELoss()


def train(config):
    # ── Device ──
    if config['device'] == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config['device'])
    print(f"🖥  Using device: {device}")

    # ── Data ──
    print("📂 Initializing Data Loader...")
    try:
        full_dataset = GeoTIFFDataset(config, mode='train')
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # Empty Dataset Guard
    if len(full_dataset) == 0:
        print("❌ Dataset is empty! Possible causes:")
        print(f"   • Not enough consecutive months (need {config['data']['sequence_length']} + {config['data']['prediction_horizon']})")
        print(f"   • No files matching pattern: {config['data']['base_path']}")
        print(f"   • Image smaller than patch_size ({config['data']['image_size']}px)")
        sys.exit(1)

    # Validation Split
    val_split = config['training'].get('val_split', 0.2)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"📊 Split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == 'cuda'
    )

    # ── Model ──
    print("🧠 Initializing ConvLSTM Model...")
    kernel_size = tuple(config['model']['kernel_size'])  # Safe: [3, 3] → (3, 3)

    model = ConvLSTM(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        kernel_size=kernel_size,
        num_layers=config['model']['num_layers'],
        batch_first=True,
        return_all_layers=False,
        prediction_horizon=config['data']['prediction_horizon'],
        output_dim=config['model'].get('output_dim', None),
        output_activation=config['model'].get('output_activation', 'tanh'),
        use_attention=config['model'].get('attention', False)
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📐 Parameters: {trainable_params:,} trainable / {total_params:,} total")

    # ── Loss, Optimizer, Scheduler ──
    criterion = build_loss(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # LR Scheduler
    sched_cfg = config['training'].get('scheduler', {})
    scheduler = None
    if sched_cfg:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=sched_cfg.get('factor', 0.5),
            patience=sched_cfg.get('patience', 5),
            min_lr=sched_cfg.get('min_lr', 1e-6),
            verbose=True
        )

    # ── Training Config ──
    epochs = config['training']['epochs']
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    grad_clip = config['training'].get('grad_clip', None)
    log_interval = config['training'].get('log_interval', 10)

    # Early Stopping
    early_stopping = config['training'].get('early_stopping', False)
    patience = config['training'].get('patience', 10)
    best_val_loss = float('inf')
    counter = 0

    # TensorBoard
    writer = None
    if HAS_TB:
        writer = SummaryWriter(log_dir=os.path.join(save_dir, "runs"))
        print("📈 TensorBoard enabled: tensorboard --logdir checkpoints/runs")

    # ── Training Loop ──
    print(f"\n{'='*60}")
    print(f"  Starting Training: {epochs} epochs")
    print(f"  Loss: {config.get('loss', {}).get('type', 'mse')}")
    print(f"  Attention: {config['model'].get('attention', False)}")
    print(f"  Grad Clip: {grad_clip}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # Gradient Clipping
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            train_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"  Epoch [{epoch+1}/{epochs}] Step [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.6f}")

        avg_train_loss = train_loss / max(len(train_loader), 1)

        # ── VALIDATE ──
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # Current LR
        current_lr = optimizer.param_groups[0]['lr']

        print(f"\n📊 Epoch [{epoch+1}/{epochs}]: "
              f"Train={avg_train_loss:.6f} | Val={avg_val_loss:.6f} | LR={current_lr:.2e}")

        # ── TensorBoard ──
        if writer:
            writer.add_scalar('Loss/train', avg_train_loss, epoch)
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # ── LR Scheduler ──
        if scheduler:
            scheduler.step(avg_val_loss)

        # ── Checkpoint & Early Stopping ──
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_convlstm_model.pt"))
            print(f"  ✅ Saved Best Model (Val Loss: {best_val_loss:.6f})")
        else:
            counter += 1
            print(f"  ⏳ EarlyStopping: {counter}/{patience}")

        if early_stopping and counter >= patience:
            print("\n🛑 Early Stopping triggered.")
            break

    # ── Save Final ──
    torch.save(model.state_dict(), os.path.join(save_dir, "convlstm_final.pt"))

    # Save config snapshot
    import shutil
    shutil.copy2("config.yaml", os.path.join(save_dir, "config_snapshot.yaml"))

    if writer:
        writer.close()

    print(f"\n{'='*60}")
    print(f"  ✅ Training Complete!")
    print(f"  Best Val Loss: {best_val_loss:.6f}")
    print(f"  Models saved to: {save_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ConvLSTM V2")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train(cfg)
