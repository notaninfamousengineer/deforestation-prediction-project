"""
GeoTIFF Dataset Loader for ConvLSTM V2 — Multi-Region Edition.
Features:
  - Scans multiple region subfolders (each region = independent timeline)
  - Memory caching (pre-loads all TIFs)
  - Temporal encoding (month_sin, month_cos channels)
  - Data augmentation (random flips, 90° rotations)
  - Gap-aware sequence validation
  - Corrupt/tiny file filtering
"""

import os
import math
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio
except ImportError:
    rasterio = None

# Files smaller than this are likely empty/corrupt GEE exports
MIN_FILE_SIZE_BYTES = 50_000


class GeoTIFFDataset(Dataset):
    def __init__(self, config, mode='train'):
        """
        config: dict loaded from config.yaml
        mode: 'train' or 'test'
        """
        self.config = config
        self.mode = mode
        self.base_path = config['data']['base_path']
        self.seq_len = config['data']['sequence_length']
        self.pred_len = config['data']['prediction_horizon']
        self.patch_size = config['data']['image_size']
        self.normalize = config['data'].get('normalize', False)
        self.use_temporal = config['data'].get('temporal_encoding', False)
        self.augment = config['data'].get('augment', False) and mode == 'train'
        self.cache_enabled = config['data'].get('cache_in_memory', False)

        if rasterio is None:
            raise ImportError("Please install rasterio: pip install rasterio")

        # Memory cache: {filepath: np.array}  — must init before _discover_regions
        # because _interpolate_corrupt writes interpolated images here
        self.img_cache = {}

        # Discover regions and build per-region timelines
        self.regions = []  # List of {name, date_to_file, valid_dates}
        self._discover_regions()

        # Cache remaining real files
        if self.cache_enabled and self.regions:
            self._cache_all_files()

        # Build samples across all regions
        self.samples = []
        self._prepare_all_samples()

        total_regions = len(self.regions)
        total_files = sum(len(r['valid_dates']) for r in self.regions)
        print(f"✅ Dataset ready: {len(self.samples)} samples from "
              f"{total_regions} regions ({total_files} total files, "
              f"{'cached' if self.cache_enabled else 'disk'} mode)")

    # ──────────────────────────────────────────
    # REGION DISCOVERY
    # ──────────────────────────────────────────
    def _discover_regions(self):
        """
        Scans base_path for region subfolders.
        If base_path contains TIF files directly, treat it as a single region.
        If base_path contains subdirectories, treat each as a separate region.
        """
        if not os.path.isdir(self.base_path):
            # base_path might be a glob pattern
            parent = os.path.dirname(self.base_path)
            if os.path.isdir(parent):
                self._add_region(parent, os.path.basename(parent))
            return

        # Check if base_path itself contains TIF files
        tifs_in_root = glob.glob(os.path.join(self.base_path, "*.tif"))
        if tifs_in_root:
            self._add_region(self.base_path, os.path.basename(self.base_path))
            return

        # Otherwise, scan subdirectories
        for entry in sorted(os.listdir(self.base_path)):
            subdir = os.path.join(self.base_path, entry)
            if os.path.isdir(subdir):
                self._add_region(subdir, entry)

    def _add_region(self, folder_path, region_name):
        """Parse a single region folder, track corrupt files, and interpolate them."""
        tif_files = sorted(glob.glob(os.path.join(folder_path, "*.tif")))

        date_to_file = {}
        all_dates = []
        corrupt_dates = []  # Dates with corrupt/tiny files

        for f in tif_files:
            basename = os.path.basename(f)
            name_clean = basename.replace(".tif", "").replace("IMG_", "")
            parts = name_clean.split("_")

            if len(parts) >= 2:
                try:
                    year, month = int(parts[0]), int(parts[1])
                    date_str = f"{year}-{month:02d}"

                    file_size = os.path.getsize(f)
                    if file_size < MIN_FILE_SIZE_BYTES:
                        corrupt_dates.append((year, month))
                    else:
                        date_to_file[date_str] = f
                        all_dates.append((year, month))
                except ValueError:
                    continue

        all_dates.sort()
        corrupt_dates.sort()

        if not all_dates:
            return

        # Interpolate corrupt months
        interpolated = 0
        if corrupt_dates:
            interpolated = self._interpolate_corrupt(
                corrupt_dates, all_dates, date_to_file, region_name
            )

        # Re-sort after interpolation
        all_dates.sort()

        self.regions.append({
            'name': region_name,
            'date_to_file': date_to_file,
            'valid_dates': all_dates,
        })
        status = f"{len(all_dates)} files ({all_dates[0][0]}-{all_dates[0][1]:02d} → {all_dates[-1][0]}-{all_dates[-1][1]:02d})"
        if interpolated:
            status += f", 🔧 interpolated {interpolated}/{len(corrupt_dates)} corrupt"
        if len(corrupt_dates) - interpolated > 0:
            status += f", ⚠ {len(corrupt_dates) - interpolated} unfixable"
        print(f"  📍 Region '{region_name}': {status}")

    def _interpolate_corrupt(self, corrupt_dates, valid_dates, date_to_file, region_name):
        """
        For each corrupt month, find the nearest valid previous & next month
        and create an interpolated image = (prev + next) / 2.
        If only one neighbor exists (edge case), use that neighbor directly.
        Returns the number of successfully interpolated months.
        """
        valid_set = set(valid_dates)
        interpolated = 0

        for (c_year, c_month) in corrupt_dates:
            # Search backward for nearest valid month
            prev_img = None
            for offset in range(1, 13):  # Look up to 12 months back
                pm = c_month - offset
                py = c_year + (pm - 1) // 12
                pm = (pm - 1) % 12 + 1
                if (py, pm) in valid_set:
                    prev_key = f"{py}-{pm:02d}"
                    try:
                        with rasterio.open(date_to_file[prev_key]) as src:
                            prev_img = src.read().astype(np.float32)
                            prev_img = np.nan_to_num(prev_img, nan=0.0)
                    except:
                        pass
                    break

            # Search forward for nearest valid month
            next_img = None
            for offset in range(1, 13):  # Look up to 12 months ahead
                nm = c_month + offset
                ny = c_year + (nm - 1) // 12
                nm = (nm - 1) % 12 + 1
                if (ny, nm) in valid_set:
                    next_key = f"{ny}-{nm:02d}"
                    try:
                        with rasterio.open(date_to_file[next_key]) as src:
                            next_img = src.read().astype(np.float32)
                            next_img = np.nan_to_num(next_img, nan=0.0)
                    except:
                        pass
                    break

            # Interpolate
            if prev_img is not None and next_img is not None:
                synthetic = (prev_img + next_img) / 2.0
            elif prev_img is not None:
                synthetic = prev_img
            elif next_img is not None:
                synthetic = next_img
            else:
                continue  # No neighbors found — unfixable

            if self.normalize:
                synthetic = np.clip(synthetic, -1.0, 1.0)

            # Store as an in-memory "file" by caching directly
            date_str = f"{c_year}-{c_month:02d}"
            date_to_file[date_str] = f"__interpolated__:{date_str}"
            valid_dates.append((c_year, c_month))
            self.img_cache[f"__interpolated__:{date_str}"] = synthetic
            interpolated += 1

        return interpolated

    # ──────────────────────────────────────────
    # CACHING
    # ──────────────────────────────────────────
    def _cache_all_files(self):
        """Pre-load all TIF files into memory as numpy arrays."""
        print("💾 Caching all TIFs into memory...")
        for region in self.regions:
            for date_str, filepath in region['date_to_file'].items():
                if filepath not in self.img_cache:
                    with rasterio.open(filepath) as src:
                        img = src.read().astype(np.float32)
                        img = np.nan_to_num(img, nan=0.0)
                        if self.normalize:
                            img = np.clip(img, -1.0, 1.0)
                        self.img_cache[filepath] = img

        total_mb = sum(v.nbytes for v in self.img_cache.values()) / 1e6
        print(f"💾 Cached {len(self.img_cache)} files ({total_mb:.1f} MB)")

    # ──────────────────────────────────────────
    # SEQUENCE BUILDING (per region)
    # ──────────────────────────────────────────
    def _prepare_all_samples(self):
        """Build valid samples across all regions."""
        for region in self.regions:
            self._prepare_region_samples(region)

    def _prepare_region_samples(self, region):
        """Find all valid consecutive-month windows for a single region."""
        needed = self.seq_len + self.pred_len
        valid_dates = region['valid_dates']
        date_to_file = region['date_to_file']

        if len(valid_dates) < needed:
            print(f"  ⚠ Region '{region['name']}': only {len(valid_dates)} files, "
                  f"need {needed}")
            return

        # Get image dimensions from first file
        try:
            first_key = f"{valid_dates[0][0]}-{valid_dates[0][1]:02d}"
            first_file = date_to_file[first_key]
            if self.cache_enabled and first_file in self.img_cache:
                _, height, width = self.img_cache[first_file].shape
            else:
                with rasterio.open(first_file) as src:
                    height, width = src.shape
        except Exception as e:
            print(f"  ❌ Error reading '{region['name']}': {e}")
            return

        h_steps = height // self.patch_size
        w_steps = width // self.patch_size

        if h_steps == 0 or w_steps == 0:
            print(f"  ⚠ Region '{region['name']}': image ({height}x{width}) "
                  f"smaller than patch_size ({self.patch_size})")
            return

        region_samples = 0

        for i in range(len(valid_dates)):
            is_continuous = True
            seq_dates = []

            start_y, start_m = valid_dates[i]

            for j in range(needed):
                if i + j >= len(valid_dates):
                    is_continuous = False
                    break

                exp_m = start_m + j
                exp_y = start_y + (exp_m - 1) // 12
                exp_m = (exp_m - 1) % 12 + 1

                if (exp_y, exp_m) != valid_dates[i + j]:
                    is_continuous = False
                    break
                seq_dates.append((exp_y, exp_m))

            if is_continuous and len(seq_dates) == needed:
                for r in range(h_steps):
                    for c in range(w_steps):
                        self.samples.append({
                            "region": region['name'],
                            "date_to_file": date_to_file,
                            "dates": seq_dates,
                            "row": r * self.patch_size,
                            "col": c * self.patch_size,
                        })
                        region_samples += 1

        print(f"  ✅ Region '{region['name']}': {region_samples} samples")

    # ──────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def _read_patch(self, filepath, row, col):
        """Read a patch from cache or disk. Interpolated files are always in cache."""
        # Always check cache first (interpolated files live here)
        if filepath in self.img_cache:
            full_img = self.img_cache[filepath]
            patch = full_img[:, row:row + self.patch_size, col:col + self.patch_size]
        else:
            window = rasterio.windows.Window(col, row, self.patch_size, self.patch_size)
            with rasterio.open(filepath) as src:
                patch = src.read(window=window).astype(np.float32)
                patch = np.nan_to_num(patch, nan=0.0)
                if self.normalize:
                    patch = np.clip(patch, -1.0, 1.0)
        return patch  # [C, H, W]

    def _add_temporal_encoding(self, patch, month):
        """Append month_sin and month_cos as extra channels."""
        _, h, w = patch.shape
        sin_val = math.sin(2 * math.pi * month / 12)
        cos_val = math.cos(2 * math.pi * month / 12)

        sin_ch = np.full((1, h, w), sin_val, dtype=np.float32)
        cos_ch = np.full((1, h, w), cos_val, dtype=np.float32)

        return np.concatenate([patch, sin_ch, cos_ch], axis=0)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        dates = sample["dates"]
        date_to_file = sample["date_to_file"]
        row = sample["row"]
        col = sample["col"]

        input_frames = []
        target_frames = []

        for i, (year, month) in enumerate(dates):
            date_key = f"{year}-{month:02d}"
            filepath = date_to_file[date_key]
            patch = self._read_patch(filepath, row, col)

            if i < self.seq_len:
                if self.use_temporal:
                    patch = self._add_temporal_encoding(patch, month)
                input_frames.append(torch.from_numpy(patch))
            else:
                target_frames.append(torch.from_numpy(patch))

        X = torch.stack(input_frames)   # [Seq, C_in, H, W]
        y = torch.stack(target_frames)  # [Pred, C_out, H, W]

        if self.augment:
            X, y = self._augment(X, y)

        if self.pred_len == 1:
            y = y.squeeze(0)

        return X, y

    # ──────────────────────────────────────────
    # AUGMENTATION
    # ──────────────────────────────────────────
    def _augment(self, X, y):
        """Apply random spatial transforms consistently across all frames."""
        if torch.rand(1).item() > 0.5:
            X = torch.flip(X, dims=[-1])
            y = torch.flip(y, dims=[-1])

        if torch.rand(1).item() > 0.5:
            X = torch.flip(X, dims=[-2])
            y = torch.flip(y, dims=[-2])

        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            X = torch.rot90(X, k, dims=[-2, -1])
            y = torch.rot90(y, k, dims=[-2, -1])

        return X, y
