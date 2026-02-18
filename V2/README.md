# Spatio-Temporal Deforestation Forecasting (V2)

This module trains a **ConvLSTM** deep learning model to predict future vegetation indices (NDVI, NDMI, etc.) based on satellite imagery.

## 🚀 1. Environment Setup




### Option B: Using venv
```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r requirements.txt
```

---

## 📂 2. Data Setup

1.  **Download your Data**: Get the `.tif` files exported from Google Earth Engine.
2.  **Place them in a folder**: e.g., `data/GeoTIFFs/`
3.  **Update Config**:
    Open `config.yaml` and update the `base_path` to point to your data folder.
    
    ```yaml
    data:
      base_path: "C:/Path/To/Your/Data/  # <--- CHANGE THIS use forward slashes /
    ```

---

## 🧠 3. Train the Model

To start training, simply run:

```bash
python train_v2.py
```

### Configuration
You can modify `config.yaml` to change:
*   `batch_size`: Decrease to 4 or 2 if you run out of GPU memory.
*   `epochs`: Increase for longer training.
*   `channels`: Ensure this matches your Tiff bands (Default: 4).

### Monitoring
*   Logs will be printed to the console.
*   Checkpoints are saved in specified `checkpoints/` folder.
*   **Early Stopping** is enabled (stops if no improvement for 10 epochs).

---

## 📊 4. Training on Cloud (Colab/Kaggle) ??

If using Google Colab:
1.  Upload this `V2` folder to Google Drive.
2.  Mount Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    %cd /content/drive/MyDrive/Path/To/V2
    ```
3.  Install dependencies:
    ```python
    !pip install rasterio
    ```
4.  Run training:
    ```python
    !python train_v2.py
    ```

