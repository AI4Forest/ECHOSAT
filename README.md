# ECHOSAT: Estimating Canopy Height Over Space And Time

![Global Map](figures/global_map.png)

<p align="center">
  <img src="https://img.shields.io/badge/Resolution-10m-brightgreen" alt="Resolution">
  <img src="https://img.shields.io/badge/Years-2018--2024-blue" alt="Years">
  <img src="https://img.shields.io/badge/Global-Coverage-orange" alt="Global">
  <img src="https://img.shields.io/badge/Code--License-Apache%202.0-green" alt="Code-License">
  <img src="https://img.shields.io/badge/Data--License-CC--BY%204.0-green" alt="Data-License">
</p>

This repository contains the code for training the ECHOSAT model, which produces the first global, temporally consistent tree height map at 10m resolution spanning multiple years (2018–2024).

📄 [**Paper**](http://arxiv.org/abs/2602.21421)

🌍 [**Google Earth Engine App (Compare to existing models)**](https://ai4forest.projects.earthengine.app/view/echosat-comparison)

🌍 [**Google Earth Engine App (Temporal analysis)**](https://ai4forest.projects.earthengine.app/view/echosat-temporal)

🗺️ **ECHOSAT Asset on GEE**: projects/ai4forest/assets/echosat

💻 [**Direct Dataset Access**](https://echosat.uni-muenster.de)

📋 [**Project Page**](https://janpauls.org/projects/echosat)

---

## Overview

Forest monitoring is critical for climate change mitigation. However, existing global tree height maps provide only static snapshots and do not capture temporal forest dynamics, which are essential for accurate carbon accounting.

**ECHOSAT** addresses this gap by:

- Providing the first high-resolution (10m) spatio-temporal tree height map covering the entire globe across seven years (2018–2024)
- Using a specialized vision transformer model (Swin Video UNet) for pixel-level temporal regression
- Introducing a novel **GrowthLoss** framework that enforces physically realistic forest growth patterns
- Capturing both natural tree growth and abrupt disturbances (fires, deforestation) without post-processing

![Year Comparison](figures/year_comparison.png)

---


## Model Architecture

ECHOSAT uses a **Swin Video UNet** architecture that processes multi-sensor satellite data:

- **Sentinel-2**: Multi-spectral optical imagery (12 bands, monthly composites)
- **Sentinel-1**: SAR imagery (2 polarizations, quarterly composites)
- **ALOS-2**: L-band SAR (HH, HV polarizations, yearly)
- **TanDEM-X**: Digital Elevation Model and Forest/Non-Forest mask

### Reduced Model Configuration

Due to memory constraints (~50GB per sample with original architecture), this repository uses a lighter configuration. Please change the configuration in the `pretraining/lightning_module.py` and `fine-tuning/main.py` files to the original configuration if you want to use the original architecture.

| Parameter | Original | Reduced |
|-----------|----------|---------|
| Embedding dimension | 72 | **24** |
| Encoder depths | [6, 4, 4, 6] | **[2, 2, 2, 2]** |
| Decoder depths | [4, 6, 8, 16] | **[2, 2, 2, 2]** |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ai4forest/ECHOSAT.git
cd ECHOSAT

# For pretraining:
cd pretraining
uv run train_lightning.py

# Fine-tuning is usually done via wandb sweeps.
# For debugging purposes, you can run the fine-tuning script directly (be aware that certain settings, like number of workers are reduced in debug mode):
cd fine-tuning
uv run main.py --debug
```

---

## Training Pipeline

ECHOSAT training consists of two stages: **Pretraining** and **Fine-tuning**.

### Stage 1: Pretraining

Pretraining uses a **Huber loss** to learn basic height regression from sparse GEDI labels.

```bash
cd pretraining

# Single GPU
python train_lightning.py --data_path ../dataset --batch_size 1 --max_epochs 3

# Multi-GPU (DDP)
torchrun --nproc_per_node=2 train_lightning.py --data_path ../dataset --batch_size 1
```

**Key arguments:**
- `--data_path`: Path to the dataset directory
- `--batch_size`: Batch size per GPU (default: 1)
- `--max_epochs`: Maximum training epochs (default: 3)
- `--max_steps`: Maximum training steps (default: 400,000)

Checkpoints are saved to `pretraining/checkpoints/`.

### Stage 2: Fine-tuning

Fine-tuning uses the **GrowthLoss** framework to enforce physically realistic growth patterns:

1. **Regression Loss**: Standard L1/L2 loss on available GEDI labels
2. **Growth Constraint**: Penalizes unrealistic height increases between consecutive years
3. **Disturbance Detection**: Allows abrupt height decreases (fires, deforestation)

```bash
cd fine-tuning

python main.py --debug  # For testing with default config
```

**Configuration** (in `main.py`):
```python
defaults = dict(
    # Data
    dataset='../dataset',
    batch_size=1,
    
    # Architecture (reduced for memory)
    embed_dim=24,                    # Original: 72
    encoder_depths=(2,2,2,2),        # Original: (6,4,4,6)
    decoder_depths=(2,2,2,2),        # Original: (4,6,8,16)
    
    # Optimization
    loss_name='combi_2heads',
    initial_lr=0.0001,
    n_iterations=10,
    
    # Load pretrained checkpoint
    model_checkpoint='../pretraining/checkpoints/last.ckpt',
)
```

Fine-tuning uses a pretrained checkpoint from Stage 1 and applies the GrowthLoss for temporal consistency. Be aware that the console logs missing keys. This is due to the new head being added to the model.

---

## Downloading ECHOSAT from Google Earth Engine

You can download ECHOSAT height maps for a specific region using the Google Earth Engine JavaScript Code Editor.

### JavaScript Code for GEE Code Editor

```javascript
// ============================================
// ECHOSAT Download Script
// ============================================

// 1. Load the ECHOSAT asset
var echosat = ee.ImageCollection('projects/ai4forest/assets/echosat').mosaic();

// 2. Load your region of interest (from uploaded shapefile)
var roi = ee.FeatureCollection('users/YOUR_USERNAME/YOUR_SHAPEFILE');
// Or define a geometry manually:

// 3. Export all years as multi-band image
Export.image.toDrive({
  image: echosat.clip(roi),
  description: 'ECHOSAT_all_years_export',
  folder: 'ECHOSAT_exports',
  region: roi.geometry(),
  scale: 10,
  crs: 'EPSG:4326',
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

print('Export tasks created. Go to the Tasks tab to run them.');
```

### Steps to Download:
1. Open [Google Earth Engine Code Editor](https://code.earthengine.google.com/)
2. Upload your shapefile as an asset (Assets → New → Shape files)
3. Copy the above code and replace the asset ID for your shapefile
4. Click "Run" and go to the "Tasks" tab
5. Click "Run" next to your export task
6. Download the GeoTIFF from your Google Drive

---

## Dataset

### Provided Examples

This repository includes **8 example samples** from the T30TXP Sentinel-2 tile for testing and development. We further provide a jupyter notebook to inspect the dataset:

```
dataset/
├── metadata.csv
└── samples/
    └── T30TXP/
        ├── T30TXP_1.npz
        ├── T30TXP_2.npz
        ├── ...
        └── T30TXP_8.npz
```

### Data Format

Each `.npz` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `sentinel2` | (7, 12, 12, H, W) | Sentinel-2 imagery (years, months, bands, height, width) |
| `sentinel1` | (7, 4, 2, H, W) | Sentinel-1 SAR (years, quarters, polarizations, H, W) |
| `gedi` | (7, 6, H, W) | GEDI height labels (years, tracks, H, W) |
| `gedi_supplemental` | (5, 7, 6, H, W) | Additional GEDI attributes (years, attributes, tracks, H, W)|
| `tandemx_fnf` | (H, W) | TanDEM-X Forest/Non-Forest mask |
| `tandemx_dem` | (H, W) | TanDEM-X Digital Elevation Model |
| `alos` | (7, 3, H, W) | ALOS-2 SAR (years, bands, H, W) |

### Preparing Your Own Dataset

To train on your own data, you need to:

1. **Collect the best (least cloudy) available Sentinel-2 per month** (2018-2024) at 10m resolution
   - Bands: B1-B12 (12 spectral bands)

2. **Collect Sentinel-1 quarterly composites**
   - VH polarizations
   - Radiometrically terrain-corrected
   - ascending and descending orbits separated
   - see `download_s1_from_gee.py` for how to download the data from GEE

3. **Download ALOS-2 yearly mosaics**
   - HH and HV polarizations
   - Available from JAXA (https://www.eorc.jaxa.jp/ALOS/en/dataset/fnf_e.htm)

4. **Obtain TanDEM-X products**
   - DEM (Digital Elevation Model)
   - FNF (Forest/Non-Forest mask)

5. **Download GEDI L2A labels**
   - Filter by quality flags and other filters described in the paper
   - Aggregate to 10m grid (fill pixel at center of the GEDI footprint)

6. **Create the NPZ files** matching the format above

7. **Create `metadata.csv`** with columns:
   ```
   tile,sample_id,fixed_val
   T30TXP,1,0
   T30TXP,2,1
   ...
   ```
   Where `fixed_val=1` indicates validation samples that are plotted in wandb (use 10 at most).

### Inspecting the Dataset

Use the provided Jupyter notebook to explore the data:

```bash
jupyter notebook inspect_dataset.ipynb
```

---

## Project Structure

```
ECHOSAT/
├── pretraining/
│   ├── train_lightning.py      # Main training script (PyTorch Lightning)
│   ├── lightning_module.py     # Model wrapper
│   ├── lightning_dataset.py    # Data loading
│   ├── models/
│   │   └── swin_video_unet.py  # Swin Video UNet architecture
│   └── losses/
│       └── huber_loss.py       # Pretraining loss
│
├── fine-tuning/
│   ├── main.py                 # Main fine-tuning script
│   ├── runner.py               # Training runner
│   ├── datasetClass.py         # Dataset class
│   ├── models/
│   │   └── swin_video_unet.py  # Model architecture
│   └── losses/
│       ├── combi_loss_2heads.py    # Combined loss with disturbance head
│       ├── regression_loss.py       # Basic regression loss
│       └── ...
│
├── dataset/
│   ├── metadata.csv
│   └── samples/
│       └── T30TXP/
│
├── inspect_dataset.ipynb       # Dataset exploration notebook
├── pyproject.toml              # Project dependencies
└── README.md
```

---

## Citation

If you use ECHOSAT in your research, please cite:

```bibtex
@misc{pauls2026echosatestimatingcanopyheight,
      title={ECHOSAT: Estimating Canopy Height Over Space And Time}, 
      author={Jan Pauls and Karsten Schrödter and Sven Ligensa and Martin Schwartz and Berkant Turan and Max Zimmer and Sassan Saatchi and Sebastian Pokutta and Philippe Ciais and Fabian Gieseke},
      year={2026},
      eprint={2602.21421},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.21421}, 
}
```

---

## License

All source code in this repository is licensed under the
Apache License 2.0 — see [LICENSE](./LICENSE).

All data products (tree height maps, derived datasets) are
licensed under Creative Commons Attribution 4.0 International
(CC-BY 4.0) — see [DATA_LICENSE](./DATA_LICENSE).

---

## Acknowledgments

This work was supported via the AI4Forest project, which is funded by the German Federal Ministry of Education and Research (BMBF; grant number 01IS23025A) and the French National Research Agency (ANR). We also acknowledge the computational resources provided by the PALMA II cluster at the University of Münster (subsidized by the DFG; INST 211/667-1) as well as by the Zuse Institute Berlin. We also appreciate the hardware donation of an A100 Tensor Core GPU from Nvidia and thank Google for their compute resources provided (Google Earth Engine). Our work was further supported by the DFG Cluster of Excellence MATH+ (EXC-2046/2, project id 390685689), as well as by the German Federal Ministry of Research, Technology and Space (research campus Modal, fund number 05M14ZAM, 05M20ZBM) and the VDI/VDE Innovation + Technik GmbH (fund number 16IS23025B).

We further thank the following data providers:
- **Sentinel-2** and **Sentinel-1**: Copernicus Programme, European Space Agency
- **GEDI**: NASA
- **ALOS-2**: Japan Aerospace Exploration Agency
- **TanDEM-X**: Deutsches Zentrum für Luft- und Raumfahrt
