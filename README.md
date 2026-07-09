# SVI*: Learning-based SAR Structural Index for Alpine Shrub Encroachment Mapping

This repository provides the code to reproduce the feature-ablation and
classifier-comparison experiments in the manuscript:

> *Fine-scale annual mapping and dynamic monitoring of shrub encroachment in
> alpine grasslands by integrating Sentinel-2 phenological features with
> Sentinel-1 structural information* (JAG-D-26-02007).

## 1. What this code does
`ablation_balanced500.py` reproduces:
- **Feature ablation** (Optical Only / Optical + Env / Optical + SVI* / Full)
  using both XGBoost and Random Forest;
- **Classifier comparison** (RF, GBDT, XGBoost, LightGBM) under the Full feature set;
- **Paired t-tests** across bootstrap repetitions;
- **Confusion matrix** for the Full + XGBoost model on the 500-point balanced
  validation set;
- Output tables (CSV) and figures (PNG/PDF) used in the paper.

## 2. Input data
Two CSV files are required:
- `Training_Set.csv` (n = 3204)
- `Validation_Set.csv` (n = 500, class-balanced 1:1)

### Required columns
| Column | Description |
|--------|-------------|
| `label` | Class label: `0` = grassland, `1` = shrubland |
| `SVI_star` | Learning-based SAR structural index (required) |
| `elevation`, `slope`, `aspect`, `twi` | Environmental/topographic features |
| Optical phenological features | Column names containing any of: `ndvi`, `evi`, `gndvi`, `msavi`, `ndgi`, `ndmi`, `ndpi`, `ndsvi`, `ndti` |

> Note: The sample dataset is available from the corresponding author upon
> reasonable request. A small synthetic example (`example_data/`) is provided
> only to demonstrate the required file format.

## 3. Environment
Install dependencies:
```bash
pip install -r requirements.txt
