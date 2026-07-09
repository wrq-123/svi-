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


4. How to run
Bash

python ablation_balanced500.py
By default, input CSVs are read from the path defined in CFG.data_dir, and all outputs are written to Paper_Results_Models_Balanced500/. Edit CFG.data_dir / CFG.out_dir in the script to match your local paths.

5. Reproducibility settings
Bootstrap repetitions: 30
Training subsampling ratio: 0.9
Validation bootstrap: stratified by class (keeps 1:1 balance)
Base random seed: 42 (per-repetition seed = 42 + i)
Software versions are printed at runtime (see console output).
6. Google Earth Engine workflow
A step-by-step description of the Sentinel-1/2 preprocessing and feature extraction workflow (data collections, filtering, masking, compositing, smoothing, feature extraction, and export settings) is provided in the Supplementary Materials of the paper.

7. Outputs
Running the script generates (in the output folder):

Table1_Feature_XGB.csv, Table2_Feature_RF.csv, Table3_Model_Comparison.csv
Paired_ttests_OpticalOnly_vs_OpticalPlusSVI.csv
All_bootstrap_results.csv
Fig1_Ablation_and_Models_v3.png/pdf
Fig2_Confusion_Matrix_v3.png/pdf
