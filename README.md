# GeoAI Framework for Large-Scale Land Subsidence Forecasting with InSAR and Environmental Variables

[![Status](https://img.shields.io/badge/Status-Submitted-blue)](https://www.springer.com/journal/11069)
[![Journal](https://img.shields.io/badge/Journal-Natural%20Hazards-orange)](https://www.springer.com/journal/11069)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Abstract
Land subsidence is a critical geohazard driven by unsustainable groundwater extraction and complex hydro-climatic factors. This repository provides a state-of-the-art **GeoAI framework** for forecasting subsidence by integrating multi-source geospatial data.

The study investigates **8 strategic regions in Iran** (Isfahan, Jiroft, Lake Urmia/Tabriz, Marvdasht, Nishapur, Qazvin-Alborz-Tehran, Rafsanjan, and Semnan). Our approach leverages **Sentinel-1 InSAR** time-series fused with **ERA5-Land** monthly climate reanalysis and **ISRIC SoilGrids** physical properties. The framework employs a **Nested Leave-One-City-Out Cross-Validation** (CV) strategy to ensure robust geographical generalizability and uses **SHAP** (SHapley Additive exPlanations) for model interpretability.

---

## 👥 Authors

* **Amirreza Shahmiri**
    * *Department of Civil and Architectural Engineering, Sultan Qaboos University, Muscat, Oman*
    * [![ORCID](https://img.shields.io/badge/ORCID-0009--0002--0746--3887-green)](https://orcid.org/0009-0002-0746-3887)
* **Masoud Ebrahimi Derakhshan**
    * *School of Civil Engineering, Iran University of Science and Technology (IUST), Tehran, Iran*
    * [![ORCID](https://img.shields.io/badge/ORCID-0009--0009--3453--8304-green)](https://orcid.org/0009-0009-3453-8304)
* **Seyed Mostafa Siadatmousavi** (Corresponding Author)
    * *School of Civil Engineering, Iran University of Science and Technology (IUST), Tehran, Iran*
    * [![ORCID](https://img.shields.io/badge/ORCID-0000--0002--0068--7506-green)](https://orcid.org/0000-0002-0068-7506)

---

## 🛠 Methodology & Features
- **Multi-Source Data Fusion:** Spatial alignment of SAR (InSAR), Climate (ERA5), and Soil properties (SoilGrids) into a unified 3D data cube.
- **Nested Feature Selection:** Defensible Recursive Feature Elimination (RFECV) with XGBoost to identify hydro-climatic drivers without data leakage.
- **Advanced Forecasting:** Comparative analysis of **ElasticNet**, **XGBoost**, **LightGBM**, and **BiLSTM** architectures.
- **Explainable AI (XAI):** Global and local feature impact analysis using SHAP values.
- **Uncertainty Analysis:** Implementation of Confidence Intervals (CI) for pixel-wise time-series predictions.

---

## 📂 Repository Contents

### 1. Data Extraction (Google Earth Engine)
* `data_extraction_era5.js`: Monthly ERA5-Land variables export.
* `data_extraction_soilgrids.js`: Weighted mean calculation for soil properties (0-100 cm).

### 2. Preprocessing
* `data_fusion_alignment.py`: Aligns HDF5 InSAR data with GeoTIFFs into `.npz` format and handles temporal gaps.

### 3. Feature Selection
* `feature_selection_nested_cv.py`: Nested Leave-One-City-Out RFECV to optimize lag-depth and feature sets.

### 4. Forecasting Models
* `subsidence_forecasting_pipeline.py`: Comprehensive script for training, SHAP analysis, and plotting results for all architectures.

---

## 📊 Performance Summary (Test Set Results)

The following results represent model performance under the **Combined Cumulative** scenario on **Unseen Holdout Regions**:

| Model | $R^2$ Score | RMSE (mm) | MAE (mm) | Optimal Lags |
| :--- | :---: | :---: | :---: | :---: |
| **ElasticNet** (Winner) | **0.9943** | **6.38** | **4.07** | 11 months |
| **BiLSTM** | 0.9883 | 8.79 | 5.83 | 2 months |
| **LightGBM** | 0.9882 | 9.18 | 5.96 | 11 months |
| **XGBoost** | 0.9879 | 9.31 | 6.04 | 11 months |

### 🌍 Regional Stability (ElasticNet Snapshot)
| City Name | $R^2$ Score | RMSE (mm) | MAE (mm) |
| :--- | :---: | :---: | :---: |
| Qazvin-Alborz-Tehran | 0.9995 | 3.81 | 2.84 |
| Rafsanjan | 0.9954 | 2.68 | 2.06 |
| Nishapur | 0.9875 | 3.61 | 3.04 |
| Isfahan | 0.9710 | 2.85 | 2.22 |
| Lake Urmia - Tabriz | 0.8653 | 9.49 | 7.02 |

---

## 🚀 Usage

1. **Extraction:** Run `.js` scripts in GEE to download environment data.
2. **Alignment:** Use `data_fusion_alignment.py` to create the unified dataset.
3. **Selection:** Run `feature_selection_nested_cv.py` to identify optimal predictors.
4. **Forecasting:** Execute `subsidence_forecasting_pipeline.py` to train models and generate interpretability plots.

---

## 📜 Citation
If you use this code or dataset, please cite our paper:

```bibtex
@article{Shahmiri2026Subsidence,
  title={Large-Scale Land Subsidence Forecasting with InSAR and Environmental Variables via Machine Learning Approaches},
  author={Shahmiri, Amirreza and Ebrahimi Derakhshan, Masoud and Siadatmousavi, Seyed Mostafa},
  journal={Natural Hazards},
  year={2026},
  note={Submitted}
}
