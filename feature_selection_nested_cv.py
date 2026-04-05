import os
import json
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.feature_selection import RFECV
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")


# =============================================================================
# SETTINGS
# =============================================================================
BASE_DIR = r"C:\Users\DFMRendering\Desktop\subsidence\Revise\Data"

FILE_PATHS = {
    "Isfahan": os.path.join(BASE_DIR, "Isfahan", "Merged_Dataset_3D.npz"),
    "Jiroft": os.path.join(BASE_DIR, "Jiroft", "Merged_Dataset_3D.npz"),
    "Lake Urmia  Tabriz": os.path.join(BASE_DIR, "Lake Urmia  Tabriz", "Merged_Dataset_3D.npz"),
    "Marvdasht": os.path.join(BASE_DIR, "Marvdasht", "Merged_Dataset_3D.npz"),
    "Nishapur": os.path.join(BASE_DIR, "Nishapur", "Merged_Dataset_3D.npz"),
    "Qazvin-Alborz-Tehran": os.path.join(BASE_DIR, "Qazvin-Alborz-Tehran", "Merged_Dataset_3D.npz"),
    "Rafsanjan": os.path.join(BASE_DIR, "Rafsanjan", "Merged_Dataset_3D.npz"),
    "Semnan": os.path.join(BASE_DIR, "Semnan", "Merged_Dataset_3D.npz"),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "Nested_Defensible_Feature_Selection_XGBoost_NoHistory")
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
FIG_DPI = 600
PAST_STEPS_GRID = [1, 2, 3, 4, 5, 6]
N_PERMUTATION_REPEATS = 30
RFECV_STEP = 1
MIN_FEATURES_TO_SELECT = 1

# =============================================================================
# FEATURE NAME DEFINITIONS
# =============================================================================
PRETTY_FEATURE_ORDER = [
    "Cumulative InSAR Displacement",
    "InSAR Displacement Difference",
    "Average Coherence",
    "Height",
    "Geopotential U Component",
    "Mask",
    "Standard Deviation of Velocity",
    "Total Precipitation",
    "Total Evaporation",
    "Runoff Volume",
    "Volumetric Soil Water Layer 1",
    "Volumetric Soil Water Layer 2",
    "Volumetric Soil Water Layer 3",
    "Volumetric Soil Water Layer 4",
    "Temperature at 2 meters above the surface",
    "Skin Temperature",
    "Soil Temperature at Level 1",
    "Soil Temperature at Level 4",
    "Surface Net Solar Radiation",
    "Surface Sensible Heat Flux",
    "Surface Pressure",
    "U Component of Wind at 10 meters",
    "V Component of Wind at 10 meters",
    "Dewpoint Temperature at 2 meters",
    "Leaf Area Index for High Vegetation",
    "Leaf Area Index for Low Vegetation",
    "Bulk Density of Soil (g/cm³)",
    "Clay Percentage in Soil",
    "Soil pH in Water",
    "Sand Percentage in Soil",
    "Silt Percentage in Soil",
    "Soil Organic Carbon Content (g/kg)",
]

RAW_TO_PRETTY = {
    "insar_cum": "Cumulative InSAR Displacement",
    "insar_diff": "InSAR Displacement Difference",
    "coh_avg": "Average Coherence",
    "hgt": "Height",
    "U.geo": "Geopotential U Component",
    "mask": "Mask",
    "vstd": "Standard Deviation of Velocity",
    "total_precipitation_sum": "Total Precipitation",
    "total_evaporation_sum": "Total Evaporation",
    "runoff_sum": "Runoff Volume",
    "volumetric_soil_water_layer_1": "Volumetric Soil Water Layer 1",
    "volumetric_soil_water_layer_2": "Volumetric Soil Water Layer 2",
    "volumetric_soil_water_layer_3": "Volumetric Soil Water Layer 3",
    "volumetric_soil_water_layer_4": "Volumetric Soil Water Layer 4",
    "temperature_2m": "Temperature at 2 meters above the surface",
    "skin_temperature": "Skin Temperature",
    "soil_temperature_level_1": "Soil Temperature at Level 1",
    "soil_temperature_level_4": "Soil Temperature at Level 4",
    "surface_net_solar_radiation_sum": "Surface Net Solar Radiation",
    "surface_sensible_heat_flux_sum": "Surface Sensible Heat Flux",
    "surface_pressure": "Surface Pressure",
    "u_component_of_wind_10m": "U Component of Wind at 10 meters",
    "v_component_of_wind_10m": "V Component of Wind at 10 meters",
    "dewpoint_temperature_2m": "Dewpoint Temperature at 2 meters",
    "leaf_area_index_high_vegetation": "Leaf Area Index for High Vegetation",
    "leaf_area_index_low_vegetation": "Leaf Area Index for Low Vegetation",
    "bdod_gcm3": "Bulk Density of Soil (g/cm³)",
    "clay_pct": "Clay Percentage in Soil",
    "phh2o_pH": "Soil pH in Water",
    "sand_pct": "Sand Percentage in Soil",
    "silt_pct": "Silt Percentage in Soil",
    "soc_dgkg": "Soil Organic Carbon Content (g/kg)",
}
PRETTY_TO_RAW = {v: k for k, v in RAW_TO_PRETTY.items()}

# =============================================================================
# CANDIDATE FEATURE SET
# =============================================================================
DYNAMIC_CANDIDATES = [
    "total_precipitation_sum",
    "total_evaporation_sum",
    "runoff_sum",
    "volumetric_soil_water_layer_1",
    "volumetric_soil_water_layer_2",
    "volumetric_soil_water_layer_3",
    "volumetric_soil_water_layer_4",
    "temperature_2m",
    "skin_temperature",
    "soil_temperature_level_1",
    "soil_temperature_level_4",
    "surface_net_solar_radiation_sum",
    "surface_sensible_heat_flux_sum",
    "surface_pressure",
    "u_component_of_wind_10m",
    "v_component_of_wind_10m",
    "dewpoint_temperature_2m",
    "leaf_area_index_high_vegetation",
    "leaf_area_index_low_vegetation",
]

STATIC_CANDIDATES = [
    "hgt",
    "bdod_gcm3",
    "clay_pct",
    "phh2o_pH",
    "sand_pct",
    "silt_pct",
    "soc_dgkg",
]

EXCLUDED_FEATURES = [
    "insar_cum",
    "insar_diff",
    "mask",
    "U.geo",
    "vstd",
    "coh_avg",
]

TARGET_SOURCE_FEATURE = "insar_cum"


# =============================================================================
# MODEL
# =============================================================================
def make_xgb_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        importance_type="gain",
    )


# =============================================================================
# DATA STRUCTURES
# =============================================================================
@dataclass
class OuterFoldResult:
    heldout_city: str
    chosen_past_steps: int
    inner_cv_rmse_for_chosen_steps: float
    n_selected_lagged_features: int
    selected_lagged_features: List[str]
    selected_base_features: List[str]
    test_r2: float
    test_rmse: float
    test_mae: float


# =============================================================================
# UTILITIES
# =============================================================================
def safe_load_city_npz(path: str) -> Tuple[np.ndarray, List[str]]:
    with np.load(path, allow_pickle=True) as loader:
        raw_data = loader["data"]
        features = [str(f) for f in loader["features"]]
    if raw_data.ndim != 3:
        raise ValueError(f"Expected 3D array in {path}, got shape {raw_data.shape}")
    return raw_data, features


def validate_feature_availability(city_feature_lists: List[List[str]]) -> Tuple[List[str], List[str], List[str]]:
    common = set(city_feature_lists[0])
    for feats in city_feature_lists[1:]:
        common &= set(feats)

    dynamic_final = [f for f in DYNAMIC_CANDIDATES if f in common and f not in EXCLUDED_FEATURES]
    static_final = [f for f in STATIC_CANDIDATES if f in common and f not in EXCLUDED_FEATURES]

    return dynamic_final, static_final, sorted(list(common))


def raw_to_pretty(raw_name: str) -> str:
    return RAW_TO_PRETTY.get(raw_name, raw_name)


def lagged_raw_to_pretty(lagged_name: str) -> str:
    if "__lag" not in lagged_name:
        return raw_to_pretty(lagged_name)
    raw_name, lag_token = lagged_name.split("__lag")
    lag_idx = int(lag_token)
    return f"{raw_to_pretty(raw_name)} (t-{lag_idx})"


def extract_base_feature_name(lagged_name: str) -> str:
    if "__lag" in lagged_name:
        return lagged_name.split("__lag")[0]
    return lagged_name


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def evaluate_logo_mean_baseline(y: pd.Series, groups: pd.Series) -> Dict[str, float]:
    logo = LeaveOneGroupOut()
    rmse_list, r2_list, mae_list = [], [], []

    for train_idx, test_idx in logo.split(np.zeros(len(y)), y, groups):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        y_pred = np.full(len(y_test), float(y_train.mean()), dtype=float)

        fold_metrics = metric_dict(y_test.to_numpy(), y_pred)
        rmse_list.append(fold_metrics["rmse"])
        r2_list.append(fold_metrics["r2"])
        mae_list.append(fold_metrics["mae"])

    return {
        "RMSE_Mean": float(np.mean(rmse_list)),
        "RMSE_Std": float(np.std(rmse_list)),
        "R2_Mean": float(np.mean(r2_list)),
        "MAE_Mean": float(np.mean(mae_list)),
    }


# =============================================================================
# DATA BUILDING
# =============================================================================
def load_all_city_data() -> Tuple[Dict[str, Tuple[np.ndarray, List[str]]], List[str], List[str], List[str]]:
    city_cache = {}
    all_feature_lists = []

    for city, path in FILE_PATHS.items():
        if not os.path.exists(path):
            continue
        raw_data, features = safe_load_city_npz(path)
        city_cache[city] = (raw_data, features)
        all_feature_lists.append(features)

    if not city_cache:
        raise RuntimeError("No valid NPZ files were found.")

    dynamic_features, static_features, common_features = validate_feature_availability(all_feature_lists)

    if TARGET_SOURCE_FEATURE not in common_features:
        raise ValueError(f"Required target source feature '{TARGET_SOURCE_FEATURE}' was not found in all cities.")

    return city_cache, dynamic_features, static_features, common_features


def build_supervised_lagged_dataset(
    city_cache: Dict[str, Tuple[np.ndarray, List[str]]],
    dynamic_features: List[str],
    static_features: List[str],
    past_steps: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Target:
        next-step differential subsidence
        Target_Next_Step = diff(insar_cum).shift(-1)

    Predictors:
        dynamic variables expanded into lagged features
        static variables included once
        no historical subsidence predictors are used
    """
    rows = []
    city_summaries = []

    for city, (raw_data, features) in city_cache.items():
        n_timesteps, n_pixels, _ = raw_data.shape
        pixel_frames = []
        valid_samples_city = 0

        for pixel_idx in range(n_pixels):
            pixel_series = raw_data[:, pixel_idx, :]
            df_p = pd.DataFrame(pixel_series, columns=features)

            # Build target exclusively from cumulative displacement
            df_p["Target_Source_Diff"] = df_p[TARGET_SOURCE_FEATURE].diff()
            df_p["Target_Next_Step"] = df_p["Target_Source_Diff"].shift(-1)

            lagged_block = pd.DataFrame(index=df_p.index)

            for feat in dynamic_features:
                for lag in range(past_steps):
                    lagged_block[f"{feat}__lag{lag}"] = df_p[feat].shift(lag)

            for sf in static_features:
                lagged_block[sf] = df_p[sf].iloc[0]

            lagged_block["City"] = city
            lagged_block["Pixel_ID"] = f"{city}_P{pixel_idx}"
            lagged_block["Time_Index"] = np.arange(len(df_p))
            lagged_block["Target_Next_Step"] = df_p["Target_Next_Step"]

            lagged_block = lagged_block.dropna().reset_index(drop=True)

            if not lagged_block.empty:
                valid_samples_city += len(lagged_block)
                pixel_frames.append(lagged_block)

        if pixel_frames:
            city_df = pd.concat(pixel_frames, ignore_index=True)
            rows.append(city_df)

        city_summaries.append({
            "City": city,
            "Original_TimeSteps": n_timesteps,
            "Pixels": n_pixels,
            "Valid_Samples_After_Lagging": valid_samples_city,
            "Past_Steps": past_steps,
        })

    if not rows:
        raise RuntimeError("No valid supervised rows were produced after lagging and NA removal.")

    master_df = pd.concat(rows, ignore_index=True)
    city_summary_df = pd.DataFrame(city_summaries)
    return master_df, city_summary_df


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================
def build_correlation_dataset(
    city_cache: Dict[str, Tuple[np.ndarray, List[str]]],
    common_features: List[str],
) -> pd.DataFrame:
    stacked = []

    available_raw_features = [PRETTY_TO_RAW[p] for p in PRETTY_FEATURE_ORDER if PRETTY_TO_RAW.get(p) in common_features]

    for city, (raw_data, features) in city_cache.items():
        feat_to_idx = {f: i for i, f in enumerate(features)}
        raw_subset = raw_data[:, :, [feat_to_idx[f] for f in available_raw_features]]
        n_t, n_p, n_f = raw_subset.shape
        flat = raw_subset.reshape(n_t * n_p, n_f)
        df_city = pd.DataFrame(flat, columns=available_raw_features)
        df_city["City"] = city
        stacked.append(df_city)

    df = pd.concat(stacked, ignore_index=True)
    return df


def save_correlation_outputs(
    corr_input_df: pd.DataFrame,
    output_dir: str,
) -> None:
    corr_feature_cols = [c for c in corr_input_df.columns if c != "City"]

    # Remove constant columns for valid correlation computation
    stds = corr_input_df[corr_feature_cols].std(numeric_only=True)
    non_constant_cols = stds[stds > 0].index.tolist()
    constant_cols = [c for c in corr_feature_cols if c not in non_constant_cols]

    if not non_constant_cols:
        raise RuntimeError("All candidate columns are constant. Correlation matrices cannot be computed.")

    corr_df = corr_input_df[non_constant_cols].copy()
    pretty_order_available = [p for p in PRETTY_FEATURE_ORDER if PRETTY_TO_RAW.get(p) in non_constant_cols]
    ordered_raw = [PRETTY_TO_RAW[p] for p in pretty_order_available]

    corr_df = corr_df[ordered_raw]
    corr_df.columns = [RAW_TO_PRETTY[c] for c in corr_df.columns]

    pearson_corr = corr_df.corr(method="pearson")
    spearman_corr = corr_df.corr(method="spearman")

    pearson_corr.to_csv(os.path.join(output_dir, "correlation_matrix_pearson.csv"), encoding="utf-8-sig")
    spearman_corr.to_csv(os.path.join(output_dir, "correlation_matrix_spearman.csv"), encoding="utf-8-sig")

    def plot_corr_heatmap(corr_matrix: pd.DataFrame, title: str, save_path: str) -> None:
        plt.figure(figsize=(20, 17))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.75, "label": "Correlation"},
        )
        plt.title(title, fontsize=20, pad=20, fontweight="bold")
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()

    plot_corr_heatmap(
        pearson_corr,
        "Pearson Correlation Matrix of Available Original Predictors",
        os.path.join(output_dir, "correlation_matrix_pearson.png"),
    )
    plot_corr_heatmap(
        spearman_corr,
        "Spearman Correlation Matrix of Available Original Predictors",
        os.path.join(output_dir, "correlation_matrix_spearman.png"),
    )

    def strongest_pairs(corr_matrix: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        rows = []
        cols = corr_matrix.columns.tolist()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                rows.append({
                    "Feature_1": cols[i],
                    "Feature_2": cols[j],
                    "Correlation": float(corr_matrix.iloc[i, j]),
                    "Abs_Correlation": float(abs(corr_matrix.iloc[i, j])),
                })
        out = pd.DataFrame(rows).sort_values("Abs_Correlation", ascending=False).reset_index(drop=True)
        return out.head(top_n)

    top_pearson = strongest_pairs(pearson_corr, top_n=50)
    top_spearman = strongest_pairs(spearman_corr, top_n=50)

    top_pearson.to_csv(os.path.join(output_dir, "top_absolute_pearson_pairs.csv"), index=False, encoding="utf-8-sig")
    top_spearman.to_csv(os.path.join(output_dir, "top_absolute_spearman_pairs.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(output_dir, "correlation_report.txt"), "w", encoding="utf-8") as f:
        f.write("CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 100 + "\n\n")
        f.write("Input level: original feature space (non-lagged), stacked across all monthly samples and pixels.\n")
        f.write("Purpose: descriptive multicollinearity assessment before nested feature selection.\n\n")

        f.write("Columns excluded because of zero variance:\n")
        if constant_cols:
            for c in constant_cols:
                f.write(f"- {RAW_TO_PRETTY.get(c, c)}\n")
        else:
            f.write("- None\n")
        f.write("\n")

        f.write("Top 50 absolute Pearson correlation pairs:\n")
        f.write(top_pearson.to_string(index=False))
        f.write("\n\n")

        f.write("Top 50 absolute Spearman correlation pairs:\n")
        f.write(top_spearman.to_string(index=False))
        f.write("\n")


# =============================================================================
# FEATURE SELECTION CORE
# =============================================================================
def fit_inner_rfecv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups_train: pd.Series,
) -> RFECV:
    selector = RFECV(
        estimator=make_xgb_model(),
        step=RFECV_STEP,
        min_features_to_select=MIN_FEATURES_TO_SELECT,
        cv=LeaveOneGroupOut(),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    selector.fit(X_train, y_train, groups=groups_train)
    return selector


def get_rfecv_selected_score(selector: RFECV) -> float:
    if not hasattr(selector, "cv_results_"):
        raise RuntimeError("RFECV object does not contain cv_results_.")
    idx = int(selector.n_features_) - 1
    return float(-selector.cv_results_["mean_test_score"][idx])


def evaluate_candidate_lag_in_inner_cv(
    master_df: pd.DataFrame,
    outer_train_cities: List[str],
) -> Tuple[RFECV, float, List[str]]:
    train_df = master_df[master_df["City"].isin(outer_train_cities)].reset_index(drop=True)
    feature_cols = [c for c in train_df.columns if c not in ["Target_Next_Step", "City", "Pixel_ID", "Time_Index"]]

    X_train = train_df[feature_cols]
    y_train = train_df["Target_Next_Step"]
    groups_train = train_df["City"]

    selector = fit_inner_rfecv(X_train, y_train, groups_train)
    selected_features = X_train.columns[selector.support_].tolist()
    inner_cv_rmse = get_rfecv_selected_score(selector)

    return selector, inner_cv_rmse, selected_features


def fit_and_evaluate_outer_fold(
    master_df: pd.DataFrame,
    heldout_city: str,
    selected_features: List[str],
) -> Tuple[XGBRegressor, Dict[str, float], pd.DataFrame]:
    train_df = master_df[master_df["City"] != heldout_city].reset_index(drop=True)
    test_df = master_df[master_df["City"] == heldout_city].reset_index(drop=True)

    X_train = train_df[selected_features]
    y_train = train_df["Target_Next_Step"]
    X_test = test_df[selected_features]
    y_test = test_df["Target_Next_Step"]

    model = make_xgb_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = metric_dict(y_test.to_numpy(), y_pred)

    fold_pred_df = test_df[["City", "Pixel_ID", "Time_Index"]].copy()
    fold_pred_df["Actual"] = y_test.to_numpy()
    fold_pred_df["Predicted"] = y_pred
    fold_pred_df["Residual"] = fold_pred_df["Actual"] - fold_pred_df["Predicted"]

    return model, metrics, fold_pred_df


def compute_outer_fold_importance_tables(
    model: XGBRegressor,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    heldout_city: str,
    fold_id: int,
    selected_features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gain_importance = pd.DataFrame({
        "Fold": fold_id,
        "Heldout_City": heldout_city,
        "Lagged_Feature": selected_features,
        "Gain_Importance": model.feature_importances_,
    })

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    perm_importance = pd.DataFrame({
        "Fold": fold_id,
        "Heldout_City": heldout_city,
        "Lagged_Feature": selected_features,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
    })

    return gain_importance, perm_importance


def summarize_final_lagged_and_base_features(
    selected_lagged_features: List[str],
    gain_df: pd.DataFrame,
    perm_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lagged_summary = pd.DataFrame({"Lagged_Feature": selected_lagged_features})
    lagged_summary["Pretty_Lagged_Feature"] = lagged_summary["Lagged_Feature"].map(lagged_raw_to_pretty)
    lagged_summary["Base_Raw_Feature"] = lagged_summary["Lagged_Feature"].map(extract_base_feature_name)
    lagged_summary["Base_Pretty_Feature"] = lagged_summary["Base_Raw_Feature"].map(raw_to_pretty)

    if not gain_df.empty:
        gain_agg = gain_df.groupby("Lagged_Feature", as_index=False)["Gain_Importance"].mean()
        lagged_summary = lagged_summary.merge(gain_agg, on="Lagged_Feature", how="left")
    else:
        lagged_summary["Gain_Importance"] = np.nan

    if not perm_df.empty:
        perm_agg = perm_df.groupby("Lagged_Feature", as_index=False)["Permutation_Importance_Mean"].mean()
        lagged_summary = lagged_summary.merge(perm_agg, on="Lagged_Feature", how="left")
    else:
        lagged_summary["Permutation_Importance_Mean"] = np.nan

    lagged_summary = lagged_summary.sort_values(
        ["Permutation_Importance_Mean", "Gain_Importance", "Pretty_Lagged_Feature"],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    base_summary = (
        lagged_summary
        .groupby(["Base_Raw_Feature", "Base_Pretty_Feature"], as_index=False)
        .agg(
            Number_of_Selected_Lagged_Terms=("Lagged_Feature", "count"),
            Mean_Gain_Importance=("Gain_Importance", "mean"),
            Mean_Permutation_Importance=("Permutation_Importance_Mean", "mean"),
        )
        .sort_values(
            ["Mean_Permutation_Importance", "Mean_Gain_Importance", "Base_Pretty_Feature"],
            ascending=[False, False, True]
        )
        .reset_index(drop=True)
    )

    return lagged_summary, base_summary


# =============================================================================
# NESTED CV PIPELINE
# =============================================================================
def run_nested_feature_selection_pipeline(
    city_cache: Dict[str, Tuple[np.ndarray, List[str]]],
    dynamic_features: List[str],
    static_features: List[str],
) -> Dict[str, object]:
    logo = LeaveOneGroupOut()
    outer_results: List[OuterFoldResult] = []
    outer_predictions = []
    outer_gain_records = []
    outer_perm_records = []
    lag_choice_records = []

    city_names = list(city_cache.keys())

    for fold_id, heldout_city in enumerate(city_names):
        outer_train_cities = [c for c in city_names if c != heldout_city]

        best_selector = None
        best_master_df = None
        best_inner_rmse = np.inf
        best_selected_features = None
        best_past_steps = None

        for past_steps in PAST_STEPS_GRID:
            master_df, _ = build_supervised_lagged_dataset(
                city_cache=city_cache,
                dynamic_features=dynamic_features,
                static_features=static_features,
                past_steps=past_steps,
            )

            selector, inner_cv_rmse, selected_features = evaluate_candidate_lag_in_inner_cv(
                master_df=master_df,
                outer_train_cities=outer_train_cities,
            )

            lag_choice_records.append({
                "Fold": fold_id,
                "Heldout_City": heldout_city,
                "Candidate_Past_Steps": past_steps,
                "Inner_CV_RMSE": inner_cv_rmse,
                "Selected_Lagged_Features_Count": len(selected_features),
            })

            if inner_cv_rmse < best_inner_rmse:
                best_inner_rmse = inner_cv_rmse
                best_selector = selector
                best_master_df = master_df
                best_selected_features = selected_features
                best_past_steps = past_steps

        # Outer evaluation with chosen lag and chosen features
        model, metrics, fold_pred_df = fit_and_evaluate_outer_fold(
            master_df=best_master_df,
            heldout_city=heldout_city,
            selected_features=best_selected_features,
        )

        test_df = best_master_df[best_master_df["City"] == heldout_city].reset_index(drop=True)
        X_test = test_df[best_selected_features]
        y_test = test_df["Target_Next_Step"]

        fold_gain_df, fold_perm_df = compute_outer_fold_importance_tables(
            model=model,
            X_test=X_test,
            y_test=y_test,
            heldout_city=heldout_city,
            fold_id=fold_id,
            selected_features=best_selected_features,
        )

        outer_gain_records.append(fold_gain_df)
        outer_perm_records.append(fold_perm_df)
        outer_predictions.append(fold_pred_df)

        selected_base = sorted({raw_to_pretty(extract_base_feature_name(f)) for f in best_selected_features})

        outer_results.append(
            OuterFoldResult(
                heldout_city=heldout_city,
                chosen_past_steps=best_past_steps,
                inner_cv_rmse_for_chosen_steps=best_inner_rmse,
                n_selected_lagged_features=len(best_selected_features),
                selected_lagged_features=best_selected_features,
                selected_base_features=selected_base,
                test_r2=metrics["r2"],
                test_rmse=metrics["rmse"],
                test_mae=metrics["mae"],
            )
        )

    outer_summary = pd.DataFrame([{
        "Heldout_City": r.heldout_city,
        "Chosen_Past_Steps": r.chosen_past_steps,
        "Inner_CV_RMSE_for_Chosen_Steps": r.inner_cv_rmse_for_chosen_steps,
        "Selected_Lagged_Features_Count": r.n_selected_lagged_features,
        "Test_R2": r.test_r2,
        "Test_RMSE": r.test_rmse,
        "Test_MAE": r.test_mae,
        "Selected_Base_Features": "; ".join(r.selected_base_features),
    } for r in outer_results])

    outer_performance = {
        "Mean_Test_R2": float(outer_summary["Test_R2"].mean()),
        "Std_Test_R2": float(outer_summary["Test_R2"].std()),
        "Mean_Test_RMSE": float(outer_summary["Test_RMSE"].mean()),
        "Std_Test_RMSE": float(outer_summary["Test_RMSE"].std()),
        "Mean_Test_MAE": float(outer_summary["Test_MAE"].mean()),
        "Std_Test_MAE": float(outer_summary["Test_MAE"].std()),
    }

    lag_choice_df = pd.DataFrame(lag_choice_records)
    all_outer_predictions_df = pd.concat(outer_predictions, ignore_index=True)
    all_outer_gain_df = pd.concat(outer_gain_records, ignore_index=True)
    all_outer_perm_df = pd.concat(outer_perm_records, ignore_index=True)

    return {
        "outer_summary": outer_summary,
        "outer_performance": outer_performance,
        "lag_choice_df": lag_choice_df,
        "outer_predictions_df": all_outer_predictions_df,
        "outer_gain_df": all_outer_gain_df,
        "outer_perm_df": all_outer_perm_df,
    }


# =============================================================================
# FINAL FULL-DATA FIT AFTER NESTED SELECTION
# =============================================================================
def determine_final_lag_depth(lag_choice_df: pd.DataFrame, outer_summary: pd.DataFrame) -> int:
    counts = lag_choice_df.groupby("Candidate_Past_Steps").apply(
        lambda x: int((x["Inner_CV_RMSE"] == x.groupby("Heldout_City")["Inner_CV_RMSE"].transform("min")).sum())
    )
    counts = counts.sort_values(ascending=False)

    top_count = counts.iloc[0]
    candidate_lags = counts[counts == top_count].index.tolist()

    if len(candidate_lags) == 1:
        return int(candidate_lags[0])

    tied = outer_summary[outer_summary["Chosen_Past_Steps"].isin(candidate_lags)]
    lag_rmse = tied.groupby("Chosen_Past_Steps")["Test_RMSE"].mean().sort_values()
    return int(lag_rmse.index[0])


def run_final_full_data_selection(
    city_cache: Dict[str, Tuple[np.ndarray, List[str]]],
    dynamic_features: List[str],
    static_features: List[str],
    final_past_steps: int,
) -> Dict[str, object]:
    master_df, city_summary = build_supervised_lagged_dataset(
        city_cache=city_cache,
        dynamic_features=dynamic_features,
        static_features=static_features,
        past_steps=final_past_steps,
    )

    feature_cols = [c for c in master_df.columns if c not in ["Target_Next_Step", "City", "Pixel_ID", "Time_Index"]]
    X = master_df[feature_cols]
    y = master_df["Target_Next_Step"]
    groups = master_df["City"]

    selector = fit_inner_rfecv(X, y, groups)
    selected_lagged_features = X.columns[selector.support_].tolist()

    final_model = make_xgb_model()
    final_model.fit(X[selected_lagged_features], y)

    full_gain_df = pd.DataFrame({
        "Lagged_Feature": selected_lagged_features,
        "Gain_Importance": final_model.feature_importances_,
    })

    # LOGO permutation importance on the final selected set
    logo = LeaveOneGroupOut()
    perm_rows = []
    for fold_id, (train_idx, test_idx) in enumerate(logo.split(X[selected_lagged_features], y, groups)):
        heldout_city = groups.iloc[test_idx].iloc[0]
        X_train = X[selected_lagged_features].iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X[selected_lagged_features].iloc[test_idx]
        y_test = y.iloc[test_idx]

        fold_model = make_xgb_model()
        fold_model.fit(X_train, y_train)

        perm = permutation_importance(
            fold_model,
            X_test,
            y_test,
            n_repeats=N_PERMUTATION_REPEATS,
            random_state=RANDOM_STATE,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        for fname, m, s in zip(selected_lagged_features, perm.importances_mean, perm.importances_std):
            perm_rows.append({
                "Fold": fold_id,
                "Heldout_City": heldout_city,
                "Lagged_Feature": fname,
                "Permutation_Importance_Mean": float(m),
                "Permutation_Importance_Std": float(s),
            })

    full_perm_df = pd.DataFrame(perm_rows)
    lagged_summary, base_summary = summarize_final_lagged_and_base_features(
        selected_lagged_features=selected_lagged_features,
        gain_df=full_gain_df,
        perm_df=full_perm_df,
    )

    return {
        "master_df": master_df,
        "city_summary": city_summary,
        "selector": selector,
        "selected_lagged_features": selected_lagged_features,
        "final_model": final_model,
        "full_gain_df": full_gain_df,
        "full_perm_df": full_perm_df,
        "lagged_summary": lagged_summary,
        "base_summary": base_summary,
    }


# =============================================================================
# PLOTTING
# =============================================================================
def plot_rfecv_curve(selector: RFECV, save_path: str, title: str) -> None:
    scores = selector.cv_results_["mean_test_score"]
    stds = selector.cv_results_["std_test_score"]
    n_features = np.arange(1, len(scores) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(n_features, scores, marker="o", linewidth=2)
    plt.fill_between(n_features, scores - stds, scores + stds, alpha=0.2)
    plt.axvline(selector.n_features_, color="red", linestyle="--", linewidth=2, label=f"Selected = {selector.n_features_}")
    plt.xlabel("Number of selected lagged features")
    plt.ylabel("Inner CV score (negative RMSE)")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_top_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, save_path: str, top_n: int = 20) -> None:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return

    d = df.head(top_n).copy().sort_values(x_col, ascending=True)
    plt.figure(figsize=(11, 8))
    plt.barh(d[y_col], d[x_col])
    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def plot_lag_selection_summary(lag_choice_df: pd.DataFrame, save_path: str) -> None:
    best_rows = (
        lag_choice_df.sort_values(["Heldout_City", "Inner_CV_RMSE"])
        .groupby("Heldout_City", as_index=False)
        .first()
    )
    counts = best_rows["Candidate_Past_Steps"].value_counts().sort_index()

    plt.figure(figsize=(9, 6))
    plt.bar(counts.index.astype(str), counts.values)
    plt.xlabel("Selected past steps")
    plt.ylabel("Number of outer folds choosing this lag depth")
    plt.title("Lag-depth selection frequency across outer leave-one-city-out folds", fontsize=14, fontweight="bold")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    sns.set_theme(style="white")

    process_log = []
    process_log.append("NESTED, CITY-WISE, DEFENSIBLE FEATURE SELECTION PIPELINE")
    process_log.append("=" * 100)
    process_log.append("Target: next-step differential subsidence, built as diff(insar_cum).shift(-1)")
    process_log.append("Historical subsidence predictors are excluded from the candidate predictor set.")
    process_log.append("Selection-evaluation separation is enforced through nested leave-one-city-out validation.")
    process_log.append("Lag depth is selected only from outer-training cities using inner leave-one-city-out RFECV.")
    process_log.append("The final published feature set is obtained only after the nested stage identifies the final lag-depth setting.")
    process_log.append("No arbitrary feature-importance threshold is used anywhere in the pipeline.")
    process_log.append("The number of selected features is chosen directly by RFECV through cross-validated predictive performance.")
    process_log.append("Permutation importance is computed on held-out cities only.")
    process_log.append("Correlation analysis is descriptive only and is not used to manually prune features.")
    process_log.append("")

    print("\n" + "=" * 110)
    print("RUNNING NESTED, DEFENSIBLE FEATURE SELECTION PIPELINE")
    print("=" * 110)

    # -------------------------------------------------------------------------
    # Load and validate data
    # -------------------------------------------------------------------------
    city_cache, dynamic_features, static_features, common_features = load_all_city_data()

    with open(os.path.join(OUTPUT_DIR, "common_features_available.json"), "w", encoding="utf-8") as f:
        json.dump({
            "common_features": common_features,
            "dynamic_features_used": dynamic_features,
            "static_features_used": static_features,
            "excluded_features": EXCLUDED_FEATURES,
        }, f, indent=2)

    # -------------------------------------------------------------------------
    # Correlation analysis
    # -------------------------------------------------------------------------
    corr_input_df = build_correlation_dataset(city_cache, common_features)
    save_correlation_outputs(corr_input_df=corr_input_df, output_dir=OUTPUT_DIR)
    process_log.append("Correlation matrices (Pearson and Spearman) were generated and saved.")

    # -------------------------------------------------------------------------
    # Nested feature selection
    # -------------------------------------------------------------------------
    nested_outputs = run_nested_feature_selection_pipeline(
        city_cache=city_cache,
        dynamic_features=dynamic_features,
        static_features=static_features,
    )

    outer_summary = nested_outputs["outer_summary"]
    outer_performance = nested_outputs["outer_performance"]
    lag_choice_df = nested_outputs["lag_choice_df"]
    outer_predictions_df = nested_outputs["outer_predictions_df"]
    outer_gain_df = nested_outputs["outer_gain_df"]
    outer_perm_df = nested_outputs["outer_perm_df"]

    outer_summary.to_csv(os.path.join(OUTPUT_DIR, "outer_fold_summary.csv"), index=False, encoding="utf-8-sig")
    lag_choice_df.to_csv(os.path.join(OUTPUT_DIR, "inner_lag_selection_records.csv"), index=False, encoding="utf-8-sig")
    outer_predictions_df.to_csv(os.path.join(OUTPUT_DIR, "outer_fold_predictions.csv"), index=False, encoding="utf-8-sig")
    outer_gain_df.to_csv(os.path.join(OUTPUT_DIR, "outer_fold_gain_importance.csv"), index=False, encoding="utf-8-sig")
    outer_perm_df.to_csv(os.path.join(OUTPUT_DIR, "outer_fold_permutation_importance.csv"), index=False, encoding="utf-8-sig")

    plot_lag_selection_summary(
        lag_choice_df=lag_choice_df,
        save_path=os.path.join(OUTPUT_DIR, "lag_selection_frequency.png"),
    )

    # Outer-fold stability summaries
    selection_rows = []
    for _, row in outer_summary.iterrows():
        base_feats = [x.strip() for x in str(row["Selected_Base_Features"]).split(";") if x.strip()]
        for feat in base_feats:
            selection_rows.append({
                "Heldout_City": row["Heldout_City"],
                "Base_Pretty_Feature": feat,
            })
    base_selection_df = pd.DataFrame(selection_rows)

    if not base_selection_df.empty:
        base_selection_summary = (
            base_selection_df.groupby("Base_Pretty_Feature", as_index=False)
            .agg(Outer_Fold_Selection_Count=("Heldout_City", "count"))
            .sort_values(["Outer_Fold_Selection_Count", "Base_Pretty_Feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        base_selection_summary = pd.DataFrame(columns=["Base_Pretty_Feature", "Outer_Fold_Selection_Count"])

    if not outer_perm_df.empty:
        outer_perm_df["Base_Raw_Feature"] = outer_perm_df["Lagged_Feature"].map(extract_base_feature_name)
        outer_perm_df["Base_Pretty_Feature"] = outer_perm_df["Base_Raw_Feature"].map(raw_to_pretty)
        base_perm_summary = (
            outer_perm_df.groupby("Base_Pretty_Feature", as_index=False)
            .agg(Mean_Outer_Permutation_Importance=("Permutation_Importance_Mean", "mean"))
            .sort_values(["Mean_Outer_Permutation_Importance", "Base_Pretty_Feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        base_perm_summary = pd.DataFrame(columns=["Base_Pretty_Feature", "Mean_Outer_Permutation_Importance"])

    if not outer_gain_df.empty:
        outer_gain_df["Base_Raw_Feature"] = outer_gain_df["Lagged_Feature"].map(extract_base_feature_name)
        outer_gain_df["Base_Pretty_Feature"] = outer_gain_df["Base_Raw_Feature"].map(raw_to_pretty)
        base_gain_summary = (
            outer_gain_df.groupby("Base_Pretty_Feature", as_index=False)
            .agg(Mean_Outer_Gain_Importance=("Gain_Importance", "mean"))
            .sort_values(["Mean_Outer_Gain_Importance", "Base_Pretty_Feature"], ascending=[False, True])
            .reset_index(drop=True)
        )
    else:
        base_gain_summary = pd.DataFrame(columns=["Base_Pretty_Feature", "Mean_Outer_Gain_Importance"])

    outer_stability_summary = base_selection_summary.merge(base_perm_summary, on="Base_Pretty_Feature", how="outer")
    outer_stability_summary = outer_stability_summary.merge(base_gain_summary, on="Base_Pretty_Feature", how="outer")
    outer_stability_summary = outer_stability_summary.sort_values(
        ["Outer_Fold_Selection_Count", "Mean_Outer_Permutation_Importance", "Mean_Outer_Gain_Importance", "Base_Pretty_Feature"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    outer_stability_summary.to_csv(
        os.path.join(OUTPUT_DIR, "outer_fold_base_feature_stability_summary.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------------------
    # Determine final lag depth and run final full-data selection
    # -------------------------------------------------------------------------
    final_past_steps = determine_final_lag_depth(lag_choice_df=lag_choice_df, outer_summary=outer_summary)
    process_log.append(f"Final lag depth determined from nested analysis: {final_past_steps}")

    final_outputs = run_final_full_data_selection(
        city_cache=city_cache,
        dynamic_features=dynamic_features,
        static_features=static_features,
        final_past_steps=final_past_steps,
    )

    final_master_df = final_outputs["master_df"]
    final_city_summary = final_outputs["city_summary"]
    final_selector = final_outputs["selector"]
    final_selected_lagged_features = final_outputs["selected_lagged_features"]
    final_model = final_outputs["final_model"]
    final_gain_df = final_outputs["full_gain_df"]
    final_perm_df = final_outputs["full_perm_df"]
    final_lagged_summary = final_outputs["lagged_summary"]
    final_base_summary = final_outputs["base_summary"]

    final_city_summary.to_csv(os.path.join(OUTPUT_DIR, "final_city_summary.csv"), index=False, encoding="utf-8-sig")
    final_lagged_summary.to_csv(os.path.join(OUTPUT_DIR, "final_selected_lagged_features_summary.csv"), index=False, encoding="utf-8-sig")
    final_base_summary.to_csv(os.path.join(OUTPUT_DIR, "final_selected_base_features_summary.csv"), index=False, encoding="utf-8-sig")
    final_gain_df.to_csv(os.path.join(OUTPUT_DIR, "final_full_data_gain_importance.csv"), index=False, encoding="utf-8-sig")
    final_perm_df.to_csv(os.path.join(OUTPUT_DIR, "final_full_data_logo_permutation_importance.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(OUTPUT_DIR, "FINAL_SELECTED_LAGGED_FEATURES.txt"), "w", encoding="utf-8") as f:
        for feat in final_selected_lagged_features:
            f.write(f"{lagged_raw_to_pretty(feat)}\n")

    with open(os.path.join(OUTPUT_DIR, "FINAL_SELECTED_BASE_FEATURES.txt"), "w", encoding="utf-8") as f:
        for feat in final_base_summary["Base_Pretty_Feature"].tolist():
            f.write(f"{feat}\n")

    plot_rfecv_curve(
        selector=final_selector,
        save_path=os.path.join(OUTPUT_DIR, "final_rfecv_curve.png"),
        title=f"Final RFECV Curve | Full Data | Past Steps = {final_past_steps}",
    )

    plot_top_bar(
        final_lagged_summary,
        x_col="Permutation_Importance_Mean",
        y_col="Pretty_Lagged_Feature",
        title="Final selected lagged features by LOGO permutation importance",
        save_path=os.path.join(OUTPUT_DIR, "final_selected_lagged_features_permutation.png"),
        top_n=25,
    )

    plot_top_bar(
        final_base_summary,
        x_col="Mean_Permutation_Importance",
        y_col="Base_Pretty_Feature",
        title="Final selected base features by LOGO permutation importance",
        save_path=os.path.join(OUTPUT_DIR, "final_selected_base_features_permutation.png"),
        top_n=25,
    )

    plot_top_bar(
        final_base_summary,
        x_col="Mean_Gain_Importance",
        y_col="Base_Pretty_Feature",
        title="Final selected base features by XGBoost gain importance",
        save_path=os.path.join(OUTPUT_DIR, "final_selected_base_features_gain.png"),
        top_n=25,
    )

    # -------------------------------------------------------------------------
    # Save comprehensive text outputs
    # -------------------------------------------------------------------------
    baseline_master_df, _ = build_supervised_lagged_dataset(
        city_cache=city_cache,
        dynamic_features=dynamic_features,
        static_features=static_features,
        past_steps=final_past_steps,
    )
    baseline_y = baseline_master_df["Target_Next_Step"]
    baseline_groups = baseline_master_df["City"]
    baseline_logo = evaluate_logo_mean_baseline(baseline_y, baseline_groups)

    with open(os.path.join(OUTPUT_DIR, "pipeline_description.txt"), "w", encoding="utf-8") as f:
        f.write("DEFENSIBLE NESTED FEATURE-SELECTION PIPELINE\n")
        f.write("=" * 100 + "\n\n")
        f.write("1. Candidate predictors were predefined based on domain relevance.\n")
        f.write("2. Historical subsidence predictors were excluded from the candidate set.\n")
        f.write("3. The target was defined as next-step differential subsidence: diff(insar_cum).shift(-1).\n")
        f.write("4. Dynamic predictors were expanded into lagged representations across candidate lag depths.\n")
        f.write("5. Static predictors were included once.\n")
        f.write("6. A nested leave-one-city-out design was used.\n")
        f.write("   - Outer loop: unbiased evaluation on held-out cities.\n")
        f.write("   - Inner loop: lag-depth selection and RFECV feature selection using only outer-training cities.\n")
        f.write("7. The number of selected features was chosen directly by RFECV from cross-validated predictive performance.\n")
        f.write("8. No manual feature-importance threshold was used.\n")
        f.write("9. Final published features were obtained only after nested model-selection determined the final lag depth.\n")
        f.write("10. Predictor relevance was summarized using held-out-city permutation importance and model-based gain importance.\n")
        f.write("11. Correlation analysis was used descriptively and did not manually pre-prune predictors.\n")

    with open(os.path.join(OUTPUT_DIR, "nested_feature_selection_report.txt"), "w", encoding="utf-8") as f:
        f.write("NESTED FEATURE-SELECTION REPORT\n")
        f.write("=" * 100 + "\n\n")

        f.write("Target definition:\n")
        f.write("  Target_Next_Step = diff(insar_cum).shift(-1)\n\n")

        f.write("Candidate dynamic predictors:\n")
        for feat in dynamic_features:
            f.write(f"  - {raw_to_pretty(feat)}\n")
        f.write("\n")

        f.write("Candidate static predictors:\n")
        for feat in static_features:
            f.write(f"  - {raw_to_pretty(feat)}\n")
        f.write("\n")

        f.write("Excluded predictors:\n")
        for feat in EXCLUDED_FEATURES:
            f.write(f"  - {raw_to_pretty(feat)}\n")
        f.write("\n")

        f.write("Nested outer-fold performance:\n")
        f.write(json.dumps(outer_performance, indent=2))
        f.write("\n\n")

        f.write("LOGO mean baseline at final lag depth:\n")
        f.write(json.dumps(baseline_logo, indent=2))
        f.write("\n\n")

        f.write("Outer-fold summary:\n")
        f.write(outer_summary.to_string(index=False))
        f.write("\n\n")

        f.write("Outer-fold base-feature stability summary:\n")
        if not outer_stability_summary.empty:
            f.write(outer_stability_summary.to_string(index=False))
        else:
            f.write("No stability summary available.\n")
        f.write("\n\n")

        f.write(f"Final lag depth selected for the full-data feature-selection fit: {final_past_steps}\n\n")

        f.write("Final selected lagged features:\n")
        for feat in final_selected_lagged_features:
            f.write(f"  - {lagged_raw_to_pretty(feat)}\n")
        f.write("\n")

        f.write("Final selected base features:\n")
        for feat in final_base_summary["Base_Pretty_Feature"].tolist():
            f.write(f"  - {feat}\n")
        f.write("\n")

        f.write("Final selected lagged-feature summary:\n")
        if not final_lagged_summary.empty:
            f.write(final_lagged_summary.to_string(index=False))
        else:
            f.write("No final lagged-feature summary available.\n")
        f.write("\n\n")

        f.write("Final selected base-feature summary:\n")
        if not final_base_summary.empty:
            f.write(final_base_summary.to_string(index=False))
        else:
            f.write("No final base-feature summary available.\n")
        f.write("\n")

    with open(os.path.join(OUTPUT_DIR, "full_process_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(process_log))

    print("\n[SUCCESS] Nested, defensible feature-selection pipeline completed.")
    print(f"\nFinal lag depth: {final_past_steps}")
    print("\nFinal selected base features:")
    for i, feat in enumerate(final_base_summary["Base_Pretty_Feature"].tolist(), 1):
        print(f"{i}. {feat}")
