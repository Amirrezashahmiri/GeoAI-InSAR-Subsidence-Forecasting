import gc
import logging
import pickle
import random
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import tensorflow as tf
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models, optimizers


# =========================================================
# GLOBAL MODEL TOGGLES
# =========================================================
ENABLE_ELASTICNET = True
ENABLE_LIGHTGBM = False
ENABLE_XGBOOST = False
ENABLE_BILSTM = False

MODEL_TOGGLES = {
    "elasticnet": ENABLE_ELASTICNET,
    "lightgbm": ENABLE_LIGHTGBM,
    "xgboost": ENABLE_XGBOOST,
    "bilstm": ENABLE_BILSTM,
}

# =========================================================
# LOGGING / VISUAL STYLE
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("subsidence_model_forecast.log", "w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info(f"GPU detected and enabled: {gpus}")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {e}")
else:
    logger.info("No GPU detected, using CPU.")

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["axes.titlesize"] = 18
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14
plt.rcParams["figure.titlesize"] = 20
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3
plt.rcParams["axes.facecolor"] = "#f9f9f9"
plt.rcParams["figure.facecolor"] = "white"

COLORS = [
    "#0077B6", "#D9534F", "#5CB85C", "#F0AD4E", "#5BC0DE",
    "#428BCA", "#777777", "#F4A261", "#264653", "#E76F51"
]

warnings.filterwarnings("ignore")


class ElasticNetSubsidencePredictor:
    def __init__(
        self,
        train_data_path: Union[str, List[str]],
        val_data_path: Union[str, List[str]],
        test_data_path: Union[str, List[str]],
        random_state: int = 42,
    ):
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.test_data_path = test_data_path
        self.random_state = random_state
        self.future_steps = 1

        np.random.seed(random_state)
        random.seed(random_state)
        tf.random.set_seed(random_state)

        logger.info(f"Forecast horizon: {self.future_steps} month ahead")

        self.feature_names = [
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

        self.environmental_feature_names = [
            "Volumetric Soil Water Layer 4",
            "Leaf Area Index for High Vegetation",
            "Total Precipitation",
            "Temperature at 2 meters above the surface",
            "Surface Net Solar Radiation",
            "Dewpoint Temperature at 2 meters",
            "Soil Temperature at Level 4",
            "Total Evaporation",
            "Soil Organic Carbon Content (g/kg)",
            "Clay Percentage in Soil",
            "Height",
            "Soil pH in Water",
        ]

        self.scenario_configs = {
            "subsidence_history_only_cumulative": {
                "input_features": ["Cumulative InSAR Displacement"],
                "target_name": "Cumulative InSAR Displacement",
            },
            "subsidence_history_only_differential": {
                "input_features": ["InSAR Displacement Difference"],
                "target_name": "InSAR Displacement Difference",
            },
            "combined_cumulative": {
                "input_features": ["Cumulative InSAR Displacement"] + self.environmental_feature_names,
                "target_name": "Cumulative InSAR Displacement",
            },
            "combined_differential": {
                "input_features": ["InSAR Displacement Difference"] + self.environmental_feature_names,
                "target_name": "InSAR Displacement Difference",
            },
            "environmental_only_cumulative": {
                "input_features": self.environmental_feature_names.copy(),
                "target_name": "Cumulative InSAR Displacement",
            },
            "environmental_only_differential": {
                "input_features": self.environmental_feature_names.copy(),
                "target_name": "InSAR Displacement Difference",
            },
        }

        self.model_display_names = {
            "elasticnet": "ElasticNet",
            "lightgbm": "LightGBM",
            "xgboost": "XGBoost",
            "bilstm": "BiLSTM",
        }

        self.model_toggles = MODEL_TOGGLES.copy()
        self.model_names = [m for m, enabled in self.model_toggles.items() if enabled]
        logger.info(f"Enabled models: {self.model_names}")

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.train_pixel_group_labels = None
        self.val_pixel_group_labels = None
        self.test_pixel_group_labels = None

        self.city_datasets: Dict[str, Dict[str, Any]] = {}

        self.model = None
        self.scaler = None
        self.model_results: Dict[str, Any] = {}
        self.cv_results: Dict[str, Any] = {}
        self.optimal_time_steps = 0
        self.feature_indices: List[int] = []
        self.X_test_flat = None
        self.X_test_seq = None
        self.flat_feature_names = None
        self.best_params: Dict[str, Any] = {}
        self.scenario_results: Dict[str, Any] = {}
        self.current_model_name = "elasticnet"
        self.current_scenario_name = ""
        self.current_train_sequences = None
        self.current_val_sequences = None
        self.current_test_sequences = None
        self.sample_weight_summary = {}
        self.current_target_name = "Cumulative InSAR Displacement"
        self.current_target_index = self.feature_names.index(self.current_target_name)
        self.incity_results_dict: Dict[str, Any] = {}

        self.output_root_dir = Path(__file__).resolve().parent / "elasticnet_six_scenarios_outputs"
        self.output_root_dir.mkdir(exist_ok=True)
        self.output_dir = self.output_root_dir

    # =========================================================
    # RESUME / SKIP COMPLETED RUNS
    # =========================================================
    def _get_model_output_dir(self, scenario_name: str, model_name: str) -> Path:
        return self.output_root_dir / scenario_name / model_name

    def _get_model_result_pickle_path(self, scenario_name: str, model_name: str) -> Path:
        model_output_dir = self._get_model_output_dir(scenario_name, model_name)
        return model_output_dir / f"{model_name}_results.pickle"

    def _get_model_artifact_paths(self, scenario_name: str, model_name: str) -> List[Path]:
        model_output_dir = self._get_model_output_dir(scenario_name, model_name)

        if model_name == "bilstm":
            return [
                model_output_dir / "bilstm_results.pickle",
                model_output_dir / "bilstm_model.keras",
                model_output_dir / "bilstm_scaler.pickle",
                model_output_dir / "bilstm_scenario_summary.txt",
            ]

        return [
            model_output_dir / f"{model_name}_results.pickle",
            model_output_dir / f"{model_name}_model_and_scaler.pickle",
            model_output_dir / f"{model_name}_scenario_summary.txt",
        ]

    def _is_completed_model_run(self, scenario_name: str, model_name: str) -> bool:
        required_paths = self._get_model_artifact_paths(scenario_name, model_name)
        if not all(path.exists() for path in required_paths):
            return False

        results_pickle = self._get_model_result_pickle_path(scenario_name, model_name)
        try:
            with open(results_pickle, "rb") as f:
                payload = pickle.load(f)

            required_keys = [
                "model_name",
                "scenario_name",
                "target_name",
                "model_results",
                "cv_results",
                "optimal_time_steps",
                "best_hyperparameters",
                "selected_features",
            ]
            if not all(k in payload for k in required_keys):
                return False

            if "test" not in payload["model_results"]:
                return False

            if "best_result" not in payload["cv_results"]:
                return False

            return True

        except Exception as e:
            logger.warning(
                f"Existing result file for scenario='{scenario_name}', model='{model_name}' "
                f"could not be read cleanly and will be recomputed. Details: {e}"
            )
            return False

    def _load_completed_model_run(self, scenario_name: str, model_name: str) -> Dict[str, Any]:
        results_pickle = self._get_model_result_pickle_path(scenario_name, model_name)
        with open(results_pickle, "rb") as f:
            payload = pickle.load(f)

        return {
            "model_name": payload["model_name"],
            "scenario_name": payload["scenario_name"],
            "target_name": payload["target_name"],
            "selected_features": payload.get("selected_features", []),
            "model_results": payload["model_results"],
            "cv_results": payload["cv_results"],
            "incity_results": payload.get("incity_results", payload.get("incity_results_dict", payload.get("incity_results", {}))),
            "optimal_time_steps": payload["optimal_time_steps"],
            "best_hyperparameters": payload.get("best_hyperparameters", {}),
            "output_dir": str(self._get_model_output_dir(scenario_name, model_name)),
            "skill_vs_persistence": payload.get("skill_vs_persistence", {}),
        }

    # =========================================================
    # DATA LOADING
    # =========================================================
    def _extract_group_name_from_path(self, path: str) -> str:
        return Path(path).parent.name

    def _load_single_npz_with_labels(self, path: str) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Loading data from: {path}")
        npz_file = np.load(path)
        data = npz_file["data"]
        if len(data.shape) != 3:
            raise ValueError(f"Expected 3D data in {path}, but got {len(data.shape)}D")
        logger.info(f"  - Data shape: {data.shape}")
        city_name = self._extract_group_name_from_path(path)
        pixel_labels = np.array([city_name] * data.shape[1], dtype=object)
        return data, pixel_labels

    def _load_data_and_pixel_labels_from_path(
        self, path: Union[str, List[str], Tuple[str, ...]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        try:
            if isinstance(path, (list, tuple)):
                data_list = []
                label_list = []

                for p in path:
                    data, pixel_labels = self._load_single_npz_with_labels(p)
                    data_list.append(data)
                    label_list.append(pixel_labels)

                min_time = min(d.shape[0] for d in data_list)
                logger.info(f"Aligning {len(data_list)} datasets to min common time length = {min_time}")

                merged_data = np.concatenate([d[:min_time, :, :] for d in data_list], axis=1)
                merged_labels = np.concatenate(label_list, axis=0)

                logger.info(f"  - Final merged shape: {merged_data.shape}")
                return merged_data, merged_labels

            return self._load_single_npz_with_labels(path)

        except Exception as e:
            logger.error(f"Error loading data from {path}: {e}")
            raise

    def _list_all_paths(self) -> List[str]:
        train_paths = self.train_data_path if isinstance(self.train_data_path, list) else [self.train_data_path]
        val_paths = self.val_data_path if isinstance(self.val_data_path, list) else [self.val_data_path]
        test_paths = self.test_data_path if isinstance(self.test_data_path, list) else [self.test_data_path]
        return train_paths + val_paths + test_paths

    def _build_city_datasets(self) -> None:
        self.city_datasets = {}
        for path in self._list_all_paths():
            city_name = self._extract_group_name_from_path(path)
            data, pixel_labels = self._load_single_npz_with_labels(path)
            self.city_datasets[city_name] = {
                "data": data,
                "pixel_labels": pixel_labels,
                "path": path,
            }
        logger.info(f"Loaded {len(self.city_datasets)} city datasets for city-wise CV.")

    def load_datasets(self) -> None:
        gc.collect()
        logger.info("Loading and aligning merged datasets...")

        self.train_data, self.train_pixel_group_labels = self._load_data_and_pixel_labels_from_path(self.train_data_path)
        self.val_data, self.val_pixel_group_labels = self._load_data_and_pixel_labels_from_path(self.val_data_path)
        self.test_data, self.test_pixel_group_labels = self._load_data_and_pixel_labels_from_path(self.test_data_path)

        logger.info(f"Final Train Shape: {self.train_data.shape}")
        logger.info(f"Final Validation Shape: {self.val_data.shape}")
        logger.info(f"Final Test Shape: {self.test_data.shape}")

        self._build_city_datasets()

    # =========================================================
    # FEATURE / TARGET CONFIG
    # =========================================================
    def _set_scenario_features(self, scenario_name: str) -> None:
        if scenario_name not in self.scenario_configs:
            raise ValueError(f"Unknown scenario: {scenario_name}")

        scenario_cfg = self.scenario_configs[scenario_name]
        scenario_feature_names = scenario_cfg["input_features"]
        self.current_target_name = scenario_cfg["target_name"]
        self.current_target_index = self.feature_names.index(self.current_target_name)
        self.feature_indices = [self.feature_names.index(name) for name in scenario_feature_names]

        logger.info(f"Scenario '{scenario_name}' selected features: {scenario_feature_names}")
        logger.info(f"Scenario '{scenario_name}' selected indices: {self.feature_indices}")
        logger.info(f"Scenario '{scenario_name}' target: {self.current_target_name} (idx={self.current_target_index})")

    def _is_current_target_differential(self) -> bool:
        return self.current_target_name == "InSAR Displacement Difference"

    def _get_target_plot_label(self) -> str:
        if self.current_target_name == "InSAR Displacement Difference":
            return "Differential Subsidence (mm)"
        return "Cumulative Subsidence (mm)"

    # =========================================================
    # SEQUENCE BUILDING
    # =========================================================
    def create_sequences(
        self,
        data: np.ndarray,
        past_months: int,
        pixel_group_labels: Optional[np.ndarray] = None,
        return_metadata: bool = False,
    ):
        sequences, targets = [], []
        meta_group_labels = []
        meta_target_abs = []

        num_timesteps, num_grid_cells, _ = data.shape

        if pixel_group_labels is None:
            pixel_group_labels = np.array(["Unknown"] * num_grid_cells, dtype=object)

        for grid_idx in range(num_grid_cells):
            city_name = str(pixel_group_labels[grid_idx])

            for t in range(past_months, num_timesteps - self.future_steps + 1):
                seq = data[t - past_months:t, grid_idx, self.feature_indices]
                target = data[t:t + self.future_steps, grid_idx, self.current_target_index]

                sequences.append(seq)
                targets.append(target)
                meta_group_labels.append(city_name)
                meta_target_abs.append(float(np.abs(target).ravel()[0]))

        sequences = np.array(sequences)
        targets = np.array(targets)
        meta_group_labels = np.array(meta_group_labels, dtype=object)
        meta_target_abs = np.array(meta_target_abs, dtype=float)

        valid_mask = ~(np.isnan(sequences).any(axis=(1, 2)) | np.isnan(targets).any(axis=1))

        if return_metadata:
            metadata = {
                "group_labels": meta_group_labels[valid_mask],
                "target_abs": meta_target_abs[valid_mask],
            }
            return sequences[valid_mask], targets[valid_mask], metadata

        return sequences[valid_mask], targets[valid_mask]

    def create_persistence_targets(self, data: np.ndarray, past_months: int) -> Tuple[np.ndarray, np.ndarray]:
        persistence_preds, targets = [], []
        num_timesteps, num_grid_cells, _ = data.shape

        for grid_idx in range(num_grid_cells):
            for t in range(past_months, num_timesteps - self.future_steps + 1):
                last_observed_target = data[t - 1, grid_idx, self.current_target_index]
                target = data[t:t + self.future_steps, grid_idx, self.current_target_index]
                persistence_preds.append(last_observed_target)
                targets.append(target)

        persistence_preds = np.array(persistence_preds)
        targets = np.array(targets)
        valid_mask = ~(np.isnan(persistence_preds) | np.isnan(targets).any(axis=1))
        return persistence_preds[valid_mask], targets[valid_mask]

    # =========================================================
    # DATA SPLITS / UTILS
    # =========================================================
    def _flatten_sequences(self, X_seq: np.ndarray) -> np.ndarray:
        return X_seq.reshape(X_seq.shape[0], -1)

    def _fit_sequence_scaler(self, X_train_seq: np.ndarray) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(self._flatten_sequences(X_train_seq))
        return scaler

    def _transform_sequence_with_scaler(
        self, scaler: StandardScaler, X_seq: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        X_flat = self._flatten_sequences(X_seq)
        X_flat_scaled = scaler.transform(X_flat)
        X_seq_scaled = X_flat_scaled.reshape(X_seq.shape[0], X_seq.shape[1], X_seq.shape[2])
        return X_seq_scaled, X_flat_scaled

    def _create_temporal_train_val_split(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        validation_fraction: float = 0.2,
    ):
        n_samples = len(X_seq)
        if n_samples < 10:
            return X_seq, X_seq, y, y

        split_index = max(1, int(n_samples * (1 - validation_fraction)))
        split_index = min(split_index, n_samples - 1)

        X_train = X_seq[:split_index]
        X_val = X_seq[split_index:]
        y_train = y[:split_index]
        y_val = y[split_index:]

        return X_train, X_val, y_train, y_val

    def _create_citywise_temporal_split(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        test_fraction: float = 0.2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n_samples = len(X_seq)
        if n_samples < 10:
            raise ValueError("Not enough samples for temporal city split.")

        split_index = max(1, int(n_samples * (1 - test_fraction)))
        split_index = min(split_index, n_samples - 1)

        X_train = X_seq[:split_index]
        X_test = X_seq[split_index:]
        y_train = y[:split_index]
        y_test = y[split_index:]
        return X_train, X_test, y_train, y_test

    def _build_bilstm_model(
        self,
        input_shape: Tuple[int, int],
        units: int,
        dropout_rate: float,
        learning_rate: float,
    ) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=input_shape),
            layers.Bidirectional(
                layers.LSTM(
                    units,
                    return_sequences=False,
                    activation="tanh",
                    recurrent_activation="sigmoid",
                    recurrent_dropout=0.0,
                    use_bias=True,
                )
            ),
            layers.Dropout(dropout_rate),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ])
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="mse",
        )
        return model

    # =========================================================
    # TUNING
    # =========================================================
    def _tune_elasticnet(
        self,
        X_train_flat: np.ndarray,
        y_train_flat: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        model = ElasticNetCV(
            cv=5,
            random_state=self.random_state,
            n_jobs=-1,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        )
        model.fit(X_train_flat, y_train_flat)

        params = {
            "alpha": float(model.alpha_),
            "l1_ratio": float(model.l1_ratio_),
        }
        return model, params

    def _tune_lightgbm(
        self,
        X_train_flat: np.ndarray,
        y_train_flat: np.ndarray,
        X_val_flat: np.ndarray,
        y_val_flat: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        param_grid = [
            {"n_estimators": 200, "learning_rate": 0.05, "num_leaves": 15, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 10, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 400, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9},
            {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 63, "min_child_samples": 20, "subsample": 0.9, "colsample_bytree": 0.9},
        ]

        best_model = None
        best_params = None
        best_score = -np.inf

        for params in param_grid:
            model = LGBMRegressor(
                objective="regression",
                random_state=self.random_state,
                n_jobs=-1,
                **params,
            )
            model.fit(X_train_flat, y_train_flat)
            y_val_pred = model.predict(X_val_flat)
            score = r2_score(y_val_flat, y_val_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params.copy()

        return best_model, best_params

    def _tune_xgboost(
        self,
        X_train_flat: np.ndarray,
        y_train_flat: np.ndarray,
        X_val_flat: np.ndarray,
        y_val_flat: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        param_grid = [
            {
                "n_estimators": 300,
                "learning_rate": 0.05,
                "max_depth": 3,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "learning_rate": 0.03,
                "max_depth": 4,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 5,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
            },
            {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "max_depth": 6,
                "min_child_weight": 5,
                "subsample": 0.9,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 2.0,
            },
        ]

        best_model = None
        best_params = None
        best_score = -np.inf

        for params in param_grid:
            model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
                **params,
            )
            model.fit(
                X_train_flat,
                y_train_flat,
                eval_set=[(X_val_flat, y_val_flat)],
                verbose=False,
            )
            y_val_pred = model.predict(X_val_flat)
            score = r2_score(y_val_flat, y_val_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params.copy()

        return best_model, best_params

    def _tune_bilstm(
        self,
        X_train_seq: np.ndarray,
        y_train_flat: np.ndarray,
        X_val_seq: np.ndarray,
        y_val_flat: np.ndarray,
    ) -> Tuple[Any, Dict[str, Any]]:
        param_grid = [
            {"units": 32, "dropout_rate": 0.1, "learning_rate": 1e-3, "batch_size": 32, "epochs": 80},
            {"units": 64, "dropout_rate": 0.1, "learning_rate": 1e-3, "batch_size": 32, "epochs": 80},
            {"units": 64, "dropout_rate": 0.2, "learning_rate": 5e-4, "batch_size": 32, "epochs": 100},
        ]

        best_model = None
        best_params = None
        best_score = -np.inf

        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        for params in param_grid:
            tf.keras.backend.clear_session()
            model = self._build_bilstm_model(
                input_shape=(X_train_seq.shape[1], X_train_seq.shape[2]),
                units=params["units"],
                dropout_rate=params["dropout_rate"],
                learning_rate=params["learning_rate"],
            )
            model.fit(
                X_train_seq,
                y_train_flat,
                validation_data=(X_val_seq, y_val_flat),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=[early_stop],
                verbose=0,
            )
            y_val_pred = model.predict(X_val_seq, verbose=0).ravel()
            score = r2_score(y_val_flat, y_val_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = params.copy()

        return best_model, best_params

    # =========================================================
    # MODEL FIT / PREDICT CORE
    # =========================================================
    def _fit_and_predict_model(
        self,
        model_name: str,
        X_train_seq: np.ndarray,
        y_train_flat: np.ndarray,
        X_test_seq: np.ndarray,
    ) -> Tuple[Any, Optional[StandardScaler], Dict[str, Any], np.ndarray]:
        if model_name == "elasticnet":
            scaler = self._fit_sequence_scaler(X_train_seq)
            _, X_train_flat_scaled = self._transform_sequence_with_scaler(scaler, X_train_seq)
            _, X_test_flat_scaled = self._transform_sequence_with_scaler(scaler, X_test_seq)

            model, best_params = self._tune_elasticnet(X_train_flat_scaled, y_train_flat)
            y_test_pred = model.predict(X_test_flat_scaled).ravel()
            return model, scaler, best_params, y_test_pred

        if model_name == "lightgbm":
            inner_X_train_seq, inner_X_val_seq, inner_y_train, inner_y_val = self._create_temporal_train_val_split(
                X_train_seq, y_train_flat
            )

            inner_X_train_flat = self._flatten_sequences(inner_X_train_seq)
            inner_X_val_flat = self._flatten_sequences(inner_X_val_seq)
            X_train_flat = self._flatten_sequences(X_train_seq)
            X_test_flat = self._flatten_sequences(X_test_seq)

            _, best_params = self._tune_lightgbm(
                inner_X_train_flat, inner_y_train, inner_X_val_flat, inner_y_val
            )

            final_model = LGBMRegressor(
                objective="regression",
                random_state=self.random_state,
                n_jobs=-1,
                **best_params,
            )
            final_model.fit(X_train_flat, y_train_flat)
            y_test_pred = final_model.predict(X_test_flat).ravel()
            return final_model, None, best_params, y_test_pred

        if model_name == "xgboost":
            inner_X_train_seq, inner_X_val_seq, inner_y_train, inner_y_val = self._create_temporal_train_val_split(
                X_train_seq, y_train_flat
            )

            inner_X_train_flat = self._flatten_sequences(inner_X_train_seq)
            inner_X_val_flat = self._flatten_sequences(inner_X_val_seq)
            X_train_flat = self._flatten_sequences(X_train_seq)
            X_test_flat = self._flatten_sequences(X_test_seq)

            _, best_params = self._tune_xgboost(
                inner_X_train_flat, inner_y_train, inner_X_val_flat, inner_y_val
            )

            final_model = xgb.XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=self.random_state,
                n_jobs=-1,
                **best_params,
            )
            final_model.fit(X_train_flat, y_train_flat, verbose=False)
            y_test_pred = final_model.predict(X_test_flat).ravel()
            return final_model, None, best_params, y_test_pred

        if model_name == "bilstm":
            scaler = self._fit_sequence_scaler(X_train_seq)
            X_train_seq_scaled, _ = self._transform_sequence_with_scaler(scaler, X_train_seq)
            X_test_seq_scaled, _ = self._transform_sequence_with_scaler(scaler, X_test_seq)

            inner_X_train_seq, inner_X_val_seq, inner_y_train, inner_y_val = self._create_temporal_train_val_split(
                X_train_seq_scaled, y_train_flat
            )

            _, best_params = self._tune_bilstm(
                inner_X_train_seq, inner_y_train, inner_X_val_seq, inner_y_val
            )

            tf.keras.backend.clear_session()
            final_model = self._build_bilstm_model(
                input_shape=(X_train_seq_scaled.shape[1], X_train_seq_scaled.shape[2]),
                units=best_params["units"],
                dropout_rate=best_params["dropout_rate"],
                learning_rate=best_params["learning_rate"],
            )
            early_stop = callbacks.EarlyStopping(
                monitor="loss",
                patience=10,
                restore_best_weights=True,
            )
            final_model.fit(
                X_train_seq_scaled,
                y_train_flat,
                epochs=best_params["epochs"],
                batch_size=best_params["batch_size"],
                callbacks=[early_stop],
                verbose=0,
            )
            y_test_pred = final_model.predict(X_test_seq_scaled, verbose=0).ravel()
            return final_model, scaler, best_params, y_test_pred

        raise ValueError(f"Unsupported model: {model_name}")

    # =========================================================
    # CITY-WISE CV
    # =========================================================
    def run_leave_one_city_out_cv(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        if model_name is None:
            model_name = self.current_model_name

        logger.info(f"Running leave-one-city-out CV for model '{self.model_display_names[model_name]}'...")
        city_names = list(self.city_datasets.keys())
        time_steps_range = range(2, 12)

        time_step_results = []
        best_overall_result = None

        for past_months in time_steps_range:
            fold_results = []

            for held_out_city in city_names:
                try:
                    held_bundle = self.city_datasets[held_out_city]
                    X_test_fold, y_test_fold = self.create_sequences(
                        held_bundle["data"],
                        past_months,
                        pixel_group_labels=held_bundle["pixel_labels"],
                        return_metadata=False,
                    )

                    X_train_parts, y_train_parts = [], []

                    for train_city in city_names:
                        if train_city == held_out_city:
                            continue
                        city_bundle = self.city_datasets[train_city]
                        X_part, y_part = self.create_sequences(
                            city_bundle["data"],
                            past_months,
                            pixel_group_labels=city_bundle["pixel_labels"],
                            return_metadata=False,
                        )
                        if len(X_part) > 0:
                            X_train_parts.append(X_part)
                            y_train_parts.append(y_part)

                    if not X_train_parts or len(X_test_fold) < 5:
                        continue

                    X_train_fold = np.concatenate(X_train_parts, axis=0)
                    y_train_fold = np.concatenate(y_train_parts, axis=0)

                    if len(X_train_fold) < 30:
                        continue

                    trained_model, _, best_params, y_test_pred = self._fit_and_predict_model(
                        model_name=model_name,
                        X_train_seq=X_train_fold,
                        y_train_flat=y_train_fold.ravel(),
                        X_test_seq=X_test_fold,
                    )

                    fold_metrics = {
                        "held_out_group": held_out_city,
                        "r2": r2_score(y_test_fold.ravel(), y_test_pred),
                        "rmse": np.sqrt(mean_squared_error(y_test_fold.ravel(), y_test_pred)),
                        "mae": mean_absolute_error(y_test_fold.ravel(), y_test_pred),
                        "params": best_params,
                        "n_train_samples": int(len(X_train_fold)),
                        "n_test_samples": int(len(X_test_fold)),
                    }
                    fold_results.append(fold_metrics)

                    del trained_model
                    gc.collect()

                except Exception as e:
                    logger.error(f"Error in city CV fold ({held_out_city}): {e}", exc_info=True)
                    continue

            if len(fold_results) == 0:
                continue

            avg_r2 = float(np.mean([f["r2"] for f in fold_results]))
            avg_rmse = float(np.mean([f["rmse"] for f in fold_results]))
            avg_mae = float(np.mean([f["mae"] for f in fold_results]))

            result = {
                "past_months": past_months,
                "avg_r2": avg_r2,
                "avg_rmse": avg_rmse,
                "avg_mae": avg_mae,
                "folds": fold_results,
            }
            time_step_results.append(result)

            if best_overall_result is None or avg_r2 > best_overall_result["avg_r2"]:
                best_overall_result = result

        self.cv_results = {
            "model_name": model_name,
            "best_result": best_overall_result,
            "time_step_results": time_step_results,
            "cv_type": "leave_one_city_out",
        }
        return self.cv_results

    def find_optimal_time_steps(self, model_name: Optional[str] = None) -> int:
        cv_results = self.run_leave_one_city_out_cv(model_name=model_name)
        return cv_results["best_result"]["past_months"]

    # =========================================================
    # FINAL TRAINING PER MODEL
    # =========================================================
    def _prepare_final_train_val_test(self):
        X_train, y_train, _ = self.create_sequences(
            self.train_data,
            self.optimal_time_steps,
            pixel_group_labels=self.train_pixel_group_labels,
            return_metadata=True,
        )
        X_val, y_val = self.create_sequences(
            self.val_data,
            self.optimal_time_steps,
            pixel_group_labels=self.val_pixel_group_labels,
            return_metadata=False,
        )
        X_test, y_test = self.create_sequences(
            self.test_data,
            self.optimal_time_steps,
            pixel_group_labels=self.test_pixel_group_labels,
            return_metadata=False,
        )
        return X_train, y_train, X_val, y_val, X_test, y_test

    def _build_flat_feature_names(self) -> None:
        self.flat_feature_names = [
            f"{f.split('(')[0].strip()} (t-{self.optimal_time_steps - t - 1})"
            for t in range(self.optimal_time_steps)
            for f in [self.feature_names[idx] for idx in self.feature_indices]
        ]

    def train_elasticnet_model(self) -> None:
        self.current_model_name = "elasticnet"
        self.optimal_time_steps = self.find_optimal_time_steps(model_name="elasticnet")
        self._build_flat_feature_names()

        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_final_train_val_test()

        self.current_train_sequences = X_train
        self.current_val_sequences = X_val
        self.current_test_sequences = X_test

        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)
        self.X_test_flat = self._flatten_sequences(X_test)
        self.X_test_seq = X_test

        y_train_flat = y_train.ravel()
        y_val_flat = y_val.ravel()
        y_test_flat = y_test.ravel()

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_flat)
        X_val_scaled = self.scaler.transform(X_val_flat)
        X_test_scaled = self.scaler.transform(self.X_test_flat)

        self.model = ElasticNetCV(
            cv=5,
            random_state=self.random_state,
            n_jobs=-1,
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
        )
        self.model.fit(X_train_scaled, y_train_flat)

        self.best_params = {
            "alpha": float(self.model.alpha_),
            "l1_ratio": float(self.model.l1_ratio_),
        }

        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        self.model_results = {
            "train": {
                "r2": r2_score(y_train_flat, y_train_pred),
                "rmse": np.sqrt(mean_squared_error(y_train_flat, y_train_pred)),
                "mae": mean_absolute_error(y_train_flat, y_train_pred),
            },
            "val": {
                "r2": r2_score(y_val_flat, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val_flat, y_val_pred)),
                "mae": mean_absolute_error(y_val_flat, y_val_pred),
            },
            "test": {
                "r2": r2_score(y_test_flat, y_test_pred),
                "rmse": np.sqrt(mean_squared_error(y_test_flat, y_test_pred)),
                "mae": mean_absolute_error(y_test_flat, y_test_pred),
            },
            "predictions": {
                "y_test_pred": y_test_pred,
                "y_test": y_test_flat,
            },
        }

    def train_lightgbm_model(self) -> None:
        self.current_model_name = "lightgbm"
        self.optimal_time_steps = self.find_optimal_time_steps(model_name="lightgbm")
        self._build_flat_feature_names()

        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_final_train_val_test()

        self.current_train_sequences = X_train
        self.current_val_sequences = X_val
        self.current_test_sequences = X_test

        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)
        self.X_test_flat = self._flatten_sequences(X_test)
        self.X_test_seq = X_test

        y_train_flat = y_train.ravel()
        y_val_flat = y_val.ravel()
        y_test_flat = y_test.ravel()

        _, best_params = self._tune_lightgbm(X_train_flat, y_train_flat, X_val_flat, y_val_flat)

        self.model = LGBMRegressor(
            objective="regression",
            random_state=self.random_state,
            n_jobs=-1,
            **best_params,
        )
        self.model.fit(X_train_flat, y_train_flat)
        self.scaler = None
        self.best_params = best_params

        y_train_pred = self.model.predict(X_train_flat)
        y_val_pred = self.model.predict(X_val_flat)
        y_test_pred = self.model.predict(self.X_test_flat)

        self.model_results = {
            "train": {
                "r2": r2_score(y_train_flat, y_train_pred),
                "rmse": np.sqrt(mean_squared_error(y_train_flat, y_train_pred)),
                "mae": mean_absolute_error(y_train_flat, y_train_pred),
            },
            "val": {
                "r2": r2_score(y_val_flat, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val_flat, y_val_pred)),
                "mae": mean_absolute_error(y_val_flat, y_val_pred),
            },
            "test": {
                "r2": r2_score(y_test_flat, y_test_pred),
                "rmse": np.sqrt(mean_squared_error(y_test_flat, y_test_pred)),
                "mae": mean_absolute_error(y_test_flat, y_test_pred),
            },
            "predictions": {
                "y_test_pred": y_test_pred,
                "y_test": y_test_flat,
            },
        }

    def train_xgboost_model(self) -> None:
        self.current_model_name = "xgboost"
        self.optimal_time_steps = self.find_optimal_time_steps(model_name="xgboost")
        self._build_flat_feature_names()

        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_final_train_val_test()

        self.current_train_sequences = X_train
        self.current_val_sequences = X_val
        self.current_test_sequences = X_test

        X_train_flat = self._flatten_sequences(X_train)
        X_val_flat = self._flatten_sequences(X_val)
        self.X_test_flat = self._flatten_sequences(X_test)
        self.X_test_seq = X_test

        y_train_flat = y_train.ravel()
        y_val_flat = y_val.ravel()
        y_test_flat = y_test.ravel()

        _, best_params = self._tune_xgboost(X_train_flat, y_train_flat, X_val_flat, y_val_flat)

        self.model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            random_state=self.random_state,
            n_jobs=-1,
            **best_params,
        )
        self.model.fit(
            X_train_flat,
            y_train_flat,
            eval_set=[(X_val_flat, y_val_flat)],
            verbose=False,
        )

        self.scaler = None
        self.best_params = best_params

        y_train_pred = self.model.predict(X_train_flat)
        y_val_pred = self.model.predict(X_val_flat)
        y_test_pred = self.model.predict(self.X_test_flat)

        self.model_results = {
            "train": {
                "r2": r2_score(y_train_flat, y_train_pred),
                "rmse": np.sqrt(mean_squared_error(y_train_flat, y_train_pred)),
                "mae": mean_absolute_error(y_train_flat, y_train_pred),
            },
            "val": {
                "r2": r2_score(y_val_flat, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val_flat, y_val_pred)),
                "mae": mean_absolute_error(y_val_flat, y_val_pred),
            },
            "test": {
                "r2": r2_score(y_test_flat, y_test_pred),
                "rmse": np.sqrt(mean_squared_error(y_test_flat, y_test_pred)),
                "mae": mean_absolute_error(y_test_flat, y_test_pred),
            },
            "predictions": {
                "y_test_pred": y_test_pred,
                "y_test": y_test_flat,
            },
        }

    def train_bilstm_model(self) -> None:
        self.current_model_name = "bilstm"
        self.optimal_time_steps = self.find_optimal_time_steps(model_name="bilstm")
        self._build_flat_feature_names()

        X_train, y_train, X_val, y_val, X_test, y_test = self._prepare_final_train_val_test()

        self.current_train_sequences = X_train
        self.current_val_sequences = X_val
        self.current_test_sequences = X_test

        y_train_flat = y_train.ravel()
        y_val_flat = y_val.ravel()
        y_test_flat = y_test.ravel()

        self.scaler = self._fit_sequence_scaler(X_train)
        X_train_scaled_seq, X_train_scaled_flat = self._transform_sequence_with_scaler(self.scaler, X_train)
        X_val_scaled_seq, X_val_scaled_flat = self._transform_sequence_with_scaler(self.scaler, X_val)
        X_test_scaled_seq, X_test_scaled_flat = self._transform_sequence_with_scaler(self.scaler, X_test)

        self.X_test_flat = X_test_scaled_flat
        self.X_test_seq = X_test_scaled_seq

        _, best_params = self._tune_bilstm(
            X_train_scaled_seq, y_train_flat, X_val_scaled_seq, y_val_flat
        )

        tf.keras.backend.clear_session()
        self.model = self._build_bilstm_model(
            input_shape=(X_train_scaled_seq.shape[1], X_train_scaled_seq.shape[2]),
            units=best_params["units"],
            dropout_rate=best_params["dropout_rate"],
            learning_rate=best_params["learning_rate"],
        )
        early_stop = callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )

        self.model.fit(
            X_train_scaled_seq,
            y_train_flat,
            validation_data=(X_val_scaled_seq, y_val_flat),
            epochs=best_params["epochs"],
            batch_size=best_params["batch_size"],
            callbacks=[early_stop],
            verbose=0,
        )

        self.best_params = best_params

        y_train_pred = self.model.predict(X_train_scaled_seq, verbose=0).ravel()
        y_val_pred = self.model.predict(X_val_scaled_seq, verbose=0).ravel()
        y_test_pred = self.model.predict(X_test_scaled_seq, verbose=0).ravel()

        self.model_results = {
            "train": {
                "r2": r2_score(y_train_flat, y_train_pred),
                "rmse": np.sqrt(mean_squared_error(y_train_flat, y_train_pred)),
                "mae": mean_absolute_error(y_train_flat, y_train_pred),
            },
            "val": {
                "r2": r2_score(y_val_flat, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val_flat, y_val_pred)),
                "mae": mean_absolute_error(y_val_flat, y_val_pred),
            },
            "test": {
                "r2": r2_score(y_test_flat, y_test_pred),
                "rmse": np.sqrt(mean_squared_error(y_test_flat, y_test_pred)),
                "mae": mean_absolute_error(y_test_flat, y_test_pred),
            },
            "predictions": {
                "y_test_pred": y_test_pred,
                "y_test": y_test_flat,
            },
        }

    # =========================================================
    # PERSISTENCE / SKILL
    # =========================================================
    def evaluate_persistence_model(self, past_months: int) -> Dict[str, Any]:
        train_pred, train_true = self.create_persistence_targets(self.train_data, past_months)
        val_pred, val_true = self.create_persistence_targets(self.val_data, past_months)
        test_pred, test_true = self.create_persistence_targets(self.test_data, past_months)

        return {
            "train": {
                "r2": r2_score(train_true.ravel(), train_pred),
                "rmse": np.sqrt(mean_squared_error(train_true.ravel(), train_pred)),
                "mae": mean_absolute_error(train_true.ravel(), train_pred),
            },
            "val": {
                "r2": r2_score(val_true.ravel(), val_pred),
                "rmse": np.sqrt(mean_squared_error(val_true.ravel(), val_pred)),
                "mae": mean_absolute_error(val_true.ravel(), val_pred),
            },
            "test": {
                "r2": r2_score(test_true.ravel(), test_pred),
                "rmse": np.sqrt(mean_squared_error(test_true.ravel(), test_pred)),
                "mae": mean_absolute_error(test_true.ravel(), test_pred),
            },
            "predictions": {
                "y_test_pred": test_pred,
                "y_test": test_true.ravel(),
            },
        }

    def _compute_skill_against_persistence(self, model_results: Dict[str, Any], persistence_results: Dict[str, Any]) -> Dict[str, float]:
        return {
            "delta_r2": model_results["test"]["r2"] - persistence_results["test"]["r2"],
            "rmse_reduction": persistence_results["test"]["rmse"] - model_results["test"]["rmse"],
            "mae_reduction": persistence_results["test"]["mae"] - model_results["test"]["mae"],
        }

    # =========================================================
    # PLOTTING
    # =========================================================
    def plot_performance_summary(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        metrics = ["r2", "rmse", "mae"]
        metric_names = ["R² Score", "RMSE (mm)", "MAE (mm)"]

        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            values = [self.model_results[split][metric] for split in ["train", "val", "test"]]
            bars = ax.bar(["Train", "Validation", "Test"], values, color=COLORS[:3], alpha=0.8)
            ax.set_title(f"{self.model_display_names[self.current_model_name]} - {name}", fontsize=16, fontweight="bold")
            ax.set_ylabel(name, fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}",
                        ha="center", va="bottom", fontsize=12)

        plt.suptitle(f"{self.model_display_names[self.current_model_name]} Performance Summary", fontsize=20, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / f"{self.current_model_name}_performance_summary.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")

    def plot_actual_vs_predicted(self) -> None:
        y_test = self.model_results["predictions"]["y_test"]
        y_test_pred = self.model_results["predictions"]["y_test_pred"]
        r2 = self.model_results["test"]["r2"]
        rmse = self.model_results["test"]["rmse"]
        mae = self.model_results["test"]["mae"]

        fig = plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_test_pred, alpha=0.5, color=COLORS[0], edgecolors="k", s=50)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2.5, label="Ideal Fit")
        plt.xlabel(f"Actual {self._get_target_plot_label()}", fontsize=14)
        plt.ylabel(f"Predicted {self._get_target_plot_label()}", fontsize=14)
        plt.title(
            f"Actual vs. Predicted ({self.model_display_names[self.current_model_name]} - Test Set)\nR² = {r2:.4f}",
            fontsize=16,
            fontweight="bold",
        )
        plt.text(
            0.05, 0.95, f"RMSE: {rmse:.3f} mm\nMAE: {mae:.3f} mm",
            transform=plt.gca().transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig(self.output_dir / f"{self.current_model_name}_actual_vs_predicted.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")

    def plot_prediction_residuals(self) -> None:
        residuals = self.model_results["predictions"]["y_test"] - self.model_results["predictions"]["y_test_pred"]

        fig = plt.figure(figsize=(12, 7))
        sns.histplot(residuals, kde=True, color=COLORS[4], bins=50)
        plt.title(
            f"Distribution of Prediction Residuals ({self.model_display_names[self.current_model_name]} - Test Set)",
            fontsize=16, fontweight="bold",
        )
        plt.xlabel("Residual (Actual - Predicted) (mm)", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.axvline(0, color="r", linestyle="--", lw=2)
        plt.grid(True, alpha=0.3)
        plt.savefig(self.output_dir / f"{self.current_model_name}_prediction_residuals.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")

    def plot_feature_importance(self) -> None:
        if self.model is None:
            logger.warning("Model not trained. Cannot plot feature importance.")
            return

        if self.current_model_name == "elasticnet" and hasattr(self.model, "coef_"):
            importance = np.abs(self.model.coef_)
            feature_names = self.flat_feature_names
            indices = np.argsort(importance)[::-1][:20]

            fig = plt.figure(figsize=(12, 10))
            plt.title("Top 20 Feature Importances for ElasticNet", fontsize=16, fontweight="bold")
            plt.barh(range(len(indices)), importance[indices][::-1], color=COLORS[0], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
            plt.xlabel("Absolute Coefficient Value", fontsize=14)
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(self.output_dir / "elasticnet_feature_importance.png", dpi=400, bbox_inches="tight")
            plt.close(fig)
            plt.close("all")
            return

        if self.current_model_name in ["lightgbm", "xgboost"] and hasattr(self.model, "feature_importances_"):
            importance = np.array(self.model.feature_importances_, dtype=float)
            feature_names = self.flat_feature_names
            importance = importance[:len(feature_names)]
            indices = np.argsort(importance)[::-1][:20]

            fig = plt.figure(figsize=(12, 10))
            plt.title(f"Top 20 Feature Importances for {self.model_display_names[self.current_model_name]}", fontsize=16, fontweight="bold")
            plt.barh(range(len(indices)), importance[indices][::-1], color=COLORS[1], align="center")
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices][::-1])
            plt.xlabel("Gain-based Importance", fontsize=14)
            plt.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{self.current_model_name}_feature_importance.png", dpi=400, bbox_inches="tight")
            plt.close(fig)
            plt.close("all")
            return

        if self.current_model_name == "bilstm":
            placeholder_path = self.output_dir / "bilstm_feature_importance.txt"
            placeholder_path.write_text(
                "Feature importance plot is not implemented for BiLSTM in this pipeline.\n"
                "Performance plots and model comparisons are available.\n",
                encoding="utf-8",
            )
            return

        logger.warning("Feature importance could not be generated for current model.")

    def plot_shap_analysis(self) -> None:
        if self.current_model_name not in ["elasticnet", "lightgbm", "xgboost"]:
            logger.info(f"Skipping SHAP for {self.current_model_name}.")
            return

        logger.info(f"Generating SHAP analysis for {self.current_model_name}...")
        X_test_input = self.X_test_flat

        if self.current_model_name == "elasticnet" and self.scaler is not None:
            X_test_input = self.scaler.transform(self.X_test_flat)

        feature_names = self.flat_feature_names
        X_df = pd.DataFrame(X_test_input, columns=feature_names)

        try:
            if self.current_model_name == "elasticnet":
                explainer = shap.LinearExplainer(self.model, X_test_input)
                shap_values = explainer.shap_values(X_test_input)
            else:
                explainer = shap.TreeExplainer(self.model)
                shap_values = explainer.shap_values(X_test_input)
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]

            plt.figure(figsize=(12, 10))
            shap.summary_plot(shap_values, X_df, show=False, plot_size=None, max_display=20)
            plt.title(f"Global Impact - {self.model_display_names[self.current_model_name]}", fontsize=18)
            plt.savefig(self.output_dir / f"{self.current_model_name}_shap_beeswarm.png", dpi=400, bbox_inches="tight")
            plt.close()

            current_shap_values = np.array(shap_values)
            pixel_dir = self.output_dir / "shap_local_pixels"
            pixel_dir.mkdir(exist_ok=True)

            y_test_true = self.model_results["predictions"]["y_test"]
            top_pixel_indices = np.argsort(y_test_true)[::-1]

            pixels_plotted = 0
            for pixel_idx in top_pixel_indices:
                if pixels_plotted >= 6:
                    break

                single_shap_values = current_shap_values[pixel_idx]
                mask = single_shap_values != 0
                if np.sum(mask) == 0:
                    continue

                filtered_values = single_shap_values[mask]
                filtered_names = np.array(feature_names)[mask]

                top_20_indices = np.argsort(np.abs(filtered_values))[::-1][:20]
                final_values = filtered_values[top_20_indices]
                final_names = filtered_names[top_20_indices]

                sort_idx = np.argsort(np.abs(final_values))
                final_values = final_values[sort_idx]
                final_names = final_names[sort_idx]

                local_height = max(6, len(final_names) * 0.45)
                plt.figure(figsize=(12, local_height))
                bars = plt.barh(range(len(final_names)), final_values, color="#5BC0DE", align="center",
                                edgecolor="black", alpha=0.8)
                plt.yticks(range(len(final_names)), final_names, fontsize=12)
                plt.title(f"Local Explanation (Sample {pixel_idx}) - {self.model_display_names[self.current_model_name]}", fontsize=16, pad=20)
                plt.xlabel("SHAP Value", fontsize=13)
                plt.axvline(0, color="black", linewidth=1.2)
                plt.grid(True, axis="x", alpha=0.3, linestyle="--")

                max_abs_val = np.max(np.abs(final_values))
                padding_factor = 1.35
                plt.xlim(-max_abs_val * padding_factor, max_abs_val * padding_factor)

                for bar in bars:
                    width = bar.get_width()
                    label_offset = max_abs_val * 0.03 if max_abs_val > 0 else 0.05
                    label_x_pos = width + (label_offset if width >= 0 else -label_offset)
                    plt.text(
                        label_x_pos,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.3f}",
                        va="center",
                        ha="left" if width >= 0 else "right",
                        fontsize=10,
                        fontweight="bold",
                    )

                plt.tight_layout()
                plt.savefig(pixel_dir / f"{self.current_model_name}_shap_sample_{pixel_idx}_explanation.png",
                            dpi=400, bbox_inches="tight")
                plt.close()
                pixels_plotted += 1

        except Exception as e:
            logger.error(f"Failed SHAP generation: {e}", exc_info=True)

    # =========================================================
    # UNCERTAINTY / PIXEL PLOTS
    # =========================================================
    def _get_confidence_intervals(self, model, X, model_name: str) -> np.ndarray:
        y_test_pred = self.model_results["predictions"]["y_test_pred"]
        y_test_true = self.model_results["predictions"]["y_test"]

        if model_name in ["elasticnet", "lightgbm", "xgboost"]:
            sigma = np.std(y_test_true - y_test_pred)
            return 1.96 * sigma * np.ones(len(X))

        if model_name == "bilstm":
            logger.info("Computing MC Dropout CI for BiLSTM...")
            mc_predictions = []
            for _ in range(20):
                preds = model(X, training=True).numpy().ravel()
                mc_predictions.append(preds)
            sigma = np.std(mc_predictions, axis=0)
            return 1.96 * sigma

        return np.zeros(len(X))

    def plot_pixel_timeseries_with_ci(self, start_pixel_offset: int = 0, max_pixels: int = 6, file_suffix: str = "set_1") -> None:
        logger.info(f"Generating pixel grid for {self.current_model_name} - {file_suffix}...")
        data = self.test_data
        past_months = self.optimal_time_steps
        num_timesteps, num_grid_cells, _ = data.shape

        X_input = self.X_test_seq if self.current_model_name == "bilstm" else self.X_test_flat
        ci_values = self._get_confidence_intervals(self.model, X_input, self.current_model_name)

        y_test_pred = self.model_results["predictions"]["y_test_pred"]
        y_test_actual = self.model_results["predictions"]["y_test"]

        pixel_series = {
            grid_idx: {"actual": [], "pred": [], "ci": [], "time_axis": []}
            for grid_idx in range(num_grid_cells)
        }

        current_idx = 0
        for grid_idx in range(num_grid_cells):
            for t in range(past_months, num_timesteps - self.future_steps + 1):
                seq = data[t - past_months:t, grid_idx, self.feature_indices]
                target = data[t:t + self.future_steps, grid_idx, self.current_target_index]

                if np.isnan(seq).any() or np.isnan(target).any():
                    continue
                if current_idx >= len(y_test_actual):
                    break

                pixel_series[grid_idx]["actual"].append(y_test_actual[current_idx])
                pixel_series[grid_idx]["pred"].append(y_test_pred[current_idx])
                pixel_series[grid_idx]["ci"].append(ci_values[current_idx])
                pixel_series[grid_idx]["time_axis"].append(t)
                current_idx += 1

        nonempty_pixels = [idx for idx in range(num_grid_cells) if len(pixel_series[idx]["actual"]) > 0]
        selected_pixels = nonempty_pixels[start_pixel_offset:start_pixel_offset + max_pixels]

        if len(selected_pixels) == 0:
            logger.warning(f"No valid pixels found for {file_suffix}.")
            return

        cols = 2
        rows = int(np.ceil(len(selected_pixels) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), sharex=False)
        axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes])

        for plot_idx, grid_idx in enumerate(selected_pixels):
            ax = axes_flat[plot_idx]
            p_actual = pixel_series[grid_idx]["actual"]
            p_pred = pixel_series[grid_idx]["pred"]
            p_ci = pixel_series[grid_idx]["ci"]
            time_axis = pixel_series[grid_idx]["time_axis"]

            ax.plot(time_axis, p_actual, "k-o", label="Actual", markersize=4, alpha=0.7)
            ax.plot(time_axis, p_pred, color="#D9534F", linestyle="--", label="Predicted", linewidth=2)

            lower_bound = np.array(p_pred) - np.array(p_ci)
            upper_bound = np.array(p_pred) + np.array(p_ci)
            ax.fill_between(time_axis, lower_bound, upper_bound, color="#D9534F", alpha=0.15, label="95% CI")

            ax.set_title(f"Pixel {grid_idx}", fontsize=14, fontweight="bold")
            ax.set_ylabel(self._get_target_plot_label())
            ax.set_xlabel("Months")
            ax.grid(True, alpha=0.2)
            if plot_idx == 0:
                ax.legend(loc="upper right", fontsize=10)

        for i in range(len(selected_pixels), len(axes_flat)):
            fig.delaxes(axes_flat[i])

        plt.tight_layout()
        output_path = self.output_dir / f"{self.current_model_name}_pixel_grid_6_plots_{file_suffix}.png"
        plt.savefig(output_path, dpi=500, bbox_inches="tight")
        plt.close(fig)

    # =========================================================
    # REPORTING
    # =========================================================
    def print_comprehensive_results(self, scenario_name: str, persistence_results: Optional[Dict[str, Any]] = None) -> None:
        logger.info("\n" + "=" * 110)
        logger.info(f"{self.model_display_names[self.current_model_name].upper()} RESULTS - SCENARIO: {scenario_name}")
        logger.info("=" * 110)
        logger.info(f"Training Data:    {self.train_data_path}")
        logger.info(f"Validation Data: {self.val_data_path}")
        logger.info(f"Test Data:       {self.test_data_path}")
        logger.info(f"Target feature:  {self.current_target_name}")
        logger.info(f"Optimal past steps: {self.optimal_time_steps}")
        logger.info(f"Model hyperparameters: {self.best_params}")
        logger.info(f"Scenario features: {[self.feature_names[idx] for idx in self.feature_indices]}")
        logger.info("-" * 110)
        logger.info(f"{'Split':<12}{'R² Score':<12}{'RMSE (mm)':<12}{'MAE (mm)':<12}")
        logger.info("-" * 110)
        logger.info(f"{'Training':<12}{self.model_results['train']['r2']:<12.4f}{self.model_results['train']['rmse']:<12.4f}{self.model_results['train']['mae']:<12.4f}")
        logger.info(f"{'Validation':<12}{self.model_results['val']['r2']:<12.4f}{self.model_results['val']['rmse']:<12.4f}{self.model_results['val']['mae']:<12.4f}")
        logger.info(f"{'Test':<12}{self.model_results['test']['r2']:<12.4f}{self.model_results['test']['rmse']:<12.4f}{self.model_results['test']['mae']:<12.4f}")

        if persistence_results is not None:
            skill = self._compute_skill_against_persistence(self.model_results, persistence_results)
            logger.info("-" * 110)
            logger.info(f"Persistence Test R²:  {persistence_results['test']['r2']:.4f}")
            logger.info(f"ΔR² vs Persistence:   {skill['delta_r2']:.4f}")
            logger.info(f"RMSE Reduction:       {skill['rmse_reduction']:.4f}")
            logger.info(f"MAE Reduction:        {skill['mae_reduction']:.4f}")

        if self.cv_results:
            best_cv = self.cv_results["best_result"]
            logger.info("-" * 110)
            logger.info("LEAVE-ONE-CITY-OUT CV SUMMARY")
            logger.info(f"Best past steps: {best_cv['past_months']} | Avg R²: {best_cv['avg_r2']:.4f} | Avg RMSE: {best_cv['avg_rmse']:.4f} | Avg MAE: {best_cv['avg_mae']:.4f}")
            logger.info(f"{'Held-out City':<28}{'R² Score':<12}{'RMSE (mm)':<12}{'MAE (mm)':<12}{'Train N':<12}{'Test N':<12}")
            for fold in best_cv["folds"]:
                logger.info(
                    f"{fold['held_out_group']:<28}{fold['r2']:<12.4f}{fold['rmse']:<12.4f}{fold['mae']:<12.4f}{fold['n_train_samples']:<12}{fold['n_test_samples']:<12}"
                )
        logger.info("=" * 110)

    def save_model_and_results(self, scenario_name: str, persistence_results: Optional[Dict[str, Any]] = None) -> None:
        if self.current_model_name == "bilstm":
            self.model.save(self.output_dir / "bilstm_model.keras")
            with open(self.output_dir / "bilstm_scaler.pickle", "wb") as f:
                pickle.dump(self.scaler, f)
        else:
            model_and_scaler = {"model": self.model, "scaler": self.scaler}
            with open(self.output_dir / f"{self.current_model_name}_model_and_scaler.pickle", "wb") as f:
                pickle.dump(model_and_scaler, f)

        skill = None
        if persistence_results is not None:
            skill = self._compute_skill_against_persistence(self.model_results, persistence_results)

        payload = {
            "model_name": self.current_model_name,
            "scenario_name": scenario_name,
            "target_name": self.current_target_name,
            "model_results": self.model_results,
            "cv_results": self.cv_results,
            "optimal_time_steps": self.optimal_time_steps,
            "best_hyperparameters": self.best_params,
            "selected_features": [self.feature_names[idx] for idx in self.feature_indices],
            "future_steps_forecasted": self.future_steps,
            "persistence_results": persistence_results,
            "skill_vs_persistence": skill,
            "incity_results": self.incity_results_dict,
        }

        with open(self.output_dir / f"{self.current_model_name}_results.pickle", "wb") as f:
            pickle.dump(payload, f)

    def save_scenario_summary_text(self, scenario_name: str, persistence_results: Optional[Dict[str, Any]] = None) -> None:
        lines = []
        lines.append("=" * 110)
        lines.append(f"MODEL: {self.model_display_names[self.current_model_name]}")
        lines.append(f"SCENARIO: {scenario_name}")
        lines.append(f"TARGET FEATURE: {self.current_target_name}")
        lines.append("=" * 110)
        lines.append(f"Training Data:    {self.train_data_path}")
        lines.append(f"Validation Data: {self.val_data_path}")
        lines.append(f"Test Data:       {self.test_data_path}")
        lines.append(f"Future steps forecasted: {self.future_steps}")
        lines.append(f"Optimal past steps: {self.optimal_time_steps}")
        lines.append(f"Scenario features: {[self.feature_names[idx] for idx in self.feature_indices]}")
        lines.append(f"Best hyperparameters: {self.best_params}")
        lines.append("")

        lines.append("MODEL PERFORMANCE SUMMARY")
        lines.append("-" * 60)
        lines.append(f"{'Split':<12}{'R² Score':<12}{'RMSE (mm)':<12}{'MAE (mm)':<12}")
        lines.append("-" * 60)
        lines.append(f"{'Training':<12}{self.model_results['train']['r2']:<12.4f}{self.model_results['train']['rmse']:<12.4f}{self.model_results['train']['mae']:<12.4f}")
        lines.append(f"{'Validation':<12}{self.model_results['val']['r2']:<12.4f}{self.model_results['val']['rmse']:<12.4f}{self.model_results['val']['mae']:<12.4f}")
        lines.append(f"{'Test':<12}{self.model_results['test']['r2']:<12.4f}{self.model_results['test']['rmse']:<12.4f}{self.model_results['test']['mae']:<12.4f}")
        lines.append("")

        if persistence_results is not None:
            skill = self._compute_skill_against_persistence(self.model_results, persistence_results)
            lines.append("PERSISTENCE BASELINE SUMMARY")
            lines.append("-" * 60)
            lines.append(f"{'Split':<12}{'R² Score':<12}{'RMSE (mm)':<12}{'MAE (mm)':<12}")
            lines.append("-" * 60)
            lines.append(f"{'Training':<12}{persistence_results['train']['r2']:<12.4f}{persistence_results['train']['rmse']:<12.4f}{persistence_results['train']['mae']:<12.4f}")
            lines.append(f"{'Validation':<12}{persistence_results['val']['r2']:<12.4f}{persistence_results['val']['rmse']:<12.4f}{persistence_results['val']['mae']:<12.4f}")
            lines.append(f"{'Test':<12}{persistence_results['test']['r2']:<12.4f}{persistence_results['test']['rmse']:<12.4f}{persistence_results['test']['mae']:<12.4f}")
            lines.append("")
            lines.append("MODEL VS PERSISTENCE ON TEST SET")
            lines.append("-" * 60)
            lines.append(f"Delta R²: {skill['delta_r2']:.4f}")
            lines.append(f"RMSE Reduction: {skill['rmse_reduction']:.4f}")
            lines.append(f"MAE Reduction: {skill['mae_reduction']:.4f}")
            lines.append("")

        if self.incity_results_dict:
            lines.append("IN-CITY TEMPORAL SPLIT SUMMARY (80% earliest / 20% latest)")
            lines.append("-" * 90)
            lines.append(f"{'City Name':<25}{'R² Score':<15}{'RMSE (mm)':<15}{'MAE (mm)':<15}{'Total Samples':<15}")
            lines.append("-" * 90)
            for city, res in self.incity_results_dict.items():
                lines.append(
                    f"{city:<25}{res['r2']:<15.4f}{res['rmse']:<15.4f}{res['mae']:<15.4f}{res['samples']:<15}"
                )
            lines.append("")

        if self.cv_results:
            best_cv = self.cv_results["best_result"]
            lines.append("LEAVE-ONE-CITY-OUT CV SUMMARY")
            lines.append("-" * 110)
            lines.append(f"Best past steps: {best_cv['past_months']} | Average R²: {best_cv['avg_r2']:.4f} | Average RMSE: {best_cv['avg_rmse']:.4f} | Average MAE: {best_cv['avg_mae']:.4f}")
            lines.append("-" * 110)
            for fold in best_cv["folds"]:
                lines.append(
                    f"{fold['held_out_group']:<28}{fold['r2']:<12.4f}{fold['rmse']:<12.4f}{fold['mae']:<12.4f}{fold['n_train_samples']:<12}{fold['n_test_samples']:<12}"
                )

        summary_path = self.output_dir / f"{self.current_model_name}_scenario_summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")

    def save_scenario_model_comparison(self, scenario_name: str) -> None:
        scenario_dir = self.output_root_dir / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 150)
        lines.append(f"MODEL COMPARISON FOR SCENARIO: {scenario_name}")
        lines.append("=" * 150)
        lines.append(f"{'Model':<15}{'Best Past Steps':<18}{'CV Avg R2':<12}{'CV Avg RMSE':<14}{'Test R2':<12}{'ΔR2 vs Pers':<14}{'RMSE Red.':<12}")
        lines.append("-" * 150)

        for model_name in self.model_names:
            result = self.scenario_results[scenario_name].get(model_name)
            if not result or "cv_results" not in result or "best_result" not in result["cv_results"]:
                lines.append(f"{self.model_display_names[model_name]:<15} No valid CV results found.")
                continue

            best_cv = result["cv_results"]["best_result"]
            skill = result.get("skill_vs_persistence", {"delta_r2": np.nan, "rmse_reduction": np.nan})
            lines.append(
                f"{self.model_display_names[model_name]:<15}"
                f"{result['optimal_time_steps']:<18}"
                f"{best_cv['avg_r2']:<12.4f}"
                f"{best_cv['avg_rmse']:<14.4f}"
                f"{result['model_results']['test']['r2']:<12.4f}"
                f"{skill['delta_r2']:<14.4f}"
                f"{skill['rmse_reduction']:<12.4f}"
            )

        comparison_path = scenario_dir / "model_comparison_summary.txt"
        comparison_path.write_text("\n".join(lines), encoding="utf-8")

    # =========================================================
    # IN-CITY EVALUATION
    # =========================================================
    def run_incity_scenario(self, scenario_name: str, model_name: Optional[str] = None) -> None:
        if model_name is None:
            model_name = self.current_model_name

        logger.info("\n" + "!" * 90)
        logger.info(f"STARTING IN-CITY TEMPORAL EVALUATION | MODEL: {self.model_display_names[model_name]} | SCENARIO: {scenario_name}")
        logger.info("!" * 90)

        self.incity_results_dict = {}

        incity_out_dir = self.output_root_dir / "InCity_Results" / scenario_name / model_name
        incity_out_dir.mkdir(parents=True, exist_ok=True)
        summary_lines = []

        for city_name, bundle in self.city_datasets.items():
            try:
                city_raw_data = bundle["data"]
                city_pixel_labels = bundle["pixel_labels"]

                X_seq, y_flat, _ = self.create_sequences(
                    city_raw_data,
                    past_months=self.optimal_time_steps,
                    pixel_group_labels=city_pixel_labels,
                    return_metadata=True,
                )

                if len(X_seq) < 10:
                    logger.warning(f"Skipping {city_name}: only {len(X_seq)} samples.")
                    continue

                X_train, X_test, y_train, y_test = self._create_citywise_temporal_split(
                    X_seq, y_flat, test_fraction=0.2
                )

                model, _, best_params, y_pred = self._fit_and_predict_model(
                    model_name,
                    X_train,
                    y_train.ravel(),
                    X_test,
                )

                r2 = r2_score(y_test.ravel(), y_pred)
                rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred))
                mae = mean_absolute_error(y_test.ravel(), y_pred)

                self.incity_results_dict[city_name] = {
                    "r2": r2,
                    "rmse": rmse,
                    "mae": mae,
                    "samples": len(X_seq),
                    "best_params": best_params,
                }

                logger.info(f"City: {city_name:<25} | In-City Temporal R²: {r2:.4f} | RMSE: {rmse:.4f} | N={len(X_seq)}")
                summary_lines.append(
                    f"{city_name:<25} | R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f} | Samples: {len(X_seq)}"
                )

                del model
                gc.collect()

            except Exception as e:
                logger.error(f"Error in in-city evaluation for {city_name}: {e}", exc_info=True)

        report_path = incity_out_dir / "incity_performance_report.txt"
        report_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # =========================================================
    # PIPELINE PER SCENARIO
    # =========================================================
    def run_single_scenario_pipeline(self, scenario_name: str) -> Dict[str, Any]:
        scenario_results = {}

        for model_name in self.model_names:
            self.current_model_name = model_name
            self.current_scenario_name = scenario_name
            self.output_dir = self.output_root_dir / scenario_name / model_name
            self.output_dir.mkdir(parents=True, exist_ok=True)

            self.model = None
            self.scaler = None
            self.model_results = {}
            self.cv_results = {}
            self.optimal_time_steps = 0
            self.feature_indices = []
            self.X_test_flat = None
            self.X_test_seq = None
            self.flat_feature_names = None
            self.best_params = {}
            self.sample_weight_summary = {}
            self.incity_results_dict = {}

            self._set_scenario_features(scenario_name)

            logger.info("\n" + "-" * 120)
            logger.info(f"RUNNING MODEL: {self.model_display_names[model_name]} | SCENARIO: {scenario_name}")
            logger.info("-" * 120)

            if self._is_completed_model_run(scenario_name, model_name):
                logger.info(
                    f"Skipping already completed run for scenario='{scenario_name}', "
                    f"model='{model_name}'. Loading saved results from disk."
                )
                loaded_result = self._load_completed_model_run(scenario_name, model_name)
                scenario_results[model_name] = loaded_result
                gc.collect()
                continue

            if model_name == "elasticnet":
                self.train_elasticnet_model()
            elif model_name == "lightgbm":
                self.train_lightgbm_model()
            elif model_name == "xgboost":
                self.train_xgboost_model()
            elif model_name == "bilstm":
                self.train_bilstm_model()
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            persistence_results = self.evaluate_persistence_model(self.optimal_time_steps)
            skill = self._compute_skill_against_persistence(self.model_results, persistence_results)

            self.run_incity_scenario(scenario_name, model_name=model_name)

            logger.info(f"\nCreating figures for {self.model_display_names[model_name]}...")
            self.plot_performance_summary()
            self.plot_actual_vs_predicted()
            self.plot_prediction_residuals()
            self.plot_feature_importance()
            self.plot_shap_analysis()
            self.plot_pixel_timeseries_with_ci(start_pixel_offset=0, max_pixels=6, file_suffix="set_1")
            self.plot_pixel_timeseries_with_ci(start_pixel_offset=6, max_pixels=6, file_suffix="set_2")

            self.print_comprehensive_results(scenario_name, persistence_results=persistence_results)
            self.save_model_and_results(scenario_name=scenario_name, persistence_results=persistence_results)
            self.save_scenario_summary_text(scenario_name=scenario_name, persistence_results=persistence_results)

            scenario_results[model_name] = {
                "model_name": model_name,
                "scenario_name": scenario_name,
                "target_name": self.current_target_name,
                "selected_features": [self.feature_names[idx] for idx in self.feature_indices],
                "model_results": self.model_results,
                "cv_results": self.cv_results,
                "incity_results": self.incity_results_dict,
                "optimal_time_steps": self.optimal_time_steps,
                "best_hyperparameters": self.best_params,
                "output_dir": str(self.output_dir),
                "skill_vs_persistence": skill,
            }

            gc.collect()

        return scenario_results

    # =========================================================
    # BEST SCENARIO / GLOBAL REPORTS
    # =========================================================
    def select_best_scenario(self) -> Tuple[str, str, Dict[str, Any]]:
        best_scenario_name = None
        best_model_name = None
        best_result = None

        best_delta_r2 = -np.inf
        best_test_r2 = -np.inf

        for scenario_name, model_dict in self.scenario_results.items():
            for model_name, result in model_dict.items():
                skill = result.get("skill_vs_persistence", {})
                delta_r2 = skill.get("delta_r2", -np.inf)
                test_r2 = result["model_results"]["test"]["r2"]

                if (delta_r2 > best_delta_r2) or (np.isclose(delta_r2, best_delta_r2) and test_r2 > best_test_r2):
                    best_delta_r2 = delta_r2
                    best_test_r2 = test_r2
                    best_scenario_name = scenario_name
                    best_model_name = model_name
                    best_result = result

        return best_scenario_name, best_model_name, best_result

    def plot_best_scenario_vs_persistence(
        self,
        scenario_name: str,
        model_results: Dict[str, Any],
        persistence_results: Dict[str, Any],
        output_dir: Path,
        model_name: Optional[str] = None,
    ) -> None:
        if model_name is None:
            model_name = self.current_model_name

        metrics = ["r2", "rmse", "mae"]
        metric_names = ["R² Score", "RMSE (mm)", "MAE (mm)"]

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            values = [model_results["test"][metric], persistence_results["test"][metric]]
            bars = ax.bar([self.model_display_names[model_name], "Persistence"], values, color=COLORS[:2], alpha=0.85)
            ax.set_title(f"{scenario_name} - Test {metric_name}", fontsize=16, fontweight="bold")
            ax.set_ylabel(metric_name, fontsize=14)
            ax.grid(True, alpha=0.3, axis="y")

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.3f}",
                        ha="center", va="bottom", fontsize=12)

        plt.suptitle(f"{self.model_display_names[model_name]} vs Persistence - Best Scenario: {scenario_name}",
                     fontsize=20, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_dir / f"{model_name}_best_scenario_vs_persistence.png", dpi=400, bbox_inches="tight")
        plt.close(fig)
        plt.close("all")

    def run_best_scenario_persistence_comparison(self, best_scenario_name: str, best_model_name: str) -> None:
        best_result = self.scenario_results[best_scenario_name][best_model_name]
        best_time_steps = best_result["optimal_time_steps"]

        self.current_model_name = best_model_name
        self.current_scenario_name = best_scenario_name
        self.output_dir = self.output_root_dir / best_scenario_name / best_model_name
        self._set_scenario_features(best_scenario_name)
        self.optimal_time_steps = best_time_steps
        self.model_results = best_result["model_results"]
        self.cv_results = best_result["cv_results"]
        self.best_params = best_result["best_hyperparameters"]
        self.incity_results_dict = best_result.get("incity_results", {})

        self._build_flat_feature_names()

        persistence_results = self.evaluate_persistence_model(best_time_steps)
        skill = self._compute_skill_against_persistence(best_result["model_results"], persistence_results)

        payload = {
            "best_scenario_name": best_scenario_name,
            "best_model_name": best_model_name,
            "best_scenario_optimal_time_steps": best_time_steps,
            "target_name": self.current_target_name,
            "model_results": best_result["model_results"],
            "persistence_results": persistence_results,
            "best_cv": best_result["cv_results"]["best_result"],
            "skill_vs_persistence": skill,
        }

        with open(self.output_dir / f"{best_model_name}_persistence_comparison_results.pickle", "wb") as f:
            pickle.dump(payload, f)

        self.plot_best_scenario_vs_persistence(
            scenario_name=best_scenario_name,
            model_results=best_result["model_results"],
            persistence_results=persistence_results,
            output_dir=self.output_dir,
            model_name=best_model_name,
        )
        self.save_scenario_summary_text(scenario_name=best_scenario_name, persistence_results=persistence_results)

        logger.info("\nBEST SCENARIO VS PERSISTENCE")
        logger.info("=" * 100)
        logger.info(f"Best scenario: {best_scenario_name}")
        logger.info(f"Best model: {self.model_display_names[best_model_name]}")
        logger.info(f"Target: {self.current_target_name}")
        logger.info(f"Optimal past steps: {best_time_steps}")
        logger.info(f"Model Test R²: {best_result['model_results']['test']['r2']:.4f}")
        logger.info(f"Persistence Test R²: {persistence_results['test']['r2']:.4f}")
        logger.info(f"ΔR²: {skill['delta_r2']:.4f}")
        logger.info(f"RMSE reduction: {skill['rmse_reduction']:.4f}")
        logger.info(f"MAE reduction: {skill['mae_reduction']:.4f}")
        logger.info("=" * 100)

    def save_global_summary(self, best_scenario_name: str, best_model_name: str, best_result: Dict[str, Any]) -> None:
        lines = []
        lines.append("=" * 170)
        lines.append("MULTI-MODEL SIX-SCENARIO SUMMARY")
        lines.append("=" * 170)
        lines.append(f"Training Data:    {self.train_data_path}")
        lines.append(f"Validation Data: {self.val_data_path}")
        lines.append(f"Test Data:       {self.test_data_path}")
        lines.append("")
        lines.append(
            f"{'Scenario':<38}{'Model':<15}{'Target':<14}{'Best Past Steps':<18}"
            f"{'CV Avg R²':<12}{'CV Avg RMSE':<14}{'CV Avg MAE':<12}"
            f"{'Test R²':<12}{'Test RMSE':<12}{'Test MAE':<12}{'ΔR² vs Pers':<14}"
        )
        lines.append("-" * 170)

        for scenario_name, model_dict in self.scenario_results.items():
            for model_name, result in model_dict.items():
                best_cv = result.get("cv_results", {}).get("best_result", {"avg_r2": 0, "avg_rmse": 0, "avg_mae": 0})
                skill = result.get("skill_vs_persistence", {"delta_r2": np.nan})
                lines.append(
                    f"{scenario_name:<38}"
                    f"{self.model_display_names[model_name]:<15}"
                    f"{result.get('target_name', 'N/A'):<14}"
                    f"{result['optimal_time_steps']:<18}"
                    f"{best_cv['avg_r2']:<12.4f}"
                    f"{best_cv['avg_rmse']:<14.4f}"
                    f"{best_cv['avg_mae']:<12.4f}"
                    f"{result['model_results']['test']['r2']:<12.4f}"
                    f"{result['model_results']['test']['rmse']:<12.4f}"
                    f"{result['model_results']['test']['mae']:<12.4f}"
                    f"{skill['delta_r2']:<14.4f}"
                )

        lines.append("")
        lines.append("=" * 170)
        lines.append("BEST MODEL / SCENARIO COMBINATION")
        lines.append("=" * 170)
        lines.append(f"Best Scenario: {best_scenario_name}")
        lines.append(f"Best Model: {self.model_display_names[best_model_name]}")
        lines.append(f"Target feature: {best_result.get('target_name', 'N/A')}")
        lines.append(f"Optimal past time steps: {best_result['optimal_time_steps']}")
        lines.append(f"Selected features: {best_result['selected_features']}")
        lines.append(f"Best CV average R²: {best_result['cv_results']['best_result']['avg_r2']:.4f}")
        lines.append(f"Test R²: {best_result['model_results']['test']['r2']:.4f}")
        lines.append(f"Test RMSE: {best_result['model_results']['test']['rmse']:.4f}")
        lines.append(f"Test MAE: {best_result['model_results']['test']['mae']:.4f}")
        lines.append(f"ΔR² vs Persistence: {best_result['skill_vs_persistence']['delta_r2']:.4f}")

        summary_path = self.output_root_dir / "all_scenarios_summary.txt"
        summary_path.write_text("\n".join(lines), encoding="utf-8")

        with open(self.output_root_dir / "all_scenarios_summary.pickle", "wb") as f:
            pickle.dump(self.scenario_results, f)

    def save_final_comparative_report(self) -> None:
        report_path = self.output_root_dir / "final_comprehensive_comparative_report.txt"
        lines = []
        lines.append("=" * 130)
        lines.append("FINAL COMPREHENSIVE SUBSIDENCE FORECASTING REPORT")
        lines.append("=" * 130)
        lines.append(f"Training Data:    {self.train_data_path}")
        lines.append(f"Validation Data: {self.val_data_path}")
        lines.append(f"Test Data:       {self.test_data_path}\n")

        for scenario_name, models_data in self.scenario_results.items():
            lines.append(f"SCENARIO: {scenario_name.upper()}")
            lines.append("-" * 70)

            for model_name, data in models_data.items():
                lines.append(f"MODEL: {self.model_display_names[model_name]}")
                lines.append(f"  - Target Feature: {data.get('target_name', 'N/A')}")
                lines.append(f"  - Optimal Past Steps: {data['optimal_time_steps']}")
                lines.append(f"  - Best CV Hyperparameters: {data.get('best_hyperparameters', 'N/A')}")

                cv_best = data.get("cv_results", {}).get("best_result", {"avg_r2": 0, "avg_rmse": 0, "avg_mae": 0})
                lines.append(f"  - [City-CV Avg] R²: {cv_best.get('avg_r2', 0):.4f} | RMSE: {cv_best.get('avg_rmse', 0):.4f} | MAE: {cv_best.get('avg_mae', 0):.4f}")

                res = data["model_results"]
                lines.append(f"  - [Holdout] R²: {res['test']['r2']:.4f} | RMSE: {res['test']['rmse']:.4f} | MAE: {res['test']['mae']:.4f}")

                self._set_scenario_features(scenario_name)
                pers = self.evaluate_persistence_model(data["optimal_time_steps"])
                skill = data.get("skill_vs_persistence", {})
                lines.append(f"  - [Persistence] R²: {pers['test']['r2']:.4f} | RMSE: {pers['test']['rmse']:.4f}")
                lines.append(f"  - [Skill] ΔR² vs Persistence: {skill.get('delta_r2', np.nan):.4f} | RMSE Reduction: {skill.get('rmse_reduction', np.nan):.4f}\n")

            lines.append("-" * 70 + "\n")

        best_sc, best_mod, best_res = self.select_best_scenario()
        lines.append("=" * 130)
        lines.append(f"OVERALL WINNER: Scenario '{best_sc}' using Model '{self.model_display_names[best_mod]}'")
        lines.append(f"Final Test R²: {best_res['model_results']['test']['r2']:.4f}")
        lines.append(f"ΔR² vs Persistence: {best_res['skill_vs_persistence']['delta_r2']:.4f}")
        lines.append("=" * 130)

        report_path.write_text("\n".join(lines), encoding="utf-8")

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    def run_complete_pipeline(self) -> None:
        logger.info("Starting subsidence forecasting pipeline...")

        try:
            if len(self.model_names) == 0:
                raise ValueError("No models are enabled. Please enable at least one model.")

            self.load_datasets()

            scenario_order = [
                "subsidence_history_only_cumulative",
                "subsidence_history_only_differential",
                "combined_cumulative",
                "combined_differential",
                "environmental_only_cumulative",
                "environmental_only_differential",
            ]

            for scenario_name in scenario_order:
                logger.info("\n" + "#" * 130)
                logger.info(f"PROCESSING SCENARIO: {scenario_name}")
                logger.info("#" * 130)

                self.scenario_results[scenario_name] = self.run_single_scenario_pipeline(scenario_name)
                self.save_scenario_model_comparison(scenario_name)
                gc.collect()

            best_scenario_name, best_model_name, best_result = self.select_best_scenario()
            logger.info("\n" + "*" * 130)
            logger.info(f"WINNER (persistence-aware): {self.model_display_names[best_model_name]} | Scenario: {best_scenario_name}")
            logger.info("*" * 130)

            self.run_best_scenario_persistence_comparison(best_scenario_name, best_model_name)
            self.save_global_summary(best_scenario_name, best_model_name, best_result)
            self.save_final_comparative_report()

            logger.info("\nPipeline completed successfully.")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise


if __name__ == "__main__":
    BASE_DATA_DIR = r"C:\Users\DFMRendering\Desktop\subsidence\Revise\Data"

    train_files = [
        str(Path(BASE_DATA_DIR) / "Rafsanjan" / "Merged_Dataset_3D.npz"),
        str(Path(BASE_DATA_DIR) / "Qazvin-Alborz-Tehran" / "Merged_Dataset_3D.npz"),
        str(Path(BASE_DATA_DIR) / "Marvdasht" / "Merged_Dataset_3D.npz"),
        str(Path(BASE_DATA_DIR) / "Isfahan" / "Merged_Dataset_3D.npz"),
    ]

    val_file = [
        str(Path(BASE_DATA_DIR) / "Semnan" / "Merged_Dataset_3D.npz"),
        str(Path(BASE_DATA_DIR) / "Jiroft" / "Merged_Dataset_3D.npz"),
    ]

    test_file = [
        str(Path(BASE_DATA_DIR) / "Lake Urmia  Tabriz" / "Merged_Dataset_3D.npz"),
        str(Path(BASE_DATA_DIR) / "Nishapur" / "Merged_Dataset_3D.npz"),
    ]

    predictor = ElasticNetSubsidencePredictor(
        train_data_path=train_files,
        val_data_path=val_file,
        test_data_path=test_file,
        random_state=42,
    )

    predictor.run_complete_pipeline()
