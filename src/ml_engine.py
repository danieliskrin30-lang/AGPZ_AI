"""
Ядро машинного обучения (ML Engine) системы предиктивной диагностики.
Включает 3 модели:
1. Isolation Forest — неконтролируемое раннее обнаружение скрытых аномалий.
2. Random Forest Classifier — диагностика точной причины отказа и вероятности поломки (%).
3. XGBoost Regressor — прогноз остаточного ресурса (RUL в часах) до критического отказа.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from xgboost import XGBRegressor

from src.data_generator import (
    generate_training_dataset,
    FAULT_CLASSES,
)

FEATURE_COLUMNS = [
    "vibration_rms",
    "vibration_kurtosis",
    "crest_factor",
    "temperature_c",
    "pressure_bar",
    "current_a",
    "rpm_dev_pct",
    "vib_temp_index",
    "power_load_factor",
]

FEATURE_NAMES_RU = {
    "vibration_rms": "Вибрация СКЗ (мм/с)",
    "vibration_kurtosis": "Эксцесс вибросигнала (импульсы)",
    "crest_factor": "Пик-фактор вибрации",
    "temperature_c": "Температура узла (°C)",
    "pressure_bar": "Давление в магистрали (бар)",
    "current_a": "Ток нагрузки двигателя (А)",
    "rpm_dev_pct": "Отклонение оборотов вала (%)",
    "vib_temp_index": "Индекс термо-вибронагруженности",
    "power_load_factor": "Коэффициент силовой нагрузки",
}

NOMINAL_BASELINE = {
    "vibration_rms": 2.15,
    "vibration_kurtosis": 3.00,
    "crest_factor": 1.42,
    "temperature_c": 54.0,
    "pressure_bar": 14.4,
    "current_a": 95.0,
    "rpm_dev_pct": 0.0,
    "vib_temp_index": 1.16,
    "power_load_factor": 1.37,
}

NOMINAL_STD = {
    "vibration_rms": 0.45,
    "vibration_kurtosis": 0.35,
    "crest_factor": 0.15,
    "temperature_c": 3.5,
    "pressure_bar": 0.55,
    "current_a": 4.5,
    "rpm_dev_pct": 0.45,
    "vib_temp_index": 0.30,
    "power_load_factor": 0.12,
}


def get_project_paths():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    return base_dir, data_dir, models_dir


def train_and_save_all_models(force_retrain: bool = False):
    """
    Обучает ансамбль моделей на датасете датчиков цеха, сохраняет веса и метрики валидации.
    """
    _, data_dir, models_dir = get_project_paths()
    clf_path = os.path.join(models_dir, "rf_fault_classifier.joblib")
    rul_path = os.path.join(models_dir, "xgb_rul_regressor.joblib")
    iso_path = os.path.join(models_dir, "isolation_forest.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if (
        not force_retrain
        and os.path.exists(clf_path)
        and os.path.exists(rul_path)
        and os.path.exists(iso_path)
        and os.path.exists(metrics_path)
    ):
        return load_models_and_metrics()

    dataset_path = os.path.join(data_dir, "industrial_sensor_dataset.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
    else:
        df = generate_training_dataset(n_samples=4000, seed=42)
        df.to_csv(dataset_path, index=False, encoding="utf-8")

    X = df[FEATURE_COLUMNS]
    y_class = df["fault_class"]
    y_rul = df["rul_hours"]

    X_train, X_test, y_class_train, y_class_test, y_rul_train, y_rul_test = (
        train_test_split(
            X, y_class, y_rul, test_size=0.22, random_state=42, stratify=y_class
        )
    )

    X_train_normal = X_train[y_class_train == 0]
    iso_model = IsolationForest(
        n_estimators=180, contamination=0.04, random_state=42
    )
    iso_model.fit(X_train_normal)

    rf_clf = RandomForestClassifier(
        n_estimators=220,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, y_class_train)

    xgb_rul = XGBRegressor(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.88,
        colsample_bytree=0.9,
        random_state=42,
    )
    xgb_rul.fit(X_train, y_rul_train)

    y_pred_class = rf_clf.predict(X_test)
    y_pred_rul = xgb_rul.predict(X_test)

    acc = float(accuracy_score(y_class_test, y_pred_class))
    f1_mac = float(f1_score(y_class_test, y_pred_class, average="macro"))
    cm = confusion_matrix(y_class_test, y_pred_class).tolist()

    mae_rul = float(mean_absolute_error(y_rul_test, y_pred_rul))
    rmse_rul = float(root_mean_squared_error(y_rul_test, y_pred_rul))
    r2_rul = float(r2_score(y_rul_test, y_pred_rul))

    iso_preds = np.where(iso_model.predict(X_test) == -1, 1, 0)
    true_anom = (y_class_test != 0).astype(int)
    anom_acc = float(accuracy_score(true_anom, iso_preds))

    importances = {
        FEATURE_NAMES_RU[col]: round(float(imp), 4)
        for col, imp in zip(FEATURE_COLUMNS, rf_clf.feature_importances_)
    }

    metrics = {
        "classifier_accuracy": round(acc * 100, 2),
        "classifier_f1_macro": round(f1_mac * 100, 2),
        "anomaly_detection_accuracy": round(anom_acc * 100, 2),
        "rul_mae_hours": round(mae_rul, 2),
        "rul_rmse_hours": round(rmse_rul, 2),
        "rul_r2_score": round(r2_rul, 4),
        "confusion_matrix": cm,
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_importances": importances,
        "class_names": [FAULT_CLASSES[i]["short"] for i in range(5)],
    }

    joblib.dump(rf_clf, clf_path)
    joblib.dump(xgb_rul, rul_path)
    joblib.dump(iso_model, iso_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return rf_clf, xgb_rul, iso_model, metrics


def load_models_and_metrics():
    _, _, models_dir = get_project_paths()
    clf_path = os.path.join(models_dir, "rf_fault_classifier.joblib")
    rul_path = os.path.join(models_dir, "xgb_rul_regressor.joblib")
    iso_path = os.path.join(models_dir, "isolation_forest.joblib")
    metrics_path = os.path.join(models_dir, "metrics.json")

    if not os.path.exists(clf_path):
        return train_and_save_all_models(force_retrain=True)

    rf_clf = joblib.load(clf_path)
    xgb_rul = joblib.load(rul_path)
    iso_model = joblib.load(iso_path)
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return rf_clf, xgb_rul, iso_model, metrics


def predict_single_state(
    vibration_rms: float,
    vibration_kurtosis: float,
    temperature_c: float,
    pressure_bar: float,
    current_a: float,
    rpm_dev_pct: float = -0.2,
    rf_clf=None,
    xgb_rul=None,
    iso_model=None,
) -> dict:
    if rf_clf is None or xgb_rul is None or iso_model is None:
        rf_clf, xgb_rul, iso_model, _ = load_models_and_metrics()

    crest_factor = 1.414 + (vibration_kurtosis - 3.0) * 0.38
    vib_temp_index = (vibration_rms * temperature_c) / 100.0
    power_load_factor = (current_a * pressure_bar) / 1000.0

    row_dict = {
        "vibration_rms": float(vibration_rms),
        "vibration_kurtosis": float(vibration_kurtosis),
        "crest_factor": float(max(1.2, crest_factor)),
        "temperature_c": float(temperature_c),
        "pressure_bar": float(pressure_bar),
        "current_a": float(current_a),
        "rpm_dev_pct": float(rpm_dev_pct),
        "vib_temp_index": float(vib_temp_index),
        "power_load_factor": float(power_load_factor),
    }

    X_input = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

    probs = rf_clf.predict_proba(X_input)[0]
    normal_prob = float(probs[0])
    failure_prob_pct = round((1.0 - normal_prob) * 100.0, 1)

    if failure_prob_pct < 25.0:
        pred_class = 0
    else:
        fault_probs = probs[1:]
        pred_class = int(np.argmax(fault_probs) + 1)

    raw_rul = float(xgb_rul.predict(X_input)[0])
    rul_hours = round(float(np.clip(raw_rul, 1.0, 240.0)), 1)

    iso_score = float(iso_model.decision_function(X_input)[0])
    is_anomaly = bool(iso_score < 0.0 or failure_prob_pct >= 28.0)

    if failure_prob_pct < 25.0 and rul_hours > 120:
        status_level = "NORMAL"
        status_label = "НОРМА (Штатный режим)"
        status_color = "#10B981"
    elif failure_prob_pct < 75.0 and rul_hours >= 24.0:
        status_level = "WARNING"
        status_label = "РАННЕЕ ПРЕДУПРЕЖДЕНИЕ ИИ"
        status_color = "#F59E0B"
    else:
        status_level = "CRITICAL"
        status_label = "КРИТИЧЕСКИЙ РИСК ОТКАЗА"
        status_color = "#EF4444"

    scada_alarm = bool(
        vibration_rms >= 7.1
        or temperature_c >= 85.0
        or pressure_bar <= 9.5
        or current_a >= 135.0
    )

    contributions = {}
    total_z = 0.0
    z_scores = {}
    for col in [
        "vibration_rms",
        "vibration_kurtosis",
        "temperature_c",
        "pressure_bar",
        "current_a",
    ]:
        val = row_dict[col]
        base = NOMINAL_BASELINE[col]
        std = NOMINAL_STD[col]
        if col == "pressure_bar":
            z = abs(val - base) / std
        else:
            z = max(0.0, (val - base) / std)
        z_scores[col] = z
        total_z += z

    for col, z in z_scores.items():
        if total_z > 0.01:
            pct = round((z / total_z) * 100.0, 1)
        else:
            pct = 20.0
        contributions[FEATURE_NAMES_RU[col]] = pct

    fault_info = FAULT_CLASSES[pred_class]

    return {
        "status_level": status_level,
        "status_label": status_label,
        "status_color": status_color,
        "failure_prob_pct": failure_prob_pct,
        "health_index_pct": round(100.0 - failure_prob_pct * 0.85, 1),
        "rul_hours": rul_hours,
        "fault_class_id": pred_class,
        "fault_name": fault_info["name"],
        "fault_short": fault_info["short"],
        "recommended_action": fault_info["action"],
        "fix_button": fault_info.get("fix_button", "🛠️ Устранить неисправность"),
        "fix_report": fault_info.get("fix_report", "Параметры выровнены в норму."),
        "required_parts": fault_info["parts"],
        "assigned_brigade": fault_info["brigade"],
        "scada_alarm_active": scada_alarm,
        "ai_early_catch": bool(status_level in ("WARNING", "CRITICAL") and not scada_alarm),
        "anomaly_score": round(iso_score, 3),
        "is_anomaly": is_anomaly,
        "feature_contributions": contributions,
        "class_probabilities": {
            FAULT_CLASSES[i]["short"]: round(float(probs[i]) * 100.0, 1)
            for i in range(5)
        },
        "raw_features": row_dict,
    }


if __name__ == "__main__":
    rf_clf, xgb_rul, iso_model, metrics = train_and_save_all_models(force_retrain=True)
    print("Trained successfully! Metrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
