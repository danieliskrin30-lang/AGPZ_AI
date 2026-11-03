# -*- coding: utf-8 -*-
"""
Soft-sensor (виртуальный анализатор) качества товарного продукта.

Задача: по телеметрии процесса (расходы, давления, температуры, вибрация, токи)
предсказать качество продукта, которого никто не измеряет в реальном времени,
и заранее дать упреждение до выхода за норму.

Обучается на истории physical-модели процесса (quality_process.py), при этом
признаки — только те, что реально измеряются на установке + физические
индикаторы, которые инженер может посчитать по этим датчикам.

Выход:
  * nowcast 4 KPI качества;
  * вероятность выхода за норму в ближайшие 8 часов;
  * накопленное упреждение (часов до нарушения);
  * деньги: потеря дорогого C3+ и штраф за некондиционный газ;
  * вклад признаков (SHAP) — «почему ИИ так решил».
"""
from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score, precision_recall_fscore_support
from xgboost import XGBClassifier, XGBRegressor

from quality_process import (
    FEATURES_MEASURED,
    QUALITY_KPI,
    QUALITY_LIMITS,
    generate_plant_history,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")

# Экономика (оценки, помечены как допущения — уточнять по факту договоров)
SHFLU_PRICE_TENGE_PER_T = 350_000       # цена ШФЛУ, ₸/т
OFFSPEC_PENALTY_TENGE_PER_H = 2_400_000  # некондиционный газ: возврат/факел, ₸/ч

FEATURES_DERIVED = [
    "load_ratio",
    "foam_index",
    "tray_eff_proxy",
    "lean_margin",
    "nts_superheat",
    "entrainment_vib",
    "glycol_deficit",
    "reboiler_deficit",
    "flow_deficit",
    "vib_per_load",
    "dp_margin",
]
ALL_FEATURES = FEATURES_MEASURED + FEATURES_DERIVED

FEATURE_RU = {
    "gas_flow_knm3h": "Расход сырого газа",
    "h2s_in_pct": "Содержание H₂S на входе",
    "amine_flow_m3h": "Расход раствора амина",
    "lean_loading": "Остаточная нагрузка амина",
    "absorber_dp_kpa": "Перепад давления абсорбера",
    "vib_amine_mms": "Вибрация насоса орошения",
    "vib_comp_mms": "Вибрация компрессора",
    "pump_current_a": "Ток насоса орошения",
    "comp_current_a": "Ток компрессора",
    "glycol_flow_m3h": "Расход гликоля",
    "glycol_purity_pct": "Чистота гликоля",
    "reboiler_temp_c": "Температура регенератора",
    "nts_temp_c": "Температура сепарации (НТС)",
    "nts_press_mpa": "Давление НТС",
    "ambient_temp_c": "Температура окружающей среды",
    "load_ratio": "Загрузка по газу (доля проектной)",
    "foam_index": "Индекс вспенивания абсорбера",
    "tray_eff_proxy": "Оценка КПД тарелки",
    "lean_margin": "Запас по нагрузке амина",
    "nts_superheat": "Перегрев сепарации",
    "entrainment_vib": "Брызгоунос от вибрации",
    "glycol_deficit": "Дефицит чистоты гликоля",
    "reboiler_deficit": "Недогрев регенератора",
    "flow_deficit": "Недолив гликоля",
    "vib_per_load": "Вибрация на единицу нагрузки",
    "dp_margin": "Запас по перепаду абсорбера",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Физические индикаторы, которые считаются из имеющихся датчиков."""
    out = df.copy()
    out["load_ratio"] = out["gas_flow_knm3h"] / 100.0
    foam = (
        0.05
        + 0.40 * np.maximum(0.0, out["vib_amine_mms"] - 4.4) / 2.6
        + 0.28 * np.maximum(0.0, out["absorber_dp_kpa"] - 21.0) / 5.0
        + 0.16 * np.maximum(0.0, out["gas_flow_knm3h"] - 108.0) / 7.0
    )
    out["foam_index"] = np.clip(foam, 0.0, 1.0)
    out["tray_eff_proxy"] = (1 - 0.60 * out["foam_index"]) * (
        1 - 0.28 * np.maximum(0.0, out["gas_flow_knm3h"] - 110.0) / 10.0
    )
    out["lean_margin"] = np.clip(1.0 - (out["lean_loading"] - 0.20) * 3.4, 0.15, 1.2)
    out["nts_superheat"] = out["nts_temp_c"] + 36.0
    out["entrainment_vib"] = np.maximum(0.0, out["vib_comp_mms"] - 4.5)
    out["glycol_deficit"] = np.maximum(0.0, 99.30 - out["glycol_purity_pct"])
    out["reboiler_deficit"] = np.maximum(0.0, 198.0 - out["reboiler_temp_c"])
    out["flow_deficit"] = np.maximum(0.0, 100.0 - out["glycol_flow_m3h"])
    out["vib_per_load"] = out["vib_comp_mms"] / np.maximum(0.5, out["load_ratio"])
    out["dp_margin"] = out["absorber_dp_kpa"] - 21.0
    return out


def train_and_save(force: bool = False, n_hours: int = 2160, seed: int = 11) -> dict:
    os.makedirs(MODELS_DIR, exist_ok=True)
    metrics_path = os.path.join(MODELS_DIR, "quality_metrics.json")
    if not force and os.path.exists(metrics_path) and os.path.exists(os.path.join(MODELS_DIR, "kpi_h2s_out_ppm.joblib")):
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    hist = generate_plant_history(n_hours=n_hours, seed=seed)
    X = build_features(hist)[ALL_FEATURES]
    split = int(len(hist) * 0.70)  # честное деление по времени, без перемешивания

    Xtr, Xte = X.iloc[:split], X.iloc[split:]
    results = {}
    models = {}

    for kpi in QUALITY_KPI:
        m = XGBRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
        )
        m.fit(Xtr, hist[kpi].iloc[:split])
        pred = m.predict(Xte)
        results[kpi] = {
            "mae": float(mean_absolute_error(hist[kpi].iloc[split:], pred)),
            "r2": float(r2_score(hist[kpi].iloc[split:], pred)),
            "bool_acc_within_10pct": float(
                np.mean(np.abs(pred - hist[kpi].iloc[split:].values) <= 0.10 * np.abs(hist[kpi].iloc[split:].values) + 1e-9)
            ),
        }
        models[kpi] = m
        joblib.dump(m, os.path.join(MODELS_DIR, f"kpi_{kpi}.joblib"))

    # упреждение до нарушения нормы
    tte = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, random_state=42)
    tte.fit(Xtr, np.minimum(hist["hours_to_offspec"].iloc[:split], 48))
    tte_pred = tte.predict(Xte)
    results["hours_to_offspec"] = {
        "mae_hours": float(mean_absolute_error(hist["hours_to_offspec"].iloc[split:], tte_pred)),
        "r2": float(r2_score(hist["hours_to_offspec"].iloc[split:], tte_pred)),
    }
    models["hours_to_offspec"] = tte
    joblib.dump(tte, os.path.join(MODELS_DIR, "hours_to_offspec.joblib"))

    # классификатор: нарушение нормы в ближайшие 8 часов
    y_cls = (hist["hours_to_offspec"] <= 8).astype(int)
    clf = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf.fit(Xtr, y_cls.iloc[:split])
    p = clf.predict_proba(Xte)[:, 1]
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_cls.iloc[split:], (p >= 0.5).astype(int), average="binary", zero_division=0
    )
    results["offspec_8h"] = {
        "auc": float(roc_auc_score(y_cls.iloc[split:], p)) if y_cls.iloc[split:].nunique() > 1 else None,
        "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "positives_test": int(y_cls.iloc[split:].sum()),
    }
    models["offspec_8h"] = clf
    joblib.dump(clf, os.path.join(MODELS_DIR, "offspec_8h.joblib"))

    # плановый горизонт: нарушение в ближайшие 24 часа
    y24 = (hist["hours_to_offspec"] <= 24).astype(int)
    clf24 = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                          subsample=0.9, colsample_bytree=0.9, random_state=42)
    clf24.fit(Xtr, y24.iloc[:split])
    p24 = clf24.predict_proba(Xte)[:, 1]
    prec24, rec24, f124, _ = precision_recall_fscore_support(
        y24.iloc[split:], (p24 >= 0.5).astype(int), average="binary", zero_division=0
    )
    results["offspec_24h"] = {
        "auc": float(roc_auc_score(y24.iloc[split:], p24)) if y24.iloc[split:].nunique() > 1 else None,
        "precision": float(prec24), "recall": float(rec24), "f1": float(f124),
        "positives_test": int(y24.iloc[split:].sum()),
    }
    models["offspec_24h"] = clf24
    joblib.dump(clf24, os.path.join(MODELS_DIR, "offspec_24h.joblib"))

    results["dataset"] = {
        "hours": int(len(hist)), "train": int(split), "test": int(len(hist) - split),
        "offspec_share_pct": round(float(hist["off_spec"].mean() * 100), 2),
        "features": len(ALL_FEATURES),
        "split": "по времени: первые 70% — обучение, последние 30% — проверка",
    }
    results["thresholds"] = {k: {"direction": v[0], "limit": v[1], "unit": v[2]} for k, v in QUALITY_LIMITS.items()}

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


_BUNDLE = {}


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Пакетный расчёт качества по кадрам процесса (для графиков и статистики)."""
    bundle = load_bundle()
    X = build_features(df)[ALL_FEATURES]
    out = df.copy()
    for kpi in QUALITY_KPI:
        out["pred_" + kpi] = bundle[kpi].predict(X)
    out["pred_hours_to_offspec"] = np.clip(bundle["hours_to_offspec"].predict(X), 0.0, 48.0)
    out["prob_offspec_8h"] = bundle["offspec_8h"].predict_proba(X)[:, 1]
    out["prob_offspec_24h"] = bundle["offspec_24h"].predict_proba(X)[:, 1]
    return out



def load_bundle() -> dict:
    if _BUNDLE:
        return _BUNDLE
    if not os.path.exists(os.path.join(MODELS_DIR, "quality_metrics.json")):
        train_and_save(force=True)
    bundle = {k: joblib.load(os.path.join(MODELS_DIR, f"kpi_{k}.joblib")) for k in QUALITY_KPI}
    bundle["hours_to_offspec"] = joblib.load(os.path.join(MODELS_DIR, "hours_to_offspec.joblib"))
    bundle["offspec_8h"] = joblib.load(os.path.join(MODELS_DIR, "offspec_8h.joblib"))
    bundle["offspec_24h"] = joblib.load(os.path.join(MODELS_DIR, "offspec_24h.joblib"))
    with open(os.path.join(MODELS_DIR, "quality_metrics.json"), encoding="utf-8") as f:
        bundle["metrics"] = json.load(f)
    _BUNDLE.update(bundle)
    return _BUNDLE


def evaluate_soft_sensor(row: dict) -> pd.DataFrame:
    """Расчёт качества по одному состоянию процесса (dict с измеряемыми датчиками)."""
    bundle = load_bundle()
    one = pd.DataFrame([{k: float(row[k]) for k in FEATURES_MEASURED}])
    X = build_features(one)[ALL_FEATURES]
    out = {"raw": one, "features": X}
    for kpi in QUALITY_KPI:
        out[kpi] = float(bundle[kpi].predict(X)[0])
    out["hours_to_offspec"] = float(np.clip(bundle["hours_to_offspec"].predict(X)[0], 0, 48))
    out["prob_offspec_8h"] = float(bundle["offspec_8h"].predict_proba(X)[0, 1])
    out["prob_offspec_24h"] = float(bundle["offspec_24h"].predict_proba(X)[0, 1])
    return out


def quality_verdict(state: dict) -> dict:
    """Человеческий вердикт + деньги + объяснение (SHAP)."""
    bundle = load_bundle()
    q = evaluate_soft_sensor(state)
    violations = []
    for kpi, (direction, limit, unit) in QUALITY_LIMITS.items():
        val = q[kpi]
        bad = val > limit if direction == "max" else val < limit
        if bad:
            violations.append({"kpi": kpi, "value": round(val, 3), "limit": limit, "unit": unit})

    gas = float(state["gas_flow_knm3h"])
    c3 = max(0.0, q["c3_carryover_g_m3"] - 0.3)
    loss_t_h = gas * c3 / 1000.0
    loss_tenge_h = loss_t_h * SHFLU_PRICE_TENGE_PER_T
    if any(v["kpi"] == "h2s_out_ppm" for v in violations):
        loss_tenge_h += OFFSPEC_PENALTY_TENGE_PER_H

    shap_pct = {}
    try:
        import shap

        if "explainer" not in _BUNDLE:
            _BUNDLE["explainer"] = shap.TreeExplainer(bundle["c3_carryover_g_m3"])
        vals = _BUNDLE["explainer"].shap_values(q["features"])
        contrib = np.abs(np.asarray(vals)[0])
        total = contrib.sum() or 1.0
        order = np.argsort(-contrib)[:5]
        for i in order:
            if contrib[i] <= 0:
                continue
            name = ALL_FEATURES[i]
            shap_pct[FEATURE_RU.get(name, name)] = round(float(contrib[i] / total * 100), 1)
    except Exception:
        shap_pct = {}

    return {
        "quality": {k: round(float(q[k]), 3) for k in QUALITY_KPI},
        "prob_offspec_8h_pct": round(q["prob_offspec_8h"] * 100, 1),
        "prob_offspec_24h_pct": round(q["prob_offspec_24h"] * 100, 1),
        "hours_to_offspec": round(q["hours_to_offspec"], 1),
        "violations": violations,
        "shap_top": shap_pct,
        "loss_tenge_per_hour": round(loss_tenge_h),
        "loss_tenge_per_day": round(loss_tenge_h * 24),
        "target_note": "цена ШФЛУ и штраф за некондицию — допущения, уточнить по договорам АГПЗ",
    }
