# -*- coding: utf-8 -*-
"""
Физическая модель участка газопереработки АГПЗ (аминовая очистка -> осушка -> НТС).

Зачем: дать ядру ИИ не «набор цифр», а физически связанную картину процесса,
где качество товарного продукта падает по понятной причине, а не по случайности.

Контур (упрощённо, по реальной технологии):
  сырой газ (H2S ~2.4%) -> абсорбер аминовой очистки -> абсорбер осушки (гликоль)
  -> блок НТС (низкотемпературная сепарация) -> товарный газ + ШФЛУ

KPI качества (то, что не измеряется онлайн, а считается косвенно):
  h2s_out_ppm        остаточный сероводород, ppm            норма <= 4
  c3_carryover_g_m3  унос C3+ (дорогой продукт) в газ, г/м3 норма <= 0.8
  dew_point_c        точка росы по воде, °C                 норма <= -10
  shflu_purity_pct   чистота ШФЛУ (доля целевых фракций), %  норма >= 97

Механика деградации (именно её должен ловить ИИ):
  рост вибрации насоса орошения -> вспенивание в абсорбере -> падение КПД тарелки
  -> проскок H2S; рост вибрации компрессора + тёплая сепарация НТС -> брызгоунос
  -> потеря C3+ в товарный газ; деградация гликоля/недогрев регенератора -> рост
  точки росы; тёплая НТС -> проскок метана в ШФЛУ -> падение чистоты.

Эпизоды («разносы») намеренно коррелированы: на реальной установке вибрация,
расход орошения и температура сепарации уходят вразнос вместе.

ВАЖНО: это модель процесса, а не данные завода. Параметры откалиброваны по
типовым проектным значениям и подлежат уточнению по данным АГПЗ.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES_MEASURED = [
    "gas_flow_knm3h",
    "h2s_in_pct",
    "amine_flow_m3h",
    "lean_loading",
    "absorber_dp_kpa",
    "vib_amine_mms",
    "vib_comp_mms",
    "pump_current_a",
    "comp_current_a",
    "glycol_flow_m3h",
    "glycol_purity_pct",
    "reboiler_temp_c",
    "nts_temp_c",
    "nts_press_mpa",
    "ambient_temp_c",
]

QUALITY_LIMITS = {
    "h2s_out_ppm": ("max", 4.0, "ppm"),
    "c3_carryover_g_m3": ("max", 0.8, "г/м³"),
    "dew_point_c": ("max", -10.0, "°C"),
    "shflu_purity_pct": ("min", 97.0, "%"),
}
QUALITY_KPI = list(QUALITY_LIMITS.keys())
KPI_RU = {
    "h2s_out_ppm": "Остаточный H₂S в газе",
    "c3_carryover_g_m3": "Унос C₃+ в товарный газ",
    "dew_point_c": "Точка росы по воде",
    "shflu_purity_pct": "Чистота ШФЛУ",
}


def _upsets(n_hours: int, rng, n_events: int = 9) -> np.ndarray:
    """Связанные эпизоды разноса: 0 — норма, 1 — пик возмущения."""
    env = np.zeros(n_hours)
    starts = rng.integers(0, max(1, n_hours - 90), size=n_events)
    for s in starts:
        length = int(rng.integers(16, 48))
        amp = float(rng.uniform(0.35, 0.95))
        idx = np.arange(s, min(n_hours, s + length))
        ramp = np.sin(np.linspace(0, np.pi, len(idx))) ** 1.3
        env[idx] = np.maximum(env[idx], amp * ramp)
    return env


def _ar1(n_hours: int, mean: float, phi: float, sigma: float, rng) -> np.ndarray:
    """Средство, возвращающееся к норме (не уползает в одну сторону)."""
    x = np.empty(n_hours)
    x[0] = mean
    for i in range(1, n_hours):
        x[i] = x[i - 1] + phi * (mean - x[i - 1]) + rng.normal(0, sigma)
    return x


def _fouling(n_hours: int, rng) -> np.ndarray:
    """Загрязнение демистера: медленный рост с промывками (скрытое состояние)."""
    f = np.empty(n_hours)
    f[0] = 0.05
    for i in range(1, n_hours):
        f[i] = f[i - 1] + 0.85 / max(1, n_hours) + rng.normal(0, 0.0008)
        if f[i] > 0.55 and rng.random() < 0.006:
            f[i] = 0.03
        f[i] = float(np.clip(f[i], 0.0, 1.0))
    return f


def _quality_from_state(df: pd.DataFrame, fouling: np.ndarray, rng) -> pd.DataFrame:
    """Физика качества по измеряемому состоянию процесса."""
    gas = df["gas_flow_knm3h"].values
    dp = df["absorber_dp_kpa"].values
    vib_a = df["vib_amine_mms"].values
    vib_c = df["vib_comp_mms"].values
    amine = df["amine_flow_m3h"].values
    lean = df["lean_loading"].values
    glycol = df["glycol_purity_pct"].values
    glycol_f = df["glycol_flow_m3h"].values
    reboiler = df["reboiler_temp_c"].values
    nts = df["nts_temp_c"].values
    n = len(df)

    # 1) вспенивание абсорбера: вибрация + перепад + нагрузка
    foam = np.clip(
        0.05
        + 0.40 * np.maximum(0.0, vib_a - 4.4) / 2.6
        + 0.28 * np.maximum(0.0, dp - 21.0) / 5.0
        + 0.16 * np.maximum(0.0, gas - 108.0) / 7.0,
        0.0, 1.0,
    )
    # 2) КПД тарелки: вспенивание + захлёбывание
    eta = (1.0 - 0.60 * foam) * (1.0 - 0.28 * np.maximum(0.0, gas - 110.0) / 10.0)
    h2s = df["h2s_in_pct"].values * 1e4 * np.exp(
        -11.5 * (amine / 100.0) * np.clip(1.0 - (lean - 0.20) * 3.4, 0.15, 1.2) * eta
    )
    h2s = np.clip(h2s * (1 + rng.normal(0, 0.03, n)), 0.05, 800.0)

    # 3) унос C3+: тёплая сепарация + вибрация компрессора + загрязнение демистера
    c3 = (
        0.08
        + 0.32 * np.exp((nts + 34.0) / 8.0)
        + 1.10 * np.maximum(0.0, vib_c - 4.6) / 2.4
        + 0.20 * fouling ** 2
        + 0.10 * np.maximum(0.0, gas - 110.0) / 8.0
    )
    c3 = np.clip(c3 * (1 + rng.normal(0, 0.035, n)), 0.02, 9.0)

    # 4) точка росы: деградация гликоля, недолив, недогрев регенератора
    dew = (
        -24.0
        + 30.0 * np.maximum(0.0, 99.30 - glycol)
        + 0.55 * np.maximum(0.0, 100.0 - glycol_f)
        + 0.40 * np.maximum(0.0, 198.0 - reboiler)
    )
    dew = np.clip(dew + rng.normal(0, 0.22, n), -48.0, 15.0)

    # 5) чистота ШФЛУ: проскок метана при тёплой сепарации
    shflu = (
        99.25
        - 1.15 * np.maximum(0.0, nts + 31.0)
        - 0.40 * fouling
        - 0.12 * np.maximum(0.0, gas - 112.0) / 8.0
    )
    shflu = np.clip(shflu + rng.normal(0, 0.05, n), 88.0, 99.8)

    out = df.copy()
    out["h2s_out_ppm"] = h2s
    out["c3_carryover_g_m3"] = c3
    out["dew_point_c"] = dew
    out["shflu_purity_pct"] = shflu
    return out


def _hours_to_next(off: np.ndarray, cap: int = 48) -> np.ndarray:
    n = len(off)
    out = np.full(n, float(cap))
    nxt = n + cap
    for i in range(n - 1, -1, -1):
        if off[i]:
            nxt = i
            out[i] = 0.0
        else:
            out[i] = float(min(cap, nxt - i))
    return out


def generate_plant_history(n_hours: int = 2160, seed: int = 11) -> pd.DataFrame:
    """История работы блока с часовым шагом (90 суток по умолчанию)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    upset = _upsets(n_hours, rng, 13)

    ambient = 8.0 + 14.0 * np.sin(2 * np.pi * t / (24 * 90) - np.pi / 2) + 3.0 * np.sin(2 * np.pi * t / 24)

    load = _ar1(n_hours, 1.0, 0.05, 0.010, rng)
    load = np.clip(load + 0.02 * np.sin(2 * np.pi * t / 168), 0.90, 1.10)

    vib_amine = 2.30 + 5.20 * upset + rng.normal(0, 0.09, n_hours)
    vib_comp = 2.40 + 5.60 * upset + rng.normal(0, 0.10, n_hours)
    vib_amine = np.clip(vib_amine, 1.4, 8.6)
    vib_comp = np.clip(vib_comp, 1.5, 9.0)

    gas_flow = 100.0 * load + rng.normal(0, 1.0, n_hours)
    h2s_in = np.clip(
        _ar1(n_hours, 2.40, 0.01, 0.02, rng) + 0.06 * np.sin(2 * np.pi * t / 500), 1.8, 3.2
    )
    amine_flow = np.clip(103.0 - 13.0 * upset + rng.normal(0, 0.8, n_hours), 80.0, 118.0)
    lean_loading = np.clip(_ar1(n_hours, 0.205, 0.02, 0.0025, rng) + 0.075 * upset, 0.16, 0.40)
    absorber_dp = np.clip(19.5 + 2.0 * (load - 1.0) * 10 + 3.2 * upset ** 1.3
                          + rng.normal(0, 0.22, n_hours), 16.0, 31.0)

    glycol_flow = np.clip(99.0 - 8.0 * upset + rng.normal(0, 1.2, n_hours), 75.0, 112.0)
    glycol_purity = np.clip(
        _ar1(n_hours, 99.22, 0.02, 0.035, rng) - 0.30 * upset + rng.normal(0, 0.02, n_hours), 98.4, 99.8
    )
    reboiler_temp = np.clip(198.0 - 6.0 * upset + rng.normal(0, 0.9, n_hours), 180.0, 205.0)

    nts_temp = np.clip(
        -33.0 + 0.18 * (ambient - 8.0) + 6.2 * upset ** 1.1 + rng.normal(0, 0.35, n_hours), -36.0, -22.0
    )
    nts_press = np.clip(4.66 + rng.normal(0, 0.02, n_hours), 4.45, 4.85)

    pump_current = np.clip(96.0 + 3.5 * (load - 1.0) * 10 + 3.2 * (vib_amine - 2.3)
                           + rng.normal(0, 0.8, n_hours), 80.0, 140.0)
    comp_current = np.clip(124.0 + 9.0 * (load - 1.0) * 10 + 4.0 * (vib_comp - 2.4)
                           + rng.normal(0, 1.0, n_hours), 100.0, 180.0)

    fouling = _fouling(n_hours, rng)

    df = pd.DataFrame({
        "hour": t,
        "ambient_temp_c": ambient,
        "gas_flow_knm3h": gas_flow,
        "h2s_in_pct": h2s_in,
        "amine_flow_m3h": amine_flow,
        "lean_loading": lean_loading,
        "absorber_dp_kpa": absorber_dp,
        "vib_amine_mms": vib_amine,
        "vib_comp_mms": vib_comp,
        "pump_current_a": pump_current,
        "comp_current_a": comp_current,
        "glycol_flow_m3h": glycol_flow,
        "glycol_purity_pct": glycol_purity,
        "reboiler_temp_c": reboiler_temp,
        "nts_temp_c": nts_temp,
        "nts_press_mpa": nts_press,
        "demister_fouling": fouling,
    })

    df = _quality_from_state(df, fouling, rng)

    off = np.zeros(n_hours, dtype=bool)
    for k, (direction, limit, _u) in QUALITY_LIMITS.items():
        off |= (df[k] > limit).to_numpy() if direction == "max" else (df[k] < limit).to_numpy()
    df["off_spec"] = off.astype(int)
    df["hours_to_offspec"] = _hours_to_next(off, cap=48)
    return df


def quality_from_sensors(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Пересчитать KPI качества по измеряемым датчикам.

    Нужно для ручного сценария на стенде: оператор «наддувает» вибрацию,
    а качество пересчитывается по той же физике, что и в истории.
    Внутренние параметры модели (загрязнение демистера) при этом не измеряются.
    """
    rng = np.random.default_rng(seed)
    if "demister_fouling" in df.columns:
        fouling = df["demister_fouling"].to_numpy()
    else:
        fouling = np.full(len(df), 0.15)
    return _quality_from_state(df.copy(), fouling, rng)


def generate_demo_trajectory(mode: str = "QUALITY_RISK", n_hours: int = 48, seed: int = 7) -> pd.DataFrame:
    """Сценарий для показа: 48 часов, шаг 1 час.

    NORMAL       — всё в норме;
    QUALITY_RISK — постепенный разнос: вибрация компрессора и насоса растут,
                   сепарация теплеет, к концу сценария унос C3+ выходит за норму;
    FIXED        — после вмешательства параметры возвращаются в норму.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_hours)
    if mode == "NORMAL":
        sev = np.full(n_hours, 0.03)
    elif mode == "FIXED":
        sev = 0.03 + 0.87 * np.clip((t - 6) / 26.0, 0, 1) ** 1.2
        sev = np.where(t >= 36, 0.03 + 0.87 * np.exp(-(t - 36) / 2.6), sev)
    else:
        sev = 0.03 + 0.87 * np.clip((t - 6) / 30.0, 0, 1) ** 1.25

    gas_flow = 100 + 2.5 * np.sin(t / 5.0) + rng.normal(0, 0.7, n_hours)
    vib_comp = 2.40 + 5.30 * sev + rng.normal(0, 0.07, n_hours)
    vib_amine = 2.30 + 5.00 * sev + rng.normal(0, 0.06, n_hours)
    nts_temp = -33.0 + 6.0 * sev + rng.normal(0, 0.22, n_hours)
    nts_press = 4.66 + rng.normal(0, 0.015, n_hours)
    fouling = np.clip(0.10 + 0.40 * sev, 0, 1)
    h2s_in = 2.40 + rng.normal(0, 0.03, n_hours)
    amine_flow = 103.0 - 12.0 * sev + rng.normal(0, 0.6, n_hours)
    lean_loading = 0.205 + 0.070 * sev + rng.normal(0, 0.002, n_hours)
    absorber_dp = 19.6 + 3.0 * sev + rng.normal(0, 0.18, n_hours)
    glycol_flow = 99.0 - 6.0 * sev + rng.normal(0, 1.0, n_hours)
    glycol_purity = 99.22 - 0.28 * sev + rng.normal(0, 0.02, n_hours)
    reboiler_temp = 198.0 - 5.0 * sev + rng.normal(0, 0.6, n_hours)
    ambient = 9 + 3 * np.sin(2 * np.pi * t / 24)

    df = pd.DataFrame({
        "hour": t,
        "ambient_temp_c": ambient,
        "gas_flow_knm3h": gas_flow,
        "h2s_in_pct": h2s_in,
        "amine_flow_m3h": amine_flow,
        "lean_loading": lean_loading,
        "absorber_dp_kpa": absorber_dp,
        "vib_amine_mms": vib_amine,
        "vib_comp_mms": vib_comp,
        "pump_current_a": 96.0 + 3.2 * (vib_amine - 2.3),
        "comp_current_a": 124.0 + 4.0 * (vib_comp - 2.4),
        "glycol_flow_m3h": glycol_flow,
        "glycol_purity_pct": glycol_purity,
        "reboiler_temp_c": reboiler_temp,
        "nts_temp_c": nts_temp,
        "nts_press_mpa": nts_press,
        "demister_fouling": fouling,
    })
    return _quality_from_state(df, fouling, rng)


if __name__ == "__main__":
    h = generate_plant_history()
    print(h[FEATURES_MEASURED + QUALITY_KPI].describe().T.to_string())
    print("\nдоля часов вне нормы:", round(h["off_spec"].mean() * 100, 2), "%")
    for k, (d, lim, u) in QUALITY_LIMITS.items():
        bad = (h[k] > lim) if d == "max" else (h[k] < lim)
        print(f"  {k:20s} вне нормы {bad.mean()*100:5.1f}%")
