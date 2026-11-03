# -*- coding: utf-8 -*-
"""
Планирование выпуска продуктов на установке первичной перегонки нефти (АВТ/CDU).

Задача в терминах производства:
  пришла партия нефти с паспортом качества (ассей: плотность, сера, кривая
  разгонки) -> сколько выйдет нафты, керосина, дизеля и остатка при разных
  температурах отбора -> какой режим закрывает план по продуктам, не уводя
  керосин в некондицию по концу кипения.

Физика (проверяемая, без магии):
  * выходы считаются по кривой истинных температур кипения (TBP) — это
    стандартный лабораторный паспорт нефти;
  * граница нафты фиксирована 180 °C, керосин отбирается 180...T_кк,
    дизель T_кк...T_дк, остаток — выше;
  * объёмные проценты переводятся в массовые через плотности фракций;
  * качество керосина определяется температурой конца отбора: выше 250 °C
    продукт не проходит по концу кипения и уходит в повторную переработку.

Экономика и экология:
  * выручка по продуктам с учётом плана: излишек продаётся дешевле, дефицит
    штрафуется (недопоставка = импорт);
  * некондиция требует повторной переработки -> дополнительное топливо печей
    -> дополнительные выбросы CO2 (расчётная оценка, не замер).

ВАЖНО: ассеи — справочные типовые значения, не паспорт конкретной партии.
Их надо заменить реальными паспортами с НПЗ.
"""
from __future__ import annotations

import numpy as np

# Плотности фракций, г/см3 (типовые)
FRACTION_DENSITY = {
    "naphtha": 0.705,
    "kerosene": 0.790,
    "diesel": 0.845,
    "residue": 0.950,
}
FRACTION_RU = {
    "naphtha": "Нафта (бензиновая фракция)",
    "kerosene": "Керосин (авиатопливо)",
    "diesel": "Дизельное топливо",
    "residue": "Мазут / остаток",
}

# Температура конца отбора нафты (фиксированная граница фракции), °C
NAPHTHA_END_C = 180.0
# Предел качества керосина по концу кипения, °C
KEROSENE_END_MAX_C = 250.0

# Управляемые параметры режима, °C
CUT_BOUNDS = {
    "kerosene_end": (205.0, 265.0),
    "diesel_end": (330.0, 385.0),
}

# Типовой режим оператора «погнать объём» — для сравнения с рекомендацией ИИ
NAIVE_CUTS = {"kerosene_end": 258.0, "diesel_end": 372.0}

# Оценки экономики (₸/т) — уточнять по договорам
PRICE_TENGE_PER_T = {
    "naphtha": 250_000,
    "kerosene": 380_000,
    "diesel": 310_000,
    "residue": 110_000,
}

FURNACE_FUEL_BASE_KG_PER_T = 26.0      # топливо печей, кг у.т./т нефти
REPROCESS_FUEL_KG_PER_T = 45.0         # доп. топливо на повторную переработку
CO2_PER_KG_FUEL = 2.75                 # кг CO2 на кг топлива

ASSAYS = {
    "Тенгиз (лёгкая, малосернистая)": {
        "api": 46.5,
        "sulfur_pct": 0.55,
        "tbp_c": [35, 85, 175, 275, 380, 495, 560],
    },
    "Кумколь (лёгкая, низкосернистая)": {
        "api": 41.5,
        "sulfur_pct": 0.10,
        "tbp_c": [40, 105, 200, 300, 405, 515, 575],
    },
    "Кашаган (лёгкая, сернистая)": {
        "api": 45.0,
        "sulfur_pct": 0.80,
        "tbp_c": [38, 95, 185, 285, 390, 500, 565],
    },
    "Уральская смесь (средняя, сернистая)": {
        "api": 31.0,
        "sulfur_pct": 1.70,
        "tbp_c": [45, 150, 290, 390, 470, 560, 610],
    },
}
TBP_PERCENT_POINTS = np.array([0, 10, 30, 50, 70, 90, 100], dtype=float)


def tbp_curve(assay: dict) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(assay["tbp_c"], dtype=float), TBP_PERCENT_POINTS


def distilled_vol_pct(assay: dict, temp_c) -> np.ndarray | float:
    """Накопленный объёмный % нефти, выкипающий до температуры (поддерживает массивы)."""
    temps, pct = tbp_curve(assay)
    return np.clip(np.interp(temp_c, temps, pct), 0.0, 100.0)


def cut_yields_vol(assay: dict, cuts: dict) -> dict:
    """Объёмные выходы фракций, % на нефть."""
    k_end = float(cuts["kerosene_end"])
    d_end = float(cuts["diesel_end"])
    v_naphtha = float(distilled_vol_pct(assay, NAPHTHA_END_C))
    v_ker = float(distilled_vol_pct(assay, k_end))
    v_diesel = float(distilled_vol_pct(assay, d_end))
    return {
        "naphtha": max(0.0, v_naphtha),
        "kerosene": max(0.0, v_ker - v_naphtha),
        "diesel": max(0.0, v_diesel - v_ker),
        "residue": max(0.0, 100.0 - v_diesel),
    }


def cut_yields_mass(assay: dict, cuts: dict) -> dict:
    """Массовые выходы фракций, % на нефть (нормировка по плотностям)."""
    vol = cut_yields_vol(assay, cuts)
    raw = {k: v * FRACTION_DENSITY[k] for k, v in vol.items()}
    total = sum(raw.values()) or 1.0
    return {k: v / total * 100.0 for k, v in raw.items()}


def quality_status(cuts: dict) -> dict:
    ker_ok = float(cuts["kerosene_end"]) <= KEROSENE_END_MAX_C
    issues = []
    if not ker_ok:
        issues.append(
            f"керосин не проходит по концу кипения: {cuts['kerosene_end']:.0f} °C "
            f"при пределе {KEROSENE_END_MAX_C:.0f} °C"
        )
    return {"kerosene_ok": ker_ok, "issues": issues}


def production_tpd(assay: dict, cuts: dict, throughput_tpd: float) -> dict:
    mass = cut_yields_mass(assay, cuts)
    prod = {k: throughput_tpd * mass[k] / 100.0 for k in mass}
    q = quality_status(cuts)
    offspec = 0.0 if q["kerosene_ok"] else prod["kerosene"]
    return {"mass_pct": mass, "tpd": prod, "offspec_tpd": offspec, "quality": q}


def evaluate_mode(assay: dict, cuts: dict, throughput_tpd: float, demand_tpd: dict) -> dict:
    """Деньги, выполнение плана и выбросы для конкретного режима отбора."""
    p = production_tpd(assay, cuts, throughput_tpd)
    prod = dict(p["tpd"])
    offspec = p["offspec_tpd"]

    revenue = 0.0
    fulfilment, deficit = {}, {}
    for k, made in prod.items():
        plan = float(demand_tpd.get(k, 0.0))
        price = PRICE_TENGE_PER_T[k]
        if k == "kerosene" and offspec > 0:
            revenue += made * PRICE_TENGE_PER_T["residue"]
            fulfilment[k] = 0.0
            deficit[k] = round(plan, 1)
            continue
        covered = min(made, plan) if plan > 0 else 0.0
        surplus = max(0.0, made - plan) if plan > 0 else made
        revenue += covered * price + surplus * price * 0.60
        fulfilment[k] = round(100.0 * covered / plan, 1) if plan > 0 else 100.0
        deficit[k] = round(max(0.0, plan - made), 1)

    reprocess_fuel_kg = offspec * REPROCESS_FUEL_KG_PER_T
    base_fuel_kg = throughput_tpd * FURNACE_FUEL_BASE_KG_PER_T
    co2_tpd = (base_fuel_kg + reprocess_fuel_kg) * CO2_PER_KG_FUEL / 1000.0

    return {
        "cuts": {k: round(float(v), 1) for k, v in cuts.items()},
        "production_tpd": {k: round(v, 1) for k, v in prod.items()},
        "mass_pct": {k: round(v, 2) for k, v in p["mass_pct"].items()},
        "offspec_tpd": round(offspec, 1),
        "revenue_tenge_day": round(revenue),
        "fulfilment_pct": fulfilment,
        "deficit_tpd": deficit,
        "co2_tpd": round(co2_tpd, 1),
        "quality_issues": p["quality"]["issues"],
    }


def optimize_mode(
    assay: dict,
    throughput_tpd: float,
    demand_tpd: dict,
    step: float = 1.5,
) -> dict:
    """Подбор температур отбора под план: векторизованный перебор по сетке.

    Критерий — максимум выручки с штрафом за невыполнение плана. Некондиция
    наказывается автоматически: керосин идёт по цене остатка и жжёт лишнее топливо.
    """
    k_grid = np.arange(CUT_BOUNDS["kerosene_end"][0], CUT_BOUNDS["kerosene_end"][1] + 1e-9, step)
    d_grid = np.arange(CUT_BOUNDS["diesel_end"][0], CUT_BOUNDS["diesel_end"][1] + 1e-9, step)
    k_arr = k_grid[:, None]
    d_arr = d_grid[None, :]

    temps, pct = tbp_curve(assay)
    v_naphtha = float(np.interp(NAPHTHA_END_C, temps, pct))
    v_ker = np.interp(k_arr, temps, pct)
    v_diesel = np.interp(d_arr, temps, pct)

    vol = {
        "naphtha": np.full_like(v_ker + v_diesel, v_naphtha),
        "kerosene": np.clip(v_ker - v_naphtha, 0, 100),
        "diesel": np.clip(v_diesel - v_ker, 0, 100),
        "residue": np.clip(100.0 - v_diesel, 0, 100),
    }
    raw = {k: v * FRACTION_DENSITY[k] for k, v in vol.items()}
    total = sum(raw.values())
    mass = {k: v / np.maximum(total, 1e-9) * 100.0 for k, v in raw.items()}
    prod = {k: throughput_tpd * mass[k] / 100.0 for k in mass}

    ker_ok = k_arr <= KEROSENE_END_MAX_C
    revenue = np.zeros_like(prod["naphtha"])
    penalty = np.zeros_like(revenue)
    for k, made in prod.items():
        plan = float(demand_tpd.get(k, 0.0))
        price = PRICE_TENGE_PER_T[k]
        covered = np.minimum(made, plan) if plan > 0 else made
        surplus = np.maximum(0.0, made - plan) if plan > 0 else np.zeros_like(made)
        value = covered * price + surplus * price * 0.60
        gap = np.maximum(0.0, plan - made) if plan > 0 else np.zeros_like(made)
        if k == "kerosene":
            value = np.where(ker_ok, value, made * PRICE_TENGE_PER_T["residue"])
            gap = np.where(ker_ok, gap, np.full_like(made, plan))
        revenue = revenue + value
        penalty = penalty + gap * price * 0.5

    score = revenue - penalty
    idx_k, idx_d = np.unravel_index(int(np.argmax(score)), score.shape)
    cuts = {"kerosene_end": float(k_grid[idx_k]), "diesel_end": float(d_grid[idx_d])}
    ev = evaluate_mode(assay, cuts, throughput_tpd, demand_tpd)
    return {
        "score": round(float(score[idx_k, idx_d])),
        "mode": ev,
        "bounds": CUT_BOUNDS,
        "evaluated_variants": int(score.size),
        "against_naive": compare_naive(assay, throughput_tpd, demand_tpd, ev),
    }


def compare_naive(assay: dict, throughput_tpd: float, demand_tpd: dict, best_eval: dict) -> dict:
    """Сравнение рекомендуемого режима с типовым «погнать объём»."""
    naive = evaluate_mode(assay, NAIVE_CUTS, throughput_tpd, demand_tpd)
    return {
        "naive_mode": naive,
        "delta_revenue_day": best_eval["revenue_tenge_day"] - naive["revenue_tenge_day"],
        "delta_co2_tpd": round(best_eval["co2_tpd"] - naive["co2_tpd"], 1),
        "delta_offspec_tpd": round(best_eval["offspec_tpd"] - naive["offspec_tpd"], 1),
    }


def plan_report(assay_name: str, throughput_tpd: float, demand_tpd: dict) -> dict:
    """Готовый отчёт для интерфейса и для LLM-объяснения."""
    assay = ASSAYS[assay_name]
    res = optimize_mode(assay, throughput_tpd, demand_tpd)
    mode = res["mode"]
    return {
        "asset": "НПЗ, установка первичной перегонки нефти (АВТ)",
        "assay": assay_name,
        "assay_props": {
            "api": assay["api"],
            "sulfur_pct": assay["sulfur_pct"],
            "note": "справочные типовые значения",
        },
        "throughput_tpd": throughput_tpd,
        "demand_tpd": demand_tpd,
        "naphtha_cut_c": NAPHTHA_END_C,
        "recommended_cuts": mode["cuts"],
        "production_tpd": mode["production_tpd"],
        "mass_pct": mode["mass_pct"],
        "fulfilment_pct": mode["fulfilment_pct"],
        "deficit_tpd": mode["deficit_tpd"],
        "offspec_tpd": mode["offspec_tpd"],
        "revenue_tenge_day": mode["revenue_tenge_day"],
        "revenue_tenge_year": round(mode["revenue_tenge_day"] * 330),
        "co2_tpd": mode["co2_tpd"],
        "co2_year_t": round(mode["co2_tpd"] * 330),
        "quality_issues": mode["quality_issues"],
        "naive": res["against_naive"],
        "limits": {"kerosene_end_max_c": KEROSENE_END_MAX_C, "bounds": CUT_BOUNDS},
        "prices": PRICE_TENGE_PER_T,
        "evaluated_variants": res["evaluated_variants"],
        "note": "ассеи справочные, экономика расчётная",
    }


def tbp_series(assay_name: str) -> dict:
    """Кривая разгонки для графика."""
    assay = ASSAYS[assay_name]
    temps, pct = tbp_curve(assay)
    return {"temp_c": temps.tolist(), "vol_pct": pct.tolist()}


if __name__ == "__main__":
    import time

    demand = {"naphtha": 3000, "kerosene": 2000, "diesel": 5000, "residue": 5500}
    for name in ASSAYS:
        t0 = time.time()
        r = plan_report(name, 16000.0, demand)
        dt = (time.time() - t0) * 1000
        print(f"\n{name}  (расчёт {dt:.0f} мс, вариантов {r['evaluated_variants']})")
        print("  режим:", r["recommended_cuts"], "| нафта до", r["naphtha_cut_c"], "°C")
        print("  выход, %:", r["mass_pct"])
        print("  выпуск, т/сут:", r["production_tpd"])
        print("  план, %:", r["fulfilment_pct"], "| дефицит:", r["deficit_tpd"])
        print("  некондиция:", r["offspec_tpd"], "| CO2:", r["co2_tpd"], "т/сут")
        print("  выручка:", f"{r['revenue_tenge_day']:,}", "₸/сут")
        n = r["naive"]["naive_mode"]
        print("  наивный режим: план", n["fulfilment_pct"], "| некондиция", n["offspec_tpd"],
              "| CO2", n["co2_tpd"], "| ₸", f"{n['revenue_tenge_day']:,}")
        print("  дельта:", f"{r['naive']['delta_revenue_day']:,}", "₸/сут | CO2:",
              r["naive"]["delta_co2_tpd"], "т/сут")
