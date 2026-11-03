# -*- coding: utf-8 -*-
"""Проверка связки: soft-sensor -> JSON -> локальная LLM (без интернета)."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import quality_model as qm
import quality_process as qp
from llm_explainer import explain, _payload_to_text

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_llm_report.txt")
buf = []


def w(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    buf.append(line)


def payload_from_state(state: dict, equipment: dict | None = None) -> dict:
    v = qm.quality_verdict(state)
    limits = {k: {"direction": d, "limit": l, "unit": u} for k, (d, l, u) in qp.QUALITY_LIMITS.items()}
    v["limits"] = limits
    v["asset"] = "АГПЗ: аминовая очистка + осушка + НТС, товарный газ и ШФЛУ"
    if equipment:
        v["equipment"] = equipment
    return v


def scenario(mode: str, hour: int | None = None) -> dict:
    df = qm.predict_frame(qp.generate_demo_trajectory(mode, n_hours=48))
    row = df.iloc[hour] if hour is not None else df.iloc[-1]
    return {k: float(row[k]) for k in qp.FEATURES_MEASURED}


w("=" * 96)
w("СВЯЗКА ML -> LLM: сценарий 'разнос вибрации'")
w("=" * 96)

equip = {
    "unit": "Компрессор ВП3-20/9 (COMP-302)",
    "rul_hours": 41.5,
    "failure_prob_pct": 87.0,
    "fault": "износ опорного подшипника, рост вибрации",
}

for label, mode, hour in [
    ("НОРМА (час 2)", "NORMAL", 2),
    ("НАЧАЛО РАЗНОСА (час 18)", "QUALITY_RISK", 18),
    ("ПИК (час 40)", "QUALITY_RISK", 40),
]:
    st = scenario(mode, hour)
    pl = payload_from_state(st, equip if mode != "NORMAL" else None)
    w("")
    w("-" * 96)
    w(f"СЦЕНАРИЙ: {label}")
    w("-" * 96)
    w("ЧТО УХОДИТ В МОДЕЛЬ (факты, ничего лишнего):")
    w(_payload_to_text(pl))
    t0 = time.time()
    ans = explain(pl)
    g = ans["guardrail"]
    w("")
    w(f"ОТВЕТ ЛОКАЛЬНОЙ LLM  (модель: {ans['model']}, {ans['latency_s']} с, fallback: {ans['fallback']}, токенов: {ans.get('tokens')})")
    w(f"guardrail: пройден={g['passed']}, лишние числа в 1-й попытке={g['first_attempt_extra'] or 'нет'}, "
      f"повтор={g['retried']}")
    w(ans["text"])

w("")
w("=" * 96)
w("ИТОГ ПО ЗАЩИТЕ ОТ ВЫДУМАННЫХ ЦИФР")
w("=" * 96)
w("Каждый ответ проходит автоматическую сверку: любое число, которого не было во")
w("входных данных, ловится, ответ перезаписывается с запретом, а при повторе")
w("подменяется детерминированным шаблоном. См. строки guardrail выше.")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("\nОТЧЁТ:", OUT)
