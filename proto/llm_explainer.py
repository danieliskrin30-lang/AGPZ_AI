# -*- coding: utf-8 -*-
"""
Локальный LLM-объяснитель (Ollama, без интернета).

Роль языковой модели здесь строго ограничена: она НЕ считает и НЕ прогнозирует.
Все числа приходят из soft-sensor (quality_model.py) и диагностики оборудования,
а LLM переводит их в понятную рекомендацию механику/технологу и объясняет причину.

Почему так, а не «спросить у модели»: LLM не является численной моделью процесса.
Если позволить ей считать, она придумает правдоподобные, но неверные цифры —
на защите это ловится первым же вопросом «откуда это число?».

Если Ollama недоступна — срабатывает шаблонный fallback: демонстрация не падает
даже на чужом ноутбуке без модели.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("AGPZ_OLLAMA_URL", "http://127.0.0.1:11434/api/chat")
MODEL = os.environ.get("AGPZ_LLM", "qwen2.5:7b-instruct-q4_K_M")
TIMEOUT_S = float(os.environ.get("AGPZ_LLM_TIMEOUT", "180"))

SYSTEM_PROMPT = """Ты — старший технолог газоперерабатывающего завода (АГПЗ, Казахстан).
Ты получаешь результат работы модели машинного обучения и объясняешь его оператору
и механику цеха. Твои правила:
1. Используй ТОЛЬКО числа из входного блока. Ничего не досчитывай и не придумывай.
2. Пиши по-русски, 4–6 предложений, простым языком инженера, без markdown и списков.
3. Структура: что происходит сейчас -> физическая причина -> что сделать в первую очередь -> срок и риск в деньгах.
4. Не выдумывай названия оборудования и регламентов, которых нет во входных данных.
5. Не пиши вводных фраз вроде «Конечно» и не повторяй входные числа списком.
6. Не выполняй никакой арифметики: не умножай, не пересчитывай суммы за период и не
   округляй потери. Приводи только те числа, что даны, в их исходных единицах.
7. Соблюдай физику: унос, потери и вибрацию СНИЖАЮТ, а не «улучшают»; давление и расход
   ВОССТАНАВЛИВАЮТ. Не пиши «улучшить унос» или «увеличить потери».
"""


def _payload_to_text(p: dict) -> str:
    """Строгий текстовый блок с числами — единственный источник фактов для модели."""
    q = p.get("quality", {})
    limits = p.get("limits", {})
    lines = [
        f"Участок: {p.get('asset', 'АГПЗ, блок очистки и НТС')}",
        "Текущее качество продукта (модель soft-sensor):",
    ]
    for k, v in q.items():
        lim = limits.get(k)
        tgt = f", норма {'<= ' if lim.get('direction') == 'max' else '>= '}{lim.get('limit')} {lim.get('unit', '')}" if lim else ""
        lines.append(f"  - {k}: {v}{tgt}")
    lines.append(
        f"Вероятность любого выхода за любую норму в ближайшие 8 часов: {p.get('prob_offspec_8h_pct')}%"
    )
    lines.append(f"Оценка времени до нарушения нормы: {p.get('hours_to_offspec')} ч")
    if p.get("violations"):
        lines.append("Уже вне нормы: " + ", ".join(f"{v['kpi']}={v['value']} {v['unit']}" for v in p["violations"]))
    top = p.get("shap_top") or {}
    if top:
        lines.append("Вклад факторов в прогноз уноса C3+ (SHAP, %):")
        for name, pct in top.items():
            lines.append(f"  - {name}: {pct}%")
    if p.get("equipment"):
        e = p["equipment"]
        lines.append(
            f"Оборудование: {e.get('unit')}, остаточный ресурс {e.get('rul_hours')} ч, "
            f"вероятность отказа {e.get('failure_prob_pct')}%, диагноз: {e.get('fault')}"
        )
    if p.get("loss_tenge_per_hour") is not None:
        lines.append(f"Потери от некондиции/уноса: {p['loss_tenge_per_hour']} тенге в час")
    return "\n".join(lines)


def build_messages(payload: dict) -> list[dict]:
    user = (
        "Данные установки:\n\n"
        + _payload_to_text(payload)
        + "\n\nДай объяснение оператору по правилам. Не добавляй чисел, которых нет выше."
    )
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def _numbers(text: str) -> set[str]:
    """Числа в тексте с нормализацией десятичного разделителя."""
    import re

    norm = text.replace(",", ".")
    return set(re.findall(r"\d+(?:\.\d+)?", norm))


def validate_answer(text: str, payload: dict) -> list[str]:
    """Какие числа появились в ответе, но отсутствуют во входных данных.

    Это защита от галлюцинаций: языковая модель не имеет права добавлять цифры,
    которых ей не дали. Если такие числа есть — ответ не показываем как есть.
    """
    allowed = _numbers(_payload_to_text(payload))
    return sorted(_numbers(text) - allowed)


def _call_ollama(messages: list[dict], temperature: float = 0.2) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 260, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError("пустой ответ модели")
    return text, data


def explain(payload: dict) -> dict:
    """Возвращает {'text', 'model', 'latency_s', 'fallback', 'guardrail'}."""
    t0 = time.time()
    guard = {"checked": True, "first_attempt_extra": [], "retried": False, "passed": True}
    try:
        text, data = _call_ollama(build_messages(payload))
        extra = validate_answer(text, payload)
        guard["first_attempt_extra"] = extra
        if extra:
            # одна повторная попытка с прямым запретом на эти числа
            guard["retried"] = True
            msgs = build_messages(payload) + [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": (
                        f"В твоём ответе появились числа {', '.join(extra)}, которых нет во входных данных. "
                        "Это ошибка. Перепиши ответ, используя только числа из блока данных, без арифметики."
                    ),
                },
            ]
            text2, data = _call_ollama(msgs, temperature=0.0)
            extra2 = validate_answer(text2, payload)
            if extra2:
                guard["passed"] = False
                guard["second_attempt_extra"] = extra2
                return {
                    "text": fallback_text(payload),
                    "model": "guardrail->template",
                    "latency_s": round(time.time() - t0, 2),
                    "fallback": True,
                    "guardrail": guard,
                }
            text = text2
        return {
            "text": text,
            "model": MODEL,
            "latency_s": round(time.time() - t0, 2),
            "fallback": False,
            "guardrail": guard,
            "tokens": data.get("eval_count"),
        }
    except Exception as e:  # noqa: BLE001 — демо не должно падать
        return {
            "text": fallback_text(payload),
            "model": "template",
            "latency_s": round(time.time() - t0, 2),
            "fallback": True,
            "error": f"{type(e).__name__}: {e}",
            "guardrail": guard,
        }


def fallback_text(p: dict) -> str:
    """Детерминированный ответ без LLM — если модель не поднята."""
    q = p.get("quality", {})
    top = list((p.get("shap_top") or {}).items())[:2]
    cause = ", ".join(f"{n} ({v}%)" for n, v in top) if top else "вибрационный фон узла"
    viol = p.get("violations") or []
    state = "Показатели уже вне нормы" if viol else "Показатели пока в норме, но тренд неблагоприятный"
    return (
        f"{state}. Прогноз уноса C3+ {q.get('c3_carryover_g_m3', '—')} г/м³ против нормы 0,8. "
        f"Основной вклад дают {cause}. "
        f"В ближайшие 8 часов риск выхода за норму {p.get('prob_offspec_8h_pct')}%, "
        f"расчётное время до нарушения {p.get('hours_to_offspec')} ч. "
        f"Рекомендация: снизить нагрузку по газу и восстановить режим сепарации, "
        f"проверить вибрацию компрессора и состояние демистера. "
        f"Цена бездействия {p.get('loss_tenge_per_hour')} тенге в час."
    )
