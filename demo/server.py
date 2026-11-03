# -*- coding: utf-8 -*-
"""
AI-PromPlatform · локальный сервер мнемосхемы НТС-1.

Что делает:
  * отдаёт страницу nts_hmi.html на http://127.0.0.1:8765;
  * принимает вопросы из чата и тревоги (/api/ask) и передаёт их локальной
    модели Qwen2.5 через Ollama вместе со снимком данных узла;
  * проверяет ответ модели: если в нём есть число, которого нет в данных,
    просит переписать, а при повторной ошибке возвращает шаблонный ответ
    (та же защита, что в proto/llm_explainer.py);
  * пишет каждый вопрос и ответ в llm_log.json — из него собирается демо-видео.

Только стандартная библиотека Python, ничего ставить не нужно.
Запуск: START_NTS.bat или  python server.py
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
OLLAMA = os.environ.get("AGPZ_OLLAMA", "http://127.0.0.1:11434")
MODEL = os.environ.get("AGPZ_LLM", "qwen2.5:7b-instruct-q4_K_M")
TIMEOUT_S = float(os.environ.get("AGPZ_LLM_TIMEOUT", "60"))
LOG_PATH = os.path.join(HERE, "llm_log.json")
PAGE = "nts_hmi.html"

SYSTEM_PROMPT = """Ты — ИИ-технолог установки низкотемпературной сепарации (НТС) газоперерабатывающего завода.
Расчёты уже выполнила платформа: модели качества, прогноз, оптимизатор режима, ядро баланса. Они верны.
Твоя задача — пересказать оператору готовые факты понятным языком инженера и коротко объяснить физическую причину.
Правила:
1. Используй только факты из блока «Факты». Не меняй числа, выводы и рекомендации, не добавляй новых чисел, не считай.
2. Если в фактах есть рекомендация (уставка, ремонт, поверка) — назови её дословно с тем же числом.
3. Пиши по-русски, 2–4 коротких предложения, без markdown, списков и вводных слов. Не повторяй служебные строки и время.
4. Физику можешь объяснить своими словами без чисел: вибрация компрессора даёт пульсации и брызгоунос, загрязнённый демистер пропускает капли, тёплая сепарация оставляет C3+ в товарном газе, холодная — растворяет метан в ШФЛУ.
5. Ты советуешь, решение принимает оператор. Не выдумывай оборудование, регламенты и людей.
6. Каждую рекомендацию называй один раз. Не перечисляй нормы и уставки, о которых не спрашивали, и не пересказывай, как устроен расчёт.
7. Говори только о том, что есть в фактах. Если вопрос вообще не про этот узел — скажи, что отвечаешь только по установке НТС-1."""

_log_lock = threading.Lock()
_llm_lock = threading.Lock()  # одна модель на ноутбуке — запросы по очереди


# ---------------------------------------------------------------- защита чисел
def _numbers(text: str) -> set[str]:
    norm = text.replace("−", "-").replace(",", ".")
    out = set()
    for m in re.findall(r"\d+(?:\.\d+)?", norm):
        out.add(m)
        if "." in m:  # «35.50» и «35.5» — одно и то же число
            out.add(m.rstrip("0").rstrip("."))
    return out


def extra_numbers(answer: str, allowed_text: str) -> list[str]:
    allowed = _numbers(allowed_text)
    bad = []
    for n in _numbers(answer):
        if n in allowed:
            continue
        if "." in n and n.rstrip("0").rstrip(".") in allowed:
            continue
        bad.append(n)
    return sorted(set(bad), key=lambda s: float(s))


# ---------------------------------------------------------------- Ollama
def ollama_chat(messages: list[dict], temperature: float = 0.2) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": "30m",
        "options": {"temperature": temperature, "num_predict": 260, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        OLLAMA + "/api/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    text = (data.get("message") or {}).get("content", "").strip()
    if not text:
        raise RuntimeError("пустой ответ модели")
    return text, data


def clean(text: str) -> str:
    """Убираем эхо служебных строк и markdown, если модель их повторила."""
    text = re.sub(r"^\s*(Время процесса|Факты|Вопрос оператора)[^.\n]*[.:]\s*", "", text, flags=re.I)
    text = text.replace("**", "").replace("### ", "")
    return text.strip()


def ollama_ok() -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(OLLAMA + "/api/tags", timeout=3) as resp:
            tags = json.loads(resp.read().decode("utf-8"))
        names = [m.get("name", "") for m in tags.get("models", [])]
        if MODEL in names:
            return True, MODEL
        return False, f"модель {MODEL} не найдена в Ollama"
    except Exception as e:  # noqa: BLE001
        return False, f"Ollama недоступна: {type(e).__name__}"


def ask_model(question: str, context: str, kind: str, fallback: str) -> dict:
    t0 = time.time()
    task = (
        "Сообщи оператору об этом событии." if kind == "alert" else f"Вопрос оператора: {question}"
    )
    user = f"Факты:\n{context}\n\n{task}\nОтветь по фактам выше, числа и рекомендации — без изменений."
    allowed = context + "\n" + question
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    guard = {"checked": True, "retried": False, "passed": True, "first_extra": []}
    try:
        with _llm_lock:
            text, data = ollama_chat(msgs, temperature=0.1)
            extra = extra_numbers(text, allowed)
            guard["first_extra"] = extra
            if extra:
                guard["retried"] = True
                msgs += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": (
                        f"В ответе есть числа {', '.join(extra)}, которых нет в данных узла. Это ошибка. "
                        "Перепиши ответ, используя только числа из блока данных, без арифметики."
                    )},
                ]
                text, data = ollama_chat(msgs, temperature=0.0)
                extra2 = extra_numbers(text, allowed)
                if extra2:
                    guard["passed"] = False
                    guard["second_extra"] = extra2
                    return {"text": fallback, "model": "шаблон (защита от выдуманных чисел)", "fallback": True,
                            "latency_s": round(time.time() - t0, 1), "guard": guard, "rejected": text}
        text = clean(text)
        return {"text": text, "model": MODEL, "fallback": False,
                "latency_s": round(time.time() - t0, 1), "guard": guard, "tokens": data.get("eval_count")}
    except Exception as e:  # noqa: BLE001 — демо не должно падать
        return {"text": fallback, "model": "шаблон (модель недоступна)", "fallback": True,
                "latency_s": round(time.time() - t0, 1), "guard": guard, "error": f"{type(e).__name__}: {e}"}


def write_log(entry: dict) -> None:
    with _log_lock:
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                log = json.load(f)
        except Exception:  # noqa: BLE001
            log = []
        log.append(entry)
        tmp = LOG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        os.replace(tmp, LOG_PATH)


# ---------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def end_headers(self):  # браузер не должен показывать старую версию страницы
        self.send_header("Cache-Control", "no-store, max-age=0")
        super().end_headers()

    def log_message(self, fmt, *args):  # тише в консоли
        if args and "/api/" in str(args[0]):
            sys.stdout.write("  " + (fmt % args) + "\n")

    def _json(self, code: int, obj: dict) -> None:
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/" + PAGE
        if self.path.startswith("/favicon"):  # у страницы нет иконки — отвечаем пусто, без ошибки
            self.send_response(204)
            self.end_headers()
            return
        if self.path.startswith("/api/health"):
            ok, info = ollama_ok()
            return self._json(200, {"online": ok, "model": MODEL if ok else None, "info": info})
        return super().do_GET()

    def do_POST(self):
        if not self.path.startswith("/api/ask"):
            return self._json(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:  # noqa: BLE001
            return self._json(400, {"error": "bad json"})
        question = str(req.get("question", ""))[:600]
        context = str(req.get("context", ""))[:9000]
        kind = "alert" if req.get("kind") == "alert" else "chat"
        fallback = str(req.get("fallback", "Модель недоступна."))[:3000]
        res = ask_model(question, context, kind, fallback)
        write_log({
            "time": datetime.now().isoformat(timespec="seconds"),
            "process_time": req.get("ptime"),
            "scenario": req.get("scenario"),
            "event": req.get("event"),
            "kind": kind,
            "question": question,
            "context": context,
            "answer": res["text"],
            "model": res["model"],
            "latency_s": res["latency_s"],
            "guard": res["guard"],
            "error": res.get("error"),
        })
        tag = "OK " if not res["fallback"] else "ШАБЛОН"
        print(f"  [{tag}] {kind}: {question[:60] or req.get('event')} · {res['latency_s']} с")
        return self._json(200, res)


def warm_up() -> None:
    ok, info = ollama_ok()
    if not ok:
        print(f"  ! {info}. Страница будет работать на шаблонных ответах.")
        return
    print("  Прогреваю модель (первый ответ бывает долгим)…")
    try:
        ollama_chat([{"role": "user", "content": "Ответь одним словом: готов?"}])
        print("  Модель готова.")
    except Exception as e:  # noqa: BLE001
        print(f"  ! прогрев не удался: {e}")


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    port = None
    for p in range(8765, 8776):
        try:
            srv = ThreadingHTTPServer(("127.0.0.1", p), Handler)
            port = p
            break
        except OSError:
            continue
    if port is None:
        print("Нет свободного порта 8765–8775.")
        sys.exit(1)
    url = f"http://127.0.0.1:{port}/"
    print("AI-PromPlatform · НТС-1")
    print(f"  Страница: {url}")
    print(f"  Модель:   {MODEL} через {OLLAMA}")
    print(f"  Журнал:   {LOG_PATH}")
    threading.Thread(target=warm_up, daemon=True).start()
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print("  Остановить: Ctrl+C или закрыть это окно.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
