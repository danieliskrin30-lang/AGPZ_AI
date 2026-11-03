# -*- coding: utf-8 -*-
"""Прототип ядра «Сквозной материальный баланс»: согласование данных (DVR) + поиск неисправного расходомера.
Граф установки -> матрица инцидентности A -> взвешенный МНК с ограничением A x = 0.
"""
import numpy as np
from scipy import stats

# Узлы и потоки ГПЗ (т/ч). (откуда, куда); None = граница установки
STREAMS = {
    "F01 сырой газ":            (None, "SEP"),
    "F02 газ после сепарации":  ("SEP", "AMINE"),
    "F03 конденсат":            ("SEP", None),
    "F04 пластовая вода":       ("SEP", None),
    "F05 очищенный газ":        ("AMINE", "DEHY"),
    "F06 кислый газ":           ("AMINE", "CLAUS"),
    "F07 сера":                 ("CLAUS", None),
    "F08 хвостовые газы":       ("CLAUS", None),
    "F09 осушенный газ":        ("DEHY", "NTS"),
    "F10 вода из осушки":       ("DEHY", None),
    "F11 товарный газ":         ("NTS", None),
    "F12 ШФЛУ":                 ("NTS", None),
    "F13 топливный газ":        ("NTS", None),
    "F14 факел":                ("NTS", None),
}
NODES = ["SEP", "AMINE", "CLAUS", "DEHY", "NTS"]
names = list(STREAMS)

A = np.zeros((len(NODES), len(names)))
for j, (src, dst) in enumerate(STREAMS.values()):
    if src: A[NODES.index(src), j] -= 1
    if dst: A[NODES.index(dst), j] += 1

# «истина» (согласована по балансу)
true = np.array([600, 560, 32, 8, 470, 90, 82, 8, 466, 4, 405, 45, 12, 4], float)
assert np.allclose(A @ true, 0)

# погрешность приборов (σ, доля от значения): факел меряется плохо
rel = np.array([.010, .010, .015, .02, .010, .015, .010, .05, .010, .05, .010, .015, .02, .10])
sigma = rel * true
rng = np.random.default_rng(3)
meas = true + rng.normal(0, sigma)
BAD = names.index("F05 очищенный газ")
meas[BAD] *= 1.04          # неисправный (смещённый) расходомер +4%


def reconcile(y, s, exclude=()):
    """WLS: min Σ((x-y)/σ)^2  при A x = 0. Исключённые приборы -> σ огромная (поток становится неизмеряемым)."""
    s = s.copy()
    for k in exclude: s[k] = 1e6
    V = np.diag(s ** 2)
    r = A @ y                                  # невязки узлов до согласования
    S = A @ V @ A.T
    Sinv = np.linalg.inv(S)
    x = y - V @ A.T @ Sinv @ r
    chi2 = float(r @ Sinv @ r)                 # глобальный тест
    dof = np.linalg.matrix_rank(A) - 0
    p = 1 - stats.chi2.cdf(chi2, dof)
    # нормированные поправки (measurement test)
    W = V @ A.T @ Sinv @ A @ V
    adj = y - x
    z = np.where(np.diag(W) > 1e-12, np.abs(adj) / np.sqrt(np.maximum(np.diag(W), 1e-12)), 0)
    for k in exclude: z[k] = 0
    # ковариация согласованных значений -> неопределённость
    Vx = V - W
    return x, r, chi2, p, z, np.sqrt(np.clip(np.diag(Vx), 0, None))


print("Невязка по узлам ДО согласования, т/ч:", dict(zip(NODES, np.round(A @ meas, 2))))
x, r, chi2, p, z, sx = reconcile(meas, sigma)
print(f"\nГлобальный тест χ²={chi2:.1f}, p={p:.4f} -> {'ЕСТЬ грубая ошибка' if p < 0.05 else 'ок'}")
worst = int(np.argmax(z))
print("Подозреваемый прибор:", names[worst], f"(нормированная поправка {z[worst]:.1f}σ)")

x2, r2, chi22, p2, z2, sx2 = reconcile(meas, sigma, exclude=(worst,))
print(f"После исключения прибора: χ²={chi22:.2f}, p={p2:.3f}")
print(f"Оценка смещения прибора: {(meas[worst] / x2[worst] - 1) * 100:+.1f}%  (заложено +4.0%)")

print("\nПоток                      замер   согласовано  ±σ     истина")
for j, n in enumerate(names):
    sd = "  искл." if j == worst else f"{sx2[j]:6.2f}"
    print(f"{n:26s} {meas[j]:7.1f} {x2[j]:10.1f} {sd} {true[j]:8.1f}")
print("Невязка ПОСЛЕ, т/ч:", np.round(A @ x2, 6))
fl = names.index("F14 факел")
print(f"\nФакел: прибор {meas[fl]:.1f} т/ч (±{sigma[fl]:.1f}) -> согласовано {x2[fl]:.1f} ±{sx2[fl]:.1f} т/ч")


# ---- Монте-Карло: как часто ядро находит неисправный прибор ----
def trial(seed, bad_name, bias):
    g = np.random.default_rng(seed)
    y = true + g.normal(0, sigma)
    b = names.index(bad_name); y[b] *= 1 + bias
    _, _, c2, pp, zz, _ = reconcile(y, sigma)
    return pp < 0.05, int(np.argmax(zz)) == b

for bad_name in ["F02 газ после сепарации", "F05 очищенный газ", "F06 кислый газ", "F11 товарный газ"]:
    for bias in (0.02, 0.04):
        res = [trial(s, bad_name, bias) for s in range(1000)]
        det = np.mean([a for a, _ in res]) * 100; ident = np.mean([b for _, b in res]) * 100
        print(f"{bad_name:24s} смещение {bias*100:.0f}%: тревога {det:5.1f}%  верно указан прибор {ident:5.1f}%")
res = [reconcile(true + np.random.default_rng(s).normal(0, sigma), sigma)[3] < 0.05 for s in range(1000)]
print(f"Ложные тревоги без неисправности: {np.mean(res)*100:.1f}%")
