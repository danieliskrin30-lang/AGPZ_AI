# -*- coding: utf-8 -*-
"""Прогон прототипа: обучение soft-sensor + оценка упреждения + демо-сценарий."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import quality_process as qp
import quality_model as qm

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_proto_report.txt")
buf = []


def w(*a):
    line = " ".join(str(x) for x in a)
    print(line)
    buf.append(line)


w("=" * 96)
w("1. ДАННЫЕ ПРОЦЕССА (90 суток, часовой шаг)")
w("=" * 96)
hist = qp.generate_plant_history(n_hours=2160, seed=11)
w("строк:", len(hist), "| часов вне нормы:", round(hist["off_spec"].mean() * 100, 2), "%")
for kpi in qp.QUALITY_KPI:
    s = hist[kpi]
    d, lim, unit = qp.QUALITY_LIMITS[kpi]
    bad = (s > lim) if d == "max" else (s < lim)
    w(f"  {kpi:20s} median={s.median():8.2f} min={s.min():8.2f} max={s.max():8.2f}  "
      f"норма {d} {lim} {unit}  вне нормы: {bad.mean()*100:5.1f}%")

w("")
w("=" * 96)
w("2. ОБУЧЕНИЕ SOFT-SENSOR (деление по времени: 70% прошлое / 30% будущее)")
w("=" * 96)
metrics = qm.train_and_save(force=True, n_hours=2160, seed=11)
w("признаков:", metrics["dataset"]["features"],
  "| обучение:", metrics["dataset"]["train"], "| проверка:", metrics["dataset"]["test"],
  "| доля вне нормы:", metrics["dataset"]["offspec_share_pct"], "%")
for kpi in qp.QUALITY_KPI:
    m = metrics[kpi]
    _d, lim, unit = qp.QUALITY_LIMITS[kpi]
    w(f"  {kpi:20s} MAE={m['mae']:8.3f} {unit:4s}  R2={m['r2']:7.4f}  "
      f"в пределах 10% от значения: {m['bool_acc_within_10pct']*100:5.1f}%")
w(f"  {'часов до нарушения':20s} MAE={metrics['hours_to_offspec']['mae_hours']:6.2f} ч  R2={metrics['hours_to_offspec']['r2']:7.4f}")
o = metrics["offspec_8h"]
o24 = metrics["offspec_24h"]
w(f"  {'нарушение в 8 ч':20s} AUC={o['auc'] if o['auc'] is None else round(o['auc'],4)}  "
  f"precision={o['precision']:.3f}  recall={o['recall']:.3f}  F1={o['f1']:.3f}  (позитивов в тесте: {o['positives_test']})")
w(f"  {'нарушение в 24 ч':20s} AUC={o24['auc'] if o24['auc'] is None else round(o24['auc'],4)}  "
  f"precision={o24['precision']:.3f}  recall={o24['recall']:.3f}  F1={o24['f1']:.3f}  (позитивов в тесте: {o24['positives_test']})")

w("")
w("=" * 96)
w("3. УПРЕЖДЕНИЕ НА ВСЕЙ ИСТОРИИ (плановый горизонт 24 ч)")
w("=" * 96)
pred = qm.predict_frame(hist)
test = pred.iloc[int(len(pred) * 0.70):].reset_index(drop=True)
off = test["off_spec"].to_numpy()
starts = [i for i in range(1, len(off)) if off[i] == 1 and off[i - 1] == 0]
if off[0] == 1:
    starts.insert(0, 0)
leads, missed = [], 0
for s in starts:
    lo = max(0, s - 48)
    hit = [i for i in range(lo, s) if test["prob_offspec_24h"].iloc[i] >= 0.5]
    if hit:
        leads.append(s - hit[0])
    else:
        missed += 1
w(f"эпизодов выхода за норму в тестовой части: {len(starts)}")
w(f"предупреждено заранее: {len(leads)} из {len(starts)} (пропущено: {missed})")
if leads:
    w(f"упреждение: медиана {np.median(leads):.1f} ч, минимум {min(leads)} ч, максимум {max(leads)} ч")
alarm_hours = int((test["prob_offspec_24h"] >= 0.5).sum())
w(f"часов с активным плановым предупреждением: {alarm_hours} из {len(test)} ({alarm_hours/len(test)*100:.1f}%)")

w("")
w("=" * 96)
w("4. ДЕМО-СЦЕНАРИЙ: разнос вибрации -> потеря качества")
w("=" * 96)
demo = qm.predict_frame(qp.generate_demo_trajectory("QUALITY_RISK", n_hours=48))
first_alarm = next((int(r["hour"]) for _, r in demo.iterrows() if r["prob_offspec_8h"] >= 0.5), None)
first_bad = next((i for i in range(len(demo))
                  if any((demo.iloc[i][k] > lim) if d == "max" else (demo.iloc[i][k] < lim)
                         for k, (d, lim, _u) in qp.QUALITY_LIMITS.items())), None)
w(f"{'час':>4} {'вибр':>6} {'C3+ факт':>9} {'C3+ прогноз':>12} {'H2S':>7} {'т.росы':>8} {'ШФЛУ':>7} {'P(норма)':>9} {'до нар.':>8}")
for i, r in demo.iterrows():
    bad = any((r[k] > lim) if d == "max" else (r[k] < lim) for k, (d, lim, _u) in qp.QUALITY_LIMITS.items())
    marker = " <-- НОРМА НАРУШЕНА" if bad else ""
    if int(r["hour"]) % 4 == 0 or (first_alarm is not None and abs(int(r["hour"]) - first_alarm) <= 1):
        w(f"{int(r['hour']):>4} {r['vib_comp_mms']:>6.2f} {r['c3_carryover_g_m3']:>9.3f} "
          f"{r['pred_c3_carryover_g_m3']:>12.3f} {r['h2s_out_ppm']:>7.2f} {r['dew_point_c']:>8.1f} "
          f"{r['shflu_purity_pct']:>7.2f} {r['prob_offspec_8h']*100:>8.1f}% {r['pred_hours_to_offspec']:>8.1f}{marker}")
w("")
w("первое предупреждение ИИ (P>=50%):", f"час {first_alarm}")
w("первое фактическое нарушение нормы: ", f"час {first_bad}")
if first_alarm is not None and first_bad is not None:
    w("УПРЕЖДЕНИЕ ДО НАРУШЕНИЯ:", f"{first_bad - first_alarm} ч")

w("")
w("=" * 96)
w("5. ОБЪЯСНЕНИЕ (SHAP) на пике сценария + деньги")
w("=" * 96)
peak_row = demo.iloc[int(demo["prob_offspec_8h"].idxmax())]
state = {k: float(peak_row[k]) for k in qp.FEATURES_MEASURED}
verdict = qm.quality_verdict(state)
for k, v in verdict["shap_top"].items():
    w(f"  {k:<48} {v:>5}%")
w("качество (прогноз):", json.dumps(verdict["quality"], ensure_ascii=False))
w("потери:", f"{verdict['loss_tenge_per_hour']:,} ₸/час", "|", f"{verdict['loss_tenge_per_day']:,} ₸/сутки")

w("")
w("=" * 96)
w("6. КОНТРОЛЬ: штатный режим и режим после устранения")
w("=" * 96)
for mode in ("NORMAL", "FIXED"):
    d = qm.predict_frame(qp.generate_demo_trajectory(mode, n_hours=48)).iloc[-1]
    w(f"  {mode:8s} C3+={d['c3_carryover_g_m3']:.3f} (прогноз {d['pred_c3_carryover_g_m3']:.3f})  "
      f"H2S={d['h2s_out_ppm']:.2f}  т.росы={d['dew_point_c']:.1f}  ШФЛУ={d['shflu_purity_pct']:.2f}  "
      f"P(нарушение 8ч)={d['prob_offspec_8h']*100:.1f}%")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(buf))
print("\nОТЧЁТ:", OUT)
