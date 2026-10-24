"""
Генератор физической телеметрии промышленного оборудования цеха.
Моделирует 4 ключевых узла завода (насос, электродвигатель, компрессор, привод)
с физически связанными параметрами: вибрация (СКЗ и эксцесс), температура, давление, ток и обороты.
"""

import os
import numpy as np
import pandas as pd


UNITS_CONFIG = {
    "PUMP-101": {
        "name": "Насосный агрегат ЦНС-180",
        "type": "Центробежный секционный насос",
        "shop": "Цех водоснабжения и перекачки №1",
        "purpose": "Отвечает за непрерывную подачу технической жидкости и охлаждающей воды под высоким давлением в магистраль цеха.",
        "what_breaks": "Чаще всего страдает от кавитации (падения давления на входе), износа подшипников вала и разгерметизации торцевого уплотнения.",
        "default_fault_id": 4,  # Падение давления / кавитация
        "norm_vibration": 2.1,      # мм/с
        "scada_crit_vib": 7.1,      # мм/с (порог аварийного останова SCADA)
        "norm_temp": 52.0,          # °C
        "scada_crit_temp": 85.0,    # °C
        "norm_pressure": 14.5,      # бар
        "scada_min_pressure": 9.5,  # бар
        "norm_current": 92.0,       # А
        "scada_crit_current": 135.0,# А
        "norm_rpm": 2950.0,
        "downtime_cost_per_hour": 1_800_000,  # тенге/час простоя линии
        "replacement_cost": 14_500_000,       # тенге (стоимость агрегата при разрушении)
        "preventive_repair_cost": 185_000,    # тенге (плановая замена подшипника/уплотнения)
    },
    "MOTOR-204": {
        "name": "Электродвигатель АИР-315S4",
        "type": "Силовой асинхронный привод 160 кВт",
        "shop": "Главный производственный корпус",
        "purpose": "Главный силовой электродвигатель цеха, вращающий рабочие агрегаты технологической линии.",
        "what_breaks": "Уязвим к износу опорных подшипников ротора (рост вибрации) и перегреву медной обмотки статора при нарушении охлаждения.",
        "default_fault_id": 1,  # Износ подшипника
        "norm_vibration": 1.9,
        "scada_crit_vib": 6.8,
        "norm_temp": 56.0,
        "scada_crit_temp": 90.0,
        "norm_pressure": 14.0,
        "scada_min_pressure": 9.0,
        "norm_current": 96.0,
        "scada_crit_current": 140.0,
        "norm_rpm": 1485.0,
        "downtime_cost_per_hour": 2_400_000,
        "replacement_cost": 18_000_000,
        "preventive_repair_cost": 220_000,
    },
    "COMP-302": {
        "name": "Компрессор ВП3-20/9",
        "type": "Компрессорная установка высокого давления",
        "shop": "Компрессорная станция №2",
        "purpose": "Сжимает и подает рабочий газ/воздух под давлением в пневмосистему и технологические реакторы завода.",
        "what_breaks": "Критичен к перегреву цилиндров и клапанов, а также к скачкам давления и вибрации поршневой группы.",
        "default_fault_id": 2,  # Перегрев узла
        "norm_vibration": 2.3,
        "scada_crit_vib": 7.5,
        "norm_temp": 58.0,
        "scada_crit_temp": 88.0,
        "norm_pressure": 15.0,
        "scada_min_pressure": 10.0,
        "norm_current": 98.0,
        "scada_crit_current": 145.0,
        "norm_rpm": 1470.0,
        "downtime_cost_per_hour": 3_200_000,
        "replacement_cost": 28_000_000,
        "preventive_repair_cost": 340_000,
    },
    "DRIVE-405": {
        "name": "Привод конвейера ПЛ-12",
        "type": "Мотор-редуктор конвейерной линии",
        "shop": "Линия транспортировки и отгрузки",
        "purpose": "Тяговый привод главного конвейера, перемещающего сырье и готовую продукцию между участками завода.",
        "what_breaks": "Подвержен механическим перегрузкам по току при застревании/перегрузе ленты и перекосу зубчатой передачи редуктора.",
        "default_fault_id": 3,  # Перегрузка по току
        "norm_vibration": 2.2,
        "scada_crit_vib": 7.2,
        "norm_temp": 54.0,
        "scada_crit_temp": 86.0,
        "norm_pressure": 14.2,
        "scada_min_pressure": 9.2,
        "norm_current": 94.0,
        "scada_crit_current": 138.0,
        "norm_rpm": 980.0,
        "downtime_cost_per_hour": 1_500_000,
        "replacement_cost": 9_500_000,
        "preventive_repair_cost": 120_000,
    },
}

FAULT_CLASSES = {
    0: {
        "name": "Норма (Штатный режим)",
        "short": "Норма",
        "action": "Продолжать штатную эксплуатацию. Все показатели в пределах зеленого коридора.",
        "fix_button": "✅ Узел работает в норме",
        "fix_report": "Параметры стабильны, вмешательство не требуется.",
        "parts": "Не требуются",
        "brigade": "Дежурный оператор АСУ ТП",
    },
    1: {
        "name": "Износ подшипника качения (рост микроударов и вибрации)",
        "short": "Износ подшипника",
        "action": "Перевести нагрузку на резервный агрегат, выполнить замену подшипника и лазерную центровку вала.",
        "fix_button": "🛠️ Переключить на резерв и заменить подшипниковый узел",
        "fix_report": "Нагрузка переведена на резерв, изношенный подшипник заменен. Вибрация снижена с 5.6 до 2.1 мм/с!",
        "parts": "Подшипник SKF 6314-2Z + высокотемпературная смазка",
        "brigade": "Слесарь-ремонтник + инженер по вибродиагностике",
    },
    2: {
        "name": "Перегрев обмотки / узла трения (нарушение теплоотвода)",
        "short": "Перегрев узла",
        "action": "Включить резервный контур принудительного охлаждения, прочистить теплообменник и снизить нагрузку на 15%.",
        "fix_button": "🛠️ Включить резервное охлаждение и стабилизировать температуру",
        "fix_report": "Резервный контур охлаждения запущен, теплообменник очищен. Температура снижена с 79.5 °C до 54.0 °C!",
        "parts": "Модуль обдува / Термопаста / Фильтр охлаждения",
        "brigade": "Электромонтер КИПиА и дежурный энергетик цеха",
    },
    3: {
        "name": "Перегрузка по току / подклинивание привода",
        "short": "Перегрузка по току",
        "action": "Автоматически разгрузить подачу на 20%, устранить перекос муфты сцепления и выровнять рабочий ток.",
        "fix_button": "🛠️ Разгрузить привод и выровнять ток электродвигателя",
        "fix_report": "Подача сырья скорректирована, заклинивание муфты устранено. Ток двигателя выровнен со 128 А до 95 А!",
        "parts": "Эластичная вставка муфты / Контактор пускателя",
        "brigade": "Механик цеха + инженер АСУ ТП",
    },
    4: {
        "name": "Падение давления в магистрали / кавитация",
        "short": "Падение давления (кавитация)",
        "action": "Открыть перепускной клапан выравнивания давления, подтянуть фланец всасывающего коллектора и заменить уплотнение.",
        "fix_button": "🛠️ Выровнять давление клапаном и устранить утечку",
        "fix_report": "Перепускной клапан открыт, герметичность восстановлена. Давление выровнено с 10.2 до 14.5 бар, кавитация прекращена!",
        "parts": "Торцевое уплотнение вала / Ремкомплект клапана",
        "brigade": "Слесарь технологических установок",
    },
}


def generate_training_dataset(n_samples: int = 4000, seed: int = 42) -> pd.DataFrame:
    """
    Генерирует репрезентативный табличный датасет показаний датчиков для обучения моделей ИИ.
    """
    rng = np.random.default_rng(seed)

    probs = [0.48, 0.13, 0.13, 0.13, 0.13]
    fault_types = rng.choice([0, 1, 2, 3, 4], size=n_samples, p=probs)

    rows = []
    for f_type in fault_types:
        if f_type == 0:
            severity = rng.uniform(0.0, 0.28)
            vib_rms = 2.1 + severity * 2.4 + rng.normal(0, 0.32)
            vib_kurtosis = 2.9 + severity * 1.6 + rng.normal(0, 0.28)
            temp_c = 54.0 + severity * 14.0 + rng.normal(0, 2.4)
            pressure_bar = 14.5 - severity * 1.1 + rng.normal(0, 0.38)
            current_a = 95.0 + severity * 12.0 + rng.normal(0, 3.2)
            rpm_dev_pct = rng.normal(0.0, 0.55) - severity * 0.65
            rul_hours = 240.0 - severity * 290.0 + rng.normal(0, 4.5)
            rul_hours = np.clip(rul_hours, 140.0, 240.0)

        elif f_type == 1:
            severity = rng.uniform(0.22, 1.0)
            vib_rms = 3.0 + severity * 6.2 + rng.normal(0, 0.42)
            vib_kurtosis = 3.9 + severity * 6.5 + rng.normal(0, 0.48)
            temp_c = 58.0 + severity * 22.0 + rng.normal(0, 2.5)
            pressure_bar = 14.2 - severity * 1.4 + rng.normal(0, 0.35)
            current_a = 98.0 + severity * 18.0 + rng.normal(0, 3.0)
            rpm_dev_pct = -0.4 - severity * 2.2 + rng.normal(0, 0.45)
            rul_hours = 135.0 * (1.0 - severity) ** 1.15 + 3.0 + rng.normal(0, 2.5)
            rul_hours = np.clip(rul_hours, 2.0, 135.0)

        elif f_type == 2:
            severity = rng.uniform(0.22, 1.0)
            vib_rms = 2.4 + severity * 2.4 + rng.normal(0, 0.38)
            vib_kurtosis = 3.1 + severity * 1.6 + rng.normal(0, 0.35)
            temp_c = 65.0 + severity * 35.0 + rng.normal(0, 2.8)
            pressure_bar = 14.1 - severity * 1.2 + rng.normal(0, 0.35)
            current_a = 103.0 + severity * 26.0 + rng.normal(0, 3.5)
            rpm_dev_pct = -0.5 - severity * 2.0 + rng.normal(0, 0.45)
            rul_hours = 125.0 * (1.0 - severity) ** 1.2 + 2.5 + rng.normal(0, 2.5)
            rul_hours = np.clip(rul_hours, 2.0, 125.0)

        elif f_type == 3:
            severity = rng.uniform(0.22, 1.0)
            vib_rms = 2.7 + severity * 3.8 + rng.normal(0, 0.40)
            vib_kurtosis = 3.3 + severity * 2.2 + rng.normal(0, 0.38)
            temp_c = 61.0 + severity * 24.0 + rng.normal(0, 2.5)
            pressure_bar = 13.8 - severity * 2.2 + rng.normal(0, 0.40)
            current_a = 114.0 + severity * 54.0 + rng.normal(0, 4.2)
            rpm_dev_pct = -1.8 - severity * 6.5 + rng.normal(0, 0.65)
            rul_hours = 110.0 * (1.0 - severity) ** 1.25 + 2.0 + rng.normal(0, 2.2)
            rul_hours = np.clip(rul_hours, 1.5, 115.0)

        else:
            severity = rng.uniform(0.22, 1.0)
            vib_rms = 2.8 + severity * 4.5 + rng.normal(0, 0.40)
            vib_kurtosis = 3.6 + severity * 3.8 + rng.normal(0, 0.42)
            temp_c = 56.0 + severity * 16.0 + rng.normal(0, 2.2)
            pressure_bar = 12.6 - severity * 5.8 + rng.normal(0, 0.45)
            current_a = 91.0 - severity * 12.0 + rng.normal(0, 3.2)
            rpm_dev_pct = 0.6 + severity * 3.2 + rng.normal(0, 0.55)
            rul_hours = 120.0 * (1.0 - severity) ** 1.15 + 3.0 + rng.normal(0, 2.5)
            rul_hours = np.clip(rul_hours, 2.0, 120.0)

        vib_temp_index = (vib_rms * temp_c) / 100.0
        power_load_factor = (current_a * pressure_bar) / 1000.0
        crest_factor = 1.414 + (vib_kurtosis - 3.0) * 0.38

        rows.append({
            "vibration_rms": round(float(max(0.5, vib_rms)), 3),
            "vibration_kurtosis": round(float(max(2.0, vib_kurtosis)), 3),
            "crest_factor": round(float(max(1.2, crest_factor)), 3),
            "temperature_c": round(float(temp_c), 2),
            "pressure_bar": round(float(max(1.5, pressure_bar)), 2),
            "current_a": round(float(max(30.0, current_a)), 2),
            "rpm_dev_pct": round(float(rpm_dev_pct), 2),
            "vib_temp_index": round(float(vib_temp_index), 3),
            "power_load_factor": round(float(power_load_factor), 3),
            "fault_class": int(f_type),
            "is_anomaly": int(1 if f_type != 0 else 0),
            "rul_hours": round(float(rul_hours), 1),
        })

    return pd.DataFrame(rows)


def generate_dynamic_trajectory(
    mode: str = "NORMAL",
    fault_id: int = 1,
    n_points: int = 60,
    seed: int = 101,
) -> pd.DataFrame:
    """
    Генерирует наглядную динамику работы узла (60 временных точек):
    - mode == "NORMAL": ровная работа в зеленом коридоре.
    - mode == "THREAT": после 25-й точки начинает развиваться выбранная угроза (fault_id 1..4).
    - mode == "FIXED": угроза развивалась с 20-й по 42-ю точку, на 42-й точке оператор нажал
      «Исправить ошибку», и показатели плавно выровнялись обратно в зеленую норму!
    """
    rng = np.random.default_rng(seed + fault_id * 7)
    steps = np.arange(1, n_points + 1)

    records = []
    for i, step in enumerate(steps):
        if mode == "NORMAL":
            sev = 0.04 + 0.02 * np.sin(i / 4.0)
            phase = "Штатная работа"
        elif mode == "THREAT":
            if i < 24:
                sev = 0.04 + 0.02 * np.sin(i / 4.0)
                phase = "Штатная работа"
            else:
                prog = (i - 24) / float(n_points - 24)
                sev = 0.05 + 0.56 * (prog ** 1.25)
                phase = "Развитие угрозы (ИИ-перехват)"
        else:  # FIXED (Угроза возникла -> перехвачена ИИ -> исправлена кнопкой стабилизации!)
            if i < 18:
                sev = 0.04 + 0.02 * np.sin(i / 4.0)
                phase = "Штатная работа"
            elif i < 42:
                prog = (i - 18) / 24.0
                sev = 0.05 + 0.54 * (prog ** 1.25)
                phase = "Развитие угрозы"
            else:
                # Плавное выравнивание в норму после нажатия кнопки «Исправить ошибку»
                decay = np.exp(-(i - 42) / 4.2)
                sev = 0.04 + 0.55 * decay
                phase = "Стабилизация после исправления"

        # Формируем физическую картину в зависимости от типа угрозы (fault_id)
        if fault_id == 1:  # Износ подшипника
            vib_rms = 2.1 + sev * 6.0 + rng.normal(0, 0.08)
            vib_kurtosis = 2.95 + sev * 6.5 + rng.normal(0, 0.10)
            temp_c = 53.5 + sev * 21.0 + rng.normal(0, 0.4)
            pressure_bar = 14.5 - sev * 1.1 + rng.normal(0, 0.08)
            current_a = 94.5 + sev * 17.0 + rng.normal(0, 0.7)
            rpm_dev_pct = -0.15 - sev * 1.9 + rng.normal(0, 0.1)
        elif fault_id == 2:  # Перегрев узла
            vib_rms = 2.15 + sev * 2.2 + rng.normal(0, 0.08)
            vib_kurtosis = 3.0 + sev * 1.5 + rng.normal(0, 0.08)
            temp_c = 54.0 + sev * 44.0 + rng.normal(0, 0.5)
            pressure_bar = 14.4 - sev * 1.0 + rng.normal(0, 0.08)
            current_a = 95.0 + sev * 27.0 + rng.normal(0, 0.8)
            rpm_dev_pct = -0.2 - sev * 1.8 + rng.normal(0, 0.1)
        elif fault_id == 3:  # Перегрузка по току / заклинивание
            vib_rms = 2.2 + sev * 3.6 + rng.normal(0, 0.09)
            vib_kurtosis = 3.05 + sev * 2.1 + rng.normal(0, 0.09)
            temp_c = 54.5 + sev * 25.0 + rng.normal(0, 0.4)
            pressure_bar = 14.3 - sev * 1.9 + rng.normal(0, 0.09)
            current_a = 95.0 + sev * 62.0 + rng.normal(0, 1.1)
            rpm_dev_pct = -0.3 - sev * 6.2 + rng.normal(0, 0.15)
        else:  # 4: Падение давления / кавитация
            vib_rms = 2.15 + sev * 4.4 + rng.normal(0, 0.09)
            vib_kurtosis = 3.0 + sev * 3.8 + rng.normal(0, 0.10)
            temp_c = 53.5 + sev * 14.0 + rng.normal(0, 0.4)
            pressure_bar = 14.5 - sev * 7.5 + rng.normal(0, 0.10)
            current_a = 94.0 - sev * 14.0 + rng.normal(0, 0.8)
            rpm_dev_pct = 0.2 + sev * 3.0 + rng.normal(0, 0.12)

        crest_factor = 1.414 + (vib_kurtosis - 3.0) * 0.38
        vib_temp_index = (vib_rms * temp_c) / 100.0
        power_load_factor = (current_a * pressure_bar) / 1000.0

        records.append({
            "step": int(step),
            "hour_offset": int(step - n_points),
            "phase": phase,
            "vibration_rms": round(float(vib_rms), 2),
            "vibration_kurtosis": round(float(vib_kurtosis), 2),
            "crest_factor": round(float(crest_factor), 2),
            "temperature_c": round(float(temp_c), 1),
            "pressure_bar": round(float(pressure_bar), 2),
            "current_a": round(float(current_a), 1),
            "rpm_dev_pct": round(float(rpm_dev_pct), 2),
            "vib_temp_index": round(float(vib_temp_index), 3),
            "power_load_factor": round(float(power_load_factor), 3),
        })

    return pd.DataFrame(records)


def generate_time_series_scenario(
    scenario: str = "early_warning",
    n_points: int = 60,
    seed: int = 101,
) -> pd.DataFrame:
    if scenario == "normal":
        return generate_dynamic_trajectory("NORMAL", fault_id=1, n_points=n_points, seed=seed)
    elif scenario == "fixed":
        return generate_dynamic_trajectory("FIXED", fault_id=1, n_points=n_points, seed=seed)
    else:
        return generate_dynamic_trajectory("THREAT", fault_id=1, n_points=n_points, seed=seed)


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    df_train = generate_training_dataset(n_samples=4000, seed=42)
    train_path = os.path.join(data_dir, "industrial_sensor_dataset.csv")
    df_train.to_csv(train_path, index=False, encoding="utf-8")

    df_sample_shop = generate_dynamic_trajectory("THREAT", fault_id=1, n_points=60)
    sample_path = os.path.join(data_dir, "sample_shop_telemetry.csv")
    df_sample_shop.to_csv(sample_path, index=False, encoding="utf-8")

    print(f"Dataset generated: {len(df_train)} rows -> {train_path}")
    print(f"Sample telemetry generated: {len(df_sample_shop)} rows -> {sample_path}")
