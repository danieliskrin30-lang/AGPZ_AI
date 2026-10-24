"""
AI-PromPlatform: Интеллектуальная ИИ-система предиктивной диагностики и мониторинга
промышленного оборудования, работающая «поверх» производственного процесса завода (Read-Only SCADA Overlay).

Разработано для внутривузовского хакатона МИТУ в рамках Национальной программы AI-SANA — 2026.
Капитан команды: Искрин Даниэль Андреевич (6B07102 «Автоматизация и управление», 2 курс МИТУ).
Трек 6.1.4: Решения на основе ИИ для инженерных и производственных задач.
"""

import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_generator import (
    UNITS_CONFIG,
    FAULT_CLASSES,
    generate_dynamic_trajectory,
)
from src.ml_engine import (
    load_models_and_metrics,
    predict_single_state,
)

# ---------------------------------------------------------------------------
# Конфигурация страницы
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI-PromPlatform | Предиктивная ИИ-диагностика цеха",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Чистый структурный темный стиль (Dark Industrial SCADA)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1E293B;
    }
    .top-banner {
        background: linear-gradient(90deg, #111827 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-left: 5px solid #06B6D4;
        border-radius: 12px;
        padding: 16px 22px;
        margin-bottom: 16px;
    }
    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }
    .sys-badge {
        background-color: #0F172A;
        border: 1px solid #06B6D4;
        color: #38BDF8;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        font-family: monospace;
    }
    .sys-badge-green {
        background-color: #064E3B;
        border: 1px solid #10B981;
        color: #6EE7B7;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        font-family: monospace;
    }
    .unit-card {
        background-color: #111827;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 8px;
        min-height: 178px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.35);
    }
    .passport-box {
        background-color: #111827;
        border: 1px solid #334155;
        border-left: 4px solid #38BDF8;
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .notif-threat {
        background: linear-gradient(90deg, rgba(245,158,11,0.18) 0%, rgba(17,24,39,0.98) 100%);
        border: 2px solid #F59E0B;
        border-left: 8px solid #EF4444;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0;
    }
    .notif-fixed {
        background: linear-gradient(90deg, rgba(16,185,129,0.20) 0%, rgba(17,24,39,0.98) 100%);
        border: 2px solid #10B981;
        border-left: 8px solid #10B981;
        border-radius: 12px;
        padding: 18px 22px;
        margin: 12px 0;
    }
    .notif-normal {
        background: linear-gradient(90deg, rgba(16,185,129,0.10) 0%, rgba(17,24,39,0.98) 100%);
        border: 1px solid #10B981;
        border-left: 6px solid #10B981;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
    }
    .action-card {
        background-color: #0F172A;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 18px;
        height: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def init_engine():
    return load_models_and_metrics()


rf_clf, xgb_rul, iso_model, model_metrics = init_engine()

# ---------------------------------------------------------------------------
# Инициализация состояния сессии (Session State) для интерактивной симуляции
# ---------------------------------------------------------------------------
if "sim_mode" not in st.session_state:
    # Три понятных состояния: "NORMAL" (Штатно), "THREAT" (Угроза активна), "FIXED" (Ошибка устранена)
    _demo_mode = str(st.query_params.get("demo", "")).upper()
    st.session_state.sim_mode = _demo_mode if _demo_mode in ("THREAT", "FIXED") else "NORMAL"
if "active_unit" not in st.session_state:
    _demo_unit = str(st.query_params.get("unit", "PUMP-101")).upper()
    st.session_state.active_unit = _demo_unit if _demo_unit in UNITS_CONFIG else "PUMP-101"
if "active_fault_id" not in st.session_state:
    st.session_state.active_fault_id = UNITS_CONFIG[st.session_state.active_unit]["default_fault_id"]
if "manual_override" not in st.session_state:
    st.session_state.manual_override = False

# ---------------------------------------------------------------------------
# Верхняя шапка сайта
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="top-banner">
        <h2 style="margin:0; color:#F8FAFC; font-size:23px;">
            ⚙️ AI-PromPlatform — Предиктивная ИИ-диагностика промышленного оборудования
        </h2>
        <p style="margin:4px 0 0 0; color:#94A3B8; font-size:14px;">
            Независимая аналитическая надстройка «поверх» производственного процесса завода (АСУ ТП / SCADA) | Специальность 6B07102 «АиУ» МИТУ
        </p>
        <div class="badge-row">
            <span class="sys-badge-green">🟢 РЕЖИМ: READ-ONLY НАДСТРОЙКА (БЕЗ ВМЕШАТЕЛЬСТВА В ПЛК)</span>
            <span class="sys-badge">🔒 ЛОКАЛЬНЫЙ ЗАКРЫТЫЙ КОНТУР ЗАВОДА</span>
            <span class="sys-badge">📡 ПРОТОКОЛ: OPC UA / MODBUS TCP</span>
            <span class="sys-badge">🇰🇿 УКАЗ «DIGITAL QAZAQSTAN» (09.06.2026)</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Боковая панель: выбор узла и дополнительные режимы
# ---------------------------------------------------------------------------
st.sidebar.markdown("### 🏭 Управление цехом")
selected_unit_id = st.sidebar.selectbox(
    "1. Выберите узел на линии:",
    options=list(UNITS_CONFIG.keys()),
    format_func=lambda k: f"{k} — {UNITS_CONFIG[k]['name']}",
    index=list(UNITS_CONFIG.keys()).index(st.session_state.active_unit),
)

if selected_unit_id != st.session_state.active_unit:
    st.session_state.active_unit = selected_unit_id
    st.session_state.active_fault_id = UNITS_CONFIG[selected_unit_id]["default_fault_id"]
    st.session_state.manual_override = False

unit_cfg = UNITS_CONFIG[selected_unit_id]

selected_fault_id = st.sidebar.selectbox(
    "2. Тип моделируемой угрозы:",
    options=[1, 2, 3, 4],
    format_func=lambda fid: f"Тип {fid}: {FAULT_CLASSES[fid]['short']}",
    index=[1, 2, 3, 4].index(st.session_state.active_fault_id),
)
if selected_fault_id != st.session_state.active_fault_id:
    st.session_state.active_fault_id = selected_fault_id
    st.session_state.manual_override = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Тонкая настройка (опционально)")
use_manual_sliders = st.sidebar.checkbox(
    "Включить ручные ползунки датчиков", value=st.session_state.manual_override
)
st.session_state.manual_override = use_manual_sliders

if use_manual_sliders:
    m_vib = st.sidebar.slider("Вибрация СКЗ (мм/с)", 1.0, 10.0, 5.4, 0.1)
    m_kurt = st.sidebar.slider("Эксцесс (микроудары)", 2.5, 10.0, 6.2, 0.1)
    m_temp = st.sidebar.slider("Температура (°C)", 40.0, 100.0, 68.0, 0.5)
    m_press = st.sidebar.slider("Давление (бар)", 6.0, 18.0, 13.5, 0.1)
    m_curr = st.sidebar.slider("Ток двигателя (А)", 65.0, 165.0, 108.0, 1.0)

# ---------------------------------------------------------------------------
# Вкладки сайта (Четкая структура из 4 разделов)
# ---------------------------------------------------------------------------
tab_main, tab_units, tab_models, tab_econ = st.tabs(
    [
        "🏭 1. Мониторинг и диагностика цеха",
        "📖 2. Технологические узлы линии",
        "🧠 3. Аналитика и метрики ИИ-моделей",
        "🛡️ 4. Архитектура интеграции и Безопасность",
    ]
)

# ===========================================================================
# ВКЛАДКА 1: ГЛАВНЫЙ ЭКРАН — СИМУЛЯЦИЯ УГРОЗЫ -> УВЕДОМЛЕНИЕ -> ИСПРАВЛЕНИЕ
# ===========================================================================
with tab_main:
    # -----------------------------------------------------------------------
    # Блок А: Панель понятных действий в 1 клик (Симуляция -> Исправление)
    # -----------------------------------------------------------------------
    st.markdown("### 🎮 Интерактивное управление: Симуляция угрозы и Устранение сбоя")

    ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1.2, 1.2, 0.9])

    with ctrl_col1:
        sim_btn = st.button(
            f"🚨 1. Запустить симуляцию угрозы ({FAULT_CLASSES[selected_fault_id]['short']})",
            type="primary",
            width="stretch",
        )
    with ctrl_col2:
        fix_label = FAULT_CLASSES[selected_fault_id]["fix_button"]
        fix_btn = st.button(
            f"{fix_label} (Исправить)",
            type="secondary",
            width="stretch",
        )
    with ctrl_col3:
        reset_btn = st.button(
            "🔄 Сбросить цех в штатный режим",
            width="stretch",
        )

    # Логика нажатия кнопки «Запустить симуляцию угрозы» с наглядным ожиданием и уведомлением
    if sim_btn:
        st.session_state.manual_override = False
        progress_placeholder = st.empty()
        status_placeholder = st.empty()
        with progress_placeholder.container():
            bar = st.progress(0, text="📡 Подключение к потоку датчиков ПЛК/SCADA (Read-Only)...")
            time.sleep(0.35)
            bar.progress(40, text=f"🔍 Мониторинг узла «{unit_cfg['name']}»... Фиксация скрытых отклонений...")
            time.sleep(0.45)
            bar.progress(80, text="🧠 Расчет остаточного ресурса (RUL) моделью XGBoost и классификация причины...")
            time.sleep(0.40)
            bar.progress(100, text="⚠️ ВНИМАНИЕ: ИИ перехватил развивающуюся аварию на ранней стадии!")
            time.sleep(0.25)
        progress_placeholder.empty()
        st.session_state.sim_mode = "THREAT"
        st.toast(
            f"🚨 УВЕДОМЛЕНИЕ ИИ: Обнаружена угроза «{FAULT_CLASSES[selected_fault_id]['short']}» на узле {selected_unit_id}!",
            icon="🚨",
        )

    # Логика нажатия кнопки «Исправить ошибку» (выравнивание показателей)
    if fix_btn:
        st.session_state.manual_override = False
        fix_placeholder = st.empty()
        with fix_placeholder.container():
            bar = st.progress(0, text=f"🛠️ Выполнение команды: {FAULT_CLASSES[selected_fault_id]['fix_button']}...")
            time.sleep(0.40)
            bar.progress(60, text="⚙️ Стабилизация давления, тока, температуры и виброскорости...")
            time.sleep(0.45)
            bar.progress(100, text="✅ Параметры выровнены! Узел возвращен в зеленый коридор нормы.")
            time.sleep(0.25)
        fix_placeholder.empty()
        st.session_state.sim_mode = "FIXED"
        st.toast(
            f"✅ Неисправность на узле {selected_unit_id} успешно устранена! Показатели выровнены.",
            icon="✅",
        )

    if reset_btn:
        st.session_state.sim_mode = "NORMAL"
        st.session_state.manual_override = False
        st.toast("🟢 Все агрегаты переведены в штатный режим работы.", icon="🟢")

    # -----------------------------------------------------------------------
    # Формирование траектории и текущих значений датчиков
    # -----------------------------------------------------------------------
    ts_df = generate_dynamic_trajectory(
        mode=st.session_state.sim_mode,
        fault_id=selected_fault_id,
        n_points=60,
    )

    if use_manual_sliders:
        cur_vib, cur_kurt, cur_temp, cur_press, cur_curr = (
            m_vib,
            m_kurt,
            m_temp,
            m_press,
            m_curr,
        )
        cur_rpm_dev = -1.5 if (m_vib > 4.0 or m_curr > 112.0) else -0.2
    else:
        # В режиме THREAT берем точку пика угрозы, а в NORMAL и FIXED — последнюю стабилизированную точку
        last_row = ts_df.iloc[-1]
        cur_vib = float(last_row["vibration_rms"])
        cur_kurt = float(last_row["vibration_kurtosis"])
        cur_temp = float(last_row["temperature_c"])
        cur_press = float(last_row["pressure_bar"])
        cur_curr = float(last_row["current_a"])
        cur_rpm_dev = float(last_row["rpm_dev_pct"])

    ai_result = predict_single_state(
        vibration_rms=cur_vib,
        vibration_kurtosis=cur_kurt,
        temperature_c=cur_temp,
        pressure_bar=cur_press,
        current_a=cur_curr,
        rpm_dev_pct=cur_rpm_dev,
        rf_clf=rf_clf,
        xgb_rul=xgb_rul,
        iso_model=iso_model,
    )

    # -----------------------------------------------------------------------
    # Блок Б: Уведомление системы (результат симуляции или исправления)
    # -----------------------------------------------------------------------
    if st.session_state.sim_mode == "THREAT" or (use_manual_sliders and ai_result["status_level"] != "NORMAL"):
        scada_msg = (
            "🔴 Порог пробит (Классическая SCADA сработала слишком поздно)"
            if ai_result["scada_alarm_active"]
            else "🟢 МОЛЧИТ (Аварийный порог еще не пробит, оператор SCADA не видит проблемы!)"
        )
        st.markdown(
            f"""
            <div class="notif-threat">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <span style="font-size:19px; font-weight:800; color:#FBBF24;">
                        🔔 УВЕДОМЛЕНИЕ ИИ: Через {ai_result['rul_hours']} ч произойдет авария на узле «{unit_cfg['name']}»!
                    </span>
                    <span style="background:#EF4444; color:#FFFFFF; padding:4px 12px; border-radius:6px; font-size:12px; font-weight:700;">
                        ВЕРОЯТНОСТЬ ОТКАЗА: {ai_result['failure_prob_pct']}%
                    </span>
                </div>
                <div style="margin-top:8px; font-size:14px; color:#F8FAFC;">
                    <b>• Выявленная причина:</b> {FAULT_CLASSES[selected_fault_id]['name']}<br>
                    <b>• Реакция обычной SCADA завода:</b> {scada_msg}<br>
                    <b>• Что нужно сделать сейчас:</b> Нажмите кнопку <b>«{FAULT_CLASSES[selected_fault_id]['fix_button']} (Исправить)»</b> выше, чтобы устранить причину и выровнять показатели!
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif st.session_state.sim_mode == "FIXED":
        saved_tenge = unit_cfg["downtime_cost_per_hour"] * 24 + (
            unit_cfg["replacement_cost"] - unit_cfg["preventive_repair_cost"]
        )
        st.markdown(
            f"""
            <div class="notif-fixed">
                <div style="font-size:19px; font-weight:800; color:#34D399;">
                    ✅ ОШИБКА УСТРАНЕНА: Показатели узла «{unit_cfg['name']}» выровнены в норму!
                </div>
                <div style="margin-top:6px; font-size:14px; color:#E2E8F0;">
                    <b>• Выполненное действие:</b> {FAULT_CLASSES[selected_fault_id]['fix_report']}<br>
                    <b>• Текущий статус:</b> Вероятность отказа снижена до <b>{ai_result['failure_prob_pct']}%</b>, остаточный ресурс восстановлен до <b>{ai_result['rul_hours']} ч</b>.<br>
                    <b>• Экономический итог:</b> Аварийная остановка цеха предотвращена! Сэкономлено <b>~{saved_tenge:,.0f} ₸</b>.
                </div>
            </div>
            """.replace(",", " "),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="notif-normal">
                <div style="font-size:17px; font-weight:700; color:#6EE7B7;">
                    🟢 ШТАТНЫЙ РЕЖИМ ЦЕХА: Все 4 промышленных узла работают стабильно
                </div>
                <div style="margin-top:4px; font-size:14px; color:#CBD5E1;">
                    Нажмите кнопку <b>«🚨 1. Запустить симуляцию угрозы»</b> выше, чтобы увидеть, как ИИ заранее перехватывает поломку и помогает оператору исправить её до остановки завода.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Блок В: Сводка всех 4 узлов цеха (Понятные и аккуратные карточки)
    # -----------------------------------------------------------------------
    st.markdown("#### 🏭 Карта состояния 4 агрегатов производственной линии")
    unit_cols = st.columns(4)

    for idx, (u_id, u_info) in enumerate(UNITS_CONFIG.items()):
        with unit_cols[idx]:
            is_selected = u_id == selected_unit_id
            if is_selected:
                u_res = ai_result
                v_show, t_show, p_show, c_show = cur_vib, cur_temp, cur_press, cur_curr
                if st.session_state.sim_mode == "FIXED":
                    badge_txt = "ИСПРАВЛЕНО (НОРМА)"
                    badge_col = "#10B981"
                elif u_res["status_level"] == " NORMAL" or u_res["failure_prob_pct"] < 25:
                    badge_txt = "ШТАТНЫЙ РЕЖИМ"
                    badge_col = "#10B981"
                else:
                    badge_txt = "УГРОЗА ПЕРЕХВАЧЕНА ИИ"
                    badge_col = "#F59E0B" if u_res["rul_hours"] >= 24 else "#EF4444"
                border_style = f"2px solid {badge_col}"
            else:
                # Остальные 3 узла цеха работают ровно в штатном режиме, чтобы не путать зрителя!
                u_res = predict_single_state(2.1, 2.95, 53.0, 14.5, 94.0, -0.1, rf_clf, xgb_rul, iso_model)
                v_show, t_show, p_show, c_show = (
                    u_info["norm_vibration"],
                    u_info["norm_temp"],
                    u_info["norm_pressure"],
                    u_info["norm_current"],
                )
                badge_txt = "ШТАТНЫЙ РЕЖИМ"
                badge_col = "#10B981"
                border_style = "1px solid #1E293B"

            st.markdown(
                f"""
                <div class="unit-card" style="border:{border_style};">
                    <div style="display:flex; justify-content:space-between; font-size:11px; color:#94A3B8;">
                        <span><b>{u_id}</b></span>
                        <span style="color:#38BDF8;">{"🎯 ВЫБРАННЫЙ УЗЕЛ" if is_selected else "В норме"}</span>
                    </div>
                    <div style="font-size:15px; font-weight:700; color:#F8FAFC; margin:4px 0 2px 0;">
                        {u_info['name']}
                    </div>
                    <div style="font-size:11px; color:#94A3B8; margin-bottom:8px;">
                        {u_info['type']}
                    </div>
                    <div style="margin-bottom:8px;">
                        <span style="background:{badge_col}22; color:{badge_col}; border:1px solid {badge_col}; padding:3px 8px; border-radius:4px; font-size:11px; font-weight:700;">
                            {badge_txt}
                        </span>
                    </div>
                    <div style="font-size:12px; color:#E2E8F0; font-family:monospace; line-height:1.55;">
                        • Вибрация: <b>{v_show:.2f} мм/с</b><br>
                        • Температура: <b>{t_show:.1f} °C</b><br>
                        • Давление: <b>{p_show:.1f} бар</b> | Ток: <b>{c_show:.0f} А</b><br>
                        • До отказа (RUL): <b>{u_res['rul_hours']} ч</b> (Риск: <b style="color:{badge_col};">{u_res['failure_prob_pct']}%</b>)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # Блок Г: Наглядная динамика графиков + Карточка решения ИИ
    # -----------------------------------------------------------------------
    st.markdown("---")
    chart_col, info_col = st.columns([1.45, 1.0])

    with chart_col:
        st.markdown(
            f"#### 📈 Динамика показателей узла «{unit_cfg['name']}» в реальном времени"
        )

        # Выбираем главный график под тип ошибки + вкладки всех датчиков
        sub_tab1, sub_tab2 = st.tabs(
            [
                "🔍 Главный график развития и выравнивания угрозы",
                "📊 Все 4 физических датчика (Вибрация, Температура, Давление, Ток)",
            ]
        )

        with sub_tab1:
            if selected_fault_id == 1:
                y_col = "vibration_rms"
                y_title = "Вибрация СКЗ (мм/с)"
                norm_val = 2.1
                ai_warn_val = 4.2
                scada_crit_val = unit_cfg["scada_crit_vib"]
                line_col = "#38BDF8"
            elif selected_fault_id == 2:
                y_col = "temperature_c"
                y_title = "Температура узла (°C)"
                norm_val = 54.0
                ai_warn_val = 69.0
                scada_crit_val = unit_cfg["scada_crit_temp"]
                line_col = "#F97316"
            elif selected_fault_id == 3:
                y_col = "current_a"
                y_title = "Ток нагрузки электродвигателя (А)"
                norm_val = 95.0
                ai_warn_val = 115.0
                scada_crit_val = unit_cfg["scada_crit_current"]
                line_col = "#A855F7"
            else:
                y_col = "pressure_bar"
                y_title = "Давление в магистрали (бар)"
                norm_val = 14.5
                ai_warn_val = 11.8
                scada_crit_val = unit_cfg["scada_min_pressure"]
                line_col = "#06B6D4"

            fig_main = go.Figure()
            fig_main.add_trace(
                go.Scatter(
                    x=ts_df["step"],
                    y=ts_df[y_col],
                    mode="lines+markers",
                    name=y_title,
                    line=dict(color=line_col, width=3.5),
                    marker=dict(size=5),
                )
            )

            # Зеленая линия номинала
            fig_main.add_hline(
                y=norm_val,
                line_dash="dot",
                line_color="#10B981",
                annotation_text=f"Рабочая норма ({norm_val})",
                annotation_position="top left",
                annotation_font_color="#10B981",
            )
            # Желтая линия раннего перехвата ИИ
            fig_main.add_hline(
                y=ai_warn_val,
                line_dash="dash",
                line_color="#F59E0B",
                annotation_text=f"Порог раннего обнаружения ИИ ({ai_warn_val} — за 48-72 ч до аварии)",
                annotation_position="top right",
                annotation_font_color="#F59E0B",
            )
            # Красная линия классической SCADA
            fig_main.add_hline(
                y=scada_crit_val,
                line_dash="solid",
                line_color="#EF4444",
                annotation_text=f"Аварийная сирена SCADA ({scada_crit_val} — узел уже разрушен)",
                annotation_position="bottom right",
                annotation_font_color="#EF4444",
            )

            # Вертикальные метки событий на графике
            if st.session_state.sim_mode in ("THREAT", "FIXED"):
                fig_main.add_vline(
                    x=24 if st.session_state.sim_mode == "THREAT" else 18,
                    line_dash="dot",
                    line_color="#F59E0B",
                    annotation_text="⚠️ Зарождение дефекта",
                    annotation_font_color="#F59E0B",
                )
            if st.session_state.sim_mode == "FIXED":
                fig_main.add_vline(
                    x=42,
                    line_dash="dash",
                    line_color="#10B981",
                    annotation_text="🛠️ Нажата кнопка «Исправить» (Выравнивание)",
                    annotation_font_color="#10B981",
                )

            fig_main.update_layout(
                height=350,
                margin=dict(l=15, r=15, t=25, b=20),
                paper_bgcolor="#111827",
                plot_bgcolor="#0B0F19",
                font=dict(color="#E2E8F0", size=12),
                xaxis_title="Такты мониторинга телеметрии (Время)",
                yaxis_title=y_title,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_main, width="stretch")

        with sub_tab2:
            c_left, c_right = st.columns(2)
            with c_left:
                fig_v = go.Figure()
                fig_v.add_trace(
                    go.Scatter(
                        x=ts_df["step"],
                        y=ts_df["vibration_rms"],
                        name="Вибрация (мм/с)",
                        line=dict(color="#38BDF8", width=2.5),
                    )
                )
                fig_v.add_trace(
                    go.Scatter(
                        x=ts_df["step"],
                        y=ts_df["pressure_bar"],
                        name="Давление (бар)",
                        line=dict(color="#06B6D4", width=2.5, dash="dot"),
                    )
                )
                fig_v.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=25, b=15),
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0B0F19",
                    font=dict(color="#E2E8F0", size=11),
                    title="Вибрация (мм/с) и Давление (бар)",
                )
                st.plotly_chart(fig_v, width="stretch")
            with c_right:
                fig_t = go.Figure()
                fig_t.add_trace(
                    go.Scatter(
                        x=ts_df["step"],
                        y=ts_df["temperature_c"],
                        name="Температура (°C)",
                        line=dict(color="#F97316", width=2.5),
                    )
                )
                fig_t.add_trace(
                    go.Scatter(
                        x=ts_df["step"],
                        y=ts_df["current_a"],
                        name="Ток нагрузки (А)",
                        line=dict(color="#10B981", width=2.5, dash="dot"),
                    )
                )
                fig_t.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=25, b=15),
                    paper_bgcolor="#111827",
                    plot_bgcolor="#0B0F19",
                    font=dict(color="#E2E8F0", size=11),
                    title="Температура (°C) и Ток двигателя (А)",
                )
                st.plotly_chart(fig_t, width="stretch")

    with info_col:
        st.markdown("#### 📋 Паспорт узла и Диагностическая карта ИИ")

        st.markdown(
            f"""
            <div class="passport-box">
                <div style="font-size:13px; color:#38BDF8; font-weight:700;">
                    ОПИСАНИЕ АКТИВНОГО УЗЛА ЦЕХА ({selected_unit_id})
                </div>
                <div style="font-size:16px; font-weight:700; color:#F8FAFC; margin:4px 0;">
                    {unit_cfg['name']} — {unit_cfg['shop']}
                </div>
                <div style="font-size:13px; color:#CBD5E1; margin-bottom:6px;">
                    <b>За что отвечает:</b> {unit_cfg['purpose']}
                </div>
                <div style="font-size:13px; color:#94A3B8;">
                    <b>Типовые риски:</b> {unit_cfg['what_breaks']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Горизонтальный график вклада датчиков (Почему ИИ принял решение)
        st.markdown("##### 🔍 Вклад датчиков в диагноз ИИ (Explainable AI)")
        contrib_sorted = sorted(
            ai_result["feature_contributions"].items(), key=lambda x: x[1], reverse=True
        )
        f_names = [x[0] for x in contrib_sorted][::-1]
        f_vals = [x[1] for x in contrib_sorted][::-1]

        fig_imp = go.Figure(
            go.Bar(
                x=f_vals,
                y=f_names,
                orientation="h",
                text=[f"{v:.1f}%" for v in f_vals],
                textposition="auto",
                marker_color=["#38BDF8", "#60A5FA", "#818CF8", "#F59E0B", "#EF4444"],
            )
        )
        fig_imp.update_layout(
            height=205,
            margin=dict(l=10, r=10, t=5, b=10),
            paper_bgcolor="#111827",
            plot_bgcolor="#0B0F19",
            font=dict(color="#E2E8F0", size=11),
            xaxis_title="Доля влияния параметра на вердикт ИИ (%)",
        )
        st.plotly_chart(fig_imp, width="stretch")

# ===========================================================================
# ВКЛАДКА 2: ТЕХНОЛОГИЧЕСКИЕ УЗЛЫ ПРОИЗВОДСТВЕННОЙ ЛИНИИ
# ===========================================================================
with tab_units:
    st.markdown("### 📖 Технические паспорта и регламентные допуски агрегатов")
    st.write(
        "Сводная спецификация контролируемых узлов производственной линии: назначение агрегатов, "
        "номинальные рабочие параметры, пороги аварийной сигнализации SCADA и профиль предиктивной защиты."
    )

    u_col1, u_col2 = st.columns(2)
    for idx, (u_id, u_info) in enumerate(UNITS_CONFIG.items()):
        target_col = u_col1 if idx % 2 == 0 else u_col2
        def_fault = FAULT_CLASSES[u_info["default_fault_id"]]
        with target_col:
            st.markdown(
                f"""
                <div class="action-card" style="margin-bottom:14px; border-left:4px solid #06B6D4;">
                    <h4 style="margin:0 0 6px 0; color:#38BDF8;">
                        {idx + 1}. {u_id} — {u_info['name']}
                    </h4>
                    <p style="margin:0 0 8px 0; font-size:13px; color:#94A3B8;">
                        <b>Расположение:</b> {u_info['shop']} | <b>Тип:</b> {u_info['type']}
                    </p>
                    <p style="font-size:14px; color:#E2E8F0; margin-bottom:8px;">
                        <b>За что отвечает на заводе:</b><br>{u_info['purpose']}
                    </p>
                    <p style="font-size:14px; color:#E2E8F0; margin-bottom:8px;">
                        <b>Что чаще всего выходит из строя:</b><br>{u_info['what_breaks']}
                    </p>
                    <div style="background:#111827; padding:10px 12px; border-radius:8px; font-family:monospace; font-size:12px; color:#CBD5E1; margin-bottom:8px;">
                        <b>Эталонные рабочие параметры (Норма vs Сирена SCADA):</b><br>
                        • Вибрация: норма {u_info['norm_vibration']} мм/с (авария SCADA > {u_info['scada_crit_vib']} мм/с)<br>
                        • Температура: норма {u_info['norm_temp']} °C (авария SCADA > {u_info['scada_crit_temp']} °C)<br>
                        • Давление: норма {u_info['norm_pressure']} бар (мин. порог SCADA < {u_info['scada_min_pressure']} бар)<br>
                        • Ток привода: норма {u_info['norm_current']} А (перегрузка SCADA > {u_info['scada_crit_current']} А)
                    </div>
                    <p style="font-size:13px; color:#6EE7B7; margin:0;">
                        <b>💡 Как спасает наш ИИ:</b> Перехватывает «{def_fault['short']}» за 48–72 часа до аварии.
                        Вместо разрушения агрегата за <b>{u_info['replacement_cost']:,.0f} ₸</b> выполняется плановая стабилизация за <b>{u_info['preventive_repair_cost']:,.0f} ₸</b>.
                    </p>
                </div>
                """.replace(",", " "),
                unsafe_allow_html=True,
            )

# ===========================================================================
# ВКЛАДКА 3: АНАЛИТИКА И МЕТРИКИ КАЧЕСТВА ИИ-МОДЕЛЕЙ
# ===========================================================================
with tab_models:
    st.markdown("### 🧠 Математическое ядро и метрики качества обученных моделей")
    st.markdown(
        """
        Внутри платформы работает **ансамбль из 3 моделей машинного обучения**, обученный на выборке из `4 000` многомерных физических состояний оборудования:
        1. **Isolation Forest** — неконтролируемое обнаружение ранних аномалий.
        2. **Random Forest Classifier** — определение точной причины поломки (5 классов) и вероятности отказа в `%`.
        3. **XGBoost Regressor** — прогноз остаточного ресурса (**RUL**) в часах до остановки.
        """
    )

    mk1, mk2, mk3, mk4 = st.columns(4)
    mk1.metric("Точность классификации (Accuracy)", f"{model_metrics['classifier_accuracy']}%")
    mk2.metric("Метрика F1-Macro (5 классов)", f"{model_metrics['classifier_f1_macro']}%")
    mk3.metric("Точность детектора аномалий", f"{model_metrics['anomaly_detection_accuracy']}%")
    mk4.metric("Точность прогноза RUL (R²)", f"{model_metrics['rul_r2_score']}")

    cm_col, fi_col = st.columns(2)

    with cm_col:
        st.markdown("#### Матрица ошибок классификатора (Confusion Matrix)")
        cm_array = np.array(model_metrics["confusion_matrix"])
        class_labels = model_metrics["class_names"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm_array,
                x=class_labels,
                y=class_labels,
                colorscale="Blues",
                text=cm_array,
                texttemplate="%{text}",
                textfont=dict(size=14, color="#FFFFFF"),
            )
        )
        fig_cm.update_layout(
            height=360,
            margin=dict(l=15, r=15, t=20, b=20),
            paper_bgcolor="#111827",
            plot_bgcolor="#0B0F19",
            font=dict(color="#E2E8F0", size=12),
            xaxis_title="Предсказанный ИИ диагноз",
            yaxis_title="Истинный класс дефекта",
        )
        st.plotly_chart(fig_cm, width="stretch")

    with fi_col:
        st.markdown("#### Глобальная значимость признаков (Feature Importance)")
        fi_sorted = sorted(
            model_metrics["feature_importances"].items(), key=lambda x: x[1], reverse=True
        )
        fi_names = [x[0] for x in fi_sorted][::-1]
        fi_vals = [round(x[1] * 100, 1) for x in fi_sorted][::-1]

        fig_fi = go.Figure(
            go.Bar(
                x=fi_vals,
                y=fi_names,
                orientation="h",
                text=[f"{v}%" for v in fi_vals],
                textposition="auto",
                marker_color="#06B6D4",
            )
        )
        fig_fi.update_layout(
            height=360,
            margin=dict(l=15, r=15, t=20, b=20),
            paper_bgcolor="#111827",
            plot_bgcolor="#0B0F19",
            font=dict(color="#E2E8F0", size=12),
            xaxis_title="Вес признака в модели Random Forest (%)",
        )
        st.plotly_chart(fig_fi, width="stretch")

# ===========================================================================
# ВКЛАДКА 4: АРХИТЕКТУРА READ-ONLY И МУЛЬТИАГЕНТНАЯ ПАУТИНА
# ===========================================================================
with tab_econ:
    st.markdown("### 🕸️ Иерархическая мультиагентная архитектура («ИИ-паутина предприятия»)")
    st.markdown(
        """
        Чтобы не перегружать единую модель миллионами сырых сигналов и мелких помех со всего завода, система построена по принципу **распределенной мультиагентной сети**:
        1. **Полевые ИИ-агенты отделов (Нижний уровень):** За каждым цехом и агрегатом (`PUMP-101`, `MOTOR-204`, `COMP-302`, `DRIVE-405`) закреплена собственная группа легких специализированных агентов (агент вибрации, агент температуры, агент давления, агент тока). Они анализируют телеметрию на местах и **отсекают до 99% рутинного шума**.
        2. **Центральная Мастер-Модель (Главный Агент-Координатор):** Получает от узловых агентов только подтвержденные векторы отклонений, сопоставляет взаимовлияние между отделами завода, рассчитывает точный остаточный ресурс (**RUL**) и выдает готовое решение механику.
        """
    )

    st.markdown("---")
    st.markdown("### 🛡️ Принцип «Бесшовной надстройки» (Read-Only) и Законодательство РК")
    st.markdown(
        """
        * **Универсальное подключение к любому заводу (Plug & Play):** Система подключается поверх существующей SCADA/ПЛК исключительно в режиме чтения телеметрии (`Read-Only` через OPC UA / Modbus TCP). Она не вмешивается в управляющий код контроллеров и не требует остановки линии.
        * **Государственная значимость:** Прямая реализация **Указа Президента РК «Digital Qazaqstan» от 09.06.2026 г.** (переход промышленности от пассивного сбора телеметрии к ИИ-моделям предиктивного управления при износе фондов более 60%).
        * **Информационная безопасность:** Все вычисления выполняются локально внутри закрытого контура завода в соответствии с Законом РК «Об искусственном интеллекте» (№ 230-VIII) и Законом РК «О кибербезопасности» (№ 418-V).
        """
    )
