"""
Генератор дизайнерской презентации PowerPoint (6 слайдов, 16:9)
Проект: AI-PromPlatform (МИТУ / METU, AI-SANA 2026)
Дизайн-система: в точном визуальном стиле референсов SANAU ENERGY
(глубокий индиго для продуктовых слайдов + светлый лавандово-белый корпоративный стиль
со стрелками-шевронами, мультиагентной паутиной и таймлайном внедрения 1..5).
Все тексты в .pptx являются нативными редактируемыми текстовыми блоками PowerPoint.
"""

import math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "presentation_assets"
SLIDES_BG_DIR = ASSETS_DIR / "slide_layers"
PREVIEW_DIR = ASSETS_DIR / "previews"
OUTPUT_PPTX = BASE_DIR / "AI_PromPlatform_Presentation_6_Slides.pptx"

# Разрешение графической подложки: 3200 x 1800 (ровно 240 пикселей на 1 дюйм при 13.333" x 7.5")
W, H = 3200, 1800
DPI_SCALE = 240.0  # 1 inch = 240 px


def inches(px_val: float):
    return Inches(px_val / DPI_SCALE)


# ---------------------------------------------------------------------------
# ЦВЕТОВАЯ ПАЛИТРА (ТОЧНО ПО РЕФЕРЕНСАМ SANAU)
# ---------------------------------------------------------------------------
DARK_BG = (24, 28, 56)           # #181C38 — глубокий темно-синий индиго
DARK_CIRCLE = (39, 54, 145)      # #273691 — большой декоративный синий круг справа
DARK_PILL = (45, 52, 94)         # #2D345E — плашка-бейдж на темном фоне
CYAN_ACCENT = (0, 229, 255)      # #00E5FF — бирюзовый акцент под бейджем

LIGHT_BG = (244, 245, 251)       # #F4F5FB — светлый лавандово-белый фон
LIGHT_CIRCLE = (234, 236, 249)   # #EAECF9 — мягкий декоративный круг справа
LIGHT_PILL = (224, 228, 248)     # #E0E4F8 — светло-лавандовый бейдж
LIGHT_BAR = (228, 232, 249)      # #E4E8F9 — нижняя плашка на светлых слайдах

ROYAL_BLUE = (53, 69, 216)       # #3545D8 — насыщенный королевский синий
INDIGO_DARK = (27, 31, 59)       # #1B1F3B — темный графитово-индиго для заголовков и нижней плашки
SLATE_MUTED = (108, 115, 143)    # #6C738F — приглушенный серо-синий для подзаголовков
WHITE = (255, 255, 255)

# Градиентные цвета для 4 шевронов (от нежно-лавандового к королевскому индиго)
CHEVRON_COLORS = [
    (212, 217, 247),  # Шаг 1: #D4D9F7
    (152, 163, 238),  # Шаг 2: #98A3EE
    (90, 107, 226),   # Шаг 3: #5A6BE2
    (45, 62, 210),    # Шаг 4: #2D3ED2
]

# Цвета 4 мини-иконок параметров
PARAM_PURPLE = (102, 92, 245)    # Вибрация
PARAM_CYAN = (0, 196, 240)       # Давление / Поток
PARAM_ORANGE = (255, 136, 0)     # Ток привода
PARAM_PINK = (255, 56, 136)      # Температура
GREEN_PULSE = (0, 210, 106)      # Зеленый индикатор


# ---------------------------------------------------------------------------
# БАЗОВЫЕ ГРАФИЧЕСКИЕ ПРИМИТИВЫ ВЫСОКОЙ ЧЕТКОСТИ
# ---------------------------------------------------------------------------
def create_dark_canvas() -> Image.Image:
    """Создает темный индиго-фон с гигантским синим кругом справа (Стиль Фото 1)."""
    img = Image.new("RGBA", (W, H), DARK_BG + (255,))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy, r = int(W * 0.77), int(H * 0.52), int(H * 0.82)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=DARK_CIRCLE + (255,))
    r2 = int(r * 0.72)
    draw.ellipse([cx - r2 + 120, cy - r2, cx + r2 + 120, cy + r2], fill=(46, 64, 168, 140))
    return img


def create_light_canvas() -> Image.Image:
    """Создает светлый корпоративный лавандовый фон с декоративным кругом справа (Стиль Фото 2 и 3)."""
    img = Image.new("RGBA", (W, H), LIGHT_BG + (255,))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy, r = int(W * 0.80), int(H * 0.46), int(H * 0.78)
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=LIGHT_CIRCLE + (255,))
    return img


def draw_shadow_rounded_rect(
    base_img: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple,
    outline: tuple | None = None,
    outline_width: int = 2,
    shadow_offset: tuple[int, int] = (0, 14),
    shadow_blur: int = 24,
    shadow_alpha: int = 38,
):
    """Рисует карточку со скругленными углами и мягкой реалистичной тенью."""
    x1, y1, x2, y2 = box
    pad = shadow_blur * 3
    sw = (x2 - x1) + pad * 2
    sh = (y2 - y1) + pad * 2

    shadow = Image.new("RGBA", (sw, sh), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow, "RGBA")
    sx1 = pad + shadow_offset[0]
    sy1 = pad + shadow_offset[1]
    sx2 = sx1 + (x2 - x1)
    sy2 = sy1 + (y2 - y1)
    sdraw.rounded_rectangle([sx1, sy1, sx2, sy2], radius=radius, fill=(15, 20, 45, shadow_alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(shadow_blur))

    base_img.alpha_composite(shadow, (x1 - pad, y1 - pad))
    draw = ImageDraw.Draw(base_img, "RGBA")
    draw.rounded_rectangle([x1, y1, x2, y2], radius=radius, fill=fill, outline=outline, width=outline_width)


def draw_vector_icon(draw: ImageDraw.ImageDraw, kind: str, cx: int, cy: int, size: int, color: tuple):
    """Рисует четкие векторные иконки внутри бейджей."""
    r = size // 2
    lw = max(4, size // 11)

    if kind == "vibration":
        pts = [
            (cx - r, cy),
            (cx - int(r * 0.55), cy),
            (cx - int(r * 0.30), cy - int(r * 0.75)),
            (cx, cy + int(r * 0.80)),
            (cx + int(r * 0.30), cy - int(r * 0.55)),
            (cx + int(r * 0.55), cy),
            (cx + r, cy),
        ]
        draw.line(pts, fill=color, width=lw, joint="curve")

    elif kind == "droplet":
        pts = [
            (cx, cy - int(r * 0.85)),
            (cx + int(r * 0.65), cy + int(r * 0.15)),
            (cx + int(r * 0.45), cy + int(r * 0.75)),
            (cx, cy + int(r * 0.90)),
            (cx - int(r * 0.45), cy + int(r * 0.75)),
            (cx - int(r * 0.65), cy + int(r * 0.15)),
        ]
        draw.polygon(pts, outline=color, width=lw)

    elif kind == "bolt":
        pts = [
            (cx + int(r * 0.15), cy - int(r * 0.90)),
            (cx - int(r * 0.55), cy + int(r * 0.05)),
            (cx - int(r * 0.05), cy + int(r * 0.05)),
            (cx - int(r * 0.20), cy + int(r * 0.90)),
            (cx + int(r * 0.55), cy - int(r * 0.05)),
            (cx + int(r * 0.05), cy - int(r * 0.05)),
        ]
        draw.polygon(pts, fill=color)

    elif kind == "thermometer":
        bulb_r = int(r * 0.38)
        draw.ellipse(
            [cx - bulb_r, cy + int(r * 0.25) - bulb_r, cx + bulb_r, cy + int(r * 0.25) + bulb_r],
            outline=color,
            width=lw,
        )
        stem_w = int(r * 0.22)
        draw.rounded_rectangle(
            [cx - stem_w, cy - int(r * 0.85), cx + stem_w, cy + int(r * 0.15)],
            radius=stem_w,
            outline=color,
            width=lw,
        )

    elif kind == "antenna":
        draw.line([(cx, cy - int(r * 0.65)), (cx, cy + int(r * 0.85))], fill=color, width=lw)
        draw.line([(cx - int(r * 0.55), cy - int(r * 0.30)), (cx + int(r * 0.55), cy - int(r * 0.30))], fill=color, width=lw)
        draw.line([(cx - int(r * 0.35), cy - int(r * 0.65)), (cx + int(r * 0.35), cy - int(r * 0.65))], fill=color, width=lw)
        for x_off in (-int(r * 0.45), int(r * 0.45)):
            draw.line([(cx + x_off, cy - int(r * 0.50)), (cx + x_off, cy - int(r * 0.10))], fill=color, width=lw)

    elif kind == "shield":
        pts = [
            (cx, cy - int(r * 0.85)),
            (cx + int(r * 0.72), cy - int(r * 0.50)),
            (cx + int(r * 0.62), cy + int(r * 0.20)),
            (cx, cy + int(r * 0.88)),
            (cx - int(r * 0.62), cy + int(r * 0.20)),
            (cx - int(r * 0.72), cy - int(r * 0.50)),
        ]
        draw.polygon(pts, outline=color, width=lw)
        chk = [
            (cx - int(r * 0.28), cy + int(r * 0.02)),
            (cx - int(r * 0.05), cy + int(r * 0.28)),
            (cx + int(r * 0.32), cy - int(r * 0.22)),
        ]
        draw.line(chk, fill=color, width=lw, joint="curve")

    elif kind == "monitor":
        mw, mh = int(r * 0.82), int(r * 0.58)
        draw.rounded_rectangle([cx - mw, cy - mh - 6, cx + mw, cy + mh - 6], radius=8, outline=color, width=lw)
        draw.line([(cx, cy + mh - 6), (cx, cy + int(r * 0.82))], fill=color, width=lw)
        draw.line([(cx - int(r * 0.45), cy + int(r * 0.82)), (cx + int(r * 0.45), cy + int(r * 0.82))], fill=color, width=lw)
        draw.line(
            [(cx - int(r * 0.45), cy + 6), (cx - int(r * 0.12), cy - 18), (cx + int(r * 0.15), cy + 2), (cx + int(r * 0.48), cy - 26)],
            fill=color,
            width=max(3, lw - 1),
        )

    elif kind == "server":
        bw, bh = int(r * 0.78), int(r * 0.34)
        draw.rounded_rectangle([cx - bw, cy - bh * 2 - 4, cx + bw, cy - 4], radius=8, outline=color, width=lw)
        draw.rounded_rectangle([cx - bw, cy + 6, cx + bw, cy + bh * 2 + 6], radius=8, outline=color, width=lw)
        draw.ellipse([cx - bw + 14, cy - bh - 8, cx - bw + 22, cy - bh], fill=color)
        draw.ellipse([cx - bw + 14, cy + bh + 2, cx - bw + 22, cy + bh + 10], fill=color)

    elif kind == "file_data":
        fw, fh = int(r * 0.65), int(r * 0.85)
        draw.rounded_rectangle([cx - fw, cy - fh, cx + fw, cy + fh], radius=8, outline=color, width=lw)
        draw.line([(cx - int(fw * 0.55), cy - int(fh * 0.35)), (cx + int(fw * 0.55), cy - int(fh * 0.35))], fill=color, width=lw)
        draw.line([(cx - int(fw * 0.55), cy + int(fh * 0.05)), (cx + int(fw * 0.55), cy + int(fh * 0.05))], fill=color, width=lw)
        draw.line([(cx - int(fw * 0.55), cy + int(fh * 0.45)), (cx + int(fw * 0.25), cy + int(fh * 0.45))], fill=color, width=lw)

    elif kind == "gateway":
        draw.ellipse([cx - int(r * 0.28), cy - int(r * 0.78), cx + int(r * 0.28), cy - int(r * 0.22)], outline=color, width=lw)
        draw.arc([cx - int(r * 0.68), cy - int(r * 0.05), cx + int(r * 0.68), cy + int(r * 0.95)], start=190, end=350, fill=color, width=lw)
        draw.line([(cx + int(r * 0.45), cy - int(r * 0.55)), (cx + int(r * 0.88), cy - int(r * 0.55))], fill=color, width=lw)

    elif kind == "web_nodes":
        cr = int(r * 0.30)
        sat_r = int(r * 0.18)
        offsets = [(-int(r * 0.68), -int(r * 0.55)), (int(r * 0.68), -int(r * 0.55)), (-int(r * 0.68), int(r * 0.55)), (int(r * 0.68), int(r * 0.55))]
        for ox, oy in offsets:
            draw.line([(cx, cy), (cx + ox, cy + oy)], fill=color, width=max(3, lw - 1))
            draw.ellipse([cx + ox - sat_r, cy + oy - sat_r, cx + ox + sat_r, cy + oy + sat_r], fill=LIGHT_PILL, outline=color, width=lw)
        draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=color)

    elif kind == "roi_cap":
        pts = [
            (cx, cy - int(r * 0.70)),
            (cx + int(r * 0.92), cy - int(r * 0.18)),
            (cx, cy + int(r * 0.32)),
            (cx - int(r * 0.92), cy - int(r * 0.18)),
        ]
        draw.polygon(pts, outline=color, width=lw)
        draw.arc([cx - int(r * 0.52), cy - int(r * 0.05), cx + int(r * 0.52), cy + int(r * 0.75)], start=0, end=180, fill=color, width=lw)

    elif kind == "rocket":
        pts = [
            (cx + int(r * 0.75), cy - int(r * 0.75)),
            (cx + int(r * 0.55), cy + int(r * 0.15)),
            (cx - int(r * 0.15), cy + int(r * 0.55)),
            (cx - int(r * 0.65), cy + int(r * 0.15)),
            (cx - int(r * 0.15), cy - int(r * 0.55)),
        ]
        draw.polygon(pts, outline=color, width=lw)
        draw.line([(cx - int(r * 0.75), cy + int(r * 0.75)), (cx - int(r * 0.35), cy + int(r * 0.35))], fill=CYAN_ACCENT, width=lw)


def draw_icon_badge(img: Image.Image, cx: int, cy: int, size: int, kind: str, bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE, is_circle=False):
    """Рисует мягкую лавандовую иконку-плашку (квадратную со скруглением или круглую)."""
    draw = ImageDraw.Draw(img, "RGBA")
    half = size // 2
    box = [cx - half, cy - half, cx + half, cy + half]
    if is_circle:
        draw.ellipse(box, fill=bg_color)
    else:
        draw.rounded_rectangle(box, radius=int(size * 0.28), fill=bg_color)
    draw_vector_icon(draw, kind, cx, cy, int(size * 0.58), fg_color)


def draw_chevron_bar(
    img: Image.Image,
    x_start: int,
    y_top: int,
    total_w: int,
    h: int,
    steps: int = 4,
    tip_depth: int = 56,
    gap: int = 14,
):
    """Рисует 4 сцепленные градиентные стрелки-шеврона (Точная геометрия Фото 2)."""
    draw = ImageDraw.Draw(img, "RGBA")
    step_w = (total_w - gap * (steps - 1)) // steps
    ym = y_top + h // 2
    yb = y_top + h

    centers = []
    for i in range(steps):
        x1 = x_start + i * (step_w + gap)
        x2 = x1 + step_w
        color = CHEVRON_COLORS[i % len(CHEVRON_COLORS)]

        if i == 0:
            pts = [
                (x1, y_top),
                (x2 - tip_depth, y_top),
                (x2, ym),
                (x2 - tip_depth, yb),
                (x1, yb),
            ]
            cx = (x1 + x2 - tip_depth // 2) // 2
        else:
            pts = [
                (x1 - tip_depth, y_top),
                (x2 - tip_depth, y_top),
                (x2, ym),
                (x2 - tip_depth, yb),
                (x1 - tip_depth, yb),
                (x1, ym),
            ]
            cx = (x1 + x2 - tip_depth) // 2

        draw.polygon(pts, fill=color)
        centers.append((cx, y_top))
    return centers


def paste_macos_window(
    base_img: Image.Image,
    screenshot_path: Path,
    box: tuple[int, int, int, int],
    dark_frame: bool = True,
):
    """Вставляет реальный скриншот нашего сайта в стильную рамку macOS с мягкой тенью (как на Фото 1)."""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    header_h = 54
    radius = 24

    frame_bg = (22, 26, 46, 255) if dark_frame else (245, 247, 252, 255)
    border_col = (72, 84, 145, 220) if dark_frame else (205, 212, 238, 255)

    draw_shadow_rounded_rect(
        base_img,
        box,
        radius=radius,
        fill=frame_bg,
        outline=border_col,
        outline_width=3,
        shadow_offset=(0, 22),
        shadow_blur=36,
        shadow_alpha=95,
    )

    if screenshot_path.exists():
        shot = Image.open(screenshot_path).convert("RGBA")
        target_w = w - 12
        target_h = h - header_h - 8
        shot_resized = shot.resize((target_w, target_h), Image.Resampling.LANCZOS)

        mask = Image.new("L", (target_w, target_h), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle([0, 0, target_w, target_h], radius=16, fill=255)
        base_img.paste(shot_resized, (x1 + 6, y1 + header_h + 2), mask)

    draw = ImageDraw.Draw(base_img, "RGBA")
    dot_y = y1 + header_h // 2
    dot_colors = [(255, 95, 86), (255, 189, 46), (39, 201, 63)]
    for idx, dcol in enumerate(dot_colors):
        dx = x1 + 30 + idx * 24
        draw.ellipse([dx - 7, dot_y - 7, dx + 7, dot_y + 7], fill=dcol)


# ---------------------------------------------------------------------------
# ГЕНЕРАЦИЯ ГРАФИЧЕСКИХ СЛОЕВ ДЛЯ ВСЕХ 6 СЛАЙДОВ
# ---------------------------------------------------------------------------
def build_slide1_bg() -> Path:
    """Слайд 1: ТЕМНЫЙ ГЕРОЙ-СЛАЙД (Точный стиль Фото 1 SANAU ENERGY)."""
    img = create_dark_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний бейдж + яркая бирюзовая линия под ним
    draw.rounded_rectangle([140, 145, 820, 203], radius=29, fill=DARK_PILL)
    draw.rounded_rectangle([140, 217, 340, 227], radius=5, fill=CYAN_ACCENT)

    # 4 овальные плашки параметров (Вибрация, Давление, Ток сети, Температура)
    pills_x = [140, 440, 740, 1035]
    pills_w = [280, 280, 275, 350]
    p_colors = [PARAM_PURPLE, PARAM_CYAN, PARAM_ORANGE, PARAM_PINK]
    p_icons = ["vibration", "droplet", "bolt", "thermometer"]

    for px_left, pw, pcol, picon in zip(pills_x, pills_w, p_colors, p_icons):
        py1, py2 = 905, 985
        draw.rounded_rectangle([px_left, py1, px_left + pw, py2], radius=40, fill=(38, 45, 84, 240), outline=(65, 76, 132, 180), width=2)
        ic_x = px_left + 42
        ic_y = (py1 + py2) // 2
        draw.ellipse([ic_x - 28, ic_y - 28, ic_x + 28, ic_y + 28], fill=pcol)
        draw_vector_icon(draw, picon, ic_x, ic_y, 34, WHITE)

    # Тонкая разделительная линия над KPI
    draw.line([(140, 1075), (1360, 1075)], fill=(52, 62, 112, 200), width=2)

    # Справа: два перекрывающихся окна с реальными скриншотами нашего сайта (как на Фото 1)
    paste_macos_window(img, ASSETS_DIR / "shot_fixed.png", box=(1650, 790, 3040, 1620), dark_frame=False)
    paste_macos_window(img, ASSETS_DIR / "shot_threat.png", box=(1430, 130, 2960, 1030), dark_frame=True)

    # Плавающий белый бейдж поверх нижнего окна
    draw_shadow_rounded_rect(
        img,
        box=(1365, 1185, 1985, 1375),
        radius=36,
        fill=(250, 251, 255, 255),
        outline=(225, 230, 248, 255),
        outline_width=2,
        shadow_offset=(0, 18),
        shadow_blur=30,
        shadow_alpha=85,
    )
    draw_icon_badge(img, 1455, 1280, 92, "vibration", bg_color=GREEN_PULSE, fg_color=WHITE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_1_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


def build_slide2_bg() -> Path:
    """Слайд 2: БЕЗОПАСНОСТЬ И READ-ONLY ПОДКЛЮЧЕНИЕ (Точный стиль Фото 2 со стрелками-шевронами)."""
    img = create_light_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний лавандовый бейдж
    draw.rounded_rectangle([140, 110, 620, 164], radius=27, fill=LIGHT_PILL)

    # Полоса из 4 градиентных шевронов
    draw_chevron_bar(img, x_start=140, y_top=485, total_w=2920, h=190, steps=4, tip_depth=54, gap=16)

    col_centers = [465, 1195, 1930, 2665]
    icon_y = 825

    # Под 1-м шевроном: 4 цветных кружка параметров (Точно как на Фото 2)
    mini_colors = [(228, 226, 255), (218, 244, 255), (255, 236, 214), (255, 222, 238)]
    mini_fg = [PARAM_PURPLE, PARAM_CYAN, PARAM_ORANGE, PARAM_PINK]
    mini_icons = ["vibration", "droplet", "bolt", "thermometer"]
    for idx, (mbg, mfg, mic) in enumerate(zip(mini_colors, mini_fg, mini_icons)):
        mx = col_centers[0] - 135 + idx * 90
        draw_icon_badge(img, mx, icon_y, 76, mic, bg_color=mbg, fg_color=mfg, is_circle=True)

    # Под 2-м, 3-м и 4-м шевронами: мягкие лавандовые квадраты с векторными иконками
    draw_icon_badge(img, col_centers[1], icon_y, 92, "antenna", bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)
    draw_icon_badge(img, col_centers[2], icon_y, 92, "shield", bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)
    draw_icon_badge(img, col_centers[3], icon_y, 92, "monitor", bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)

    # Нижняя широкая лавандовая плашка-вывод
    draw.rounded_rectangle([140, 1455, 3060, 1615], radius=32, fill=LIGHT_BAR, outline=(208, 215, 245), width=2)
    draw_icon_badge(img, 225, 1535, 76, "shield", bg_color=(208, 216, 248), fg_color=ROYAL_BLUE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_2_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


def build_slide3_bg() -> Path:
    """Слайд 3: МУЛЬТИАГЕНТНАЯ ПАУТИНА ИИ (Визуальная схема распределения нагрузки по отделам)."""
    img = create_light_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний бейдж
    draw.rounded_rectangle([140, 110, 690, 164], radius=27, fill=LIGHT_PILL)

    hub_cx, hub_cy = 1600, 910
    dept_boxes = [
        (140, 465, 1020, 825),
        (140, 965, 1020, 1325),
        (2180, 465, 3060, 825),
        (2180, 965, 3060, 1325),
    ]
    dept_OnPoints = [
        (1020, 645),
        (1020, 1145),
        (2180, 645),
        (2180, 1145),
    ]

    for ring_r, alpha in [(520, 90), (400, 130), (290, 180)]:
        draw.ellipse(
            [hub_cx - ring_r, hub_cy - ring_r, hub_cx + ring_r, hub_cy + ring_r],
            outline=(195, 204, 244, alpha),
            width=3,
        )

    filter_badges = [
        (1195, 735),
        (1195, 1075),
        (2005, 735),
        (2005, 1075),
    ]
    for (px_pt, py_pt), (bx, by) in zip(dept_OnPoints, filter_badges):
        draw.line([(px_pt, py_pt), (hub_cx, hub_cy)], fill=ROYAL_BLUE, width=6)
        draw.ellipse([px_pt - 14, py_pt - 14, px_pt + 14, py_pt + 14], fill=ROYAL_BLUE, outline=WHITE, width=4)
        draw.rounded_rectangle([bx - 155, by - 28, bx + 155, by + 28], radius=28, fill=LIGHT_PILL, outline=ROYAL_BLUE, width=2)

    dept_icons = ["droplet", "bolt", "thermometer", "vibration"]
    for box, dic in zip(dept_boxes, dept_icons):
        draw_shadow_rounded_rect(
            img,
            box=box,
            radius=30,
            fill=WHITE,
            outline=(218, 224, 246),
            outline_width=2,
        )
        x1, y1, x2, y2 = box
        draw_icon_badge(img, x1 + 78, y1 + 82, 84, dic, bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)
        draw.rounded_rectangle([x1 + 42, y2 - 78, x2 - 42, y2 - 26], radius=18, fill=(235, 248, 242))

    # Центральное Ядро: МАСТЕР-МОДЕЛЬ
    draw.ellipse([hub_cx - 255, hub_cy - 255, hub_cx + 255, hub_cy + 255], fill=(210, 217, 250))
    draw_shadow_rounded_rect(
        img,
        box=(hub_cx - 235, hub_cy - 245, hub_cx + 235, hub_cy + 245),
        radius=235,
        fill=INDIGO_DARK,
        outline=ROYAL_BLUE,
        outline_width=8,
        shadow_offset=(0, 18),
        shadow_blur=34,
        shadow_alpha=75,
    )
    draw_icon_badge(img, hub_cx, hub_cy - 130, 88, "web_nodes", bg_color=ROYAL_BLUE, fg_color=WHITE, is_circle=True)

    # Нижняя плашка-вывод
    draw.rounded_rectangle([140, 1455, 3060, 1615], radius=32, fill=LIGHT_BAR, outline=(208, 215, 245), width=2)
    draw_icon_badge(img, 225, 1535, 76, "web_nodes", bg_color=(208, 216, 248), fg_color=ROYAL_BLUE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_3_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


def build_slide4_bg() -> Path:
    """Слайд 4: КОНВЕЙЕР ИИ-МОДЕЛЕЙ И РАСЧЕТ RUL (Точный стиль Фото 2 со стрелками-шевронами)."""
    img = create_light_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний бейдж
    draw.rounded_rectangle([140, 110, 640, 164], radius=27, fill=LIGHT_PILL)

    # Полоса из 4 градиентных шевронов
    draw_chevron_bar(img, x_start=140, y_top=485, total_w=2920, h=190, steps=4, tip_depth=54, gap=16)

    col_centers = [465, 1195, 1930, 2665]
    icon_y = 825
    icons = ["vibration", "shield", "web_nodes", "monitor"]
    for cx, ic in zip(col_centers, icons):
        draw_icon_badge(img, cx, icon_y, 92, ic, bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)

    # Плашки точности под каждой колонкой
    for cx in col_centers:
        draw.rounded_rectangle([cx - 215, 1255, cx + 215, 1320], radius=22, fill=WHITE, outline=(210, 216, 245), width=2)

    # Нижняя широкая лавандовая плашка-вывод
    draw.rounded_rectangle([140, 1455, 3060, 1615], radius=32, fill=LIGHT_BAR, outline=(208, 215, 245), width=2)
    draw_icon_badge(img, 225, 1535, 76, "monitor", bg_color=(208, 216, 248), fg_color=ROYAL_BLUE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_4_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


def build_slide5_bg() -> Path:
    """Слайд 5: ЖИВАЯ ДЕМОНСТРАЦИЯ НА САЙТЕ (Темный индиго-стиль Фото 1 с карточками шагов и окнами До/После)."""
    img = create_dark_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний бейдж + бирюзовая полоска
    draw.rounded_rectangle([140, 135, 780, 193], radius=29, fill=DARK_PILL)
    draw.rounded_rectangle([140, 207, 340, 217], radius=5, fill=CYAN_ACCENT)

    # 3 карточки интерактивных шагов слева
    step_y = [560, 760, 960]
    step_cols = [(255, 92, 92), (255, 178, 36), GREEN_PULSE]
    step_icons = ["bolt", "vibration", "shield"]

    for sy, scol, sic in zip(step_y, step_cols, step_icons):
        draw.rounded_rectangle(
            [140, sy, 1350, sy + 165],
            radius=28,
            fill=(35, 42, 80, 235),
            outline=(68, 80, 138, 200),
            width=2,
        )
        draw_icon_badge(img, 225, sy + 82, 86, sic, bg_color=scol, fg_color=WHITE, is_circle=True)

    # Тонкая линия над KPI
    draw.line([(140, 1185), (1350, 1185)], fill=(52, 62, 112, 200), width=2)

    # Справа: крупное сравнение окон "ДО" и "ПОСЛЕ"
    paste_macos_window(img, ASSETS_DIR / "shot_threat.png", box=(1430, 130, 2940, 960), dark_frame=True)
    paste_macos_window(img, ASSETS_DIR / "shot_fixed.png", box=(1620, 815, 3050, 1630), dark_frame=False)

    # Плавающий белый бейдж результата
    draw_shadow_rounded_rect(
        img,
        box=(1365, 1245, 2030, 1435),
        radius=36,
        fill=(250, 251, 255, 255),
        outline=(225, 230, 248, 255),
        outline_width=2,
        shadow_offset=(0, 18),
        shadow_blur=30,
        shadow_alpha=85,
    )
    draw_icon_badge(img, 1455, 1340, 92, "shield", bg_color=GREEN_PULSE, fg_color=WHITE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_5_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


def build_slide6_bg() -> Path:
    """Слайд 6: ЦЕПОЧКА ВНЕДРЕНИЯ НА ЗАВОД И ЭКОНОМИКА (Точный стиль Фото 3 с пунктирной линией 1..5 и белыми карточками)."""
    img = create_light_canvas()
    draw = ImageDraw.Draw(img, "RGBA")

    # Верхний бейдж
    draw.rounded_rectangle([140, 110, 660, 164], radius=27, fill=LIGHT_PILL)

    card_w = 540
    gap = 55
    x_starts = [140 + i * (card_w + gap) for i in range(5)]
    centers_x = [xs + card_w // 2 for xs in x_starts]
    circles_y = [515, 495, 475, 455, 435]

    line_pts = list(zip(centers_x, circles_y)) + [(3030, 422)]
    for i in range(len(line_pts) - 1):
        x1, y1 = line_pts[i]
        x2, y2 = line_pts[i + 1]
        dist = math.hypot(x2 - x1, y2 - y1)
        dash_len = 22
        steps = int(dist // (dash_len * 1.8))
        for s in range(steps):
            t1 = (s * 1.8 * dash_len) / dist
            t2 = min(1.0, ((s * 1.8 + 1.0) * dash_len) / dist)
            dx1 = int(x1 + (x2 - x1) * t1)
            dy1 = int(y1 + (y2 - y1) * t1)
            dx2 = int(x1 + (x2 - x1) * t2)
            dy2 = int(y1 + (y2 - y1) * t2)
            draw.line([(dx1, dy1), (dx2, dy2)], fill=(150, 160, 225), width=5)

    ax, ay = 3035, 422
    draw.polygon([(ax, ay), (ax - 28, ay - 14), (ax - 28, ay + 14)], fill=(150, 160, 225))

    for cx, cy in zip(centers_x, circles_y):
        r = 46
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=ROYAL_BLUE)

    card_icons = ["gateway", "file_data", "server", "web_nodes", "roi_cap"]
    for xs, cy, cic in zip(x_starts, circles_y, card_icons):
        top_y = cy + 80
        bot_y = 1365
        draw_shadow_rounded_rect(
            img,
            box=(xs, top_y, xs + card_w, bot_y),
            radius=32,
            fill=WHITE,
            outline=(222, 226, 246),
            outline_width=2,
            shadow_offset=(0, 12),
            shadow_blur=22,
            shadow_alpha=30,
        )
        draw_icon_badge(img, xs + 84, top_y + 92, 88, cic, bg_color=LIGHT_PILL, fg_color=ROYAL_BLUE)

    # Нижняя темно-синяя плашка с ракетой
    draw_shadow_rounded_rect(
        img,
        box=(140, 1445, 3060, 1625),
        radius=34,
        fill=INDIGO_DARK,
        outline=None,
        shadow_offset=(0, 14),
        shadow_blur=24,
        shadow_alpha=55,
    )
    draw_icon_badge(img, 230, 1535, 82, "rocket", bg_color=(42, 50, 95), fg_color=ROYAL_BLUE, is_circle=True)

    out = SLIDES_BG_DIR / "slide_6_bg.png"
    img.convert("RGB").save(out, quality=95)
    return out


# ---------------------------------------------------------------------------
# СБОРКА ПРЕЗЕНТАЦИИ POWERPOINT (.PPTX) С НАТИВНЫМИ ТЕКСТАМИ
# ---------------------------------------------------------------------------
class SlideBuilder:
    """Помощник для добавления нативных текстовых блоков в PowerPoint и одновременного рендера превью."""

    def __init__(self, prs: Presentation, bg_path: Path, slide_idx: int):
        self.slide = prs.slides.add_slide(prs.slide_layouts[6])
        self.slide.shapes.add_picture(str(bg_path), Inches(0), Inches(0), width=prs.slide_width, height=prs.slide_height)
        self.preview_img = Image.open(bg_path).convert("RGBA")
        self.preview_draw = ImageDraw.Draw(self.preview_img, "RGBA")
        self.slide_idx = slide_idx

    def _get_pil_font(self, pt_size: float, bold: bool = False):
        px_size = int(round(pt_size * (DPI_SCALE / 72.0)))
        font_candidates = (
            ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"]
            if bold
            else ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"]
        )
        for fp in font_candidates:
            if Path(fp).exists():
                return ImageFont.truetype(fp, px_size)
        return ImageFont.load_default()

    def add_text(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        text: str,
        pt_size: float = 14,
        color: tuple[int, int, int] = WHITE,
        bold: bool = False,
        align: str = "left",
        valign: str = "top",
    ):
        """Добавляет редактируемый текстовый блок в PowerPoint и синхронно отрисовывает его на превью."""
        # Даем буфер ширины для PowerPoint, чтобы перенос строк шел строго по нашим явным '\n'
        ppt_x = x - 40 if align == "center" else x
        ppt_w = w + 80
        tb = self.slide.shapes.add_textbox(inches(ppt_x), inches(y), inches(ppt_w), inches(h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.margin_left = Inches(0)
        tf.margin_right = Inches(0)
        tf.margin_top = Inches(0)
        tf.margin_bottom = Inches(0)

        if valign == "middle":
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE

        lines = text.split("\n")
        for idx, line in enumerate(lines):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = line
            p.font.name = "Segoe UI"
            p.font.size = Pt(pt_size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor(*color)
            if align == "center":
                p.alignment = PP_ALIGN.CENTER
            elif align == "right":
                p.alignment = PP_ALIGN.RIGHT
            else:
                p.alignment = PP_ALIGN.LEFT

        font = self._get_pil_font(pt_size, bold=bold)
        wrapped_lines = []
        for raw_line in lines:
            words = raw_line.split(" ")
            cur = ""
            for word in words:
                test = f"{cur} {word}".strip()
                bbox = self.preview_draw.textbbox((0, 0), test, font=font)
                if (bbox[2] - bbox[0]) <= w or not cur:
                    cur = test
                else:
                    wrapped_lines.append(cur)
                    cur = word
            wrapped_lines.append(cur)

        line_h = int(pt_size * (DPI_SCALE / 72.0) * 1.25)
        total_th = len(wrapped_lines) * line_h
        cur_y = y + (h - total_th) // 2 if valign == "middle" else y

        for wl in wrapped_lines:
            bbox = self.preview_draw.textbbox((0, 0), wl, font=font)
            tw = bbox[2] - bbox[0]
            if align == "center":
                tx = x + (w - tw) // 2
            elif align == "right":
                tx = x + w - tw
            else:
                tx = x
            self.preview_draw.text((tx, cur_y), wl, font=font, fill=color + (255,))
            cur_y += line_h

    def save_preview(self):
        out = PREVIEW_DIR / f"slide_{self.slide_idx}_preview.png"
        self.preview_img.convert("RGB").save(out, quality=92)


def generate_deck():
    SLIDES_BG_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    prs = Presentation()
    prs.slide_width = Inches(13.333333)
    prs.slide_height = Inches(7.5)

    # =========================================================================
    # СЛАЙД 1: ТИТУЛЬНЫЙ ПРОДУКТОВЫЙ СЛАЙД (Стиль Фото 1 SANAU ENERGY)
    # =========================================================================
    bg1 = build_slide1_bg()
    s1 = SlideBuilder(prs, bg1, 1)

    s1.add_text(140, 145, 680, 58, "ИИ-ПЛАТФОРМА • АСУ ТП • КАЗАХСТАН", pt_size=9.5, color=(190, 202, 255), bold=True, align="center", valign="middle")
    s1.add_text(
        140,
        265,
        1220,
        560,
        "Предиктивная система\nAI-PromPlatform —\nотечественная цифровая\nоснова безотказной\nпромышленности",
        pt_size=24,
        color=WHITE,
        bold=True,
    )

    # Тексты внутри 4 овальных плашек параметров
    s1.add_text(224, 905, 185, 80, "Вибрация", pt_size=11.5, color=WHITE, bold=True, valign="middle")
    s1.add_text(524, 905, 185, 80, "Давление", pt_size=11.5, color=WHITE, bold=True, valign="middle")
    s1.add_text(824, 905, 180, 80, "Ток сети", pt_size=11.5, color=WHITE, bold=True, valign="middle")
    s1.add_text(1119, 905, 230, 80, "Температура", pt_size=11.5, color=WHITE, bold=True, valign="middle")

    # 3 крупные цифры KPI внизу слева (как 126 041 | 15+ | 24/7 на Фото 1)
    s1.add_text(140, 1125, 390, 115, "99.89%", pt_size=27, color=WHITE, bold=True)
    s1.add_text(140, 1255, 390, 65, "точность ансамбля ИИ", pt_size=11, color=(175, 186, 225))

    s1.add_text(555, 1125, 390, 115, "48–72 ч", pt_size=27, color=WHITE, bold=True)
    s1.add_text(555, 1255, 390, 65, "фора до отказа узла", pt_size=11, color=(175, 186, 225))

    s1.add_text(975, 1125, 390, 115, "24/7", pt_size=27, color=WHITE, bold=True)
    s1.add_text(975, 1255, 390, 65, "контроль Read-Only", pt_size=11, color=(175, 186, 225))

    # Плавающий белый бейдж справа
    s1.add_text(1525, 1215, 440, 50, "на защите сейчас", pt_size=10.5, color=SLATE_MUTED)
    s1.add_text(1525, 1268, 440, 75, "4 узла • 0 сбоев", pt_size=16, color=INDIGO_DARK, bold=True)

    # Нижний колонтитул
    s1.add_text(140, 1615, 1200, 55, "МИТУ (METU) • Специальность 6B07102 «АиУ» • Искрин Д. А. • 2026", pt_size=10, color=(145, 156, 200))
    s1.save_preview()

    # =========================================================================
    # СЛАЙД 2: ПРОБЛЕМА И READ-ONLY АРХИТЕКТУРА (Стиль Фото 2 с шевронами)
    # =========================================================================
    bg2 = build_slide2_bg()
    s2 = SlideBuilder(prs, bg2, 2)

    s2.add_text(140, 110, 480, 54, "02 • БЕЗОПАСНОЕ РЕШЕНИЕ", pt_size=9.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")
    s2.add_text(
        140,
        185,
        2750,
        175,
        "AI-PromPlatform считывает потоки автоматически\nи работает поверх штатной АСУ ТП / SCADA",
        pt_size=22.5,
        color=INDIGO_DARK,
        bold=True,
    )
    s2.add_text(
        140,
        380,
        2600,
        60,
        "Износ фондов РК составляет 60–66%. Единый контур от датчика цеха до раннего предупреждения",
        pt_size=12,
        color=SLATE_MUTED,
    )

    chev_centers = [465, 1195, 1930, 2665]
    chev_titles = ["Датчики агрегатов", "Штатная SCADA", "Шлюз Read-Only", "Перехват за 72 ч"]
    chev_tcolors = [INDIGO_DARK, INDIGO_DARK, WHITE, WHITE]
    for cx, ctitle, ctcol in zip(chev_centers, chev_titles, chev_tcolors):
        s2.add_text(cx - 310, 485, 620, 190, ctitle, pt_size=15.5, color=ctcol, bold=True, align="center", valign="middle")

    col_data_s2 = [
        ("Вибрация, давление,\nток, температура", "непрерывный съем\nс 4 агрегатов линии"),
        ("OPC UA • Modbus TCP\nПЛК и контроллеры", "штатная система видит\nтолько поздний перегрев"),
        ("Одностороннее чтение\nбез команд в ПЛК", "0% риска вмешательства\nв техпроцесс завода"),
        ("Ранний прогноз ИИ\nЗащита от простоев", "таймер RUL в часах\nи кнопка стабилизации"),
    ]
    for cx, (main_txt, sub_txt) in zip(chev_centers, col_data_s2):
        s2.add_text(cx - 310, 935, 620, 120, main_txt, pt_size=13.5, color=INDIGO_DARK, bold=True, align="center")
        s2.add_text(cx - 310, 1080, 620, 110, sub_txt, pt_size=11.5, color=SLATE_MUTED, align="center")

    s2.add_text(
        295,
        1455,
        2720,
        160,
        "Безопасный закрытый контур — полное соответствие Указу «Digital Qazaqstan» (09.06.2026) и МЭК 62443",
        pt_size=13.5,
        color=INDIGO_DARK,
        bold=True,
        valign="middle",
    )
    s2.add_text(140, 1655, 1500, 50, "AI-PromPlatform • Архитектура промышленной безопасности Read-Only", pt_size=9.5, color=SLATE_MUTED)
    s2.save_preview()

    # =========================================================================
    # СЛАЙД 3: МУЛЬТИАГЕНТНАЯ ПАУТИНА ИИ (Распределение нагрузки по отделам)
    # =========================================================================
    bg3 = build_slide3_bg()
    s3 = SlideBuilder(prs, bg3, 3)

    s3.add_text(140, 110, 550, 54, "03 • МУЛЬТИАГЕНТНАЯ ПАУТИНА", pt_size=9.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")
    s3.add_text(
        140,
        185,
        2850,
        175,
        "За каждым отделом следит своя команда агентов —\nМастер-Модель принимает только главное",
        pt_size=22.5,
        color=INDIGO_DARK,
        bold=True,
    )
    s3.add_text(
        140,
        380,
        2750,
        60,
        "Иерархическая паутина ИИ: локальные агенты гасят 99% фоновой мелочи на узлах, защищая Центр от перегрузки",
        pt_size=12,
        color=SLATE_MUTED,
    )

    depts_info = [
        (140, 465, "Отдел 1 • Насосная", "Агрегат ЦНС-180 (PUMP-101)", "Агенты давления и кавитации\nгасят фоновую рябь потока", "Локальный фильтр: 99% шума отсечено"),
        (140, 965, "Отдел 2 • Электроприводы", "Двигатель АИР-315 (MOTOR-204)", "Агенты вибродиагностики и тока\nфильтруют пусковые скачки", "Локальный фильтр: 99% шума отсечено"),
        (2180, 465, "Отдел 3 • Компрессорная", "Компрессор ВП3-20 (COMP-302)", "Агенты термоконтроля клапанов\nотсекают сезонный нагрев", "Локальный фильтр: 99% шума отсечено"),
        (2180, 965, "Отдел 4 • Конвейеры", "Привод ленты ПЛ-12 (DRIVE-405)", "Агенты натяжения и редукторов\nгасят вибрацию стыков ленты", "Локальный фильтр: 99% шума отсечено"),
    ]
    for dx, dy, dtitle, dsub, ddesc, dbadge in depts_info:
        s3.add_text(dx + 140, dy + 40, 710, 50, dtitle, pt_size=14, color=INDIGO_DARK, bold=True)
        s3.add_text(dx + 140, dy + 92, 710, 45, dsub, pt_size=10.5, color=ROYAL_BLUE, bold=True)
        s3.add_text(dx + 45, dy + 158, 790, 100, ddesc, pt_size=11.5, color=SLATE_MUTED)
        s3.add_text(dx + 45, dy + 282, 790, 52, dbadge, pt_size=10.5, color=(16, 145, 82), bold=True, align="center", valign="middle")

    for bx, by in [(1195, 735), (1195, 1075), (2005, 735), (2005, 1075)]:
        s3.add_text(bx - 155, by - 28, 310, 56, "1% вектор угрозы", pt_size=9.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")

    # Текст внутри Центрального Ядра (Мастер-Модель)
    s3.add_text(1390, 840, 420, 80, "МАСТЕР-МОДЕЛЬ\nКООРДИНАТОР", pt_size=13, color=WHITE, bold=True, align="center")
    s3.add_text(1400, 940, 400, 140, "Принимает только\nподтвержденные сбои,\nсчитает ресурс RUL\nи выдает решение", pt_size=10, color=(198, 208, 255), align="center")

    s3.add_text(
        295,
        1455,
        2720,
        160,
        "Распределенная паутина ИИ — нагрузка на центральный сервер ниже в 100 раз, отклик системы менее 0.2 сек",
        pt_size=13.5,
        color=INDIGO_DARK,
        bold=True,
        valign="middle",
    )
    s3.add_text(140, 1655, 1600, 50, "AI-PromPlatform • Мультиагентная архитектура фильтрации промышленной телеметрии", pt_size=9.5, color=SLATE_MUTED)
    s3.save_preview()

    # =========================================================================
    # СЛАЙД 4: МАТЕМАТИЧЕСКОЕ ЯДРО И КОНВЕЙЕР (Стиль Фото 2 с шевронами)
    # =========================================================================
    bg4 = build_slide4_bg()
    s4 = SlideBuilder(prs, bg4, 4)

    s4.add_text(140, 110, 500, 54, "04 • МАТЕМАТИЧЕСКОЕ ЯДРО", pt_size=9.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")
    s4.add_text(
        140,
        185,
        2750,
        175,
        "Ансамбль из 3 моделей выявляет причину сбоя\nи рассчитывает точный остаточный ресурс (RUL)",
        pt_size=22.5,
        color=INDIGO_DARK,
        bold=True,
    )
    s4.add_text(
        140,
        380,
        2650,
        60,
        "Четыре ступени верификации сигнала с модулем объяснимого ИИ (Explainable AI) для главного инженера",
        pt_size=12,
        color=SLATE_MUTED,
    )

    chev4_titles = ["Съем и дельты", "Isolation Forest", "Random Forest", "XGBoost + XAI"]
    for cx, ctitle, ctcol in zip(chev_centers, chev4_titles, chev_tcolors):
        s4.add_text(cx - 310, 485, 620, 190, ctitle, pt_size=15.5, color=ctcol, bold=True, align="center", valign="middle")

    col_data_s4 = [
        ("Скользящее окно\nи градиенты", "расчет скорости роста\nвибрации и температуры", "12 признаков • 60 мин"),
        ("Узловой фильтр\nаномалий", "отсекает фоновый шум\nна уровне отдела цеха", "Точность отсева 98.52%"),
        ("Классификация\nпричины угрозы", "распознает износ, клин,\nкавитацию и перегрев", "Accuracy 99.89%"),
        ("Прогноз часов RUL\nи вклад датчиков", "точный таймер до отказа\nи объяснение причины в %", "R² = 0.9735 • XAI"),
    ]
    for cx, (mtxt, stxt, badge_txt) in zip(chev_centers, col_data_s4):
        s4.add_text(cx - 310, 935, 620, 120, mtxt, pt_size=13.5, color=INDIGO_DARK, bold=True, align="center")
        s4.add_text(cx - 310, 1075, 620, 110, stxt, pt_size=11.5, color=SLATE_MUTED, align="center")
        s4.add_text(cx - 215, 1255, 430, 65, badge_txt, pt_size=10.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")

    s4.add_text(
        295,
        1455,
        2720,
        160,
        "Никакого «чёрного ящика» — оператор видит точный вклад каждого датчика (вибрация 54%, нагрев 31%, ток 15%)",
        pt_size=13.5,
        color=INDIGO_DARK,
        bold=True,
        valign="middle",
    )
    s4.add_text(140, 1655, 1500, 50, "AI-PromPlatform • Валидация на выборке 8 000 промышленных циклов", pt_size=9.5, color=SLATE_MUTED)
    s4.save_preview()

    # =========================================================================
    # СЛАЙД 5: ЖИВАЯ ДЕМОНСТРАЦИЯ НА САЙТЕ (Темный стиль Фото 1)
    # =========================================================================
    bg5 = build_slide5_bg()
    s5 = SlideBuilder(prs, bg5, 5)

    s5.add_text(140, 135, 640, 58, "05 • ИНТЕРАКТИВНЫЙ ПРОТОТИП", pt_size=9.5, color=(190, 202, 255), bold=True, align="center", valign="middle")
    s5.add_text(
        140,
        240,
        1220,
        280,
        "Живая демонстрация:\nсимуляция угрозы, перехват\nи выравнивание в 1 клик",
        pt_size=22,
        color=WHITE,
        bold=True,
    )

    steps_s5 = [
        (560, "1. Симуляция скрытой угрозы на узле", "Запуск деградации подшипника или кавитации.\nШтатная SCADA молчит — порог еще не пробит."),
        (760, "2. Перехват ИИ и обратный отсчет RUL", "Мгновенное уведомление: «Через 77.3 ч авария!»\nРиск отказа 100%, указана точная причина."),
        (960, "3. Кнопка «Исправить» и выравнивание", "Автоматический перевод нагрузки на резерв.\nГрафики вибрации и нагрева возвращаются в норму."),
    ]
    for sy, stitle, sdesc in steps_s5:
        s5.add_text(295, sy + 24, 1030, 48, stitle, pt_size=13, color=WHITE, bold=True)
        s5.add_text(295, sy + 76, 1030, 75, sdesc, pt_size=10.5, color=(185, 196, 235))

    s5.add_text(140, 1225, 390, 100, "2.37 мм/с", pt_size=24, color=CYAN_ACCENT, bold=True)
    s5.add_text(140, 1335, 390, 65, "норма после 5.62 мм/с", pt_size=10.5, color=(175, 186, 225))

    s5.add_text(555, 1225, 410, 100, "204.1 ч", pt_size=24, color=WHITE, bold=True)
    s5.add_text(555, 1335, 410, 65, "восстановленный ресурс", pt_size=10.5, color=(175, 186, 225))

    s5.add_text(985, 1225, 370, 100, "0.0%", pt_size=24, color=GREEN_PULSE, bold=True)
    s5.add_text(985, 1335, 370, 65, "риск после стабилизации", pt_size=10.5, color=(175, 186, 225))

    s5.add_text(1525, 1275, 480, 48, "результат стабилизации", pt_size=10.5, color=SLATE_MUTED)
    s5.add_text(1525, 1328, 480, 75, "Параметры в норме", pt_size=13, color=INDIGO_DARK, bold=True)

    s5.add_text(140, 1615, 1200, 55, "Демонстрационный стенд работает локально в реальном времени (Streamlit + Plotly)", pt_size=10, color=(145, 156, 200))
    s5.save_preview()

    # =========================================================================
    # СЛАЙД 6: ЦЕПОЧКА ВНЕДРЕНИЯ 1..5 И ЭКОНОМИКА (Точный стиль Фото 3)
    # =========================================================================
    bg6 = build_slide6_bg()
    s6 = SlideBuilder(prs, bg6, 6)

    s6.add_text(140, 110, 520, 54, "06 • ВНЕДРЕНИЕ И ЭКОНОМИКА", pt_size=9.5, color=ROYAL_BLUE, bold=True, align="center", valign="middle")
    s6.add_text(
        140,
        185,
        2650,
        100,
        "Как проходит подключение к заводу и экономический эффект",
        pt_size=23.5,
        color=INDIGO_DARK,
        bold=True,
    )
    s6.add_text(
        140,
        295,
        2500,
        60,
        "Пилот на одном цехе за 14 дней с масштабированием паутины ИИ-агентов на всё предприятие",
        pt_size=12,
        color=SLATE_MUTED,
    )

    card_w = 540
    gap = 55
    x_starts = [140 + i * (card_w + gap) for i in range(5)]
    circles_y = [515, 495, 475, 455, 435]

    steps_s6 = [
        ("Доступ к системе", "Подключение шлюза\nRead-Only к штатной\nSCADA и ПЛК завода\nпо OPC UA / Modbus"),
        ("Выгрузка данных", "Импорт архивов\nтелеметрии и журналов\nотказов из базы/Excel,\nразметка датчиков"),
        ("Узловые агенты", "Установка легких\nИИ-агентов в Docker\nпо отделам цеха и\nкалибровка нормы"),
        ("Мастер-модель", "Сборка паутины ИИ,\nзапуск расчета RUL\nи интеграция алертов\nс системой ТОиР"),
        ("Экономия 24/7", "Плановый сервис за\n185 тыс. ₸ вместо\nаварийного простоя\nцеха на 57,7 млн ₸"),
    ]

    for i, (xs, cy, (ctitle, cdesc)) in enumerate(zip(x_starts, circles_y, steps_s6)):
        cx = xs + card_w // 2
        s6.add_text(cx - 46, cy - 46, 92, 92, str(i + 1), pt_size=15, color=WHITE, bold=True, align="center", valign="middle")
        top_y = cy + 80
        s6.add_text(xs + 40, top_y + 165, card_w - 80, 60, ctitle, pt_size=14.5, color=INDIGO_DARK, bold=True)
        s6.add_text(xs + 40, top_y + 240, card_w - 80, 300, cdesc, pt_size=11.5, color=SLATE_MUTED)

    s6.add_text(
        300,
        1445,
        2720,
        180,
        "Экономический эффект — 57,5 млн ₸ экономии на одном сбое и снижение простоев на 30–40%",
        pt_size=15,
        color=WHITE,
        bold=True,
        valign="middle",
    )
    s6.add_text(140, 1655, 1700, 50, "AI-PromPlatform • Расчет эффекта для непрерывной линии (120 т/ч) • Хакатон AI-SANA 2026", pt_size=9.5, color=SLATE_MUTED)
    s6.save_preview()

    prs.save(str(OUTPUT_PPTX))
    print(f"Presentation successfully saved to: {OUTPUT_PPTX}")


if __name__ == "__main__":
    generate_deck()
