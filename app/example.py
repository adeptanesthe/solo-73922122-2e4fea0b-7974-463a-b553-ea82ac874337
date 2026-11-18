"""内置“交叉遮挡”示例会话数据。

场景：6 帧 320x240 画面。两只外观不同的动物（A 偏红、B 偏蓝）
沿对角线相向而行，在第 3 帧（0 起）于画面中央交汇：
- B 在第 3 帧被 A 遮挡，缺失检测 → 需要跨 2 帧连接；
- A 在第 3 帧的检测框外观向量被 B 的颜色污染（余弦相似度下降），
  模拟遮挡时的表观退化；
- 两动物在第 2/4 帧空间位置非常接近，朴素逐帧贪心关联容易发生
  串号（ID switch），而全局最优解能给出正确的 2 条轨迹。

外观向量为 4 维（近似颜色直方图），均非零。
"""

from __future__ import annotations

import io
import math
import random

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 320, 240
N_FRAMES = 6

# 每只动物的中心轨迹（帧号 -> (cx, cy)），两条路径在画面中央呈 X 形交叉
A_CENTERS = [(40, 60), (82, 86), (124, 112), (166, 138), (208, 164), (250, 190)]
B_CENTERS = [(290, 60), (248, 86), (206, 112), (164, 138), (122, 164), (80, 190)]
B_OCCLUDED_FRAME = 3  # B 在该帧被 A 遮挡，无检测框

A_VEC = (0.90, 0.10, 0.05, 0.02)
B_VEC = (0.05, 0.10, 0.90, 0.02)
# A 在遮挡帧的表观被 B 污染
A_VEC_OCCLUDED = (0.60, 0.10, 0.35, 0.02)

BOX = 28  # 检测框边长（像素）


def _jittered(vec, rng, sigma=0.015):
    out = [max(0.01, v + rng.gauss(0, sigma)) for v in vec]
    return [round(v, 4) for v in out]


def build_example_detections():
    """返回示例检测框列表（dict）。"""
    rng = random.Random(20240611)
    dets = []
    for f, (cx, cy) in enumerate(A_CENTERS):
        vec = A_VEC_OCCLUDED if f == B_OCCLUDED_FRAME else A_VEC
        dets.append(
            {
                "id": f"A{f}",
                "frame": f,
                "rect": [cx - BOX // 2, cy - BOX // 2, BOX, BOX],
                "vector": _jittered(vec, rng),
            }
        )
    for f, (cx, cy) in enumerate(B_CENTERS):
        if f == B_OCCLUDED_FRAME:
            continue  # 被遮挡，无检测
        dets.append(
            {
                "id": f"B{f}",
                "frame": f,
                "rect": [cx - BOX // 2, cy - BOX // 2, BOX, BOX],
                "vector": _jittered(B_VEC, rng),
            }
        )
    dets.sort(key=lambda d: (d["frame"], d["id"]))
    return dets


def _draw_animal(draw, cx, cy, rgb, r=13):
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb, outline=(20, 20, 20))


def build_example_frames():
    """返回 [(content_type, png_bytes), ...] 共 6 帧。"""
    rng = random.Random(7)
    frames = []
    for f in range(N_FRAMES):
        img = Image.new("RGB", (WIDTH, HEIGHT), (34, 52, 34))
        draw = ImageDraw.Draw(img)
        # 草地噪点
        for _ in range(450):
            x = rng.randrange(WIDTH)
            y = rng.randrange(HEIGHT)
            g = rng.randrange(45, 90)
            draw.point((x, y), fill=(g // 2, g, g // 3))
        # 先画 B（被遮挡时在下层），再画 A
        bx, by = B_CENTERS[f]
        if f == B_OCCLUDED_FRAME:
            # B 几乎完全被 A 挡住，只露出边缘一小角
            _draw_animal(draw, bx, by, (60, 60, 140), r=13)
        else:
            _draw_animal(draw, bx, by, (40, 40, 220), r=13)
        ax, ay = A_CENTERS[f]
        _draw_animal(draw, ax, ay, (220, 40, 40), r=14)
        draw.text((6, 4), f"frame {f}", fill=(240, 240, 240))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append(("image/png", buf.getvalue()))
    return frames


EXAMPLE_DESCRIPTION = (
    "交叉遮挡示例：A（红）与 B（蓝）相向穿行，第 3 帧 B 被 A 遮挡缺失检测、"
    "A 的外观被污染。正确结果应为 2 条轨迹（2 只动物）；"
    "若只按最近邻逐帧贪心，容易在交汇点串号或把 B 断成两条而重复计数。"
)


# ---------------------------------------------------------------- 曝光抖动示例
#
# 6 帧 320x240 画面：背景为带亮度梯度与彩石的静态场景，两只动物
# （A 红棕、B 蓝灰）沿对角线相向而行、全程可见。每帧叠加不同的整体增益与
# 红/蓝通道白平衡系数，模拟自动曝光与自动白平衡抖动 —— 直接逐帧比对毛色
# 会有明显色差，适合演示“相邻帧色差校正”。

A2_COLOR = (168, 78, 58)   # 红棕
B2_COLOR = (72, 92, 168)   # 蓝灰
A2_CENTERS = [(round(36 + 49.6 * f), round(52 + 25.6 * f)) for f in range(N_FRAMES)]
B2_CENTERS = [(round(284 - 49.6 * f), round(52 + 25.6 * f)) for f in range(N_FRAMES)]

A2_VEC = (0.80, 0.35, 0.25, 0.02)
B2_VEC = (0.30, 0.40, 0.80, 0.02)

# 每帧整体增益与逐通道白平衡系数（绿色通道为基准 1.0）
EX_GAIN = (1.00, 1.08, 0.92, 1.10, 0.88, 1.04)
EX_WB_R = (1.00, 1.04, 0.96, 1.06, 0.94, 1.02)
EX_WB_B = (1.00, 0.96, 1.04, 0.94, 1.06, 0.98)


def _exposure_background(rng):
    """静态背景：亮度梯度 + 宽色域彩石 + 固定噪点（各帧完全一致）。

    彩石避开所有帧的动物位置，保证检测框内只有动物与平滑背景，
    动物像素标定范围检查不受彩石干扰。
    """
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    luma = 55 + 120 * (0.55 * xx / (WIDTH - 1) + 0.45 * yy / (HEIGHT - 1))
    bg = np.stack([0.62 * luma, luma, 0.52 * luma], axis=-1)
    animal_xy = A2_CENTERS + B2_CENTERS
    # 彩石：让各通道样本覆盖到高亮区，支撑可信标定范围
    placed = 0
    while placed < 140:
        cx = rng.randrange(WIDTH)
        cy = rng.randrange(HEIGHT)
        r = rng.randrange(2, 6)
        if any((cx - ax) ** 2 + (cy - ay) ** 2 < (r + 22) ** 2 for ax, ay in animal_xy):
            continue  # 不撒进检测框附近
        color = [rng.randrange(90, 236) for _ in range(3)]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        bg[mask] = color
        placed += 1
    # 固定噪点纹理
    for _ in range(500):
        x = rng.randrange(WIDTH)
        y = rng.randrange(HEIGHT)
        g = rng.randrange(40, 110)
        bg[y, x] = (g // 2, g, g // 3)
    return bg


def build_exposure_example_frames():
    """返回 [(content_type, png_bytes), ...] 共 6 帧（带曝光/白平衡抖动）。"""
    rng = random.Random(20261118)
    bg = _exposure_background(rng)
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    frames = []
    for f in range(N_FRAMES):
        arr = bg.copy()
        for (cx, cy), color in ((A2_CENTERS[f], A2_COLOR), (B2_CENTERS[f], B2_COLOR)):
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 13 * 13
            arr[mask] = color
        # 帧号标签先画进场景，再随整帧一起做曝光/白平衡变换
        label = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")
        ImageDraw.Draw(label).text((6, 4), f"frame {f}", fill=(230, 230, 230))
        arr = np.asarray(label, dtype=float)
        gains = (EX_GAIN[f] * EX_WB_R[f], EX_GAIN[f], EX_GAIN[f] * EX_WB_B[f])
        arr = np.clip(arr * np.asarray(gains), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGB").save(buf, format="PNG")
        frames.append(("image/png", buf.getvalue()))
    return frames


def build_exposure_example_detections():
    """返回曝光抖动示例的检测框列表（dict）。"""
    rng = random.Random(20261119)
    dets = []
    for f in range(N_FRAMES):
        for tag, centers, vec in (("A", A2_CENTERS, A2_VEC), ("B", B2_CENTERS, B2_VEC)):
            cx, cy = centers[f]
            dets.append(
                {
                    "id": f"{tag}{f}",
                    "frame": f,
                    "rect": [cx - BOX // 2, cy - BOX // 2, BOX, BOX],
                    "vector": _jittered(vec, rng),
                }
            )
    dets.sort(key=lambda d: (d["frame"], d["id"]))
    return dets


EXPOSURE_EXAMPLE_DESCRIPTION = (
    "曝光抖动示例：A（红棕）与 B（蓝灰）相向穿行、全程可见（2 只动物）；"
    "每帧整体增益与红/蓝白平衡系数不同，模拟自动曝光与自动白平衡抖动。"
    "在“相邻帧色差校正”中逐帧估计背景亮度映射，可消除色差后再比对毛色。"
)
