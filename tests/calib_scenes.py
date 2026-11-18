"""合成场景构造：供色差校正的单元测试与 API 测试共用。

所有场景均为 96×72 灰度/彩色渐变背景 + 两只移动动物，
通过在 B 帧上叠加逐通道增益模拟自动曝光 / 白平衡变化。
"""

import io

import numpy as np
from PIL import Image

W, H = 96, 72


def gradient_bg(w=W, h=H, lo=40.0, hi=200.0, noise=2.0, seed=1):
    """全幅灰度渐变背景（含轻微噪声），明暗跨度大。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = lo + (hi - lo) * (0.6 * xx / (w - 1) + 0.4 * yy / (h - 1))
    bg = np.repeat(base[:, :, None], 3, axis=2)
    bg = bg + rng.normal(0.0, noise, (h, w, 3))
    return np.clip(bg, 0, 255)


def draw_animals(bg, animals):
    """在背景副本上画实心圆动物。animals: [(cx, cy, r, (r, g, b))]。"""
    img = bg.copy()
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, cy, r, color in animals:
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = color
    return img


def transform_gains(img, gains):
    """逐通道增益（模拟曝光 / 白平衡变化），截断到 [0, 255]。"""
    g = np.asarray(gains, dtype=float)
    return np.clip(img.astype(float) * g, 0, 255)


def to_png(arr):
    img = Image.fromarray(np.clip(np.rint(arr), 0, 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def from_png(data):
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=float)


def boxes_for(animals, pad=2):
    return [
        [cx - r - pad, cy - r - pad, 2 * (r + pad), 2 * (r + pad)]
        for cx, cy, r, _ in animals
    ]


def _pair(bg, animals_a, animals_b, gains):
    frame_a = draw_animals(bg, animals_a)
    frame_b = transform_gains(draw_animals(bg, animals_b), gains)
    return {
        "a": to_png(frame_a),
        "b": to_png(frame_b),
        "a_arr": frame_a,
        "boxes_a": boxes_for(animals_a),
        "boxes_b": boxes_for(animals_b),
    }


def valid_pair():
    """有效场景：均匀曝光/白平衡变化，动物颜色在背景覆盖范围内。"""
    bg = gradient_bg()
    a_pos = [(20, 18, 7, (150, 90, 60)), (70, 50, 7, (60, 100, 150))]
    b_pos = [(26, 22, 7, (150, 90, 60)), (64, 46, 7, (60, 100, 150))]
    return _pair(bg, a_pos, b_pos, (1.15 * 1.05, 1.15, 1.15 * 0.95))


def conflict_pair():
    """歧义场景：左半幅变亮、右半幅变暗 —— 各区域映射相互矛盾。"""
    bg = gradient_bg()
    left = transform_gains(bg, (1.25, 1.25, 1.25))
    right = transform_gains(bg, (0.85, 0.85, 0.85))
    frame_b = bg.copy()
    frame_b[:, : W // 2] = left[:, : W // 2]
    frame_b[:, W // 2:] = right[:, W // 2:]
    animals_a = [(30, 30, 6, (120, 110, 100))]
    animals_b = [(34, 32, 6, (120, 110, 100))]
    frame_b = draw_animals(frame_b, animals_b)
    return {
        "a": to_png(draw_animals(bg, animals_a)),
        "b": to_png(frame_b),
        "boxes_a": boxes_for(animals_a),
        "boxes_b": boxes_for(animals_b),
    }


def flat_pair():
    """歧义场景：背景近乎均匀灰 —— 明暗跨度不足。"""
    rng = np.random.default_rng(5)
    bg = np.clip(100.0 + rng.normal(0, 1.5, (H, W, 3)), 0, 255)
    a_pos = [(24, 24, 6, (110, 105, 95))]
    b_pos = [(30, 28, 6, (110, 105, 95))]
    return _pair(bg, a_pos, b_pos, (1.10, 1.10, 1.10))


def saturated_pair():
    """歧义场景：B 帧 97% 像素饱和 —— 有效样本不足。"""
    bg = gradient_bg()
    rng = np.random.default_rng(7)
    frame_b = transform_gains(bg, (1.1, 1.1, 1.1))
    mask = rng.random((H, W)) < 0.97
    frame_b[mask] = 255.0
    a_pos = [(24, 24, 6, (150, 90, 60))]
    b_pos = [(28, 26, 6, (150, 90, 60))]
    return {
        "a": to_png(draw_animals(bg, a_pos)),
        "b": to_png(draw_animals(frame_b, b_pos)),
        "boxes_a": boxes_for(a_pos),
        "boxes_b": boxes_for(b_pos),
    }


def changed_pair():
    """歧义场景：左 60% 背景被未标记的大面积变化替换 —— 样本分布过于集中。"""
    bg = gradient_bg()
    rng = np.random.default_rng(11)
    frame_b = transform_gains(bg, (1.1, 1.1, 1.1))
    cut = int(W * 0.6)
    frame_b[:, :cut] = rng.uniform(0, 255, (H, cut, 3))
    a_pos = [(78, 30, 6, (150, 90, 60))]
    b_pos = [(80, 32, 6, (150, 90, 60))]
    return {
        "a": to_png(draw_animals(bg, a_pos)),
        "b": to_png(draw_animals(frame_b, b_pos)),
        "boxes_a": boxes_for(a_pos),
        "boxes_b": boxes_for(b_pos),
    }


def animal_out_of_range_pair():
    """歧义场景：动物为鲜红色，远超灰度背景覆盖的标定范围。"""
    bg = gradient_bg(lo=40.0, hi=150.0)
    a_pos = [(30, 30, 7, (240, 40, 40))]
    b_pos = [(36, 34, 7, (240, 40, 40))]
    return _pair(bg, a_pos, b_pos, (1.0, 1.0, 1.0))
