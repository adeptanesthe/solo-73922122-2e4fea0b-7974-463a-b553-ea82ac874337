"""相邻帧色差校正：由检测框外背景估计各颜色通道的单调亮度映射。

用途
----
自动曝光（AE）与自动白平衡（AWB）会使同一背景在相邻两帧呈现不同的
亮度/色偏，直接逐帧比对毛色会产生色差。本模块以“参考帧 A → 待校正帧 B”
为输入，估计逐通道单调亮度映射，把 B 帧校正到 A 帧的曝光/白平衡下。

样本选取（仅用于估计，不修改任何原图）
--------------------------------------
- 仅使用两帧检测框并集（外扩 2 像素）以外的背景像素；
- 剔除饱和像素：任一帧任一通道 ≤ SAT_LO 或 ≥ SAT_HI；
- 剔除变化区域：对“每通道鲁棒线性变换”的残差超过 CHANGED_TOL 的像素
  （未标注的运动目标、摇曳枝叶等），线性系数由分位数匹配估计并迭代一次；
  变化掩膜再外扩 1 像素，挡住边缘抗锯齿/光晕等半变化像素。

可靠性门槛（任一不满足即 ok=False，不产出校正图）
------------------------------------------------
- 有效样本数 ≥ MIN_SAMPLES；
- 样本空间分散：4×4 网格中至少 MIN_COVERED_CELLS 格各有足够样本；
- 明暗跨度：亮度 p95−p5 ≥ SPAN_LUMA_MIN，且每通道 p95−p5 ≥ SPAN_CH_MIN；
- 每通道映射：按 B 帧样本值分箱取中位数，加权保序回归（PAVA）保证单调不减，
  插值为 256 级查找表；有效分箱过少则拒绝；
- 区域一致性：2×2 子区域分别估计局部映射，与全局映射的最大偏差
  ≤ CONFLICT_TOL 灰级，否则判定各区域映射矛盾；
- 动物像素检查：B 帧检测框内像素落在样本可信标定范围
  （每通道 p0.5~p99.5 外扩 TRUST_MARGIN）之外的比例 ≤ ANIMAL_OUT_MAX_FRAC，
  否则校正需外推，判定不可信。
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

MAX_DIM = 512  # 帧宽、高上限

SAT_LO = 6  # 饱和判定：通道值 ≤ SAT_LO 或 ≥ SAT_HI 视为截断，不参与估计
SAT_HI = 249

BOX_DILATE = 2  # 检测框外扩像素，挡住边缘渗漏的动物像素

CHANGED_TOL = 26.0  # 变化区域判定：去全局变换后的逐通道残差阈值（灰级）
CHANGED_DILATE = 1  # 变化区域外扩像素：挡住边缘抗锯齿/光晕等半变化像素

MIN_SAMPLES = 300  # 有效标定样本下限
GRID = 4  # 分散度网格（GRID×GRID）
MIN_PER_CELL = 6  # 每格至少样本数，才计为“已覆盖”
MIN_COVERED_CELLS = 10  # 至少覆盖的格数（共 16 格）

SPAN_LUMA_MIN = 48.0  # 亮度（luma）p95−p5 最小跨度
SPAN_CH_MIN = 30.0  # 每通道 p95−p5 最小跨度

N_BINS = 24  # 映射分箱数
MIN_BIN_COUNT = 3  # 每箱最少样本数
MIN_CURVE_BINS = 4  # 拟合曲线至少需要的有效分箱数

REGION_GRID = 2  # 区域一致性网格（REGION_GRID×REGION_GRID）
MIN_REGION_SAMPLES = 120  # 子区域参与一致性校验的最少样本数
MIN_REGION_BIN_COUNT = 4  # 子区域内每箱最少样本数
MIN_CONSISTENCY_POINTS = 6  # 一致性可比对的（区域×箱×通道）最少点数
CONFLICT_TOL = 20.0  # 区域映射矛盾阈值（灰级）

TRUST_LO_Q = 0.5  # 可信标定范围分位（%）
TRUST_HI_Q = 99.5
TRUST_MARGIN = 8.0  # 可信范围外扩灰级
ANIMAL_OUT_MAX_FRAC = 0.02  # 动物像素越界比例上限
ANIMAL_OUT_MIN_COUNT = 12  # 且越界像素数需超过该值才判定（忽略零星边缘点）

_CHANNEL_NAMES = ("R", "G", "B")


class CalibrationInputError(Exception):
    """输入层面无法满足校正前提（尺寸、格式等），对应 HTTP 400。"""


# ---------------------------------------------------------------- 基础工具

def _decode(img_bytes: bytes, label: str) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)
    except Exception as exc:  # noqa: BLE001 - 任何解码失败都归为输入错误
        raise CalibrationInputError(f"{label}无法解码为图像：{exc}") from exc
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise CalibrationInputError(f"{label}不是有效的三通道图像。")
    return arr


def _rasterize_boxes(shape, boxes, dilate) -> np.ndarray:
    """把 [x, y, w, h] 框列表栅格化为布尔掩膜（裁剪到图内，可外扩）。"""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        x, y, bw, bh = (float(v) for v in box)
        x0 = max(0, int(math.floor(x)) - dilate)
        y0 = max(0, int(math.floor(y)) - dilate)
        x1 = min(w, int(math.ceil(x + bw)) + dilate)
        y1 = min(h, int(math.ceil(y + bh)) + dilate)
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    return mask


def _dilate(mask: np.ndarray) -> np.ndarray:
    """3×3（8 连通）布尔膨胀，纯 numpy 实现。"""
    padded = np.pad(mask, 1)
    h, w = mask.shape
    out = np.zeros_like(mask)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy:dy + h, dx:dx + w]
    return out


def _pava(y, w):
    """加权保序回归（PAVA，非递减）。返回与 y 等长的拟合值列表。"""
    levels, weights, counts = [], [], []
    for yi, wi in zip(y, w):
        levels.append(float(yi))
        weights.append(float(wi))
        counts.append(1)
        while len(levels) >= 2 and levels[-2] >= levels[-1]:
            l2 = levels.pop()
            w2 = weights.pop()
            c2 = counts.pop()
            w1 = weights[-1]
            levels[-1] = (levels[-1] * w1 + l2 * w2) / (w1 + w2)
            weights[-1] = w1 + w2
            counts[-1] += c2
    out = np.empty(len(y), dtype=float)
    pos = 0
    for lv, c in zip(levels, counts):
        out[pos:pos + c] = lv
        pos += c
    return out


def _fit_channel_lut(b_vals, a_vals):
    """由样本对 (B 帧值 → A 帧值) 拟合单通道单调查找表。

    返回 (lut[256] uint8, xs, fit, edges)；有效分箱不足时返回 None。
    """
    lo, hi = float(b_vals.min()), float(b_vals.max())
    if hi - lo < 1.0:
        return None
    edges = np.linspace(lo, hi, N_BINS + 1)
    idx = np.clip(np.digitize(b_vals, edges) - 1, 0, N_BINS - 1)
    xs, ys, ws = [], [], []
    for k in range(N_BINS):
        m = idx == k
        cnt = int(m.sum())
        if cnt < MIN_BIN_COUNT:
            continue
        xs.append(float(np.median(b_vals[m])))
        ys.append(float(np.median(a_vals[m])))
        ws.append(cnt)
    if len(xs) < MIN_CURVE_BINS:
        return None
    fit = _pava(ys, ws)
    xs_arr = np.asarray(xs, dtype=float)
    grid = np.arange(256, dtype=float)
    lut = np.interp(grid, xs_arr, fit, left=fit[0], right=fit[-1])
    lut = np.clip(np.rint(lut), 0, 255).astype(np.uint8)
    return lut, xs_arr, np.asarray(fit, dtype=float), edges


# ---------------------------------------------------------------- 主流程

def calibrate_pair(img_a_bytes: bytes, img_b_bytes: bytes, boxes_a, boxes_b) -> dict:
    """估计并应用逐通道单调亮度映射。

    img_a_bytes / img_b_bytes: 参考帧 / 待校正帧的图像字节（同尺寸，≤512×512）。
    boxes_a / boxes_b: 两帧各自的检测框 [[x, y, w, h], ...]。

    返回字典：
      ok: 是否可靠对齐；
      reasons: 失败原因列表（ok=False 时非空）；
      diagnostics: 样本、跨度、一致性等诊断数据；
      corrected_png: ok=True 时的校正后 PNG 字节，否则为 None。
    输入尺寸等问题抛出 CalibrationInputError。
    """
    a = _decode(img_a_bytes, "参考帧")
    b = _decode(img_b_bytes, "待校正帧")
    if a.shape != b.shape:
        raise CalibrationInputError(
            f"两帧尺寸不一致（{a.shape[1]}×{a.shape[0]} 与 "
            f"{b.shape[1]}×{b.shape[0]}），无法校正。"
        )
    h, w = a.shape[:2]
    if h > MAX_DIM or w > MAX_DIM:
        raise CalibrationInputError(
            f"帧尺寸 {w}×{h} 超过 {MAX_DIM}×{MAX_DIM} 上限，无法校正。"
        )

    af = a.astype(np.float64)
    bf = b.astype(np.float64)

    # ---- 掩膜：检测框外背景、非饱和、非变化区域 --------------------------
    animal_mask = _rasterize_boxes((h, w), list(boxes_a) + list(boxes_b), BOX_DILATE)
    sat_mask = (af <= SAT_LO).any(axis=2) | (af >= SAT_HI).any(axis=2) \
        | (bf <= SAT_LO).any(axis=2) | (bf >= SAT_HI).any(axis=2)
    candidate = ~animal_mask & ~sat_mask

    # 变化区域：先按候选背景估计每通道鲁棒线性变换（分位数匹配），
    # 残差过大者判为变化像素；在净化后的集合上重估一次。
    changed = np.zeros((h, w), dtype=bool)
    for _ in range(2):
        base = candidate & ~changed
        if int(base.sum()) < 50:
            break
        changed_new = np.zeros((h, w), dtype=bool)
        for c in range(3):
            av = af[:, :, c][base]
            bv = bf[:, :, c][base]
            a_lo, a_hi = np.percentile(av, [25, 75])
            b_lo, b_hi = np.percentile(bv, [25, 75])
            if a_hi - a_lo < 4.0:  # 通道近乎平坦，退化为平移估计
                gain, offset = 1.0, float(np.median(bv) - np.median(av))
            else:
                gain = (b_hi - b_lo) / (a_hi - a_lo)
                offset = b_lo - gain * a_lo
            resid = np.abs(bf[:, :, c] - (gain * af[:, :, c] + offset))
            changed_new |= resid > CHANGED_TOL
        if int(changed_new.sum()) == int(changed.sum()):
            changed = changed_new
            break
        changed = changed_new

    # 变化区域外扩：仅膨胀候选背景内的变化像素（饱和像素本就被排除，
    # 不参与膨胀，否则散布的饱和点会把光晕扩满全图），挡住边缘半变化像素
    halo = changed & candidate
    for _ in range(CHANGED_DILATE):
        halo = _dilate(halo)

    sample_mask = candidate & ~halo
    n_samples = int(sample_mask.sum())

    diag = {
        "image_size": [w, h],
        "n_pixels": int(h * w),
        "n_animal_pixels_excluded": int((animal_mask & ~sat_mask).sum()),
        "n_saturated_excluded": int((sat_mask & ~animal_mask).sum()),
        "n_changed_excluded": int((halo & candidate).sum()),
        "n_samples": n_samples,
    }

    reasons = []

    # ---- 门槛 1：样本数量 -------------------------------------------------
    if n_samples < MIN_SAMPLES:
        reasons.append(
            f"有效标定样本不足（{n_samples} < {MIN_SAMPLES}）："
            "背景被检测框、饱和像素或变化区域占据过多。"
        )

    # ---- 门槛 2：空间分散度（4×4 网格） -----------------------------------
    ys, xs = np.nonzero(sample_mask)
    cell_id = (ys * GRID // h) * GRID + (xs * GRID // w)
    cell_counts = np.bincount(cell_id, minlength=GRID * GRID)
    covered = int((cell_counts >= MIN_PER_CELL).sum())
    diag["cells_covered"] = covered
    diag["cells_total"] = GRID * GRID
    if covered < MIN_COVERED_CELLS:
        reasons.append(
            f"标定样本空间分布过于集中（仅覆盖 {covered}/{GRID * GRID} 个区域），"
            "无法保证映射全局一致。"
        )

    # ---- 门槛 3：明暗跨度 --------------------------------------------------
    if n_samples > 0:
        sb = bf[ys, xs, :]  # 样本的 B 帧值（映射输入域）
        sa = af[ys, xs, :]
        luma = sb @ np.array([0.299, 0.587, 0.114])
        span_luma = float(np.percentile(luma, 95) - np.percentile(luma, 5))
        span_ch = [float(np.percentile(sb[:, c], 95) - np.percentile(sb[:, c], 5))
                   for c in range(3)]
    else:
        sb = sa = np.zeros((0, 3))
        span_luma = 0.0
        span_ch = [0.0, 0.0, 0.0]
    diag["span_luma"] = round(span_luma, 1)
    diag["span_channels"] = [round(s, 1) for s in span_ch]
    if span_luma < SPAN_LUMA_MIN:
        reasons.append(
            f"样本明暗跨度不足（亮度跨度 {span_luma:.0f} < {SPAN_LUMA_MIN:.0f}），"
            "无法可靠估计亮度映射。"
        )
    for c, name in enumerate(_CHANNEL_NAMES):
        if span_ch[c] < SPAN_CH_MIN:
            reasons.append(
                f"颜色通道 {name} 的样本跨度不足"
                f"（{span_ch[c]:.0f} < {SPAN_CH_MIN:.0f}），该通道映射不可靠。"
            )

    if reasons:
        return {"ok": False, "reasons": reasons, "diagnostics": diag,
                "corrected_png": None}

    # ---- 每通道单调映射（分箱中位数 + 保序回归 → 256 级 LUT） --------------
    luts, curves, edges = [], [], []
    for c in range(3):
        fitted = _fit_channel_lut(sb[:, c], sa[:, c])
        if fitted is None:
            reasons.append(
                f"颜色通道 {_CHANNEL_NAMES[c]} 的样本在亮度轴上分布过于稀疏，"
                "无法构造单调映射。"
            )
        else:
            luts.append(fitted[0])
            curves.append((fitted[1], fitted[2]))
            edges.append(fitted[3])
    if reasons:
        return {"ok": False, "reasons": reasons, "diagnostics": diag,
                "corrected_png": None}

    # ---- 门槛 4：区域一致性（2×2 子区域映射不得相互矛盾） -------------------
    region_id = (ys * REGION_GRID // h) * REGION_GRID + (xs * REGION_GRID // w)
    diffs = []
    checked_regions = 0
    for r in range(REGION_GRID * REGION_GRID):
        rm = region_id == r
        if int(rm.sum()) < MIN_REGION_SAMPLES:
            continue
        checked_regions += 1
        for c in range(3):
            rb = sb[rm, c]
            ra = sa[rm, c]
            idx = np.clip(np.digitize(rb, edges[c]) - 1, 0, N_BINS - 1)
            gx, gf = curves[c]
            for k in range(N_BINS):
                bm = idx == k
                if int(bm.sum()) < MIN_REGION_BIN_COUNT:
                    continue
                med_b = float(np.median(rb[bm]))
                med_a = float(np.median(ra[bm]))
                global_at = float(np.interp(med_b, gx, gf))
                diffs.append(abs(med_a - global_at))
    max_diff = max(diffs) if diffs else 0.0
    diag["regions_checked"] = checked_regions
    diag["region_max_deviation"] = round(max_diff, 1)
    if checked_regions < 2 or len(diffs) < MIN_CONSISTENCY_POINTS:
        reasons.append(
            f"各区域样本量不足，无法校验映射的区域一致性"
            f"（需至少 2 个区域各有 {MIN_REGION_SAMPLES} 个样本），保守起见不输出校正图。"
        )
    elif max_diff > CONFLICT_TOL:
        reasons.append(
            f"各区域的亮度映射相互矛盾（区域间最大偏差 {max_diff:.0f} 灰级 "
            f"> {CONFLICT_TOL:.0f}），可能存在局部光照变化或大面积未标记运动。"
        )

    # ---- 门槛 5：动物像素不得超出可信标定范围 -------------------------------
    trusted = []
    for c in range(3):
        lo_q, hi_q = np.percentile(sb[:, c], [TRUST_LO_Q, TRUST_HI_Q])
        trusted.append((float(lo_q) - TRUST_MARGIN, float(hi_q) + TRUST_MARGIN))
    diag["trusted_range"] = [[round(lo, 1), round(hi, 1)] for lo, hi in trusted]

    boxes_b_list = list(boxes_b)
    animal_out_frac = 0.0
    animal_out_count = 0
    animal_total = 0
    if boxes_b_list:
        b_animal = _rasterize_boxes((h, w), boxes_b_list, 0)
        animal_total = int(b_animal.sum())
        if animal_total > 0:
            px = bf[b_animal, :]  # B 帧动物像素（校正对象）
            out = np.zeros(animal_total, dtype=bool)
            for c in range(3):
                lo, hi = trusted[c]
                out |= (px[:, c] < lo) | (px[:, c] > hi)
            animal_out_count = int(out.sum())
            animal_out_frac = animal_out_count / animal_total
    diag["animal_pixels"] = animal_total
    diag["animal_out_count"] = animal_out_count
    diag["animal_out_frac"] = round(animal_out_frac, 4)
    if animal_total > 0 and animal_out_frac > ANIMAL_OUT_MAX_FRAC \
            and animal_out_count > ANIMAL_OUT_MIN_COUNT:
        reasons.append(
            f"动物像素有 {animal_out_frac * 100:.1f}%（{animal_out_count} 个）"
            "超出可信标定范围，校正需依赖外推，结果不可信。"
        )

    if reasons:
        return {"ok": False, "reasons": reasons, "diagnostics": diag,
                "corrected_png": None}

    # ---- 应用映射，输出校正图 ----------------------------------------------
    corrected = np.empty_like(b)
    for c in range(3):
        corrected[:, :, c] = luts[c][b[:, :, c]]
    buf = io.BytesIO()
    Image.fromarray(corrected, "RGB").save(buf, format="PNG")

    diag["n_samples_used"] = n_samples
    return {"ok": True, "reasons": [], "diagnostics": diag,
            "corrected_png": buf.getvalue()}
