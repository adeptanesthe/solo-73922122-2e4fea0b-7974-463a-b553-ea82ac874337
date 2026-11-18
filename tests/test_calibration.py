"""色差校正核心算法单元测试：有效场景与各类歧义场景。"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_scenes as cs
from app.calibration import (
    CalibrationInputError,
    _fit_channel_lut,
    _pava,
    calibrate_pair,
)
from app.example import (
    build_example_detections,
    build_example_frames,
    build_exposure_example_detections,
    build_exposure_example_frames,
)


def run(pair):
    return calibrate_pair(pair["a"], pair["b"], pair["boxes_a"], pair["boxes_b"])


# ---------------------------------------------------------------- 保序回归与 LUT

def test_pava_pools_violators():
    fit = _pava([3.0, 1.0, 2.0, 5.0], [1, 1, 1, 1])
    assert list(fit[:3]) == [2.0, 2.0, 2.0]
    assert fit[3] == 5.0


def test_lut_is_monotonic_nondecreasing():
    rng = np.random.default_rng(0)
    b = rng.uniform(20, 230, 5000)
    a = 0.8 * b + 10 + rng.normal(0, 4, 5000)
    lut, xs, fit, _ = _fit_channel_lut(b, a)
    assert np.all(np.diff(lut.astype(int)) >= 0)
    # 线性真值下 LUT 应接近恒等反变换
    mid = lut[int(np.median(b))]
    assert abs(mid - np.median(a)) < 8


# ---------------------------------------------------------------- 有效场景

def test_valid_pair_ok_and_corrected_close_to_reference():
    pair = cs.valid_pair()
    r = run(pair)
    assert r["ok"], r["reasons"]
    assert r["corrected_png"]
    d = r["diagnostics"]
    assert d["n_samples"] >= 300
    assert d["cells_covered"] >= 10
    assert d["span_luma"] >= 48
    assert d["region_max_deviation"] <= 20
    assert d["animal_out_frac"] <= 0.02

    # 校正后的 B 帧背景应与参考帧 A 的背景基本一致（曝光差被消除）
    corrected = cs.from_png(r["corrected_png"])
    ref = pair["a_arr"]
    h, w = ref.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    bg = np.ones((h, w), bool)
    for (x, y, bw, bh) in pair["boxes_a"] + pair["boxes_b"]:
        x0, y0 = max(0, int(x) - 2), max(0, int(y) - 2)
        bg[y0:int(y + bh) + 3, x0:int(x + bw) + 3] = False
    mae = float(np.abs(corrected[bg] - ref[bg]).mean())
    assert mae < 5.0, f"校正后背景与参考帧平均色差 {mae:.2f} 灰级，过大"


def test_saturated_pixels_excluded_from_estimation():
    """30% 背景像素在 B 帧饱和：不得参与估计，其余样本仍应支撑可靠校正。"""
    pair = cs.valid_pair()
    b_arr = cs.from_png(pair["b"])
    rng = np.random.default_rng(3)
    mask = rng.random(b_arr.shape[:2]) < 0.30
    # 只漂白背景：检测框（外扩 2 像素）内的像素保持原样
    for (x, y, bw, bh) in pair["boxes_b"]:
        x0, y0 = max(0, int(x) - 2), max(0, int(y) - 2)
        mask[y0:int(y + bh) + 3, x0:int(x + bw) + 3] = False
    b_arr[mask] = 255.0
    pair["b"] = cs.to_png(b_arr)
    r = run(pair)
    assert r["ok"], r["reasons"]
    d = r["diagnostics"]
    assert d["n_saturated_excluded"] > 1000
    # 有效样本数应明显少于无饱和时（饱和像素确实被剔除）
    r_clean = run(cs.valid_pair())
    assert d["n_samples"] < r_clean["diagnostics"]["n_samples"] - 1000


# ---------------------------------------------------------------- 歧义场景

def test_conflicting_regions_rejected():
    r = run(cs.conflict_pair())
    assert not r["ok"]
    assert any("矛盾" in reason for reason in r["reasons"]), r["reasons"]
    assert r["corrected_png"] is None


def test_insufficient_span_rejected():
    r = run(cs.flat_pair())
    assert not r["ok"]
    assert any("跨度" in reason for reason in r["reasons"]), r["reasons"]


def test_mostly_saturated_rejected():
    r = run(cs.saturated_pair())
    assert not r["ok"]
    assert any("样本不足" in reason for reason in r["reasons"]), r["reasons"]


def test_large_changed_region_rejected():
    r = run(cs.changed_pair())
    assert not r["ok"]
    assert any("集中" in reason for reason in r["reasons"]), r["reasons"]


def test_animal_pixels_out_of_range_rejected():
    r = run(cs.animal_out_of_range_pair())
    assert not r["ok"]
    assert any("动物像素" in reason for reason in r["reasons"]), r["reasons"]


# ---------------------------------------------------------------- 输入校验

def test_size_mismatch_raises():
    pair = cs.valid_pair()
    other = cs.to_png(np.full((72, 80, 3), 128.0))
    with pytest.raises(CalibrationInputError, match="尺寸不一致"):
        calibrate_pair(pair["a"], other, [], [])


def test_oversize_raises():
    big = cs.to_png(np.full((100, 513, 3), 128.0))
    with pytest.raises(CalibrationInputError, match="512"):
        calibrate_pair(big, big, [], [])


def test_undecodable_raises():
    with pytest.raises(CalibrationInputError):
        calibrate_pair(b"not an image", b"not an image", [], [])


# ---------------------------------------------------------------- 内置示例

def test_exposure_example_all_pairs_calibrate_ok():
    """曝光抖动示例：全部 5 对相邻帧都应可靠对齐。"""
    frames = build_exposure_example_frames()
    dets = build_exposure_example_detections()
    assert len(frames) == 6
    for f in range(5):
        boxes_a = [d["rect"] for d in dets if d["frame"] == f]
        boxes_b = [d["rect"] for d in dets if d["frame"] == f + 1]
        r = calibrate_pair(frames[f][1], frames[f + 1][1], boxes_a, boxes_b)
        assert r["ok"], f"帧对 ({f}, {f + 1}) 失败：{r['reasons']}"
        assert r["diagnostics"]["animal_out_frac"] <= 0.02


def test_tracking_example_rejected_for_flat_background():
    """交叉遮挡示例背景近乎单色：应因明暗跨度不足被拒（不显示校正图）。"""
    frames = build_example_frames()
    dets = build_example_detections()
    boxes_a = [d["rect"] for d in dets if d["frame"] == 0]
    boxes_b = [d["rect"] for d in dets if d["frame"] == 1]
    r = calibrate_pair(frames[0][1], frames[1][1], boxes_a, boxes_b)
    assert not r["ok"]
    assert any("跨度" in reason for reason in r["reasons"])
