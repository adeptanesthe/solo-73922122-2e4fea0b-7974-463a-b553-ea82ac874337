"""相邻帧色差校正 API 测试：有效/歧义场景、token 失效、尺寸与帧号校验、
以及校正为只读操作（原图、检测框、版本号均不变）。"""

import io
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import calib_scenes as cs
from app.main import app

client = TestClient(app)


def make_session(pairs_frames, dets=None, pair=None):
    """pairs_frames: [png_bytes, ...]；dets 缺省时由 pair 的检测框构造。"""
    if dets is None:
        if pair is None:
            pair = cs.valid_pair()
        dets = []
        for f, boxes in enumerate((pair["boxes_a"], pair["boxes_b"])):
            for k, rect in enumerate(boxes):
                dets.append({"id": f"f{f}b{k}", "frame": f, "rect": rect, "vector": [1, 0]})
    files = [("images", (f"f{i}.png", data, "image/png")) for i, data in enumerate(pairs_frames)]
    files.append(("detections", ("d.json", json.dumps({"detections": dets}), "application/json")))
    r = client.post("/api/sessions", files=files)
    assert r.status_code == 200, r.text
    return r.json()


def calibrate(sid, frame):
    return client.post(f"/api/sessions/{sid}/calibrate", json={"frame": frame})


# ---------------------------------------------------------------- 有效流程

def test_calibrate_ok_and_token_rotation():
    pair = cs.valid_pair()
    p = make_session([pair["a"], pair["b"]], pair=pair)
    sid = p["session_id"]

    r = calibrate(sid, 0)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["ok"] and body["pair"] == [0, 1]
    assert body["diagnostics"]["n_samples"] >= 300
    url1 = body["corrected_url"]

    img = client.get(url1)
    assert img.status_code == 200 and img.headers["content-type"] == "image/png"
    corrected = Image.open(io.BytesIO(img.content))
    assert corrected.size == (cs.W, cs.H)

    # 再次校正：旧 token 立即失效（迟到结果不能回填）
    url2 = calibrate(sid, 0).json()["corrected_url"]
    assert url2 != url1
    assert client.get(url1).status_code == 404
    assert client.get(url2).status_code == 200


def test_calibrate_is_read_only():
    """校正不得改动原图、检测框、判断与输入版本号。"""
    pair = cs.valid_pair()
    p = make_session([pair["a"], pair["b"]], pair=pair)
    sid = p["session_id"]
    before_frames = [client.get(f"/api/sessions/{sid}/frames/{i}").content for i in range(2)]
    before_dets = client.get(f"/api/sessions/{sid}/detections.json").json()
    before_version = client.get(f"/api/sessions/{sid}").json()["version"]

    assert calibrate(sid, 0).json()["ok"]

    after_frames = [client.get(f"/api/sessions/{sid}/frames/{i}").content for i in range(2)]
    assert before_frames == after_frames  # 原图字节不变
    assert client.get(f"/api/sessions/{sid}/detections.json").json() == before_dets
    assert client.get(f"/api/sessions/{sid}").json()["version"] == before_version


def test_failed_calibrate_revokes_previous_image():
    """新一次校正失败时，旧校正图一并撤下，不残留可误用的结果。"""
    good = cs.valid_pair()
    flat = cs.flat_pair()
    p = make_session([good["a"], good["b"], flat["a"]])
    sid = p["session_id"]

    url1 = calibrate(sid, 0).json()["corrected_url"]
    assert client.get(url1).status_code == 200

    r = calibrate(sid, 1)  # 第 2 对帧背景平坦 → 无法可靠对齐
    assert r.status_code == 200
    assert r.json()["ok"] is False
    assert "corrected_url" not in r.json()
    assert client.get(url1).status_code == 404  # 旧图已撤下


# ---------------------------------------------------------------- 歧义场景

def test_flat_background_unreliable():
    pair = cs.flat_pair()
    p = make_session([pair["a"], pair["b"]], pair=pair)
    body = calibrate(p["session_id"], 0).json()
    assert body["ok"] is False
    assert any("跨度" in reason for reason in body["reasons"])
    assert "corrected_url" not in body


def test_conflicting_regions_unreliable():
    pair = cs.conflict_pair()
    p = make_session([pair["a"], pair["b"]], pair=pair)
    body = calibrate(p["session_id"], 0).json()
    assert body["ok"] is False
    assert any("矛盾" in reason for reason in body["reasons"])


def test_animal_out_of_range_unreliable():
    pair = cs.animal_out_of_range_pair()
    p = make_session([pair["a"], pair["b"]], pair=pair)
    body = calibrate(p["session_id"], 0).json()
    assert body["ok"] is False
    assert any("动物像素" in reason for reason in body["reasons"])


# ---------------------------------------------------------------- 输入校验

def test_frame_index_validation():
    pair = cs.valid_pair()
    p = make_session([pair["a"], pair["b"], pair["a"]])
    sid = p["session_id"]
    assert calibrate(sid, -1).status_code == 400
    assert calibrate(sid, 2).status_code == 400  # 最后一帧无下一帧
    assert calibrate(sid, 99).status_code == 400
    assert calibrate(sid, 1).status_code == 200  # 中间帧可组成相邻对
    assert client.post(f"/api/sessions/{sid}/calibrate", json={"frame": "x"}).status_code == 422


def test_size_mismatch_rejected():
    small = cs.to_png(__import__("numpy").full((48, 64, 3), 120.0))
    tall = cs.to_png(__import__("numpy").full((50, 64, 3), 120.0))
    p = make_session([small, tall], dets=[])
    r = calibrate(p["session_id"], 0)
    assert r.status_code == 400 and "尺寸不一致" in r.json()["detail"]


def test_oversize_rejected():
    big = cs.to_png(__import__("numpy").full((64, 520, 3), 120.0))
    p = make_session([big, big], dets=[])
    r = calibrate(p["session_id"], 0)
    assert r.status_code == 400 and "512" in r.json()["detail"]


def test_unknown_session_404():
    assert calibrate("nope", 0).status_code == 404
    assert client.get("/api/sessions/nope/calibration/x/corrected.png").status_code == 404


# ---------------------------------------------------------------- 曝光抖动示例

def test_exposure_example_end_to_end():
    p = client.post("/api/example-exposure").json()
    assert p["n_frames"] == 6 and len(p["detections"]) == 12
    sid = p["session_id"]

    body = calibrate(sid, 0).json()
    assert body["ok"], body["reasons"]
    assert client.get(body["corrected_url"]).status_code == 200

    # 示例会话在既有求解流程下应为 2 只动物（两条轨迹）
    job = client.post(f"/api/sessions/{sid}/solve").json()
    import time
    deadline = time.time() + 10
    while time.time() < deadline:
        done = client.get(f"/api/jobs/{job['job_id']}").json()
        if done["status"] != "running":
            break
        time.sleep(0.05)
    assert done["result"]["feasible"] and done["result"]["animal_count"] == 2
