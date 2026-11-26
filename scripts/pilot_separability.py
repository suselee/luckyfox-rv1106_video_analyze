#!/usr/bin/env python3
"""儿童/成人可分性 pilot：零标注探路实验。

目的：在投入 700 张标注之前，先用"冻结特征 + 线性分类器"回答一个硬问题——
**纯成人和儿童的人体裁剪，在特征空间里到底分不分得开？**

做法：
  1. 从正样本(已发布 confirmed 片段)和负样本(被拒片段 rejected/)各取若干 clip；
  2. 每个 clip 复用 P1 `_plan_verify_targets()` 在活跃窗口(best_ts/activity_peak_ts)
     抽帧（裸流 .h264/.hevc 先 -c copy 封装为临时 MP4 以支持精确 -ss 寻址）；
  3. 用 YOLOv11n 检人、优先匹配板端 best_box 对应的人体框（避免同框时误裁大人）
     并 letterbox 到 224x224；
  4. 用 MobileNetV2 (ONNX, cv2.dnn) 提取 GlobalAveragePool 1280 维池化特征；
  5. 按 clip 分组做 5 折，训练 L2 逻辑回归，分别报告：
     - 纯外观特征 (1280d) AUC
     - 外观 + 几何特征 (w, h, aspect, y2) AUC
     - 各阈值下的 precision/recall 与负样本分 decision 统计。

判定口径（写死，避免事后找理由）：
  AUC >= 0.85  可以投入标注与训练
  0.70 ~ 0.85  边缘，需要补硬负样本（坐姿成人、背影儿童）或人工去噪再试
  < 0.70       信号不足，停止该路线，改人工审核队列

依赖：只用 cv2 + numpy（符合仓库零依赖约束），模型首次运行自动下载。

用法：
  .venv/bin/python3 scripts/pilot_separability.py \
      --positive-dir /mnt/family-moments \
      --negative-dir /mnt/var/board_ingest/rejected \
      --detector-model-dir /mnt/var/person_filter_models \
      --limit 60 --out /tmp/pilot.npz
"""
from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nas_video_summarizer.ingest import _plan_verify_targets  # noqa: E402
from nas_video_summarizer.person_filter import PersonFilter  # noqa: E402

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-12.onnx"
)
FEATURE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def ensure_model(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[pilot] downloading {MODEL_URL}", flush=True)
    from urllib import request

    with request.urlopen(MODEL_URL) as resp, path.open("wb") as fh:
        while chunk := resp.read(1 << 20):
            fh.write(chunk)
    return path


def letterbox(img: np.ndarray, size: int = FEATURE_SIZE) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((size, size, 3), np.uint8)
    r = min(size / w, size / h)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    canvas = np.full((size, size, 3), 114, np.uint8)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(img, (nw, nh))
    return canvas


def _board_box_xywh(meta: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = meta.get("best_box") or meta.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in raw)
    except (TypeError, ValueError):
        return None
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1.5:
        fw = float(meta.get("frame_width") or 0)
        fh = float(meta.get("frame_height") or 0)
        if fw <= 0 or fh <= 0:
            return None
        x1, x2 = x1 / fw, x2 / fw
        y1, y2 = y1 / fh, y2 / fh
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return (x1, y1, w, h)


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _pick_person_box(
    boxes: list[list[float]],
    anchor_box: tuple[float, float, float, float] | None,
) -> list[float] | None:
    valid = [b for b in boxes if len(b) >= 4 and float(b[3]) >= 0.10]
    if not valid:
        return None
    if anchor_box is None or len(valid) == 1:
        return max(valid, key=lambda b: float(b[2]) * float(b[3]))
    # 同框多人时优先选与板端 best_box IoU 最高 (或中心最近) 的人体框, 避免误裁大人
    acx = anchor_box[0] + anchor_box[2] / 2.0
    acy = anchor_box[1] + anchor_box[3] / 2.0

    def rank(b: list[float]) -> tuple[float, float]:
        tup = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        iou = _iou_xywh(tup, anchor_box)
        cx, cy = tup[0] + tup[2] / 2.0, tup[1] + tup[3] / 2.0
        dist = ((cx - acx) ** 2 + (cy - acy) ** 2) ** 0.5
        return (iou, -dist)

    return max(valid, key=rank)


def extract_anchored_frames(
    clip: Path,
    meta: dict[str, Any],
    count: int = 3,
) -> list[np.ndarray]:
    """在片段活跃窗口抽 count 帧; 裸流先 -c copy 重封装以支持精确 -ss 寻址。"""
    targets, err = _plan_verify_targets(
        meta, width_scale=1.5, height_scale=1.5, sample_count=count
    )
    offsets = [t.offset_seconds for t in targets] if err is None and targets else []

    frames: list[np.ndarray] = []
    suffix = clip.suffix.lower()
    with tempfile.TemporaryDirectory(prefix="pilot-frames-") as tmp:
        tmp_path = Path(tmp)
        source = clip
        if suffix in {".h264", ".hevc", ".h265"}:
            es_fmt = "hevc" if suffix in {".hevc", ".h265"} else "h264"
            muxed = tmp_path / "clip.mp4"
            res = subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", es_fmt, "-i", str(clip), "-c", "copy", "-y", str(muxed),
                ],
                check=False,
                timeout=60,
            )
            if res.returncode == 0 and muxed.exists() and muxed.stat().st_size > 0:
                source = muxed

        if not offsets:
            # 无 meta 兜底: 从片尾倒数 6s 开始抽 (兼容 context_before=20s)
            out_pat = tmp_path / "f_%02d.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-sseof", "-6", "-i", str(source),
                    "-vf", "fps=1,scale=1280:-2", "-frames:v", str(count),
                    str(out_pat),
                ],
                check=False,
                timeout=60,
            )
            for path in sorted(tmp_path.glob("f_*.jpg")):
                img = cv2.imread(str(path))
                if img is not None:
                    frames.append(img)
            return frames

        for idx, off in enumerate(offsets, start=1):
            out_jpg = tmp_path / f"f_{idx:02d}.jpg"
            subprocess.run(
                [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-ss", f"{max(0.0, off):.3f}", "-i", str(source),
                    "-frames:v", "1", "-vf", "scale=1280:-2", "-q:v", "3",
                    "-y", str(out_jpg),
                ],
                check=False,
                timeout=30,
            )
            if out_jpg.exists():
                img = cv2.imread(str(out_jpg))
                if img is not None:
                    frames.append(img)
    return frames


def collect_crops(
    samples: list[tuple[Path, int, dict[str, Any], str]],
    detector: PersonFilter,
    per_clip: int = 3,
) -> tuple[list[np.ndarray], list[list[float]], list[int], list[str], list[str]]:
    crops: list[np.ndarray] = []
    geoms: list[list[float]] = []
    labels: list[int] = []
    clips: list[str] = []
    tags: list[str] = []
    for clip, label, meta, tag in samples:
        frames = extract_anchored_frames(clip, meta, count=per_clip)
        anchor = _board_box_xywh(meta)
        taken = 0
        for frame in frames:
            if taken >= per_clip:
                break
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            info = detector.detect(base64.b64encode(buf.tobytes()).decode(), classify_age=False)
            boxes = info.get("person_boxes") or []
            box = _pick_person_box(boxes, anchor)
            if box is None:
                continue
            x, y, bw, bh = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
            pad = 0.10
            x1 = max(0, int((x - bw * pad) * w))
            x2 = min(w, int((x + bw * (1 + pad)) * w))
            y1 = max(0, int((y - bh * pad) * h))
            y2 = min(h, int((y + bh * (1 + pad)) * h))
            if x2 - x1 < 24 or y2 - y1 < 48:
                continue
            crops.append(letterbox(frame[y1:y2, x1:x2]))
            geoms.append([bw, bh, bw / max(bh, 1e-4), y + bh])
            labels.append(label)
            clips.append(str(clip))
            tags.append(tag)
            taken += 1
    return crops, geoms, labels, clips, tags


def mobilenet_features(net, crops: list[np.ndarray]) -> np.ndarray:
    layer_names = set(net.getLayerNames())
    target_layer = (
        "onnx_node!GlobalAveragePool_97"
        if "onnx_node!GlobalAveragePool_97" in layer_names
        else ""
    )
    feats = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        blob = (blob - IMAGENET_MEAN) / IMAGENET_STD
        net.setInput(blob)
        out = net.forward(target_layer) if target_layer else net.forward()
        vec = out.reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        if norm > 1e-6:
            vec = vec / norm
        feats.append(vec.copy())
    return np.asarray(feats, dtype=np.float32)


def logistic_regression(x: np.ndarray, y: np.ndarray, epochs: int = 500,
                        lr: float = 0.8, l2: float = 1e-3) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, dtype=np.float32)
    b = np.float32(0.0)
    for _ in range(epochs):
        z = np.clip(x @ w + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = x.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return np.concatenate([w, [b]])


def predict(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    w, b = params[:-1], params[-1]
    z = np.clip(x @ w + b, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _load_positive_samples(
    pos_dir: Path,
    limit: int,
    confirmed_only: bool,
) -> list[tuple[Path, int, dict[str, Any], str]]:
    """按最新倒序扫描已发布片段; 默认只取 rv1106_confirmed 保证正样本纯度。"""
    candidates = sorted(pos_dir.rglob("*.mp4"), reverse=True)
    picked: list[tuple[Path, int, dict[str, Any], str]] = []
    for mp4 in candidates:
        if len(picked) >= limit:
            break
        sidecar = mp4.with_suffix(".json")
        meta: dict[str, Any] = {}
        category = "unknown"
        if sidecar.exists():
            try:
                data = json.loads(sidecar.read_text(encoding="utf-8"))
                category = str(data.get("category") or "unknown")
                raw = data.get("model_raw") or data.get("board") or {}
                if isinstance(raw, dict):
                    meta = raw
            except Exception:
                pass
        if confirmed_only and category != "rv1106_confirmed":
            continue
        picked.append((mp4, 1, meta, category))
    return picked


def _load_negative_samples(
    neg_dir: Path,
    limit: int,
    allowed_decisions: set[str] | None,
) -> list[tuple[Path, int, dict[str, Any], str]]:
    picked: list[tuple[Path, int, dict[str, Any], str]] = []
    for entry in sorted(neg_dir.iterdir(), reverse=True):
        if len(picked) >= limit:
            break
        if entry.is_dir():
            video = None
            for pattern in ("clip.h264", "clip.hevc", "clip.mp4"):
                candidate = entry / pattern
                if candidate.exists():
                    video = candidate
                    break
            if video is None:
                continue
            decision = "unknown"
            meta: dict[str, Any] = {}
            dj = entry / "decision.json"
            if dj.exists():
                try:
                    info = json.loads(dj.read_text(encoding="utf-8"))
                    decision = str(info.get("decision") or "unknown")
                    if isinstance(info.get("meta"), dict):
                        meta = info["meta"]
                except Exception:
                    pass
            if not meta and (entry / "meta.json").exists():
                try:
                    meta = json.loads((entry / "meta.json").read_text(encoding="utf-8"))
                except Exception:
                    pass
            if allowed_decisions and decision not in allowed_decisions:
                continue
            picked.append((video, 0, meta, decision))
        elif entry.suffix.lower() in {".mp4", ".h264", ".hevc", ".h265"}:
            if allowed_decisions:
                continue
            picked.append((entry, 0, {}, "file"))
    return picked


def _eval_cv(
    x: np.ndarray,
    y_all: np.ndarray,
    clips_arr: np.ndarray,
    label: str,
    tags: list[str] | None = None,
) -> float:
    unique_clips = np.unique(clips_arr)
    rng = np.random.default_rng(0)
    rng.shuffle(unique_clips)
    folds = np.array_split(unique_clips, min(5, len(unique_clips)))
    all_scores = np.zeros(len(y_all), dtype=np.float32)
    for fold in folds:
        mask = np.isin(clips_arr, fold)
        train = ~mask
        if y_all[train].sum() == 0 or (1 - y_all[train]).sum() == 0:
            continue
        params = logistic_regression(x[train], y_all[train].astype(np.float32))
        all_scores[mask] = predict(x[mask], params)

    auc = auc_score(all_scores, y_all)
    # 同时算 clip 级均值 AUC (真实部署时是对 3-5 帧聚合判定, 比单帧更贴近实际)
    clip_scores, clip_labels = [], []
    for cid in unique_clips:
        m = clips_arr == cid
        clip_scores.append(float(np.mean(all_scores[m])))
        clip_labels.append(int(y_all[m][0]))
    clip_auc = auc_score(np.asarray(clip_scores), np.asarray(clip_labels))

    print(
        f"\n[pilot] {label}: 帧级 AUC = {auc:.3f} | clip级均值 AUC = {clip_auc:.3f}",
        flush=True,
    )
    for thresh in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (all_scores >= thresh).astype(int)
        tp = int(((pred == 1) & (y_all == 1)).sum())
        fp = int(((pred == 1) & (y_all == 0)).sum())
        fn = int(((pred == 0) & (y_all == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  阈值 {thresh:.1f}: precision={prec:.2f} recall={rec:.2f} (tp={tp} fp={fp} fn={fn})", flush=True)

    if tags is not None:
        print("  负样本按 decision 分桶的预测均分 (越高越像正样本/越可能混有女儿):", flush=True)
        by_tag: dict[str, list[float]] = {}
        for score, lbl, tag in zip(all_scores, y_all, tags):
            if lbl == 0:
                by_tag.setdefault(tag, []).append(float(score))
        for tag, vals in sorted(by_tag.items(), key=lambda kv: -len(kv[1])):
            arr = np.asarray(vals)
            print(
                f"    {tag:<26} n={len(arr):>3} mean={arr.mean():.3f} >=0.5:{int((arr>=0.5).sum()):>2}",
                flush=True,
            )
    return clip_auc


def _load_feedback_samples(
    db_path: Path,
    limit: int,
) -> tuple[
    list[tuple[Path, int, dict[str, Any], str]],
    list[tuple[Path, int, dict[str, Any], str]],
]:
    """从 Web UI 人工标注的 clip_feedback 表加载干净的正/负样本。"""
    positives: list[tuple[Path, int, dict[str, Any], str]] = []
    negatives: list[tuple[Path, int, dict[str, Any], str]] = []
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT source, label, clip_path, meta_json, category
            FROM clip_feedback
            WHERE label IN ('daughter', 'adult_only') AND clip_path != ''
            ORDER BY updated_at DESC, id DESC
            """
        ).fetchall()
    for row in rows:
        clip = Path(str(row["clip_path"]))
        if not clip.is_file():
            continue
        meta: dict[str, Any] = {}
        try:
            loaded = json.loads(str(row["meta_json"] or "{}"))
            if isinstance(loaded, dict):
                meta = loaded
        except json.JSONDecodeError:
            pass
        tag = f"{row['source']}:{row['category'] or 'unknown'}"
        if row["label"] == "daughter":
            if len(positives) < limit:
                positives.append((clip, 1, meta, tag))
        elif row["label"] == "adult_only":
            if len(negatives) < limit:
                negatives.append((clip, 0, meta, tag))
        if len(positives) >= limit and len(negatives) >= limit:
            break
    return positives, negatives


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feedback-db", default=None, help="直接读取 SQLite clip_feedback 人工标注表")
    ap.add_argument("--positive-dir", default=None, help="已发布片段目录 (含 .mp4)")
    ap.add_argument("--negative-dir", default=None, help="被拒片段目录 (rejected/)")
    ap.add_argument("--limit", type=int, default=60, help="每类最多多少 clip")
    ap.add_argument("--per-clip", type=int, default=3, help="每条 clip 抽帧数")
    ap.add_argument("--model-dir", default="/tmp/pilot_models", help="MobileNetV2 缓存目录")
    ap.add_argument("--detector-model-dir", default=None, help="YOLOv11n 缓存目录 (默认同 --model-dir)")
    ap.add_argument("--all-positives", action="store_true", help="包含 probable 正样本 (默认只取 confirmed)")
    ap.add_argument("--negative-decisions", default="", help="只取指定 decision 的负样本 (逗号分隔)")
    ap.add_argument("--out", default="/tmp/pilot.npz")
    ap.add_argument("--min-crops", type=int, default=20)
    args = ap.parse_args()

    if args.feedback_db:
        positives, negatives = _load_feedback_samples(Path(args.feedback_db), args.limit)
    else:
        if not args.positive_dir or not args.negative_dir:
            ap.error("需要指定 --feedback-db，或者同时指定 --positive-dir 与 --negative-dir")
        pos_dir = Path(args.positive_dir)
        neg_dir = Path(args.negative_dir)
        allowed_neg = {d.strip() for d in args.negative_decisions.split(",") if d.strip()} or None

        positives = _load_positive_samples(pos_dir, args.limit, confirmed_only=not args.all_positives)
        negatives = _load_negative_samples(neg_dir, args.limit, allowed_decisions=allowed_neg)
    if not positives or not negatives:
        print(f"[pilot] 样本不足: 正 {len(positives)} 负 {len(negatives)}", flush=True)
        return 2
    print(
        f"[pilot] 正样本 {len(positives)} clip ({dict(Counter(x[3] for x in positives))}), "
        f"负样本 {len(negatives)} clip ({dict(Counter(x[3] for x in negatives))})",
        flush=True,
    )

    model_path = ensure_model(Path(args.model_dir) / "mobilenetv2-12.onnx")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    det_dir = Path(args.detector_model_dir) if args.detector_model_dir else Path(args.model_dir)
    detector = PersonFilter(threshold=0.35, model_dir=det_dir)

    samples = positives + negatives
    crops, geoms, labels, clips, tags = collect_crops(samples, detector, per_clip=args.per_clip)
    pos_crops = sum(labels)
    neg_crops = len(labels) - pos_crops
    print(f"[pilot] 有效裁剪 {len(crops)} 张 (正 {pos_crops} / 负 {neg_crops})", flush=True)
    if len(crops) < args.min_crops or pos_crops < 5 or neg_crops < 5:
        print("[pilot] 裁剪太少, 无法评估", flush=True)
        return 2

    features = mobilenet_features(net, crops)
    geom_arr = np.asarray(geoms, dtype=np.float32)
    # 标准化几何特征后与外观向量拼接
    geom_std = (geom_arr - geom_arr.mean(axis=0)) / np.maximum(geom_arr.std(axis=0), 1e-4)
    combined = np.hstack([features, geom_std * 0.25])

    np.savez(
        args.out,
        x=features,
        geom=geom_arr,
        y=np.array(labels),
        clips=np.array(clips),
        tags=np.array(tags),
    )
    print(f"[pilot] 特征已保存 -> {args.out} (dim={features.shape[1]})", flush=True)

    y_all = np.asarray(labels, dtype=np.int64)
    clips_arr = np.asarray(clips)

    app_auc = _eval_cv(features, y_all, clips_arr, "纯外观 MobileNetV2-1280d", tags=tags)
    comb_auc = _eval_cv(combined, y_all, clips_arr, "外观 + 几何 (w,h,aspect,y2)")

    best_auc = max(app_auc, comb_auc)
    if best_auc >= 0.85:
        verdict = "可分性足够 -> 可以投入标注与轻量化身体分类器训练"
    elif best_auc >= 0.70:
        verdict = "边缘 -> 建议先肉眼核对高分负样本是否混入真女儿, 去噪后复测"
    else:
        verdict = "信号不足 -> 停止分类器路线, 改人工审核队列"
    print(f"\n[pilot] 结论 (clip级最优 AUC={best_auc:.3f}): {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
