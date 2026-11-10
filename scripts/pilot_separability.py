#!/usr/bin/env python3
"""儿童/成人可分性 pilot：零标注探路实验。

目的：在投入 700 张标注之前，先用"冻结特征 + 线性分类器"回答一个硬问题——
**纯成人和儿童的人体裁剪，在特征空间里到底分不分得开？**

做法：
  1. 从正样本(已发布片段)和负样本(被拒片段 rejected/)各取若干 clip；
  2. 每个 clip 抽若干帧，用 YOLOv11n 检人、按框裁剪并 letterbox 到 128x128；
  3. 用 MobileNetV2 (ONNX, cv2.dnn) 提取池化特征；
  4. 按 clip 分组做 5 折，训练 L2 逻辑回归，报告 AUC / 准确率 / 阈值扫描。

判定口径（写死，避免事后找理由）：
  AUC >= 0.85  可以投入标注与训练
  0.70 ~ 0.85  边缘，需要补硬负样本（坐姿成人、背影儿童）再试
  < 0.70       信号不足，停止该路线，改人工审核队列

依赖：只用 cv2 + numpy（符合仓库零依赖约束），模型首次运行自动下载。

用法：
  .venv/bin/python3 scripts/pilot_separability.py \
      --positive-dir /mnt/family-moments \
      --negative-dir /mnt/var/board_ingest/rejected \
      --limit 40 --out /tmp/pilot.npz
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from nas_video_summarizer.person_filter import PersonFilter  # noqa: E402

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/classification/"
    "mobilenet/model/mobilenetv2-12.onnx"
)
FEATURE_SIZE = 128


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


def extract_frames(clip: Path, count: int = 3) -> list[np.ndarray]:
    """用 ffmpeg 抽 count 帧；裸流 (.h264/.hevc) 需要显式 -f。"""
    suffix = clip.suffix.lower()
    fmt = []
    if suffix in {".h264"}:
        fmt = ["-f", "h264"]
    elif suffix in {".hevc", ".h265"}:
        fmt = ["-f", "hevc"]
    frames: list[np.ndarray] = []
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "f_%02d.jpg"
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", *fmt,
            "-i", str(clip), "-vf", "fps=1,scale=1280:-2",
            "-frames:v", str(count), str(out),
        ]
        try:
            subprocess.run(cmd, check=False, timeout=120)
        except Exception:
            return frames
        for path in sorted(Path(tmp).glob("f_*.jpg")):
            img = cv2.imread(str(path))
            if img is not None:
                frames.append(img)
    return frames


def collect_crops(samples: list[tuple[Path, int]], detector: PersonFilter,
                  per_clip: int = 2) -> tuple[list[np.ndarray], list[int], list[str]]:
    crops: list[np.ndarray] = []
    labels: list[int] = []
    clips: list[str] = []
    for clip, label in samples:
        frames = extract_frames(clip, count=4)
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
            if not boxes:
                continue
            # 取面积最大的框
            box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, bw, bh = box
            if bh < 0.10:          # 太小的框没有信息量
                continue
            pad = 0.10
            x1 = max(0, int((x - bw * pad) * w)); x2 = min(w, int((x + bw * (1 + pad)) * w))
            y1 = max(0, int((y - bh * pad) * h)); y2 = min(h, int((y + bh * (1 + pad)) * h))
            if x2 - x1 < 24 or y2 - y1 < 48:
                continue
            crops.append(letterbox(frame[y1:y2, x1:x2]))
            labels.append(label)
            clips.append(clip.name)
            taken += 1
    return crops, labels, clips


def mobilenet_features(net, crops: list[np.ndarray]) -> np.ndarray:
    feats = []
    for crop in crops:
        blob = cv2.dnn.blobFromImage(
            crop, 1.0 / 255.0, (FEATURE_SIZE, FEATURE_SIZE), (0.5, 0.5, 0.5),
            swapRB=True, crop=False,
        )
        net.setInput(blob)
        out = net.forward()
        feats.append(out.reshape(-1).copy())
    return np.asarray(feats, dtype=np.float32)


def logistic_regression(x: np.ndarray, y: np.ndarray, epochs: int = 400,
                        lr: float = 0.5, l2: float = 1e-3) -> np.ndarray:
    n, d = x.shape
    w = np.zeros(d, np.float32)
    b = np.float32(0.0)
    for _ in range(epochs):
        z = x @ w + b
        p = 1.0 / (1.0 + np.exp(-z))
        grad_w = x.T @ (p - y) / n + l2 * w
        grad_b = float(np.mean(p - y))
        w -= lr * grad_w
        b -= lr * grad_b
    return np.concatenate([w, [b]])


def predict(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    w, b = params[:-1], params[-1]
    return 1.0 / (1.0 + np.exp(-(x @ w + b)))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U / (n_pos * n_neg)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive-dir", required=True, help="已发布片段目录 (含 .mp4)")
    ap.add_argument("--negative-dir", required=True, help="被拒片段目录 (rejected/)")
    ap.add_argument("--limit", type=int, default=40, help="每类最多多少 clip")
    ap.add_argument("--model-dir", default="/tmp/pilot_models")
    ap.add_argument("--out", default="/tmp/pilot.npz")
    ap.add_argument("--min-crops", type=int, default=20)
    args = ap.parse_args()

    pos_dir = Path(args.positive_dir)
    neg_dir = Path(args.negative_dir)
    positives = sorted(pos_dir.rglob("*.mp4"))[: args.limit]
    negatives: list[Path] = []
    for entry in sorted(neg_dir.iterdir()):
        if len(negatives) >= args.limit:
            break
        if entry.is_dir():
            for pattern in ("clip.h264", "clip.hevc", "clip.mp4"):
                candidate = entry / pattern
                if candidate.exists():
                    negatives.append(candidate)
                    break
        elif entry.suffix.lower() in {".mp4", ".h264", ".hevc", ".h265"}:
            negatives.append(entry)
    if not positives or not negatives:
        print(f"[pilot] 样本不足: 正 {len(positives)} 负 {len(negatives)}", flush=True)
        return 2
    print(f"[pilot] 正样本 {len(positives)} clip, 负样本 {len(negatives)} clip", flush=True)

    model_path = ensure_model(Path(args.model_dir) / "mobilenetv2-12.onnx")
    net = cv2.dnn.readNetFromONNX(str(model_path))
    detector = PersonFilter(threshold=0.35, model_dir=Path(args.model_dir))

    samples = [(p, 1) for p in positives] + [(n, 0) for n in negatives]
    crops, labels, clips = collect_crops(samples, detector)
    print(f"[pilot] 有效裁剪 {len(crops)} 张 (正 {sum(labels)} / 负 {len(labels)-sum(labels)})", flush=True)
    if len(crops) < args.min_crops:
        print("[pilot] 裁剪太少, 无法评估", flush=True)
        return 2

    features = mobilenet_features(net, crops)
    np.savez(args.out, x=features, y=np.array(labels), clips=np.array(clips))
    print(f"[pilot] 特征已保存 -> {args.out}", flush=True)

    # 按 clip 分组 5 折
    y_all = np.asarray(labels, dtype=np.int64)
    clips_arr = np.asarray(clips)
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
        params = logistic_regression(features[train], y_all[train].astype(np.float32))
        all_scores[mask] = predict(features[mask], params)

    auc = auc_score(all_scores, y_all)
    print(f"\n[pilot] 按 clip 分组的交叉验证 AUC = {auc:.3f}", flush=True)
    for thresh in (0.3, 0.4, 0.5, 0.6, 0.7):
        pred = (all_scores >= thresh).astype(int)
        tp = int(((pred == 1) & (y_all == 1)).sum())
        fp = int(((pred == 1) & (y_all == 0)).sum())
        fn = int(((pred == 0) & (y_all == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        print(f"  阈值 {thresh:.1f}: precision={prec:.2f} recall={rec:.2f}", flush=True)

    if auc >= 0.85:
        verdict = "可分性足够 -> 可以投入 700 张标注并训练"
    elif auc >= 0.70:
        verdict = "边缘 -> 需要补硬负样本(坐姿成人/背影儿童)后重试"
    else:
        verdict = "信号不足 -> 停止分类器路线, 改人工审核队列"
    print(f"\n[pilot] 结论: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
