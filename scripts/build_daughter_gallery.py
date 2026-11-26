#!/usr/bin/env python3
"""构建或刷新 NAS 侧 4K ArcFace (w600k_r50.onnx) 女儿人脸底库 daughter_r50.npz。

典型用法 (在 NAS jail 内运行):
  # 首次构建: 用 RV1106 板端 daughter.db + w600k_mbf.onnx 锚定选脸,
  # 避免把同框的妈妈/奶奶误收入底库:
  python3 scripts/build_daughter_gallery.py \\
      --anchor-board-db /tmp/pilot_models/daughter.db \\
      --anchor-mbf /tmp/pilot_models/w600k_mbf.onnx

  # 后续随着女儿长大定期刷新 (默认自动以现有 daughter_r50.npz 作为锚点):
  python3 scripts/build_daughter_gallery.py

原子写盘 (.tmp.npz -> os.replace), 在线服务检测到 mtime 变化后自动热重载,
无需重启 nas_video。
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nas_video_summarizer.config import load_settings  # noqa: E402
from nas_video_summarizer.database import Database  # noqa: E402
from nas_video_summarizer.face_verifier import DaughterFaceVerifier  # noqa: E402
from nas_video_summarizer.ingest import _plan_verify_targets  # noqa: E402


@dataclass
class CandidateFace:
    clip_id: str
    source: str
    embedding: Any
    anchor_sim: float
    face_px: int
    face_score: float
    sharpness: float


def load_board_fdb1(path: Path, np: Any) -> Any:
    raw = path.read_bytes()
    if len(raw) < 12:
        raise ValueError(f"invalid daughter.db (too short): {path}")
    magic, dim, count = struct.unpack("<4sii", raw[:12])
    if magic != b"FDB1" or count == 0 or dim == 0:
        raise ValueError(f"unexpected FDB1 header in {path}: magic={magic!r} count={count} dim={dim}")
    expected = 12 + count * dim * 4
    if len(raw) < expected:
        raise ValueError(f"truncated FDB1 file {path}: {len(raw)} < {expected}")
    arr = np.frombuffer(raw[12:expected], dtype=np.float32).reshape(count, dim).copy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-6)


def load_npz_gallery(path: Path, np: Any) -> Any:
    data = np.load(path)
    try:
        arr = np.asarray(data["embeddings"], dtype=np.float32)
    finally:
        data.close()
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, 1e-6)


def extract_frames_sync(
    ffmpeg_bin: str,
    clip_path: Path,
    meta: dict[str, Any],
    out_dir: Path,
    *,
    width: int = 1920,
    count: int = 5,
) -> list[Path]:
    source = clip_path
    suffix = clip_path.suffix.lower()
    if suffix in {".hevc", ".h265", ".h264"}:
        es_fmt = "hevc" if suffix in {".hevc", ".h265"} else "h264"
        muxed = out_dir / "clip.mp4"
        res = subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                es_fmt,
                "-i",
                str(clip_path),
                "-c",
                "copy",
                "-y",
                str(muxed),
            ],
            check=False,
            timeout=30,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0 and muxed.is_file() and muxed.stat().st_size > 0:
            source = muxed

    offsets: list[float] = []
    if meta:
        targets, err = _plan_verify_targets(
            meta, width_scale=1.5, height_scale=1.5, sample_count=count
        )
        if err is None and targets:
            offsets = [max(0.0, t.offset_seconds) for t in targets]
    if not offsets:
        offsets = [1.0, 4.0, 8.0, 12.0, 16.0][:count]

    frames: list[Path] = []
    for idx, off in enumerate(offsets, start=1):
        out = out_dir / f"frame-{idx}.jpg"
        subprocess.run(
            [
                ffmpeg_bin,
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{off:.3f}",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                f"scale={width}:-2",
                "-q:v",
                "3",
                "-y",
                str(out),
            ],
            check=False,
            timeout=25,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if out.is_file() and out.stat().st_size > 0:
            frames.append(out)
    return frames


def collect_positive_clips(
    database: Database,
    *,
    max_confirmed: int,
    include_feedback_daughter: bool,
) -> list[tuple[str, str, Path, dict[str, Any]]]:
    """收集用于构建底库的候选片段: (clip_id, source, path, board_meta)。"""
    adult_keys: set[str] = set()
    adult_paths: set[str] = set()
    for row in database.list_clip_feedback(limit=2000, label="adult_only"):
        adult_keys.add(str(row.get("item_key") or ""))
        if row.get("clip_path"):
            adult_paths.add(str(row["clip_path"]))

    clips: list[tuple[str, str, Path, dict[str, Any]]] = []
    seen_paths: set[str] = set()

    # 1. 人工标注为 daughter 的片段 (来自 Saved Moments 与 Rejected Queue)
    if include_feedback_daughter:
        for row in database.list_clip_feedback(limit=500, label="daughter"):
            clip_str = str(row.get("clip_path") or "")
            if not clip_str or clip_str in adult_paths or clip_str in seen_paths:
                continue
            path = Path(clip_str)
            if not path.is_file():
                continue
            seen_paths.add(clip_str)
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            clips.append((str(row.get("item_key") or path.name), f"feedback:{row.get('source')}", path, meta))

    # 2. 近期 rv1106_confirmed (及 rv1106_rescued) moments
    confirmed_added = 0
    for moment in database.list_moments(limit=500):
        if confirmed_added >= max_confirmed:
            break
        cat = str(moment.get("category") or "")
        if cat not in {"rv1106_confirmed", "rv1106_rescued"}:
            continue
        clip_str = str(moment.get("clip_path") or "")
        if not clip_str or clip_str in adult_paths or clip_str in seen_paths:
            continue
        trigger_key = str(moment.get("trigger_key") or "")
        if trigger_key and f"moment:{trigger_key}" in adult_keys:
            continue
        if f"moment:id:{moment['id']}" in adult_keys:
            continue
        path = Path(clip_str)
        if not path.is_file():
            continue
        meta: dict[str, Any] = {}
        meta_path = Path(str(moment.get("metadata_path") or ""))
        if meta_path.is_file():
            try:
                loaded = json.loads(meta_path.read_text(encoding="utf-8"))
                raw = loaded.get("model_raw") or loaded.get("board") or {}
                if isinstance(raw, dict):
                    meta = raw
            except (OSError, json.JSONDecodeError):
                pass
        seen_paths.add(clip_str)
        clips.append((f"moment:{moment['id']}", cat, path, meta))
        confirmed_added += 1

    return clips


def select_diverse_templates(
    candidates: list[CandidateFace],
    *,
    max_templates: int,
    dedup_similarity: float,
    np: Any,
) -> list[CandidateFace]:
    """按锚定相似度 + 人脸清晰度排序, 贪心去重选出多角度多样化底库模板。"""
    ranked = sorted(
        candidates,
        key=lambda c: (c.anchor_sim, min(c.face_px, 160) / 160.0, c.sharpness),
        reverse=True,
    )
    selected: list[CandidateFace] = []
    for cand in ranked:
        if len(selected) >= max_templates:
            break
        if selected:
            mat = np.stack([s.embedding for s in selected], axis=0)
            max_sim = float(np.max(mat @ cand.embedding))
            if max_sim >= dedup_similarity:
                continue
        selected.append(cand)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build NAS 4K ArcFace daughter gallery (.npz)")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", help="ArcFace ONNX path (default from .env)")
    parser.add_argument("--output", help="Output .npz gallery path (default from .env)")
    parser.add_argument("--anchor-board-db", help="RV1106 daughter.db (FDB1) to anchor face selection")
    parser.add_argument("--anchor-mbf", help="w600k_mbf.onnx path (required with --anchor-board-db)")
    parser.add_argument("--anchor-gallery", help="Existing R50 .npz gallery to anchor refresh")
    parser.add_argument("--anchor-threshold", type=float, default=0.38)
    parser.add_argument("--min-face-px", type=int, default=42)
    parser.add_argument("--min-face-score", type=float, default=0.75)
    parser.add_argument("--max-confirmed", type=int, default=60)
    parser.add_argument("--max-templates", type=int, default=25)
    parser.add_argument("--dedup-sim", type=float, default=0.88)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    import cv2
    import numpy as np

    settings = load_settings(args.env_file)
    database = Database(settings.database_path)

    model_path = Path(args.model) if args.model else settings.rv1106_verify_face_recog_model_path
    output_path = Path(args.output) if args.output else settings.rv1106_verify_face_recog_gallery_path
    yunet_path = settings.person_filter_model_dir / "face_detection_yunet_2023mar.onnx"

    if not model_path.is_file():
        print(f"ERROR: ArcFace model not found: {model_path}", file=sys.stderr)
        return 2
    if not yunet_path.is_file():
        print(f"ERROR: YuNet model not found: {yunet_path}", file=sys.stderr)
        return 2

    # Prepare main R50 extractor (without requiring gallery to exist yet)
    r50 = DaughterFaceVerifier(
        model_path=model_path,
        gallery_path=output_path,
        yunet_path=yunet_path,
        min_face_score=args.min_face_score,
        min_face_px=args.min_face_px,
    )
    r50._cv2 = cv2
    r50._np = np
    r50._arcface_dst = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    r50._yunet = cv2.FaceDetectorYN.create(str(yunet_path), "", (320, 320), args.min_face_score, 0.3, 5000)
    r50._net = cv2.dnn.readNetFromONNX(str(model_path))

    # Choose anchoring strategy
    anchor_mode = "cluster"
    anchor_vecs = None
    anchor_net = None

    if args.anchor_board_db:
        if not args.anchor_mbf or not Path(args.anchor_mbf).is_file():
            print("ERROR: --anchor-board-db requires --anchor-mbf pointing to w600k_mbf.onnx", file=sys.stderr)
            return 2
        anchor_vecs = load_board_fdb1(Path(args.anchor_board_db), np)
        anchor_net = cv2.dnn.readNetFromONNX(str(args.anchor_mbf))
        anchor_mode = f"board_db({len(anchor_vecs)})"
    elif args.anchor_gallery:
        anchor_vecs = load_npz_gallery(Path(args.anchor_gallery), np)
        anchor_mode = f"gallery_npz({len(anchor_vecs)})"
    elif output_path.is_file():
        anchor_vecs = load_npz_gallery(output_path, np)
        anchor_mode = f"existing_gallery({len(anchor_vecs)})"

    def embed_with_net(net: Any, aligned: Any) -> Any:
        blob = cv2.dnn.blobFromImage(
            aligned, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
        )
        net.setInput(blob)
        vec = net.forward().reshape(-1).astype(np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else vec

    clips = collect_positive_clips(
        database,
        max_confirmed=args.max_confirmed,
        include_feedback_daughter=True,
    )
    print(f"Scanning {len(clips)} positive clips (anchor={anchor_mode}, thresh={args.anchor_threshold}, min_px={args.min_face_px})...")

    per_clip_candidates: list[CandidateFace] = []
    # For cluster fallback mode: collect all faces per clip
    cluster_pool: list[tuple[str, str, list[tuple[Any, int, float, float]]]] = []

    for clip_id, source, clip_path, meta in clips:
        with tempfile.TemporaryDirectory(prefix="nas-gallery-") as tmp:
            frames = extract_frames_sync(
                settings.ffmpeg_bin,
                clip_path,
                meta,
                Path(tmp),
                width=settings.rv1106_verify_face_recog_full_width,
                count=5,
            )
            best_for_clip: CandidateFace | None = None
            clip_faces: list[tuple[Any, int, float, float]] = []
            for frame_path in frames:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                for aligned, fscore, fpx in r50.detect_and_align_faces(img):
                    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
                    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    r50_vec = r50.embed_aligned(aligned)
                    if anchor_vecs is not None:
                        probe = embed_with_net(anchor_net, aligned) if anchor_net is not None else r50_vec
                        sim = float(np.max(anchor_vecs @ probe))
                        if sim < args.anchor_threshold:
                            continue
                        cand = CandidateFace(
                            clip_id=clip_id,
                            source=source,
                            embedding=r50_vec,
                            anchor_sim=sim,
                            face_px=fpx,
                            face_score=fscore,
                            sharpness=sharp,
                        )
                        if best_for_clip is None or (
                            cand.anchor_sim, cand.face_px, cand.sharpness
                        ) > (
                            best_for_clip.anchor_sim, best_for_clip.face_px, best_for_clip.sharpness
                        ):
                            best_for_clip = cand
                    else:
                        clip_faces.append((r50_vec, fpx, fscore, sharp))
            if best_for_clip is not None:
                per_clip_candidates.append(best_for_clip)
            elif clip_faces:
                cluster_pool.append((clip_id, source, clip_faces))

    if anchor_vecs is None and cluster_pool:
        # Dominant identity cluster across clips: pick face whose embedding matches the most distinct clips
        all_vecs = [vec for _, _, faces in cluster_pool for vec, _, _, _ in faces]
        mat = np.stack(all_vecs, axis=0)
        best_seed = None
        best_clip_hits = -1
        for seed in all_vecs:
            hits = 0
            for _, _, faces in cluster_pool:
                if any(float(np.dot(seed, v)) >= 0.45 for v, _, _, _ in faces):
                    hits += 1
            if hits > best_clip_hits:
                best_clip_hits = hits
                best_seed = seed
        if best_seed is not None:
            for clip_id, source, faces in cluster_pool:
                best_for_clip = None
                for vec, fpx, fscore, sharp in faces:
                    sim = float(np.dot(best_seed, vec))
                    if sim < args.anchor_threshold:
                        continue
                    cand = CandidateFace(clip_id, source, vec, sim, fpx, fscore, sharp)
                    if best_for_clip is None or (cand.anchor_sim, cand.face_px) > (
                        best_for_clip.anchor_sim, best_for_clip.face_px
                    ):
                        best_for_clip = cand
                if best_for_clip is not None:
                    per_clip_candidates.append(best_for_clip)

    if not per_clip_candidates:
        print("ERROR: no matching daughter faces found; gallery not updated.", file=sys.stderr)
        return 1

    selected = select_diverse_templates(
        per_clip_candidates,
        max_templates=args.max_templates,
        dedup_similarity=args.dedup_sim,
        np=np,
    )
    print(f"Matched {len(per_clip_candidates)} clips -> selected {len(selected)} diverse templates:")
    for idx, item in enumerate(selected, start=1):
        print(
            f"  #{idx:02d} {item.clip_id:<28} src={item.source:<18} "
            f"anchor_sim={item.anchor_sim:.3f} px={item.face_px:>3} sharp={item.sharpness:6.1f}"
        )

    if args.dry_run:
        print("Dry run complete; no file written.")
        return 0

    embeddings = np.stack([item.embedding for item in selected], axis=0).astype(np.float32)
    meta_records = [
        {
            "clip_id": item.clip_id,
            "source": item.source,
            "anchor_sim": round(item.anchor_sim, 4),
            "face_px": item.face_px,
            "face_score": round(item.face_score, 4),
            "sharpness": round(item.sharpness, 2),
        }
        for item in selected
    ]
    summary_meta = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model_path.name,
        "anchor_mode": anchor_mode,
        "template_count": len(selected),
        "templates": meta_records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.npz")
    try:
        np.savez(
            tmp_path,
            embeddings=embeddings,
            meta_json=json.dumps(summary_meta, ensure_ascii=False),
        )
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Saved gallery: {output_path} ({embeddings.shape[0]}x{embeddings.shape[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
