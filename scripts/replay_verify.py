#!/usr/bin/env python3
"""回放已知片段, 校验 probable 裁剪校验的判定是否符合人工标注。

在 NAS jail 内运行（需要数据库 + 落盘片段 + ffmpeg）:

    cd /path/to/repo && uv run python scripts/replay_verify.py
    uv run python scripts/replay_verify.py --ids 3784,3791
    uv run python scripts/replay_verify.py --day 2026-11-05 --verbose

判定复用生产逻辑: `ingest._verify_probable_clip()` 与
`daughter_detector.verify_board_probable_paths()`, 不做任何简化, 因此结果与
线上一致（受 .env 里当前的阈值影响）。

标注来源: 2026-11-03 ~ 11-05 人工逐条抽帧确认（见 docs/verify-fixtures.md）。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nas_video_summarizer.config import load_settings  # noqa: E402
from nas_video_summarizer.database import Database  # noqa: E402
from nas_video_summarizer.daughter_detector import DaughterDetector  # noqa: E402
from nas_video_summarizer.ingest import _verify_probable_clip  # noqa: E402

# 人工标注: 画面里是否真的有女儿
EXPECT_REJECT = {3758, 3761, 3784, 3791}          # 只有大人
EXPECT_ACCEPT = {3755, 3759, 3760, 3785, 3786,  # 有女儿
                 3787, 3788, 3789, 3792}


def load_board_meta(metadata_path: Path) -> dict:
    """从 moment 的元数据里取回板端上传时携带的 meta。"""
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    meta = data.get("model_raw") or data.get("board") or {}
    if not isinstance(meta, dict):
        return {}
    return meta


async def replay_one(settings, database, detector, moment_id: int, verbose: bool):
    row = database.get_moment(moment_id)
    if row is None:
        return moment_id, "missing-in-db", None, None

    clip = Path(str(row["clip_path"]))
    meta_path = Path(str(row["metadata_path"]))
    if not clip.exists():
        return moment_id, "clip-missing", None, None
    if not meta_path.exists():
        return moment_id, "meta-missing", None, None

    meta = load_board_meta(meta_path)
    accepted, decision, verification = await _verify_probable_clip(
        settings, clip, meta, detector, job_key=f"replay-{moment_id}"
    )
    detail = None
    if verification is not None:
        detail = {
            "child_face_frames": getattr(verification, "child_face_frames", None),
            "face_frames": getattr(verification, "face_frames", None),
            "person_frames": getattr(verification, "person_frames", None),
            "max_person_height": getattr(verification, "max_person_height", None),
            "board_score": meta.get("score"),
            "board_person_score": meta.get("person_score"),
        }
    return moment_id, decision, accepted, detail


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", help="逗号分隔的 moment id")
    parser.add_argument("--day", help="按日期回放, 如 2026-11-05")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()

    settings = load_settings(args.env_file)
    database = Database(settings.database_path)
    detector = DaughterDetector(settings)

    if args.ids:
        ids = [int(v) for v in args.ids.split(",") if v.strip()]
    else:
        rows = database.list_moments(limit=500)
        if args.day:
            rows = [
                r for r in rows
                if str(r.get("source_started_at", "")).startswith(args.day)
            ]
        ids = [int(r["id"]) for r in rows if r.get("analysis_backend") == "rv1106_edge"]
        ids.sort()

    print(f"回放 {len(ids)} 条 (policy={settings.rv1106_probable_policy}, "
          f"child_face_frames={settings.rv1106_verify_child_face_frames}, "
          f"child_max_h={settings.rv1106_verify_child_face_max_person_height}, "
          f"fallback_faceless={settings.rv1106_verify_fallback_allow_faceless})")

    passed = failed = unknown = 0
    for moment_id in ids:
        mid, decision, accepted, detail = await replay_one(
            settings, database, detector, moment_id, args.verbose
        )
        expected = (
            "reject" if mid in EXPECT_REJECT
            else "accept" if mid in EXPECT_ACCEPT
            else "unknown"
        )
        if expected == "unknown" or accepted is None:
            verdict = "-"
            unknown += 1
        else:
            ok = (accepted and expected == "accept") or (
                not accepted and expected == "reject"
            )
            verdict = "PASS" if ok else "FAIL"
            if ok:
                passed += 1
            else:
                failed += 1
        extra = ""
        if detail and args.verbose:
            extra = " " + " ".join(f"{k}={v}" for k, v in detail.items())
        print(f"  {mid:>5}  expect={expected:<7} accepted={accepted!s:<5} "
              f"decision={decision:<24} {verdict}{extra}")

    print(f"\nPASS={passed} FAIL={failed} 未标注={unknown}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
