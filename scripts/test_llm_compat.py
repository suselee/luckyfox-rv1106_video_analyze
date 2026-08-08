#!/usr/bin/env python3
"""Diagnostic test: Gemma 4 E2B + response_format + images.

Run on the NAS:
    python3 /tmp/test_llm_compat.py /mnt/var/buffer/home-camera/low

Tests 6 scenarios to isolate the root cause:
  1. text-only + response_format
  2. text-only, no response_format
  3. single image + response_format
  4. single image, no response_format
  5. full analysis prompt + 8 images + response_format
  6. full analysis prompt + 8 images, no response_format
"""
import sys
import os
import json
import base64
import urllib.request
import urllib.error
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

BASE_URL = os.getenv("LLAMA_BASE_URL", "http://192.168.123.202:8892/v1")
MODEL = os.getenv("LLAMA_MODEL", "gemma-4-E2B-it-qat")
FRAME_WIDTH = 512


def extract_frame(video_path: Path, output_path: Path, offset: float = 30.0):
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", str(offset), "-i", str(video_path),
        "-frames:v", "1", "-vf", f"scale={FRAME_WIDTH}:-2", "-q:v", "4",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.strip()}")
        return False
    return output_path.exists()


def find_good_video(buffer_dir: str) -> Path | None:
    p = Path(buffer_dir)
    if not p.exists():
        print(f"Buffer dir not found: {p}")
        return None
    for f in sorted(p.glob("*.mp4"), reverse=True):
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", str(f)],
            capture_output=True, text=True
        )
        if probe.returncode == 0:
            return f
    return None


def post_chat(payload: dict) -> dict:
    url = BASE_URL.rstrip("/") + "/chat/completions"
    timeout = payload.pop("_timeout", 120)
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"error": str(exc)}


def img_content(path: Path) -> dict:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def run_test(name: str, payload: dict):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print(f"{'='*70}")
    timeout = payload.get("_timeout", 120)
    print(f"  payload keys: {list(payload.keys())}")
    if "response_format" in payload:
        print(f"  response_format: {payload['response_format']}")
    print(f"  has images: {any('image_url' in str(m.get('content','')) for m in payload.get('messages',[]))}")

    resp = post_chat(payload)
    if "error" in resp:
        print(f"  ERROR: {resp['error'][:500]}")
        return

    content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
    finish = resp.get("choices", [{}])[0].get("finish_reason", "?")
    usage = resp.get("usage", {})

    if isinstance(content, list):
        text = "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    else:
        text = content

    print(f"  finish_reason: {finish}")
    print(f"  prompt_tokens: {usage.get('prompt_tokens', '?')}")
    print(f"  completion_tokens: {usage.get('completion_tokens', '?')}")
    print(f"  raw_content ({len(text)} chars):")
    print(f"  >> {text[:2000]}")
    print()


def main():
    buffer_dir = sys.argv[1] if len(sys.argv) > 1 else "/mnt/var/buffer/home-camera/low"
    print(f"LLAMA_BASE_URL = {BASE_URL}")
    print(f"LLAMA_MODEL    = {MODEL}")
    print(f"BUFFER_DIR     = {buffer_dir}")

    # Find a good video file
    video = find_good_video(buffer_dir)
    if not video:
        print("No valid video found. Exiting.")
        sys.exit(1)
    print(f"Using video: {video.name}")

    # Extract frames
    with tempfile.TemporaryDirectory(prefix="llm-test-") as tmp:
        frames = []
        for i, offset in enumerate([15, 30, 45, 60, 75, 90, 105, 120]):
            if i >= 8:
                break
            fp = Path(tmp) / f"frame_{i+1}.jpg"
            if extract_frame(video, fp, offset=float(offset)):
                frames.append(fp)
                print(f"  extracted frame {i+1} @ {offset}s ({fp.stat().st_size} bytes)")
            else:
                print(f"  FAILED to extract frame {i+1} @ {offset}s")

        if not frames:
            print("No frames extracted. Exiting.")
            sys.exit(1)
        print(f"\nExtracted {len(frames)} frames total.")

        base_msg = [{"role": "user", "content": "Say hello in JSON: {\"greeting\": \"hi\"}"}]

        # Test 1: text-only + response_format
        run_test("1. text-only + response_format", {
            "model": MODEL, "messages": base_msg, "temperature": 0.2,
            "response_format": {"type": "json_object"},
        })

        # Test 2: text-only, no response_format
        run_test("2. text-only, no response_format", {
            "model": MODEL, "messages": base_msg, "temperature": 0.2,
        })

        # Test 3: single image + response_format
        run_test("3. single image + response_format", {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image. Return JSON: {\"description\": \"...\", \"person_visible\": true/false}"},
                img_content(frames[0]),
            ]}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        })

        # Test 4: single image, no response_format
        run_test("4. single image, no response_format", {
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Describe this image. Return JSON: {\"description\": \"...\", \"person_visible\": true/false}"},
                img_content(frames[0]),
            ]}],
            "temperature": 0.2,
        })

        # Full analysis prompt (mimicking the actual system prompt)
        full_prompt = (
            "Find short, precious indoor moments of my daughter in the living room. "
            "ONLY keep clips where my daughter is ACTIVELY doing something. "
            "Return exactly one JSON object with: keep (boolean), title, summary, tags (array), "
            "confidence (0-1), start_offset_seconds (int), end_offset_seconds (int). "
            "The images are sampled video frames in chronological order. "
        )

        # Test different image counts to find the sweet spot
        for img_count in [2, 3, 4]:
            frame_labels = ", ".join(
                f"Frame #{i+1}: ~{15*(i+1)}s"
                for i in range(img_count)
            )
            prompt_text = full_prompt + frame_labels + f"\nSegment duration: 120s"
            content = [{"type": "text", "text": prompt_text}]
            content.extend(img_content(f) for f in frames[:img_count])

            run_test(f"{img_count}. {img_count} images + response_format (timeout=300s)", {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": "You are a careful family video curator. Return JSON only."},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
                "_timeout": 300,
            })

    print("\n" + "=" * 70)
    print("SUMMARY:")
    print("  Tests 1-4 already ran. Tests 5-6 (8 images) timed out at 120s.")
    print("  Tests 2/3/4 test with 2-3-4 images at 300s timeout.")
    print("  Goal: find max image count that completes within 300s.")
    print("=" * 70)


if __name__ == "__main__":
    main()