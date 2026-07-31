from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import httpx


API_URL = "https://api.pexels.com/videos/search"


def _enabled() -> bool:
    return os.environ.get("PEXELS_VIDEO_ENABLED", "true").strip().lower() == "true"


def _queries(scene: dict[str, Any], media_job: dict[str, Any] | None) -> list[str]:
    values: list[Any] = []
    if media_job:
        values.extend((media_job.get("queries") or {}).get("en") or [])
    values.extend(scene.get("search_queries_en") or [])
    intent = str(scene.get("visual_intent") or "").strip()
    if intent:
        values.append(intent)
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))


def _best_file(video: dict[str, Any]) -> dict[str, Any] | None:
    candidates = [
        item for item in video.get("video_files", [])
        if item.get("link") and item.get("file_type") == "video/mp4"
        and int(item.get("width") or 0) >= int(item.get("height") or 0)
    ]
    if not candidates:
        return None
    # Prefer landscape HD without needlessly downloading 4K material.
    return min(
        candidates,
        key=lambda item: (
            0 if int(item.get("width") or 0) >= 1280 else 1,
            abs(int(item.get("width") or 0) - 1920),
            abs(int(item.get("height") or 0) - 1080),
        ),
    )


def acquire_scene_video(
    scene: dict[str, Any],
    media_job: dict[str, Any] | None,
    raw_path: Path,
) -> dict[str, Any] | None:
    api_key = os.environ.get("PEXELS_API_KEY", "").strip()
    if not _enabled() or not api_key:
        return None
    per_page = max(1, min(30, int(os.environ.get("PEXELS_RESULTS_PER_PAGE", "10"))))
    max_bytes = max(10, int(os.environ.get("PEXELS_MAX_DOWNLOAD_MB", "120"))) * 1024 * 1024
    headers = {"Authorization": api_key}
    queries = _queries(scene, media_job)

    with httpx.Client(timeout=httpx.Timeout(90, connect=20), follow_redirects=True) as client:
        for query in queries:
            try:
                response = client.get(
                    API_URL,
                    headers=headers,
                    params={"query": query, "orientation": "landscape", "size": "medium", "per_page": per_page},
                )
                response.raise_for_status()
            except httpx.HTTPError:
                continue
            for video in response.json().get("videos", []):
                selected = _best_file(video)
                if not selected:
                    continue
                try:
                    with client.stream("GET", selected["link"]) as download:
                        download.raise_for_status()
                        total = 0
                        with raw_path.open("wb") as output:
                            for chunk in download.iter_bytes(1024 * 1024):
                                total += len(chunk)
                                if total > max_bytes:
                                    output.close()
                                    raw_path.unlink(missing_ok=True)
                                    break
                                output.write(chunk)
                        if not raw_path.exists():
                            continue
                except (httpx.HTTPError, OSError):
                    raw_path.unlink(missing_ok=True)
                    continue
                user = video.get("user") or {}
                return {
                    "provider": "pexels",
                    "scene_number": int(scene["scene_number"]),
                    "query": query,
                    "video_id": video.get("id"),
                    "video_page_url": video.get("url"),
                    "creator": user.get("name"),
                    "creator_url": user.get("url"),
                    "source_width": selected.get("width"),
                    "source_height": selected.get("height"),
                }
    return None


def normalize_video(raw_path: Path, output_path: Path, duration: float) -> None:
    command = [
        "ffmpeg", "-y", "-v", "error", "-stream_loop", "-1", "-i", str(raw_path),
        "-t", str(duration),
        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30,format=yuv420p",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
