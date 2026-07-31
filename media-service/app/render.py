from __future__ import annotations

import subprocess
from pathlib import Path


def safe_path(value: str, work_root: str) -> Path:
    path = Path(value).resolve()
    root = Path(work_root).resolve()
    if path != root and root not in path.parents:
        raise ValueError("path must be below WORK_ROOT")
    return path


def render_video(
    video_path: str,
    voice_path: str,
    output_path: str,
    work_root: str,
    subtitle_path: str | None = None,
) -> dict:
    video = safe_path(video_path, work_root)
    voice = safe_path(voice_path, work_root)
    output = safe_path(output_path, work_root)
    if not video.is_file() or not voice.is_file():
        raise FileNotFoundError("video or voice asset missing")
    output.parent.mkdir(parents=True, exist_ok=True)

    video_filters = [
        "scale=1920:1080:force_original_aspect_ratio=decrease",
        "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:black",
        "fps=30",
        "format=yuv420p",
    ]
    if subtitle_path:
        subtitle = safe_path(subtitle_path, work_root)
        if not subtitle.is_file():
            raise FileNotFoundError("subtitle asset missing")
        escaped = str(subtitle).replace("\\", "\\\\").replace(":", "\\:")
        video_filters.append(
            "subtitles="
            + escaped
            + ":force_style='FontName=DejaVu Sans,FontSize=20,"
              "Outline=2,Shadow=1,Alignment=2,MarginV=55'"
        )

    command = [
        "ffmpeg", "-y", "-v", "error",
        "-i", str(video), "-i", str(voice),
        "-filter:v", ",".join(video_filters),
        "-filter:a", "loudnorm=I=-14:TP=-1.5:LRA=11",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest", "-movflags", "+faststart", str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return {"output_path": str(output), "bytes": output.stat().st_size, "command": command}


def render_thumbnail(
    background_path: str,
    output_path: str,
    work_root: str,
    headline: str,
) -> dict:
    background = safe_path(background_path, work_root)
    output = safe_path(output_path, work_root)
    if not background.is_file():
        raise FileNotFoundError("background asset missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_text = headline.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    filters = (
        "scale=1280:720:force_original_aspect_ratio=increase,"
        "crop=1280:720,"
        "drawbox=x=0:y=470:w=1280:h=250:color=black@0.62:t=fill,"
        f"drawtext=font='DejaVu Sans':text='{safe_text}':"
        "fontcolor=white:fontsize=68:borderw=3:bordercolor=black:"
        "x=(w-text_w)/2:y=520"
    )
    command = [
        "ffmpeg", "-y", "-v", "error", "-i", str(background),
        "-frames:v", "1", "-vf", filters, str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    if result.returncode:
        raise RuntimeError(result.stderr[-2000:])
    return {"output_path": str(output), "bytes": output.stat().st_size}

