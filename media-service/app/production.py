from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import textwrap
import time
import wave
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw, ImageFont


def _run(command: list[str], timeout: int = 3600) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _scene_card(path: Path, scene: dict[str, Any], title: str) -> None:
    n = int(scene.get("scene_number", 1))
    palettes = [(9, 25, 51), (20, 52, 74), (45, 27, 74), (17, 64, 58), (63, 38, 24)]
    base = palettes[(n - 1) % len(palettes)]
    image = Image.new("RGB", (1920, 1080), base)
    draw = ImageDraw.Draw(image)
    for x in range(0, 1920, 120):
        shade = tuple(min(255, c + (x // 120) % 2 * 6) for c in base)
        draw.rectangle((x, 0, x + 120, 1080), fill=shade)
    draw.rectangle((90, 90, 1830, 990), outline=(80, 190, 220), width=4)
    draw.text((130, 130), f"SAHNE {n:02d}", font=_font(44), fill=(90, 215, 235))
    intent = str(scene.get("visual_intent") or title)
    lines = textwrap.wrap(intent, width=38)[:5]
    y = 360
    for line in lines:
        draw.text((150, y), line, font=_font(68), fill="white")
        y += 86
    image.save(path, quality=94)


def _write_wav(path: Path, pcm: bytes, rate: int = 24000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


def _tts(text: str, voice: str, delivery: str, output: Path) -> None:
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required by the media service")
    model = os.environ.get("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    prompt = f"Türkçe olarak, metni aynen oku. Sunum tarzı: {delivery}. Metin: {text}"
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "languageCode": "tr-TR",
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}},
            },
        },
    }
    with httpx.Client(timeout=180) as client:
        for attempt in range(6):
            response = client.post(url, headers={"x-goog-api-key": api_key}, json=payload)
            if response.status_code not in (429, 500, 502, 503, 504):
                response.raise_for_status()
                break
            if attempt == 5:
                response.raise_for_status()
            retry_after = int(response.headers.get("retry-after", "0") or 0)
            time.sleep(max(retry_after, min(60, 5 * (2**attempt))))
        data = response.json()
    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    inline = next((p.get("inlineData") or p.get("inline_data") for p in parts if p.get("inlineData") or p.get("inline_data")), None)
    if not inline or not inline.get("data"):
        raise RuntimeError("Gemini TTS returned no audio")
    audio = base64.b64decode(inline["data"])
    mime = str(inline.get("mimeType") or inline.get("mime_type") or "")
    if "wav" in mime.lower() or audio[:4] == b"RIFF":
        output.write_bytes(audio)
    else:
        _write_wav(output, audio)


def _concat_wav(turns: list[dict[str, Any]], audio_dir: Path, output: Path, target_seconds: float = 0) -> float:
    frames: list[bytes] = []
    rate = 24000
    for turn in turns:
        for key in ("pause_before_ms",):
            ms = max(0, int(turn.get(key, 0) or 0))
            frames.append(b"\0\0" * int(rate * ms / 1000))
        wav_path = audio_dir / f"{int(turn['sequence']):03d}-{turn['speaker_id']}.wav"
        with wave.open(str(wav_path), "rb") as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != rate:
                raise RuntimeError(f"Unexpected TTS WAV format: {wav_path.name}")
            frames.append(wf.readframes(wf.getnframes()))
        ms = max(0, int(turn.get("pause_after_ms", 0) or 0))
        frames.append(b"\0\0" * int(rate * ms / 1000))
    pcm = b"".join(frames)
    current_seconds = len(pcm) / 2 / rate
    if target_seconds > current_seconds:
        pcm += b"\0\0" * int(rate * (target_seconds - current_seconds))
    _write_wav(output, pcm, rate)
    return len(pcm) / 2 / rate


def produce(job: dict[str, Any], job_dir: Path, public_base: str) -> dict[str, Any]:
    payload = job["payload"]
    documentary = payload["documentary"]
    turns = sorted(documentary.get("dialogue_turns", []), key=lambda x: int(x.get("sequence", 0)))
    scenes = sorted(payload.get("timeline", {}).get("scenes", []), key=lambda x: int(x.get("scene_number", 0)))
    if len(turns) != 42 or len(scenes) != 22:
        raise RuntimeError("Production requires exactly 42 turns and 22 scenes")

    audio_dir = job_dir / "audio"
    scene_dir = job_dir / "scenes"
    output_dir = job_dir / "output"
    for directory in (audio_dir, scene_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)

    speakers = {s["speaker_id"]: s for s in documentary.get("speakers", [])}
    for index, turn in enumerate(turns, 1):
        speaker = speakers.get(turn["speaker_id"], {})
        voice = speaker.get("voice_name") or {"SPEAKER_A": "Kore", "SPEAKER_B": "Charon", "SPEAKER_C": "Puck"}[turn["speaker_id"]]
        _tts(turn["text"], voice, turn.get("delivery", "calm_documentary"), audio_dir / f"{index:03d}-{turn['speaker_id']}.wav")
        job.update(progress=5 + int(index / len(turns) * 45), message=f"TTS {index}/{len(turns)}")

    narration = output_dir / "narration.wav"
    planned_duration = sum(max(4.0, float(scene.get("duration_seconds", 4))) for scene in scenes)
    audio_duration = _concat_wav(turns, audio_dir, narration, planned_duration)
    clip_list = job_dir / "clips.txt"
    clip_lines: list[str] = []
    for index, scene in enumerate(scenes, 1):
        image_path = scene_dir / f"scene-{index:03d}.jpg"
        clip_path = scene_dir / f"scene-{index:03d}.mp4"
        _scene_card(image_path, scene, documentary.get("title", "Belgesel"))
        duration = max(4.0, float(scene.get("duration_seconds", 4)))
        _run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(image_path), "-t", str(duration), "-vf", "scale=1920:1080,zoompan=z='min(zoom+0.0005,1.08)':d=1:s=1920x1080:fps=30,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-an", str(clip_path)])
        clip_lines.append(f"file '{clip_path.as_posix()}'")
        job.update(progress=52 + int(index / len(scenes) * 28), message=f"Scene {index}/{len(scenes)}")
    clip_list.write_text("\n".join(clip_lines), encoding="utf-8")
    silent_video = output_dir / "silent.mp4"
    _run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(clip_list), "-c", "copy", str(silent_video)])

    subtitle = output_dir / "documentary-tr.srt"
    subtitle.write_text(payload.get("timeline", {}).get("subtitles", {}).get("content", ""), encoding="utf-8")
    final_video = output_dir / "documentary.mp4"
    escaped_sub = str(subtitle).replace("\\", "\\\\").replace(":", "\\:")
    _run(["ffmpeg", "-y", "-v", "error", "-i", str(silent_video), "-i", str(narration), "-vf", f"subtitles={escaped_sub}:force_style='FontName=DejaVu Sans,FontSize=22,Outline=2,Shadow=1,Alignment=2,MarginV=55'", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-b:a", "192k", "-filter:a", "loudnorm=I=-14:TP=-1.5:LRA=11", "-shortest", "-movflags", "+faststart", str(final_video)], timeout=7200)
    thumbnail = output_dir / "thumbnail.jpg"
    shutil.copy2(scene_dir / "scene-001.jpg", thumbnail)

    base = public_base.rstrip("/")
    job_id = job["job_id"]
    return {
        "video_url": f"{base}/files/{job_id}/output/documentary.mp4",
        "thumbnail_url": f"{base}/files/{job_id}/output/thumbnail.jpg",
        "metrics": {
            "duration_seconds": round(audio_duration, 2), "width": 1920, "height": 1080,
            "has_video": True, "has_audio": True, "has_subtitles": True, "thumbnail_created": True,
            "video_size_bytes": final_video.stat().st_size,
        },
    }
