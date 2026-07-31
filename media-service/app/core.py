from __future__ import annotations

from typing import Any, Mapping


def opportunity_score(c: Mapping[str, Any]) -> dict[str, Any]:
    views = int(c["views"])
    age_hours = float(c["age_hours"])
    median = max(int(c.get("channel_median_views", 1)), 1)
    likes = int(c.get("likes", 0))
    comments = int(c.get("comments", 0))
    velocity = min(100.0, (views / age_hours) / 500.0 * 100.0)
    anomaly = min(100.0, views / median * 20.0)
    engagement = min(
        100.0, ((likes + comments * 4.0) / max(views, 1)) * 1000.0
    )
    search_demand = float(c.get("search_demand", 50))
    localization_fit = float(c.get("localization_fit", 50))
    competition = float(c.get("competition", 50))
    score = (
        velocity * 0.24
        + anomaly * 0.21
        + engagement * 0.12
        + search_demand * 0.18
        + localization_fit * 0.17
        + (100.0 - competition) * 0.08
    )
    rights_verified = bool(c.get("rights_verified", False))
    return {
        "score": round(score, 2),
        "eligible_for_production": score >= 70 and rights_verified,
        "requires_rights_review": not rights_verified,
        "components": {
            "velocity": round(velocity, 2),
            "anomaly": round(anomaly, 2),
            "engagement": round(engagement, 2),
            "search_demand": search_demand,
            "localization_fit": localization_fit,
            "competition_inverse": 100.0 - competition,
        },
    }


def quality_gate(m: Mapping[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    if not m.get("rights_verified"):
        failures.append("RIGHTS_NOT_VERIFIED")
    if len(str(m.get("original_script", "")).strip()) < 500:
        failures.append("SCRIPT_TOO_SHORT")
    if len(str(m.get("hook", "")).strip()) < 20:
        failures.append("WEAK_HOOK")
    if len(str(m.get("payoff", "")).strip()) < 20:
        failures.append("MISSING_PAYOFF")
    if float(m.get("subtitle_coverage", 0)) < 0.97:
        failures.append("LOW_SUBTITLE_COVERAGE")
    if not -16.5 <= float(m.get("audio_lufs", -99)) <= -13.0:
        failures.append("LOUDNESS_OUT_OF_RANGE")
    if float(m.get("audio_true_peak_db", 1)) > -1.0:
        failures.append("AUDIO_CLIPPING_RISK")
    if float(m.get("silence_ratio", 1)) > 0.08:
        failures.append("EXCESSIVE_SILENCE")
    if float(m.get("visual_changes_per_minute", 0)) < 10:
        failures.append("LOW_VISUAL_DENSITY")
    if float(m.get("duplicate_scene_ratio", 1)) > 0.18:
        failures.append("DUPLICATE_SCENES")
    if float(m.get("thumbnail_readability", 0)) < 80:
        failures.append("WEAK_THUMBNAIL")
    if m.get("channel_profile") == "kids":
        if not m.get("child_safety_passed"):
            failures.append("CHILD_SAFETY_FAILED")
        if not m.get("human_reviewed"):
            failures.append("KIDS_HUMAN_REVIEW_REQUIRED")
    return {
        "passed": not failures,
        "quality_score": max(0, 100 - len(failures) * 9),
        "failures": failures,
        "next_status": "READY_PRIVATE_UPLOAD" if not failures else "REWORK",
    }

