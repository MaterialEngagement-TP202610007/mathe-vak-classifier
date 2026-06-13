"""Simple-score fallback classifier.

Used by the backend when Lambda is unavailable. The predominant style is the
highest VAK score and confidence is each score's share of the total. Returns the
same JSON contract as the model with classifier_type = "simple_score".
"""

from __future__ import annotations

VAK_CLASSES = ["Visual", "Auditory", "Kinesthetic"]
SCORE_KEYS = ["visual_score", "auditory_score", "kinesthetic_score"]

CLEAR_MARGIN = 0.30
MIXED_MARGIN = 0.12


def _profile_type(top: float, second: float) -> str:
    gap = top - second
    if gap >= CLEAR_MARGIN:
        return "clear"
    if gap >= MIXED_MARGIN:
        return "tendency"
    return "mixed"


def classify_by_score(raw: dict) -> dict:
    scores = [float(raw[k]) for k in SCORE_KEYS]
    total = sum(scores)
    proba = [s / total for s in scores] if total > 0 else [1 / 3] * 3

    order = sorted(range(3), key=lambda i: proba[i], reverse=True)
    top, second = order[0], order[1]
    profile = _profile_type(proba[top], proba[second])

    confidence = {VAK_CLASSES[i]: round(proba[i] * 100, 2) for i in range(3)}
    return {
        "predominant_style": VAK_CLASSES[top],
        "secondary_style": VAK_CLASSES[second],
        "confidence": confidence,
        "predominant_confidence": confidence[VAK_CLASSES[top]],
        "profile_type": profile,
        "is_mixed_profile": profile == "mixed",
        "classifier_type": "simple_score",
    }


if __name__ == "__main__":
    import json

    example = {"visual_score": 12, "auditory_score": 3, "kinesthetic_score": 5}
    print(json.dumps(classify_by_score(example), indent=2, ensure_ascii=False))
