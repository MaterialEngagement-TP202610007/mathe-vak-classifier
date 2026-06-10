"""Simple-score fallback classifier.

Used by the backend when Lambda is unavailable. The predominant style is the
highest VAK score and "confianza" is each score's share of the total. Returns
the same JSON contract as the model with clasificador_tipo = "puntaje_simple".
This is the reference the Express backend reproduces.
"""

from __future__ import annotations

VAK_DISPLAY = ["Visual", "Auditivo", "Kinestesico"]
SCORE_KEYS = ["visual_score", "auditory_score", "kinesthetic_score"]

CLEAR_MARGIN = 0.30
MIXED_MARGIN = 0.12


def _profile_type(top: float, second: float) -> str:
    gap = top - second
    if gap >= CLEAR_MARGIN:
        return "claro"
    if gap >= MIXED_MARGIN:
        return "tendencia"
    return "mixto"


def classify_by_score(raw: dict) -> dict:
    scores = [float(raw[k]) for k in SCORE_KEYS]
    total = sum(scores)
    proba = [s / total for s in scores] if total > 0 else [1 / 3] * 3

    order = sorted(range(3), key=lambda i: proba[i], reverse=True)
    top, second = order[0], order[1]
    tipo = _profile_type(proba[top], proba[second])

    confianza = {VAK_DISPLAY[i]: round(proba[i] * 100, 2) for i in range(3)}
    return {
        "estilo_predominante": VAK_DISPLAY[top],
        "estilo_secundario": VAK_DISPLAY[second],
        "confianza": confianza,
        "confianza_predominante": confianza[VAK_DISPLAY[top]],
        "tipo_perfil": tipo,
        "es_perfil_mixto": tipo == "mixto",
        "clasificador_tipo": "puntaje_simple",
    }


if __name__ == "__main__":
    import json

    example = {"visual_score": 12, "auditory_score": 3, "kinesthetic_score": 5}
    print(json.dumps(classify_by_score(example), indent=2, ensure_ascii=False))
