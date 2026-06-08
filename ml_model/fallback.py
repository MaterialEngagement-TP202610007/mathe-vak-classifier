"""
fallback.py
===========
Clasificador de respaldo por PUNTAJE SIMPLE (Notion - Fase 4, tarea 2).

El plan exige que el fallback este SIEMPRE disponible como respaldo, incluso cuando
XGBoost funcione. Si Lambda falla (timeout, 500, no disponible), el backend usa esta
logica: el estilo predominante es el del mayor puntaje VAK, y la "confianza" se estima
como la proporcion de cada puntaje sobre el total.

Esta es la implementacion de referencia en Python; el backend Express.js debe
reproducir la misma logica. Devuelve el mismo contrato JSON que la Lambda, con
`clasificador_tipo = "puntaje_simple"`.
"""

from __future__ import annotations

VAK_CLASSES = ["Visual", "Auditivo", "Kinestesico"]
_SCORE_KEYS = {
    "Visual": "visual_score",
    "Auditivo": "auditory_score",
    "Kinestesico": "kinesthetic_score",
}


def classify_by_score(raw: dict) -> dict:
    """Clasifica usando solo los tres puntajes VAK.

    `raw` debe contener visual_score, auditory_score, kinesthetic_score.
    Devuelve el mismo contrato que la Lambda XGBoost.
    """
    scores = {cls: float(raw[_SCORE_KEYS[cls]]) for cls in VAK_CLASSES}
    total = sum(scores.values())

    if total <= 0:
        # Sin senal: reparto uniforme, se elige Visual por defecto.
        confidences = {cls: round(100 / len(VAK_CLASSES), 2) for cls in VAK_CLASSES}
        return {
            "estilo_predominante": "Visual",
            "confianza": confidences,
            "confianza_predominante": confidences["Visual"],
            "clasificador_tipo": "puntaje_simple",
        }

    confidences = {cls: round(scores[cls] / total * 100, 2) for cls in VAK_CLASSES}
    predominante = max(scores, key=scores.get)
    return {
        "estilo_predominante": predominante,
        "confianza": confidences,
        "confianza_predominante": confidences[predominante],
        "clasificador_tipo": "puntaje_simple",
    }


if __name__ == "__main__":
    ejemplo = {"visual_score": 12, "auditory_score": 3, "kinesthetic_score": 5}
    import json

    print(json.dumps(classify_by_score(ejemplo), indent=2, ensure_ascii=False))
