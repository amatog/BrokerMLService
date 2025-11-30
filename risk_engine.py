# risk_engine.py
from typing import List, Dict
from models import Position

def compute_risk_from_positions(positions: List[Position]) -> Dict:
    if not positions:
        return {
            "total_value": 0.0,
            "num_positions": 0,
            "concentration": 0.0,
            "risk_score": 0,
            "risk_level": "UNKNOWN",
            "explanation": "Keine Positionen im Portfolio vorhanden."
        }

    values = [p.quantity * p.last_price for p in positions]
    total_value = sum(values)

    if total_value <= 0:
        return {
            "total_value": 0.0,
            "num_positions": len(positions),
            "concentration": 0.0,
            "risk_score": 0,
            "risk_level": "UNKNOWN",
            "explanation": "Gesamtwert des Portfolios ist 0 oder negativ."
        }

    weights = [v / total_value for v in values]
    max_weight = max(weights)
    num_positions = len(positions)

    # Heuristische Bewertung:
    # - hohe Konzentration = höheres Risiko
    # - wenige Positionen = höheres Risiko
    # - viele Positionen & niedrige Konzentration = niedrigeres Risiko

    risk_score = 50  # Baseline

    # Konzentrationsrisiko
    if max_weight > 0.5:
        risk_score += 30
    elif max_weight > 0.35:
        risk_score += 20
    elif max_weight > 0.2:
        risk_score += 10
    else:
        risk_score -= 10

    # Diversifikation (Anzahl Positionen)
    if num_positions < 5:
        risk_score += 20
    elif num_positions < 10:
        risk_score += 10
    elif num_positions > 20:
        risk_score -= 10

    # Score innerhalb 0–100 clampen
    risk_score = max(0, min(100, risk_score))

    if risk_score >= 75:
        risk_level = "HIGH"
    elif risk_score >= 40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    explanation = (
        f"Portfolio hat {num_positions} Position(en) mit einem Gesamtwert von {total_value:.2f}. "
        f"Die größte Einzelposition macht {max_weight:.1%} des Portfoliowertes aus. "
        f"Daraus wird ein modellbasierter Risiko-Score von {risk_score} (0–100) abgeleitet "
        f"und als Risikoniveau '{risk_level}' klassifiziert. "
        "Hinweis: Modellbasierte Risiko-Einschätzung, keine Anlageberatung."
    )

    return {
        "total_value": total_value,
        "num_positions": num_positions,
        "concentration": max_weight,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "explanation": explanation,
    }
