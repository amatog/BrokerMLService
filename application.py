#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
application.py

WSGI-Entry-Point für den ML-Service.

Endpunkte:
  GET  /health             → einfacher Status
  POST /train-risk-model   → startet Training, gibt JSON-Zusammenfassung zurück

Dieser Service nutzt die Trainingslogik aus train_risk_model.py.
"""

import json
from datetime import datetime

from train_risk_model import train_risk_model  # train_risk_model.py muss im gleichen Verzeichnis liegen


def json_response(start_response, status_code: int, payload: dict):
    body = json.dumps(payload).encode("utf-8")
    status = f"{status_code} {'OK' if status_code == 200 else ''}".strip()
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))),
    ]
    start_response(status, headers)
    return [body]


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/")
    method = environ.get("REQUEST_METHOD", "").upper()

    # ---- /health (GET) ----
    if path.rstrip("/") == "/health" and method == "GET":
        payload = {
            "status": "ok",
            "service": "ml-risk-service",
            "time": datetime.utcnow().isoformat() + "Z",
        }
        return json_response(start_response, 200, payload)

    # ---- /train-risk-model (POST) ----
    if path.rstrip("/") == "/train-risk-model" and method == "POST":
        try:
            length = int(environ.get("CONTENT_LENGTH", 0) or 0)
            raw_body = environ["wsgi.input"].read(length) if length > 0 else b"{}"
            params = json.loads(raw_body.decode("utf-8") or "{}")

            num_portfolios = int(params.get("num_portfolios", 5000))
            horizon_days = int(params.get("horizon_days", 30))
            years = int(params.get("years", 3))

            symbols = params.get("symbols")
            if symbols is not None and not isinstance(symbols, list):
                raise ValueError("symbols muss eine Liste von Strings sein.")

            # Training starten
            summary = train_risk_model(
                num_portfolios=num_portfolios,
                horizon_days=horizon_days,
                years=years,
                symbols=symbols,
                model_path="risk_model.pkl",
                feature_path="risk_features.pkl",
            )

            return json_response(start_response, 200, summary)

        except Exception as e:
            payload = {
                "status": "error",
                "message": str(e),
            }
            return json_response(start_response, 500, payload)

    # ---- Fallback: 404 ----
    payload = {
        "status": "not_found",
        "path": path,
        "method": method,
    }
    return json_response(start_response, 404, payload)


# Für lokalen Test:
if __name__ == "__main__":
    from wsgiref.simple_server import make_server

    print("Starte lokalen ML-Service auf http://localhost:8000 ...")
    with make_server("", 8000, application) as server:
        server.serve_forever()
