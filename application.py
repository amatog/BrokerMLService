"""Bounded WSGI API. Configure ML_ADMIN_TOKEN before enabling training."""
import hmac
import json
import logging
import os
from http import HTTPStatus
from pathlib import Path

MAX_BODY = 65536


def json_response(start_response, status_code, payload):
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    start_response(f"{status_code} {HTTPStatus(status_code).phrase}", [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
    return [body]


def read_body(environ):
    length = int(environ.get("CONTENT_LENGTH") or 0)
    if length < 0 or length > MAX_BODY:
        raise ValueError("Request body exceeds limit")
    body = json.loads(environ["wsgi.input"].read(length) or b"{}")
    if not isinstance(body, dict):
        raise ValueError("JSON object required")
    return body


def application(environ, start_response):
    path = environ.get("PATH_INFO", "/").rstrip("/")
    method = environ.get("REQUEST_METHOD", "GET").upper()
    if path == "/health" and method == "GET":
        return json_response(start_response, 200, {
            "status": "ok", "service": "ml-risk-service",
            "capabilities": {"risk": "heuristic", "trend": False,
                             "training": bool(os.getenv("ML_ADMIN_TOKEN"))}})
    try:
        if path == "/risk" and method == "POST":
            from models import PortfolioRiskRequest
            from risk_engine import compute_risk_from_positions
            req = PortfolioRiskRequest(**read_body(environ))
            if req.positions is None:
                raise ValueError("positions must be supplied; database inference is not enabled")
            result = compute_risk_from_positions(req.positions)
            result.update(portfolio_id=req.portfolio_id, method="heuristic")
            return json_response(start_response, 200, result)
        if path == "/trend" and method == "GET":
            return json_response(start_response, 501, {
                "status": "unavailable", "message": "Trend inference is not implemented"})
        if path == "/train-risk-model" and method == "POST":
            expected = os.getenv("ML_ADMIN_TOKEN", "")
            if not expected:
                return json_response(start_response, 503, {"message": "Training is disabled"})
            supplied = environ.get("HTTP_AUTHORIZATION", "")
            if not hmac.compare_digest(supplied.encode(), ("Bearer " + expected).encode()):
                return json_response(start_response, 403, {"message": "Forbidden"})
            from train_risk_model import train_risk_model, validate_training_params
            params = read_body(environ)
            count = params.get("num_portfolios", 5000)
            horizon = params.get("horizon_days", 30)
            years = params.get("years", 3)
            symbols = params.get("symbols")
            validate_training_params(count, horizon, years, symbols)
            root = Path(os.getenv("ML_ARTIFACT_DIR", "artifacts"))
            root.mkdir(parents=True, exist_ok=True)
            lock = root / "training.lock"
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                return json_response(start_response, 409, {"message": "Training is already running"})
            try:
                os.close(fd)
                result = train_risk_model(count, horizon, years, symbols,
                                          str(root / "risk_model.pkl"), str(root / "risk_features.pkl"))
            finally:
                lock.unlink(missing_ok=True)
            return json_response(start_response, 200, result)
    except (ValueError, TypeError) as exc:
        return json_response(start_response, 400, {"status": "error", "message": str(exc)})
    except Exception:
        logging.getLogger(__name__).error("ML request failed", exc_info=False)
        return json_response(start_response, 500, {"status": "error", "message": "Service operation failed"})
    return json_response(start_response, 404, {"status": "not_found"})
