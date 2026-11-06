#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import re
import uuid
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional

import requests
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")


# ---------- Finnhub ----------

def _ensure_api_key() -> str:
    api_key = os.getenv("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("FINNHUB_API_KEY ist nicht gesetzt (Environment Variable).")
    return api_key



def load_prices_from_finnhub(symbol: str, years: int = 3) -> pd.DataFrame:
    api_key = _ensure_api_key()

    end = int(time.time())
    start = int((datetime.now() - timedelta(days=365 * years)).timestamp())

    url = (
        "https://finnhub.io/api/v1/stock/candle"
        f"?symbol={symbol}&resolution=D&from={start}&to={end}&token={api_key}"
    )

    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if data.get("s") != "ok":
        raise RuntimeError(f"Finnhub-Fehler für {symbol}: {data}")

    df = pd.DataFrame({
        "t": data["t"],
        "c": data["c"],
    })

    df["t"] = pd.to_datetime(df["t"], unit="s")
    df.set_index("t", inplace=True)
    df.sort_index(inplace=True)

    return df


def load_multiple_symbols(symbols: List[str], years: int = 3) -> pd.DataFrame:
    frames = {}
    for sym in symbols:
        print(f"[INFO] Lade historische Kurse für {sym} von Finnhub...")
        df = load_prices_from_finnhub(sym, years=years)
        frames[sym] = df["c"]

    prices_df = pd.DataFrame(frames)
    prices_df.dropna(inplace=True)
    print(f"[INFO] Historische Kursmatrix: {prices_df.shape[0]} Tage x {prices_df.shape[1]} Symbole")
    return prices_df


# ---------- zufällige Portfolios ----------

def simulate_random_portfolios(
        symbols: List[str],
        num_portfolios: int = 5000,
        min_positions: int = 3,
        max_positions: Optional[int] = None,
) -> List[Dict[str, float]]:
    if max_positions is None:
        max_positions = min(len(symbols), 12)

    portfolios: List[Dict[str, float]] = []

    for _ in range(num_portfolios):
        k = np.random.randint(min_positions, max_positions + 1)
        chosen = np.random.choice(symbols, size=k, replace=False)

        weights = np.random.random(k)
        weights /= weights.sum()

        portfolio = dict(zip(chosen, weights))
        portfolios.append(portfolio)

    return portfolios


# ---------- Features ----------

def build_feature_vector(portfolio: Dict[str, float], prices_df: pd.DataFrame) -> List[float]:
    symbols = list(portfolio.keys())
    weights = np.array(list(portfolio.values()))

    num_positions = len(weights)
    max_weight = float(weights.max())
    top3 = float(weights[np.argsort(weights)[-3:]].sum()) if num_positions >= 3 else float(weights.sum())

    sub_df = prices_df[symbols]
    returns = sub_df.pct_change().dropna()

    if returns.empty:
        volatility = 0.0
    else:
        port_returns = (returns * weights).sum(axis=1)
        volatility = float(port_returns.std())

    return [num_positions, max_weight, top3, volatility]


# ---------- Drawdown & VaR ----------

def compute_drawdown_and_var(
        portfolio: Dict[str, float],
        prices_df: pd.DataFrame,
        horizon_days: int,
) -> Tuple[float, float]:
    symbols = list(portfolio.keys())
    weights = np.array(list(portfolio.values()))

    sub_df = prices_df[symbols]
    returns = sub_df.pct_change().dropna()

    if returns.empty:
        return 0.0, 0.0

    port_returns = (returns * weights).sum(axis=1)

    if len(port_returns) > horizon_days:
        port_returns = port_returns[-horizon_days:]

    cumulative = (1 + port_returns).cumprod()

    peak = cumulative.cummax().clip(lower=1.0)
    drawdown = (cumulative - peak) / peak
    max_dd = float(abs(drawdown.min()))

    var_5 = float(np.quantile(port_returns, 0.05))
    return max_dd, var_5


# ---------- Label ----------

def drawdown_to_label(max_dd: float) -> str:
    if max_dd < 0.05:
        return "LOW"
    elif max_dd < 0.10:
        return "MEDIUM"
    elif max_dd < 0.20:
        return "HIGH"
    else:
        return "EXTREME"


# ---------- Training ----------

def validate_training_params(num_portfolios, horizon_days, years, symbols):
    for name, value, low, high in (("num_portfolios", num_portfolios, 100, 10000),
                                   ("horizon_days", horizon_days, 1, 126),
                                   ("years", years, 1, 10)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer between {low} and {high}")
    if symbols is not None:
        if (not isinstance(symbols, list) or not 3 <= len(symbols) <= 30
                or any(not isinstance(s, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,19}", s) for s in symbols)
                or len(set(symbols)) != len(symbols)):
            raise ValueError("symbols must contain 3 to 30 distinct ticker symbols")


def build_temporal_dataset(portfolios, prices_df, horizon_days, lookback=63):
    """Past-only features, future labels, and disjoint chronological label windows."""
    anchors = list(range(lookback, len(prices_df) - horizon_days, horizon_days))
    if len(anchors) < 6:
        raise ValueError("Insufficient history for at least six independent label windows")
    cut = max(1, int(len(anchors) * 0.8))
    test_start = anchors[cut]
    features, labels, train_mask = [], [], []
    for i, portfolio in enumerate(portfolios):
        anchor = anchors[min(len(anchors) - 1, i * len(anchors) // len(portfolios))]
        past = prices_df.iloc[anchor - lookback:anchor + 1]
        future = prices_df.iloc[anchor:anchor + horizon_days + 1]
        features.append(build_feature_vector(portfolio, past))
        drawdown, _ = compute_drawdown_and_var(portfolio, future, horizon_days)
        labels.append(drawdown_to_label(drawdown))
        train_mask.append(anchor < test_start)
    mask = np.asarray(train_mask)
    X, y = np.asarray(features), np.asarray(labels)
    return X[mask], X[~mask], y[mask], y[~mask], {
        "method": "chronological_nonoverlapping_label_windows",
        "lookback_days": lookback,
        "train_label_end": str(prices_df.index[anchors[cut - 1] + horizon_days]),
        "test_feature_asof_start": str(prices_df.index[test_start]),
        "train_samples": int(mask.sum()), "test_samples": int((~mask).sum()),
    }

def train_risk_model(
        num_portfolios: int = 5000,
        horizon_days: int = 30,
        years: int = 3,
        symbols: Optional[List[str]] = None,
        model_path: str = "risk_model.pkl",
        feature_path: str = "risk_features.pkl",
) -> Dict:
    validate_training_params(num_portfolios, horizon_days, years, symbols)
    print("[INFO] Starte Training des Risiko-Modells...")
    print(f"[INFO] num_portfolios={num_portfolios}, horizon_days={horizon_days}, years={years}")

    if symbols is None:
        symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA", "NFLX"]

    prices_df = load_multiple_symbols(symbols, years=years)
    portfolios = simulate_random_portfolios(symbols, num_portfolios=num_portfolios)
    print(f"[INFO] Simulierte Portfolios: {len(portfolios)}")

    X_train, X_test, y_train, y_test, evaluation = build_temporal_dataset(
        portfolios, prices_df, horizon_days)
    y_list = list(y_train) + list(y_test)

    feature_names = ["num_positions", "max_weight", "top3_concentration", "volatility"]


    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    print("[INFO] Klassifikationsbericht (Testdaten):")
    print(classification_report(y_test, y_pred))

    # Publish an immutable pair; readers follow the manifest only after both files exist.
    root = Path(model_path).resolve().parent
    run_dir = root / ("run_" + uuid.uuid4().hex)
    run_dir.mkdir(parents=True)
    model_path = str(run_dir / Path(model_path).name)
    feature_path = str(run_dir / Path(feature_path).name)
    joblib.dump(model, model_path)
    joblib.dump(feature_names, feature_path)
    manifest = root / (".current_" + uuid.uuid4().hex + ".json")
    manifest.write_text(json.dumps({"model": model_path, "features": feature_path,
                                    "evaluation": evaluation}), encoding="utf-8")
    os.replace(manifest, root / "current.json")
    print(f"[INFO] Modell gespeichert unter: {os.path.abspath(model_path)}")
    print(f"[INFO] Feature-Namen gespeichert unter: {os.path.abspath(feature_path)}")

    class_counts: Dict[str, int] = {}
    for label in y_list:
        class_counts[label] = class_counts.get(label, 0) + 1

    summary = {
        "status": "ok",
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "num_samples": len(y_list),
        "horizon_days": horizon_days,
        "years": years,
        "symbols": symbols,
        "class_distribution": class_counts,
        "test_report": report,
        "evaluation": evaluation,
        "model_path": os.path.abspath(model_path),
        "feature_path": os.path.abspath(feature_path),
    }

    return summary


if __name__ == "__main__":
    result = train_risk_model(
        num_portfolios=2000,
        horizon_days=30,
        years=3,
        symbols=None,
        model_path="risk_model.pkl",
        feature_path="risk_features.pkl",
    )

    print("\n[RESULT] Trainingszusammenfassung:")
    print(json.dumps(result, indent=2))
