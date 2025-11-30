# db.py
import os
from dotenv import load_dotenv
load_dotenv()

import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    # Beispiel: postgres://user:password@localhost:5432/mybroker
    raise RuntimeError("DATABASE_URL environment variable is not set")

def get_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def fetch_portfolio_positions(portfolio_id: int):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT p.id,
                       p.symbol,
                       p.quantity,
                       p.avg_price,
                       p.last_price,
                       p.sector
                FROM portfolio_positions p
                WHERE p.portfolio_id = %s
                """,
                (portfolio_id,)
            )
            rows = cur.fetchall()
            return list(rows)
    finally:
        conn.close()

def save_risk_metrics(portfolio_id: int, metrics: dict):
    """
    Optional: schreibt die berechneten Risiko-Metriken in die Tabelle portfolio_risk_metrics.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO portfolio_risk_metrics
                (portfolio_id, total_value, num_positions, concentration,
                 risk_score, risk_level, explanation)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    portfolio_id,
                    metrics["total_value"],
                    metrics["num_positions"],
                    metrics["concentration"],
                    metrics["risk_score"],
                    metrics["risk_level"],
                    metrics["explanation"],
                )
            )
            conn.commit()
    finally:
        conn.close()
