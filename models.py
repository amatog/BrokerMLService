# models.py
from typing import List, Optional
from pydantic import BaseModel, Field

class Position(BaseModel):
    symbol: str
    quantity: float = Field(gt=0, allow_inf_nan=False)
    last_price: float = Field(gt=0, allow_inf_nan=False)
    sector: Optional[str] = None

class PortfolioRiskRequest(BaseModel):
    """
    Variante A:
    - Java schickt nur portfolio_id -> Service liest Positionen aus DB
    Variante B:
    - Java schickt direkt eine Liste von Positionen (z.B. aus Alpaca-API)
    Beide Varianten werden unterstützt.
    """
    portfolio_id: Optional[int] = Field(
        default=None,
        description="ID des Portfolios in der DB"
    )
    positions: Optional[List[Position]] = Field(
        default=None,
        max_length=1000,
        description="Optional: Positionen direkt mitliefern, falls keine DB genutzt wird"
    )

class PortfolioRiskResponse(BaseModel):
    portfolio_id: Optional[int]
    total_value: float
    num_positions: int
    concentration: float
    risk_score: int
    risk_level: str
    explanation: str
