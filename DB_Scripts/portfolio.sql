-- 1. Tabelle für Portfolios
CREATE TABLE portfolios (
                            id              SERIAL PRIMARY KEY,
                            user_id         VARCHAR(100) NOT NULL,
                            name            VARCHAR(255) NOT NULL,
                            base_currency   VARCHAR(10)  NOT NULL DEFAULT 'USD',
                            created_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

-- 2. Tabelle für Positionen im Portfolio
CREATE TABLE portfolio_positions (
                                     id              SERIAL PRIMARY KEY,
                                     portfolio_id    INTEGER      NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
                                     symbol          VARCHAR(20)  NOT NULL,
                                     quantity        NUMERIC(20,4) NOT NULL,
                                     avg_price       NUMERIC(20,4) NOT NULL,
                                     last_price      NUMERIC(20,4) NOT NULL,
                                     sector          VARCHAR(100),
                                     updated_at      TIMESTAMP    NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_portfolio_positions_portfolio
    ON portfolio_positions (portfolio_id);

-- 3. Optionale Tabelle: zuletzt berechnete Risiko-Metriken cachen
CREATE TABLE portfolio_risk_metrics (
                                        id                  SERIAL PRIMARY KEY,
                                        portfolio_id        INTEGER NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
                                        as_of               TIMESTAMP NOT NULL DEFAULT NOW(),
                                        total_value         NUMERIC(20,4) NOT NULL,
                                        num_positions       INTEGER NOT NULL,
                                        concentration       NUMERIC(10,4) NOT NULL,
                                        risk_score          INTEGER NOT NULL,
                                        risk_level          VARCHAR(20) NOT NULL,
                                        explanation         TEXT
);

CREATE INDEX idx_risk_metrics_portfolio_asof
    ON portfolio_risk_metrics (portfolio_id, as_of DESC);
