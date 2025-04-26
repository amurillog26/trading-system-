-- Inicialización de la base de datos para el sistema de trading algorítmico

-- Crear tablas principales

-- Tabla para datos de mercado
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    date TIMESTAMP NOT NULL,
    open NUMERIC NOT NULL,
    high NUMERIC NOT NULL,
    low NUMERIC NOT NULL,
    close NUMERIC NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(symbol, timeframe, date)
);

-- Índices para consultas frecuentes
CREATE INDEX IF NOT EXISTS market_data_symbol_timeframe_date_idx ON market_data(symbol, timeframe, date);
CREATE INDEX IF NOT EXISTS market_data_symbol_date_idx ON market_data(symbol, date);

-- Tabla para patrones técnicos detectados
CREATE TABLE IF NOT EXISTS detected_patterns (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    pattern_type VARCHAR(50) NOT NULL,
    confidence NUMERIC NOT NULL,
    date TIMESTAMP NOT NULL,
    entry_price NUMERIC NOT NULL,
    stop_loss_price NUMERIC NOT NULL,
    target_price NUMERIC NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    notified BOOLEAN DEFAULT FALSE,
    triggered BOOLEAN DEFAULT FALSE,
    result VARCHAR(20) DEFAULT NULL, -- 'profit', 'loss', 'pending'
    UNIQUE(symbol, timeframe, pattern_type, date, entry_price)
);

CREATE INDEX IF NOT EXISTS patterns_symbol_date_idx ON detected_patterns(symbol, date);
CREATE INDEX IF NOT EXISTS patterns_confidence_idx ON detected_patterns(confidence);

-- Tabla para órdenes de trading
CREATE TABLE IF NOT EXISTS trading_orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'buy', 'sell'
    order_type VARCHAR(20) NOT NULL, -- 'market', 'limit', 'stop'
    quantity INTEGER NOT NULL,
    price NUMERIC,
    status VARCHAR(20) NOT NULL, -- 'pending', 'filled', 'canceled', 'rejected'
    timestamp TIMESTAMP NOT NULL,
    fill_price NUMERIC,
    fill_time TIMESTAMP,
    commission NUMERIC,
    pattern_id INTEGER REFERENCES detected_patterns(id),
    notes TEXT
);

CREATE INDEX IF NOT EXISTS orders_symbol_idx ON trading_orders(symbol);
CREATE INDEX IF NOT EXISTS orders_timestamp_idx ON trading_orders(timestamp);

-- Tabla para posiciones abiertas
CREATE TABLE IF NOT EXISTS trading_positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'long', 'short'
    quantity INTEGER NOT NULL,
    entry_price NUMERIC NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    stop_loss NUMERIC,
    take_profit NUMERIC,
    current_price NUMERIC,
    last_update TIMESTAMP,
    risk_amount NUMERIC,
    pattern_id INTEGER REFERENCES detected_patterns(id),
    notes TEXT,
    UNIQUE(symbol, direction, entry_time)
);

CREATE INDEX IF NOT EXISTS positions_symbol_idx ON trading_positions(symbol);

-- Tabla para historico de posiciones cerradas
CREATE TABLE IF NOT EXISTS trading_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    direction VARCHAR(10) NOT NULL, -- 'long', 'short'
    quantity INTEGER NOT NULL,
    entry_price NUMERIC NOT NULL,
    entry_time TIMESTAMP NOT NULL,
    exit_price NUMERIC NOT NULL,
    exit_time TIMESTAMP NOT NULL,
    profit_loss NUMERIC NOT NULL,
    profit_loss_pct NUMERIC NOT NULL,
    commission NUMERIC NOT NULL,
    pattern_id INTEGER REFERENCES detected_patterns(id),
    reason VARCHAR(20), -- 'take_profit', 'stop_loss', 'signal', 'manual'
    notes TEXT
);

CREATE INDEX IF NOT EXISTS history_symbol_idx ON trading_history(symbol);
CREATE INDEX IF NOT EXISTS history_entry_time_idx ON trading_history(entry_time);
CREATE INDEX IF NOT EXISTS history_exit_time_idx ON trading_history(exit_time);

-- Tabla para estado de la cuenta
CREATE TABLE IF NOT EXISTS account_status (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    account_balance NUMERIC NOT NULL,
    equity NUMERIC NOT NULL,
    margin_used NUMERIC NOT NULL,
    free_margin NUMERIC NOT NULL,
    margin_level NUMERIC,
    open_positions INTEGER NOT NULL,
    daily_pnl NUMERIC
);

CREATE INDEX IF NOT EXISTS account_timestamp_idx ON account_status(timestamp);

-- Tabla para estado de riesgo
CREATE TABLE IF NOT EXISTS risk_status (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    risk_level VARCHAR(10) NOT NULL, -- 'low', 'medium', 'high'
    total_exposure NUMERIC NOT NULL,
    exposure_percentage NUMERIC NOT NULL,
    max_drawdown NUMERIC NOT NULL,
    num_positions INTEGER NOT NULL,
    sector_exposure JSONB,
    correlation_matrix JSONB
);

CREATE INDEX IF NOT EXISTS risk_timestamp_idx ON risk_status(timestamp);

-- Tabla para alertas generadas
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    alert_type VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    acknowledged BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS alerts_timestamp_idx ON alerts(timestamp);
CREATE INDEX IF NOT EXISTS alerts_type_idx ON alerts(alert_type);

-- Funciones

-- Función para calcular el rendimiento diario
CREATE OR REPLACE FUNCTION calculate_daily_pnl()
RETURNS NUMERIC AS $$
DECLARE
    today_balance NUMERIC;
    yesterday_balance NUMERIC;
    daily_pnl NUMERIC;
BEGIN
    SELECT account_balance INTO today_balance
    FROM account_status
    ORDER BY timestamp DESC
    LIMIT 1;
    
    SELECT account_balance INTO yesterday_balance
    FROM account_status
    WHERE timestamp::date = CURRENT_DATE - INTERVAL '1 day'
    ORDER BY timestamp DESC
    LIMIT 1;
    
    IF yesterday_balance IS NULL OR yesterday_balance = 0 THEN
        daily_pnl := 0;
    ELSE
        daily_pnl := ((today_balance - yesterday_balance) / yesterday_balance) * 100;
    END IF;
    
    RETURN daily_pnl;
END;
$$ LANGUAGE plpgsql;

-- Vista para análisis de rendimiento
CREATE OR REPLACE VIEW performance_summary AS
SELECT
    symbol,
    COUNT(*) AS total_trades,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) AS losing_trades,
    ROUND(SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(*)::NUMERIC * 100, 2) AS win_rate,
    ROUND(AVG(profit_loss_pct), 2) AS avg_profit_loss_pct,
    ROUND(AVG(CASE WHEN profit_loss > 0 THEN profit_loss_pct ELSE NULL END), 2) AS avg_win_pct,
    ROUND(AVG(CASE WHEN profit_loss <= 0 THEN profit_loss_pct ELSE NULL END), 2) AS avg_loss_pct,
    ROUND(SUM(profit_loss), 2) AS total_profit_loss,
    ROUND(SUM(commission), 2) AS total_commission,
    MIN(entry_time) AS first_trade,
    MAX(exit_time) AS last_trade
FROM trading_history
GROUP BY symbol
ORDER BY total_profit_loss DESC;

-- Vista para análisis de patrones
CREATE OR REPLACE VIEW pattern_performance AS
SELECT
    p.pattern_type,
    COUNT(h.*) AS total_trades,
    SUM(CASE WHEN h.profit_loss > 0 THEN 1 ELSE 0 END) AS winning_trades,
    SUM(CASE WHEN h.profit_loss <= 0 THEN 1 ELSE 0 END) AS losing_trades,
    ROUND(SUM(CASE WHEN h.profit_loss > 0 THEN 1 ELSE 0 END)::NUMERIC / COUNT(h.*)::NUMERIC * 100, 2) AS win_rate,
    ROUND(AVG(h.profit_loss_pct), 2) AS avg_profit_loss_pct,
    ROUND(SUM(h.profit_loss), 2) AS total_profit_loss
FROM trading_history h
JOIN detected_patterns p ON h.pattern_id = p.id
GROUP BY p.pattern_type
ORDER BY win_rate DESC;

-- Datos iniciales (opcional)

-- Insertar estados de cuenta iniciales
INSERT INTO account_status 
(timestamp, account_balance, equity, margin_used, free_margin, margin_level, open_positions, daily_pnl)
VALUES 
(NOW(), 3000.00, 3000.00, 0.00, 3000.00, 0.00, 0, 0.00);

-- Insertar estado de riesgo inicial
INSERT INTO risk_status
(timestamp, risk_level, total_exposure, exposure_percentage, max_drawdown, num_positions, sector_exposure, correlation_matrix)
VALUES
(NOW(), 'low', 0.00, 0.00, 0.00, 0, '{"Telecomunicaciones": 0, "Consumo": 0, "Financiero": 0, "Comercio": 0}', '{}');

-- Mensaje de creación exitosa
DO $$
BEGIN
    RAISE NOTICE 'Base de datos inicializada correctamente para el sistema de trading algorítmico';
END
$$;
