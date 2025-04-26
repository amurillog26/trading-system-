import os
import json
import logging
import datetime as dt
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import psycopg2
from psycopg2.extras import RealDictCursor
import requests
import time
from functools import wraps

# Configuración de logging
logging.basicConfig(
    level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('dashboard')

# Configuración de base de datos
DB_CONFIG = {
    'host': os.environ.get('POSTGRES_HOST', 'postgres'),
    'port': os.environ.get('POSTGRES_PORT', '5432'),
    'database': os.environ.get('POSTGRES_DB', 'trading'),
    'user': os.environ.get('POSTGRES_USER', 'postgres'),
    'password': os.environ.get('POSTGRES_PASSWORD', 'postgres')
}

# Configuración de servicios
IBKR_SERVICE_URL = os.environ.get('IBKR_SERVICE_URL', 'http://ibkr-connector:8080/api')
RISK_SERVICE_URL = os.environ.get('RISK_SERVICE_URL', 'http://risk-manager:8080/api')

# Cargar configuración
def load_config():
    config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'market_data.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            return {
                "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"],
                "timeframes": ["1d", "1h"]
            }
    except Exception as e:
        logger.error(f"Error al cargar configuración: {str(e)}")
        return {
            "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"],
            "timeframes": ["1d", "1h"]
        }

# Conexión a base de datos
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Error al conectar a base de datos: {str(e)}")
        return None

# Función para medir tiempo de ejecución
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Función {func.__name__} ejecutada en {end_time - start_time:.4f} segundos")
        return result
    return wrapper

# Funciones para obtener datos
@timing_decorator
def get_account_status():
    try:
        # Intentar primero desde IBKR
        response = requests.get(f"{IBKR_SERVICE_URL}/account", timeout=2)
        if response.status_code == 200:
            return response.json()
        
        # Si falla, obtener desde base de datos
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT * FROM account_status 
            ORDER BY timestamp DESC 
            LIMIT 1;
            """
            cursor.execute(query)
            data = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if data:
                return {
                    "connected": True,
                    "account": {
                        "NetLiquidation": {"USD": data["equity"]},
                        "TotalCashValue": {"USD": data["account_balance"]},
                        "MaintMarginReq": {"USD": data["margin_used"]},
                        "AvailableFunds": {"USD": data["free_margin"]}
                    },
                    "timestamp": data["timestamp"].isoformat()
                }
        
        return {"connected": False, "error": "No se pudo obtener estado de cuenta"}
    except Exception as e:
        logger.error(f"Error al obtener estado de cuenta: {str(e)}")
        return {"connected": False, "error": str(e)}

@timing_decorator
def get_open_positions():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT * FROM trading_positions
            ORDER BY entry_time DESC;
            """
            cursor.execute(query)
            positions = cursor.fetchall()
            cursor.close()
            conn.close()
            return positions
        return []
    except Exception as e:
        logger.error(f"Error al obtener posiciones abiertas: {str(e)}")
        return []

@timing_decorator
def get_trading_history(limit=50):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = f"""
            SELECT * FROM trading_history
            ORDER BY exit_time DESC
            LIMIT {limit};
            """
            cursor.execute(query)
            history = cursor.fetchall()
            cursor.close()
            conn.close()
            return history
        return []
    except Exception as e:
        logger.error(f"Error al obtener historial de trading: {str(e)}")
        return []

@timing_decorator
def get_recent_patterns(days=7, min_confidence=0.7):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT * FROM detected_patterns
            WHERE timestamp >= %s
            AND confidence >= %s
            ORDER BY timestamp DESC;
            """
            start_date = dt.datetime.now() - dt.timedelta(days=days)
            cursor.execute(query, (start_date, min_confidence))
            patterns = cursor.fetchall()
            cursor.close()
            conn.close()
            return patterns
        return []
    except Exception as e:
        logger.error(f"Error al obtener patrones recientes: {str(e)}")
        return []

@timing_decorator
def get_risk_status():
    try:
        # Intentar primero desde servicio de riesgo
        response = requests.get(f"{RISK_SERVICE_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        
        # Si falla, obtener desde base de datos
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT * FROM risk_status 
            ORDER BY timestamp DESC 
            LIMIT 1;
            """
            cursor.execute(query)
            data = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if data:
                return {
                    "risk_level": data["risk_level"],
                    "total_exposure": data["total_exposure"],
                    "exposure_percentage": data["exposure_percentage"],
                    "max_drawdown": data["max_drawdown"],
                    "num_positions": data["num_positions"],
                    "sector_exposure": data["sector_exposure"] if data["sector_exposure"] else {},
                    "timestamp": data["timestamp"].isoformat()
                }
        
        return {"risk_level": "unknown", "error": "No se pudo obtener estado de riesgo"}
    except Exception as e:
        logger.error(f"Error al obtener estado de riesgo: {str(e)}")
        return {"risk_level": "unknown", "error": str(e)}

@timing_decorator
def get_market_data(symbol, timeframe='1d', limit=100):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = """
            SELECT * FROM market_data
            WHERE symbol = %s AND timeframe = %s
            ORDER BY date DESC
            LIMIT %s;
            """
            cursor.execute(query, (symbol, timeframe, limit))
            data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            # Convertir a dataframe
            if data:
                df = pd.DataFrame(data)
                df = df.sort_values('date')
                return df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error al obtener datos de mercado: {str(e)}")
        return pd.DataFrame()

@timing_decorator
def get_performance_metrics():
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            # Obtener métricas de rendimiento
            query = """
            SELECT * FROM performance_summary;
            """
            cursor.execute(query)
            performance = cursor.fetchall()
            
            # Obtener rendimiento por patrón
            query = """
            SELECT * FROM pattern_performance;
            """
            cursor.execute(query)
            pattern_perf = cursor.fetchall()
            
            # Obtener historial de equity
            query = """
            SELECT timestamp, equity FROM account_status
            ORDER BY timestamp ASC;
            """
            cursor.execute(query)
            equity_history = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            return {
                "performance": performance,
                "pattern_performance": pattern_perf,
                "equity_history": equity_history
            }
        return {}
    except Exception as e:
        logger.error(f"Error al obtener métricas de rendimiento: {str(e)}")
        return {}

@timing_decorator
def get_recent_alerts(limit=20):
    try:
        conn = get_db_connection()
        if conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            query = f"""
            SELECT * FROM alerts
            ORDER BY timestamp DESC
            LIMIT {limit};
            """
            cursor.execute(query)
            alerts = cursor.fetchall()
            cursor.close()
            conn.close()
            return alerts
        return []
    except Exception as e:
        logger.error(f"Error al obtener alertas recientes: {str(e)}")
        return []

@timing_decorator
def get_ibkr_connection_status():
    try:
        response = requests.get(f"{IBKR_SERVICE_URL}/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return {"connected": False, "error": "No se pudo conectar al servicio IBKR"}
    except Exception as e:
        logger.error(f"Error al obtener estado de conexión IBKR: {str(e)}")
        return {"connected": False, "error": str(e)}

# Función para crear gráfico de velas
def create_candlestick_chart(df, symbol, timeframe):
    if df.empty:
        return go.Figure().update_layout(title=f"No hay datos para {symbol} - {timeframe}")
    
    # Crear figura
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                      vertical_spacing=0.03, subplot_titles=(f'{symbol} - {timeframe}', 'Volumen'),
                      row_heights=[0.7, 0.3])
    
    # Añadir gráfico de velas
    fig.add_trace(
        go.Candlestick(
            x=df['date'],
            open=df['open'], 
            high=df['high'],
            low=df['low'], 
            close=df['close'],
            name=symbol
        ),
        row=1, col=1
    )
    
    # Añadir volumen
    fig.add_trace(
        go.Bar(
            x=df['date'],
            y=df['volume'],
            name='Volumen',
            marker=dict(color='rgba(58, 71, 80, 0.6)')
        ),
        row=2, col=1
    )
    
    # Actualizar diseño
    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=600,
        margin=dict(l=50, r=50, t=50, b=50),
        template='plotly_dark'
    )
    
    return fig

# Inicializar app Dash
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
server = app.server

# Cargar configuración
config = load_config()
symbols = config.get('symbols', ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"])
timeframes = config.get('timeframes', ["1d", "1h"])

# Layout principal
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Sistema de Trading Algorítmico", className="text-center my-4"),
            html.Div(id='connection-status')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Estado de Cuenta", className="text-white font-weight-bold"),
                dbc.CardBody(id="account-summary")
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Estado de Riesgo", className="text-white font-weight-bold"),
                dbc.CardBody(id="risk-summary")
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rendimiento", className="text-white font-weight-bold"),
                dbc.CardBody(id="performance-summary")
            ], className="mb-4")
        ], width=4)
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Mercado", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Datos de Mercado", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Símbolo:"),
                            dcc.Dropdown(
                                id='market-symbol-dropdown',
                                options=[{'label': s, 'value': s} for s in symbols],
                                value=symbols[0] if symbols else None,
                                className="mb-3"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Timeframe:"),
                            dcc.Dropdown(
                                id='market-timeframe-dropdown',
                                options=[{'label': t, 'value': t} for t in timeframes],
                                value=timeframes[0] if timeframes else None,
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    dcc.Graph(id='market-chart'),
                    dbc.Button("Actualizar", id='refresh-market-button', color="primary", className="mt-2")
                ])
            ])
        ]),
        
        dbc.Tab(label="Posiciones", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Posiciones Abiertas", className="mt-3"),
                    html.Div(id='open-positions'),
                    dbc.Button("Actualizar", id='refresh-positions-button', color="primary", className="mt-2")
                ], width=12)
            ])
        ]),
        
        dbc.Tab(label="Historial", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Historial de Operaciones", className="mt-3"),
                    html.Div(id='trading-history'),
                    dbc.Button("Actualizar", id='refresh-history-button', color="primary", className="mt-2")
                ], width=12)
            ])
        ]),
        
        dbc.Tab(label="Patrones", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Patrones Detectados", className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Días atrás:"),
                            dcc.Slider(
                                id='patterns-days-slider',
                                min=1,
                                max=30,
                                step=1,
                                value=7,
                                marks={i: str(i) for i in range(0, 31, 5)},
                                className="mb-3"
                            )
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Confianza mínima:"),
                            dcc.Slider(
                                id='patterns-confidence-slider',
                                min=0.5,
                                max=1.0,
                                step=0.05,
                                value=0.7,
                                marks={i/10: str(i/10) for i in range(5, 11)},
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    html.Div(id='detected-patterns'),
                    dbc.Button("Actualizar", id='refresh-patterns-button', color="primary", className="mt-2")
                ], width=12)
            ])
        ]),
        
        dbc.Tab(label="Rendimiento", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Métricas de Rendimiento", className="mt-3"),
                    dcc.Graph(id='equity-curve'),
                    html.Div(id='performance-metrics'),
                    dbc.Button("Actualizar", id='refresh-performance-button', color="primary", className="mt-2")
                ], width=12)
            ])
        ]),
        
        dbc.Tab(label="Alertas", children=[
            dbc.Row([
                dbc.Col([
                    html.H4("Alertas Recientes", className="mt-3"),
                    html.Div(id='recent-alerts'),
                    dbc.Button("Actualizar", id='refresh-alerts-button', color="primary", className="mt-2")
                ], width=12)
            ])
        ])
    ]),
    
    dcc.Interval(
        id='interval-component',
        interval=60 * 1000,  # 60 segundos
        n_intervals=0
    ),
    
    html.Footer([
        html.P("Trading System Dashboard © 2025", className="text-center text-muted mt-4")
    ])
], fluid=True)

# Callbacks
@app.callback(
    Output('connection-status', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_connection_status(n):
    status = get_ibkr_connection_status()
    if status.get('connected', False):
        return html.Div([
            html.I(className="fas fa-check-circle mr-2", style={"color": "green"}),
            html.Span("Conectado a Interactive Brokers", className="text-success")
        ], className="mb-3")
    else:
        return html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "red"}),
            html.Span("Desconectado de Interactive Brokers", className="text-danger"),
            html.Div(f"Error: {status.get('error', 'Desconocido')}", className="text-muted small")
        ], className="mb-3")

@app.callback(
    Output('account-summary', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_account_summary(n):
    account = get_account_status()
    if account.get('connected', False) and 'account' in account:
        account_data = account['account']
        
        # Extraer valores
        equity = account_data.get('NetLiquidation', {}).get('USD', 0)
        balance = account_data.get('TotalCashValue', {}).get('USD', 0)
        margin = account_data.get('MaintMarginReq', {}).get('USD', 0)
        available = account_data.get('AvailableFunds', {}).get('USD', 0)
        
        return html.Div([
            html.H4(f"${equity:,.2f}", className="text-center text-success"),
            html.Hr(),
            html.Div([
                html.Div("Balance:", className="font-weight-bold"),
                html.Div(f"${balance:,.2f}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Margen Usado:", className="font-weight-bold"),
                html.Div(f"${margin:,.2f}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Fondos Disponibles:", className="font-weight-bold"),
                html.Div(f"${available:,.2f}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Actualizado:", className="font-weight-bold small text-muted mt-3"),
                html.Div(dt.datetime.fromisoformat(account['timestamp']).strftime("%H:%M:%S"), 
                         className="small text-muted mt-3")
            ], className="d-flex justify-content-between")
        ])
    else:
        return html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "orange"}),
            html.Span("No se pudo obtener información de cuenta", className="text-warning")
        ])

@app.callback(
    Output('risk-summary', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_risk_summary(n):
    risk = get_risk_status()
    
    if 'error' not in risk:
        # Colores según nivel de riesgo
        risk_colors = {
            'low': 'success',
            'medium': 'warning',
            'high': 'danger',
            'unknown': 'secondary'
        }
        risk_level = risk.get('risk_level', 'unknown').lower()
        risk_color = risk_colors.get(risk_level, 'secondary')
        
        return html.Div([
            html.H4(risk_level.upper(), className=f"text-center text-{risk_color}"),
            html.Hr(),
            html.Div([
                html.Div("Exposición Total:", className="font-weight-bold"),
                html.Div(f"${risk.get('total_exposure', 0):,.2f}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Exposición (%):", className="font-weight-bold"),
                html.Div(f"{risk.get('exposure_percentage', 0):.2f}%")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Drawdown Máximo:", className="font-weight-bold"),
                html.Div(f"{risk.get('max_drawdown', 0):.2f}%")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Posiciones Abiertas:", className="font-weight-bold"),
                html.Div(f"{risk.get('num_positions', 0)}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Actualizado:", className="font-weight-bold small text-muted mt-3"),
                html.Div(dt.datetime.fromisoformat(risk['timestamp']).strftime("%H:%M:%S"), 
                         className="small text-muted mt-3") if 'timestamp' in risk else ""
            ], className="d-flex justify-content-between")
        ])
    else:
        return html.Div([
            html.I(className="fas fa-exclamation-circle mr-2", style={"color": "orange"}),
            html.Span("No se pudo obtener información de riesgo", className="text-warning"),
            html.Div(f"Error: {risk.get('error', 'Desconocido')}", className="text-muted small")
        ])

@app.callback(
    Output('performance-summary', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_performance_summary(n):
    history = get_trading_history(limit=100)
    
    if history:
        # Calcular métricas básicas
        total_trades = len(history)
        winning_trades = sum(1 for trade in history if trade.get('profit_loss', 0) > 0)
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calcular P&L
        total_profit = sum(trade.get('profit_loss', 0) for trade in history if trade.get('profit_loss', 0) > 0)
        total_loss = sum(trade.get('profit_loss', 0) for trade in history if trade.get('profit_loss', 0) <= 0)
        net_profit = total_profit + total_loss
        
        return html.Div([
            html.H4(f"${net_profit:,.2f}", 
                    className=f"text-center {'text-success' if net_profit >= 0 else 'text-danger'}"),
            html.Hr(),
            html.Div([
                html.Div("Operaciones Totales:", className="font-weight-bold"),
                html.Div(f"{total_trades}")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Win Rate:", className="font-weight-bold"),
                html.Div(f"{win_rate:.2f}%")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Ganancia Total:", className="font-weight-bold"),
                html.Div(f"${total_profit:,.2f}", className="text-success")
            ], className="d-flex justify-content-between"),
            html.Div([
                html.Div("Pérdida Total:", className="font-weight-bold"),
                html.Div(f"${total_loss:,.2f}", className="text-danger")
            ], className="d-flex justify-content-between")
        ])
    else:
        return html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span("No hay operaciones para mostrar", className="text-info")
        ])

@app.callback(
    Output('market-chart', 'figure'),
    [Input('refresh-market-button', 'n_clicks'),
     Input('market-symbol-dropdown', 'value'),
     Input('market-timeframe-dropdown', 'value')]
)
def update_market_chart(n_clicks, symbol, timeframe):
    if not symbol or not timeframe:
        return go.Figure().update_layout(title="Selecciona símbolo y timeframe")
    
    df = get_market_data(symbol, timeframe)
    return create_candlestick_chart(df, symbol, timeframe)

@app.callback(
    Output('open-positions', 'children'),
    [Input('refresh-positions-button', 'n_clicks'),
     Input('interval-component', 'n_intervals')]
)
def update_open_positions(n_clicks, n_intervals):
    positions = get_open_positions()
    
    if positions:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Símbolo"),
                html.Th("Dirección"),
                html.Th("Cantidad"),
                html.Th("Precio Entrada"),
                html.Th("Precio Actual"),
                html.Th("P&L"),
                html.Th("Stop Loss"),
                html.Th("Fecha Entrada")
            ]))
        ]
        
        rows = []
        for pos in positions:
            # Calcular P&L
            entry_price = pos.get('entry_price', 0)
            current_price = pos.get('current_price', entry_price)
            quantity = pos.get('quantity', 0)
            direction = pos.get('direction', 'long')
            
            if direction == 'long':
                pnl = (current_price - entry_price) * quantity
                pnl_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
            else:  # short
                pnl = (entry_price - current_price) * quantity
                pnl_pct = ((entry_price / current_price) - 1) * 100 if current_price > 0 else 0
            
            # Formatear fecha
            entry_date = pos.get('entry_time')
            if entry_date:
                entry_date = entry_date.strftime("%Y-%m-%d %H:%M")
            
            # Color según P&L
            pnl_color = "text-success" if pnl >= 0 else "text-danger"
            
            rows.append(html.Tr([
                html.Td(pos.get('symbol', '')),
                html.Td(direction.capitalize()),
                html.Td(f"{pos.get('quantity', 0):,}"),
                html.Td(f"${entry_price:,.2f}"),
                html.Td(f"${current_price:,.2f}"),
                html.Td([
                    html.Div(f"${pnl:,.2f}", className=pnl_color),
                    html.Div(f"({pnl_pct:,.2f}%)", className=f"small {pnl_color}")
                ]),
                html.Td(f"${pos.get('stop_loss', 0):,.2f}" if pos.get('stop_loss') else "-"),
                html.Td(entry_date)
            ]))
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True)
    else:
        return html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span("No hay posiciones abiertas", className="text-info")
        ], className="text-center my-4")

@app.callback(
    Output('trading-history', 'children'),
    [Input('refresh-history-button', 'n_clicks')]
)
def update_trading_history(n_clicks):
    history = get_trading_history(limit=50)
    
    if history:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Símbolo"),
                html.Th("Dirección"),
                html.Th("Cantidad"),
                html.Th("Entrada"),
                html.Th("Salida"),
                html.Th("P&L"),
                html.Th("Duración"),
                html.Th("Motivo")
            ]))
        ]
        
        rows = []
        for trade in history:
            # Calcular duración
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            duration = "-"
            if entry_time and exit_time:
                duration_seconds = (exit_time - entry_time).total_seconds()
                if duration_seconds < 3600:  # Menos de una hora
                    duration = f"{int(duration_seconds / 60)} min"
                elif duration_seconds < 86400:  # Menos de un día
                    duration = f"{int(duration_seconds / 3600)} horas"
                else:
                    duration = f"{int(duration_seconds / 86400)} días"
            
            # Formatear P&L
            pnl = trade.get('profit_loss', 0)
            pnl_pct = trade.get('profit_loss_pct', 0)
            pnl_color = "text-success" if pnl >= 0 else "text-danger"
            
            rows.append(html.Tr([
                html.Td(trade.get('symbol', '')),
                html.Td(trade.get('direction', '').capitalize()),
                html.Td(f"{trade.get('quantity', 0):,}"),
                html.Td([
                    html.Div(f"${trade.get('entry_price', 0):,.2f}"),
                    html.Div(entry_time.strftime("%Y-%m-%d %H:%M") if entry_time else "-", className="small text-muted")
                ]),
                html.Td([
                    html.Div(f"${trade.get('exit_price', 0):,.2f}"),
                    html.Div(exit_time.strftime("%Y-%m-%d %H:%M") if exit_time else "-", className="small text-muted")
                ]),
                html.Td([
                    html.Div(f"${pnl:,.2f}", className=pnl_color),
                    html.Div(f"({pnl_pct:,.2f}%)", className=f"small {pnl_color}")
                ]),
                html.Td(duration),
                html.Td(trade.get('reason', '-').capitalize())
            ]))
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
    else:
        return html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span("No hay historial de operaciones", className="text-info")
        ], className="text-center my-4")

@app.callback(
    Output('detected-patterns', 'children'),
    [Input('refresh-patterns-button', 'n_clicks'),
     Input('patterns-days-slider', 'value'),
     Input('patterns-confidence-slider', 'value')]
)
def update_detected_patterns(n_clicks, days, confidence):
    patterns = get_recent_patterns(days=days, min_confidence=confidence)
    
    if patterns:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Símbolo"),
                html.Th("Patrón"),
                html.Th("Confianza"),
                html.Th("Fecha"),
                html.Th("Precio"),
                html.Th("Stop Loss"),
                html.Th("Objetivo"),
                html.Th("R/R")
            ]))
        ]
        
        rows = []
        for pattern in patterns:
            # Obtener tipo de patrón y dirección
            pattern_type = pattern.get('pattern_type', '')
            is_bullish = '_bullish' in pattern_type
            direction = 'Alcista' if is_bullish else 'Bajista'
            pattern_name = pattern_type.replace('_bullish', '').replace('_bearish', '').replace('_', ' ').title()
            
            # Formatear fecha
            pattern_date = pattern.get('date')
            date_str = pattern_date.strftime("%Y-%m-%d %H:%M") if pattern_date else "-"
            
            # Calcular ratio riesgo/recompensa
            entry = pattern.get('entry_price', 0)
            stop = pattern.get('stop_loss_price', 0)
            target = pattern.get('target_price', 0)
            risk = abs(entry - stop) if entry > 0 and stop > 0 else 0
            reward = abs(target - entry) if entry > 0 and target > 0 else 0
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0
            
            # Color según dirección
            direction_color = "success" if is_bullish else "danger"
            
            rows.append(html.Tr([
                html.Td(pattern.get('symbol', '')),
                html.Td([
                    html.Div(pattern_name),
                    html.Div(direction, className=f"small text-{direction_color}")
                ]),
                html.Td(f"{pattern.get('confidence', 0):.2f}"),
                html.Td(date_str),
                html.Td(f"${pattern.get('entry_price', 0):,.2f}"),
                html.Td(f"${pattern.get('stop_loss_price', 0):,.2f}"),
                html.Td(f"${pattern.get('target_price', 0):,.2f}"),
                html.Td(f"{rr_ratio:.2f}")
            ]))
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
    else:
        return html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span(f"No hay patrones detectados en los últimos {days} días con confianza >= {confidence}", 
                     className="text-info")
        ], className="text-center my-4")

@app.callback(
    [Output('equity-curve', 'figure'),
     Output('performance-metrics', 'children')],
    [Input('refresh-performance-button', 'n_clicks')]
)
def update_performance_metrics(n_clicks):
    metrics = get_performance_metrics()
    
    # Crear gráfico de equity
    fig = go.Figure()
    
    if metrics and 'equity_history' in metrics and metrics['equity_history']:
        # Convertir a dataframe
        equity_df = pd.DataFrame(metrics['equity_history'])
        
        # Añadir curva de equity
        fig.add_trace(go.Scatter(
            x=equity_df['timestamp'],
            y=equity_df['equity'],
            mode='lines',
            name='Equity',
            line=dict(color='green', width=2)
        ))
        
        # Actualizar diseño
        fig.update_layout(
            title='Curva de Equity',
            xaxis_title='Fecha',
            yaxis_title='Equity ($)',
            template='plotly_dark',
            height=400
        )
    else:
        fig.update_layout(
            title='No hay datos de equity disponibles',
            template='plotly_dark',
            height=400
        )
    
    # Crear tabla de rendimiento por símbolo
    symbol_performance = None
    if metrics and 'performance' in metrics and metrics['performance']:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Símbolo"),
                html.Th("Operaciones"),
                html.Th("Win Rate"),
                html.Th("Ganancia Media"),
                html.Th("Pérdida Media"),
                html.Th("P&L Total"),
                html.Th("Comisiones")
            ]))
        ]
        
        rows = []
        for perf in metrics['performance']:
            # Color según P&L
            pnl = perf.get('total_profit_loss', 0)
            pnl_color = "text-success" if pnl >= 0 else "text-danger"
            
            rows.append(html.Tr([
                html.Td(perf.get('symbol', '')),
                html.Td(f"{perf.get('total_trades', 0)} ({perf.get('winning_trades', 0)}/{perf.get('losing_trades', 0)})"),
                html.Td(f"{perf.get('win_rate', 0):.2f}%"),
                html.Td(f"{perf.get('avg_win_pct', 0):.2f}%"),
                html.Td(f"{perf.get('avg_loss_pct', 0):.2f}%"),
                html.Td(f"${pnl:,.2f}", className=pnl_color),
                html.Td(f"${perf.get('total_commission', 0):,.2f}")
            ]))
        
        table_body = [html.Tbody(rows)]
        
        symbol_performance = dbc.Card([
            dbc.CardHeader("Rendimiento por Símbolo"),
            dbc.CardBody([
                dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
            ])
        ], className="mb-4")
    else:
        symbol_performance = html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span("No hay datos de rendimiento disponibles", className="text-info")
        ], className="text-center my-4")
    
    # Crear tabla de rendimiento por patrón
    pattern_performance = None
    if metrics and 'pattern_performance' in metrics and metrics['pattern_performance']:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Patrón"),
                html.Th("Operaciones"),
                html.Th("Win Rate"),
                html.Th("Ganancia Media"),
                html.Th("P&L Total")
            ]))
        ]
        
        rows = []
        for perf in metrics['pattern_performance']:
            # Formatear nombre de patrón
            pattern_type = perf.get('pattern_type', '')
            pattern_name = pattern_type.replace('_bullish', '').replace('_bearish', '').replace('_', ' ').title()
            
            # Color según P&L
            pnl = perf.get('total_profit_loss', 0)
            pnl_color = "text-success" if pnl >= 0 else "text-danger"
            
            rows.append(html.Tr([
                html.Td(pattern_name),
                html.Td(f"{perf.get('total_trades', 0)} ({perf.get('winning_trades', 0)}/{perf.get('losing_trades', 0)})"),
                html.Td(f"{perf.get('win_rate', 0):.2f}%"),
                html.Td(f"{perf.get('avg_profit_loss_pct', 0):.2f}%"),
                html.Td(f"${pnl:,.2f}", className=pnl_color)
            ]))
        
        table_body = [html.Tbody(rows)]
        
        pattern_performance = dbc.Card([
            dbc.CardHeader("Rendimiento por Patrón"),
            dbc.CardBody([
                dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
            ])
        ])
    else:
        pattern_performance = html.Div()
    
    return fig, html.Div([symbol_performance, pattern_performance])

@app.callback(
    Output('recent-alerts', 'children'),
    [Input('refresh-alerts-button', 'n_clicks')]
)
def update_recent_alerts(n_clicks):
    alerts = get_recent_alerts()
    
    if alerts:
        # Crear tabla
        table_header = [
            html.Thead(html.Tr([
                html.Th("Fecha/Hora"),
                html.Th("Tipo"),
                html.Th("Mensaje")
            ]))
        ]
        
        rows = []
        for alert in alerts:
            # Formatear tipo de alerta
            alert_type = alert.get('alert_type', '')
            
            # Colores según tipo
            type_colors = {
                'price_change': 'info',
                'volume_spike': 'info',
                'pattern': 'success',
                'risk': 'warning',
                'drawdown': 'warning',
                'system': 'danger',
                'order_execution': 'success',
                'order_failed': 'danger'
            }
            type_color = type_colors.get(alert_type, 'secondary')
            
            # Formatear fecha
            alert_time = alert.get('timestamp')
            time_str = alert_time.strftime("%Y-%m-%d %H:%M:%S") if alert_time else "-"
            
            rows.append(html.Tr([
                html.Td(time_str),
                html.Td(html.Span(alert_type.replace('_', ' ').title(), 
                                 className=f"badge badge-{type_color}")),
                html.Td(alert.get('message', ''))
            ]))
        
        table_body = [html.Tbody(rows)]
        
        return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, size="sm")
    else:
        return html.Div([
            html.I(className="fas fa-info-circle mr-2", style={"color": "blue"}),
            html.Span("No hay alertas recientes", className="text-info")
        ], className="text-center my-4")

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8080, debug=False)
