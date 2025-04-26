import os
import json
import time
import logging
import threading
import pandas as pd
import numpy as np
import psycopg2
import datetime as dt
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, request
from typing import Dict, List, Tuple, Union, Optional
from concurrent.futures import ThreadPoolExecutor

class RiskManager:
    def __init__(self, config_path: str = None):
        """
        Inicializa el gestor de riesgos
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None, usa valores por defecto
                        o busca en la ubicación estándar.
        """
        # Configurar logger
        self._setup_logging()
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Conectar con base de datos
        self.db_conn = self._get_db_connection()
        
        # Iniciar hilo de monitoreo de riesgo
        self.monitoring_thread = threading.Thread(target=self._risk_monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        # Crear la aplicación Flask
        self.app = self._create_app()
        
        self.logger.info("Gestor de riesgos inicializado")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('risk_manager')
    
    def _load_config(self, config_path: str = None) -> Dict:
        """
        Carga la configuración desde archivo
        
        Args:
            config_path: Ruta al archivo de configuración
            
        Returns:
            Dict con la configuración
        """
        # Configuración por defecto
        default_config = {
            "max_risk_per_trade": 0.02,  # 2% por operación
            "max_portfolio_risk": 0.06,  # 6% para todo el portafolio
            "max_correlation": 0.7,  # Máxima correlación permitida
            "max_drawdown": 0.15,  # 15% máximo drawdown
            "max_trades_per_day": 3,  # Máximo número de operaciones diarias
            "position_sizing": {
                "method": "risk_based",  # Métodos: fixed, risk_based
                "fixed_amount": 500,  # Monto fijo (si method=fixed)
                "risk_percent": 0.02,  # Porcentaje de riesgo (si method=risk_based)
                "max_capital_percent": 0.3,  # Máximo porcentaje de capital por operación
                "atr_multiplier": 2.0  # Multiplicador de ATR para stop loss
            },
            "hedging": {
                "enabled": False,
                "instruments": {
                    "MXN.USD": {
                        "weight": 0.5,
                        "direction": "inverse"
                    }
                }
            },
            "sector_limits": {
                "Telecomunicaciones": 0.25,
                "Consumo": 0.30,
                "Financiero": 0.25,
                "Comercio Minorista": 0.25,
                "Materiales": 0.20,
                "Otros": 0.10
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "monitoring_interval": 300  # segundos (5 minutos)
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'risk.json')
        
        # Intentar cargar desde archivo si existe
        config = default_config.copy()
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Actualizar configuración con valores del archivo
                    self._deep_update(config, file_config)
                    self.logger.info(f"Configuración cargada desde {config_path}")
            except Exception as e:
                self.logger.error(f"Error al cargar configuración desde archivo: {str(e)}")
                self.logger.info("Usando configuración por defecto")
        else:
            self.logger.warning(f"Archivo de configuración no encontrado: {config_path}")
            self.logger.info("Usando configuración por defecto")
        
        return config
    
    def _deep_update(self, original: Dict, update: Dict) -> None:
        """
        Actualiza recursivamente un diccionario con valores de otro
        
        Args:
            original: Diccionario original a actualizar
            update: Diccionario con nuevos valores
        """
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _get_db_connection(self):
        """Establece conexión con la base de datos PostgreSQL"""
        try:
            conn = psycopg2.connect(
                host=os.environ.get('POSTGRES_HOST', 'postgres'),
                port=os.environ.get('POSTGRES_PORT', '5432'),
                database=os.environ.get('POSTGRES_DB', 'trading'),
                user=os.environ.get('POSTGRES_USER', 'postgres'),
                password=os.environ.get('POSTGRES_PASSWORD', 'postgres')
            )
            self.logger.info("Conexión a base de datos establecida")
            return conn
        except Exception as e:
            self.logger.error(f"Error al conectar a la base de datos: {str(e)}")
            # Reintentar después de un tiempo
            time.sleep(5)
            return self._get_db_connection()
    
    def _risk_monitoring_loop(self) -> None:
        """
        Hilo principal para monitoreo continuo de riesgo
        """
        interval = self.config.get('monitoring_interval', 300)  # Por defecto, cada 5 minutos
        
        while True:
            try:
                # Evaluar riesgo actual
                self.evaluate_risk()
                
                # Esperar hasta el próximo ciclo
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error en ciclo de monitoreo de riesgo: {str(e)}")
                time.sleep(60)  # Esperar un minuto antes de reintentar
    
    def evaluate_risk(self) -> Dict:
        """
        Evalúa el riesgo actual del portafolio
        
        Returns:
            Dict con evaluación de riesgo
        """
        try:
            # 1. Obtener estado de cuenta
            account_info = self._get_account_info()
            
            # 2. Obtener posiciones abiertas
            positions = self._get_open_positions()
            
            # 3. Calcular exposición total y por sector
            exposure_info = self._calculate_exposure(positions, account_info)
            
            # 4. Calcular correlación entre posiciones
            correlation_matrix = self._calculate_correlation(positions)
            
            # 5. Calcular drawdown
            drawdown = self._calculate_drawdown(account_info)
            
            # 6. Determinar nivel de riesgo
            risk_level, risk_factors = self._determine_risk_level(exposure_info, correlation_matrix, drawdown)
            
            # Guardar evaluación en base de datos
            risk_status = {
                'timestamp': dt.datetime.now(),
                'risk_level': risk_level,
                'total_exposure': exposure_info['total_exposure'],
                'exposure_percentage': exposure_info['exposure_percentage'],
                'max_drawdown': drawdown['max_drawdown'],
                'num_positions': len(positions),
                'sector_exposure': json.dumps(exposure_info['sector_exposure']),
                'correlation_matrix': json.dumps(correlation_matrix),
                'risk_factors': risk_factors
            }
            
            self._save_risk_status(risk_status)
            
            self.logger.info(f"Evaluación de riesgo completada: nivel {risk_level}, exposición {exposure_info['exposure_percentage']:.2f}%")
            
            return {
                'risk_level': risk_level,
                'exposure': exposure_info,
                'drawdown': drawdown,
                'correlation': correlation_matrix,
                'risk_factors': risk_factors
            }
        except Exception as e:
            self.logger.error(f"Error al evaluar riesgo: {str(e)}")
            return {
                'error': str(e),
                'risk_level': 'unknown'
            }
    
    def _get_account_info(self) -> Dict:
        """
        Obtiene información de la cuenta desde base de datos
        
        Returns:
            Dict con información de cuenta
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Obtener último estado de cuenta
            query = """
            SELECT * FROM account_status
            ORDER BY timestamp DESC
            LIMIT 1;
            """
            
            cursor.execute(query)
            account = cursor.fetchone()
            cursor.close()
            
            if account:
                return dict(account)
            else:
                self.logger.warning("No se encontró información de cuenta")
                return {
                    'account_balance': 0,
                    'equity': 0,
                    'margin_used': 0,
                    'free_margin': 0,
                    'margin_level': 0,
                    'open_positions': 0,
                    'daily_pnl': 0,
                    'timestamp': dt.datetime.now()
                }
        except Exception as e:
            self.logger.error(f"Error al obtener información de cuenta: {str(e)}")
            return {
                'account_balance': 0,
                'equity': 0,
                'margin_used': 0,
                'free_margin': 0,
                'margin_level': 0,
                'open_positions': 0,
                'daily_pnl': 0,
                'timestamp': dt.datetime.now()
            }
    
    def _get_open_positions(self) -> List[Dict]:
        """
        Obtiene posiciones abiertas desde base de datos
        
        Returns:
            Lista de posiciones abiertas
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Obtener posiciones abiertas
            query = """
            SELECT * FROM trading_positions
            ORDER BY entry_time DESC;
            """
            
            cursor.execute(query)
            positions = cursor.fetchall()
            cursor.close()
            
            return positions if positions else []
        except Exception as e:
            self.logger.error(f"Error al obtener posiciones abiertas: {str(e)}")
            return []
    
    def _calculate_exposure(self, positions: List[Dict], account_info: Dict) -> Dict:
        """
        Calcula la exposición total y por sector
        
        Args:
            positions: Lista de posiciones abiertas
            account_info: Información de cuenta
            
        Returns:
            Dict con información de exposición
        """
        # Obtener saldo de cuenta
        account_balance = account_info.get('account_balance', 0)
        
        # Inicializar exposición por sector
        sector_exposure = {sector: 0 for sector in self.config.get('sector_limits', {}).keys()}
        sector_exposure['Otros'] = 0  # Para símbolos sin sector específico
        
        # Calcular exposición total y por sector
        total_exposure = 0
        for position in positions:
            # Calcular valor de posición
            position_value = position.get('quantity', 0) * position.get('current_price', 0)
            total_exposure += position_value
            
            # Asignar a sector
            symbol = position.get('symbol', '')
            sector = self._get_symbol_sector(symbol)
            
            if sector in sector_exposure:
                sector_exposure[sector] += position_value
            else:
                sector_exposure['Otros'] += position_value
        
        # Calcular porcentaje de exposición
        exposure_percentage = (total_exposure / account_balance * 100) if account_balance > 0 else 0
        
        # Calcular porcentaje por sector
        sector_percentage = {}
        for sector, value in sector_exposure.items():
            sector_percentage[sector] = (value / account_balance * 100) if account_balance > 0 else 0
        
        return {
            'total_exposure': total_exposure,
            'exposure_percentage': exposure_percentage,
            'sector_exposure': sector_exposure,
            'sector_percentage': sector_percentage
        }
    
    def _get_symbol_sector(self, symbol: str) -> str:
        """
        Obtiene el sector al que pertenece un símbolo
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Sector del símbolo
        """
        # Mapeo de símbolos a sectores (podría obtenerse de base de datos o archivo de configuración)
        sector_mapping = {
            "AMXL.MX": "Telecomunicaciones",
            "FEMSAUBD.MX": "Consumo",
            "GFNORTEO.MX": "Financiero",
            "WALMEX.MX": "Comercio Minorista",
            "CEMEXCPO.MX": "Materiales",
            "BIMBOA.MX": "Consumo",
            "KIMBERA.MX": "Consumo",
            "ALSEA.MX": "Consumo",
            "GRUMAB.MX": "Consumo",
            "KOFUBL.MX": "Consumo",
            "GCARSOA1.MX": "Telecomunicaciones"
        }
        
        return sector_mapping.get(symbol, "Otros")
    
    def _calculate_correlation(self, positions: List[Dict]) -> Dict:
        """
        Calcula la matriz de correlación entre posiciones
        
        Args:
            positions: Lista de posiciones abiertas
            
        Returns:
            Dict con matriz de correlación
        """
        if len(positions) < 2:
            return {}
        
        # Obtener símbolos de posiciones abiertas
        symbols = [position.get('symbol', '') for position in positions]
        
        try:
            # Obtener datos históricos para calcular correlación
            price_data = self._get_historical_prices(symbols, days=90)  # 90 días
            
            if not price_data or len(price_data) < 2:
                return {}
            
            # Calcular rendimientos diarios
            returns = {}
            for symbol, prices in price_data.items():
                if len(prices) > 1:
                    returns[symbol] = pd.Series(
                        [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))],
                        index=range(len(prices)-1)
                    )
            
            # Crear dataframe con todos los rendimientos
            df = pd.DataFrame(returns)
            
            # Calcular matriz de correlación
            correlation_matrix = df.corr().to_dict()
            
            # Formato para resultado final
            result = {}
            for symbol1, corrs in correlation_matrix.items():
                result[symbol1] = {}
                for symbol2, corr in corrs.items():
                    if symbol1 != symbol2:
                        result[symbol1][symbol2] = round(corr, 2)
            
            return result
        except Exception as e:
            self.logger.error(f"Error al calcular correlación: {str(e)}")
            return {}
    
    def _get_historical_prices(self, symbols: List[str], days: int = 90) -> Dict[str, List[float]]:
        """
        Obtiene precios históricos para un conjunto de símbolos
        
        Args:
            symbols: Lista de símbolos
            days: Número de días hacia atrás
            
        Returns:
            Dict con precios históricos por símbolo
        """
        if not symbols:
            return {}
        
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            result = {}
            for symbol in symbols:
                # Obtener precios diarios
                query = """
                SELECT date, close FROM market_data
                WHERE symbol = %s AND timeframe = '1d'
                AND date >= %s
                ORDER BY date ASC;
                """
                
                start_date = dt.datetime.now() - dt.timedelta(days=days)
                cursor.execute(query, (symbol, start_date))
                data = cursor.fetchall()
                
                if data:
                    # Extraer precios de cierre
                    result[symbol] = [row['close'] for row in data]
            
            cursor.close()
            return result
        except Exception as e:
            self.logger.error(f"Error al obtener precios históricos: {str(e)}")
            return {}
    
    def _calculate_drawdown(self, account_info: Dict) -> Dict:
        """
        Calcula drawdown actual y máximo
        
        Args:
            account_info: Información de cuenta
            
        Returns:
            Dict con información de drawdown
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Obtener historial de equity
            query = """
            SELECT timestamp, equity FROM account_status
            ORDER BY timestamp ASC;
            """
            
            cursor.execute(query)
            history = cursor.fetchall()
            cursor.close()
            
            if history:
                # Convertir a dataframe
                df = pd.DataFrame(history)
                
                # Calcular peak (máximo histórico)
                df['peak'] = df['equity'].cummax()
                
                # Calcular drawdown
                df['drawdown'] = (df['peak'] - df['equity']) / df['peak'] * 100
                
                # Obtener drawdown actual y máximo
                current_drawdown = df['drawdown'].iloc[-1] if not df.empty else 0
                max_drawdown = df['drawdown'].max() if not df.empty else 0
                max_drawdown_date = df.loc[df['drawdown'].idxmax(), 'timestamp'] if not df.empty and max_drawdown > 0 else None
                
                return {
                    'current_drawdown': round(current_drawdown, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'max_drawdown_date': max_drawdown_date
                }
            else:
                return {
                    'current_drawdown': 0,
                    'max_drawdown': 0,
                    'max_drawdown_date': None
                }
        except Exception as e:
            self.logger.error(f"Error al calcular drawdown: {str(e)}")
            return {
                'current_drawdown': 0,
                'max_drawdown': 0,
                'max_drawdown_date': None
            }
    
    def _determine_risk_level(self, exposure_info: Dict, correlation_matrix: Dict, drawdown: Dict) -> Tuple[str, List[str]]:
        """
        Determina el nivel de riesgo actual
        
        Args:
            exposure_info: Información de exposición
            correlation_matrix: Matriz de correlación
            drawdown: Información de drawdown
            
        Returns:
            Tupla con (nivel_riesgo, factores_riesgo)
        """
        # Inicializar factores de riesgo
        risk_factors = []
        
        # 1. Evaluar exposición total
        exposure_percentage = exposure_info.get('exposure_percentage', 0)
        max_portfolio_risk = self.config.get('max_portfolio_risk', 0.06) * 100  # convertir a porcentaje
        
        if exposure_percentage >= max_portfolio_risk:
            risk_factors.append(f"Exposición total ({exposure_percentage:.2f}%) excede el límite ({max_portfolio_risk:.2f}%)")
        
        # 2. Evaluar exposición por sector
        sector_percentage = exposure_info.get('sector_percentage', {})
        sector_limits = self.config.get('sector_limits', {})
        
        for sector, percentage in sector_percentage.items():
            if sector in sector_limits and percentage > sector_limits[sector] * 100:
                risk_factors.append(f"Exposición en sector {sector} ({percentage:.2f}%) excede el límite ({sector_limits[sector]*100:.2f}%)")
        
        # 3. Evaluar correlación
        max_correlation = self.config.get('max_correlation', 0.7)
        high_correlations = []
        
        for symbol1, corrs in correlation_matrix.items():
            for symbol2, corr in corrs.items():
                if abs(corr) > max_correlation:
                    high_correlations.append((symbol1, symbol2, corr))
        
        if high_correlations:
            symbols = ', '.join([f"{s1}/{s2}={c:.2f}" for s1, s2, c in high_correlations[:3]])
            risk_factors.append(f"Alta correlación entre símbolos: {symbols}")
        
        # 4. Evaluar drawdown
        current_drawdown = drawdown.get('current_drawdown', 0)
        max_allowed_drawdown = self.config.get('max_drawdown', 0.15) * 100  # convertir a porcentaje
        
        if current_drawdown > max_allowed_drawdown:
            risk_factors.append(f"Drawdown actual ({current_drawdown:.2f}%) excede el límite ({max_allowed_drawdown:.2f}%)")
        
        # Determinar nivel de riesgo
        if len(risk_factors) >= 3 or current_drawdown > max_allowed_drawdown:
            risk_level = "high"
        elif len(risk_factors) >= 1 or current_drawdown > max_allowed_drawdown * 0.7:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        return risk_level, risk_factors
    
    def _save_risk_status(self, risk_status: Dict) -> None:
        """
        Guarda el estado de riesgo en base de datos
        
        Args:
            risk_status: Estado de riesgo a guardar
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Insertar en risk_status
            query = """
            INSERT INTO risk_status
            (timestamp, risk_level, total_exposure, exposure_percentage, max_drawdown, num_positions, sector_exposure, correlation_matrix)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            cursor.execute(query, (
                risk_status['timestamp'],
                risk_status['risk_level'],
                risk_status['total_exposure'],
                risk_status['exposure_percentage'],
                risk_status['max_drawdown'],
                risk_status['num_positions'],
                risk_status['sector_exposure'],
                risk_status['correlation_matrix']
            ))
            
            self.db_conn.commit()
            cursor.close()
            
            # Si el riesgo es alto, generar alerta
            if risk_status['risk_level'] == 'high':
                self._generate_risk_alert(risk_status)
        except Exception as e:
            self.logger.error(f"Error al guardar estado de riesgo: {str(e)}")
    
    def _generate_risk_alert(self, risk_status: Dict) -> None:
        """
        Genera una alerta de riesgo
        
        Args:
            risk_status: Estado de riesgo
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Construir mensaje
            message = f"⚠️ ALERTA DE RIESGO: Nivel de riesgo ALTO detectado\n"
            message += f"Exposición total: ${risk_status['total_exposure']:.2f} ({risk_status['exposure_percentage']:.2f}%)\n"
            message += f"Drawdown máximo: {risk_status['max_drawdown']:.2f}%\n"
            message += f"Posiciones abiertas: {risk_status['num_positions']}\n"
            
            if 'risk_factors' in risk_status and risk_status['risk_factors']:
                message += f"Factores de riesgo:\n- " + "\n- ".join(risk_status['risk_factors'])
            
            # Insertar alerta
            query = """
            INSERT INTO alerts
            (timestamp, alert_type, message)
            VALUES (%s, %s, %s);
            """
            
            cursor.execute(query, (
                dt.datetime.now(),
                'risk',
                message
            ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al generar alerta de riesgo: {str(e)}")
    
    def calculate_position_size(self, params: Dict) -> Dict:
        """
        Calcula el tamaño de posición según el modelo de gestión de riesgo
        
        Args:
            params: Parámetros para cálculo
                {
                    "symbol": "AMXL.MX",
                    "entry_price": 15.75,
                    "stop_loss_price": 15.25,
                    "pattern_type": "ma_crossover_bullish",
                    "confidence": 0.82
                }
                
        Returns:
            Dict con tamaño de posición calculado y métricas de riesgo
        """
        try:
            # Validar parámetros
            required_params = ["symbol", "entry_price", "stop_loss_price"]
            for param in required_params:
                if param not in params:
                    return {"error": f"Parámetro requerido no encontrado: {param}"}
            
            # Obtener información de cuenta
            account_info = self._get_account_info()
            account_balance = account_info.get('account_balance', 0)
            
            if account_balance <= 0:
                return {"error": "Balance de cuenta no disponible o cero"}
            
            # Obtener posiciones actuales
            positions = self._get_open_positions()
            
            # Calcular riesgo actual
            risk_assessment = self.evaluate_risk()
            
            # Verificar si se puede abrir nueva posición
            if risk_assessment.get('risk_level') == 'high':
                return {
                    "position_size": 0,
                    "error": "Nivel de riesgo actual es ALTO. No se pueden abrir nuevas posiciones.",
                    "risk_level": "high"
                }
            
            # Extraer parámetros
            symbol = params["symbol"]
            entry_price = params["entry_price"]
            stop_loss_price = params["stop_loss_price"]
            
            # Verificar si ya existe posición para este símbolo
            existing_position = next((p for p in positions if p.get('symbol') == symbol), None)
            if existing_position:
                return {
                    "position_size": 0,
                    "error": f"Ya existe una posición abierta para {symbol}",
                    "existing_position": existing_position
                }
            
            # Calcular riesgo por unidad (diferencia entre entrada y stop loss)
            risk_per_unit = abs(entry_price - stop_loss_price)
            if risk_per_unit <= 0:
                return {"error": "Riesgo por unidad inválido. Entrada y stop loss deben ser diferentes."}
            
            # Determinar método de cálculo
            position_sizing = self.config.get('position_sizing', {})
            method = position_sizing.get('method', 'risk_based')
            
            if method == 'fixed':
                # Método de monto fijo
                fixed_amount = position_sizing.get('fixed_amount', 500)
                position_size = int(fixed_amount / entry_price)
                risk_amount = position_size * risk_per_unit
                
            elif method == 'risk_based':
                # Método basado en porcentaje de riesgo
                risk_percent = position_sizing.get('risk_percent', 0.02)
                max_risk_amount = account_balance * risk_percent
                
                # Ajustar por confianza del patrón
                confidence = params.get('confidence', 0.7)
                adjusted_risk = max_risk_amount * (confidence / 0.7)  # Normalizar contra 0.7
                
                # Calcular tamaño de posición basado en riesgo
                position_size = int(adjusted_risk / risk_per_unit)
                risk_amount = position_size * risk_per_unit
                
                # Verificar límite de porcentaje máximo de capital
                max_capital_percent = position_sizing.get('max_capital_percent', 0.3)
                max_position_size = int(account_balance * max_capital_percent / entry_price)
                
                if position_size > max_position_size:
                    position_size = max_position_size
                    risk_amount = position_size * risk_per_unit
            else:
                return {"error": f"Método de cálculo no soportado: {method}"}
            
            # Calcular porcentaje de riesgo
            risk_percentage = (risk_amount / account_balance) * 100
            
            # Verificar límite de riesgo por operación
            max_risk_per_trade = self.config.get('max_risk_per_trade', 0.02) * 100  # convertir a porcentaje
            if risk_percentage > max_risk_per_trade:
                # Ajustar tamaño para no exceder límite
                position_size = int(position_size * (max_risk_per_trade / risk_percentage))
                risk_amount = position_size * risk_per_unit
                risk_percentage = (risk_amount / account_balance) * 100
            
            # Verificar límite de riesgo de portafolio
            current_portfolio_risk = sum([
                abs(p.get('quantity', 0) * (p.get('entry_price', 0) - p.get('stop_loss', 0)))
                for p in positions if p.get('stop_loss') is not None
            ])
            
            max_portfolio_risk = self.config.get('max_portfolio_risk', 0.06) * account_balance
            
            if (current_portfolio_risk + risk_amount) > max_portfolio_risk:
                # Si excede el límite de portafolio, ajustar o rechazar
                available_risk = max_portfolio_risk - current_portfolio_risk
                if available_risk <= 0:
                    return {
                        "position_size": 0,
                        "error": "Límite de riesgo de portafolio alcanzado",
                        "current_portfolio_risk": current_portfolio_risk,
                        "max_portfolio_risk": max_portfolio_risk
                    }
                
                # Ajustar tamaño
                position_size = int(available_risk / risk_per_unit)
                risk_amount = position_size * risk_per_unit
                risk_percentage = (risk_amount / account_balance) * 100
            
            # Asegurarse de que el tamaño sea al menos 1
            if position_size < 1:
                return {
                    "position_size": 0,
                    "error": "Tamaño calculado menor que 1",
                    "calculated_size": position_size
                }
            
            # Calcular valor total de la posición
            position_value = position_size * entry_price
            position_value_pct = (position_value / account_balance) * 100
            
            return {
                "position_size": position_size,
                "risk_amount": risk_amount,
                "risk_percentage": risk_percentage,
                "position_value": position_value,
                "position_value_percentage": position_value_pct,
                "entry_price": entry_price,
                "stop_loss_price": stop_loss_price,
                "account_balance": account_balance,
                "method": method
            }
        except Exception as e:
            self.logger.error(f"Error al calcular tamaño de posición: {str(e)}")
            return {"error": str(e)}
    
    def _create_app(self) -> Flask:
        """
        Crea y configura la aplicación Flask
        
        Returns:
            Aplicación Flask configurada
        """
        app = Flask(__name__)
        
        # Definir rutas
        @app.route('/api/status', methods=['GET'])
        def get_status():
            risk_assessment = self.evaluate_risk()
            return jsonify(risk_assessment)
        
        @app.route('/api/position-size', methods=['POST'])
        def calculate_position_size():
            params = request.json
            result = self.calculate_position_size(params)
            return jsonify(result)
        
        @app.route('/api/exposure', methods=['GET'])
        def get_exposure():
            positions = self._get_open_positions()
            account_info = self._get_account_info()
            exposure = self._calculate_exposure(positions, account_info)
            return jsonify(exposure)
        
        @app.route('/api/correlation', methods=['GET'])
        def get_correlation():
            positions = self._get_open_positions()
            correlation = self._calculate_correlation(positions)
            return jsonify(correlation)
        
        @app.route('/api/drawdown', methods=['GET'])
        def get_drawdown():
            account_info = self._get_account_info()
            drawdown = self._calculate_drawdown(account_info)
            return jsonify(drawdown)
        
        return app
    
    def run_api(self) -> None:
        """
        Ejecuta la API REST
        """
        host = self.config['api'].get('host', '0.0.0.0')
        port = self.config['api'].get('port', 8080)
        
        self.logger.info(f"Iniciando API en {host}:{port}")
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    # Crear gestor de riesgos
    risk_manager = RiskManager()
    
    # Ejecutar API
    risk_manager.run_api()
