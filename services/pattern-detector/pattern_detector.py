import os
import json
import time
import logging
import argparse
import psycopg2
import datetime as dt
import pandas as pd
import numpy as np
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

class PatternDetector:
    def __init__(self, config_path: str = None):
        # Configurar logger
        self._setup_logging()
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Conectar con base de datos
        self.db_conn = self._get_db_connection()
        
        self.logger.info(f"Detector de patrones inicializado con {len(self.config['patterns'])} patrones configurados")
    
    def _setup_logging(self) -> None:
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('pattern_detector')
    
    def _load_config(self, config_path: str = None) -> Dict:
        # Configuración por defecto
        default_config = {
            "patterns": [
                "cdl_doji",
                "cdl_engulfing",
                "cdl_hammer",
                "cdl_hanging_man",
                "cdl_morning_star",
                "cdl_evening_star",
                "cdl_shooting_star"
            ],
            "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"],
            "timeframes": ["1d", "1h"],
            "min_confidence": 0.6,
            "ma_crossover": {
                "enabled": True,
                "fast_period": 9,
                "slow_period": 21
            },
            "rsi": {
                "enabled": True,
                "period": 14,
                "oversold": 30,
                "overbought": 70
            },
            "bollinger": {
                "enabled": True,
                "period": 20,
                "std_dev": 2.0
            },
            "macd": {
                "enabled": True,
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9
            },
            "confirmation_days": 2,
            "lookback_periods": 200
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'patterns.json')
        
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
        for key, value in update.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._deep_update(original[key], value)
            else:
                original[key] = value
    
    def _get_db_connection(self):
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
    
    def detect_patterns(self, symbols: List[str] = None, timeframes: List[str] = None) -> Dict:
        if symbols is None:
            symbols = self.config['symbols']
        
        if timeframes is None:
            timeframes = self.config['timeframes']
        
        self.logger.info(f"Detectando patrones para {len(symbols)} símbolos y {len(timeframes)} timeframes")
        
        results = {}
        lookback = self.config['lookback_periods']
        
        # Procesar cada combinación de símbolo y timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                
                try:
                    # Obtener datos de la base de datos
                    data = self._get_market_data(symbol, timeframe, lookback)
                    
                    if data.empty:
                        self.logger.warning(f"No hay datos disponibles para {key}")
                        continue
                    
                    # Detectar patrones
                    patterns = self._detect_all_patterns(data, symbol, timeframe)
                    
                    # Añadir a resultados
                    results[key] = patterns
                    
                    # Guardar patrones detectados en base de datos
                    self._save_patterns_to_db(patterns)
                    
                    self.logger.info(f"Detectados {len(patterns)} patrones para {key}")
                except Exception as e:
                    self.logger.error(f"Error al procesar {key}: {str(e)}")
        
        return results
    
    def _get_market_data(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame:
        try:
            cursor = self.db_conn.cursor()
            
            # Construir consulta SQL
            query = """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = %s AND timeframe = %s
            ORDER BY date DESC
            LIMIT %s;
            """
            
            cursor.execute(query, (symbol, timeframe, lookback))
            rows = cursor.fetchall()
            cursor.close()
            
            if not rows:
                return pd.DataFrame()  # DataFrame vacío
            
            # Crear DataFrame
            data = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            # Ordenar por fecha
            data = data.sort_values('date')
            
            return data
        except Exception as e:
            self.logger.error(f"Error al obtener datos de mercado: {str(e)}")
            return pd.DataFrame()
    
    def _detect_all_patterns(self, data: pd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
        # Convertir a pandas para análisis
        df = data
        
        patterns = []
        
        # 1. Detectar patrones de velas japonesas
        candle_patterns = self._detect_candlestick_patterns(df)
        patterns.extend(candle_patterns)
        
        # 2. Detectar cruces de medias móviles
        if self.config['ma_crossover']['enabled']:
            ma_patterns = self._detect_ma_crossover(df)
            patterns.extend(ma_patterns)
        
        # 3. Detectar señales de RSI
        if self.config['rsi']['enabled']:
            rsi_patterns = self._detect_rsi_signals(df)
            patterns.extend(rsi_patterns)
        
        # 4. Detectar señales de Bandas de Bollinger
        if self.config['bollinger']['enabled']:
            bollinger_patterns = self._detect_bollinger_signals(df)
            patterns.extend(bollinger_patterns)
        
        # 5. Detectar señales de MACD
        if self.config['macd']['enabled']:
            macd_patterns = self._detect_macd_signals(df)
            patterns.extend(macd_patterns)
        
        # Añadir información común a todos los patrones
        for pattern in patterns:
            pattern['symbol'] = symbol
            pattern['timeframe'] = timeframe
            pattern['timestamp'] = dt.datetime.now()
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        return []
    
    def _detect_ma_crossover(self, df: pd.DataFrame) -> List[Dict]:
        return []
    
    def _detect_rsi_signals(self, df: pd.DataFrame) -> List[Dict]:
        return []
    
    def _detect_bollinger_signals(self, df: pd.DataFrame) -> List[Dict]:
        return []
    
    def _detect_macd_signals(self, df: pd.DataFrame) -> List[Dict]:
        return []
    
    def _save_patterns_to_db(self, patterns: List[Dict]) -> None:
        if not patterns:
            return
        
        try:
            cursor = self.db_conn.cursor()
            
            for pattern in patterns:
                # Verificar si el patrón ya existe (misma fecha, símbolo y tipo)
                check_query = """
                SELECT id FROM detected_patterns
                WHERE symbol = %s AND timeframe = %s AND pattern_type = %s 
                AND date = %s AND ABS(entry_price - %s) < 0.001;
                """
                
                cursor.execute(check_query, (
                    pattern['symbol'], pattern['timeframe'], pattern['pattern_type'],
                    pattern['date'], pattern['entry_price']
                ))
                
                existing = cursor.fetchone()
                
                if existing:
                    # Actualizar patrón existente
                    update_query = """
                    UPDATE detected_patterns
                    SET confidence = %s, stop_loss_price = %s, target_price = %s, timestamp = %s
                    WHERE id = %s;
                    """
                    
                    cursor.execute(update_query, (
                        pattern['confidence'], pattern['stop_loss_price'],
                        pattern['target_price'], dt.datetime.now(), existing[0]
                    ))
                else:
                    # Insertar nuevo patrón
                    insert_query = """
                    INSERT INTO detected_patterns
                    (symbol, timeframe, pattern_type, confidence, date, entry_price, 
                    stop_loss_price, target_price, timestamp, notified)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    
                    cursor.execute(insert_query, (
                        pattern['symbol'], pattern['timeframe'], pattern['pattern_type'],
                        pattern['confidence'], pattern['date'], pattern['entry_price'],
                        pattern['stop_loss_price'], pattern['target_price'], 
                        pattern['timestamp'], pattern['notified']
                    ))
            
            self.db_conn.commit()
            self.logger.info(f"Guardados {len(patterns)} patrones en base de datos")
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Error al guardar patrones en base de datos: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Detector de patrones técnicos')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--symbols', type=str, help='Lista de símbolos separados por comas')
    parser.add_argument('--timeframes', type=str, help='Lista de timeframes separados por comas')
    
    args = parser.parse_args()
    
    # Inicializar detector
    detector = PatternDetector(args.config)
    
    # Obtener símbolos y timeframes
    symbols = args.symbols.split(',') if args.symbols else None
    timeframes = args.timeframes.split(',') if args.timeframes else None
    
    # Detectar patrones
    results = detector.detect_patterns(symbols, timeframes)
    
    # Mostrar resumen
    total_patterns = sum(len(patterns) for patterns in results.values())
    print(f"Detectados {total_patterns} patrones en total")
    
    for key, patterns in results.items():
        print(f"{key}: {len(patterns)} patrones")

if __name__ == "__main__":
    # Esperar un poco para que otros servicios estén disponibles
    time.sleep(10)
    
    main()
