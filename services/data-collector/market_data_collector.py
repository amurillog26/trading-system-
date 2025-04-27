import os
import json
import time
import logging
import argparse
import psycopg2
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import requests
from urllib.parse import urlencode


class MarketDataCollector:
    def __init__(self, config_path: str = None):
        """
        Inicializa el recolector de datos de mercado
        
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
        
        self.logger.info(f"Recolector de datos inicializado para {len(self.config['symbols'])} símbolos")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('market_data_collector')
    
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
            "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX", "CEMEXCPO.MX"],
            "timeframes": ["1d", "1h", "15m"],
            "sources": ["yfinance", "alpha_vantage"],
            "max_historical_days": 365,
            "update_interval": {
                "1d": 86400,  # Segundos en un día
                "1h": 3600,   # Segundos en una hora
                "15m": 900,   # Segundos en 15 minutos
                "5m": 300     # Segundos en 5 minutos
            },
            "retry_attempts": 3,
            "retry_delay": 5,
            "concurrent_downloads": 5,
            "symbol_data": {
                "AMXL.MX": {
                    "name": "América Móvil",
                    "sector": "Telecomunicaciones"
                },
                "FEMSAUBD.MX": {
                    "name": "FEMSA",
                    "sector": "Consumo"
                },
                "GFNORTEO.MX": {
                    "name": "Grupo Financiero Banorte",
                    "sector": "Financiero"
                },
                "WALMEX.MX": {
                    "name": "Walmart de México",
                    "sector": "Comercio Minorista"
                },
                "CEMEXCPO.MX": {
                    "name": "CEMEX",
                    "sector": "Materiales"
                }
            }
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'market_data.json')
        
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
    
    def collect_data(self, symbols: List[str] = None, timeframes: List[str] = None, 
                    days: int = None, force_update: bool = False) -> Dict:
        """
        Recolecta datos de mercado para los símbolos y timeframes especificados
        
        Args:
            symbols: Lista de símbolos a recolectar. Si es None, usa todos los de la configuración.
            timeframes: Lista de timeframes a recolectar. Si es None, usa todos los de la configuración.
            days: Número de días a recolectar. Si es None, usa el valor de la configuración.
            force_update: Si True, fuerza la actualización aunque los datos estén recientes.
            
        Returns:
            Dict con resultados de la recolección
        """
        if symbols is None:
            symbols = self.config['symbols']
        
        if timeframes is None:
            timeframes = self.config['timeframes']
        
        if days is None:
            days = self.config['max_historical_days']
        
        self.logger.info(f"Recolectando datos para {len(symbols)} símbolos y {len(timeframes)} timeframes")
        
        results = {}
        
        # Procesar cada combinación de símbolo y timeframe
        with ThreadPoolExecutor(max_workers=self.config['concurrent_downloads']) as executor:
            futures = {}
            
            for symbol in symbols:
                for timeframe in timeframes:
                    key = f"{symbol}_{timeframe}"
                    
                    # Verificar si es necesario actualizar
                    if not force_update and not self._should_update(symbol, timeframe):
                        self.logger.info(f"Datos recientes disponibles para {key}, omitiendo actualización")
                        results[key] = {"status": "skipped", "reason": "recent_data_available"}
                        continue
                    
                    # Ejecutar recolección en paralelo
                    futures[executor.submit(self._collect_symbol_data, symbol, timeframe, days)] = key
            
            # Recopilar resultados
            for future in futures:
                key = futures[future]
                try:
                    data = future.result()
                    if data is not None and not data.empty:
                        # Guardar en base de datos
                        rows_saved = self._save_to_database(data, key.split('_')[0], key.split('_')[1])
                        results[key] = {"status": "success", "rows": len(data), "saved": rows_saved}
                    else:
                        results[key] = {"status": "error", "reason": "no_data"}
                except Exception as e:
                    self.logger.error(f"Error al recolectar datos para {key}: {str(e)}")
                    results[key] = {"status": "error", "reason": str(e)}
        
        return results
    
    def _should_update(self, symbol: str, timeframe: str) -> bool:
        """
        Determina si es necesario actualizar los datos para un símbolo y timeframe
        
        Args:
            symbol: Símbolo a verificar
            timeframe: Timeframe a verificar
            
        Returns:
            True si se debe actualizar, False en caso contrario
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Obtener último registro
            query = """
            SELECT MAX(date) as last_date FROM market_data
            WHERE symbol = %s AND timeframe = %s;
            """
            
            cursor.execute(query, (symbol, timeframe))
            result = cursor.fetchone()
            cursor.close()
            
            if result[0] is None:
                # No hay datos, se debe actualizar
                return True
            
            last_date = result[0]
            now = dt.datetime.now()
            
            # Calcular tiempo transcurrido desde última actualización
            elapsed = (now - last_date).total_seconds()
            
            # Obtener intervalo de actualización para este timeframe
            update_interval = self.config['update_interval'].get(timeframe, 86400)  # 1 día por defecto
            
            # Determinar si ha pasado suficiente tiempo
            return elapsed >= update_interval
        except Exception as e:
            self.logger.error(f"Error al verificar actualización para {symbol}_{timeframe}: {str(e)}")
            # En caso de error, asumir que se debe actualizar
            return True
    
    def _collect_symbol_data(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Recolecta datos para un símbolo y timeframe específicos
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días a recolectar
            
        Returns:
            DataFrame con datos recolectados o None si no se pudieron obtener
        """
        # Determinar fuentes a utilizar
        sources = self.config['sources']
        
        # Intentar cada fuente en orden
        for source in sources:
            try:
                if source == 'yfinance':
                    data = self._collect_from_yfinance(symbol, timeframe, days)
                elif source == 'alpha_vantage':
                    data = self._collect_from_alpha_vantage(symbol, timeframe, days)
                else:
                    self.logger.warning(f"Fuente desconocida: {source}")
                    continue
                
                if data is not None and not data.empty:
                    self.logger.info(f"Datos recolectados de {source} para {symbol}_{timeframe}: {len(data)} registros")
                    return data
            except Exception as e:
                self.logger.error(f"Error al recolectar datos de {source} para {symbol}_{timeframe}: {str(e)}")
        
        # Si no se pudo obtener de ninguna fuente
        return None
    
    def _collect_from_yfinance(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Recolecta datos desde Yahoo Finance
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días a recolectar
            
        Returns:
            DataFrame con datos recolectados
        """
        # Mapear timeframe al formato de yfinance
        yf_interval = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d',
            '1wk': '1wk',
            '1mo': '1mo'
        }
        
        interval = yf_interval.get(timeframe, '1d')
        
        # Calcular rango de fechas
        end_date = dt.datetime.now()
        
        # Para intervalos intradía, yfinance tiene limitaciones en el historial disponible
        if interval in ['1m', '5m', '15m', '30m', '1h']:
            # Para intervalos intradía, el máximo es de 60 días
            start_date = end_date - dt.timedelta(days=min(days, 60))
        else:
            start_date = end_date - dt.timedelta(days=days)
        
        # Recolectar datos
        retry_attempts = self.config.get('retry_attempts', 3)
        retry_delay = self.config.get('retry_delay', 5)
        
        for attempt in range(retry_attempts):
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(interval=interval, start=start_date, end=end_date)
                
                if data.empty:
                    self.logger.warning(f"No hay datos disponibles en yfinance para {symbol} con intervalo {interval}")
                    return pd.DataFrame()
                
                # Preparar DataFrame
                df = data.reset_index()
                
                # Renombrar columnas al formato de nuestra base de datos
                df = df.rename(columns={
                    'Date': 'date', 
                    'Datetime': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Seleccionar solo las columnas necesarias
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                return df
            except Exception as e:
                self.logger.warning(f"Intento {attempt+1}/{retry_attempts} fallido para {symbol}: {str(e)}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
        
        return pd.DataFrame()
    
    def _collect_from_alpha_vantage(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """
        Recolecta datos desde Alpha Vantage
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días a recolectar
            
        Returns:
            DataFrame con datos recolectados
        """
        # Obtener API key
        api_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        
        if not api_key:
            self.logger.warning("Alpha Vantage API key no encontrada en variables de entorno")
            return pd.DataFrame()
        
        # Mapear timeframe al formato de Alpha Vantage
        av_interval = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min',
            '1d': 'daily',
            '1wk': 'weekly',
            '1mo': 'monthly'
        }
        
        interval = av_interval.get(timeframe, 'daily')
        
        # Construir URL
        base_url = 'https://www.alphavantage.co/query'
        
        if interval in ['1min', '5min', '15min', '30min', '60min']:
            # Datos intradía
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': symbol,
                'interval': interval,
                'outputsize': 'full',
                'apikey': api_key
            }
        else:
            # Datos diarios, semanales o mensuales
            function = {
                'daily': 'TIME_SERIES_DAILY',
                'weekly': 'TIME_SERIES_WEEKLY',
                'monthly': 'TIME_SERIES_MONTHLY'
            }
            
            params = {
                'function': function[interval],
                'symbol': symbol,
                'outputsize': 'full',
                'apikey': api_key
            }
        
        # Recolectar datos
        retry_attempts = self.config.get('retry_attempts', 3)
        retry_delay = self.config.get('retry_delay', 5)
        
        for attempt in range(retry_attempts):
            try:
                # Realizar solicitud a Alpha Vantage
                response = requests.get(base_url, params=params)
                data = response.json()
                
                # Verificar si hay errores
                if 'Error Message' in data:
                    self.logger.warning(f"Error de Alpha Vantage: {data['Error Message']}")
                    return pd.DataFrame()
                
                # Extraer datos según el intervalo
                if interval in ['1min', '5min', '15min', '30min', '60min']:
                    time_series_key = f"Time Series ({interval})"
                else:
                    time_series_key = {
                        'daily': 'Time Series (Daily)',
                        'weekly': 'Weekly Time Series',
                        'monthly': 'Monthly Time Series'
                    }[interval]
                
                if time_series_key not in data:
                    self.logger.warning(f"Datos no encontrados en respuesta de Alpha Vantage para {symbol}")
                    return pd.DataFrame()
                
                # Convertir a DataFrame
                time_series = data[time_series_key]
                records = []
                
                for date, values in time_series.items():
                    records.append({
                        'date': pd.to_datetime(date),
                        'open': float(values['1. open']),
                        'high': float(values['2. high']),
                        'low': float(values['3. low']),
                        'close': float(values['4. close']),
                        'volume': int(float(values['5. volume']))
                    })
                
                df = pd.DataFrame(records)
                
                # Ordenar por fecha
                df = df.sort_values('date')
                
                # Filtrar por rango de fechas
                start_date = dt.datetime.now() - dt.timedelta(days=days)
                df = df[df['date'] >= start_date]
                
                return df
            except Exception as e:
                self.logger.warning(f"Intento {attempt+1}/{retry_attempts} fallido para {symbol}: {str(e)}")
                if attempt < retry_attempts - 1:
                    time.sleep(retry_delay)
        
        return pd.DataFrame()
    
    def _save_to_database(self, data: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Guarda los datos recolectados en la base de datos
        
        Args:
            data: DataFrame con datos a guardar
            symbol: Símbolo de los datos
            timeframe: Timeframe de los datos
            
        Returns:
            Número de filas guardadas
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Inicializar contador
            rows_saved = 0
            
            # Iterar sobre cada fila y guardar
            for _, row in data.iterrows():
                # Verificar si ya existe este registro
                check_query = """
                SELECT id FROM market_data
                WHERE symbol = %s AND timeframe = %s AND date = %s;
                """
                
                cursor.execute(check_query, (symbol, timeframe, row['date']))
                existing = cursor.fetchone()
                
                if existing:
                    # Actualizar registro existente
                    update_query = """
                    UPDATE market_data
                    SET open = %s, high = %s, low = %s, close = %s, volume = %s
                    WHERE id = %s;
                    """
                    
                    cursor.execute(update_query, (
                        float(row['open']), float(row['high']), float(row['low']), 
                        float(row['close']), int(row['volume']), existing[0]
                    ))
                else:
                    # Insertar nuevo registro
                    insert_query = """
                    INSERT INTO market_data
                    (symbol, timeframe, date, open, high, low, close, volume, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    
                    cursor.execute(insert_query, (
                        symbol, timeframe, row['date'],
                        float(row['open']), float(row['high']), float(row['low']), 
                        float(row['close']), int(row['volume']), dt.datetime.now()
                    ))
                
                rows_saved += 1
            
            self.db_conn.commit()
            cursor.close()
            
            self.logger.info(f"Guardados {rows_saved} registros para {symbol}_{timeframe}")
            return rows_saved
        except Exception as e:
            self.db_conn.rollback()
            self.logger.error(f"Error al guardar datos en base de datos: {str(e)}")
            return 0


def main():
    """Función principal para ejecutar el recolector de datos"""
    parser = argparse.ArgumentParser(description='Recolector de datos de mercado')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--symbols', type=str, help='Lista de símbolos separados por comas')
    parser.add_argument('--timeframes', type=str, help='Lista de timeframes separados por comas')
    parser.add_argument('--days', type=int, help='Número de días a recolectar')
    parser.add_argument('--force', action='store_true', help='Forzar actualización aunque haya datos recientes')
    
    args = parser.parse_args()
    
    # Inicializar recolector
    collector = MarketDataCollector(args.config)
    
    # Obtener símbolos y timeframes
    symbols = args.symbols.split(',') if args.symbols else None
    timeframes = args.timeframes.split(',') if args.timeframes else None
    
    # Recolectar datos
    results = collector.collect_data(symbols, timeframes, args.days, args.force)
    
    # Mostrar resumen
    success_count = sum(1 for result in results.values() if result.get('status') == 'success')
    total_rows = sum(result.get('rows', 0) for result in results.values() if result.get('status') == 'success')
    
    print(f"Recolección completada: {success_count}/{len(results)} exitosos, {total_rows} registros recolectados")


if __name__ == "__main__":
    # Esperar un poco para que otros servicios estén disponibles
    time.sleep(10)
    
    main()
