import os
import json
import time
import logging
import argparse
import requests
import psycopg2
import datetime as dt
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        
        # Inicializar API Keys
        self.alpha_vantage_key = os.environ.get('ALPHA_VANTAGE_KEY', '')
        
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
            "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"],
            "timeframes": ["1d", "1h"],
            "sources": ["yfinance", "alpha_vantage"],
            "max_historical_days": 365,
            "update_interval": {
                "1d": 86400,  # 24 horas en segundos
                "1h": 3600,   # 1 hora en segundos
                "15m": 900,   # 15 minutos en segundos
                "5m": 300     # 5 minutos en segundos
            },
            "retry_attempts": 3,
            "retry_delay": 5,  # segundos
            "concurrent_downloads": 5
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
    
    def collect_data(self, symbols: List[str] = None, timeframes: List[str] = None) -> Dict:
        """
        Recolecta datos de mercado para los símbolos y timeframes especificados
        
        Args:
            symbols: Lista de símbolos a recolectar. Si es None, usa todos los de la configuración.
            timeframes: Lista de timeframes a recolectar. Si es None, usa todos los de la configuración.
            
        Returns:
            Dict con resultados de la recolección
        """
        if symbols is None:
            symbols = self.config['symbols']
        
        if timeframes is None:
            timeframes = self.config['timeframes']
        
        self.logger.info(f"Recolectando datos para {len(symbols)} símbolos y {len(timeframes)} timeframes")
        
        # Verificar si se necesita recolectar datos históricos completos o solo actualizaciones
        full_history_symbols = self._check_full_history_needed(symbols, timeframes)
        
        # Resultados
        results = {
            "success": 0,
            "failed": 0,
            "total": len(symbols) * len(timeframes),
            "details": {}
        }
        
        # Usar ThreadPoolExecutor para descargas concurrentes
        max_workers = self.config['concurrent_downloads']
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for symbol in symbols:
                for timeframe in timeframes:
                    key = f"{symbol}_{timeframe}"
                    
                    # Determinar si necesitamos historia completa o actualización
                    if symbol in full_history_symbols and timeframe in full_history_symbols[symbol]:
                        # Historia completa
                        days = self.config['max_historical_days']
                        self.logger.info(f"Descargando historia completa para {key} ({days} días)")
                    else:
                        # Solo actualización desde último registro
                        last_date = self._get_last_data_date(symbol, timeframe)
                        days = None
                        self.logger.info(f"Actualizando datos para {key} desde {last_date}")
                    
                    # Enviar tarea
                    futures.append(executor.submit(
                        self._collect_symbol_data, symbol, timeframe, days, last_date))
            
            # Procesar resultados
            for future in as_completed(futures):
                try:
                    result = future.result()
                    key = f"{result['symbol']}_{result['timeframe']}"
                    results['details'][key] = result
                    
                    if result['success']:
                        results['success'] += 1
                    else:
                        results['failed'] += 1
                except Exception as e:
                    self.logger.error(f"Error en tarea de recolección: {str(e)}")
                    results['failed'] += 1
        
        self.logger.info(f"Recolección completada: {results['success']} exitosos, {results['failed']} fallidos")
        return results
    
    def _check_full_history_needed(self, symbols: List[str], timeframes: List[str]) -> Dict[str, List[str]]:
        """
        Verifica qué símbolos y timeframes necesitan descarga de historia completa
        
        Args:
            symbols: Lista de símbolos a verificar
            timeframes: Lista de timeframes a verificar
            
        Returns:
            Dict con símbolos y timeframes que necesitan historia completa
        """
        full_history_needed = {}
        
        try:
            cursor = self.db_conn.cursor()
            
            for symbol in symbols:
                symbol_timeframes = []
                
                for timeframe in timeframes:
                    # Verificar si ya existen datos
                    query = """
                    SELECT COUNT(*) FROM market_data
                    WHERE symbol = %s AND timeframe = %s;
                    """
                    
                    cursor.execute(query, (symbol, timeframe))
                    count = cursor.fetchone()[0]
                    
                    if count == 0:
                        # No hay datos, necesitamos historia completa
                        symbol_timeframes.append(timeframe)
                
                if symbol_timeframes:
                    full_history_needed[symbol] = symbol_timeframes
            
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al verificar necesidad de historia completa: {str(e)}")
        
        return full_history_needed
    
    def _get_last_data_date(self, symbol: str, timeframe: str) -> Optional[dt.datetime]:
        """
        Obtiene la fecha del último dato disponible en la base de datos
        
        Args:
            symbol: Símbolo a consultar
            timeframe: Timeframe a consultar
            
        Returns:
            Fecha del último dato o None si no hay datos
        """
        try:
            cursor = self.db_conn.cursor()
            
            query = """
            SELECT MAX(date) FROM market_data
            WHERE symbol = %s AND timeframe = %s;
            """
            
            cursor.execute(query, (symbol, timeframe))
            last_date = cursor.fetchone()[0]
            cursor.close()
            
            return last_date
        except Exception as e:
            self.logger.error(f"Error al obtener última fecha para {symbol}_{timeframe}: {str(e)}")
            return None
    
    def _collect_symbol_data(self, symbol: str, timeframe: str, 
                            days: Optional[int] = None, 
                            start_date: Optional[dt.datetime] = None) -> Dict:
        """
        Recolecta datos para un símbolo y timeframe específico
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días históricos a recolectar (si es None, usa start_date)
            start_date: Fecha de inicio para la recolección (si es None, usa days)
            
        Returns:
            Dict con resultados de la recolección
        """
        result = {
            "symbol": symbol,
            "timeframe": timeframe,
            "success": False,
            "rows": 0,
            "source": None,
            "error": None
        }
        
        # Intentar con cada fuente configurada
        for source in self.config['sources']:
            try:
                if source == "yfinance":
                    data = self._get_data_from_yfinance(symbol, timeframe, days, start_date)
                elif source == "alpha_vantage":
                    data = self._get_data_from_alpha_vantage(symbol, timeframe, days, start_date)
                else:
                    self.logger.warning(f"Fuente no soportada: {source}")
                    continue
                
                if data is not None and not data.empty:
                    # Guardar datos en base de datos
                    rows_saved = self._save_data_to_db(data, symbol, timeframe)
                    result['success'] = True
                    result['rows'] = rows_saved
                    result['source'] = source
                    break
            except Exception as e:
                error_msg = f"Error al obtener datos de {source} para {symbol}_{timeframe}: {str(e)}"
                self.logger.error(error_msg)
                result['error'] = error_msg
        
        return result
    
    def _get_data_from_yfinance(self, symbol: str, timeframe: str, 
                               days: Optional[int] = None, 
                               start_date: Optional[dt.datetime] = None) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de Yahoo Finance
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días históricos a recolectar
            start_date: Fecha de inicio para la recolección
            
        Returns:
            DataFrame con datos o None si hay error
        """
        # Mapeo de timeframes a intervalos de Yahoo Finance
        timeframe_map = {
            "1d": "1d",
            "1h": "1h",
            "15m": "15m",
            "5m": "5m",
            "1m": "1m"
        }
        
        if timeframe not in timeframe_map:
            self.logger.error(f"Timeframe no soportado para yfinance: {timeframe}")
            return None
        
        interval = timeframe_map[timeframe]
        
        # Determinar período
        period = None
        start = None
        end = dt.datetime.now()
        
        if days:
            # Usar período en días
            if interval == "1d":
                # Para datos diarios, podemos usar 'max' para historia completa
                period = "max" if days > 5000 else f"{days}d"
            else:
                # Para intradiarios, hay que usar fechas específicas
                start = end - dt.timedelta(days=min(days, 60))  # yfinance tiene límite de 60 días para intradiario
        elif start_date:
            # Usar fecha de inicio específica
            start = start_date
            if interval != "1d" and (end - start).days > 60:
                # Limitar a 60 días para datos intradiarios
                start = end - dt.timedelta(days=60)
        else:
            # Si no hay días ni fecha de inicio, usar un valor predeterminado
            if interval == "1d":
                period = "max"
            else:
                period = "60d"
        
        # Intentar descargar datos
        for attempt in range(self.config['retry_attempts']):
            try:
                ticker = yf.Ticker(symbol)
                
                if period:
                    data = ticker.history(period=period, interval=interval)
                else:
                    data = ticker.history(start=start, end=end, interval=interval)
                
                if data.empty:
                    self.logger.warning(f"No hay datos disponibles para {symbol} con intervalo {interval}")
                    return None
                
                # Resetear índice para tener la fecha como columna
                data = data.reset_index()
                
                # Renombrar columnas a formato estándar
                data = data.rename(columns={
                    "Date": "date",
                    "Datetime": "date",
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                
                # Seleccionar solo columnas necesarias
                columns = ["date", "open", "high", "low", "close", "volume"]
                data = data[columns]
                
                # Asegurar que el volumen no sea NaN
                data['volume'] = data['volume'].fillna(0).astype(int)
                
                return data
            except Exception as e:
                self.logger.warning(f"Intento {attempt+1} fallido para {symbol}: {str(e)}")
                time.sleep(self.config['retry_delay'])
        
        self.logger.error(f"Todos los intentos fallidos para {symbol} con yfinance")
        return None
    
    def _get_data_from_alpha_vantage(self, symbol: str, timeframe: str, 
                                    days: Optional[int] = None, 
                                    start_date: Optional[dt.datetime] = None) -> Optional[pd.DataFrame]:
        """
        Obtiene datos de Alpha Vantage
        
        Args:
            symbol: Símbolo a recolectar
            timeframe: Timeframe a recolectar
            days: Número de días históricos a recolectar
            start_date: Fecha de inicio para la recolección
            
        Returns:
            DataFrame con datos o None si hay error
        """
        if not self.alpha_vantage_key:
            self.logger.warning("Alpha Vantage API key no configurada")
            return None
        
        # Mapeo de timeframes a intervalos de Alpha Vantage
        timeframe_map = {
            "1d": "daily",
            "1h": "60min",
            "15m": "15min",
            "5m": "5min",
            "1m": "1min"
        }
        
        if timeframe not in timeframe_map:
            self.logger.error(f"Timeframe no soportado para Alpha Vantage: {timeframe}")
            return None
        
        function = "TIME_SERIES_INTRADAY" if timeframe != "1d" else "TIME_SERIES_DAILY"
        interval = timeframe_map[timeframe]
        
        # Configurar parámetros de la API
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full"
        }
        
        if timeframe != "1d":
            params["interval"] = interval
        
        # Intentar descargar datos
        for attempt in range(self.config['retry_attempts']):
            try:
                response = requests.get(base_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                # Extraer la serie temporal
                time_series_key = f"Time Series ({interval})" if timeframe != "1d" else "Time Series (Daily)"
                
                if time_series_key not in data:
                    error_msg = data.get("Error Message", "Unknown error")
                    self.logger.error(f"Error de Alpha Vantage: {error_msg}")
                    return None
                
                time_series = data[time_series_key]
                
                # Convertir a DataFrame
                df = pd.DataFrame.from_dict(time_series, orient='index')
                
                # Renombrar columnas
                df = df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                })
                
                # Convertir tipos de datos
                for col in ["open", "high", "low", "close"]:
                    df[col] = pd.to_numeric(df[col])
                
                df["volume"] = pd.to_numeric(df["volume"]).astype(int)
                
                # Añadir columna de fecha y resetear índice
                df.index = pd.to_datetime(df.index)
                df = df.reset_index()
                df = df.rename(columns={"index": "date"})
                
                # Filtrar por fecha si es necesario
                if start_date:
                    df = df[df["date"] >= start_date]
                elif days:
                    end_date = dt.datetime.now()
                    start_date = end_date - dt.timedelta(days=days)
                    df = df[df["date"] >= start_date]
                
                return df
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Intento {attempt+1} fallido para {symbol} con Alpha Vantage: {str(e)}")
                time.sleep(self.config['retry_delay'])
            except Exception as e:
                self.logger.error(f"Error al procesar datos de Alpha Vantage para {symbol}: {str(e)}")
                return None
        
        self.logger.error(f"Todos los intentos fallidos para {symbol} con Alpha Vantage")
        return None
    
    def _save_data_to_db(self, data: pd.DataFrame, symbol: str, timeframe: str) -> int:
        """
        Guarda los datos en la base de datos
        
        Args:
            data: DataFrame con datos a guardar
            symbol: Símbolo de los datos
            timeframe: Timeframe de los datos
            
        Returns:
            Número de filas guardadas
        """
        rows_saved = 0
        
        try:
            cursor = self.db_conn.cursor()
            
            # Crear tabla si no existe
            create_table_query = """
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
            CREATE INDEX IF NOT EXISTS market_data_symbol_timeframe_date_idx ON market_data(symbol, timeframe, date);
            """
            cursor.execute(create_table_query)
            self.db_conn.commit()
            
            # Insertar datos
            for _, row in data.iterrows():
                try:
                    # Verificar si el registro ya existe
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
                        SET open = %s, high = %s, low = %s, close = %s, volume = %s, created_at = NOW()
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
                        (symbol, timeframe, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                        """
                        cursor.execute(insert_query, (
                            symbol, timeframe, row['date'],
                            float(row['open']), float(row['high']), float(row['low']),
                            float(row['close']), int(row['volume'])
                        ))
                    
                    rows_saved += 1
                except Exception as e:
                    self.logger.error(f"Error al guardar fila para {symbol}_{timeframe} en fecha {row['date']}: {str(e)}")
                    self.db_conn.rollback()
                    continue
                
                # Commit cada 100 filas para evitar transacciones muy largas
                if rows_saved % 100 == 0:
                    self.db_conn.commit()
            
            # Commit final
            self.db_conn.commit()
            
            self.logger.info(f"Guardadas {rows_saved} filas para {symbol}_{timeframe}")
        except Exception as e:
            self.logger.error(f"Error al guardar datos en base de datos para {symbol}_{timeframe}: {str(e)}")
            self.db_conn.rollback()
        finally:
            cursor.close()
        
        return rows_saved

def main():
    """Función principal para ejecutar el recolector de datos"""
    parser = argparse.ArgumentParser(description='Recolector de datos de mercado')
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--symbols', type=str, help='Lista de símbolos separados por comas')
    parser.add_argument('--timeframes', type=str, help='Lista de timeframes separados por comas')
    
    args = parser.parse_args()
    
    # Inicializar recolector
    collector = MarketDataCollector(args.config)
    
    # Obtener símbolos y timeframes
    symbols = args.symbols.split(',') if args.symbols else None
    timeframes = args.timeframes.split(',') if args.timeframes else None
    
    # Recolectar datos
    results = collector.collect_data(symbols, timeframes)
    
    # Mostrar resumen
    print(f"Recolección completada: {results['success']} exitosos, {results['failed']} fallidos de {results['total']} total")

if __name__ == "__main__":
    # Esperar un poco para que otros servicios estén disponibles
    time.sleep(10)
    
    main()
