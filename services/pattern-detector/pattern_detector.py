import os
import json
import time
import logging
import argparse
import psycopg2
import datetime as dt
import dask.dataframe as dd
import numpy as np
import pandas as pd  # Añade esta línea
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Tuple, Optional, Union
from concurrent.futures import ThreadPoolExecutor

class PatternDetector:
    def __init__(self, config_path: str = None):
        """
        Inicializa el detector de patrones técnicos
        
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
        
        self.logger.info(f"Detector de patrones inicializado con {len(self.config['patterns'])} patrones configurados")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('pattern_detector')
    
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
    
    def detect_patterns(self, symbols: List[str] = None, timeframes: List[str] = None) -> Dict:
        """
        Detecta patrones técnicos en los símbolos y timeframes especificados
        
        Args:
            symbols: Lista de símbolos a analizar. Si es None, usa todos los de la configuración.
            timeframes: Lista de timeframes a analizar. Si es None, usa todos los de la configuración.
            
        Returns:
            Dict con los resultados de detección de patrones
        """
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
    
    def _get_market_data(self, symbol: str, timeframe: str, lookback: int) -> dd.DataFrame:
        """
        Obtiene datos de mercado desde la base de datos
        
        Args:
            symbol: Símbolo a consultar
            timeframe: Timeframe a consultar
            lookback: Número de periodos a consultar
            
        Returns:
            DataFrame con los datos de mercado
        """
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
                return dd.from_pandas(pd.DataFrame(), npartitions=1)  # DataFrame vacío
            
            # SOLUCIÓN: Primero crear un DataFrame de pandas
            pd_df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
            
            # Luego convertir a DataFrame de Dask
            data = dd.from_pandas(pd_df, npartitions=max(1, len(pd_df) // 1000))
            
            # Ordenar por fecha
            data = data.sort_values('date')
            
            return data
        except Exception as e:
            self.logger.error(f"Error al obtener datos de mercado: {str(e)}")
            return dd.from_pandas(pd.DataFrame(), npartitions=1)  # Retornar DataFrame vacío en caso de error
            """
            Obtiene datos de mercado desde la base de datos
            
            Args:
                symbol: Símbolo a consultar
                timeframe: Timeframe a consultar
                lookback: Número de periodos a consultar
                
            Returns:
                DataFrame con los datos de mercado
            """
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
                    return dd.DataFrame()
                
                # Convertir a DataFrame
                data = dd.from_pandas(
                    pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume']),
                    npartitions=max(1, len(rows) // 1000)  # Dividir en particiones para procesamiento paralelo
                )
                
                # Ordenar por fecha
                data = data.sort_values('date')
                
                return data
            except Exception as e:
                self.logger.error(f"Error al obtener datos de mercado: {str(e)}")
                return dd.DataFrame()
        
        def _detect_all_patterns(self, data: dd.DataFrame, symbol: str, timeframe: str) -> List[Dict]:
            """
            Detecta todos los patrones configurados en los datos
            
            Args:
                data: DataFrame con datos de mercado
                symbol: Símbolo analizado
                timeframe: Timeframe analizado
                
            Returns:
                Lista de patrones detectados
            """
            # Convertir a pandas para usar talib (que no es compatible con dask)
            df = data.compute()
            
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
            """
            Detecta patrones de velas japonesas usando talib
            
            Args:
                df: DataFrame con datos de mercado
                
            Returns:
                Lista de patrones detectados
            """
            patterns = []
            confirmation_days = self.config['confirmation_days']
            
            # Mapeo de funciones de TALib a nombres de patrones
            pattern_functions = {
                'cdl_doji': df.ta.cdl_doji,
                'cdl_engulfing': df.ta.cdl_engulfing,
                'cdl_hammer': df.ta.cdl_hammer,
                'cdl_hanging_man': df.ta.cdl_h,
                'cdl_morning_star': df.ta.cdl_morning_star,
                'cdl_evening_star': df.ta.cdl_evening_star,
                'cdl_shooting_star': df.ta.cdl_shooting_star
            }
            
            # Detectar patrones configurados
            for pattern_name in self.config['patterns']:
                if pattern_name in pattern_functions:
                    try:
                        func = pattern_functions[pattern_name]
                        pattern_values = func(df['open'].values, df['high'].values, 
                                            df['low'].values, df['close'].values)
                        
                        # Identificar señales (valores no cero)
                        for i in range(len(pattern_values) - confirmation_days):
                            if pattern_values[i] != 0:
                                # 100 = señal alcista, -100 = señal bajista
                                bullish = pattern_values[i] > 0
                                pattern_type = f"{pattern_name}_bullish" if bullish else f"{pattern_name}_bearish"
                                
                                # Calcular confianza basada en volumen y confirmación posterior
                                confidence = self._calculate_pattern_confidence(
                                    df, i, bullish, confirmation_days)
                                
                                if confidence >= self.config['min_confidence']:
                                    # Calcular precios de entrada, stop loss y objetivo
                                    entry_price = df['close'].iloc[i]
                                    # Stop loss basado en mínimo/máximo reciente
                                    if bullish:
                                        stop_loss = df['low'].iloc[i-5:i+1].min() * 0.99
                                        target_price = entry_price + (entry_price - stop_loss) * 2
                                    else:
                                        stop_loss = df['high'].iloc[i-5:i+1].max() * 1.01
                                        target_price = entry_price - (stop_loss - entry_price) * 2
                                    
                                    patterns.append({
                                        'pattern_type': pattern_type,
                                        'confidence': confidence,
                                        'date': df['date'].iloc[i],
                                        'entry_price': entry_price,
                                        'stop_loss_price': stop_loss,
                                        'target_price': target_price,
                                        'notified': False
                                    })
                    except Exception as e:
                        self.logger.error(f"Error al detectar patrón {pattern_name}: {str(e)}")
            
            return patterns
        
        def _detect_ma_crossover(self, df: pd.DataFrame) -> List[Dict]:
            """
            Detecta cruces de medias móviles
            
            Args:
                df: DataFrame con datos de mercado
                
            Returns:
                Lista de patrones detectados
            """
            patterns = []
            fast_period = self.config['ma_crossover']['fast_period']
            slow_period = self.config['ma_crossover']['slow_period']
            confirmation_days = self.config['confirmation_days']
            
            # Calcular medias móviles
            df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
            df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
            
            # Detectar cruces
            for i in range(slow_period + 1, len(df)):
                # Cruce hacia arriba (señal alcista)
                if (df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1] and 
                    df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i]):
                    
                    # Calcular confianza
                    confidence = self._calculate_ma_crossover_confidence(df, i, True, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['low'].iloc[i-5:i+1].min() * 0.98
                        target_price = entry_price + (entry_price - stop_loss) * 2
                        
                        patterns.append({
                            'pattern_type': 'ma_crossover_bullish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
                
                # Cruce hacia abajo (señal bajista)
                elif (df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1] and 
                    df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i]):
                    
                    # Calcular confianza
                    confidence = self._calculate_ma_crossover_confidence(df, i, False, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['high'].iloc[i-5:i+1].max() * 1.02
                        target_price = entry_price - (stop_loss - entry_price) * 2
                        
                        patterns.append({
                            'pattern_type': 'ma_crossover_bearish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
            
            return patterns
        
        def _detect_rsi_signals(self, df: pd.DataFrame) -> List[Dict]:
            """
            Detecta señales basadas en RSI
            
            Args:
                df: DataFrame con datos de mercado
                
            Returns:
                Lista de patrones detectados
            """
            patterns = []
            period = self.config['rsi']['period']
            oversold = self.config['rsi']['oversold']
            overbought = self.config['rsi']['overbought']
            confirmation_days = self.config['confirmation_days']
            
            # Calcular RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Detectar señales
            for i in range(period + 1, len(df) - confirmation_days):
                # Señal de sobreventa (RSI cruza hacia arriba desde zona de sobreventa)
                if df['rsi'].iloc[i-1] <= oversold and df['rsi'].iloc[i] > oversold:
                    # Calcular confianza
                    confidence = self._calculate_rsi_confidence(df, i, True, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['low'].iloc[i-5:i+1].min() * 0.98
                        target_price = entry_price + (entry_price - stop_loss) * 2
                        
                        patterns.append({
                            'pattern_type': 'rsi_oversold_bullish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
                
                # Señal de sobrecompra (RSI cruza hacia abajo desde zona de sobrecompra)
                elif df['rsi'].iloc[i-1] >= overbought and df['rsi'].iloc[i] < overbought:
                    # Calcular confianza
                    confidence = self._calculate_rsi_confidence(df, i, False, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['high'].iloc[i-5:i+1].max() * 1.02
                        target_price = entry_price - (stop_loss - entry_price) * 2
                        
                        patterns.append({
                            'pattern_type': 'rsi_overbought_bearish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
            
            return patterns
        
        def _detect_bollinger_signals(self, df: pd.DataFrame) -> List[Dict]:
            """
            Detecta señales basadas en Bandas de Bollinger
            
            Args:
                df: DataFrame con datos de mercado
                
            Returns:
                Lista de patrones detectados
            """
            patterns = []
            period = self.config['bollinger']['period']
            std_dev = self.config['bollinger']['std_dev']
            confirmation_days = self.config['confirmation_days']
            
            # Calcular Bandas de Bollinger
            df['middle_band'] = df['close'].rolling(window=period).mean()
            df['std'] = df['close'].rolling(window=period).std()
            df['upper_band'] = df['middle_band'] + (df['std'] * std_dev)
            df['lower_band'] = df['middle_band'] - (df['std'] * std_dev)
            
            # Detectar señales
            for i in range(period + 1, len(df) - confirmation_days):
                # Señal de rebote desde banda inferior
                if (df['close'].iloc[i-1] <= df['lower_band'].iloc[i-1] and 
                    df['close'].iloc[i] > df['lower_band'].iloc[i]):
                    
                    # Calcular confianza
                    confidence = self._calculate_bollinger_confidence(df, i, True, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['low'].iloc[i-5:i+1].min() * 0.98
                        target_price = df['middle_band'].iloc[i]  # Objetivo: banda media
                        
                        patterns.append({
                            'pattern_type': 'bollinger_lower_band_bullish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
                
                # Señal de rechazo desde banda superior
                elif (df['close'].iloc[i-1] >= df['upper_band'].iloc[i-1] and 
                    df['close'].iloc[i] < df['upper_band'].iloc[i]):
                    
                    # Calcular confianza
                    confidence = self._calculate_bollinger_confidence(df, i, False, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['high'].iloc[i-5:i+1].max() * 1.02
                        target_price = df['middle_band'].iloc[i]  # Objetivo: banda media
                        
                        patterns.append({
                            'pattern_type': 'bollinger_upper_band_bearish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
            
            return patterns
        
        def _detect_macd_signals(self, df: pd.DataFrame) -> List[Dict]:
            """
            Detecta señales basadas en MACD
            
            Args:
                df: DataFrame con datos de mercado
                
            Returns:
                Lista de patrones detectados
            """
            patterns = []
            fast_period = self.config['macd']['fast_period']
            slow_period = self.config['macd']['slow_period']
            signal_period = self.config['macd']['signal_period']
            confirmation_days = self.config['confirmation_days']
            
            # Calcular MACD
            exp1 = df['close'].ewm(span=fast_period, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow_period, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['signal'] = df['macd'].ewm(span=signal_period, adjust=False).mean()
            df['histogram'] = df['macd'] - df['signal']
            
            # Detectar señales
            for i in range(slow_period + signal_period + 1, len(df) - confirmation_days):
                # Señal alcista: MACD cruza por encima de línea de señal
                if df['macd'].iloc[i-1] <= df['signal'].iloc[i-1] and df['macd'].iloc[i] > df['signal'].iloc[i]:
                    # Calcular confianza
                    confidence = self._calculate_macd_confidence(df, i, True, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['low'].iloc[i-5:i+1].min() * 0.98
                        target_price = entry_price + (entry_price - stop_loss) * 2
                        
                        patterns.append({
                            'pattern_type': 'macd_crossover_bullish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
                
                # Señal bajista: MACD cruza por debajo de línea de señal
                elif df['macd'].iloc[i-1] >= df['signal'].iloc[i-1] and df['macd'].iloc[i] < df['signal'].iloc[i]:
                    # Calcular confianza
                    confidence = self._calculate_macd_confidence(df, i, False, confirmation_days)
                    
                    if confidence >= self.config['min_confidence']:
                        # Calcular precios
                        entry_price = df['close'].iloc[i]
                        stop_loss = df['high'].iloc[i-5:i+1].max() * 1.02
                        target_price = entry_price - (stop_loss - entry_price) * 2
                        
                        patterns.append({
                            'pattern_type': 'macd_crossover_bearish',
                            'confidence': confidence,
                            'date': df['date'].iloc[i],
                            'entry_price': entry_price,
                            'stop_loss_price': stop_loss,
                            'target_price': target_price,
                            'notified': False
                        })
            
            return patterns
        
        def _calculate_pattern_confidence(self, df: pd.DataFrame, index: int, 
                                        bullish: bool, confirmation_days: int) -> float:
            """
            Calcula la confianza del patrón de velas japonesas
            
            Args:
                df: DataFrame con datos de mercado
                index: Índice de la señal en el DataFrame
                bullish: True si es señal alcista, False si es bajista
                confirmation_days: Días de confirmación a considerar
                
            Returns:
                Valor de confianza entre 0 y 1
            """
            # Factores para el cálculo de confianza
            volume_factor = 0.3  # Peso del volumen
            confirmation_factor = 0.5  # Peso de la confirmación posterior
            trend_factor = 0.2  # Peso de la tendencia previa
            
            # 1. Evaluar volumen
            avg_volume = df['volume'].iloc[index-5:index].mean()
            current_volume = df['volume'].iloc[index]
            
            volume_score = min(1.0, current_volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # 2. Evaluar confirmación posterior
            confirmation_score = 0
            if index + confirmation_days < len(df):
                if bullish:
                    # Para señal alcista, queremos ver precios subiendo en días posteriores
                    price_change = (df['close'].iloc[index+confirmation_days] - df['close'].iloc[index]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, price_change * 10))  # 10% de subida = confianza 1.0
                else:
                    # Para señal bajista, queremos ver precios bajando en días posteriores
                    price_change = (df['close'].iloc[index] - df['close'].iloc[index+confirmation_days]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, price_change * 10))  # 10% de bajada = confianza 1.0
            
            # 3. Evaluar tendencia previa
            trend_score = 0
            if index >= 10:
                if bullish:
                    # Para señal alcista, una tendencia bajista previa es favorable (reversión)
                    price_change = (df['close'].iloc[index-10:index].min() - df['close'].iloc[index-10]) / df['close'].iloc[index-10]
                    trend_score = min(1.0, max(0, -price_change * 5))  # 20% de bajada previa = confianza 1.0
                else:
                    # Para señal bajista, una tendencia alcista previa es favorable (reversión)
                    price_change = (df['close'].iloc[index-10:index].max() - df['close'].iloc[index-10]) / df['close'].iloc[index-10]
                    trend_score = min(1.0, max(0, price_change * 5))  # 20% de subida previa = confianza 1.0
            
            # Calcular confianza total ponderada
            confidence = (volume_score * volume_factor + 
                        confirmation_score * confirmation_factor + 
                        trend_score * trend_factor)
            
            return confidence
        
        def _calculate_ma_crossover_confidence(self, df: pd.DataFrame, index: int, 
                                            bullish: bool, confirmation_days: int) -> float:
            """
            Calcula la confianza del cruce de medias móviles
            
            Args:
                df: DataFrame con datos de mercado
                index: Índice de la señal en el DataFrame
                bullish: True si es señal alcista, False si es bajista
                confirmation_days: Días de confirmación a considerar
                
            Returns:
                Valor de confianza entre 0 y 1
            """
            # Factores para el cálculo de confianza
            angle_factor = 0.4  # Peso del ángulo de cruce
            volume_factor = 0.2  # Peso del volumen
            confirmation_factor = 0.4  # Peso de la confirmación posterior
            
            # 1. Evaluar ángulo de cruce
            ma_diff_before = df['fast_ma'].iloc[index-1] - df['slow_ma'].iloc[index-1]
            ma_diff_after = df['fast_ma'].iloc[index] - df['slow_ma'].iloc[index]
            
            angle_change = abs(ma_diff_after - ma_diff_before) / df['close'].iloc[index]
            angle_score = min(1.0, angle_change * 100)  # Cambio de 1% relativo = confianza 1.0
            
            # 2. Evaluar volumen
            avg_volume = df['volume'].iloc[index-5:index].mean()
            current_volume = df['volume'].iloc[index]
            
            volume_score = min(1.0, current_volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # 3. Evaluar confirmación posterior
            confirmation_score = 0
            if index + confirmation_days < len(df):
                if bullish:
                    # Para señal alcista, queremos que la media rápida siga por encima de la lenta
                    distance = (df['fast_ma'].iloc[index+confirmation_days] - df['slow_ma'].iloc[index+confirmation_days]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, distance * 100))
                else:
                    # Para señal bajista, queremos que la media rápida siga por debajo de la lenta
                    distance = (df['slow_ma'].iloc[index+confirmation_days] - df['fast_ma'].iloc[index+confirmation_days]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, distance * 100))
            
            # Calcular confianza total ponderada
            confidence = (angle_score * angle_factor + 
                        volume_score * volume_factor + 
                        confirmation_score * confirmation_factor)
            
            return confidence
        
        def _calculate_rsi_confidence(self, df: pd.DataFrame, index: int, 
                                    bullish: bool, confirmation_days: int) -> float:
            """
            Calcula la confianza de la señal de RSI
            
            Args:
                df: DataFrame con datos de mercado
                index: Índice de la señal en el DataFrame
                bullish: True si es señal alcista, False si es bajista
                confirmation_days: Días de confirmación a considerar
                
            Returns:
                Valor de confianza entre 0 y 1
            """
            # Factores para el cálculo de confianza
            extremity_factor = 0.3  # Peso de la extremidad del RSI
            volume_factor = 0.2  # Peso del volumen
            confirmation_factor = 0.5  # Peso de la confirmación posterior
            
            # 1. Evaluar extremidad del RSI
            if bullish:
                # Para señal alcista, cuanto más bajo haya sido el RSI mejor
                min_rsi = df['rsi'].iloc[index-3:index].min()
                extremity_score = min(1.0, (30 - min_rsi) / 30) if min_rsi < 30 else 0.5
            else:
                # Para señal bajista, cuanto más alto haya sido el RSI mejor
                max_rsi = df['rsi'].iloc[index-3:index].max()
                extremity_score = min(1.0, (max_rsi - 70) / 30) if max_rsi > 70 else 0.5
            
            # 2. Evaluar volumen
            avg_volume = df['volume'].iloc[index-5:index].mean()
            current_volume = df['volume'].iloc[index]
            
            volume_score = min(1.0, current_volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # 3. Evaluar confirmación posterior
            confirmation_score = 0
            if index + confirmation_days < len(df):
                if bullish:
                    # Para señal alcista, queremos ver precios subiendo en días posteriores
                    price_change = (df['close'].iloc[index+confirmation_days] - df['close'].iloc[index]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, price_change * 10))  # 10% de subida = confianza 1.0
                else:
                    # Para señal bajista, queremos ver precios bajando en días posteriores
                    price_change = (df['close'].iloc[index] - df['close'].iloc[index+confirmation_days]) / df['close'].iloc[index]
                    confirmation_score = min(1.0, max(0, price_change * 10))  # 10% de bajada = confianza 1.0
            
            # Calcular confianza total ponderada
            confidence = (extremity_score * extremity_factor + 
                        volume_score * volume_factor + 
                        confirmation_score * confirmation_factor)
            
            return confidence
        
        def _calculate_bollinger_confidence(self, df: pd.DataFrame, index: int, 
                                        bullish: bool, confirmation_days: int) -> float:
            """
            Calcula la confianza de la señal de Bandas de Bollinger
            
            Args:
                df: DataFrame con datos de mercado
                index: Índice de la señal en el DataFrame
                bullish: True si es señal alcista, False si es bajista
                confirmation_days: Días de confirmación a considerar
                
            Returns:
                Valor de confianza entre 0 y 1
            """
            # Factores para el cálculo de confianza
            distance_factor = 0.3  # Peso de la distancia a la banda
            volume_factor = 0.2  # Peso del volumen
            confirmation_factor = 0.5  # Peso de la confirmación posterior
            
            # 1. Evaluar distancia a la banda
            if bullish:
                # Para señal alcista, evaluamos qué tan por debajo de la banda inferior llegó el precio
                min_price = df['low'].iloc[index-2:index+1].min()
                distance = (df['lower_band'].iloc[index] - min_price) / df['close'].iloc[index]
                distance_score = min(1.0, max(0, distance * 50))  # 2% por debajo = confianza 1.0
            else:
                # Para señal bajista, evaluamos qué tan por encima de la banda superior llegó el precio
                max_price = df['high'].iloc[index-2:index+1].max()
                distance = (max_price - df['upper_band'].iloc[index]) / df['close'].iloc[index]
                distance_score = min(1.0, max(0, distance * 50))  # 2% por encima = confianza 1.0
            
            # 2. Evaluar volumen
            avg_volume = df['volume'].iloc[index-5:index].mean()
            current_volume = df['volume'].iloc[index]
            
            volume_score = min(1.0, current_volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # 3. Evaluar confirmación posterior
            confirmation_score = 0
            if index + confirmation_days < len(df):
                if bullish:
                    # Para señal alcista, queremos ver precios subiendo hacia la banda media
                    price_change = (df['close'].iloc[index+confirmation_days] - df['close'].iloc[index]) / df['close'].iloc[index]
                    target_change = (df['middle_band'].iloc[index] - df['close'].iloc[index]) / df['close'].iloc[index]
                    if target_change > 0:
                        confirmation_score = min(1.0, price_change / target_change)
                    else:
                        confirmation_score = 0.5
                else:
                    # Para señal bajista, queremos ver precios bajando hacia la banda media
                    price_change = (df['close'].iloc[index] - df['close'].iloc[index+confirmation_days]) / df['close'].iloc[index]
                    target_change = (df['close'].iloc[index] - df['middle_band'].iloc[index]) / df['close'].iloc[index]
                    if target_change > 0:
                        confirmation_score = min(1.0, price_change / target_change)
                    else:
                        confirmation_score = 0.5
            
            # Calcular confianza total ponderada
            confidence = (distance_score * distance_factor + 
                        volume_score * volume_factor + 
                        confirmation_score * confirmation_factor)
            
            return confidence
        
        def _calculate_macd_confidence(self, df: pd.DataFrame, index: int, 
                                    bullish: bool, confirmation_days: int) -> float:
            """
            Calcula la confianza de la señal de MACD
            
            Args:
                df: DataFrame con datos de mercado
                index: Índice de la señal en el DataFrame
                bullish: True si es señal alcista, False si es bajista
                confirmation_days: Días de confirmación a considerar
                
            Returns:
                Valor de confianza entre 0 y 1
            """
            # Factores para el cálculo de confianza
            histogram_factor = 0.3  # Peso del cambio en el histograma
            volume_factor = 0.2  # Peso del volumen
            confirmation_factor = 0.5  # Peso de la confirmación posterior
            
            # 1. Evaluar cambio en el histograma
            hist_before = df['histogram'].iloc[index-1]
            hist_after = df['histogram'].iloc[index]
            
            if bullish:
                # Para señal alcista, queremos un histograma que crece significativamente
                histogram_change = hist_after - hist_before
                histogram_score = min(1.0, max(0, histogram_change / df['close'].iloc[index] * 1000))
            else:
                # Para señal bajista, queremos un histograma que decrece significativamente
                histogram_change = hist_before - hist_after
                histogram_score = min(1.0, max(0, histogram_change / df['close'].iloc[index] * 1000))
            
            # 2. Evaluar volumen
            avg_volume = df['volume'].iloc[index-5:index].mean()
            current_volume = df['volume'].iloc[index]
            
            volume_score = min(1.0, current_volume / (avg_volume * 1.5)) if avg_volume > 0 else 0.5
            
            # 3. Evaluar confirmación posterior
            confirmation_score = 0
            if index + confirmation_days < len(df):
                if bullish:
                    # Para señal alcista, queremos ver el histograma creciendo más
                    histogram_after_confirmation = df['histogram'].iloc[index+confirmation_days]
                    if histogram_after_confirmation > hist_after:
                        confirmation_score = min(1.0, (histogram_after_confirmation - hist_after) / abs(hist_after) if hist_after != 0 else 0.5)
                    else:
                        confirmation_score = 0
                else:
                    # Para señal bajista, queremos ver el histograma decreciendo más
                    histogram_after_confirmation = df['histogram'].iloc[index+confirmation_days]
                    if histogram_after_confirmation < hist_after:
                        confirmation_score = min(1.0, (hist_after - histogram_after_confirmation) / abs(hist_after) if hist_after != 0 else 0.5)
                    else:
                        confirmation_score = 0
            
            # Calcular confianza total ponderada
            confidence = (histogram_score * histogram_factor + 
                        volume_score * volume_factor + 
                        confirmation_score * confirmation_factor)
            
            return confidence
        
        def _save_patterns_to_db(self, patterns: List[Dict]) -> None:
            """
            Guarda los patrones detectados en la base de datos
            
            Args:
                patterns: Lista de patrones detectados
            """
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
            finally:
                cursor.close()

def main():
    """Función principal para ejecutar el detector de patrones"""
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
