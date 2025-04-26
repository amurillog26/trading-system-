import os
import json
import time
import logging
import threading
import pandas as pd
import numpy as np
import psycopg2
import datetime as dt
import requests
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, request
from typing import Dict, List, Tuple, Union, Optional

class TradeExecutor:
    def __init__(self, config_path: str = None):
        """
        Inicializa el ejecutor de operaciones
        
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
        
        # URLs de servicios
        self.ibkr_service_url = os.environ.get('IBKR_SERVICE_URL', 'http://ibkr-connector:8080/api')
        self.risk_service_url = os.environ.get('RISK_SERVICE_URL', 'http://risk-manager:8080/api')
        
        # Crear la aplicación Flask
        self.app = self._create_app()
        
        # Iniciar hilo para procesar señales pendientes
        self.processing_thread = threading.Thread(target=self._process_pending_signals, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("Ejecutor de operaciones inicializado")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('trade_executor')
    
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
            "execution": {
                "enabled": True,  # Habilitar ejecución automática
                "min_confidence": 0.7,  # Mínima confianza para ejecutar
                "default_order_type": "LIMIT",  # MARKET o LIMIT
                "limit_price_buffer": 0.002,  # 0.2% buffer para órdenes limit
                "stop_loss_enabled": True,
                "take_profit_enabled": True,
                "outside_rth": False,  # Outside Regular Trading Hours
                "risk_reward_min": 1.5  # Mínima relación riesgo/recompensa
            },
            "processing": {
                "check_interval": 300,  # Revisar señales cada 5 minutos
                "max_signals_per_run": 5,  # Máximas señales a procesar por ciclo
                "max_age_days": 1  # Máxima antigüedad de señales a procesar
            },
            "filters": {
                "patterns": ["ma_crossover", "rsi", "bollinger", "macd"],  # Patrones permitidos
                "min_volume": 100000,  # Volumen mínimo
                "max_spread_pct": 1.0  # Spread máximo como % del precio
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080
            }
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'trade_execution.json')
        
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
    
    def _process_pending_signals(self) -> None:
        """
        Hilo principal para procesar señales pendientes
        """
        interval = self.config['processing'].get('check_interval', 300)  # 5 minutos por defecto
        
        while True:
            try:
                # Verificar si la ejecución automática está habilitada
                if self.config['execution'].get('enabled', True):
                    # Procesar señales pendientes
                    pending_signals = self._get_pending_signals()
                    
                    for signal in pending_signals:
                        try:
                            # Procesar señal
                            result = self.process_signal(signal)
                            
                            # Actualizar estado en base de datos
                            self._update_signal_status(signal['id'], result)
                            
                            # Si se generó una orden, guardar referencia
                            if result.get('success') and 'order_id' in result:
                                self._link_order_to_signal(result['order_id'], signal['id'])
                        except Exception as e:
                            self.logger.error(f"Error al procesar señal {signal['id']}: {str(e)}")
                            # Actualizar estado como error
                            self._update_signal_status(signal['id'], {'success': False, 'error': str(e)})
                else:
                    self.logger.debug("Ejecución automática deshabilitada")
            except Exception as e:
                self.logger.error(f"Error en ciclo de procesamiento: {str(e)}")
            
            # Esperar hasta el próximo ciclo
            time.sleep(interval)
    
    def _get_pending_signals(self) -> List[Dict]:
        """
        Obtiene señales pendientes de procesar
        
        Returns:
            Lista de señales pendientes
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Configuración de procesamiento
            max_signals = self.config['processing'].get('max_signals_per_run', 5)
            max_age_days = self.config['processing'].get('max_age_days', 1)
            min_confidence = self.config['execution'].get('min_confidence', 0.7)
            
            # Patrones permitidos
            allowed_patterns = self.config['filters'].get('patterns', [])
            pattern_filter = ""
            if allowed_patterns:
                patterns_str = ', '.join([f"'{p}'" for p in allowed_patterns])
                pattern_filter = f"AND (pattern_type LIKE ANY(array[{patterns_str}])) "
            
            # Consultar señales pendientes
            query = f"""
            SELECT id, symbol, timeframe, pattern_type, confidence, date, entry_price, 
                   stop_loss_price, target_price, timestamp
            FROM detected_patterns
            WHERE notified = TRUE 
            AND triggered = FALSE
            AND timestamp >= %s
            AND confidence >= %s
            {pattern_filter}
            ORDER BY confidence DESC, timestamp DESC
            LIMIT %s;
            """
            
            # Calcular fecha mínima
            min_date = dt.datetime.now() - dt.timedelta(days=max_age_days)
            
            cursor.execute(query, (min_date, min_confidence, max_signals))
            signals = cursor.fetchall()
            cursor.close()
            
            self.logger.info(f"Encontradas {len(signals)} señales pendientes de procesar")
            return signals
        except Exception as e:
            self.logger.error(f"Error al obtener señales pendientes: {str(e)}")
            return []
    
    def _update_signal_status(self, signal_id: int, result: Dict) -> None:
        """
        Actualiza el estado de una señal
        
        Args:
            signal_id: ID de la señal
            result: Resultado del procesamiento
        """
        try:
            cursor = self.db_conn.cursor()
            
            if result.get('success', False):
                # Marcar como procesada
                query = """
                UPDATE detected_patterns
                SET triggered = TRUE, result = %s
                WHERE id = %s;
                """
                cursor.execute(query, ('success', signal_id))
            else:
                # Marcar como error
                query = """
                UPDATE detected_patterns
                SET result = %s
                WHERE id = %s;
                """
                cursor.execute(query, (f"error: {result.get('error', 'unknown')}", signal_id))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al actualizar estado de señal {signal_id}: {str(e)}")
    
    def _link_order_to_signal(self, order_id: str, signal_id: int) -> None:
        """
        Vincula una orden con la señal que la generó
        
        Args:
            order_id: ID de la orden
            signal_id: ID de la señal
        """
        try:
            cursor = self.db_conn.cursor()
            
            query = """
            UPDATE trading_orders
            SET pattern_id = %s
            WHERE order_id = %s;
            """
            
            cursor.execute(query, (signal_id, order_id))
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al vincular orden {order_id} con señal {signal_id}: {str(e)}")
    
    def process_signal(self, signal: Dict) -> Dict:
        """
        Procesa una señal de trading
        
        Args:
            signal: Señal a procesar
                {
                    "id": 123,
                    "symbol": "AMXL.MX",
                    "pattern_type": "ma_crossover_bullish",
                    "confidence": 0.85,
                    "entry_price": 15.75,
                    "stop_loss_price": 15.25,
                    "target_price": 16.75,
                    ...
                }
                
        Returns:
            Dict con resultado del procesamiento
        """
        try:
            # 1. Validar la señal
            validation = self._validate_signal(signal)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': validation['reason'],
                    'signal_id': signal['id']
                }
            
            # 2. Obtener tamaño de posición del gestor de riesgos
            position_size = self._calculate_position_size(signal)
            
            if not position_size.get('position_size', 0) > 0:
                return {
                    'success': False,
                    'error': position_size.get('error', 'No se pudo calcular tamaño de posición'),
                    'signal_id': signal['id'],
                    'risk_data': position_size
                }
            
            # 3. Verificar datos actuales de mercado
            market_data = self._get_current_market_data(signal['symbol'])
            
            if 'error' in market_data:
                return {
                    'success': False,
                    'error': f"Error al obtener datos de mercado: {market_data['error']}",
                    'signal_id': signal['id']
                }
            
            # 4. Ajustar precios según datos actuales
            prices = self._adjust_prices(signal, market_data)
            
            # 5. Ejecutar la operación
            order_result = self._execute_order(signal['symbol'], signal['pattern_type'], 
                                             position_size['position_size'], prices)
            
            if order_result.get('success', False):
                # Generar alerta de ejecución
                self._generate_execution_alert(signal, position_size, order_result)
                
                return {
                    'success': True,
                    'signal_id': signal['id'],
                    'order_id': order_result.get('order_id', ''),
                    'position_size': position_size['position_size'],
                    'entry_price': prices['entry_price'],
                    'stop_loss': prices['stop_loss'],
                    'take_profit': prices['take_profit']
                }
            else:
                # Generar alerta de error
                self._generate_error_alert(signal, order_result)
                
                return {
                    'success': False,
                    'error': order_result.get('error', 'Error desconocido al ejecutar orden'),
                    'signal_id': signal['id'],
                    'order_data': order_result
                }
        except Exception as e:
            self.logger.error(f"Error al procesar señal {signal['id']}: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'signal_id': signal['id']
            }
    
    def _validate_signal(self, signal: Dict) -> Dict:
        """
        Valida una señal antes de procesarla
        
        Args:
            signal: Señal a validar
            
        Returns:
            Dict con resultado de validación
        """
        # Verificar campos requeridos
        required_fields = ['id', 'symbol', 'pattern_type', 'confidence', 'entry_price', 
                          'stop_loss_price', 'target_price']
        
        for field in required_fields:
            if field not in signal or signal[field] is None:
                return {
                    'valid': False,
                    'reason': f"Campo requerido faltante o nulo: {field}"
                }
        
        # Verificar confianza mínima
        min_confidence = self.config['execution'].get('min_confidence', 0.7)
        if signal['confidence'] < min_confidence:
            return {
                'valid': False,
                'reason': f"Confianza ({signal['confidence']}) menor que mínimo requerido ({min_confidence})"
            }
        
        # Verificar patrón permitido
        allowed_patterns = self.config['filters'].get('patterns', [])
        if allowed_patterns:
            pattern_base = signal['pattern_type'].split('_')[0]
            if pattern_base not in allowed_patterns:
                return {
                    'valid': False,
                    'reason': f"Patrón {signal['pattern_type']} no está en la lista de permitidos"
                }
        
        # Verificar precios válidos
        if signal['entry_price'] <= 0 or signal['stop_loss_price'] <= 0 or signal['target_price'] <= 0:
            return {
                'valid': False,
                'reason': "Precios inválidos (deben ser mayores que cero)"
            }
        
        # Verificar ratio riesgo/recompensa
        direction = 'bullish' if 'bullish' in signal['pattern_type'] else 'bearish'
        
        if direction == 'bullish':
            risk = signal['entry_price'] - signal['stop_loss_price']
            reward = signal['target_price'] - signal['entry_price']
        else:  # bearish
            risk = signal['stop_loss_price'] - signal['entry_price']
            reward = signal['entry_price'] - signal['target_price']
        
        if risk <= 0:
            return {
                'valid': False,
                'reason': "Riesgo inválido (stop loss debe ser menor que entrada para señales alcistas o mayor para bajistas)"
            }
        
        risk_reward_ratio = reward / risk if risk > 0 else 0
        min_rr_ratio = self.config['execution'].get('risk_reward_min', 1.5)
        
        if risk_reward_ratio < min_rr_ratio:
            return {
                'valid': False,
                'reason': f"Ratio riesgo/recompensa ({risk_reward_ratio:.2f}) menor que mínimo requerido ({min_rr_ratio})"
            }
        
        # Si pasa todas las validaciones
        return {
            'valid': True,
            'risk_reward_ratio': risk_reward_ratio
        }
    
    def _calculate_position_size(self, signal: Dict) -> Dict:
        """
        Calcula el tamaño de posición utilizando el servicio de gestión de riesgos
        
        Args:
            signal: Señal para la que calcular tamaño
            
        Returns:
            Dict con tamaño de posición y métricas de riesgo
        """
        try:
            # Preparar datos para el servicio de riesgo
            risk_params = {
                "symbol": signal['symbol'],
                "entry_price": signal['entry_price'],
                "stop_loss_price": signal['stop_loss_price'],
                "pattern_type": signal['pattern_type'],
                "confidence": signal['confidence']
            }
            
            # Llamar al servicio de riesgo
            response = requests.post(f"{self.risk_service_url}/position-size", 
                                   json=risk_params, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(f"Error del servicio de riesgo: {response.text}")
                return {
                    "error": f"Error del servicio de riesgo: {response.status_code}",
                    "position_size": 0
                }
        except requests.RequestException as e:
            self.logger.error(f"Error al conectar con servicio de riesgo: {str(e)}")
            return {
                "error": f"Error al conectar con servicio de riesgo: {str(e)}",
                "position_size": 0
            }
        except Exception as e:
            self.logger.error(f"Error al calcular tamaño de posición: {str(e)}")
            return {
                "error": str(e),
                "position_size": 0
            }
    
    def _get_current_market_data(self, symbol: str) -> Dict:
        """
        Obtiene datos actuales de mercado
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Dict con datos de mercado
        """
        try:
            # Intentar obtener de IBKR
            response = requests.get(f"{self.ibkr_service_url}/market-data/{symbol}", timeout=5)
            
            if response.status_code == 200:
                market_data = response.json()
                
                if 'data' in market_data:
                    return market_data['data']
                else:
                    return {"error": "Datos no disponibles en respuesta de IBKR"}
            else:
                # Si falla, intentar obtener de base de datos
                return self._get_latest_market_data_from_db(symbol)
        except requests.RequestException as e:
            self.logger.warning(f"Error al conectar con IBKR: {str(e)}, intentando base de datos")
            return self._get_latest_market_data_from_db(symbol)
        except Exception as e:
            self.logger.error(f"Error al obtener datos de mercado: {str(e)}")
            return {"error": str(e)}
    
    def _get_latest_market_data_from_db(self, symbol: str) -> Dict:
        """
        Obtiene los últimos datos de mercado desde la base de datos
        
        Args:
            symbol: Símbolo a consultar
            
        Returns:
            Dict con datos de mercado
        """
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            query = """
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT 1;
            """
            
            cursor.execute(query, (symbol,))
            data = cursor.fetchone()
            cursor.close()
            
            if data:
                return {
                    "symbol": symbol,
                    "last_price": data['close'],
                    "bid": data['close'] * 0.999,  # Estimación
                    "ask": data['close'] * 1.001,  # Estimación
                    "high": data['high'],
                    "low": data['low'],
                    "volume": data['volume'],
                    "timestamp": data['date'].isoformat(),
                    "source": "database"
                }
            else:
                return {"error": f"No hay datos para {symbol} en base de datos"}
        except Exception as e:
            self.logger.error(f"Error al obtener datos de mercado desde BD: {str(e)}")
            return {"error": str(e)}
    
    def _adjust_prices(self, signal: Dict, market_data: Dict) -> Dict:
        """
        Ajusta precios de entrada, stop loss y take profit según datos actuales
        
        Args:
            signal: Señal original
            market_data: Datos actuales de mercado
            
        Returns:
            Dict con precios ajustados
        """
        # Obtener precios actuales
        last_price = market_data.get('last_price')
        bid = market_data.get('bid')
        ask = market_data.get('ask')
        
        if not last_price:
            # Si no hay precio actual, usar precios originales
            return {
                'entry_price': signal['entry_price'],
                'stop_loss': signal['stop_loss_price'],
                'take_profit': signal['target_price']
            }
        
        # Detectar dirección (compra o venta)
        is_bullish = 'bullish' in signal['pattern_type']
        direction = 'buy' if is_bullish else 'sell'
        
        # Ajustar precios según dirección y tipo de orden
        order_type = self.config['execution'].get('default_order_type', 'LIMIT')
        buffer = self.config['execution'].get('limit_price_buffer', 0.002)  # 0.2%
        
        if direction == 'buy':
            # Para órdenes de compra
            if order_type == 'MARKET':
                # Usar precio actual para cálculos
                entry_price = ask if ask else last_price
            else:  # LIMIT
                # Usar precio ligeramente por debajo para mayor probabilidad de ejecución
                entry_price = last_price * (1 - buffer) if last_price else signal['entry_price']
            
            # Ajustar stop loss y take profit proporcionalmente
            original_risk = signal['entry_price'] - signal['stop_loss_price']
            original_reward = signal['target_price'] - signal['entry_price']
            
            risk_ratio = original_risk / signal['entry_price'] if signal['entry_price'] > 0 else 0
            reward_ratio = original_reward / signal['entry_price'] if signal['entry_price'] > 0 else 0
            
            stop_loss = entry_price * (1 - risk_ratio)
            take_profit = entry_price * (1 + reward_ratio)
        else:
            # Para órdenes de venta
            if order_type == 'MARKET':
                # Usar precio actual para cálculos
                entry_price = bid if bid else last_price
            else:  # LIMIT
                # Usar precio ligeramente por encima para mayor probabilidad de ejecución
                entry_price = last_price * (1 + buffer) if last_price else signal['entry_price']
            
            # Ajustar stop loss y take profit proporcionalmente
            original_risk = signal['stop_loss_price'] - signal['entry_price']
            original_reward = signal['entry_price'] - signal['target_price']
            
            risk_ratio = original_risk / signal['entry_price'] if signal['entry_price'] > 0 else 0
            reward_ratio = original_reward / signal['entry_price'] if signal['entry_price'] > 0 else 0
            
            stop_loss = entry_price * (1 + risk_ratio)
            take_profit = entry_price * (1 - reward_ratio)
        
        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction
        }
    
    def _execute_order(self, symbol: str, pattern_type: str, quantity: int, prices: Dict) -> Dict:
        """
        Ejecuta una orden a través del servicio de IBKR
        
        Args:
            symbol: Símbolo para la orden
            pattern_type: Tipo de patrón que generó la orden
            quantity: Cantidad de acciones
            prices: Precios ajustados
            
        Returns:
            Dict con resultado de la ejecución
        """
        try:
            # Extraer datos
            direction = prices.get('direction', 'buy').upper()
            entry_price = prices.get('entry_price')
            stop_loss = prices.get('stop_loss')
            take_profit = prices.get('take_profit')
            
            # Verificar si se debe usar bracket order o simple
            use_bracket = (self.config['execution'].get('stop_loss_enabled', True) and 
                          self.config['execution'].get('take_profit_enabled', True))
            
            if use_bracket:
                # Crear orden bracket (entrada + stop loss + take profit)
                order_params = {
                    "symbol": symbol,
                    "action": direction,
                    "quantity": quantity,
                    "entry_price": entry_price,
                    "stop_loss_price": stop_loss,
                    "take_profit_price": take_profit,
                    "entry_order_type": self.config['execution'].get('default_order_type', 'LIMIT'),
                    "outside_rth": self.config['execution'].get('outside_rth', False),
                    "tag": f"Pattern: {pattern_type}"
                }
                
                # Enviar al servicio de IBKR
                response = requests.post(f"{self.ibkr_service_url}/bracket-order", 
                                       json=order_params, timeout=10)
            else:
                # Crear orden simple
                order_type = self.config['execution'].get('default_order_type', 'LIMIT')
                
                order_params = {
                    "symbol": symbol,
                    "action": direction,
                    "order_type": order_type,
                    "quantity": quantity,
                    "outside_rth": self.config['execution'].get('outside_rth', False),
                    "tag": f"Pattern: {pattern_type}"
                }
                
                # Añadir precio límite si es necesario
                if order_type == "LIMIT":
                    order_params["limit_price"] = entry_price
                
                # Enviar al servicio de IBKR
                response = requests.post(f"{self.ibkr_service_url}/order", 
                                       json=order_params, timeout=10)
            
            # Procesar respuesta
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    self.logger.info(f"Orden ejecutada: {symbol} {direction} x{quantity} @ {entry_price}")
                    
                    # Si es bracket, guardar IDs de órdenes secundarias
                    if use_bracket and 'parent_order_id' in result:
                        # Guardar en base de datos
                        self._save_bracket_order_details(
                            result['parent_order_id'],
                            result.get('stop_loss_order_id'),
                            result.get('take_profit_order_id'),
                            symbol, direction, quantity, entry_price, stop_loss, take_profit
                        )
                        
                        return {
                            'success': True,
                            'order_id': result['parent_order_id'],
                            'stop_loss_order_id': result.get('stop_loss_order_id'),
                            'take_profit_order_id': result.get('take_profit_order_id')
                        }
                    else:
                        # Guardar en base de datos
                        self._save_order_details(
                            result['order_id'], symbol, direction, quantity, 
                            entry_price, stop_loss, take_profit, order_type
                        )
                        
                        return {
                            'success': True,
                            'order_id': result['order_id']
                        }
                else:
                    self.logger.error(f"Error al ejecutar orden: {result.get('error', 'Error desconocido')}")
                    return {
                        'success': False,
                        'error': result.get('error', 'Error desconocido'),
                        'ibkr_response': result
                    }
            else:
                self.logger.error(f"Error del servicio IBKR: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Error del servicio IBKR: {response.status_code}",
                    'response_text': response.text
                }
        except requests.RequestException as e:
            self.logger.error(f"Error al conectar con servicio IBKR: {str(e)}")
            return {
                'success': False,
                'error': f"Error al conectar con servicio IBKR: {str(e)}"
            }
        except Exception as e:
            self.logger.error(f"Error al ejecutar orden: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _save_order_details(self, order_id: str, symbol: str, direction: str, 
                           quantity: int, entry_price: float, stop_loss: float, 
                           take_profit: float, order_type: str) -> None:
        """
        Guarda detalles de una orden en base de datos
        
        Args:
            order_id: ID de la orden
            symbol: Símbolo
            direction: Dirección (BUY/SELL)
            quantity: Cantidad
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
            order_type: Tipo de orden
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Guardar en trading_orders
            query = """
            INSERT INTO trading_orders
            (order_id, symbol, direction, order_type, quantity, price, status, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            cursor.execute(query, (
                order_id, symbol, direction, order_type, quantity, 
                entry_price, 'pending', dt.datetime.now()
            ))
            
            # Si es BUY también crear entrada en trading_positions
            if direction == 'BUY':
                query = """
                INSERT INTO trading_positions
                (symbol, direction, quantity, entry_price, entry_time, stop_loss, current_price, last_update)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                cursor.execute(query, (
                    symbol, 'long', quantity, entry_price, dt.datetime.now(),
                    stop_loss, entry_price, dt.datetime.now()
                ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar detalles de orden: {str(e)}")
    
    def _save_bracket_order_details(self, parent_id: str, stop_loss_id: str, 
                                   take_profit_id: str, symbol: str, direction: str, 
                                   quantity: int, entry_price: float, stop_loss: float, 
                                   take_profit: float) -> None:
        """
        Guarda detalles de una orden bracket en base de datos
        
        Args:
            parent_id: ID de orden principal
            stop_loss_id: ID de orden stop loss
            take_profit_id: ID de orden take profit
            symbol: Símbolo
            direction: Dirección (BUY/SELL)
            quantity: Cantidad
            entry_price: Precio de entrada
            stop_loss: Precio de stop loss
            take_profit: Precio de take profit
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Guardar orden principal
            query = """
            INSERT INTO trading_orders
            (order_id, symbol, direction, order_type, quantity, price, status, timestamp, notes)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            
            order_type = self.config['execution'].get('default_order_type', 'LIMIT')
            timestamp = dt.datetime.now()
            
            # Orden principal
            cursor.execute(query, (
                parent_id, symbol, direction, order_type, quantity, 
                entry_price, 'pending', timestamp, 'Bracket order parent'
            ))
            
            # Orden stop loss
            reverse_direction = 'SELL' if direction == 'BUY' else 'BUY'
            cursor.execute(query, (
                stop_loss_id, symbol, reverse_direction, 'STOP', quantity, 
                stop_loss, 'pending', timestamp, f'Stop loss for {parent_id}'
            ))
            
            # Orden take profit
            cursor.execute(query, (
                take_profit_id, symbol, reverse_direction, 'LIMIT', quantity, 
                take_profit, 'pending', timestamp, f'Take profit for {parent_id}'
            ))
            
            # Si es BUY también crear entrada en trading_positions
            if direction == 'BUY':
                query = """
                INSERT INTO trading_positions
                (symbol, direction, quantity, entry_price, entry_time, stop_loss, current_price, last_update)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
                """
                
                cursor.execute(query, (
                    symbol, 'long', quantity, entry_price, timestamp,
                    stop_loss, entry_price, timestamp
                ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar detalles de orden bracket: {str(e)}")
    
    def _generate_execution_alert(self, signal: Dict, position_size: Dict, order_result: Dict) -> None:
        """
        Genera una alerta de ejecución de orden
        
        Args:
            signal: Señal procesada
            position_size: Información de tamaño de posición
            order_result: Resultado de la ejecución
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Construir mensaje
            direction = 'compra' if 'bullish' in signal['pattern_type'] else 'venta'
            pattern_name = signal['pattern_type'].replace('_bullish', '').replace('_bearish', '')
            
            message = f"🎯 ORDEN EJECUTADA: {direction.upper()} para {signal['symbol']}, "
            message += f"{position_size.get('position_size', 0)} acciones a ${position_size.get('entry_price', 0):.2f}\n"
            message += f"Stop Loss: ${position_size.get('stop_loss_price', 0):.2f}, "
            message += f"Objetivo: ${position_size.get('target_price', 0):.2f}\n"
            message += f"Patrón: {pattern_name.upper()} (confianza: {signal['confidence']:.2f})\n"
            message += f"Riesgo: ${position_size.get('risk_amount', 0):.2f} ({position_size.get('risk_percentage', 0):.2f}%)"
            
            # Insertar alerta
            query = """
            INSERT INTO alerts
            (timestamp, alert_type, message)
            VALUES (%s, %s, %s);
            """
            
            cursor.execute(query, (
                dt.datetime.now(),
                'order_execution',
                message
            ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al generar alerta de ejecución: {str(e)}")
    
    def _generate_error_alert(self, signal: Dict, error_result: Dict) -> None:
        """
        Genera una alerta de error en ejecución
        
        Args:
            signal: Señal procesada
            error_result: Resultado del error
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Construir mensaje
            direction = 'compra' if 'bullish' in signal['pattern_type'] else 'venta'
            pattern_name = signal['pattern_type'].replace('_bullish', '').replace('_bearish', '')
            
            message = f"⚠️ ERROR EN ORDEN: {direction.upper()} para {signal['symbol']}\n"
            message += f"Patrón: {pattern_name.upper()} (confianza: {signal['confidence']:.2f})\n"
            message += f"Error: {error_result.get('error', 'Error desconocido')}"
            
            # Insertar alerta
            query = """
            INSERT INTO alerts
            (timestamp, alert_type, message)
            VALUES (%s, %s, %s);
            """
            
            cursor.execute(query, (
                dt.datetime.now(),
                'order_failed',
                message
            ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al generar alerta de error: {str(e)}")
    
    def execute_trade(self, trade_params: Dict) -> Dict:
        """
        Ejecuta una operación de trading (endpoint API)
        
        Args:
            trade_params: Parámetros de la operación
                {
                    "symbol": "AMXL.MX",
                    "pattern_type": "manual",
                    "direction": "buy",
                    "quantity": 100,
                    "entry_price": 15.75,
                    "stop_loss": 15.25,
                    "take_profit": 16.75,
                    "order_type": "LIMIT"
                }
                
        Returns:
            Dict con resultado de la ejecución
        """
        try:
            # Validar parámetros
            required_params = ["symbol", "direction", "quantity"]
            for param in required_params:
                if param not in trade_params:
                    return {"success": False, "error": f"Parámetro requerido no encontrado: {param}"}
            
            # Obtener precios actuales si no se proporcionan
            if 'entry_price' not in trade_params or trade_params['entry_price'] is None:
                market_data = self._get_current_market_data(trade_params['symbol'])
                if 'error' in market_data:
                    return {
                        "success": False,
                        "error": f"No se pudo obtener precio actual: {market_data['error']}"
                    }
                
                # Usar precio bid/ask según dirección
                direction = trade_params['direction'].lower()
                if direction == 'buy':
                    trade_params['entry_price'] = market_data.get('ask', market_data.get('last_price'))
                else:  # sell
                    trade_params['entry_price'] = market_data.get('bid', market_data.get('last_price'))
            
            # Preparar parámetros de orden
            order_type = trade_params.get('order_type', self.config['execution'].get('default_order_type', 'LIMIT'))
            
            # Verificar si se debe usar orden bracket
            use_bracket = ('stop_loss' in trade_params and 'take_profit' in trade_params and
                          trade_params['stop_loss'] is not None and trade_params['take_profit'] is not None)
            
            if use_bracket:
                # Crear orden bracket
                order_params = {
                    "symbol": trade_params['symbol'],
                    "action": trade_params['direction'].upper(),
                    "quantity": trade_params['quantity'],
                    "entry_price": trade_params['entry_price'],
                    "stop_loss_price": trade_params['stop_loss'],
                    "take_profit_price": trade_params['take_profit'],
                    "entry_order_type": order_type,
                    "outside_rth": trade_params.get('outside_rth', self.config['execution'].get('outside_rth', False)),
                    "tag": f"Manual: {trade_params.get('pattern_type', 'manual')}"
                }
                
                # Enviar al servicio de IBKR
                response = requests.post(f"{self.ibkr_service_url}/bracket-order", 
                                       json=order_params, timeout=10)
            else:
                # Crear orden simple
                order_params = {
                    "symbol": trade_params['symbol'],
                    "action": trade_params['direction'].upper(),
                    "order_type": order_type,
                    "quantity": trade_params['quantity'],
                    "outside_rth": trade_params.get('outside_rth', self.config['execution'].get('outside_rth', False)),
                    "tag": f"Manual: {trade_params.get('pattern_type', 'manual')}"
                }
                
                # Añadir precio límite si es necesario
                if order_type == "LIMIT":
                    order_params["limit_price"] = trade_params['entry_price']
                
                # Enviar al servicio de IBKR
                response = requests.post(f"{self.ibkr_service_url}/order", 
                                       json=order_params, timeout=10)
            
            # Procesar respuesta
            if response.status_code == 200:
                result = response.json()
                
                if result.get('success', False):
                    self.logger.info(f"Orden manual ejecutada: {trade_params['symbol']} "
                                    f"{trade_params['direction']} x{trade_params['quantity']}")
                    
                    # Generar alerta
                    self._generate_manual_execution_alert(trade_params, result)
                    
                    return result
                else:
                    self.logger.error(f"Error al ejecutar orden manual: {result.get('error', 'Error desconocido')}")
                    return result
            else:
                self.logger.error(f"Error del servicio IBKR: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"Error del servicio IBKR: {response.status_code}",
                    'response_text': response.text
                }
        except Exception as e:
            self.logger.error(f"Error al ejecutar operación manual: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _generate_manual_execution_alert(self, trade_params: Dict, result: Dict) -> None:
        """
        Genera una alerta para ejecución manual
        
        Args:
            trade_params: Parámetros de la operación
            result: Resultado de la ejecución
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Construir mensaje
            direction = 'COMPRA' if trade_params['direction'].lower() == 'buy' else 'VENTA'
            
            message = f"👤 ORDEN MANUAL: {direction} para {trade_params['symbol']}, "
            message += f"{trade_params['quantity']} acciones a ${trade_params.get('entry_price', 0):.2f}\n"
            
            if 'stop_loss' in trade_params and trade_params['stop_loss'] is not None:
                message += f"Stop Loss: ${trade_params['stop_loss']:.2f}, "
            
            if 'take_profit' in trade_params and trade_params['take_profit'] is not None:
                message += f"Objetivo: ${trade_params['take_profit']:.2f}\n"
            
            message += f"Tipo: {trade_params.get('order_type', 'LIMIT')}"
            
            # Insertar alerta
            query = """
            INSERT INTO alerts
            (timestamp, alert_type, message)
            VALUES (%s, %s, %s);
            """
            
            cursor.execute(query, (
                dt.datetime.now(),
                'order_execution',
                message
            ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al generar alerta de ejecución manual: {str(e)}")
    
    def _create_app(self) -> Flask:
        """
        Crea y configura la aplicación Flask
        
        Returns:
            Aplicación Flask configurada
        """
        app = Flask(__name__)
        
        # Definir rutas
        @app.route('/api/execute', methods=['POST'])
        def execute_trade():
            trade_params = request.json
            return jsonify(self.execute_trade(trade_params))
        
        @app.route('/api/signal', methods=['POST'])
        def process_signal():
            signal = request.json
            return jsonify(self.process_signal(signal))
        
        @app.route('/api/status', methods=['GET'])
        def get_status():
            # Verificar estado de servicios dependientes
            ibkr_status = {"connected": False}
            risk_status = {"status": "unknown"}
            
            try:
                # Verificar IBKR
                ibkr_response = requests.get(f"{self.ibkr_service_url}/status", timeout=2)
                if ibkr_response.status_code == 200:
                    ibkr_status = ibkr_response.json()
                
                # Verificar Risk Manager
                risk_response = requests.get(f"{self.risk_service_url}/status", timeout=2)
                if risk_response.status_code == 200:
                    risk_status = risk_response.json()
            except:
                pass
            
            return jsonify({
                "execution_enabled": self.config['execution'].get('enabled', True),
                "ibkr_status": ibkr_status,
                "risk_status": risk_status,
                "config": self.config
            })
        
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
    # Crear ejecutor de operaciones
    executor = TradeExecutor()
    
    # Ejecutar API
    executor.run_api()
