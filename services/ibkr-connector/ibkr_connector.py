import os
import json
import time
import logging
import threading
import pandas as pd
import psycopg2
import datetime as dt
from psycopg2.extras import RealDictCursor
from flask import Flask, jsonify, request
from typing import Dict, List, Optional, Union
from ib_insync import *

class IBKRConnector:
    def __init__(self, config_path: str = None):
        """
        Inicializa el conector con Interactive Brokers
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None, usa valores por defecto
                        o busca en la ubicación estándar.
        """
        # Configurar logger
        self._setup_logging()
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Inicializar conexión con IB
        self.ib = IB()
        self.connected = False
        self.last_error = None
        self.connection_time = None
        
        # Conectar con base de datos
        self.db_conn = self._get_db_connection()
        
        # Iniciar hilo para mantener conexión
        self.connection_thread = threading.Thread(target=self._maintain_connection, daemon=True)
        self.connection_thread.start()
        
        # Crear la aplicación Flask
        self.app = self._create_app()
        
        self.logger.info("Conector IBKR inicializado")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('ibkr_connector')
    
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
            "ibkr": {
                "host": os.environ.get('IBKR_HOST', '127.0.0.1'),
                "port": int(os.environ.get('IBKR_PORT', '7496')),
                "client_id": int(os.environ.get('IBKR_CLIENT_ID', '1')),
                "read_only": False,
                "reconnect_interval": 60,  # segundos
                "max_reconnect_attempts": 10
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8080
            },
            "order_defaults": {
                "market_order": {
                    "transmit": True,
                    "outsideRth": False,
                    "tif": "GTC"  # Good Till Cancelled
                },
                "limit_order": {
                    "transmit": True,
                    "outsideRth": False,
                    "tif": "GTC"
                },
                "stop_order": {
                    "transmit": True,
                    "outsideRth": False,
                    "tif": "GTC"
                }
            },
            "symbols": {
                "AMXL.MX": {"exchange": "MEXI", "currency": "MXN", "symbol_type": "STK"},
                "FEMSAUBD.MX": {"exchange": "MEXI", "currency": "MXN", "symbol_type": "STK"},
                "GFNORTEO.MX": {"exchange": "MEXI", "currency": "MXN", "symbol_type": "STK"},
                "WALMEX.MX": {"exchange": "MEXI", "currency": "MXN", "symbol_type": "STK"}
            }
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'ibkr.json')
        
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
    
    def connect_to_ibkr(self) -> bool:
        """
        Establece conexión con Interactive Brokers TWS o IB Gateway
        
        Returns:
            True si se conectó correctamente, False en caso contrario
        """
        # Si ya estamos conectados, no hacer nada
        if self.connected and self.ib.isConnected():
            return True
        
        # Intentar conectar
        try:
            host = self.config['ibkr']['host']
            port = self.config['ibkr']['port']
            client_id = self.config['ibkr']['client_id']
            read_only = self.config['ibkr']['read_only']
            
            self.logger.info(f"Conectando a IBKR en {host}:{port} con ID {client_id}")
            
            # Conectar
            self.ib.connect(host, port, clientId=client_id, readonly=read_only)
            
            # Verificar conexión
            if self.ib.isConnected():
                self.connected = True
                self.connection_time = dt.datetime.now()
                self.last_error = None
                self.logger.info("Conexión con IBKR establecida correctamente")
                
                # Registrar error handlers
                self.ib.errorEvent += self._handle_error
                
                # Guardar estado de conexión en DB
                self._save_connection_status(True)
                
                return True
            else:
                self.connected = False
                self.last_error = "No se pudo conectar a IBKR"
                self.logger.error(self.last_error)
                
                # Guardar estado de conexión en DB
                self._save_connection_status(False)
                
                return False
        except Exception as e:
            self.connected = False
            self.last_error = str(e)
            self.logger.error(f"Error al conectar a IBKR: {self.last_error}")
            
            # Guardar estado de conexión en DB
            self._save_connection_status(False)
            
            return False
    
    def disconnect_from_ibkr(self) -> bool:
        """
        Cierra la conexión con Interactive Brokers
        
        Returns:
            True si se desconectó correctamente, False en caso contrario
        """
        try:
            if self.ib.isConnected():
                self.ib.disconnect()
                self.logger.info("Desconexión de IBKR exitosa")
            
            self.connected = False
            
            # Guardar estado de conexión en DB
            self._save_connection_status(False)
            
            return True
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Error al desconectar de IBKR: {self.last_error}")
            return False
    
    def _maintain_connection(self) -> None:
        """
        Hilo para mantener la conexión con IBKR activa
        """
        reconnect_interval = self.config['ibkr']['reconnect_interval']
        max_attempts = self.config['ibkr']['max_reconnect_attempts']
        attempts = 0
        
        while True:
            try:
                # Si no estamos conectados, intentar reconectar
                if not self.connected or not self.ib.isConnected():
                    attempts += 1
                    self.logger.info(f"Intento de reconexión {attempts}/{max_attempts}")
                    
                    # Intentar conectar
                    success = self.connect_to_ibkr()
                    
                    if success:
                        attempts = 0
                    elif attempts >= max_attempts:
                        self.logger.error(f"Se alcanzó el máximo de intentos de reconexión ({max_attempts})")
                        time.sleep(reconnect_interval * 5)  # Esperar más tiempo antes de volver a intentar
                        attempts = 0
                else:
                    # Si estamos conectados, verificar estado periódicamente
                    self.logger.debug("Conexión IBKR activa")
                    attempts = 0
            except Exception as e:
                self.logger.error(f"Error en hilo de mantenimiento de conexión: {str(e)}")
            
            # Esperar antes del siguiente intento
            time.sleep(reconnect_interval)
    
    def _handle_error(self, reqId, errorCode, errorString, contract) -> None:
        """
        Maneja errores reportados por IB
        
        Args:
            reqId: ID de la solicitud
            errorCode: Código de error
            errorString: Descripción del error
            contract: Contrato relacionado con el error
        """
        # Ignorar errores informativos
        if errorCode in [2104, 2106, 2158]:
            self.logger.debug(f"IB Info ({errorCode}): {errorString}")
            return
        
        # Loguear error
        symbol = contract.symbol if contract else 'unknown'
        error_msg = f"IB Error {errorCode} para {symbol}: {errorString}"
        self.logger.error(error_msg)
        
        # Almacenar último error
        self.last_error = error_msg
        
        # Guardar error en base de datos
        self._save_error_to_db(reqId, errorCode, errorString, symbol)
        
        # Si es un error crítico que indica problemas de conexión, intentar reconectar
        if errorCode in [1100, 1101, 1102, 1300, 2110]:
            self.logger.warning("Error crítico de conexión detectado, intentando reconectar...")
            self.disconnect_from_ibkr()
            self.connected = False
    
    def _save_error_to_db(self, req_id: int, error_code: int, error_msg: str, symbol: str) -> None:
        """
        Guarda un error en la base de datos
        
        Args:
            req_id: ID de la solicitud
            error_code: Código de error
            error_msg: Descripción del error
            symbol: Símbolo relacionado con el error
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Crear tabla si no existe
            create_table_query = """
            CREATE TABLE IF NOT EXISTS ibkr_errors (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                req_id INTEGER,
                error_code INTEGER NOT NULL,
                error_message TEXT NOT NULL,
                symbol VARCHAR(20)
            );
            """
            cursor.execute(create_table_query)
            self.db_conn.commit()
            
            # Insertar error
            insert_query = """
            INSERT INTO ibkr_errors
            (timestamp, req_id, error_code, error_message, symbol)
            VALUES (%s, %s, %s, %s, %s);
            """
            cursor.execute(insert_query, (
                dt.datetime.now(), req_id, error_code, error_msg, symbol
            ))
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar error en base de datos: {str(e)}")
    
    def _save_connection_status(self, connected: bool) -> None:
        """
        Guarda el estado de conexión en la base de datos
        
        Args:
            connected: True si está conectado, False en caso contrario
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Crear tabla si no existe
            create_table_query = """
            CREATE TABLE IF NOT EXISTS ibkr_connection_status (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL,
                connected BOOLEAN NOT NULL,
                last_error TEXT
            );
            """
            cursor.execute(create_table_query)
            self.db_conn.commit()
            
            # Insertar estado
            insert_query = """
            INSERT INTO ibkr_connection_status
            (timestamp, connected, last_error)
            VALUES (%s, %s, %s);
            """
            cursor.execute(insert_query, (
                dt.datetime.now(), connected, self.last_error
            ))
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar estado de conexión en base de datos: {str(e)}")
    
    def get_account_summary(self) -> Dict:
        """
        Obtiene un resumen de la cuenta
        
        Returns:
            Dict con información de la cuenta
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"error": "No conectado a IBKR", "connected": False}
        
        try:
            # Obtener información de cuenta
            account_values = self.ib.accountSummary()
            
            # Agrupar valores por etiqueta y moneda
            summary = {}
            for av in account_values:
                if av.tag not in summary:
                    summary[av.tag] = {}
                summary[av.tag][av.currency] = float(av.value)
            
            # Obtener posiciones
            positions = self.ib.positions()
            pos_summary = []
            
            for pos in positions:
                contract = pos.contract
                pos_summary.append({
                    "symbol": contract.symbol,
                    "exchange": contract.exchange,
                    "currency": contract.currency,
                    "position": pos.position,
                    "avg_cost": pos.avgCost,
                    "market_value": pos.marketValue if hasattr(pos, 'marketValue') else None
                })
            
            # Guardar en base de datos
            self._save_account_summary(summary, pos_summary)
            
            return {
                "connected": True,
                "account": summary,
                "positions": pos_summary,
                "timestamp": dt.datetime.now().isoformat()
            }
        except Exception as e:
            error_msg = f"Error al obtener resumen de cuenta: {str(e)}"
            self.logger.error(error_msg)
            self.last_error = error_msg
            return {"error": error_msg, "connected": self.connected}
    
    def _save_account_summary(self, account_summary: Dict, positions_summary: List) -> None:
        """
        Guarda el resumen de cuenta en la base de datos
        
        Args:
            account_summary: Dict con información de cuenta
            positions_summary: Lista de posiciones
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Obtener saldo y equidad
            account_balance = 0
            equity = 0
            margin_used = 0
            free_margin = 0
            
            if 'NetLiquidation' in account_summary and 'USD' in account_summary['NetLiquidation']:
                equity = account_summary['NetLiquidation']['USD']
            
            if 'TotalCashValue' in account_summary and 'USD' in account_summary['TotalCashValue']:
                account_balance = account_summary['TotalCashValue']['USD']
            
            if 'MaintMarginReq' in account_summary and 'USD' in account_summary['MaintMarginReq']:
                margin_used = account_summary['MaintMarginReq']['USD']
            
            if 'AvailableFunds' in account_summary and 'USD' in account_summary['AvailableFunds']:
                free_margin = account_summary['AvailableFunds']['USD']
            
            # Calcular nivel de margen
            margin_level = (equity / margin_used * 100) if margin_used > 0 else 0
            
            # Insertar en account_status
            insert_query = """
            INSERT INTO account_status
            (timestamp, account_balance, equity, margin_used, free_margin, margin_level, open_positions, daily_pnl)
            VALUES (%s, %s, %s, %s, %s, %s, %s, calculate_daily_pnl());
            """
            cursor.execute(insert_query, (
                dt.datetime.now(), account_balance, equity, margin_used, 
                free_margin, margin_level, len(positions_summary)
            ))
            self.db_conn.commit()
            
            # Actualizar posiciones en trading_positions
            for pos in positions_summary:
                # Verificar si la posición ya existe
                check_query = """
                SELECT id FROM trading_positions
                WHERE symbol = %s AND direction = %s;
                """
                cursor.execute(check_query, (
                    pos['symbol'], 'long' if pos['position'] > 0 else 'short'
                ))
                existing = cursor.fetchone()
                
                if existing:
                    # Actualizar posición existente
                    update_query = """
                    UPDATE trading_positions
                    SET quantity = %s, current_price = %s, last_update = %s
                    WHERE id = %s;
                    """
                    cursor.execute(update_query, (
                        abs(pos['position']), pos['avg_cost'], dt.datetime.now(), existing[0]
                    ))
                else:
                    # Crear nueva posición si cantidad > 0
                    if pos['position'] != 0:
                        insert_pos_query = """
                        INSERT INTO trading_positions
                        (symbol, direction, quantity, entry_price, entry_time, current_price, last_update)
                        VALUES (%s, %s, %s, %s, %s, %s, %s);
                        """
                        cursor.execute(insert_pos_query, (
                            pos['symbol'], 'long' if pos['position'] > 0 else 'short',
                            abs(pos['position']), pos['avg_cost'], dt.datetime.now(),
                            pos['avg_cost'], dt.datetime.now()
                        ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar resumen de cuenta en base de datos: {str(e)}")
    
    def get_contract(self, symbol: str) -> Optional[Contract]:
        """
        Obtiene un contrato para un símbolo
        
        Args:
            symbol: Símbolo del contrato
            
        Returns:
            Contrato de IB o None si no se encontró
        """
        if symbol not in self.config['symbols']:
            self.logger.error(f"Símbolo no configurado: {symbol}")
            return None
        
        try:
            symbol_config = self.config['symbols'][symbol]
            
            contract = Contract()
            contract.symbol = symbol.split('.')[0]  # Eliminar sufijo .MX
            contract.secType = symbol_config.get('symbol_type', 'STK')
            contract.exchange = symbol_config.get('exchange', 'MEXI')
            contract.currency = symbol_config.get('currency', 'MXN')
            
            # Calificar el contrato para obtener los detalles completos
            if self.connected and self.ib.isConnected():
                contracts = self.ib.qualifyContracts(contract)
                if contracts:
                    return contracts[0]
            
            return contract
        except Exception as e:
            self.logger.error(f"Error al obtener contrato para {symbol}: {str(e)}")
            return None
    
    def get_market_data(self, symbol: str) -> Dict:
        """
        Obtiene datos de mercado para un símbolo
        
        Args:
            symbol: Símbolo para el que obtener datos
            
        Returns:
            Dict con datos de mercado
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"error": "No conectado a IBKR", "connected": False}
        
        try:
            contract = self.get_contract(symbol)
            if not contract:
                return {"error": f"No se pudo obtener contrato para {symbol}", "connected": True}
            
            # Solicitar datos de mercado
            self.ib.reqMktData(contract)
            self.ib.sleep(1)  # Esperar a recibir datos
            
            # Obtener ticker
            ticker = self.ib.ticker(contract)
            
            # Extraer datos
            market_data = {
                "symbol": symbol,
                "last_price": ticker.last if hasattr(ticker, 'last') and ticker.last else None,
                "bid": ticker.bid if hasattr(ticker, 'bid') and ticker.bid else None,
                "ask": ticker.ask if hasattr(ticker, 'ask') and ticker.ask else None,
                "high": ticker.high if hasattr(ticker, 'high') and ticker.high else None,
                "low": ticker.low if hasattr(ticker, 'low') and ticker.low else None,
                "volume": ticker.volume if hasattr(ticker, 'volume') and ticker.volume else None,
                "open": ticker.open if hasattr(ticker, 'open') and ticker.open else None,
                "close": ticker.close if hasattr(ticker, 'close') and ticker.close else None,
                "timestamp": dt.datetime.now().isoformat()
            }
            
            return {"data": market_data, "connected": True}
        except Exception as e:
            error_msg = f"Error al obtener datos de mercado para {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return {"error": error_msg, "connected": self.connected}
    
    def place_order(self, order_params: Dict) -> Dict:
        """
        Coloca una orden en IBKR
        
        Args:
            order_params: Parámetros de la orden
                {
                    "symbol": "AMXL.MX",
                    "action": "BUY",  # o "SELL"
                    "order_type": "MARKET",  # o "LIMIT" o "STOP"
                    "quantity": 100,
                    "limit_price": 10.5,  # solo para órdenes LIMIT
                    "stop_price": 9.5,  # solo para órdenes STOP
                    "tif": "GTC",  # Time in force
                    "outside_rth": false,  # Fuera de horario de trading
                    "transmit": true,  # Transmitir orden inmediatamente
                    "parent_id": null,  # ID de orden padre para bracket/OCO
                    "tag": "Estrategia XYZ"  # Tag personalizado
                }
                
        Returns:
            Dict con resultado de la operación
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"success": False, "error": "No conectado a IBKR", "connected": False}
        
        try:
            # Validar parámetros
            required_params = ["symbol", "action", "order_type", "quantity"]
            for param in required_params:
                if param not in order_params:
                    return {"success": False, "error": f"Parámetro requerido no encontrado: {param}"}
            
            # Obtener contrato
            symbol = order_params["symbol"]
            contract = self.get_contract(symbol)
            if not contract:
                return {"success": False, "error": f"No se pudo obtener contrato para {symbol}"}
            
            # Crear orden según el tipo
            order_type = order_params["order_type"].upper()
            order = None
            
            if order_type == "MARKET":
                order = MarketOrder(
                    order_params["action"],
                    order_params["quantity"]
                )
            elif order_type == "LIMIT":
                if "limit_price" not in order_params:
                    return {"success": False, "error": "Precio límite no especificado para orden LIMIT"}
                
                order = LimitOrder(
                    order_params["action"],
                    order_params["quantity"],
                    order_params["limit_price"]
                )
            elif order_type == "STOP":
                if "stop_price" not in order_params:
                    return {"success": False, "error": "Precio stop no especificado para orden STOP"}
                
                order = StopOrder(
                    order_params["action"],
                    order_params["quantity"],
                    order_params["stop_price"]
                )
            else:
                return {"success": False, "error": f"Tipo de orden no soportado: {order_type}"}
            
            # Configurar parámetros adicionales
            default_config = self.config['order_defaults'][order_type.lower() + '_order']
            
            # Aplicar configuración por defecto y luego parámetros específicos
            order.tif = order_params.get("tif", default_config.get("tif", "GTC"))
            order.outsideRth = order_params.get("outside_rth", default_config.get("outsideRth", False))
            order.transmit = order_params.get("transmit", default_config.get("transmit", True))
            
            if "parent_id" in order_params and order_params["parent_id"]:
                order.parentId = order_params["parent_id"]
            
            # Colocar orden
            trade = self.ib.placeOrder(contract, order)
            
            # Esperar confirmación
            for _ in range(5):  # Intentar hasta 5 veces
                self.ib.sleep(1)
                if trade.orderStatus.status:
                    break
            
            # Guardar orden en base de datos
            order_id = trade.order.orderId
            self._save_order_to_db(order_id, order_params, trade)
            
            # Devolver resultado
            return {
                "success": True,
                "order_id": order_id,
                "status": trade.orderStatus.status,
                "filled": trade.orderStatus.filled,
                "remaining": trade.orderStatus.remaining,
                "avg_fill_price": trade.orderStatus.avgFillPrice,
                "last_fill_price": trade.orderStatus.lastFillPrice,
                "why_held": trade.orderStatus.whyHeld
            }
        except Exception as e:
            error_msg = f"Error al colocar orden: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "connected": self.connected}
    
    def place_bracket_order(self, order_params: Dict) -> Dict:
        """
        Coloca una orden bracket (entrada con stop loss y take profit)
        
        Args:
            order_params: Parámetros de la orden
                {
                    "symbol": "AMXL.MX",
                    "action": "BUY",  # o "SELL"
                    "quantity": 100,
                    "entry_price": 10.5,  # Precio de entrada (limit o None para market)
                    "stop_loss_price": 9.5,  # Precio de stop loss
                    "take_profit_price": 11.5,  # Precio de take profit
                    "entry_order_type": "LIMIT",  # o "MARKET"
                    "outside_rth": false,
                    "tag": "Estrategia XYZ"
                }
                
        Returns:
            Dict con resultado de la operación
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"success": False, "error": "No conectado a IBKR", "connected": False}
        
        try:
            # Validar parámetros
            required_params = ["symbol", "action", "quantity", "stop_loss_price", "take_profit_price"]
            for param in required_params:
                if param not in order_params:
                    return {"success": False, "error": f"Parámetro requerido no encontrado: {param}"}
            
            # Obtener contrato
            symbol = order_params["symbol"]
            contract = self.get_contract(symbol)
            if not contract:
                return {"success": False, "error": f"No se pudo obtener contrato para {symbol}"}
            
            # Crear orden principal
            entry_order_type = order_params.get("entry_order_type", "MARKET").upper()
            
            if entry_order_type == "MARKET":
                parent = MarketOrder(
                    order_params["action"],
                    order_params["quantity"],
                    transmit=False
                )
            elif entry_order_type == "LIMIT":
                if "entry_price" not in order_params:
                    return {"success": False, "error": "Precio de entrada no especificado para orden LIMIT"}
                
                parent = LimitOrder(
                    order_params["action"],
                    order_params["quantity"],
                    order_params["entry_price"],
                    transmit=False
                )
            else:
                return {"success": False, "error": f"Tipo de orden no soportado: {entry_order_type}"}
            
            # Configurar parámetros adicionales
            parent.outsideRth = order_params.get("outside_rth", False)
            
            # Crear órdenes bracket
            reverse_action = "SELL" if order_params["action"] == "BUY" else "BUY"
            
            # Stop loss
            stop_loss = StopOrder(
                reverse_action,
                order_params["quantity"],
                order_params["stop_loss_price"],
                transmit=False
            )
            
            # Take profit
            take_profit = LimitOrder(
                reverse_action,
                order_params["quantity"],
                order_params["take_profit_price"],
                transmit=True
            )
            
            # Colocar órdenes bracket
            bracket_orders = self.ib.bracketOrder(parent, stop_loss, take_profit)
            
            # Colocar orden principal
            parent_trade = self.ib.placeOrder(contract, bracket_orders[0])
            
            # Esperar confirmación
            for _ in range(5):  # Intentar hasta 5 veces
                self.ib.sleep(1)
                if parent_trade.orderStatus.status:
                    break
            
            # Colocar órdenes secundarias (stop loss y take profit)
            sl_trade = self.ib.placeOrder(contract, bracket_orders[1])
            tp_trade = self.ib.placeOrder(contract, bracket_orders[2])
            
            # Guardar órdenes en base de datos
            parent_id = bracket_orders[0].orderId
            sl_id = bracket_orders[1].orderId
            tp_id = bracket_orders[2].orderId
            
            # Modificar parámetros para cada orden
            parent_params = order_params.copy()
            parent_params["order_type"] = entry_order_type
            if entry_order_type == "LIMIT":
                parent_params["limit_price"] = order_params["entry_price"]
            
            sl_params = order_params.copy()
            sl_params["action"] = reverse_action
            sl_params["order_type"] = "STOP"
            sl_params["stop_price"] = order_params["stop_loss_price"]
            sl_params["parent_id"] = parent_id
            
            tp_params = order_params.copy()
            tp_params["action"] = reverse_action
            tp_params["order_type"] = "LIMIT"
            tp_params["limit_price"] = order_params["take_profit_price"]
            tp_params["parent_id"] = parent_id
            
            # Guardar en base de datos
            self._save_order_to_db(parent_id, parent_params, parent_trade)
            self._save_order_to_db(sl_id, sl_params, sl_trade)
            self._save_order_to_db(tp_id, tp_params, tp_trade)
            
            # Devolver resultado
            return {
                "success": True,
                "parent_order_id": parent_id,
                "stop_loss_order_id": sl_id,
                "take_profit_order_id": tp_id,
                "status": parent_trade.orderStatus.status
            }
        except Exception as e:
            error_msg = f"Error al colocar orden bracket: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "connected": self.connected}
    
    def cancel_order(self, order_id: int) -> Dict:
        """
        Cancela una orden
        
        Args:
            order_id: ID de la orden a cancelar
            
        Returns:
            Dict con resultado de la operación
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"success": False, "error": "No conectado a IBKR", "connected": False}
        
        try:
            # Buscar orden
            open_orders = self.ib.openOrders()
            order = next((o for o in open_orders if o.orderId == order_id), None)
            
            if not order:
                return {"success": False, "error": f"Orden {order_id} no encontrada"}
            
            # Cancelar orden
            self.ib.cancelOrder(order)
            
            # Esperar confirmación
            for _ in range(5):  # Intentar hasta 5 veces
                self.ib.sleep(1)
                trade = next((t for t in self.ib.trades() if t.order.orderId == order_id), None)
                if trade and trade.orderStatus.status in ['Cancelled', 'Canceled']:
                    break
            
            # Actualizar en base de datos
            self._update_order_status_in_db(order_id, 'canceled')
            
            return {"success": True, "order_id": order_id, "status": "canceled"}
        except Exception as e:
            error_msg = f"Error al cancelar orden {order_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "connected": self.connected}
    
    def get_order_status(self, order_id: int) -> Dict:
        """
        Obtiene el estado de una orden
        
        Args:
            order_id: ID de la orden
            
        Returns:
            Dict con estado de la orden
        """
        if not self.connected or not self.ib.isConnected():
            success = self.connect_to_ibkr()
            if not success:
                return {"success": False, "error": "No conectado a IBKR", "connected": False}
        
        try:
            # Buscar orden en trades activos
            trade = next((t for t in self.ib.trades() if t.order.orderId == order_id), None)
            
            if trade:
                return {
                    "success": True,
                    "order_id": order_id,
                    "status": trade.orderStatus.status,
                    "filled": trade.orderStatus.filled,
                    "remaining": trade.orderStatus.remaining,
                    "avg_fill_price": trade.orderStatus.avgFillPrice,
                    "last_fill_price": trade.orderStatus.lastFillPrice,
                    "why_held": trade.orderStatus.whyHeld
                }
            else:
                # Buscar en base de datos
                status = self._get_order_status_from_db(order_id)
                if status:
                    return {"success": True, "order_id": order_id, "status": status}
                else:
                    return {"success": False, "error": f"Orden {order_id} no encontrada"}
        except Exception as e:
            error_msg = f"Error al obtener estado de orden {order_id}: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg, "connected": self.connected}
    
    def _save_order_to_db(self, order_id: int, order_params: Dict, trade) -> None:
        """
        Guarda una orden en la base de datos
        
        Args:
            order_id: ID de la orden
            order_params: Parámetros de la orden
            trade: Objeto trade de IB
        """
        try:
            cursor = self.db_conn.cursor()
            
            # Obtener estado actual
            status = trade.orderStatus.status if hasattr(trade, 'orderStatus') and hasattr(trade.orderStatus, 'status') else 'pending'
            
            # Datos para guardar
            symbol = order_params["symbol"]
            direction = order_params["action"]
            order_type = order_params["order_type"]
            quantity = order_params["quantity"]
            
            # Precio según tipo de orden
            price = None
            if order_type == "LIMIT" and "limit_price" in order_params:
                price = order_params["limit_price"]
            elif order_type == "STOP" and "stop_price" in order_params:
                price = order_params["stop_price"]
            
            # Verificar si ya existe
            check_query = """
            SELECT id FROM trading_orders
            WHERE order_id = %s;
            """
            cursor.execute(check_query, (str(order_id),))
            existing = cursor.fetchone()
            
            if existing:
                # Actualizar orden existente
                update_query = """
                UPDATE trading_orders
                SET status = %s, fill_price = %s, fill_time = %s
                WHERE order_id = %s;
                """
                cursor.execute(update_query, (
                    status,
                    trade.orderStatus.avgFillPrice if hasattr(trade, 'orderStatus') and hasattr(trade.orderStatus, 'avgFillPrice') else None,
                    dt.datetime.now() if status == 'Filled' else None,
                    str(order_id)
                ))
            else:
                # Insertar nueva orden
                insert_query = """
                INSERT INTO trading_orders
                (order_id, symbol, direction, order_type, quantity, price, status, timestamp, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """
                cursor.execute(insert_query, (
                    str(order_id), symbol, direction, order_type, quantity, price, status, 
                    dt.datetime.now(), order_params.get("tag", None)
                ))
            
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al guardar orden en base de datos: {str(e)}")
    
    def _update_order_status_in_db(self, order_id: int, status: str) -> bool:
        """
        Actualiza el estado de una orden en la base de datos
        
        Args:
            order_id: ID de la orden
            status: Nuevo estado
            
        Returns:
            True si se actualizó correctamente, False en caso contrario
        """
        try:
            cursor = self.db_conn.cursor()
            
            update_query = """
            UPDATE trading_orders
            SET status = %s, fill_time = %s
            WHERE order_id = %s;
            """
            cursor.execute(update_query, (
                status,
                dt.datetime.now() if status == 'filled' else None,
                str(order_id)
            ))
            
            self.db_conn.commit()
            cursor.close()
            return True
        except Exception as e:
            self.logger.error(f"Error al actualizar estado de orden en base de datos: {str(e)}")
            return False
    
    def _get_order_status_from_db(self, order_id: int) -> Optional[str]:
        """
        Obtiene el estado de una orden desde la base de datos
        
        Args:
            order_id: ID de la orden
            
        Returns:
            Estado de la orden o None si no se encontró
        """
        try:
            cursor = self.db_conn.cursor()
            
            query = """
            SELECT status FROM trading_orders
            WHERE order_id = %s;
            """
            cursor.execute(query, (str(order_id),))
            result = cursor.fetchone()
            cursor.close()
            
            if result:
                return result[0]
            else:
                return None
        except Exception as e:
            self.logger.error(f"Error al obtener estado de orden desde base de datos: {str(e)}")
            return None
    
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
            return jsonify({
                "connected": self.connected and self.ib.isConnected(),
                "last_error": self.last_error,
                "connection_time": self.connection_time.isoformat() if self.connection_time else None
            })
        
        @app.route('/api/connect', methods=['POST'])
        def connect():
            success = self.connect_to_ibkr()
            return jsonify({
                "success": success,
                "connected": self.connected,
                "last_error": self.last_error
            })
        
        @app.route('/api/disconnect', methods=['POST'])
        def disconnect():
            success = self.disconnect_from_ibkr()
            return jsonify({
                "success": success,
                "connected": self.connected,
                "last_error": self.last_error
            })
        
        @app.route('/api/account', methods=['GET'])
        def get_account():
            return jsonify(self.get_account_summary())
        
        @app.route('/api/market-data/<symbol>', methods=['GET'])
        def get_market_data(symbol):
            return jsonify(self.get_market_data(symbol))
        
        @app.route('/api/order', methods=['POST'])
        def place_order():
            order_params = request.json
            return jsonify(self.place_order(order_params))
        
        @app.route('/api/bracket-order', methods=['POST'])
        def place_bracket_order():
            order_params = request.json
            return jsonify(self.place_bracket_order(order_params))
        
        @app.route('/api/order/<int:order_id>', methods=['GET'])
        def get_order_status(order_id):
            return jsonify(self.get_order_status(order_id))
        
        @app.route('/api/order/<int:order_id>/cancel', methods=['POST'])
        def cancel_order(order_id):
            return jsonify(self.cancel_order(order_id))
        
        return app
    
    def run_api(self) -> None:
        """
        Ejecuta la API REST
        """
        host = self.config['api']['host']
        port = self.config['api']['port']
        
        self.logger.info(f"Iniciando API en {host}:{port}")
        self.app.run(host=host, port=port)

if __name__ == "__main__":
    # Crear conector
    connector = IBKRConnector()
    
    # Intentar conectar
    connector.connect_to_ibkr()
    
    # Ejecutar API
    connector.run_api()
