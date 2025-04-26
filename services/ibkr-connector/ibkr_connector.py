from ib_insync import *
import json
import logging
from typing import Dict, List

class IBKRConnector:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.ib = IB()
        self.logger = logging.getLogger('ibkr_connector')
        self.connected = False
    
    def connect(self):
        """Conecta con Interactive Brokers TWS o IB Gateway"""
        try:
            self.ib.connect(
                self.config['host'],
                self.config['port'],
                clientId=self.config['client_id']
            )
            self.connected = True
            self.logger.info(f"Conectado a IBKR: {self.ib.client.isConnected()}")
            return True
        except Exception as e:
            self.logger.error(f"Error al conectar con IBKR: {str(e)}")
            return False
    
    def place_order(self, order_params: Dict) -> Dict:
        """Coloca una orden en IBKR"""
        if not self.connected:
            if not self.connect():
                return {'success': False, 'error': 'No se pudo conectar a IBKR'}
        
        try:
            # Crear contrato
            contract = Stock(
                symbol=order_params['symbol'],
                exchange=order_params.get('exchange', 'MEXI'),  # Bolsa Mexicana de Valores
                currency=order_params.get('currency', 'MXN')
            )
            
            # Crear orden
            if order_params['order_type'] == 'MARKET':
                order = MarketOrder(
                    order_params['action'],  # 'BUY' o 'SELL'
                    order_params['quantity']
                )
            elif order_params['order_type'] == 'LIMIT':
                order = LimitOrder(
                    order_params['action'],
                    order_params['quantity'],
                    order_params['limit_price']
                )
            else:
                return {'success': False, 'error': 'Tipo de orden no soportado'}
            
            # Añadir stop loss si está configurado
            if 'stop_loss' in order_params:
                # Crear orden de stop loss
                stop_action = 'SELL' if order_params['action'] == 'BUY' else 'BUY'
                stop_order = StopOrder(
                    stop_action,
                    order_params['quantity'],
                    order_params['stop_loss']
                )
                parent_order = order
                
                # Crear orden bracket
                bracket = self.ib.bracketOrder(
                    parent_order,
                    stop_order,
                    None  # Sin take profit por ahora
                )
                
                # Colocar órdenes bracket
                trade = self.ib.placeOrder(contract, bracket[0])
                return {
                    'success': True,
                    'order_id': trade.order.orderId,
                    'status': trade.orderStatus.status
                }
            else:
                # Colocar orden simple
                trade = self.ib.placeOrder(contract, order)
                return {
                    'success': True,
                    'order_id': trade.order.orderId,
                    'status': trade.orderStatus.status
                }
                
        except Exception as e:
            self.logger.error(f"Error al colocar orden: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_account_summary(self) -> Dict:
        """Obtiene el resumen de la cuenta"""
        if not self.connected:
            if not self.connect():
                return {'success': False, 'error': 'No se pudo conectar a IBKR'}
        
        try:
            account_values = self.ib.accountSummary()
            
            # Convertir a formato más amigable
            summary = {}
            for av in account_values:
                if av.tag not in summary:
                    summary[av.tag] = {}
                summary[av.tag][av.currency] = float(av.value)
            
            return {'success': True, 'account': summary}
        except Exception as e:
            self.logger.error(f"Error al obtener resumen de cuenta: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def disconnect(self):
        """Desconecta de IBKR"""
        if self.connected:
            self.ib.disconnect()
            self.connected = False
            self.logger.info("Desconectado de IBKR")
