import os
import json
import time
import logging
import smtplib
import requests
import psycopg2
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from psycopg2.extras import RealDictCursor
from typing import Dict, List, Union, Optional

class AlertService:
    def __init__(self):
        """Inicializa el servicio de alertas"""
        # Configurar logging
        logging.basicConfig(
            level=getattr(logging, os.environ.get('LOG_LEVEL', 'INFO')),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('alert_service')
        
        # Cargar configuración
        self.config = self._load_config()
        
        # Conectar a la base de datos
        self.db_conn = self._get_db_connection()
        
        # Inicializar canales de notificación
        self.telegram_enabled = bool(os.environ.get('TELEGRAM_TOKEN', ''))
        self.email_enabled = bool(os.environ.get('EMAIL_USERNAME', ''))
        
        self.logger.info("Servicio de alertas inicializado")
    
    def _load_config(self) -> Dict:
        """Carga la configuración desde archivo o variables de entorno"""
        config = {
            'check_interval': int(os.environ.get('CHECK_INTERVAL', '60')),  # En segundos
            'alert_thresholds': {
                'price_change': float(os.environ.get('PRICE_CHANGE_THRESHOLD', '1.5')),  # Porcentaje
                'volume_spike': float(os.environ.get('VOLUME_SPIKE_THRESHOLD', '2.0')),  # Multiplicador
                'pattern_confidence': float(os.environ.get('PATTERN_CONFIDENCE_THRESHOLD', '0.75')),  # 0-1
                'risk_warning': float(os.environ.get('RISK_WARNING_THRESHOLD', '0.1'))  # Porcentaje (10%)
            },
            'notification': {
                'telegram': {
                    'token': os.environ.get('TELEGRAM_TOKEN', ''),
                    'chat_id': os.environ.get('TELEGRAM_CHAT_ID', '')
                },
                'email': {
                    'smtp_server': os.environ.get('EMAIL_SMTP_SERVER', 'smtp.gmail.com'),
                    'smtp_port': int(os.environ.get('EMAIL_SMTP_PORT', '587')),
                    'username': os.environ.get('EMAIL_USERNAME', ''),
                    'password': os.environ.get('EMAIL_PASSWORD', ''),
                    'recipients': os.environ.get('EMAIL_RECIPIENTS', '').split(','),
                    'subject_prefix': '[Trading Alert] '
                }
            }
        }
        
        # Intentar cargar desde archivo si existe
        config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'alerts.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                    # Actualizar configuración con valores del archivo
                    self._deep_update(config, file_config)
                    self.logger.info(f"Configuración cargada desde {config_path}")
            except Exception as e:
                self.logger.error(f"Error al cargar configuración desde archivo: {str(e)}")
        
        return config
    
    def _deep_update(self, original: Dict, update: Dict) -> None:
        """Actualiza recursivamente un diccionario con valores de otro"""
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
    
    def run(self):
        """Ejecuta el ciclo principal del servicio de alertas"""
        self.logger.info("Iniciando ciclo de monitoreo de alertas")
        
        while True:
            try:
                # Verificar diferentes tipos de alertas
                self._check_price_alerts()
                self._check_pattern_alerts()
                self._check_risk_alerts()
                self._check_system_alerts()
                
                # Esperar hasta el próximo ciclo
                time.sleep(self.config['check_interval'])
            except Exception as e:
                self.logger.error(f"Error en ciclo de monitoreo: {str(e)}")
                # Evitar bucles de error demasiado rápidos
                time.sleep(10)
    
    def _check_price_alerts(self):
        """Verifica alertas relacionadas con cambios de precio"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Consultar precios actuales y compararlos con promedio reciente
            query = """
            WITH recent_prices AS (
                SELECT 
                    symbol,
                    LAST_VALUE(close) OVER (PARTITION BY symbol ORDER BY date) AS latest_price,
                    AVG(close) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_recent_price,
                    LAST_VALUE(volume) OVER (PARTITION BY symbol ORDER BY date) AS latest_volume,
                    AVG(volume) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING) AS avg_recent_volume,
                    date
                FROM market_data
                WHERE date >= %s
            )
            SELECT DISTINCT ON (symbol)
                symbol,
                latest_price,
                avg_recent_price,
                latest_volume,
                avg_recent_volume,
                date,
                ((latest_price - avg_recent_price) / avg_recent_price * 100) AS price_change_pct,
                (latest_volume / NULLIF(avg_recent_volume, 0)) AS volume_ratio
            FROM recent_prices
            ORDER BY symbol, date DESC;
            """
            
            # Fecha desde la cual buscar datos (últimas 48 horas)
            start_date = datetime.now() - timedelta(hours=48)
            cursor.execute(query, (start_date,))
            results = cursor.fetchall()
            
            for row in results:
                # Verificar cambios significativos de precio
                if (abs(row['price_change_pct']) >= self.config['alert_thresholds']['price_change'] and 
                    row['price_change_pct'] is not None):
                    direction = "subida" if row['price_change_pct'] > 0 else "bajada"
                    message = (f"🔔 ALERTA DE PRECIO: {row['symbol']} ha tenido una {direction} de "
                              f"{abs(row['price_change_pct']):.2f}% desde {row['avg_recent_price']:.2f} "
                              f"hasta {row['latest_price']:.2f}")
                    self._send_alert(message, alert_type="price_change")
                
                # Verificar picos de volumen
                if (row['volume_ratio'] is not None and 
                    row['volume_ratio'] >= self.config['alert_thresholds']['volume_spike']):
                    message = (f"📊 ALERTA DE VOLUMEN: {row['symbol']} ha tenido un pico de volumen "
                              f"{row['volume_ratio']:.2f}x mayor que el promedio reciente. "
                              f"Volumen actual: {row['latest_volume']:.0f}")
                    self._send_alert(message, alert_type="volume_spike")
            
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al verificar alertas de precio: {str(e)}")
    
    def _check_pattern_alerts(self):
        """Verifica alertas relacionadas con patrones técnicos detectados"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Consultar patrones detectados recientemente
            query = """
            SELECT 
                id, symbol, pattern_type, confidence, entry_price, 
                stop_loss_price, target_price, timestamp
            FROM detected_patterns
            WHERE timestamp >= %s
            AND notified = FALSE
            AND confidence >= %s
            ORDER BY confidence DESC;
            """
            
            # Patrones detectados en la última hora
            start_time = datetime.now() - timedelta(hours=1)
            confidence_threshold = self.config['alert_thresholds']['pattern_confidence']
            
            cursor.execute(query, (start_time, confidence_threshold))
            patterns = cursor.fetchall()
            
            for pattern in patterns:
                # Crear mensaje de alerta
                direction = "alcista" if pattern['pattern_type'].endswith('_bullish') else "bajista"
                pattern_name = pattern['pattern_type'].replace('_bullish', '').replace('_bearish', '')
                
                message = (f"🔍 PATRÓN DETECTADO: {pattern_name.upper()} ({direction}) "
                          f"en {pattern['symbol']} con confianza {pattern['confidence']:.2f}\n"
                          f"Precio de entrada sugerido: {pattern['entry_price']:.2f}\n"
                          f"Stop loss: {pattern['stop_loss_price']:.2f}\n"
                          f"Objetivo: {pattern['target_price']:.2f}\n"
                          f"R/R: {(pattern['target_price'] - pattern['entry_price']) / (pattern['entry_price'] - pattern['stop_loss_price']):.2f}")
                
                # Enviar alerta
                self._send_alert(message, alert_type="pattern")
                
                # Marcar como notificado
                update_query = "UPDATE detected_patterns SET notified = TRUE WHERE id = %s;"
                cursor.execute(update_query, (pattern['id'],))
                self.db_conn.commit()
            
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al verificar alertas de patrones: {str(e)}")
    
    def _check_risk_alerts(self):
        """Verifica alertas relacionadas con el riesgo de la cartera"""
        try:
            cursor = self.db_conn.cursor(cursor_factory=RealDictCursor)
            
            # Consultar estado actual de riesgo
            query = """
            SELECT 
                id, timestamp, risk_level, total_exposure, 
                exposure_percentage, max_drawdown, num_positions
            FROM risk_status
            ORDER BY timestamp DESC
            LIMIT 1;
            """
            
            cursor.execute(query)
            risk_status = cursor.fetchone()
            
            if risk_status and risk_status['risk_level'] == 'HIGH':
                message = (f"⚠️ ALERTA DE RIESGO: Nivel de riesgo ALTO detectado\n"
                          f"Exposición total: ${risk_status['total_exposure']:.2f} "
                          f"({risk_status['exposure_percentage']:.2f}%)\n"
                          f"Drawdown máximo: {risk_status['max_drawdown']:.2f}%\n"
                          f"Posiciones abiertas: {risk_status['num_positions']}")
                
                self._send_alert(message, alert_type="risk")
            
            # Verificar drawdown significativo
            query_drawdown = """
            SELECT 
                account_balance, previous_balance,
                ((previous_balance - account_balance) / previous_balance * 100) AS drawdown_pct
            FROM (
                SELECT 
                    account_balance,
                    LAG(account_balance, 1) OVER (ORDER BY timestamp) AS previous_balance
                FROM account_status
                ORDER BY timestamp DESC
                LIMIT 2
            ) AS latest_status
            WHERE previous_balance IS NOT NULL;
            """
            
            cursor.execute(query_drawdown)
            drawdown = cursor.fetchone()
            
            if (drawdown and 
                drawdown['drawdown_pct'] is not None and 
                drawdown['drawdown_pct'] >= self.config['alert_thresholds']['risk_warning']):
                message = (f"📉 ALERTA DE DRAWDOWN: Se ha detectado un drawdown de {drawdown['drawdown_pct']:.2f}%\n"
                          f"Balance anterior: ${drawdown['previous_balance']:.2f}\n"
                          f"Balance actual: ${drawdown['account_balance']:.2f}\n"
                          f"Pérdida: ${(drawdown['previous_balance'] - drawdown['account_balance']):.2f}")
                
                self._send_alert(message, alert_type="drawdown")
            
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al verificar alertas de riesgo: {str(e)}")
    
    def _check_system_alerts(self):
        """Verifica alertas relacionadas con el estado del sistema"""
        try:
            # Verificar conexión con IBKR
            ibkr_url = f"http://{os.environ.get('IBKR_SERVICE_HOST', 'ibkr-connector')}:{os.environ.get('IBKR_SERVICE_PORT', '8080')}/api/status"
            
            try:
                response = requests.get(ibkr_url, timeout=5)
                if response.status_code != 200 or not response.json().get('connected', False):
                    message = "❌ ALERTA DE SISTEMA: Conexión con Interactive Brokers perdida. Verificar estado del conector."
                    self._send_alert(message, alert_type="system")
            except requests.RequestException:
                message = "❌ ALERTA DE SISTEMA: No se pudo contactar el servicio de Interactive Brokers."
                self._send_alert(message, alert_type="system")
            
            # Verificar otros servicios críticos aquí...
            
        except Exception as e:
            self.logger.error(f"Error al verificar alertas de sistema: {str(e)}")
    
    def _send_alert(self, message: str, alert_type: str = "general") -> bool:
        """
        Envía una alerta a través de los canales configurados
        
        Args:
            message: Texto de la alerta
            alert_type: Tipo de alerta (price_change, pattern, risk, system, etc.)
            
        Returns:
            bool: True si se envió correctamente al menos por un canal
        """
        self.logger.info(f"Alerta tipo [{alert_type}]: {message}")
        
        # Registrar alerta en base de datos
        try:
            cursor = self.db_conn.cursor()
            query = """
            INSERT INTO alerts (timestamp, alert_type, message)
            VALUES (%s, %s, %s);
            """
            cursor.execute(query, (datetime.now(), alert_type, message))
            self.db_conn.commit()
            cursor.close()
        except Exception as e:
            self.logger.error(f"Error al registrar alerta en BD: {str(e)}")
        
        # Enviar por canales configurados
        success = False
        
        # 1. Telegram
        if self.telegram_enabled:
            telegram_success = self._send_telegram(message)
            success = success or telegram_success
        
        # 2. Email
        if self.email_enabled:
            email_success = self._send_email(message, alert_type)
            success = success or email_success
        
        return success
    
    def _send_telegram(self, message: str) -> bool:
        """
        Envía un mensaje a través de Telegram
        
        Args:
            message: Texto del mensaje
            
        Returns:
            bool: True si se envió correctamente
        """
        if not self.config['notification']['telegram']['token'] or not self.config['notification']['telegram']['chat_id']:
            return False
        
        try:
            token = self.config['notification']['telegram']['token']
            chat_id = self.config['notification']['telegram']['chat_id']
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            
            response = requests.post(
                url, 
                data={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML'
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Alerta enviada por Telegram")
                return True
            else:
                self.logger.error(f"Error al enviar por Telegram: {response.text}")
                return False
        except Exception as e:
            self.logger.error(f"Error al enviar por Telegram: {str(e)}")
            return False
    
    def _send_email(self, message: str, alert_type: str) -> bool:
        """
        Envía un mensaje por correo electrónico
        
        Args:
            message: Texto del mensaje
            alert_type: Tipo de alerta
            
        Returns:
            bool: True si se envió correctamente
        """
        config = self.config['notification']['email']
        if not config['username'] or not config['password'] or not config['recipients']:
            return False
        
        try:
            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = config['username']
            msg['To'] = ', '.join(config['recipients'])
            msg['Subject'] = f"{config['subject_prefix']}{alert_type.upper()}"
            
            # Añadir contenido
            msg.attach(MIMEText(message, 'plain'))
            
            # Conectar al servidor SMTP
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            
            # Enviar mensaje
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Alerta enviada por correo a {len(config['recipients'])} destinatarios")
            return True
        except Exception as e:
            self.logger.error(f"Error al enviar por correo: {str(e)}")
            return False

if __name__ == "__main__":
    # Esperar un poco para que otros servicios estén disponibles
    time.sleep(10)
    
    # Iniciar servicio de alertas
    alert_service = AlertService()
    alert_service.run()
