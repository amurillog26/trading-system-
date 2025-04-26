import os
import json
import logging
import argparse
import dask.dataframe as dd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

class Backtester:
    def __init__(self, config_path: str = None):
        """
        Inicializa el sistema de backtesting
        
        Args:
            config_path: Ruta al archivo de configuración. Si es None, usa valores por defecto
                         o busca en la ubicación estándar.
        """
        # Configurar logger
        self._setup_logging()
        
        # Cargar configuración
        self.config = self._load_config(config_path)
        
        # Establecer directorios
        self._setup_directories()
        
        # Datos cargados
        self.data = {}
        
        # Resultados
        self.results = {}
        
        self.logger.info(f"Backtester inicializado: {self.config['system_name']}")
    
    def _setup_logging(self) -> None:
        """Configura el sistema de logging"""
        log_level = os.environ.get('LOG_LEVEL', 'INFO')
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=getattr(logging, log_level),
            format=log_format
        )
        self.logger = logging.getLogger('backtester')
    
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
            "system_name": "Trading System Backtester",
            "version": "1.0.0",
            "data_dir": "/app/data",
            "results_dir": "/app/results",
            "start_date": "2020-01-01",
            "end_date": "2023-12-31",
            "initial_capital": 100000,
            "commission_rate": 0.0025,  # 0.25% por operación
            "slippage": 0.001,  # 0.1% de slippage
            "symbols": ["AMXL.MX", "FEMSAUBD.MX", "GFNORTEO.MX", "WALMEX.MX"],
            "timeframes": ["1d"],
            "strategies": {
                "ma_crossover": {
                    "fast_period": 9,
                    "slow_period": 21,
                    "stop_loss_atr": 2.0,
                    "take_profit_atr": 4.0,
                    "atr_period": 14
                },
                "rsi_strategy": {
                    "rsi_period": 14,
                    "rsi_oversold": 30,
                    "rsi_overbought": 70,
                    "stop_loss_pct": 0.05
                }
            },
            "optimization": {
                "parallel_jobs": 8,
                "parameter_grids": {
                    "ma_crossover": {
                        "fast_period": [5, 8, 9, 10, 12, 15],
                        "slow_period": [20, 21, 25, 30, 35, 40],
                        "stop_loss_atr": [1.5, 2.0, 2.5, 3.0]
                    },
                    "rsi_strategy": {
                        "rsi_period": [9, 14, 21],
                        "rsi_oversold": [20, 25, 30, 35],
                        "rsi_overbought": [65, 70, 75, 80]
                    }
                }
            }
        }
        
        # Si no se proporciona ruta, buscar en ubicación estándar
        if not config_path:
            config_path = os.path.join(os.environ.get('CONFIG_DIR', '/app/config'), 'backtester.json')
        
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
        
        # Convertir fechas a objetos datetime
        config['start_date'] = dt.datetime.strptime(config['start_date'], '%Y-%m-%d')
        config['end_date'] = dt.datetime.strptime(config['end_date'], '%Y-%m-%d')
        
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
    
    def _setup_directories(self) -> None:
        """Crea los directorios necesarios si no existen"""
        os.makedirs(self.config['data_dir'], exist_ok=True)
        os.makedirs(self.config['results_dir'], exist_ok=True)
        
        # Crear directorio para resultados del sistema actual
        self.system_results_dir = os.path.join(
            self.config['results_dir'], 
            self.config['system_name'].replace(' ', '_').lower()
        )
        os.makedirs(self.system_results_dir, exist_ok=True)
        
        # Crear directorios para cada estrategia
        for strategy in self.config['strategies'].keys():
            strategy_dir = os.path.join(self.system_results_dir, strategy)
            os.makedirs(strategy_dir, exist_ok=True)
    
    def load_data(self, symbols: List[str] = None, timeframes: List[str] = None) -> None:
        """
        Carga los datos históricos para los símbolos y timeframes especificados
        
        Args:
            symbols: Lista de símbolos a cargar. Si es None, usa todos los de la configuración.
            timeframes: Lista de timeframes a cargar. Si es None, usa todos los de la configuración.
        """
        if symbols is None:
            symbols = self.config['symbols']
        
        if timeframes is None:
            timeframes = self.config['timeframes']
        
        self.logger.info(f"Cargando datos para {len(symbols)} símbolos y {len(timeframes)} timeframes")
        
        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                try:
                    # Construir ruta al archivo
                    file_path = os.path.join(self.config['data_dir'], f"{key}.csv")
                    
                    if not os.path.exists(file_path):
                        self.logger.warning(f"Archivo no encontrado: {file_path}")
                        continue
                    
                    # Cargar datos con Dask para procesamiento paralelo
                    df = dd.read_csv(file_path, parse_dates=['date'])
                    
                    # Filtrar por fechas
                    df = df[(df['date'] >= self.config['start_date']) & 
                            (df['date'] <= self.config['end_date'])]
                    
                    # Ordenar por fecha
                    df = df.sort_values('date')
                    
                    # Almacenar en diccionario de datos
                    self.data[key] = df
                    
                    self.logger.info(f"Datos cargados para {key}: {len(df)} registros (aprox)")
                except Exception as e:
                    self.logger.error(f"Error al cargar datos para {key}: {str(e)}")
    
    def run_backtest(self, strategy_function: Callable, strategy_params: Dict) -> Dict:
        """
        Ejecuta el backtest para una estrategia específica
        
        Args:
            strategy_function: Función de estrategia que toma datos y parámetros
            strategy_params: Parámetros para la estrategia
            
        Returns:
            Dict con resultados del backtest
        """
        symbol = strategy_params.get('symbol')
        timeframe = strategy_params.get('timeframe', '1d')
        key = f"{symbol}_{timeframe}"
        
        if key not in self.data:
            self.logger.error(f"No hay datos disponibles para {key}")
            return {'error': f"No hay datos disponibles para {key}"}
        
        # Obtener datos
        data = self.data[key].compute()  # Convertir Dask DataFrame a Pandas DataFrame
        
        # Registrar información de la ejecución
        self.logger.info(f"Ejecutando backtest para {symbol} con estrategia {strategy_params.get('strategy_name', 'unnamed')}")
        self.logger.info(f"Rango de fechas: {data['date'].min()} a {data['date'].max()}")
        self.logger.info(f"Parámetros de estrategia: {strategy_params}")
        
        # Inicializar variables
        initial_capital = strategy_params.get('initial_capital', self.config['initial_capital'])
        commission_rate = strategy_params.get('commission_rate', self.config['commission_rate'])
        slippage = strategy_params.get('slippage', self.config['slippage'])
        
        capital = initial_capital
        position = 0
        entry_price = 0
        trades = []
        equity_curve = []
        
        # Ejecutar estrategia para generar señales
        signals = strategy_function(data, strategy_params)
        
        # Procesar señales
        for i, row in enumerate(signals):
            date = row['date']
            signal = row['signal']
            price = row['price']
            stop_loss = row.get('stop_loss', 0)
            take_profit = row.get('take_profit', 0)
            
            # Añadir punto a la curva de capital
            equity_curve.append({'date': date, 'equity': capital + position * price})
            
            # Procesar señal
            if signal == 'buy' and position == 0:
                # Aplicar slippage al precio de entrada
                actual_price = price * (1 + slippage)
                
                # Calcular tamaño de posición
                max_shares = int(capital * 0.95 / actual_price)  # Usar 95% del capital
                position = max_shares
                entry_price = actual_price
                cost = position * actual_price
                commission = cost * commission_rate
                capital -= (cost + commission)
                
                trades.append({
                    'type': 'entry',
                    'direction': 'long',
                    'date': date,
                    'price': actual_price,
                    'shares': position,
                    'cost': cost,
                    'commission': commission,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit
                })
                
                self.logger.debug(f"Compra: {date}, Precio: {actual_price}, Acciones: {position}, Capital: {capital}")
            
            elif signal == 'sell' and position > 0:
                # Aplicar slippage al precio de salida
                actual_price = price * (1 - slippage)
                
                # Vender posición
                cost = position * actual_price
                commission = cost * commission_rate
                capital += (cost - commission)
                
                # Calcular ganancia/pérdida
                entry_trade = next((t for t in trades if t['type'] == 'entry' and t['direction'] == 'long'), None)
                if entry_trade:
                    entry_cost = entry_trade['cost']
                    profit = cost - entry_cost - commission - entry_trade['commission']
                    profit_pct = profit / entry_cost * 100
                else:
                    profit = 0
                    profit_pct = 0
                
                trades.append({
                    'type': 'exit',
                    'direction': 'long',
                    'date': date,
                    'price': actual_price,
                    'shares': position,
                    'cost': cost,
                    'commission': commission,
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'reason': signal
                })
                
                self.logger.debug(f"Venta: {date}, Precio: {actual_price}, Acciones: {position}, "
                                 f"Ganancia: {profit_pct:.2f}%, Capital: {capital}")
                
                position = 0
                entry_price = 0
            
            elif signal == 'short' and position == 0:
                # Implementar operativa en corto si está soportada
                pass
            
            elif signal == 'cover' and position < 0:
                # Implementar cobertura de cortos si está soportada
                pass
        
        # Si quedó posición abierta, cerrarla al último precio
        if position > 0:
            last_price = data['close'].iloc[-1] * (1 - slippage)
            cost = position * last_price
            commission = cost * commission_rate
            capital += (cost - commission)
            
            # Calcular ganancia/pérdida
            entry_trade = next((t for t in trades if t['type'] == 'entry' and t['direction'] == 'long'), None)
            if entry_trade:
                entry_cost = entry_trade['cost']
                profit = cost - entry_cost - commission - entry_trade['commission']
                profit_pct = profit / entry_cost * 100
            else:
                profit = 0
                profit_pct = 0
            
            trades.append({
                'type': 'exit',
                'direction': 'long',
                'date': data['date'].iloc[-1],
                'price': last_price,
                'shares': position,
                'cost': cost,
                'commission': commission,
                'profit': profit,
                'profit_pct': profit_pct,
                'reason': 'end_of_period'
            })
            
            position = 0
        
        # Calcular métricas
        metrics = self._calculate_metrics(trades, equity_curve, initial_capital)
        
        # Agregar información adicional
        results = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy_params.get('strategy_name', 'unnamed'),
            'start_date': self.config['start_date'].strftime('%Y-%m-%d'),
            'end_date': self.config['end_date'].strftime('%Y-%m-%d'),
            'initial_capital': initial_capital,
            'final_capital': capital + (position * data['close'].iloc[-1] if position > 0 else 0),
            'trades': trades,
            'equity_curve': equity_curve,
            'parameters': strategy_params,
            **metrics
        }
        
        # Guardar en el diccionario de resultados
        strategy_id = f"{symbol}_{timeframe}_{strategy_params.get('strategy_name', 'unnamed')}_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.results[strategy_id] = results
        
        # Guardar resultados en archivo
        self._save_results(strategy_id, results)
        
        return results
    
    def _calculate_metrics(self, trades: List[Dict], equity_curve: List[Dict], initial_capital: float) -> Dict:
        """
        Calcula métricas de rendimiento a partir de los resultados del backtest
        
        Args:
            trades: Lista de operaciones realizadas
            equity_curve: Curva de capital
            initial_capital: Capital inicial
            
        Returns:
            Dict con métricas calculadas
        """
        exit_trades = [t for t in trades if t['type'] == 'exit']
        total_trades = len(exit_trades)
        
        if total_trades == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'max_profit': 0,
                'max_loss': 0,
                'avg_trade': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'avg_days_in_trade': 0
            }
        
        # Calcular métricas básicas
        winning_trades = len([t for t in exit_trades if t['profit'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        profit_trades = [t['profit'] for t in exit_trades if t['profit'] > 0]
        loss_trades = [t['profit'] for t in exit_trades if t['profit'] <= 0]
        
        avg_profit = np.mean(profit_trades) if profit_trades else 0
        avg_loss = np.mean(loss_trades) if loss_trades else 0
        
        if sum(loss_trades) != 0:
            profit_factor = abs(sum(profit_trades) / sum(loss_trades)) if sum(loss_trades) != 0 else float('inf')
        else:
            profit_factor = float('inf')
        
        max_profit = max(profit_trades) if profit_trades else 0
        max_loss = min(loss_trades) if loss_trades else 0
        avg_trade = np.mean([t['profit'] for t in exit_trades]) if exit_trades else 0
        
        # Calcular curva de capital y drawdown
        equity_df = pd.DataFrame(equity_curve)
        
        if not equity_df.empty:
            # Asegurar que las fechas están en orden
            equity_df = equity_df.sort_values('date')
            
            # Calcular drawdown
            equity_df['peak'] = equity_df['equity'].cummax()
            equity_df['drawdown'] = equity_df['peak'] - equity_df['equity']
            equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
            max_drawdown = equity_df['drawdown_pct'].max()
            
            # Calcular rendimiento total
            initial_equity = equity_df['equity'].iloc[0]
            final_equity = equity_df['equity'].iloc[-1]
            total_return = (final_equity - initial_equity) / initial_equity * 100
            
            # Calcular rendimiento anualizado
            days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
            if days > 0:
                annual_return = (1 + total_return/100) ** (365 / days) - 1
                annual_return *= 100  # Convertir a porcentaje
            else:
                annual_return = 0
            
            # Calcular Sharpe Ratio (asumiendo rendimientos diarios)
            equity_df['daily_return'] = equity_df['equity'].pct_change()
            avg_daily_return = equity_df['daily_return'].mean()
            std_daily_return = equity_df['daily_return'].std()
            
            if std_daily_return > 0:
                sharpe_ratio = (avg_daily_return / std_daily_return) * np.sqrt(252)  # Anualizado
            else:
                sharpe_ratio = 0
            
            # Calcular Sortino Ratio (solo considerando rendimientos negativos)
            negative_returns = equity_df['daily_return'][equity_df['daily_return'] < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                sortino_ratio = (avg_daily_return / negative_returns.std()) * np.sqrt(252)
            else:
                sortino_ratio = 0
        else:
            max_drawdown = 0
            total_return = 0
            annual_return = 0
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calcular duración promedio de las operaciones
        if exit_trades and 'date' in exit_trades[0]:
            trade_durations = []
            
            for i, exit_trade in enumerate(exit_trades):
                exit_date = exit_trade['date']
                
                # Encontrar la entrada correspondiente
                entry_trades = [t for t in trades if t['type'] == 'entry' and t['date'] <= exit_date]
                if entry_trades:
                    entry_date = max([t['date'] for t in entry_trades], key=lambda d: d)
                    if isinstance(entry_date, dt.datetime) and isinstance(exit_date, dt.datetime):
                        duration = (exit_date - entry_date).days
                        trade_durations.append(duration)
            
            avg_days_in_trade = np.mean(trade_durations) if trade_durations else 0
        else:
            avg_days_in_trade = 0
        
        # Retornar todas las métricas
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'avg_loss': avg_loss,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'avg_trade': avg_trade,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'avg_days_in_trade': avg_days_in_trade
        }
    
    def run_parameter_optimization(self, strategy_function: Callable, 
                                  base_params: Dict, 
                                  param_grid: Dict = None) -> Dict:
        """
        Optimiza los parámetros de una estrategia mediante backtesting
        
        Args:
            strategy_function: Función de estrategia
            base_params: Parámetros base de la estrategia
            param_grid: Diccionario con parámetros a optimizar y valores a probar.
                       Si es None, usa los valores de la configuración.
            
        Returns:
            Dict con resultados de optimización
        """
        symbol = base_params.get('symbol')
        strategy_name = base_params.get('strategy_name', 'unnamed')
        
        self.logger.info(f"Iniciando optimización de parámetros para {symbol} con estrategia {strategy_name}")
        
        # Si no se proporciona grid, usar el de la configuración
        if param_grid is None:
            try:
                param_grid = self.config['optimization']['parameter_grids'][strategy_name]
                self.logger.info(f"Usando grid de configuración para {strategy_name}")
            except KeyError:
                self.logger.warning(f"No se encontró grid para {strategy_name} en la configuración")
                return {'error': f"No se encontró grid para {strategy_name}"}
        
        # Generar todas las combinaciones de parámetros
        param_combinations = self._generate_param_combinations(param_grid)
        self.logger.info(f"Probando {len(param_combinations)} combinaciones de parámetros")
        
        # Resultados
        optimization_results = []
        
        # Ejecutar backtests en paralelo
        max_workers = self.config['optimization'].get('parallel_jobs', 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for params in param_combinations:
                # Combinar parámetros base con esta combinación
                test_params = base_params.copy()
                test_params.update(params)
                
                # Añadir identificador único
                param_str = '_'.join([f"{k}_{v}" for k, v in params.items()])
                test_params['strategy_name'] = f"{strategy_name}_{param_str}"
                
                # Enviar tarea
                futures.append(executor.submit(self.run_backtest, strategy_function, test_params))
            
            # Recopilar resultados
            for future in as_completed(futures):
                try:
                    result = future.result()
                    optimization_results.append(result)
                except Exception as e:
                    self.logger.error(f"Error en optimización: {str(e)}")
        
        # Ordenar resultados por métrica seleccionada
        optimization_metric = self.config['optimization'].get('metric', 'sharpe_ratio')
        optimization_results.sort(key=lambda x: x.get(optimization_metric, 0), reverse=True)
        
        # Guardar resultados de optimización
        optimization_id = f"{symbol}_{strategy_name}_opt_{dt.datetime.now().strftime('%Y%m%d%H%M%S')}"
        self._save_optimization_results(optimization_id, optimization_results)
        
        self.logger.info(f"Optimización completada. Mejor {optimization_metric}: "
                       f"{optimization_results[0].get(optimization_metric, 0) if optimization_results else 'N/A'}")
        
        return {
            'best_params': optimization_results[0] if optimization_results else None,
            'all_results': optimization_results,
            'optimization_id': optimization_id
        }
    
    def _generate_param_combinations(self, param_grid: Dict) -> List[Dict]:
        """
        Genera todas las combinaciones posibles de parámetros
        
        Args:
            param_grid: Diccionario con parámetros a optimizar y valores a probar
            
        Returns:
            Lista de diccionarios con combinaciones de parámetros
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        if not keys:
            return [{}]
        
        combinations = []
        
        # Función recursiva para generar combinaciones
        def generate(index, current_params):
            if index == len(keys):
                combinations.append(current_params.copy())
                return
            
            for value in values[index]:
                current_params[keys[index]] = value
                generate(index + 1, current_params)
        
        generate(0, {})
        return combinations
    
    def _save_results(self, strategy_id: str, results: Dict) -> None:
        """
        Guarda los resultados del backtest en un archivo
        
        Args:
            strategy_id: Identificador de la estrategia
            results: Resultados del backtest
        """
        strategy_name = results.get('strategy', 'unnamed')
        strategy_dir = os.path.join(self.system_results_dir, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Guardar JSON
        json_path = os.path.join(strategy_dir, f"{strategy_id}.json")
        try:
            with open(json_path, 'w') as f:
                # Eliminar datos muy grandes para el archivo JSON
                results_copy = results.copy()
                results_copy.pop('equity_curve', None)
                results_copy.pop('trades', None)
                json.dump(results_copy, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error al guardar resultados JSON: {str(e)}")
        
        # Guardar datos completos en formato más eficiente (pickle)
        pickle_path = os.path.join(strategy_dir, f"{strategy_id}.pkl")
        try:
            pd.to_pickle(results, pickle_path)
        except Exception as e:
            self.logger.error(f"Error al guardar resultados pickle: {str(e)}")
        
        # Guardar gráficos
        try:
            charts_dir = os.path.join(strategy_dir, f"{strategy_id}_charts")
            os.makedirs(charts_dir, exist_ok=True)
            self._generate_result_charts(strategy_id, results, charts_dir)
        except Exception as e:
            self.logger.error(f"Error al generar gráficos: {str(e)}")
        
        self.logger.info(f"Resultados guardados en {json_path}")
    
    def _save_optimization_results(self, optimization_id: str, results: List[Dict]) -> None:
        """
        Guarda los resultados de la optimización de parámetros
        
        Args:
            optimization_id: Identificador de la optimización
            results: Lista de resultados
        """
        if not results:
            self.logger.warning("No hay resultados para guardar")
            return
        
        # Determinar estrategia
        strategy_name = results[0].get('strategy', 'unnamed').split('_')[0]
        
        # Crear directorio para resultados de optimización
        strategy_dir = os.path.join(self.system_results_dir, strategy_name)
        opt_dir = os.path.join(strategy_dir, optimization_id)
        os.makedirs(opt_dir, exist_ok=True)
        
        # Guardar resumen
        summary = []
        for r in results:
            params = r.get('parameters', {})
            # Excluir parámetros no relevantes para el resumen
            exclude_keys = ['symbol', 'timeframe', 'strategy_name', 'initial_capital']
            params_summary = {k: v for k, v in params.items() if k not in exclude_keys}
            
            summary.append({
                'strategy_name': r.get('strategy', ''),
                'parameters': params_summary,
                'total_return': r.get('total_return', 0),
                'annual_return': r.get('annual_return', 0),
                'max_drawdown': r.get('max_drawdown', 0),
                'win_rate': r.get('win_rate', 0),
                'profit_factor': r.get('profit_factor', 0),
                'sharpe_ratio': r.get('sharpe_ratio', 0),
                'sortino_ratio': r.get('sortino_ratio', 0),
                'total_trades': r.get('total_trades', 0)
            })
        
        # Guardar a JSON
        try:
            summary_path = os.path.join(opt_dir, "summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error al guardar resumen de optimización: {str(e)}")
        
        # Guardar resumen en CSV para fácil análisis
        try:
            summary_df = pd.DataFrame(summary)
            csv_path = os.path.join(opt_dir, "summary.csv")
            summary_df.to_csv(csv_path, index=False)
        except Exception as e:
            self.logger.error(f"Error al guardar CSV de optimización: {str(e)}")
        
        # Guardar gráficos de comparación
        try:
            self._generate_optimization_charts(optimization_id, results, opt_dir)
        except Exception as e:
            self.logger.error(f"Error al generar gráficos de optimización: {str(e)}")
        
        self.logger.info(f"Resultados de optimización guardados en {opt_dir}")
    
    def _generate_result_charts(self, strategy_id: str, results: Dict, charts_dir: str) -> None:
        """
        Genera gráficos para visualizar los resultados del backtest
        
        Args:
            strategy_id: Identificador de la estrategia
            results: Resultados del backtest
            charts_dir: Directorio donde guardar los gráficos
        """
        # Configurar el estilo de los gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Convertir a DataFrame para facilitar el graficado
        if 'equity_curve' in results and results['equity_curve']:
            try:
                equity_df = pd.DataFrame(results['equity_curve'])
                
                # 1. Gráfico de curva de capital
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df['date'], equity_df['equity'])
                plt.title(f"Curva de Capital - {results.get('strategy', 'unnamed')}")
                plt.xlabel('Fecha')
                plt.ylabel('Capital ($)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, "equity_curve.png"), dpi=150)
                plt.close()
                
                # 2. Gráfico de drawdown
                equity_df['peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = equity_df['peak'] - equity_df['equity']
                equity_df['drawdown_pct'] = (equity_df['drawdown'] / equity_df['peak']) * 100
                
                plt.figure(figsize=(12, 6))
                plt.plot(equity_df['date'], equity_df['drawdown_pct'])
                plt.title(f"Drawdown (%) - {results.get('strategy', 'unnamed')}")
                plt.xlabel('Fecha')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(charts_dir, "drawdown.png"), dpi=150)
                plt.close()
                
                # 3. Gráfico de rendimientos mensuales
                if len(equity_df) > 30:
                    equity_df['month'] = equity_df['date'].dt.to_period('M')
                    monthly_returns = equity_df.groupby('month').apply(
                        lambda x: (x['equity'].iloc[-1] / x['equity'].iloc[0] - 1) * 100
                    )
                    
                    plt.figure(figsize=(14, 7))
                    ax = monthly_returns.plot(kind='bar', color=monthly_returns.map(lambda x: 'g' if x >= 0 else 'r'))
                    plt.title(f"Rendimientos Mensuales (%) - {results.get('strategy', 'unnamed')}")
                    plt.xlabel('Mes')
                    plt.ylabel('Rendimiento (%)')
                    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                    plt.grid(True, axis='y')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(charts_dir, "monthly_returns.png"), dpi=150)
                    plt.close()
            except Exception as e:
                self.logger.error(f"Error al generar gráficos de equity curve: {str(e)}")
        
        # 4. Gráfico de distribución de ganancias/pérdidas
        if 'trades' in results and results['trades']:
            try:
                trades_df = pd.DataFrame([t for t in results['trades'] if t['type'] == 'exit'])
                if not trades_df.empty and 'profit_pct' in trades_df.columns:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(trades_df['profit_pct'], kde=True, bins=20)
                    plt.title(f"Distribución de Ganancias/Pérdidas (%) - {results.get('strategy', 'unnamed')}")
                    plt.xlabel('Ganancia/Pérdida (%)')
                    plt.ylabel('Frecuencia')
                    plt.axvline(x=0, color='r', linestyle='--')
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(charts_dir, "profit_distribution.png"), dpi=150)
                    plt.close()
                    
                    # 5. Gráfico de ganancias/pérdidas por orden cronológico
                    if 'date' in trades_df.columns:
                        plt.figure(figsize=(12, 6))
                        plt.scatter(range(len(trades_df)), trades_df['profit_pct'],
                                   c=trades_df['profit_pct'].apply(lambda x: 'g' if x >= 0 else 'r'))
                        plt.title(f"Secuencia de Operaciones - {results.get('strategy', 'unnamed')}")
                        plt.xlabel('Número de Operación')
                        plt.ylabel('Ganancia/Pérdida (%)')
                        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig(os.path.join(charts_dir, "trade_sequence.png"), dpi=150)
                        plt.close()
                        
                        # 6. Gráfico de duración de las operaciones vs rendimiento
                        if 'date' in trades_df.columns:
                            # Calcular duración de operaciones
                            trades_df['entry_date'] = None
                            for i, row in trades_df.iterrows():
                                exit_date = row['date']
                                # Encontrar la entrada correspondiente
                                entry_trades = [t for t in results['trades'] if t['type'] == 'entry' and t['date'] <= exit_date]
                                if entry_trades:
                                    entry_date = max([t['date'] for t in entry_trades], key=lambda d: d)
                                    trades_df.at[i, 'entry_date'] = entry_date
                            
                            trades_df['duration'] = (trades_df['date'] - trades_df['entry_date']).dt.days
                            
                            plt.figure(figsize=(10, 6))
                            plt.scatter(trades_df['duration'], trades_df['profit_pct'],
                                      c=trades_df['profit_pct'].apply(lambda x: 'g' if x >= 0 else 'r'),
                                      alpha=0.7)
                            plt.title(f"Duración vs Rendimiento - {results.get('strategy', 'unnamed')}")
                            plt.xlabel('Duración (días)')
                            plt.ylabel('Ganancia/Pérdida (%)')
                            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                            plt.grid(True)
                            plt.tight_layout()
                            plt.savefig(os.path.join(charts_dir, "duration_vs_return.png"), dpi=150)
                            plt.close()
            except Exception as e:
                self.logger.error(f"Error al generar gráficos de operaciones: {str(e)}")
        
        # 7. Métricas clave en un solo gráfico
        try:
            plt.figure(figsize=(10, 8))
            metrics = {
                'Retorno Total (%)': results.get('total_return', 0),
                'Retorno Anual (%)': results.get('annual_return', 0),
                'Drawdown Máx. (%)': results.get('max_drawdown', 0),
                'Win Rate (%)': results.get('win_rate', 0),
                'Factor de Beneficio': results.get('profit_factor', 0),
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Sortino Ratio': results.get('sortino_ratio', 0)
            }
            
            colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', 
                     '#9467bd', '#8c564b', '#e377c2']
            
            plt.barh(list(metrics.keys()), list(metrics.values()), color=colors)
            plt.title(f"Métricas Clave - {results.get('strategy', 'unnamed')}")
            plt.xlabel('Valor')
            for i, v in enumerate(metrics.values()):
                plt.text(v, i, f"{v:.2f}", va='center')
            plt.tight_layout()
            plt.savefig(os.path.join(charts_dir, "key_metrics.png"), dpi=150)
            plt.close()
        except Exception as e:
            self.logger.error(f"Error al generar gráfico de métricas clave: {str(e)}")
    
    def _generate_optimization_charts(self, optimization_id: str, results: List[Dict], opt_dir: str) -> None:
        """
        Genera gráficos para visualizar los resultados de la optimización
        
        Args:
            optimization_id: Identificador de la optimización
            results: Lista de resultados
            opt_dir: Directorio donde guardar los gráficos
        """
        # Configurar el estilo de los gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Convertir a DataFrame para facilitar el graficado
        try:
            # Extraer parámetros y métricas
            optimization_data = []
            for r in results:
                params = r.get('parameters', {})
                # Excluir parámetros no relevantes
                exclude_keys = ['symbol', 'timeframe', 'strategy_name', 'initial_capital']
                params_extract = {k: v for k, v in params.items() if k not in exclude_keys}
                
                optimization_data.append({
                    'strategy': r.get('strategy', ''),
                    'total_return': r.get('total_return', 0),
                    'annual_return': r.get('annual_return', 0),
                    'max_drawdown': r.get('max_drawdown', 0),
                    'win_rate': r.get('win_rate', 0),
                    'profit_factor': r.get('profit_factor', 0),
                    'sharpe_ratio': r.get('sharpe_ratio', 0),
                    'sortino_ratio': r.get('sortino_ratio', 0),
                    'total_trades': r.get('total_trades', 0),
                    **params_extract
                })
            
            df = pd.DataFrame(optimization_data)
            
            if df.empty:
                self.logger.warning("No hay datos para generar gráficos de optimización")
                return
            
            # 1. Comparativa de retorno anual vs drawdown
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['max_drawdown'], df['annual_return'], 
                                 alpha=0.6, c=df['sharpe_ratio'], s=df['total_trades']*2,
                                 cmap='viridis')
            plt.colorbar(scatter, label='Sharpe Ratio')
            plt.title('Retorno Anual vs Drawdown Máximo')
            plt.xlabel('Drawdown Máximo (%)')
            plt.ylabel('Retorno Anual (%)')
            plt.grid(True)
            
            # Añadir etiquetas a puntos destacados
            top_returns = df.nlargest(3, 'sharpe_ratio')
            for i, row in top_returns.iterrows():
                label = ', '.join([f"{k}={v}" for k, v in row.items() 
                                 if k not in ['strategy', 'total_return', 'annual_return', 
                                             'max_drawdown', 'win_rate', 'profit_factor',
                                             'sharpe_ratio', 'sortino_ratio', 'total_trades']])
                plt.annotate(label, 
                            (row['max_drawdown'], row['annual_return']),
                            textcoords="offset points",
                            xytext=(0,10),
                            ha='center',
                            fontsize=8)
            
            plt.tight_layout()
            plt.savefig(os.path.join(opt_dir, "return_vs_drawdown.png"), dpi=150)
            plt.close()
            
            # 2. Top 10 estrategias por Sharpe Ratio
            top10 = df.nlargest(10, 'sharpe_ratio')
            param_cols = [col for col in top10.columns if col not in ['strategy', 'total_return', 
                                                                     'annual_return', 'max_drawdown', 
                                                                     'win_rate', 'profit_factor',
                                                                     'sharpe_ratio', 'sortino_ratio', 
                                                                     'total_trades']]
            
            # Crear etiquetas para las estrategias
            strategy_labels = []
            for i, row in top10.iterrows():
                label = ', '.join([f"{k}={row[k]}" for k in param_cols])
                strategy_labels.append(label)
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(strategy_labels)), top10['sharpe_ratio'], color='skyblue')
            plt.yticks(range(len(strategy_labels)), strategy_labels, fontsize=8)
            plt.title('Top 10 Estrategias por Sharpe Ratio')
            plt.xlabel('Sharpe Ratio')
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(opt_dir, "top10_sharpe.png"), dpi=150)
            plt.close()
            
            # 3. Relación entre Win Rate y Sharpe Ratio
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(df['win_rate'], df['sharpe_ratio'], 
                                 alpha=0.6, c=df['profit_factor'], s=df['total_trades']*2,
                                 cmap='viridis')
            plt.colorbar(scatter, label='Profit Factor')
            plt.title('Win Rate vs Sharpe Ratio')
            plt.xlabel('Win Rate (%)')
            plt.ylabel('Sharpe Ratio')
            plt.grid(True)
            
            # Añadir línea de tendencia
            if len(df) > 1:
                z = np.polyfit(df['win_rate'], df['sharpe_ratio'], 1)
                p = np.poly1d(z)
                plt.plot(df['win_rate'], p(df['win_rate']), "r--", alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(opt_dir, "winrate_vs_sharpe.png"), dpi=150)
            plt.close()
            
            # 4. Heatmap para visualizar relaciones de parámetros (para 2 parámetros)
            param_cols = [col for col in df.columns if col not in ['strategy', 'total_return', 
                                                                  'annual_return', 'max_drawdown', 
                                                                  'win_rate', 'profit_factor',
                                                                  'sharpe_ratio', 'sortino_ratio', 
                                                                  'total_trades']]
            
            if len(param_cols) >= 2:
                param1, param2 = param_cols[0], param_cols[1]
                
                if df[param1].nunique() > 1 and df[param2].nunique() > 1:
                    pivot_table = df.pivot_table(values='sharpe_ratio', 
                                               index=param1, 
                                               columns=param2, 
                                               aggfunc='mean')
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(pivot_table, annot=True, cmap='viridis', fmt=".2f")
                    plt.title(f'Sharpe Ratio: {param1} vs {param2}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(opt_dir, f"heatmap_{param1}_{param2}.png"), dpi=150)
                    plt.close()
            
                    # 5. Gráfico 3D para 3 parámetros si existen
                    if len(param_cols) >= 3:
                        param3 = param_cols[2]
                        
                        # Seleccionar solo algunos valores para claridad
                        unique_values = sorted(df[param3].unique())
                        selected_values = unique_values[:min(4, len(unique_values))]
                        
                        plt.figure(figsize=(12, 10))
                        for i, val in enumerate(selected_values):
                            subset = df[df[param3] == val]
                            pivot = subset.pivot_table(values='sharpe_ratio', 
                                                      index=param1, 
                                                      columns=param2, 
                                                      aggfunc='mean')
                            
                            plt.subplot(2, 2, i+1)
                            sns.heatmap(pivot, annot=True, cmap='viridis', fmt=".2f")
                            plt.title(f'{param3} = {val}')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(opt_dir, f"multi_heatmap_{param3}.png"), dpi=150)
                        plt.close()
        except Exception as e:
            self.logger.error(f"Error al generar gráficos de optimización: {str(e)}")

# Estrategias para backtesting

def ma_crossover_strategy(data: pd.DataFrame, params: Dict) -> List[Dict]:
    """
    Estrategia de cruce de medias móviles para backtesting
    
    Args:
        data: DataFrame con datos históricos
        params: Parámetros de la estrategia
    
    Returns:
        Lista de señales generadas
    """
    # Extraer parámetros
    fast_period = params.get('fast_period', 9)
    slow_period = params.get('slow_period', 21)
    atr_period = params.get('atr_period', 14)
    stop_loss_atr = params.get('stop_loss_atr', 2.0)
    take_profit_atr = params.get('take_profit_atr', 4.0)
    
    # Hacer una copia para no modificar los datos originales
    df = data.copy()
    
    # Calcular medias móviles
    df['fast_ma'] = df['close'].rolling(window=fast_period).mean()
    df['slow_ma'] = df['close'].rolling(window=slow_period).mean()
    
    # Calcular ATR para stop loss
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift())
    df['low_close'] = abs(df['low'] - df['close'].shift())
    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period).mean()
    
    # Inicializar señales
    signals = []
    position = 0
    stop_loss = 0
    take_profit = 0
    
    # Generar señales
    for i in range(1, len(df)):
        signal = None
        price = df['close'].iloc[i]
        date = df['date'].iloc[i]
        atr = df['atr'].iloc[i]
        
        # Si tenemos posición, verificar stop loss o take profit
        if position == 1:
            if price <= stop_loss:
                signal = 'sell'
                position = 0
                signals.append({
                    'date': date,
                    'signal': signal,
                    'price': price,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'reason': 'stop_loss'
                })
                continue
            
            if price >= take_profit:
                signal = 'sell'
                position = 0
                signals.append({
                    'date': date,
                    'signal': signal,
                    'price': price,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'reason': 'take_profit'
                })
                continue
        
        # Señal de compra: cruce hacia arriba
        if (df['fast_ma'].iloc[i-1] <= df['slow_ma'].iloc[i-1] and 
            df['fast_ma'].iloc[i] > df['slow_ma'].iloc[i] and
            position == 0):
            signal = 'buy'
            position = 1
            stop_loss = price - (atr * stop_loss_atr)
            take_profit = price + (atr * take_profit_atr)
        
        # Señal de venta: cruce hacia abajo
        elif (df['fast_ma'].iloc[i-1] >= df['slow_ma'].iloc[i-1] and 
              df['fast_ma'].iloc[i] < df['slow_ma'].iloc[i] and
              position == 1):
            signal = 'sell'
            position = 0
            stop_loss = 0
            take_profit = 0
        
        if signal:
            signals.append({
                'date': date,
                'signal': signal,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': 'signal'
            })
    
    return signals

def rsi_strategy(data: pd.DataFrame, params: Dict) -> List[Dict]:
    """
    Estrategia basada en RSI para backtesting
    
    Args:
        data: DataFrame con datos históricos
        params: Parámetros de la estrategia
    
    Returns:
        Lista de señales generadas
    """
    # Extraer parámetros
    rsi_period = params.get('rsi_period', 14)
    rsi_oversold = params.get('rsi_oversold', 30)
    rsi_overbought = params.get('rsi_overbought', 70)
    stop_loss_pct = params.get('stop_loss_pct', 0.05)  # 5%
    take_profit_pct = params.get('take_profit_pct', 0.1)  # 10%
    
    # Hacer una copia para no modificar los datos originales
    df = data.copy()
    
    # Calcular RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Inicializar señales
    signals = []
    position = 0
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    
    # Generar señales
    for i in range(1, len(df)):
        signal = None
        price = df['close'].iloc[i]
        date = df['date'].iloc[i]
        rsi = df['rsi'].iloc[i]
        rsi_prev = df['rsi'].iloc[i-1]
        
        # Si tenemos posición, verificar stop loss o take profit
        if position == 1:
            if price <= stop_loss:
                signal = 'sell'
                position = 0
                entry_price = 0
                signals.append({
                    'date': date,
                    'signal': signal,
                    'price': price,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'reason': 'stop_loss'
                })
                continue
            
            if price >= take_profit:
                signal = 'sell'
                position = 0
                entry_price = 0
                signals.append({
                    'date': date,
                    'signal': signal,
                    'price': price,
                    'stop_loss': 0,
                    'take_profit': 0,
                    'reason': 'take_profit'
                })
                continue
        
        # Señal de compra: RSI cruza hacia arriba desde zona de sobreventa
        if rsi_prev < rsi_oversold and rsi > rsi_oversold and position == 0:
            signal = 'buy'
            position = 1
            entry_price = price
            stop_loss = price * (1 - stop_loss_pct)
            take_profit = price * (1 + take_profit_pct)
        
        # Señal de venta: RSI cruza hacia abajo desde zona de sobrecompra
        elif rsi_prev > rsi_overbought and rsi < rsi_overbought and position == 1:
            signal = 'sell'
            position = 0
            entry_price = 0
            stop_loss = 0
            take_profit = 0
        
        if signal:
            signals.append({
                'date': date,
                'signal': signal,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': 'signal'
            })
    
    return signals

def run_backtest_command(args):
    """
    Función para ejecutar backtest desde línea de comandos
    
    Args:
        args: Argumentos del parser
    """
    backtester = Backtester(args.config)
    
    # Cargar datos
    if args.symbols:
        symbols = args.symbols.split(',')
    else:
        symbols = backtester.config['symbols']
    
    if args.timeframes:
        timeframes = args.timeframes.split(',')
    else:
        timeframes = backtester.config['timeframes']
    
    backtester.load_data(symbols, timeframes)
    
    # Seleccionar estrategia
    if args.strategy == 'ma_crossover':
        strategy_function = ma_crossover_strategy
    elif args.strategy == 'rsi':
        strategy_function = rsi_strategy
    else:
        raise ValueError(f"Estrategia no soportada: {args.strategy}")
    
    # Configurar parámetros
    params = {
        'symbol': args.symbol,
        'timeframe': args.timeframe,
        'strategy_name': args.strategy
    }
    
    # Añadir parámetros específicos de la estrategia
    if args.strategy == 'ma_crossover':
        params.update({
            'fast_period': args.fast_period,
            'slow_period': args.slow_period,
            'atr_period': args.atr_period,
            'stop_loss_atr': args.stop_loss_atr,
            'take_profit_atr': args.take_profit_atr
        })
    elif args.strategy == 'rsi':
        params.update({
            'rsi_period': args.rsi_period,
            'rsi_oversold': args.rsi_oversold,
            'rsi_overbought': args.rsi_overbought,
            'stop_loss_pct': args.stop_loss_pct,
            'take_profit_pct': args.take_profit_pct
        })
    
    # Ejecutar backtest
    if args.optimize:
        # Configurar grid de parámetros para optimización
        param_grid = {}
        
        if args.strategy == 'ma_crossover':
            if args.fast_periods:
                param_grid['fast_period'] = [int(p) for p in args.fast_periods.split(',')]
            
            if args.slow_periods:
                param_grid['slow_period'] = [int(p) for p in args.slow_periods.split(',')]
            
            if args.stop_loss_atrs:
                param_grid['stop_loss_atr'] = [float(p) for p in args.stop_loss_atrs.split(',')]
        
        elif args.strategy == 'rsi':
            if args.rsi_periods:
                param_grid['rsi_period'] = [int(p) for p in args.rsi_periods.split(',')]
            
            if args.rsi_oversolds:
                param_grid['rsi_oversold'] = [int(p) for p in args.rsi_oversolds.split(',')]
            
            if args.rsi_overboughts:
                param_grid['rsi_overbought'] = [int(p) for p in args.rsi_overboughts.split(',')]
        
        # Si no se especificó grid, usar el de la configuración
        if not param_grid:
            param_grid = None
        
        results = backtester.run_parameter_optimization(strategy_function, params, param_grid)
        print(f"Optimización completada. ID: {results.get('optimization_id')}")
        
        if results.get('best_params'):
            best = results['best_params']
            print(f"\nMejores parámetros encontrados:")
            for k, v in best.get('parameters', {}).items():
                if k not in ['symbol', 'timeframe', 'strategy_name']:
                    print(f"  {k}: {v}")
            
            print(f"\nMétricas con los mejores parámetros:")
            print(f"  Retorno Total: {best.get('total_return', 0):.2f}%")
            print(f"  Retorno Anual: {best.get('annual_return', 0):.2f}%")
            print(f"  Max Drawdown: {best.get('max_drawdown', 0):.2f}%")
            print(f"  Win Rate: {best.get('win_rate', 0):.2f}%")
            print(f"  Sharpe Ratio: {best.get('sharpe_ratio', 0):.2f}")
            print(f"  Operaciones: {best.get('total_trades', 0)}")
    else:
        results = backtester.run_backtest(strategy_function, params)
        print(f"Backtest completado para {params['symbol']} con {params['strategy_name']}")
        print(f"\nMétricas:")
        print(f"  Retorno Total: {results.get('total_return', 0):.2f}%")
        print(f"  Retorno Anual: {results.get('annual_return', 0):.2f}%")
        print(f"  Max Drawdown: {results.get('max_drawdown', 0):.2f}%")
        print(f"  Win Rate: {results.get('win_rate', 0):.2f}%")
        print(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        print(f"  Operaciones: {results.get('total_trades', 0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sistema de backtesting para estrategias de trading')
    
    # Argumentos generales
    parser.add_argument('--config', type=str, help='Ruta al archivo de configuración')
    parser.add_argument('--symbols', type=str, help='Lista de símbolos separados por comas')
    parser.add_argument('--timeframes', type=str, help='Lista de timeframes separados por comas')
    
    # Argumentos para backtest específico
    parser.add_argument('--strategy', type=str, default='ma_crossover', 
                       choices=['ma_crossover', 'rsi'], help='Estrategia a ejecutar')
    parser.add_argument('--symbol', type=str, required=True, help='Símbolo para el backtest')
    parser.add_argument('--timeframe', type=str, default='1d', help='Timeframe para el backtest')
    
    # Optimización
    parser.add_argument('--optimize', action='store_true', help='Realizar optimización de parámetros')
    
    # Parámetros para MA Crossover
    parser.add_argument('--fast-period', type=int, default=9, help='Periodo de la media rápida')
    parser.add_argument('--slow-period', type=int, default=21, help='Periodo de la media lenta')
    parser.add_argument('--atr-period', type=int, default=14, help='Periodo para el ATR')
    parser.add_argument('--stop-loss-atr', type=float, default=2.0, help='Multiplicador de ATR para stop loss')
    parser.add_argument('--take-profit-atr', type=float, default=4.0, help='Multiplicador de ATR para take profit')
    
    # Parámetros para RSI
    parser.add_argument('--rsi-period', type=int, default=14, help='Periodo para el RSI')
    parser.add_argument('--rsi-oversold', type=int, default=30, help='Nivel de sobreventa')
    parser.add_argument('--rsi-overbought', type=int, default=70, help='Nivel de sobrecompra')
    parser.add_argument('--stop-loss-pct', type=float, default=0.05, help='Porcentaje para stop loss')
    parser.add_argument('--take-profit-pct', type=float, default=0.1, help='Porcentaje para take profit')
    
    # Parámetros para optimización
    parser.add_argument('--fast-periods', type=str, help='Periodos para optimización de media rápida')
    parser.add_argument('--slow-periods', type=str, help='Periodos para optimización de media lenta')
    parser.add_argument('--stop-loss-atrs', type=str, help='Multiplicadores ATR para optimización')
    parser.add_argument('--rsi-periods', type=str, help='Periodos para optimización de RSI')
    parser.add_argument('--rsi-oversolds', type=str, help='Niveles de sobreventa para optimización')
    parser.add_argument('--rsi-overboughts', type=str, help='Niveles de sobrecompra para optimización')
    
    args = parser.parse_args()
    
    run_backtest_command(args)
