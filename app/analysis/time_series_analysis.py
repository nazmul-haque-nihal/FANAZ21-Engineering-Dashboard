# app/analysis/time_series_analysis.py
import numpy as np
import pandas as pd
from scipy import signal
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
from app.config import Config

class TimeSeriesAnalyzer:
    def __init__(self, config):
        self.config = config
        self.datasets = {}  # Dictionary to store time series datasets
    
    def load_data(self, data, name='timeseries', time_index=None):
        """Load time series data for analysis"""
        if isinstance(data, np.ndarray):
            if time_index is None:
                time_index = pd.date_range(start='2020-01-01', periods=len(data), freq='D')
            series = pd.Series(data, index=time_index)
            self.datasets[name] = {'data': series, 'type': 'series'}
        elif isinstance(data, pd.Series):
            self.datasets[name] = {'data': data, 'type': 'series'}
        elif isinstance(data, pd.DataFrame):
            if time_index is not None:
                data.index = time_index
            self.datasets[name] = {'data': data, 'type': 'dataframe'}
        elif isinstance(data, dict):
            if time_index is None:
                time_index = pd.date_range(start='2020-01-01', periods=len(next(iter(data.values()))), freq='D')
            df = pd.DataFrame(data, index=time_index)
            self.datasets[name] = {'data': df, 'type': 'dataframe'}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return name
    
    def check_stationarity(self, dataset_name, column=None):
        """Check stationarity of a time series using Augmented Dickey-Fuller test"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for stationarity check: {data_type}")
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'used_lag': result[2],
            'n_obs': result[3],
            'critical_values': result[4],
            'is_stationary': result[1] <= 0.05,
            'conclusion': 'Stationary' if result[1] <= 0.05 else 'Non-stationary'
        }
    
    def decompose_time_series(self, dataset_name, column=None, model='additive', period=None):
        """Decompose a time series into trend, seasonal, and residual components"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for decomposition: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        # Determine period if not provided
        if period is None:
            # Try to infer period from data frequency
            if hasattr(series.index, 'freq') and series.index.freq is not None:
                # For daily data with yearly seasonality
                if series.index.freqstr == 'D':
                    period = 365
                # For monthly data with yearly seasonality
                elif series.index.freqstr == 'M':
                    period = 12
                # For quarterly data with yearly seasonality
                elif series.index.freqstr == 'Q':
                    period = 4
                else:
                    # Default to a reasonable value
                    period = min(12, len(series) // 2)
            else:
                # Default to a reasonable value
                period = min(12, len(series) // 2)
        
        # Perform decomposition
        decomposition = seasonal_decompose(series, model=model, period=period)
        
        return {
            'original': series,
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'model': model,
            'period': period
        }
    
    def autocorrelation_analysis(self, dataset_name, column=None, lags=40):
        """Perform autocorrelation and partial autocorrelation analysis"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for autocorrelation analysis: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        # Calculate autocorrelation function
        acf_values, acf_confint = acf(series, nlags=lags, alpha=0.05, fft=True)
        
        # Calculate partial autocorrelation function
        pacf_values, pacf_confint = pacf(series, nlags=lags, alpha=0.05)
        
        return {
            'acf': {
                'values': acf_values,
                'confint': acf_confint,
                'lags': np.arange(len(acf_values))
            },
            'pacf': {
                'values': pacf_values,
                'confint': pacf_confint,
                'lags': np.arange(len(pacf_values))
            }
        }
    
    def fit_arima_model(self, dataset_name, column=None, order=(1, 1, 1), seasonal_order=None):
        """Fit an ARIMA model to a time series"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for ARIMA modeling: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        # Fit ARIMA model
        if seasonal_order is None:
            model = ARIMA(series, order=order)
        else:
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
        
        fitted_model = model.fit()
        
        # Get model summary
        summary = fitted_model.summary().as_text()
        
        # Get residuals
        residuals = fitted_model.resid
        
        # Get predictions
        predictions = fitted_model.predict(start=1, end=len(series))
        
        # Get forecast
        forecast_steps = min(10, len(series) // 4)
        forecast_result = fitted_model.forecast(steps=forecast_steps)
        forecast_conf_int = fitted_model.get_forecast(steps=forecast_steps).conf_int()
        
        return {
            'model_type': 'ARIMA',
            'order': order,
            'seasonal_order': seasonal_order,
            'summary': summary,
            'params': fitted_model.params.to_dict(),
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'residuals': residuals,
            'predictions': predictions,
            'forecast': forecast_result,
            'forecast_conf_int': forecast_conf_int,
            'original': series
        }
    
    def detect_anomalies(self, dataset_name, column=None, method='zscore', threshold=3.0):
        """Detect anomalies in a time series"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for anomaly detection: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        anomalies = None
        anomaly_indices = None
        
        if method == 'zscore':
            # Z-score method
            z_scores = np.abs((series - series.mean()) / series.std())
            anomaly_mask = z_scores > threshold
            anomalies = series[anomaly_mask]
            anomaly_indices = series.index[anomaly_mask]
            
            result = {
                'method': 'Z-score',
                'threshold': threshold,
                'z_scores': z_scores,
                'anomalies': anomalies,
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomalies)
            }
        
        elif method == 'iqr':
            # IQR method
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            anomaly_mask = (series < lower_bound) | (series > upper_bound)
            anomalies = series[anomaly_mask]
            anomaly_indices = series.index[anomaly_mask]
            
            result = {
                'method': 'IQR',
                'threshold': threshold,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'anomalies': anomalies,
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomalies)
            }
        
        elif method == 'rolling_zscore':
            # Rolling Z-score method
            window = min(30, len(series) // 4)
            rolling_mean = series.rolling(window=window).mean()
            rolling_std = series.rolling(window=window).std()
            rolling_z_scores = np.abs((series - rolling_mean) / rolling_std)
            anomaly_mask = rolling_z_scores > threshold
            anomalies = series[anomaly_mask]
            anomaly_indices = series.index[anomaly_mask]
            
            result = {
                'method': 'Rolling Z-score',
                'threshold': threshold,
                'window': window,
                'rolling_mean': rolling_mean,
                'rolling_std': rolling_std,
                'rolling_z_scores': rolling_z_scores,
                'anomalies': anomalies,
                'anomaly_indices': anomaly_indices,
                'anomaly_count': len(anomalies)
            }
        
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        return result
    
    def calculate_moving_averages(self, dataset_name, column=None, windows=[7, 14, 30]):
        """Calculate moving averages for a time series"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for moving averages: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        # Calculate moving averages
        moving_averages = {}
        for window in windows:
            if window < len(series):
                moving_averages[f'MA_{window}'] = series.rolling(window=window).mean()
        
        return {
            'original': series,
            'moving_averages': moving_averages
        }
    
    def calculate_exponential_smoothing(self, dataset_name, column=None, alpha=0.3):
        """Calculate exponential smoothing for a time series"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract time series
        if data_type == 'series':
            series = data
        elif data_type == 'dataframe':
            if column is not None:
                series = data[column]
            else:
                # Use the first column
                series = data.iloc[:, 0]
        else:
            raise ValueError(f"Unsupported data type for exponential smoothing: {data_type}")
        
        # Drop NaN values
        series = series.dropna()
        
        # Calculate exponential smoothing
        exp_smoothing = series.ewm(alpha=alpha).mean()
        
        return {
            'original': series,
            'exp_smoothing': exp_smoothing,
            'alpha': alpha
        }
    
    def get_dataset(self, dataset_name):
        """Get a dataset by name"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        return self.datasets[dataset_name]
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.datasets.keys())
    
    def remove_dataset(self, dataset_name):
        """Remove a dataset by name"""
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            return True
        return False