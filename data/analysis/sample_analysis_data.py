# data/analysis/sample_analysis_data.py
import numpy as np
import pandas as pd
from scipy import signal

def get_analysis_data():
    """Generate sample analysis data for electrical engineering"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate signal with multiple frequency components
    y = 1.0 * np.sin(2 * np.pi * 7 * t) + 0.5 * np.sin(2 * np.pi * 13 * t) + 0.2 * np.sin(2 * np.pi * 20 * t)
    
    # Add some noise
    y += np.random.normal(0, 0.1, len(t))
    
    # Generate time series data
    dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
    trend = np.linspace(100, 200, 365)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 365.25 * 4)  # Quarterly seasonality
    noise = np.random.normal(0, 5, 365)
    time_series = trend + seasonality + noise
    
    # Generate multivariate data for correlation analysis
    np.random.seed(42)
    n_samples = 100
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)  # Correlated with x1
    x3 = np.random.normal(0, 1, n_samples)  # Uncorrelated with x1 and x2
    x4 = -0.3 * x1 + 0.7 * x3 + np.random.normal(0, 0.3, n_samples)  # Partially correlated
    
    # Generate data for regression analysis
    x_reg = np.linspace(0, 10, 100)
    y_reg_linear = 2.5 * x_reg + np.random.normal(0, 1, 100)  # Linear relationship
    y_reg_poly = 0.1 * x_reg**2 - 0.5 * x_reg + 2 + np.random.normal(0, 1, 100)  # Polynomial relationship
    
    # Generate data for distribution analysis
    normal_data = np.random.normal(0, 1, 1000)
    exponential_data = np.random.exponential(1, 1000)
    uniform_data = np.random.uniform(-1, 1, 1000)
    
    # Generate data with outliers
    outlier_data = np.random.normal(0, 1, 100)
    # Add some outliers
    outlier_data[10] = 5.0
    outlier_data[20] = -4.5
    outlier_data[30] = 6.0
    
    # Generate frequency domain data
    fs = 1000  # Sampling frequency
    t_freq = np.arange(0, 1, 1/fs)
    
    # Create a signal with specific frequency components
    y_freq = 0.5 * np.sin(2 * np.pi * 50 * t_freq) + \
             0.3 * np.sin(2 * np.pi * 120 * t_freq) + \
             0.2 * np.sin(2 * np.pi * 300 * t_freq)
    
    # Add noise
    y_freq += 0.1 * np.random.normal(0, 1, len(t_freq))
    
    # Create a chirp signal (frequency changes over time)
    y_chirp = signal.chirp(t_freq, f0=10, f1=200, t1=1, method='linear')
    
    # Create a modulated signal
    carrier = np.sin(2 * np.pi * 100 * t_freq)
    modulator = np.sin(2 * np.pi * 10 * t_freq)
    y_am = (1 + 0.5 * modulator) * carrier
    
    # Return all data
    return {
        'time_domain': {
            't': t,
            'y': y
        },
        'time_series': {
            'dates': dates,
            'values': time_series
        },
        'multivariate': {
            'x1': x1,
            'x2': x2,
            'x3': x3,
            'x4': x4
        },
        'regression': {
            'x': x_reg,
            'y_linear': y_reg_linear,
            'y_poly': y_reg_poly
        },
        'distribution': {
            'normal': normal_data,
            'exponential': exponential_data,
            'uniform': uniform_data
        },
        'outliers': {
            'data': outlier_data
        },
        'frequency_domain': {
            't': t_freq,
            'y': y_freq,
            'y_chirp': y_chirp,
            'y_am': y_am,
            'fs': fs
        }
    }