# data/real_time_data.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import json
import os

def generate_real_time_data():
    """Generate real-time data for the dashboard"""
    # Generate new data point
    products = ['Product A', 'Product B', 'Product C', 'Product D']
    regions = ['North', 'South', 'East', 'West']
    
    new_data = {
        'Date': datetime.now().strftime('%Y-%m-%d'),
        'Product': random.choice(products),
        'Region': random.choice(regions),
        'Sales': random.randint(8000, 15000),
        'Units': random.randint(90, 180)
    }
    
    # Read existing data
    try:
        df = pd.read_csv('data/sample_data.csv')
    except FileNotFoundError:
        # Create new DataFrame if file doesn't exist
        df = pd.DataFrame(columns=['Date', 'Product', 'Region', 'Sales', 'Units'])
    
    # Append new data
    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
    
    # Keep only last 100 records
    df = df.tail(100)
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # Save back to CSV
    df.to_csv('data/sample_data.csv', index=False)
    
    return new_data

def generate_real_time_simulation_data():
    """Generate real-time simulation data"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate signal with random frequency
    freq = random.uniform(1, 20)
    y = np.sin(2 * np.pi * freq * t) + 0.1 * np.random.normal(0, 1, len(t))
    
    # Generate circuit response with random parameters
    R = random.uniform(100, 1000)
    C = random.uniform(1e-6, 10e-6)
    tau = R * C
    
    v_in = np.ones_like(t)
    v_out = 1.0 * (1 - np.exp(-t / tau))
    
    # Generate control system response with random parameters
    K = random.uniform(0.5, 2.0)
    tau = random.uniform(0.05, 0.2)
    
    u = np.ones_like(t)
    y = K * (1 - np.exp(-t / tau))
    
    # Ensure simulation data directory exists
    os.makedirs('data/simulation', exist_ok=True)
    
    # Save data
    data = {
        'timestamp': datetime.now().isoformat(),
        'signal': {
            't': t.tolist(),
            'y': y.tolist(),
            'frequency': freq
        },
        'circuit': {
            't': t.tolist(),
            'v_in': v_in.tolist(),
            'v_out': v_out.tolist(),
            'R': R,
            'C': C,
            'tau': tau
        },
        'control': {
            't': t.tolist(),
            'u': u.tolist(),
            'y': y.tolist(),
            'K': K,
            'tau': tau
        }
    }
    
    with open('data/simulation/real_time_simulation_data.json', 'w') as f:
        json.dump(data, f)
    
    return data

def generate_real_time_analysis_data():
    """Generate real-time analysis data"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate signal with random frequency components
    freq1 = random.uniform(5, 15)
    freq2 = random.uniform(20, 50)
    freq3 = random.uniform(60, 100)
    
    y = (1.0 * np.sin(2 * np.pi * freq1 * t) + 
         0.5 * np.sin(2 * np.pi * freq2 * t) + 
         0.2 * np.sin(2 * np.pi * freq3 * t) + 
         0.1 * np.random.normal(0, 1, len(t)))
    
    # Generate time series data
    dates = pd.date_range(start=datetime.now() - timedelta(days=30), periods=30, freq='D')
    trend = np.linspace(100, 200, 30)
    seasonality = 10 * np.sin(2 * np.pi * np.arange(30) / 30.25 * 4)
    noise = np.random.normal(0, 5, 30)
    time_series = trend + seasonality + noise
    
    # Generate multivariate data
    n_samples = 100
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)
    x3 = np.random.normal(0, 1, n_samples)
    x4 = -0.3 * x1 + 0.7 * x3 + np.random.normal(0, 0.3, n_samples)
    
    # Ensure analysis data directory exists
    os.makedirs('data/analysis', exist_ok=True)
    
    # Save data
    data = {
        'timestamp': datetime.now().isoformat(),
        'time_domain': {
            't': t.tolist(),
            'y': y.tolist(),
            'frequencies': [freq1, freq2, freq3]
        },
        'time_series': {
            'dates': [d.isoformat() for d in dates],
            'values': time_series.tolist()
        },
        'multivariate': {
            'x1': x1.tolist(),
            'x2': x2.tolist(),
            'x3': x3.tolist(),
            'x4': x4.tolist()
        }
    }
    
    with open('data/analysis/real_time_analysis_data.json', 'w') as f:
        json.dump(data, f)
    
    return data

def generate_real_time_modeling_data():
    """Generate real-time modeling data"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate sine wave with random parameters
    A = random.uniform(0.5, 2.0)
    f = random.uniform(1, 20)
    phi = random.uniform(0, 2 * np.pi)
    offset = random.uniform(-0.5, 0.5)
    
    y = A * np.sin(2 * np.pi * f * t + phi) + offset + 0.1 * np.random.normal(0, 1, len(t))
    
    # Generate exponential decay with random parameters
    A_exp = random.uniform(0.5, 2.0)
    tau = random.uniform(0.05, 0.3)
    offset_exp = random.uniform(-0.5, 0.5)
    
    y_exp = A_exp * np.exp(-t / tau) + offset_exp + 0.05 * np.random.normal(0, 1, len(t))
    
    # Generate polynomial with random parameters
    a = random.uniform(-0.2, 0.2)
    b = random.uniform(-1.0, 1.0)
    c = random.uniform(0.0, 5.0)
    
    y_poly = a * t**2 + b * t + c + 0.2 * np.random.normal(0, 1, len(t))
    
    # Generate circuit response with random parameters
    R = random.uniform(100, 1000)
    C = random.uniform(1e-6, 10e-6)
    tau = R * C
    
    v_in = np.ones_like(t)
    v_out = 1.0 * (1 - np.exp(-t / tau)) + 0.02 * np.random.normal(0, 1, len(t))
    
    # Generate system response with random parameters
    K = random.uniform(0.5, 2.0)
    tau_sys = random.uniform(0.05, 0.3)
    
    u = np.ones_like(t)
    y_sys = K * (1 - np.exp(-t / tau_sys)) + 0.02 * np.random.normal(0, 1, len(t))
    
    # Ensure modeling data directory exists
    os.makedirs('data/modeling', exist_ok=True)
    
    # Save data
    data = {
        'timestamp': datetime.now().isoformat(),
        'signal': {
            't': t.tolist(),
            'y': y.tolist(),
            'A': A,
            'f': f,
            'phi': phi,
            'offset': offset
        },
        'exponential': {
            't': t.tolist(),
            'y': y_exp.tolist(),
            'A': A_exp,
            'tau': tau,
            'offset': offset_exp
        },
        'polynomial': {
            't': t.tolist(),
            'y': y_poly.tolist(),
            'a': a,
            'b': b,
            'c': c
        },
        'circuit': {
            't': t.tolist(),
            'v_in': v_in.tolist(),
            'v_out': v_out.tolist(),
            'R': R,
            'C': C,
            'tau': tau
        },
        'system': {
            't': t.tolist(),
            'u': u.tolist(),
            'y': y_sys.tolist(),
            'K': K,
            'tau': tau_sys
        }
    }
    
    with open('data/modeling/real_time_modeling_data.json', 'w') as f:
        json.dump(data, f)
    
    return data