# app/modeling/models.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
import base64

def generate_model_data(model_type='default'):
    """Generate model data based on the model type"""
    np.random.seed(42)
    
    if model_type == 'linear':
        return generate_linear_model_data()
    elif model_type == 'polynomial':
        return generate_polynomial_model_data()
    elif model_type == 'exponential':
        return generate_exponential_model_data()
    elif model_type == 'logarithmic':
        return generate_logarithmic_model_data()
    else:
        return generate_default_model_data()

def generate_linear_model_data():
    """Generate linear model data"""
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def generate_polynomial_model_data():
    """Generate polynomial model data"""
    x = np.linspace(0, 10, 100)
    y = 0.5 * x**2 - 2 * x + 3 + np.random.normal(0, 2, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def generate_exponential_model_data():
    """Generate exponential model data"""
    x = np.linspace(0, 5, 100)
    y = 2 * np.exp(0.5 * x) + np.random.normal(0, 0.5, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def generate_logarithmic_model_data():
    """Generate logarithmic model data"""
    x = np.linspace(1, 10, 100)
    y = 3 * np.log(x) + 1 + np.random.normal(0, 0.2, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def generate_default_model_data():
    """Generate default model data"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def fit_linear_model(df):
    """Fit a linear model to the data"""
    X = df[['X']]
    y = df['Y']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Coefficients
    coefficients = {
        'intercept': model.intercept_,
        'slope': model.coef_[0]
    }
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': coefficients
    }

def fit_polynomial_model(df, degree=2):
    """Fit a polynomial model to the data"""
    X = df[['X']]
    y = df['Y']
    
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(X, y)
    
    # Predictions
    y_pred = model.predict(X)
    
    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Coefficients
    coefficients = {
        'intercept': model.named_steps['linearregression'].intercept_,
        'coefficients': model.named_steps['linearregression'].coef_[1:]
    }
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': coefficients,
        'degree': degree
    }

def fit_exponential_model(df):
    """Fit an exponential model to the data"""
    X = df[['X']]
    y = df['Y']
    
    # Transform y to log space
    log_y = np.log(y)
    
    # Fit linear model in log space
    model = LinearRegression()
    model.fit(X, log_y)
    
    # Predictions
    log_y_pred = model.predict(X)
    y_pred = np.exp(log_y_pred)
    
    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Coefficients
    coefficients = {
        'intercept': model.intercept_,
        'slope': model.coef_[0]
    }
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': coefficients
    }

def fit_logarithmic_model(df):
    """Fit a logarithmic model to the data"""
    X = df[['X']]
    y = df['Y']
    
    # Transform X to log space
    log_X = np.log(X)
    
    # Fit linear model with log X
    model = LinearRegression()
    model.fit(log_X, y)
    
    # Predictions
    y_pred = model.predict(log_X)
    
    # Metrics
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # Coefficients
    coefficients = {
        'intercept': model.intercept_,
        'slope': model.coef_[0]
    }
    
    return {
        'model': model,
        'predictions': y_pred,
        'mse': mse,
        'r2': r2,
        'coefficients': coefficients
    }

def fit_model(df, model_type='linear', **kwargs):
    """Fit a model to the data based on the model type"""
    if model_type == 'linear':
        return fit_linear_model(df)
    elif model_type == 'polynomial':
        degree = kwargs.get('degree', 2)
        return fit_polynomial_model(df, degree)
    elif model_type == 'exponential':
        return fit_exponential_model(df)
    elif model_type == 'logarithmic':
        return fit_logarithmic_model(df)
    else:
        return fit_linear_model(df)

def predict_with_model(model, X, model_type='linear'):
    """Make predictions using a fitted model"""
    if model_type == 'linear' or model_type == 'polynomial':
        return model.predict(X)
    elif model_type == 'exponential':
        log_y_pred = model.predict(X)
        return np.exp(log_y_pred)
    elif model_type == 'logarithmic':
        log_X = np.log(X)
        return model.predict(log_X)
    else:
        return model.predict(X)