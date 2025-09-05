import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import psutil
from sklearn.linear_model import LinearRegression
import json
import os
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import h5py
import scipy.io as sio

class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.sample_data_paths = config.SAMPLE_DATA_PATHS
        self.load_sample_data()
    
    def load_sample_data(self):
        self.sample_data = {
            'simulation': self._load_simulation_data(),
            'analysis': self._load_analysis_data(),
            'modeling': self._load_modeling_data()
        }
    
    def _load_simulation_data(self):
        try:
            from data.simulation.sample_simulation_data import get_simulation_data
            return get_simulation_data()
        except ImportError:
            return self._generate_default_simulation_data()
    
    def _load_analysis_data(self):
        try:
            from data.analysis.sample_analysis_data import get_analysis_data
            return get_analysis_data()
        except ImportError:
            return self._generate_default_analysis_data()
    
    def _load_modeling_data(self):
        try:
            from data.modeling.sample_model_data import get_modeling_data
            return get_modeling_data()
        except ImportError:
            return self._generate_default_modeling_data()
    
    def _generate_default_simulation_data(self):
        # Generate default simulation data for electrical engineering
        t = np.linspace(0, 1, 1000)
        signal_data = {
            'sine': np.sin(2 * np.pi * 5 * t),
            'square': signal.square(2 * np.pi * 5 * t),
            'sawtooth': signal.sawtooth(2 * np.pi * 5 * t),
            'noise': np.random.normal(0, 0.1, len(t))
        }
        return {'t': t, 'signals': signal_data}
    
    def _generate_default_analysis_data(self):
        # Generate default analysis data for electrical engineering
        t = np.linspace(0, 1, 1000)
        # Create a signal with multiple frequency components
        y = 1.0 * np.sin(2 * np.pi * 7 * t) + 0.5 * np.sin(2 * np.pi * 13 * t) + 0.2 * np.sin(2 * np.pi * 20 * t)
        # Add some noise
        y += np.random.normal(0, 0.1, len(t))
        return {'t': t, 'y': y}
    
    def _generate_default_modeling_data(self):
        # Generate default modeling data for electrical engineering
        # RLC circuit parameters
        R = 100  # Ohms
        L = 0.1  # Henry
        C = 1e-6  # Farad
        
        # Time vector
        t = np.linspace(0, 0.01, 1000)
        
        # Input voltage (step function)
        Vin = np.ones_like(t) * 5  # 5V step
        
        # Solve RLC circuit differential equation
        omega_0 = 1 / np.sqrt(L * C)
        zeta = R / (2 * np.sqrt(L / C))
        
        if zeta < 1:  # Underdamped
            omega_d = omega_0 * np.sqrt(1 - zeta**2)
            Vout = Vin * (1 - np.exp(-zeta * omega_0 * t) * 
                         (np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t)))
        elif zeta == 1:  # Critically damped
            Vout = Vin * (1 - np.exp(-omega_0 * t) * (1 + omega_0 * t))
        else:  # Overdamped
            s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
            s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
            Vout = Vin * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))
        
        return {'t': t, 'Vin': Vin, 'Vout': Vout, 'R': R, 'L': L, 'C': C}
    
    def get_system_metrics(self):
        cpu_percent = psutil.cpu_percent()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        net_io = psutil.net_io_counters()
        
        return {
            'cpu': cpu_percent,
            'memory': mem.percent,
            'disk': disk.percent,
            'network_sent': net_io.bytes_sent,
            'network_recv': net_io.bytes_recv
        }
    
    def get_top_processes(self):
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
            try:
                processes.append(proc.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
        # Sort by CPU percent and return top 5
        processes = sorted(processes, key=lambda p: p['cpu_percent'], reverse=True)
        return processes[:5]
    
    def process_uploaded_file(self, file_path, file_type):
        """Process uploaded file based on its type"""
        try:
            if file_type == 'csv':
                return pd.read_csv(file_path)
            elif file_type == 'xlsx':
                return pd.read_excel(file_path)
            elif file_type == 'json':
                with open(file_path, 'r') as f:
                    return json.load(f)
            elif file_type == 'txt':
                with open(file_path, 'r') as f:
                    return f.read()
            elif file_type == 'mat':
                return sio.loadmat(file_path)
            elif file_type == 'h5':
                with h5py.File(file_path, 'r') as f:
                    return dict(f)
            else:
                return None
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None
    
    def save_figure_to_image(self, fig, format='png'):
        """Save plotly figure to image and return as base64 string"""
        img_bytes = fig.to_image(format=format)
        encoded = base64.b64encode(img_bytes).decode()
        return f"data:image/{format};base64,{encoded}"
    
    def create_time_series_plot(self, t, y, title="Time Series", xlabel="Time", ylabel="Amplitude"):
        """Create a time series plot"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='Signal'))
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            template=self.config.CHART_TEMPLATE,
            height=self.config.CHART_HEIGHT
        )
        return fig
    
    def create_spectrum_plot(self, y, fs, title="Frequency Spectrum"):
        """Create a frequency spectrum plot"""
        n = len(y)
        yf = fft(y)
        xf = fftfreq(n, 1/fs)[:n//2]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xf, y=2.0/n * np.abs(yf[0:n//2]), mode='lines', name='Spectrum'))
        fig.update_layout(
            title=title,
            xaxis_title='Frequency (Hz)',
            yaxis_title='Amplitude',
            template=self.config.CHART_TEMPLATE,
            height=self.config.CHART_HEIGHT
        )
        return fig
    
    def create_phase_plot(self, y1, y2, title="Phase Plot"):
        """Create a phase plot (y1 vs y2)"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y1, y=y2, mode='lines', name='Phase'))
        fig.update_layout(
            title=title,
            xaxis_title='Signal 1',
            yaxis_title='Signal 2',
            template=self.config.CHART_TEMPLATE,
            height=self.config.CHART_HEIGHT
        )
        return fig
    
    def create_bode_plot(self, frequencies, magnitude, phase, title="Bode Plot"):
        """Create a Bode plot"""
        fig = go.Figure()
        
        # Magnitude plot
        fig.add_trace(go.Scatter(
            x=frequencies, 
            y=20 * np.log10(magnitude),
            mode='lines',
            name='Magnitude',
            yaxis='y'
        ))
        
        # Phase plot
        fig.add_trace(go.Scatter(
            x=frequencies, 
            y=phase,
            mode='lines',
            name='Phase',
            yaxis='y2'
        ))
        
        # Update layout with two y-axes
        fig.update_layout(
            title=title,
            xaxis_title='Frequency (Hz)',
            yaxis=dict(
                title='Magnitude (dB)',
                side='left'
            ),
            yaxis2=dict(
                title='Phase (degrees)',
                overlaying='y',
                side='right'
            ),
            template=self.config.CHART_TEMPLATE,
            height=self.config.CHART_HEIGHT
        )
        
        return fig
    
    def create_nyquist_plot(self, real, imag, title="Nyquist Plot"):
        """Create a Nyquist plot"""
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=real, y=imag, mode='lines', name='Nyquist'))
        fig.update_layout(
            title=title,
            xaxis_title='Real Part',
            yaxis_title='Imaginary Part',
            template=self.config.CHART_TEMPLATE,
            height=self.config.CHART_HEIGHT
        )
        return fig
