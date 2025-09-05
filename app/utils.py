import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
import plotly.graph_objs as go
import plotly.express as px
from flask_login import current_user
from app.auth import is_admin, is_engineer, is_analyst

def check_permission(required_permission):
    """Check if current user has the required permission"""
    if required_permission == 'admin':
        return is_admin(current_user)
    elif required_permission == 'engineer':
        return is_engineer(current_user)
    elif required_permission == 'analyst':
        return is_analyst(current_user)
    return False

def generate_sine_wave(frequency, duration, sample_rate=1000, amplitude=1.0, phase=0):
    """Generate a sine wave"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    y = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return t, y

def generate_square_wave(frequency, duration, sample_rate=1000, amplitude=1.0, duty=0.5):
    """Generate a square wave"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    y = amplitude * signal.square(2 * np.pi * frequency * t, duty=duty)
    return t, y

def generate_sawtooth_wave(frequency, duration, sample_rate=1000, amplitude=1.0, width=0.5):
    """Generate a sawtooth wave"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    y = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=width)
    return t, y

def generate_noise(duration, sample_rate=1000, amplitude=0.1, noise_type='white'):
    """Generate noise signal"""
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    if noise_type == 'white':
        y = amplitude * np.random.normal(0, 1, len(t))
    elif noise_type == 'pink':
        # Generate white noise
        white = np.random.normal(0, 1, len(t))
        # Apply 1/f filter
        f = np.fft.rfft(white)
        f[1:] = f[1:] / np.sqrt(np.arange(1, len(f)))
        y = amplitude * np.fft.irfft(f)
    elif noise_type == 'brown':
        # Generate white noise
        white = np.random.normal(0, 1, len(t))
        # Apply 1/f^2 filter
        f = np.fft.rfft(white)
        f[1:] = f[1:] / np.arange(1, len(f))
        y = amplitude * np.fft.irfft(f)
    else:
        y = amplitude * np.random.normal(0, 1, len(t))
    
    return t, y

def calculate_fft(y, sample_rate):
    """Calculate FFT of a signal"""
    n = len(y)
    yf = fft(y)
    xf = fftfreq(n, 1/sample_rate)[:n//2]
    return xf, 2.0/n * np.abs(yf[0:n//2])

def calculate_psd(y, sample_rate, method='welch'):
    """Calculate Power Spectral Density"""
    if method == 'welch':
        f, Pxx = signal.welch(y, sample_rate)
        return f, Pxx
    elif method == 'periodogram':
        f, Pxx = signal.periodogram(y, sample_rate)
        return f, Pxx
    else:
        # Default to FFT
        xf, yf = calculate_fft(y, sample_rate)
        return xf, yf**2

def calculate_snr(signal, noise):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    return 10 * np.log10(signal_power / noise_power)

def calculate_thd(signal, fundamental_freq, sample_rate):
    """Calculate Total Harmonic Distortion"""
    # Find fundamental component
    n = len(signal)
    yf = fft(signal)
    xf = fftfreq(n, 1/sample_rate)[:n//2]
    
    # Find index of fundamental frequency
    fund_idx = np.argmin(np.abs(xf - fundamental_freq))
    fundamental_amplitude = 2.0/n * np.abs(yf[fund_idx])
    
    # Calculate sum of harmonics (2nd to 9th)
    harmonic_sum = 0
    for h in range(2, 10):
        harmonic_freq = h * fundamental_freq
        if harmonic_freq < sample_rate/2:
            harmonic_idx = np.argmin(np.abs(xf - harmonic_freq))
            harmonic_amplitude = 2.0/n * np.abs(yf[harmonic_idx])
            harmonic_sum += harmonic_amplitude**2
    
    # Calculate THD
    thd = np.sqrt(harmonic_sum) / fundamental_amplitude
    return thd * 100  # Return as percentage

def filter_signal(y, sample_rate, filter_type, cutoff_freq, order=5):
    """Apply a filter to a signal"""
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    
    if filter_type == 'lowpass':
        b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    elif filter_type == 'highpass':
        b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    elif filter_type == 'bandpass':
        if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
            low = cutoff_freq[0] / nyquist
            high = cutoff_freq[1] / nyquist
            b, a = signal.butter(order, [low, high], btype='band', analog=False)
        else:
            raise ValueError("Bandpass filter requires a list/tuple of two cutoff frequencies")
    elif filter_type == 'bandstop':
        if isinstance(cutoff_freq, (list, tuple)) and len(cutoff_freq) == 2:
            low = cutoff_freq[0] / nyquist
            high = cutoff_freq[1] / nyquist
            b, a = signal.butter(order, [low, high], btype='bandstop', analog=False)
        else:
            raise ValueError("Bandstop filter requires a list/tuple of two cutoff frequencies")
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    filtered_signal = signal.filtfilt(b, a, y)
    return filtered_signal

def create_transfer_function(num, den, frequencies):
    """Create transfer function from numerator and denominator coefficients"""
    s = 1j * 2 * np.pi * frequencies
    H = np.zeros_like(frequencies, dtype=complex)
    
    for i, freq in enumerate(frequencies):
        numerator = 0
        for j, coef in enumerate(num):
            numerator += coef * (s[i] ** (len(num) - 1 - j))
        
        denominator = 0
        for j, coef in enumerate(den):
            denominator += coef * (s[i] ** (len(den) - 1 - j))
        
        H[i] = numerator / denominator
    
    return H

def calculate_step_response(num, den, t):
    """Calculate step response of a system"""
    _, y = signal.step((num, den), T=t)
    return y

def calculate_impulse_response(num, den, t):
    """Calculate impulse response of a system"""
    _, y = signal.impulse((num, den), T=t)
    return y

def calculate_system_metrics(t, y, u):
    """Calculate system metrics like rise time, settling time, overshoot"""
    # Find steady state value
    steady_state = np.mean(y[-int(len(y)*0.1):])
    
    # Find rise time (10% to 90% of steady state)
    y_10 = 0.1 * steady_state
    y_90 = 0.9 * steady_state
    
    idx_10 = np.where(y >= y_10)[0]
    idx_90 = np.where(y >= y_90)[0]
    
    if len(idx_10) > 0 and len(idx_90) > 0:
        rise_time = t[idx_90[0]] - t[idx_10[0]]
    else:
        rise_time = np.nan
    
    # Find settling time (within 2% of steady state)
    settling_threshold = 0.02 * steady_state
    settling_idx = np.where(np.abs(y - steady_state) <= settling_threshold)[0]
    
    if len(settling_idx) > 0:
        settling_time = t[settling_idx[0]]
    else:
        settling_time = np.nan
    
    # Find overshoot
    max_value = np.max(y)
    overshoot = ((max_value - steady_state) / steady_state) * 100 if steady_state != 0 else np.nan
    
    return {
        'rise_time': rise_time,
        'settling_time': settling_time,
        'overshoot': overshoot,
        'steady_state': steady_state
    }
