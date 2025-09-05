# app/simulation/signal_simulation.py
import numpy as np
from scipy import signal
import plotly.graph_objs as go
from app.utils import (
    generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise,
    calculate_fft, calculate_psd, calculate_snr, calculate_thd, filter_signal
)

class SignalSimulator:
    def __init__(self, config):
        self.config = config
        self.sample_rate = 1000  # Default sample rate
        self.duration = 1.0  # Default duration in seconds
        self.signals = {}  # Dictionary to store generated signals
    
    def set_parameters(self, sample_rate=None, duration=None):
        """Set simulation parameters"""
        if sample_rate is not None:
            self.sample_rate = sample_rate
        if duration is not None:
            self.duration = duration
    
    def generate_signal(self, signal_type, **kwargs):
        """Generate a signal based on the specified type"""
        if signal_type == 'sine':
            frequency = kwargs.get('frequency', 5)
            amplitude = kwargs.get('amplitude', 1.0)
            phase = kwargs.get('phase', 0)
            t, y = generate_sine_wave(frequency, self.duration, self.sample_rate, amplitude, phase)
        
        elif signal_type == 'square':
            frequency = kwargs.get('frequency', 5)
            amplitude = kwargs.get('amplitude', 1.0)
            duty = kwargs.get('duty', 0.5)
            t, y = generate_square_wave(frequency, self.duration, self.sample_rate, amplitude, duty)
        
        elif signal_type == 'sawtooth':
            frequency = kwargs.get('frequency', 5)
            amplitude = kwargs.get('amplitude', 1.0)
            width = kwargs.get('width', 0.5)
            t, y = generate_sawtooth_wave(frequency, self.duration, self.sample_rate, amplitude, width)
        
        elif signal_type == 'noise':
            amplitude = kwargs.get('amplitude', 0.1)
            noise_type = kwargs.get('noise_type', 'white')
            t, y = generate_noise(self.duration, self.sample_rate, amplitude, noise_type)
        
        elif signal_type == 'composite':
            # Composite signal: sine + square + noise
            frequency = kwargs.get('frequency', 5)
            amplitude = kwargs.get('amplitude', 1.0)
            noise_amplitude = kwargs.get('noise_amplitude', 0.1)
            
            t, sine = generate_sine_wave(frequency, self.duration, self.sample_rate, amplitude)
            _, square = generate_square_wave(frequency * 0.5, self.duration, self.sample_rate, amplitude * 0.5)
            _, noise = generate_noise(self.duration, self.sample_rate, noise_amplitude)
            
            y = sine + square + noise
        
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Store the signal
        signal_id = f"{signal_type}_{len(self.signals)}"
        self.signals[signal_id] = {
            't': t,
            'y': y,
            'type': signal_type,
            'params': kwargs
        }
        
        return signal_id, t, y
    
    def add_signal(self, signal_id1, signal_id2, operation='add'):
        """Perform arithmetic operations on two signals"""
        if signal_id1 not in self.signals or signal_id2 not in self.signals:
            raise ValueError("One or both signal IDs not found")
        
        sig1 = self.signals[signal_id1]
        sig2 = self.signals[signal_id2]
        
        # Ensure signals have the same length
        min_len = min(len(sig1['t']), len(sig2['t']))
        t = sig1['t'][:min_len]
        y1 = sig1['y'][:min_len]
        y2 = sig2['y'][:min_len]
        
        if operation == 'add':
            y = y1 + y2
        elif operation == 'subtract':
            y = y1 - y2
        elif operation == 'multiply':
            y = y1 * y2
        elif operation == 'divide':
            # Avoid division by zero
            y2_safe = np.where(np.abs(y2) < 1e-10, 1e-10, y2)
            y = y1 / y2_safe
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Store the new signal
        new_signal_id = f"{signal_id1}_{operation}_{signal_id2}"
        self.signals[new_signal_id] = {
            't': t,
            'y': y,
            'type': 'combined',
            'params': {'operation': operation, 'signal1': signal_id1, 'signal2': signal_id2}
        }
        
        return new_signal_id, t, y
    
    def apply_filter(self, signal_id, filter_type, cutoff_freq, order=5):
        """Apply a filter to a signal"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal ID not found: {signal_id}")
        
        sig = self.signals[signal_id]
        t = sig['t']
        y = sig['y']
        
        # Apply the filter
        filtered_y = filter_signal(y, self.sample_rate, filter_type, cutoff_freq, order)
        
        # Store the filtered signal
        filtered_signal_id = f"{signal_id}_filtered"
        self.signals[filtered_signal_id] = {
            't': t,
            'y': filtered_y,
            'type': 'filtered',
            'params': {
                'original_signal': signal_id,
                'filter_type': filter_type,
                'cutoff_freq': cutoff_freq,
                'order': order
            }
        }
        
        return filtered_signal_id, t, filtered_y
    
    def analyze_signal(self, signal_id):
        """Analyze a signal and return metrics"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal ID not found: {signal_id}")
        
        sig = self.signals[signal_id]
        t = sig['t']
        y = sig['y']
        
        # Calculate FFT
        xf, yf = calculate_fft(y, self.sample_rate)
        
        # Calculate PSD
        f_psd, Pxx = calculate_psd(y, self.sample_rate)
        
        # Calculate SNR (assuming noise is the last 10% of the signal)
        signal_part = y[:int(len(y)*0.9)]
        noise_part = y[int(len(y)*0.9):]
        snr = calculate_snr(signal_part, noise_part)
        
        # Calculate THD (for periodic signals)
        # Find the fundamental frequency (peak in FFT)
        fundamental_idx = np.argmax(yf)
        fundamental_freq = xf[fundamental_idx]
        thd = calculate_thd(y, fundamental_freq, self.sample_rate)
        
        # Calculate RMS value
        rms = np.sqrt(np.mean(y**2))
        
        # Calculate peak value
        peak = np.max(np.abs(y))
        
        # Calculate crest factor
        crest_factor = peak / rms if rms > 0 else np.nan
        
        return {
            'fft': {'frequencies': xf, 'magnitude': yf},
            'psd': {'frequencies': f_psd, 'magnitude': Pxx},
            'snr': snr,
            'thd': thd,
            'rms': rms,
            'peak': peak,
            'crest_factor': crest_factor
        }
    
    def modulate_signal(self, signal_id, carrier_freq, modulation_type='am', modulation_index=0.5):
        """Modulate a signal with a carrier wave"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal ID not found: {signal_id}")
        
        sig = self.signals[signal_id]
        t = sig['t']
        y = sig['y']
        
        # Generate carrier wave
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        if modulation_type == 'am':  # Amplitude modulation
            # Normalize signal to [-1, 1]
            y_norm = y / (np.max(np.abs(y)) + 1e-10)
            # Modulate
            y_modulated = (1 + modulation_index * y_norm) * carrier
        
        elif modulation_type == 'fm':  # Frequency modulation
            # Normalize signal to [-1, 1]
            y_norm = y / (np.max(np.abs(y)) + 1e-10)
            # Calculate instantaneous frequency
            inst_freq = carrier_freq * (1 + modulation_index * y_norm)
            # Calculate phase by integrating frequency
            phase = 2 * np.pi * np.cumsum(inst_freq) / self.sample_rate
            # Generate FM signal
            y_modulated = np.sin(phase)
        
        elif modulation_type == 'pm':  # Phase modulation
            # Normalize signal to [-1, 1]
            y_norm = y / (np.max(np.abs(y)) + 1e-10)
            # Modulate phase
            phase = 2 * np.pi * carrier_freq * t + modulation_index * y_norm
            # Generate PM signal
            y_modulated = np.sin(phase)
        
        else:
            raise ValueError(f"Unknown modulation type: {modulation_type}")
        
        # Store the modulated signal
        modulated_signal_id = f"{signal_id}_modulated"
        self.signals[modulated_signal_id] = {
            't': t,
            'y': y_modulated,
            'type': 'modulated',
            'params': {
                'original_signal': signal_id,
                'carrier_freq': carrier_freq,
                'modulation_type': modulation_type,
                'modulation_index': modulation_index
            }
        }
        
        return modulated_signal_id, t, y_modulated
    
    def demodulate_signal(self, signal_id, carrier_freq, modulation_type='am'):
        """Demodulate a signal"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal ID not found: {signal_id}")
        
        sig = self.signals[signal_id]
        t = sig['t']
        y = sig['y']
        
        if modulation_type == 'am':  # Amplitude demodulation
            # Multiply by carrier
            y_demod = y * np.sin(2 * np.pi * carrier_freq * t)
            # Low-pass filter
            y_demod = filter_signal(y_demod, self.sample_rate, 'lowpass', carrier_freq * 2, order=5)
            # Remove DC component
            y_demod = y_demod - np.mean(y_demod)
        
        elif modulation_type == 'fm':  # Frequency demodulation
            # Hilbert transform to get analytic signal
            from scipy.signal import hilbert
            analytic_signal = hilbert(y)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * self.sample_rate
            # Scale to get original signal
            y_demod = (instantaneous_frequency - carrier_freq) / carrier_freq
            # Pad to match original length
            y_demod = np.pad(y_demod, (0, 1), 'constant')
        
        elif modulation_type == 'pm':  # Phase demodulation
            # Hilbert transform to get analytic signal
            from scipy.signal import hilbert
            analytic_signal = hilbert(y)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            # Remove carrier phase
            y_demod = instantaneous_phase - 2 * np.pi * carrier_freq * t
            # Unwrap phase to get continuous signal
            y_demod = np.unwrap(y_demod)
        
        else:
            raise ValueError(f"Unknown modulation type: {modulation_type}")
        
        # Store the demodulated signal
        demodulated_signal_id = f"{signal_id}_demodulated"
        self.signals[demodulated_signal_id] = {
            't': t,
            'y': y_demod,
            'type': 'demodulated',
            'params': {
                'original_signal': signal_id,
                'carrier_freq': carrier_freq,
                'modulation_type': modulation_type
            }
        }
        
        return demodulated_signal_id, t, y_demod
    
    def get_signal(self, signal_id):
        """Get a signal by ID"""
        if signal_id not in self.signals:
            raise ValueError(f"Signal ID not found: {signal_id}")
        return self.signals[signal_id]
    
    def list_signals(self):
        """List all available signals"""
        return list(self.signals.keys())
    
    def remove_signal(self, signal_id):
        """Remove a signal by ID"""
        if signal_id in self.signals:
            del self.signals[signal_id]
            return True
        return False