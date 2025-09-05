# app/analysis/frequency_analysis.py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq, fftshift
from scipy.signal import spectrogram, stft, csd, coherence, hilbert
import plotly.graph_objs as go
from app.config import Config

class FrequencyAnalyzer:
    def __init__(self, config):
        self.config = config
        self.datasets = {}  # Dictionary to store datasets
    
    def load_data(self, data, name='dataset', sample_rate=None):
        """Load data for frequency analysis"""
        if isinstance(data, np.ndarray):
            self.datasets[name] = {
                'data': data, 
                'type': 'array',
                'sample_rate': sample_rate
            }
        elif isinstance(data, pd.DataFrame):
            self.datasets[name] = {
                'data': data, 
                'type': 'dataframe',
                'sample_rate': sample_rate
            }
        elif isinstance(data, dict):
            self.datasets[name] = {
                'data': data, 
                'type': 'dict',
                'sample_rate': sample_rate
            }
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return name
    
    def fft_analysis(self, dataset_name, column=None, window='hann', detrend='constant'):
        """Perform FFT analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    signal_data = data[:, column]
                else:
                    signal_data = data.flatten()
            else:
                signal_data = data
        elif data_type == 'dataframe':
            if column is not None:
                signal_data = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    signal_data = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                signal_data = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signal_data = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for FFT analysis: {data_type}")
        
        # Apply window function if specified
        if window is not None:
            if window == 'hann':
                window_func = np.hanning(len(signal_data))
            elif window == 'hamming':
                window_func = np.hamming(len(signal_data))
            elif window == 'blackman':
                window_func = np.blackman(len(signal_data))
            elif window == 'flattop':
                window_func = signal.flattop(len(signal_data))
            else:
                window_func = np.ones(len(signal_data))
            
            signal_data = signal_data * window_func
        
        # Apply detrending if specified
        if detrend is not None:
            if detrend == 'constant':
                signal_data = signal_data - np.mean(signal_data)
            elif detrend == 'linear':
                signal_data = signal.detrend(signal_data, type='linear')
        
        # Compute FFT
        n = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(n, 1/sample_rate)
        
        # Only return the positive frequencies
        positive_freq_idx = xf >= 0
        xf = xf[positive_freq_idx]
        yf = 2.0/n * np.abs(yf[positive_freq_idx])
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(yf, height=np.max(yf)*0.1)[0]
        peak_freqs = xf[peak_indices]
        peak_magnitudes = yf[peak_indices]
        
        # Sort peaks by magnitude
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_magnitudes = peak_magnitudes[sorted_idx]
        
        return {
            'frequencies': xf,
            'magnitude': yf,
            'peak_frequencies': peak_freqs.tolist(),
            'peak_magnitudes': peak_magnitudes.tolist(),
            'dominant_frequency': peak_freqs[0] if len(peak_freqs) > 0 else None,
            'window': window,
            'detrend': detrend,
            'sample_rate': sample_rate
        }
    
    def power_spectrum_analysis(self, dataset_name, column=None, method='welch', window='hann', nperseg=None):
        """Perform power spectrum analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    signal_data = data[:, column]
                else:
                    signal_data = data.flatten()
            else:
                signal_data = data
        elif data_type == 'dataframe':
            if column is not None:
                signal_data = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    signal_data = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                signal_data = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signal_data = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for power spectrum analysis: {data_type}")
        
        # Set default segment length if not provided
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 4)
        
        # Compute power spectrum
        if method == 'welch':
            f, Pxx = signal.welch(signal_data, fs=sample_rate, window=window, nperseg=nperseg)
        elif method == 'periodogram':
            f, Pxx = signal.periodogram(signal_data, fs=sample_rate, window=window)
        else:
            raise ValueError(f"Unknown power spectrum method: {method}")
        
        # Convert to dB scale
        Pxx_db = 10 * np.log10(Pxx)
        
        # Find dominant frequencies
        peak_indices = signal.find_peaks(Pxx_db, height=np.max(Pxx_db)-10)[0]
        peak_freqs = f[peak_indices]
        peak_powers = Pxx_db[peak_indices]
        
        # Sort peaks by power
        sorted_idx = np.argsort(peak_powers)[::-1]
        peak_freqs = peak_freqs[sorted_idx]
        peak_powers = peak_powers[sorted_idx]
        
        return {
            'frequencies': f,
            'power': Pxx,
            'power_db': Pxx_db,
            'peak_frequencies': peak_freqs.tolist(),
            'peak_powers': peak_powers.tolist(),
            'dominant_frequency': peak_freqs[0] if len(peak_freqs) > 0 else None,
            'method': method,
            'window': window,
            'nperseg': nperseg,
            'sample_rate': sample_rate
        }
    
    def spectrogram_analysis(self, dataset_name, column=None, window='hann', nperseg=None, noverlap=None):
        """Perform spectrogram analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    signal_data = data[:, column]
                else:
                    signal_data = data.flatten()
            else:
                signal_data = data
        elif data_type == 'dataframe':
            if column is not None:
                signal_data = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    signal_data = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                signal_data = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signal_data = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for spectrogram analysis: {data_type}")
        
        # Set default segment length if not provided
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 8)
        
        # Set default overlap if not provided
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute spectrogram
        f, t, Sxx = spectrogram(signal_data, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx)
        
        return {
            'frequencies': f,
            'times': t,
            'spectrogram': Sxx,
            'spectrogram_db': Sxx_db,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'sample_rate': sample_rate
        }
    
    def stft_analysis(self, dataset_name, column=None, window='hann', nperseg=None, noverlap=None):
        """Perform Short-Time Fourier Transform (STFT) analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    signal_data = data[:, column]
                else:
                    signal_data = data.flatten()
            else:
                signal_data = data
        elif data_type == 'dataframe':
            if column is not None:
                signal_data = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    signal_data = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                signal_data = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signal_data = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for STFT analysis: {data_type}")
        
        # Set default segment length if not provided
        if nperseg is None:
            nperseg = min(256, len(signal_data) // 8)
        
        # Set default overlap if not provided
        if noverlap is None:
            noverlap = nperseg // 2
        
        # Compute STFT
        f, t, Zxx = stft(signal_data, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)
        
        # Calculate magnitude and phase
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Convert magnitude to dB scale
        magnitude_db = 20 * np.log10(magnitude + 1e-10)
        
        return {
            'frequencies': f,
            'times': t,
            'stft': Zxx,
            'magnitude': magnitude,
            'magnitude_db': magnitude_db,
            'phase': phase,
            'window': window,
            'nperseg': nperseg,
            'noverlap': noverlap,
            'sample_rate': sample_rate
        }
    
    def coherence_analysis(self, dataset_name, column1=None, column2=None, window='hann', nperseg=None):
        """Perform coherence analysis between two signals in a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim < 2:
                raise ValueError("Coherence analysis requires at least 2D data")
            
            if column1 is not None and column2 is not None:
                signal1 = data[:, column1]
                signal2 = data[:, column2]
            else:
                # Use first two columns
                signal1 = data[:, 0]
                signal2 = data[:, 1]
        
        elif data_type == 'dataframe':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if column1 is not None and column2 is not None:
                if column1 in numeric_columns and column2 in numeric_columns:
                    signal1 = data[column1].values
                    signal2 = data[column2].values
                else:
                    raise ValueError("One or both specified columns are not numeric")
            else:
                if len(numeric_columns) >= 2:
                    signal1 = data[numeric_columns[0]].values
                    signal2 = data[numeric_columns[1]].values
                else:
                    raise ValueError("DataFrame must have at least two numeric columns")
        
        elif data_type == 'dict':
            keys = list(data.keys())
            arrays = []
            
            for key in keys:
                if isinstance(data[key], (list, np.ndarray)):
                    arrays.append(np.array(data[key]))
            
            if len(arrays) < 2:
                raise ValueError("Dictionary must contain at least two arrays")
            
            if column1 is not None and column2 is not None:
                if column1 in keys and column2 in keys:
                    signal1 = np.array(data[column1])
                    signal2 = np.array(data[column2])
                else:
                    raise ValueError("One or both specified keys not found in dictionary")
            else:
                signal1 = arrays[0]
                signal2 = arrays[1]
        
        else:
            raise ValueError(f"Unsupported data type for coherence analysis: {data_type}")
        
        # Set default segment length if not provided
        if nperseg is None:
            nperseg = min(256, min(len(signal1), len(signal2)) // 4)
        
        # Compute coherence
        f, Cxy = coherence(signal1, signal2, fs=sample_rate, window=window, nperseg=nperseg)
        
        # Find frequencies with high coherence
        high_coherence_idx = np.where(Cxy > 0.8)[0]
        high_coherence_freqs = f[high_coherence_idx]
        high_coherence_values = Cxy[high_coherence_idx]
        
        return {
            'frequencies': f,
            'coherence': Cxy,
            'high_coherence_frequencies': high_coherence_freqs.tolist(),
            'high_coherence_values': high_coherence_values.tolist(),
            'max_coherence': np.max(Cxy),
            'max_coherence_frequency': f[np.argmax(Cxy)],
            'window': window,
            'nperseg': nperseg,
            'sample_rate': sample_rate
        }
    
    def cross_spectral_analysis(self, dataset_name, column1=None, column2=None, window='hann', nperseg=None):
        """Perform cross-spectral density analysis between two signals in a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data (same as coherence analysis)
        if data_type == 'array':
            if data.ndim < 2:
                raise ValueError("Cross-spectral analysis requires at least 2D data")
            
            if column1 is not None and column2 is not None:
                signal1 = data[:, column1]
                signal2 = data[:, column2]
            else:
                # Use first two columns
                signal1 = data[:, 0]
                signal2 = data[:, 1]
        
        elif data_type == 'dataframe':
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            
            if column1 is not None and column2 is not None:
                if column1 in numeric_columns and column2 in numeric_columns:
                    signal1 = data[column1].values
                    signal2 = data[column2].values
                else:
                    raise ValueError("One or both specified columns are not numeric")
            else:
                if len(numeric_columns) >= 2:
                    signal1 = data[numeric_columns[0]].values
                    signal2 = data[numeric_columns[1]].values
                else:
                    raise ValueError("DataFrame must have at least two numeric columns")
        
        elif data_type == 'dict':
            keys = list(data.keys())
            arrays = []
            
            for key in keys:
                if isinstance(data[key], (list, np.ndarray)):
                    arrays.append(np.array(data[key]))
            
            if len(arrays) < 2:
                raise ValueError("Dictionary must contain at least two arrays")
            
            if column1 is not None and column2 is not None:
                if column1 in keys and column2 in keys:
                    signal1 = np.array(data[column1])
                    signal2 = np.array(data[column2])
                else:
                    raise ValueError("One or both specified keys not found in dictionary")
            else:
                signal1 = arrays[0]
                signal2 = arrays[1]
        
        else:
            raise ValueError(f"Unsupported data type for cross-spectral analysis: {data_type}")
        
        # Set default segment length if not provided
        if nperseg is None:
            nperseg = min(256, min(len(signal1), len(signal2)) // 4)
        
        # Compute cross-spectral density
        f, Pxy = csd(signal1, signal2, fs=sample_rate, window=window, nperseg=nperseg)
        
        # Calculate magnitude and phase
        magnitude = np.abs(Pxy)
        phase = np.angle(Pxy, deg=True)
        
        # Convert magnitude to dB scale
        magnitude_db = 10 * np.log10(magnitude + 1e-10)
        
        return {
            'frequencies': f,
            'cross_spectrum': Pxy,
            'magnitude': magnitude,
            'magnitude_db': magnitude_db,
            'phase': phase,
            'window': window,
            'nperseg': nperseg,
            'sample_rate': sample_rate
        }
    
    def hilbert_transform_analysis(self, dataset_name, column=None):
        """Perform Hilbert transform analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        sample_rate = dataset.get('sample_rate', 1.0)
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    signal_data = data[:, column]
                else:
                    signal_data = data.flatten()
            else:
                signal_data = data
        elif data_type == 'dataframe':
            if column is not None:
                signal_data = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    signal_data = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                signal_data = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        signal_data = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for Hilbert transform analysis: {data_type}")
        
        # Compute Hilbert transform
        analytic_signal = hilbert(signal_data)
        
        # Extract amplitude and phase
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        
        # Calculate instantaneous frequency
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi) * sample_rate
        
        # Time vector
        t = np.arange(len(signal_data)) / sample_rate
        
        return {
            'time': t,
            'original_signal': signal_data,
            'analytic_signal': analytic_signal,
            'amplitude_envelope': amplitude_envelope,
            'instantaneous_phase': instantaneous_phase,
            'instantaneous_frequency': instantaneous_frequency,
            'sample_rate': sample_rate
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