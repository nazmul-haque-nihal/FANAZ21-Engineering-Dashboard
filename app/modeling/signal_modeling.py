# app/modeling/signal_modeling.py
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from app.config import Config

class SignalModeler:
    def __init__(self, config):
        self.config = config
        self.models = {}  # Dictionary to store signal models
    
    def fit_sinusoidal_model(self, t, y, model_type='single_sine'):
        """Fit a sinusoidal model to data"""
        def single_sine_func(t, A, f, phi, offset):
            return A * np.sin(2 * np.pi * f * t + phi) + offset
        
        def multi_sine_func(t, *params):
            result = np.zeros_like(t)
            for i in range(0, len(params), 3):
                A = params[i]
                f = params[i+1]
                phi = params[i+2]
                result += A * np.sin(2 * np.pi * f * t + phi)
            return result
        
        # Initial guess
        if model_type == 'single_sine':
            # Estimate amplitude
            A_guess = (np.max(y) - np.min(y)) / 2
            # Estimate frequency using FFT
            yf = np.fft.fft(y)
            xf = np.fft.fftfreq(len(y), t[1] - t[0])
            f_guess = xf[np.argmax(np.abs(yf[1:len(yf)//2])) + 1]
            # Estimate phase and offset
            phi_guess = 0
            offset_guess = np.mean(y)
            
            initial_guess = [A_guess, f_guess, phi_guess, offset_guess]
            bounds = ([0, 0, -np.pi, -np.inf], [np.inf, np.inf, np.pi, np.inf])
            
            try:
                popt, pcov = curve_fit(single_sine_func, t, y, p0=initial_guess, bounds=bounds)
                A, f, phi, offset = popt
                
                # Calculate fitted values
                y_fit = single_sine_func(t, A, f, phi, offset)
                
                # Calculate R-squared
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store the model
                model_id = f"single_sine_{len(self.models)}"
                self.models[model_id] = {
                    'type': 'single_sine',
                    'parameters': {'A': A, 'f': f, 'phi': phi, 'offset': offset},
                    'covariance': pcov.tolist(),
                    'r_squared': r_squared,
                    't': t,
                    'y_original': y,
                    'y_fit': y_fit
                }
                
                return model_id, {'A': A, 'f': f, 'phi': phi, 'offset': offset}, r_squared
            
            except Exception as e:
                raise ValueError(f"Failed to fit single sine model: {str(e)}")
        
        elif model_type == 'multi_sine':
            # Estimate number of components using FFT
            yf = np.abs(np.fft.fft(y)[1:len(y)//2])
            peaks, _ = signal.find_peaks(yf, height=np.max(yf) * 0.1)
            n_components = min(5, len(peaks))  # Limit to 5 components
            
            # Initial guess for each component
            initial_guess = []
            bounds_lower = []
            bounds_upper = []
            
            xf = np.fft.fftfreq(len(y), t[1] - t[0])
            for i in range(n_components):
                # Get the i-th largest peak
                peak_idx = peaks[np.argsort(yf[peaks])[-(i+1)]]
                f_guess = xf[peak_idx + 1]  # +1 because we skipped DC component
                A_guess = yf[peak_idx] * 2 / len(y)  # Scale factor
                phi_guess = 0
                
                initial_guess.extend([A_guess, f_guess, phi_guess])
                bounds_lower.extend([0, 0, -np.pi])
                bounds_upper.extend([np.inf, np.inf, np.pi])
            
            try:
                popt, pcov = curve_fit(multi_sine_func, t, y, p0=initial_guess, 
                                      bounds=(bounds_lower, bounds_upper))
                
                # Calculate fitted values
                y_fit = multi_sine_func(t, *popt)
                
                # Calculate R-squared
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Format parameters
                params = {}
                for i in range(n_components):
                    params[f'A_{i+1}'] = popt[i*3]
                    params[f'f_{i+1}'] = popt[i*3+1]
                    params[f'phi_{i+1}'] = popt[i*3+2]
                
                # Store the model
                model_id = f"multi_sine_{len(self.models)}"
                self.models[model_id] = {
                    'type': 'multi_sine',
                    'n_components': n_components,
                    'parameters': params,
                    'covariance': pcov.tolist(),
                    'r_squared': r_squared,
                    't': t,
                    'y_original': y,
                    'y_fit': y_fit
                }
                
                return model_id, params, r_squared
            
            except Exception as e:
                raise ValueError(f"Failed to fit multi-sine model: {str(e)}")
        
        else:
            raise ValueError(f"Unknown sinusoidal model type: {model_type}")
    
    def fit_exponential_model(self, t, y, model_type='single_exp'):
        """Fit an exponential model to data"""
        def single_exp_func(t, A, tau, offset):
            return A * np.exp(-t / tau) + offset
        
        def double_exp_func(t, A1, tau1, A2, tau2, offset):
            return A1 * np.exp(-t / tau1) + A2 * np.exp(-t / tau2) + offset
        
        # Initial guess
        if model_type == 'single_exp':
            # Estimate parameters
            A_guess = y[0] - y[-1]  # Initial value minus final value
            offset_guess = y[-1]  # Final value as offset
            # Estimate time constant (time to reach 1/e of initial value)
            e_fold_value = A_guess / np.e + offset_guess
            e_fold_idx = np.argmin(np.abs(y - e_fold_value))
            tau_guess = t[e_fold_idx]
            
            initial_guess = [A_guess, tau_guess, offset_guess]
            bounds = ([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf])
            
            try:
                popt, pcov = curve_fit(single_exp_func, t, y, p0=initial_guess, bounds=bounds)
                A, tau, offset = popt
                
                # Calculate fitted values
                y_fit = single_exp_func(t, A, tau, offset)
                
                # Calculate R-squared
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store the model
                model_id = f"single_exp_{len(self.models)}"
                self.models[model_id] = {
                    'type': 'single_exp',
                    'parameters': {'A': A, 'tau': tau, 'offset': offset},
                    'covariance': pcov.tolist(),
                    'r_squared': r_squared,
                    't': t,
                    'y_original': y,
                    'y_fit': y_fit
                }
                
                return model_id, {'A': A, 'tau': tau, 'offset': offset}, r_squared
            
            except Exception as e:
                raise ValueError(f"Failed to fit single exponential model: {str(e)}")
        
        elif model_type == 'double_exp':
            # For double exponential, we'll use a more complex approach
            # First fit a single exponential
            A_guess = y[0] - y[-1]
            offset_guess = y[-1]
            e_fold_value = A_guess / np.e + offset_guess
            e_fold_idx = np.argmin(np.abs(y - e_fold_value))
            tau_guess = t[e_fold_idx]
            
            # Second component parameters
            # Assume second component is faster and smaller
            A2_guess = A_guess * 0.2
            tau2_guess = tau_guess * 0.2
            
            initial_guess = [A_guess, tau_guess, A2_guess, tau2_guess, offset_guess]
            bounds = ([-np.inf, 0, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
            
            try:
                popt, pcov = curve_fit(double_exp_func, t, y, p0=initial_guess, bounds=bounds)
                A1, tau1, A2, tau2, offset = popt
                
                # Calculate fitted values
                y_fit = double_exp_func(t, A1, tau1, A2, tau2, offset)
                
                # Calculate R-squared
                ss_res = np.sum((y - y_fit) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store the model
                model_id = f"double_exp_{len(self.models)}"
                self.models[model_id] = {
                    'type': 'double_exp',
                    'parameters': {'A1': A1, 'tau1': tau1, 'A2': A2, 'tau2': tau2, 'offset': offset},
                    'covariance': pcov.tolist(),
                    'r_squared': r_squared,
                    't': t,
                    'y_original': y,
                    'y_fit': y_fit
                }
                
                return model_id, {'A1': A1, 'tau1': tau1, 'A2': A2, 'tau2': tau2, 'offset': offset}, r_squared
            
            except Exception as e:
                raise ValueError(f"Failed to fit double exponential model: {str(e)}")
        
        else:
            raise ValueError(f"Unknown exponential model type: {model_type}")
    
    def fit_polynomial_model(self, t, y, degree=2):
        """Fit a polynomial model to data"""
        try:
            # Fit polynomial
            coeffs = np.polyfit(t, y, degree)
            
            # Calculate fitted values
            poly_func = np.poly1d(coeffs)
            y_fit = poly_func(t)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Format parameters
            params = {}
            for i, coef in enumerate(coeffs):
                params[f'a_{degree-i}'] = coef
            
            # Store the model
            model_id = f"poly_{degree}_{len(self.models)}"
            self.models[model_id] = {
                'type': 'polynomial',
                'degree': degree,
                'parameters': params,
                'coefficients': coeffs.tolist(),
                'r_squared': r_squared,
                't': t,
                'y_original': y,
                'y_fit': y_fit
            }
            
            return model_id, params, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to fit polynomial model: {str(e)}")
    
    def fit_fourier_model(self, t, y, n_harmonics=3):
        """Fit a Fourier series model to data"""
        def fourier_func(t, *params):
            result = params[0]  # DC component
            for i in range(n_harmonics):
                A = params[1 + i*2]
                B = params[2 + i*2]
                result += A * np.cos(2 * np.pi * (i+1) * t / (t[-1] - t[0])) + \
                         B * np.sin(2 * np.pi * (i+1) * t / (t[-1] - t[0]))
            return result
        
        # Initial guess
        T = t[-1] - t[0]  # Time period
        initial_guess = [np.mean(y)]  # DC component
        
        # Add initial guesses for harmonics
        for i in range(n_harmonics):
            # Use FFT to estimate coefficients
            freq = (i+1) / T
            idx = int(freq * len(t))
            if idx < len(y) // 2:
                A_guess = 2 * np.mean(y * np.cos(2 * np.pi * freq * t))
                B_guess = 2 * np.mean(y * np.sin(2 * np.pi * freq * t))
            else:
                A_guess = 0
                B_guess = 0
            
            initial_guess.extend([A_guess, B_guess])
        
        try:
            popt, pcov = curve_fit(fourier_func, t, y, p0=initial_guess)
            
            # Calculate fitted values
            y_fit = fourier_func(t, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Format parameters
            params = {'DC': popt[0]}
            for i in range(n_harmonics):
                params[f'A_{i+1}'] = popt[1 + i*2]
                params[f'B_{i+1}'] = popt[2 + i*2]
            
            # Store the model
            model_id = f"fourier_{n_harmonics}_{len(self.models)}"
            self.models[model_id] = {
                'type': 'fourier',
                'n_harmonics': n_harmonics,
                'parameters': params,
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'y_original': y,
                'y_fit': y_fit
            }
            
            return model_id, params, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to fit Fourier model: {str(e)}")
    
    def fit_custom_model(self, t, y, model_func, initial_guess=None, bounds=None):
        """Fit a custom model to data"""
        try:
            if initial_guess is None:
                raise ValueError("Initial guess must be provided for custom model")
            
            popt, pcov = curve_fit(model_func, t, y, p0=initial_guess, bounds=bounds)
            
            # Calculate fitted values
            y_fit = model_func(t, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Format parameters
            params = {f'param_{i}': val for i, val in enumerate(popt)}
            
            # Store the model
            model_id = f"custom_{len(self.models)}"
            self.models[model_id] = {
                'type': 'custom',
                'parameters': params,
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'y_original': y,
                'y_fit': y_fit
            }
            
            return model_id, params, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to fit custom model: {str(e)}")
    
    def predict(self, model_id, t_new):
        """Use a fitted model to make predictions"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        model_type = model['type']
        params = model['parameters']
        
        if model_type == 'single_sine':
            A = params['A']
            f = params['f']
            phi = params['phi']
            offset = params['offset']
            y_pred = A * np.sin(2 * np.pi * f * t_new + phi) + offset
        
        elif model_type == 'multi_sine':
            y_pred = np.zeros_like(t_new)
            n_components = model['n_components']
            for i in range(n_components):
                A = params[f'A_{i+1}']
                f = params[f'f_{i+1}']
                phi = params[f'phi_{i+1}']
                y_pred += A * np.sin(2 * np.pi * f * t_new + phi)
        
        elif model_type == 'single_exp':
            A = params['A']
            tau = params['tau']
            offset = params['offset']
            y_pred = A * np.exp(-t_new / tau) + offset
        
        elif model_type == 'double_exp':
            A1 = params['A1']
            tau1 = params['tau1']
            A2 = params['A2']
            tau2 = params['tau2']
            offset = params['offset']
            y_pred = A1 * np.exp(-t_new / tau1) + A2 * np.exp(-t_new / tau2) + offset
        
        elif model_type == 'polynomial':
            degree = model['degree']
            coeffs = model['coefficients']
            poly_func = np.poly1d(coeffs)
            y_pred = poly_func(t_new)
        
        elif model_type == 'fourier':
            n_harmonics = model['n_harmonics']
            T = model['t'][-1] - model['t'][0]  # Time period from original data
            y_pred = np.full_like(t_new, params['DC'])  # DC component
            for i in range(n_harmonics):
                A = params[f'A_{i+1}']
                B = params[f'B_{i+1}']
                y_pred += A * np.cos(2 * np.pi * (i+1) * t_new / T) + \
                          B * np.sin(2 * np.pi * (i+1) * t_new / T)
        
        elif model_type == 'custom':
            # For custom models, we need the original function
            # This is a limitation of our current approach
            raise ValueError("Prediction not supported for custom models in this implementation")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return y_pred
    
    def get_model(self, model_id):
        """Get a model by ID"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        return self.models[model_id]
    
    def list_models(self):
        """List all available models"""
        return list(self.models.keys())
    
    def remove_model(self, model_id):
        """Remove a model by ID"""
        if model_id in self.models:
            del self.models[model_id]
            return True
        return False