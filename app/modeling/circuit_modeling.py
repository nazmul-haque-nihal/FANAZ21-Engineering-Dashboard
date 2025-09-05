# app/modeling/circuit_modeling.py
import numpy as np
from scipy import signal, optimize
import plotly.graph_objs as go
from app.config import Config

class CircuitModeler:
    def __init__(self, config):
        self.config = config
        self.models = {}  # Dictionary to store circuit models
    
    def model_rc_circuit(self, t, v_in, v_out):
        """Model an RC circuit from input-output data"""
        def rc_response(t, tau, v_final):
            return v_final * (1 - np.exp(-t / tau))
        
        # Initial guess
        v_final_guess = v_out[-1]  # Final voltage
        # Estimate time constant (time to reach 63% of final value)
        v_63_percent = 0.63 * v_final_guess
        tau_guess = t[np.argmin(np.abs(v_out - v_63_percent))]
        
        initial_guess = [tau_guess, v_final_guess]
        bounds = ([0, 0], [np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(rc_response, t, v_out, p0=initial_guess, bounds=bounds)
            tau, v_final = popt
            
            # Calculate fitted values
            v_fit = rc_response(t, tau, v_final)
            
            # Calculate R-squared
            ss_res = np.sum((v_out - v_fit) ** 2)
            ss_tot = np.sum((v_out - np.mean(v_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate resistance (assuming C = 1μF)
            C = 1e-6  # 1μF
            R = tau / C
            
            # Store the model
            model_id = f"RC_{len(self.models)}"
            self.models[model_id] = {
                'type': 'RC',
                'parameters': {'tau': tau, 'v_final': v_final, 'R': R, 'C': C},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'v_in': v_in,
                'v_out': v_out,
                'v_fit': v_fit
            }
            
            return model_id, {'tau': tau, 'v_final': v_final, 'R': R, 'C': C}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model RC circuit: {str(e)}")
    
    def model_rl_circuit(self, t, i_in, i_out):
        """Model an RL circuit from input-output data"""
        def rl_response(t, tau, i_final):
            return i_final * (1 - np.exp(-t / tau))
        
        # Initial guess
        i_final_guess = i_out[-1]  # Final current
        # Estimate time constant (time to reach 63% of final value)
        i_63_percent = 0.63 * i_final_guess
        tau_guess = t[np.argmin(np.abs(i_out - i_63_percent))]
        
        initial_guess = [tau_guess, i_final_guess]
        bounds = ([0, 0], [np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(rl_response, t, i_out, p0=initial_guess, bounds=bounds)
            tau, i_final = popt
            
            # Calculate fitted values
            i_fit = rl_response(t, tau, i_final)
            
            # Calculate R-squared
            ss_res = np.sum((i_out - i_fit) ** 2)
            ss_tot = np.sum((i_out - np.mean(i_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate inductance (assuming R = 100Ω)
            R = 100  # 100Ω
            L = tau * R
            
            # Store the model
            model_id = f"RL_{len(self.models)}"
            self.models[model_id] = {
                'type': 'RL',
                'parameters': {'tau': tau, 'i_final': i_final, 'R': R, 'L': L},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'i_in': i_in,
                'i_out': i_out,
                'i_fit': i_fit
            }
            
            return model_id, {'tau': tau, 'i_final': i_final, 'R': R, 'L': L}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model RL circuit: {str(e)}")
    
    def model_rlc_circuit(self, t, v_in, v_out):
        """Model an RLC circuit from input-output data"""
        def rlc_response(t, R, L, C):
            # Solve RLC circuit differential equation
            omega_0 = 1 / np.sqrt(L * C)
            zeta = R / (2 * np.sqrt(L / C))
            
            if zeta < 1:  # Underdamped
                omega_d = omega_0 * np.sqrt(1 - zeta**2)
                return 1 - np.exp(-zeta * omega_0 * t) * (
                    np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t)
                )
            elif zeta == 1:  # Critically damped
                return 1 - np.exp(-omega_0 * t) * (1 + omega_0 * t)
            else:  # Overdamped
                s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
                s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
                return 1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)
        
        # Scale output to match input (assuming step input)
        v_scale = np.max(v_out)
        v_out_scaled = v_out / v_scale
        
        # Initial guess
        R_guess = 100  # 100Ω
        L_guess = 0.1  # 0.1H
        C_guess = 1e-6  # 1μF
        
        initial_guess = [R_guess, L_guess, C_guess]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(rlc_response, t, v_out_scaled, p0=initial_guess, bounds=bounds)
            R, L, C = popt
            
            # Calculate fitted values
            v_fit_scaled = rlc_response(t, R, L, C)
            v_fit = v_fit_scaled * v_scale
            
            # Calculate R-squared
            ss_res = np.sum((v_out - v_fit) ** 2)
            ss_tot = np.sum((v_out - np.mean(v_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate circuit parameters
            omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
            zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
            
            # Store the model
            model_id = f"RLC_{len(self.models)}"
            self.models[model_id] = {
                'type': 'RLC',
                'parameters': {'R': R, 'L': L, 'C': C, 'omega_0': omega_0, 'zeta': zeta},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'v_in': v_in,
                'v_out': v_out,
                'v_fit': v_fit
            }
            
            return model_id, {'R': R, 'L': L, 'C': C, 'omega_0': omega_0, 'zeta': zeta}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model RLC circuit: {str(e)}")
    
    def model_diode_circuit(self, v_in, i_out):
        """Model a diode circuit from input-output data"""
        def diode_model(v, Is, n, Vt):
            # Shockley diode equation
            return Is * (np.exp(v / (n * Vt)) - 1)
        
        # Initial guess
        Is_guess = 1e-12  # Reverse saturation current
        n_guess = 1.0  # Ideality factor
        Vt_guess = 0.026  # Thermal voltage at room temperature
        
        initial_guess = [Is_guess, n_guess, Vt_guess]
        bounds = ([0, 1, 0.01], [1e-6, 2, 0.05])
        
        try:
            popt, pcov = optimize.curve_fit(diode_model, v_in, i_out, p0=initial_guess, bounds=bounds)
            Is, n, Vt = popt
            
            # Calculate fitted values
            i_fit = diode_model(v_in, Is, n, Vt)
            
            # Calculate R-squared
            ss_res = np.sum((i_out - i_fit) ** 2)
            ss_tot = np.sum((i_out - np.mean(i_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store the model
            model_id = f"Diode_{len(self.models)}"
            self.models[model_id] = {
                'type': 'Diode',
                'parameters': {'Is': Is, 'n': n, 'Vt': Vt},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'v_in': v_in,
                'i_out': i_out,
                'i_fit': i_fit
            }
            
            return model_id, {'Is': Is, 'n': n, 'Vt': Vt}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model diode circuit: {str(e)}")
    
    def model_bjt_amplifier(self, v_in, v_out):
        """Model a BJT amplifier from input-output data"""
        def bjt_model(v, A, v_offset):
            # Simple linear model for small signal
            return A * v + v_offset
        
        # Initial guess
        A_guess = (np.max(v_out) - np.min(v_out)) / (np.max(v_in) - np.min(v_in))
        v_offset_guess = np.mean(v_out) - A_guess * np.mean(v_in)
        
        initial_guess = [A_guess, v_offset_guess]
        bounds = ([-np.inf, -np.inf], [np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(bjt_model, v_in, v_out, p0=initial_guess, bounds=bounds)
            A, v_offset = popt
            
            # Calculate fitted values
            v_fit = bjt_model(v_in, A, v_offset)
            
            # Calculate R-squared
            ss_res = np.sum((v_out - v_fit) ** 2)
            ss_tot = np.sum((v_out - np.mean(v_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store the model
            model_id = f"BJT_{len(self.models)}"
            self.models[model_id] = {
                'type': 'BJT',
                'parameters': {'A': A, 'v_offset': v_offset},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'v_in': v_in,
                'v_out': v_out,
                'v_fit': v_fit
            }
            
            return model_id, {'A': A, 'v_offset': v_offset}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model BJT amplifier: {str(e)}")
    
    def model_opamp_circuit(self, v_in, v_out, circuit_type='inverting'):
        """Model an op-amp circuit from input-output data"""
        def inverting_model(v, Rf, Rin):
            # Inverting amplifier: Vout = - (Rf/Rin) * Vin
            return - (Rf / Rin) * v
        
        def non_inverting_model(v, R1, R2):
            # Non-inverting amplifier: Vout = (1 + R2/R1) * Vin
            return (1 + R2 / R1) * v
        
        if circuit_type == 'inverting':
            model_func = inverting_model
            # Initial guess
            gain_guess = -np.mean(v_out) / np.mean(v_in) if np.mean(v_in) != 0 else -1
            Rf_guess = 10000  # 10kΩ
            Rin_guess = Rf_guess / abs(gain_guess) if gain_guess != 0 else 10000  # 10kΩ
            
            initial_guess = [Rf_guess, Rin_guess]
            bounds = ([0, 0], [np.inf, np.inf])
            
        elif circuit_type == 'non_inverting':
            model_func = non_inverting_model
            # Initial guess
            gain_guess = np.mean(v_out) / np.mean(v_in) if np.mean(v_in) != 0 else 1
            R1_guess = 10000  # 10kΩ
            R2_guess = R1_guess * (gain_guess - 1) if gain_guess > 1 else 10000  # 10kΩ
            
            initial_guess = [R1_guess, R2_guess]
            bounds = ([0, 0], [np.inf, np.inf])
            
        else:
            raise ValueError(f"Unknown op-amp circuit type: {circuit_type}")
        
        try:
            popt, pcov = optimize.curve_fit(model_func, v_in, v_out, p0=initial_guess, bounds=bounds)
            
            # Calculate fitted values
            v_fit = model_func(v_in, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((v_out - v_fit) ** 2)
            ss_tot = np.sum((v_out - np.mean(v_out)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Format parameters
            if circuit_type == 'inverting':
                Rf, Rin = popt
                params = {'Rf': Rf, 'Rin': Rin, 'gain': -Rf/Rin}
            else:  # non_inverting
                R1, R2 = popt
                params = {'R1': R1, 'R2': R2, 'gain': 1 + R2/R1}
            
            # Store the model
            model_id = f"OpAmp_{circuit_type}_{len(self.models)}"
            self.models[model_id] = {
                'type': f'OpAmp_{circuit_type}',
                'parameters': params,
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'v_in': v_in,
                'v_out': v_out,
                'v_fit': v_fit
            }
            
            return model_id, params, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model op-amp circuit: {str(e)}")
    
    def predict(self, model_id, input_data):
        """Use a fitted circuit model to make predictions"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        model_type = model['type']
        params = model['parameters']
        
        if model_type == 'RC':
            tau = params['tau']
            v_final = params['v_final']
            output = v_final * (1 - np.exp(-input_data / tau))
        
        elif model_type == 'RL':
            tau = params['tau']
            i_final = params['i_final']
            output = i_final * (1 - np.exp(-input_data / tau))
        
        elif model_type == 'RLC':
            R = params['R']
            L = params['L']
            C = params['C']
            
            # Solve RLC circuit differential equation
            omega_0 = 1 / np.sqrt(L * C)
            zeta = R / (2 * np.sqrt(L / C))
            
            if zeta < 1:  # Underdamped
                omega_d = omega_0 * np.sqrt(1 - zeta**2)
                output = 1 - np.exp(-zeta * omega_0 * input_data) * (
                    np.cos(omega_d * input_data) + (zeta * omega_0 / omega_d) * np.sin(omega_d * input_data)
                )
            elif zeta == 1:  # Critically damped
                output = 1 - np.exp(-omega_0 * input_data) * (1 + omega_0 * input_data)
            else:  # Overdamped
                s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
                s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
                output = 1 - (s2 * np.exp(s1 * input_data) - s1 * np.exp(s2 * input_data)) / (s2 - s1)
        
        elif model_type == 'Diode':
            Is = params['Is']
            n = params['n']
            Vt = params['Vt']
            output = Is * (np.exp(input_data / (n * Vt)) - 1)
        
        elif model_type == 'BJT':
            A = params['A']
            v_offset = params['v_offset']
            output = A * input_data + v_offset
        
        elif model_type == 'OpAmp_inverting':
            Rf = params['Rf']
            Rin = params['Rin']
            output = - (Rf / Rin) * input_data
        
        elif model_type == 'OpAmp_non_inverting':
            R1 = params['R1']
            R2 = params['R2']
            output = (1 + R2 / R1) * input_data
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return output
    
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