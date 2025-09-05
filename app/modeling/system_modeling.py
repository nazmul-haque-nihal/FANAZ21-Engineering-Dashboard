# app/modeling/system_modeling.py
import numpy as np
from scipy import signal, optimize
import plotly.graph_objs as go
from app.config import Config

class SystemModeler:
    def __init__(self, config):
        self.config = config
        self.models = {}  # Dictionary to store system models
    
    def model_first_order_system(self, t, u, y):
        """Model a first-order system from input-output data"""
        def first_order_response(t, K, tau):
            # For step input
            return K * (1 - np.exp(-t / tau))
        
        # Check if input is a step function
        if not np.all(u == u[0]):
            raise ValueError("First-order system modeling currently only supports step input")
        
        # Scale output to match input
        y_scale = np.max(y)
        y_scaled = y / y_scale
        
        # Initial guess
        K_guess = 1.0  # Gain
        # Estimate time constant (time to reach 63% of final value)
        y_63_percent = 0.63 * K_guess
        tau_guess = t[np.argmin(np.abs(y_scaled - y_63_percent))]
        
        initial_guess = [K_guess, tau_guess]
        bounds = ([0, 0], [np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(first_order_response, t, y_scaled, p0=initial_guess, bounds=bounds)
            K, tau = popt
            
            # Calculate fitted values
            y_fit_scaled = first_order_response(t, K, tau)
            y_fit = y_fit_scaled * y_scale
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate system parameters
            time_constant = tau
            settling_time = 4 * tau  # Settling time (2% criterion)
            rise_time = 2.2 * tau  # Rise time (10% to 90%)
            
            # Store the model
            model_id = f"FirstOrder_{len(self.models)}"
            self.models[model_id] = {
                'type': 'FirstOrder',
                'parameters': {'K': K * y_scale, 'tau': tau},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'system_params': {
                    'time_constant': time_constant,
                    'settling_time': settling_time,
                    'rise_time': rise_time
                },
                't': t,
                'u': u,
                'y': y,
                'y_fit': y_fit
            }
            
            return model_id, {'K': K * y_scale, 'tau': tau}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model first-order system: {str(e)}")
    
    def model_second_order_system(self, t, u, y):
        """Model a second-order system from input-output data"""
        def second_order_response(t, K, zeta, omega_n):
            # For step input
            if zeta < 1:  # Underdamped
                omega_d = omega_n * np.sqrt(1 - zeta**2)
                return K * (1 - np.exp(-zeta * omega_n * t) * (
                    np.cos(omega_d * t) + (zeta * omega_n / omega_d) * np.sin(omega_d * t)
                ))
            elif zeta == 1:  # Critically damped
                return K * (1 - np.exp(-omega_n * t) * (1 + omega_n * t))
            else:  # Overdamped
                s1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
                s2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
                return K * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))
        
        # Check if input is a step function
        if not np.all(u == u[0]):
            raise ValueError("Second-order system modeling currently only supports step input")
        
        # Scale output to match input
        y_scale = np.max(y)
        y_scaled = y / y_scale
        
        # Initial guess
        K_guess = 1.0  # Gain
        zeta_guess = 0.5  # Damping ratio
        # Estimate natural frequency from oscillation period if underdamped
        if np.max(y_scaled) > 1.05 * K_guess:  # Overshoot indicates underdamped
            # Find peaks
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(y_scaled)
            if len(peaks) > 1:
                period = t[peaks[1]] - t[peaks[0]]
                omega_d_guess = 2 * np.pi / period
                omega_n_guess = omega_d_guess / np.sqrt(1 - zeta_guess**2)
            else:
                omega_n_guess = 10.0  # Default guess
        else:
            omega_n_guess = 10.0  # Default guess
        
        initial_guess = [K_guess, zeta_guess, omega_n_guess]
        bounds = ([0, 0, 0], [np.inf, np.inf, np.inf])
        
        try:
            popt, pcov = optimize.curve_fit(second_order_response, t, y_scaled, p0=initial_guess, bounds=bounds)
            K, zeta, omega_n = popt
            
            # Calculate fitted values
            y_fit_scaled = second_order_response(t, K, zeta, omega_n)
            y_fit = y_fit_scaled * y_scale
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate system parameters
            if zeta < 1:  # Underdamped
                omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped natural frequency
                T_p = np.pi / omega_d  # Peak time
                M_p = np.exp(-np.pi * zeta / np.sqrt(1 - zeta**2)) * 100  # Percent overshoot
                T_s = 4 / (zeta * omega_n)  # Settling time (2% criterion)
            elif zeta == 1:  # Critically damped
                omega_d = 0
                T_p = np.inf
                M_p = 0
                T_s = 4 / omega_n  # Approximate settling time
            else:  # Overdamped
                omega_d = 0
                T_p = np.inf
                M_p = 0
                T_s = 4 / (zeta * omega_n)  # Approximate settling time
            
            # Store the model
            model_id = f"SecondOrder_{len(self.models)}"
            self.models[model_id] = {
                'type': 'SecondOrder',
                'parameters': {'K': K * y_scale, 'zeta': zeta, 'omega_n': omega_n},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'system_params': {
                    'omega_d': omega_d,
                    'T_p': T_p,
                    'M_p': M_p,
                    'T_s': T_s
                },
                't': t,
                'u': u,
                'y': y,
                'y_fit': y_fit
            }
            
            return model_id, {'K': K * y_scale, 'zeta': zeta, 'omega_n': omega_n}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model second-order system: {str(e)}")
    
    def model_pid_controller(self, t, r, u, y):
        """Model a PID controller from input-output data"""
        def pid_system(t, Kp, Ki, Kd):
            # Simulate PID controller with a simple first-order plant
            # This is a simplified approach for demonstration
            dt = t[1] - t[0]
            
            # Initialize arrays
            e = np.zeros_like(t)
            integral = 0
            prev_error = 0
            y_sim = np.zeros_like(t)
            
            # Simulate closed-loop system
            for i in range(1, len(t)):
                # Calculate error
                e[i] = r[i] - y_sim[i-1]
                
                # PID controller
                proportional = Kp * e[i]
                integral += Ki * e[i] * dt
                derivative = Kd * (e[i] - prev_error) / dt
                
                # Control signal
                u_sim = proportional + integral + derivative
                
                # Simple first-order plant (tau = 1)
                tau = 1.0
                y_sim[i] = y_sim[i-1] + (u_sim - y_sim[i-1]) * dt / tau
                
                # Update previous error
                prev_error = e[i]
            
            return y_sim
        
        # Initial guess
        Kp_guess = 1.0
        Ki_guess = 0.1
        Kd_guess = 0.01
        
        initial_guess = [Kp_guess, Ki_guess, Kd_guess]
        bounds = ([0, 0, 0], [100, 10, 1])
        
        try:
            popt, pcov = optimize.curve_fit(pid_system, t, y, p0=initial_guess, bounds=bounds)
            Kp, Ki, Kd = popt
            
            # Calculate fitted values
            y_fit = pid_system(t, Kp, Ki, Kd)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Store the model
            model_id = f"PID_{len(self.models)}"
            self.models[model_id] = {
                'type': 'PID',
                'parameters': {'Kp': Kp, 'Ki': Ki, 'Kd': Kd},
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                't': t,
                'r': r,
                'u': u,
                'y': y,
                'y_fit': y_fit
            }
            
            return model_id, {'Kp': Kp, 'Ki': Ki, 'Kd': Kd}, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model PID controller: {str(e)}")
    
    def model_state_space_system(self, t, u, y, n_states=2):
        """Model a state-space system from input-output data"""
        def state_space_response(t, *params):
            # Extract parameters
            A = np.array(params[:n_states*n_states]).reshape(n_states, n_states)
            B = np.array(params[n_states*n_states:n_states*n_states+n_states]).reshape(n_states, 1)
            C = np.array(params[n_states*n_states+n_states:n_states*n_states+n_states+1]).reshape(1, n_states)
            D = np.array(params[n_states*n_states+n_states+1])
            
            # Simulate state-space system
            dt = t[1] - t[0]
            x = np.zeros((len(t), n_states))
            y_sim = np.zeros_like(t)
            
            for i in range(1, len(t)):
                # State equation: x_dot = A*x + B*u
                x_dot = A @ x[i-1] + B * u[i-1]
                # Euler integration
                x[i] = x[i-1] + x_dot * dt
                # Output equation: y = C*x + D*u
                y_sim[i] = C @ x[i] + D * u[i]
            
            return y_sim
        
        # Initial guess
        # Simple second-order system guess
        A_guess = np.array([[-0.1, 0], [0, -0.2]])
        B_guess = np.array([[0.1], [0.2]])
        C_guess = np.array([[1, 0]])
        D_guess = np.array([0])
        
        initial_guess = np.concatenate([A_guess.flatten(), B_guess.flatten(), C_guess.flatten(), D_guess])
        
        # No bounds for simplicity
        bounds = None
        
        try:
            popt, pcov = optimize.curve_fit(state_space_response, t, y, p0=initial_guess, bounds=bounds)
            
            # Extract parameters
            A = np.array(popt[:n_states*n_states]).reshape(n_states, n_states)
            B = np.array(popt[n_states*n_states:n_states*n_states+n_states]).reshape(n_states, 1)
            C = np.array(popt[n_states*n_states+n_states:n_states*n_states+n_states+1]).reshape(1, n_states)
            D = np.array(popt[n_states*n_states+n_states+1])
            
            # Calculate fitted values
            y_fit = state_space_response(t, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate system eigenvalues
            eigenvalues = np.linalg.eigvals(A)
            
            # Store the model
            model_id = f"StateSpace_{len(self.models)}"
            self.models[model_id] = {
                'type': 'StateSpace',
                'parameters': {
                    'A': A.tolist(),
                    'B': B.tolist(),
                    'C': C.tolist(),
                    'D': D.tolist()
                },
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'system_params': {
                    'eigenvalues': eigenvalues.tolist(),
                    'is_stable': np.all(np.real(eigenvalues) < 0)
                },
                't': t,
                'u': u,
                'y': y,
                'y_fit': y_fit
            }
            
            return model_id, {
                'A': A.tolist(),
                'B': B.tolist(),
                'C': C.tolist(),
                'D': D.tolist()
            }, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model state-space system: {str(e)}")
    
    def model_transfer_function(self, t, u, y, order_num=2, order_den=2):
        """Model a transfer function from input-output data"""
        def transfer_function_response(t, *params):
            # Extract numerator and denominator coefficients
            num = params[:order_num+1]
            den = params[order_num+1:order_num+order_den+2]
            
            # Create transfer function
            sys = signal.TransferFunction(num, den)
            
            # Simulate system
            t_out, y_out, _ = signal.lsim(sys, u, t)
            
            return y_out
        
        # Initial guess
        # Simple second-order system guess
        num_guess = [1.0, 0.0, 0.0]  # 1
        den_guess = [1.0, 1.0, 1.0]  # s^2 + s + 1
        
        initial_guess = num_guess + den_guess
        
        # No bounds for simplicity
        bounds = None
        
        try:
            popt, pcov = optimize.curve_fit(transfer_function_response, t, y, p0=initial_guess, bounds=bounds)
            
            # Extract parameters
            num = popt[:order_num+1]
            den = popt[order_num+1:order_num+order_den+2]
            
            # Calculate fitted values
            y_fit = transfer_function_response(t, *popt)
            
            # Calculate R-squared
            ss_res = np.sum((y - y_fit) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate poles and zeros
            zeros = np.roots(num)
            poles = np.roots(den)
            
            # Store the model
            model_id = f"TransferFunction_{len(self.models)}"
            self.models[model_id] = {
                'type': 'TransferFunction',
                'parameters': {
                    'num': num.tolist(),
                    'den': den.tolist()
                },
                'covariance': pcov.tolist(),
                'r_squared': r_squared,
                'system_params': {
                    'zeros': zeros.tolist(),
                    'poles': poles.tolist(),
                    'is_stable': np.all(np.real(poles) < 0)
                },
                't': t,
                'u': u,
                'y': y,
                'y_fit': y_fit
            }
            
            return model_id, {
                'num': num.tolist(),
                'den': den.tolist()
            }, r_squared
        
        except Exception as e:
            raise ValueError(f"Failed to model transfer function: {str(e)}")
    
    def predict(self, model_id, t_new, u_new=None):
        """Use a fitted system model to make predictions"""
        if model_id not in self.models:
            raise ValueError(f"Model not found: {model_id}")
        
        model = self.models[model_id]
        model_type = model['type']
        params = model['parameters']
        
        if model_type == 'FirstOrder':
            K = params['K']
            tau = params['tau']
            
            # For step input
            if u_new is None or np.all(u_new == u_new[0]):
                output = K * (1 - np.exp(-t_new / tau))
            else:
                # For arbitrary input, solve differential equation
                dt = t_new[1] - t_new[0]
                output = np.zeros_like(t_new)
                for i in range(1, len(t_new)):
                    output[i] = output[i-1] + (K * u_new[i-1] - output[i-1]) * dt / tau
        
        elif model_type == 'SecondOrder':
            K = params['K']
            zeta = params['zeta']
            omega_n = params['omega_n']
            
            # For step input
            if u_new is None or np.all(u_new == u_new[0]):
                if zeta < 1:  # Underdamped
                    omega_d = omega_n * np.sqrt(1 - zeta**2)
                    output = K * (1 - np.exp(-zeta * omega_n * t_new) * (
                        np.cos(omega_d * t_new) + (zeta * omega_n / omega_d) * np.sin(omega_d * t_new)
                    ))
                elif zeta == 1:  # Critically damped
                    output = K * (1 - np.exp(-omega_n * t_new) * (1 + omega_n * t_new))
                else:  # Overdamped
                    s1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
                    s2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
                    output = K * (1 - (s2 * np.exp(s1 * t_new) - s1 * np.exp(s2 * t_new)) / (s2 - s1))
            else:
                # For arbitrary input, solve differential equation
                # This would require a more complex ODE solver
                raise ValueError("Arbitrary input prediction not implemented for second-order system")
        
        elif model_type == 'PID':
            Kp = params['Kp']
            Ki = params['Ki']
            Kd = params['Kd']
            
            if u_new is None:
                raise ValueError("Input required for PID prediction")
            
            # Simulate PID controller with a simple first-order plant
            dt = t_new[1] - t_new[0]
            
            # Initialize arrays
            e = np.zeros_like(t_new)
            integral = 0
            prev_error = 0
            output = np.zeros_like(t_new)
            
            # Simulate closed-loop system
            for i in range(1, len(t_new)):
                # Calculate error (assuming setpoint is 1)
                e[i] = 1 - output[i-1]
                
                # PID controller
                proportional = Kp * e[i]
                integral += Ki * e[i] * dt
                derivative = Kd * (e[i] - prev_error) / dt
                
                # Control signal
                u_sim = proportional + integral + derivative
                
                # Simple first-order plant (tau = 1)
                tau = 1.0
                output[i] = output[i-1] + (u_sim - output[i-1]) * dt / tau
                
                # Update previous error
                prev_error = e[i]
        
        elif model_type == 'StateSpace':
            A = np.array(params['A'])
            B = np.array(params['B'])
            C = np.array(params['C'])
            D = np.array(params['D'])
            
            if u_new is None:
                raise ValueError("Input required for state-space prediction")
            
            # Simulate state-space system
            dt = t_new[1] - t_new[0]
            n_states = A.shape[0]
            x = np.zeros((len(t_new), n_states))
            output = np.zeros_like(t_new)
            
            for i in range(1, len(t_new)):
                # State equation: x_dot = A*x + B*u
                x_dot = A @ x[i-1] + B * u_new[i-1]
                # Euler integration
                x[i] = x[i-1] + x_dot * dt
                # Output equation: y = C*x + D*u
                output[i] = C @ x[i] + D * u_new[i]
        
        elif model_type == 'TransferFunction':
            num = np.array(params['num'])
            den = np.array(params['den'])
            
            if u_new is None:
                raise ValueError("Input required for transfer function prediction")
            
            # Create transfer function
            sys = signal.TransferFunction(num, den)
            
            # Simulate system
            t_out, output, _ = signal.lsim(sys, u_new, t_new)
            
            # Interpolate to match t_new if needed
            if not np.array_equal(t_out, t_new):
                output = np.interp(t_new, t_out, output)
        
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