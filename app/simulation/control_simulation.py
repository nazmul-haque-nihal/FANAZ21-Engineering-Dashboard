# app/simulation/control_simulation.py
import numpy as np
from scipy import signal
import plotly.graph_objs as go
from app.utils import (
    create_transfer_function, calculate_step_response, calculate_impulse_response,
    calculate_system_metrics
)

class ControlSimulator:
    def __init__(self, config):
        self.config = config
        self.systems = {}  # Dictionary to store control system simulations
    
    def simulate_first_order_system(self, K, tau, duration=10.0, sample_rate=100, input_type='step'):
        """Simulate a first-order system: K / (tau*s + 1)"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function
        num = [K]
        den = [tau, 1]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        elif input_type == 'ramp':
            u = t
            # For ramp input, we need to integrate the step response
            _, y_step = calculate_step_response(num, den, t)
            y = np.cumsum(y_step) * (t[1] - t[0])  # Integrate
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
        # Calculate time constant
        time_constant = tau
        
        # Calculate settling time (4 * time constant for first-order)
        settling_time = 4 * tau
        
        # Store the system simulation
        system_id = f"FirstOrder_{K}_{tau}_{input_type}"
        self.systems[system_id] = {
            'type': 'FirstOrder',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'K': K, 'tau': tau, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics,
            'system_params': {'time_constant': time_constant, 'settling_time': settling_time}
        }
        
        return system_id, t, u, y, metrics
    
    def simulate_second_order_system(self, K, zeta, omega_n, duration=10.0, sample_rate=100, input_type='step'):
        """Simulate a second-order system: K * omega_n^2 / (s^2 + 2*zeta*omega_n*s + omega_n^2)"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function
        num = [K * omega_n**2]
        den = [1, 2 * zeta * omega_n, omega_n**2]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        elif input_type == 'ramp':
            u = t
            # For ramp input, we need to integrate the step response
            _, y_step = calculate_step_response(num, den, t)
            y = np.cumsum(y_step) * (t[1] - t[0])  # Integrate
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
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
        
        # Store the system simulation
        system_id = f"SecondOrder_{K}_{zeta}_{omega_n}_{input_type}"
        self.systems[system_id] = {
            'type': 'SecondOrder',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'K': K, 'zeta': zeta, 'omega_n': omega_n, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics,
            'system_params': {
                'omega_d': omega_d, 'T_p': T_p, 'M_p': M_p, 'T_s': T_s
            }
        }
        
        return system_id, t, u, y, metrics
    
    def simulate_pid_controller(self, Kp, Ki, Kd, plant_num, plant_den, duration=10.0, sample_rate=100, setpoint=1.0):
        """Simulate a PID controller with a given plant"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        dt = t[1] - t[0]
        
        # Initialize arrays
        u = np.zeros_like(t)  # Control signal
        y = np.zeros_like(t)  # Plant output
        e = np.zeros_like(t)  # Error signal
        
        # Setpoint
        r = setpoint * np.ones_like(t)
        
        # PID controller state
        integral = 0
        prev_error = 0
        
        # Simulate closed-loop system
        for i in range(1, len(t)):
            # Calculate error
            e[i] = r[i] - y[i-1]
            
            # PID controller
            proportional = Kp * e[i]
            integral += Ki * e[i] * dt
            derivative = Kd * (e[i] - prev_error) / dt
            
            # Control signal
            u[i] = proportional + integral + derivative
            
            # Limit control signal (anti-windup)
            u[i] = np.clip(u[i], -10, 10)
            
            # Simulate plant response (simplified)
            # For a more accurate simulation, we would use a proper ODE solver
            # Here we'll use a simple Euler method for demonstration
            if len(plant_den) == 2:  # First-order plant
                tau = plant_den[1] / plant_den[0]
                K = plant_num[0] / plant_den[0]
                y[i] = y[i-1] + (K * u[i] - y[i-1]) * dt / tau
            elif len(plant_den) == 3:  # Second-order plant
                # Convert to state-space form
                a1 = plant_den[1] / plant_den[0]
                a2 = plant_den[2] / plant_den[0]
                b0 = plant_num[0] / plant_den[0]
                
                # State variables
                if i == 1:
                    x1 = 0
                    x2 = 0
                else:
                    # Euler method
                    x1_dot = x2
                    x2_dot = -a2 * x1 - a1 * x2 + b0 * u[i]
                    x1 += x1_dot * dt
                    x2 += x2_dot * dt
                
                y[i] = x1
            else:
                # Default to first-order approximation
                tau = plant_den[-1] / plant_den[0]
                K = plant_num[0] / plant_den[0]
                y[i] = y[i-1] + (K * u[i] - y[i-1]) * dt / tau
            
            # Update previous error
            prev_error = e[i]
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, r)
        
        # Store the system simulation
        system_id = f"PID_{Kp}_{Ki}_{Kd}"
        self.systems[system_id] = {
            'type': 'PID',
            't': t,
            'r': r,
            'u': u,
            'y': y,
            'e': e,
            'parameters': {
                'Kp': Kp, 'Ki': Ki, 'Kd': Kd,
                'plant_num': plant_num, 'plant_den': plant_den,
                'setpoint': setpoint
            },
            'metrics': metrics
        }
        
        return system_id, t, r, u, y, e, metrics
    
    def simulate_state_space_system(self, A, B, C, D, duration=10.0, sample_rate=100, input_type='step'):
        """Simulate a state-space system"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        dt = t[1] - t[0]
        
        # System dimensions
        n_states = A.shape[0]
        n_inputs = B.shape[1]
        n_outputs = C.shape[0]
        
        # Initialize arrays
        u = np.zeros((len(t), n_inputs))
        y = np.zeros((len(t), n_outputs))
        x = np.zeros((len(t), n_states))
        
        # Generate input signal
        if input_type == 'step':
            u[:, 0] = 1.0  # Step input to first input
        elif input_type == 'impulse':
            u[0, 0] = 1.0  # Impulse at t=0 to first input
        elif input_type == 'sine':
            freq = 1.0  # Hz
            u[:, 0] = np.sin(2 * np.pi * freq * t)  # Sine input to first input
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Simulate state-space system
        for i in range(1, len(t)):
            # State equation: x_dot = A*x + B*u
            x_dot = A @ x[i-1] + B @ u[i-1]
            # Euler integration
            x[i] = x[i-1] + x_dot * dt
            # Output equation: y = C*x + D*u
            y[i] = C @ x[i] + D @ u[i]
        
        # Calculate system metrics for each output
        metrics = []
        for i in range(n_outputs):
            metrics.append(calculate_system_metrics(t, y[:, i], u[:, 0]))
        
        # Store the system simulation
        system_id = f"StateSpace_{input_type}"
        self.systems[system_id] = {
            'type': 'StateSpace',
            't': t,
            'u': u,
            'y': y,
            'x': x,
            'parameters': {
                'A': A.tolist(), 'B': B.tolist(), 'C': C.tolist(), 'D': D.tolist(),
                'input_type': input_type
            },
            'metrics': metrics
        }
        
        return system_id, t, u, y, x, metrics
    
    def simulate_lead_lag_compensator(self, z, p, K, duration=10.0, sample_rate=100, input_type='step'):
        """Simulate a lead-lag compensator: K * (s + z) / (s + p)"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function
        num = [K, K * z]
        den = [1, p]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
        # Calculate compensator parameters
        if z < p:  # Lead compensator
            alpha = z / p
            phi_max = np.arcsin((1 - alpha) / (1 + alpha)) * 180 / np.pi  # Maximum phase lead (degrees)
            w_max = np.sqrt(z * p)  # Frequency at maximum phase lead
        else:  # Lag compensator
            alpha = z / p
            phi_max = np.arcsin((1 - alpha) / (1 + alpha)) * 180 / np.pi  # Maximum phase lag (degrees)
            w_max = np.sqrt(z * p)  # Frequency at maximum phase lag
        
        # Store the system simulation
        system_id = f"LeadLag_{z}_{p}_{K}_{input_type}"
        self.systems[system_id] = {
            'type': 'LeadLag',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'z': z, 'p': p, 'K': K, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics,
            'compensator_params': {
                'alpha': alpha, 'phi_max': phi_max, 'w_max': w_max
            }
        }
        
        return system_id, t, u, y, metrics
    
    def get_frequency_response(self, system_id, frequencies=None):
        """Get the frequency response of a system"""
        if system_id not in self.systems:
            raise ValueError(f"System ID not found: {system_id}")
        
        system = self.systems[system_id]
        
        if 'transfer_function' not in system:
            raise ValueError(f"System {system_id} does not have a transfer function")
        
        num = system['transfer_function']['num']
        den = system['transfer_function']['den']
        
        if frequencies is None:
            # Generate logarithmically spaced frequencies
            frequencies = np.logspace(-3, 3, 1000)  # 0.001 Hz to 1000 Hz
        
        # Calculate transfer function
        H = create_transfer_function(num, den, frequencies)
        
        # Calculate magnitude and phase
        magnitude = np.abs(H)
        phase = np.angle(H, deg=True)
        
        return {
            'frequencies': frequencies,
            'magnitude': magnitude,
            'phase': phase
        }
    
    def get_root_locus(self, system_id, K_range=None):
        """Get the root locus of a system"""
        if system_id not in self.systems:
            raise ValueError(f"System ID not found: {system_id}")
        
        system = self.systems[system_id]
        
        if 'transfer_function' not in system:
            raise ValueError(f"System {system_id} does not have a transfer function")
        
        num = system['transfer_function']['num']
        den = system['transfer_function']['den']
        
        # Calculate root locus
        if K_range is None:
            K_range = np.logspace(-2, 2, 100)
        
        roots = []
        for K in K_range:
            # Characteristic equation: 1 + K * G(s) = 0
            # Where G(s) = num(s) / den(s)
            # So the roots are the roots of den(s) + K * num(s) = 0
            char_poly = np.zeros(max(len(num), len(den)))
            for i, coef in enumerate(den):
                char_poly[i] += coef
            for i, coef in enumerate(num):
                char_poly[i] += K * coef
            
            # Find roots
            r = np.roots(char_poly)
            roots.append(r)
        
        return {
            'K': K_range,
            'roots': roots
        }
    
    def get_system(self, system_id):
        """Get a system simulation by ID"""
        if system_id not in self.systems:
            raise ValueError(f"System ID not found: {system_id}")
        return self.systems[system_id]
    
    def list_systems(self):
        """List all available system simulations"""
        return list(self.systems.keys())
    
    def remove_system(self, system_id):
        """Remove a system simulation by ID"""
        if system_id in self.systems:
            del self.systems[system_id]
            return True
        return False