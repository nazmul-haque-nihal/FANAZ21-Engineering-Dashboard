# app/simulation/circuit_simulation.py
import numpy as np
from scipy import signal
import plotly.graph_objs as go
from app.utils import (
    create_transfer_function, calculate_step_response, calculate_impulse_response,
    calculate_system_metrics
)

class CircuitSimulator:
    def __init__(self, config):
        self.config = config
        self.circuits = {}  # Dictionary to store circuit simulations
    
    def simulate_rc_circuit(self, R, C, duration=0.01, sample_rate=10000, input_type='step'):
        """Simulate an RC circuit"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function for RC circuit: H(s) = 1 / (1 + RCs)
        num = [1]
        den = [R*C, 1]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        elif input_type == 'sine':
            freq = 100  # Hz
            u = np.sin(2 * np.pi * freq * t)
            # Frequency response
            frequencies = np.array([freq])
            H = create_transfer_function(num, den, frequencies)
            y = np.abs(H[0]) * np.sin(2 * np.pi * freq * t + np.angle(H[0]))
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
        # Store the circuit simulation
        circuit_id = f"RC_{R}_{C}_{input_type}"
        self.circuits[circuit_id] = {
            'type': 'RC',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'R': R, 'C': C, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics
        }
        
        return circuit_id, t, u, y, metrics
    
    def simulate_rl_circuit(self, R, L, duration=0.01, sample_rate=10000, input_type='step'):
        """Simulate an RL circuit"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function for RL circuit: H(s) = 1 / (1 + (L/R)s)
        num = [1]
        den = [L/R, 1]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        elif input_type == 'sine':
            freq = 100  # Hz
            u = np.sin(2 * np.pi * freq * t)
            # Frequency response
            frequencies = np.array([freq])
            H = create_transfer_function(num, den, frequencies)
            y = np.abs(H[0]) * np.sin(2 * np.pi * freq * t + np.angle(H[0]))
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
        # Store the circuit simulation
        circuit_id = f"RL_{R}_{L}_{input_type}"
        self.circuits[circuit_id] = {
            'type': 'RL',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'R': R, 'L': L, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics
        }
        
        return circuit_id, t, u, y, metrics
    
    def simulate_rlc_circuit(self, R, L, C, duration=0.01, sample_rate=10000, input_type='step'):
        """Simulate an RLC circuit"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transfer function for RLC circuit: H(s) = 1 / (LCs² + RCs + 1)
        num = [1]
        den = [L*C, R*C, 1]
        
        # Generate input signal
        if input_type == 'step':
            u = np.ones_like(t)
            y = calculate_step_response(num, den, t)
        elif input_type == 'impulse':
            u = np.zeros_like(t)
            u[0] = 1  # Impulse at t=0
            y = calculate_impulse_response(num, den, t)
        elif input_type == 'sine':
            freq = 100  # Hz
            u = np.sin(2 * np.pi * freq * t)
            # Frequency response
            frequencies = np.array([freq])
            H = create_transfer_function(num, den, frequencies)
            y = np.abs(H[0]) * np.sin(2 * np.pi * freq * t + np.angle(H[0]))
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Calculate system metrics
        metrics = calculate_system_metrics(t, y, u)
        
        # Calculate resonant frequency and damping ratio
        omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
        zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
        
        # Store the circuit simulation
        circuit_id = f"RLC_{R}_{L}_{C}_{input_type}"
        self.circuits[circuit_id] = {
            'type': 'RLC',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'R': R, 'L': L, 'C': C, 'input_type': input_type},
            'transfer_function': {'num': num, 'den': den},
            'metrics': metrics,
            'circuit_params': {'omega_0': omega_0, 'zeta': zeta}
        }
        
        return circuit_id, t, u, y, metrics
    
    def simulate_diode_circuit(self, R, duration=0.01, sample_rate=10000, input_type='sine'):
        """Simulate a simple diode circuit"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Diode parameters (simplified Shockley diode model)
        Is = 1e-12  # Reverse saturation current (A)
        Vt = 0.026  # Thermal voltage at room temperature (V)
        n = 1.0  # Ideality factor
        
        # Generate input signal
        if input_type == 'sine':
            freq = 50  # Hz
            amplitude = 5  # V
            u = amplitude * np.sin(2 * np.pi * freq * t)
        elif input_type == 'step':
            u = 5 * np.ones_like(t)  # 5V step
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Simulate diode circuit (iterative solution)
        y = np.zeros_like(t)
        for i in range(len(t)):
            # Initial guess
            v_diode = 0.7  # Initial guess for diode voltage
            
            # Newton-Raphson iteration
            for _ in range(10):  # Max 10 iterations
                # Diode current
                i_diode = Is * (np.exp(v_diode / (n * Vt)) - 1)
                # Resistor current
                i_resistor = (u[i] - v_diode) / R
                # Function to zero
                f = i_diode - i_resistor
                # Derivative
                df_dv = (Is / (n * Vt)) * np.exp(v_diode / (n * Vt)) + 1/R
                # Update
                v_diode_new = v_diode - f / df_dv
                
                # Check convergence
                if abs(v_diode_new - v_diode) < 1e-6:
                    break
                
                v_diode = v_diode_new
            
            # Output voltage (across resistor)
            y[i] = u[i] - v_diode
        
        # Store the circuit simulation
        circuit_id = f"Diode_{R}_{input_type}"
        self.circuits[circuit_id] = {
            'type': 'Diode',
            't': t,
            'u': u,
            'y': y,
            'parameters': {'R': R, 'input_type': input_type},
            'diode_params': {'Is': Is, 'Vt': Vt, 'n': n}
        }
        
        return circuit_id, t, u, y
    
    def simulate_transistor_amplifier(self, R1, R2, Rc, Re, Vcc, beta=100, duration=0.01, sample_rate=10000, input_type='sine'):
        """Simulate a simple BJT amplifier circuit"""
        # Time vector
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        # Transistor parameters
        Vbe = 0.7  # Base-emitter voltage (V)
        
        # Calculate DC operating point
        Vb = Vcc * R2 / (R1 + R2)  # Base voltage
        Ve = Vb - Vbe  # Emitter voltage
        Ie = Ve / Re  # Emitter current
        Ic = Ie * beta / (beta + 1)  # Collector current
        Vc = Vcc - Ic * Rc  # Collector voltage
        
        # Generate input signal
        if input_type == 'sine':
            freq = 1000  # Hz
            amplitude = 0.01  # V (small signal)
            u = amplitude * np.sin(2 * np.pi * freq * t)
        elif input_type == 'step':
            u = 0.01 * np.ones_like(t)  # 10mV step
        else:
            raise ValueError(f"Unknown input type: {input_type}")
        
        # Small signal parameters
        r_e = 0.026 / Ie  # Emitter resistance
        Av = -Rc / (Re + r_e)  # Voltage gain
        
        # Simulate amplifier (simplified small signal model)
        y = Vc + Av * u  # Output voltage
        
        # Store the circuit simulation
        circuit_id = f"BJT_{R1}_{R2}_{Rc}_{Re}_{input_type}"
        self.circuits[circuit_id] = {
            'type': 'BJT',
            't': t,
            'u': u,
            'y': y,
            'parameters': {
                'R1': R1, 'R2': R2, 'Rc': Rc, 'Re': Re, 'Vcc': Vcc, 
                'input_type': input_type, 'beta': beta
            },
            'operating_point': {
                'Vb': Vb, 'Ve': Ve, 'Ie': Ie, 'Ic': Ic, 'Vc': Vc
            },
            'small_signal': {
                'r_e': r_e, 'Av': Av
            }
        }
        
        return circuit_id, t, u, y
    
    def get_frequency_response(self, circuit_id, frequencies=None):
        """Get the frequency response of a circuit"""
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit ID not found: {circuit_id}")
        
        circuit = self.circuits[circuit_id]
        
        if 'transfer_function' not in circuit:
            raise ValueError(f"Circuit {circuit_id} does not have a transfer function")
        
        num = circuit['transfer_function']['num']
        den = circuit['transfer_function']['den']
        
        if frequencies is None:
            # Generate logarithmically spaced frequencies
            frequencies = np.logspace(-1, 5, 1000)  # 0.1 Hz to 100 kHz
        
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
    
    def get_circuit(self, circuit_id):
        """Get a circuit simulation by ID"""
        if circuit_id not in self.circuits:
            raise ValueError(f"Circuit ID not found: {circuit_id}")
        return self.circuits[circuit_id]
    
    def list_circuits(self):
        """List all available circuit simulations"""
        return list(self.circuits.keys())
    
    def remove_circuit(self, circuit_id):
        """Remove a circuit simulation by ID"""
        if circuit_id in self.circuits:
            del self.circuits[circuit_id]
            return True
        return False