# data/simulation/sample_simulation_data.py
import numpy as np
import pandas as pd
from scipy import signal

def get_simulation_data():
    """Generate sample simulation data for electrical engineering"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate different types of signals
    signals = {
        'sine': np.sin(2 * np.pi * 5 * t),
        'square': signal.square(2 * np.pi * 5 * t),
        'sawtooth': signal.sawtooth(2 * np.pi * 5 * t),
        'noise': np.random.normal(0, 0.1, len(t))
    }
    
    # Generate composite signal
    composite = signals['sine'] + 0.5 * signals['square'] + 0.2 * signals['noise']
    
    # Generate modulated signals
    carrier_freq = 50  # Hz
    modulation_index = 0.5
    
    # AM modulated signal
    am_signal = (1 + modulation_index * signals['sine']) * np.sin(2 * np.pi * carrier_freq * t)
    
    # FM modulated signal
    fm_signal = np.sin(2 * np.pi * carrier_freq * t + modulation_index * signals['sine'])
    
    # Generate filtered signals
    from app.utils import filter_signal
    filtered_signal = filter_signal(signals['sine'], 1000, 'lowpass', 10)
    
    # Generate circuit responses
    # RC circuit response
    tau = 0.01  # Time constant
    rc_response = 1 - np.exp(-t / tau)
    
    # RL circuit response
    tau = 0.01  # Time constant
    rl_response = 1 - np.exp(-t / tau)
    
    # RLC circuit response
    R = 100  # Ohms
    L = 0.1  # Henry
    C = 1e-6  # Farad
    omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
    zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
    
    if zeta < 1:  # Underdamped
        omega_d = omega_0 * np.sqrt(1 - zeta**2)
        rlc_response = 1 - np.exp(-zeta * omega_0 * t) * (
            np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t)
        )
    elif zeta == 1:  # Critically damped
        rlc_response = 1 - np.exp(-omega_0 * t) * (1 + omega_0 * t)
    else:  # Overdamped
        s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
        s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
        rlc_response = 1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)
    
    # Generate control system responses
    # First-order system response
    K = 1.0  # Gain
    tau = 0.1  # Time constant
    first_order_response = K * (1 - np.exp(-t / tau))
    
    # Second-order system response
    K = 1.0  # Gain
    zeta = 0.5  # Damping ratio
    omega_n = 10  # Natural frequency
    
    if zeta < 1:  # Underdamped
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        second_order_response = K * (1 - np.exp(-zeta * omega_n * t) * (
            np.cos(omega_d * t) + (zeta * omega_n / omega_d) * np.sin(omega_d * t)
        ))
    elif zeta == 1:  # Critically damped
        second_order_response = K * (1 - np.exp(-omega_n * t) * (1 + omega_n * t))
    else:  # Overdamped
        s1 = -omega_n * (zeta - np.sqrt(zeta**2 - 1))
        s2 = -omega_n * (zeta + np.sqrt(zeta**2 - 1))
        second_order_response = K * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))
    
    # PID controller response
    Kp = 1.0  # Proportional gain
    Ki = 0.1  # Integral gain
    Kd = 0.01  # Derivative gain
    
    # Simple first-order plant
    plant_tau = 0.1
    
    # Simulate PID controller
    dt = t[1] - t[0]
    integral = 0
    prev_error = 0
    pid_response = np.zeros_like(t)
    
    for i in range(1, len(t)):
        # Calculate error (setpoint = 1)
        error = 1 - pid_response[i-1]
        
        # PID controller
        proportional = Kp * error
        integral += Ki * error * dt
        derivative = Kd * (error - prev_error) / dt
        
        # Control signal
        u = proportional + integral + derivative
        
        # Plant response
        pid_response[i] = pid_response[i-1] + (u - pid_response[i-1]) * dt / plant_tau
        
        # Update previous error
        prev_error = error
    
    # Return all data
    return {
        't': t,
        'signals': signals,
        'composite': composite,
        'am_signal': am_signal,
        'fm_signal': fm_signal,
        'filtered_signal': filtered_signal,
        'rc_response': rc_response,
        'rl_response': rl_response,
        'rlc_response': rlc_response,
        'first_order_response': first_order_response,
        'second_order_response': second_order_response,
        'pid_response': pid_response
    }