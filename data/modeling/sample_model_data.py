# data/modeling/sample_model_data.py
import numpy as np
import pandas as pd
from scipy import signal

def get_modeling_data():
    """Generate sample modeling data for electrical engineering"""
    # Time vector
    t = np.linspace(0, 1, 1000)
    
    # Generate data for signal modeling
    # Single sine wave with noise
    A = 1.0  # Amplitude
    f = 5.0  # Frequency
    phi = 0.0  # Phase
    offset = 0.0  # Offset
    
    y_sine = A * np.sin(2 * np.pi * f * t + phi) + offset + np.random.normal(0, 0.1, len(t))
    
    # Multi-sine wave with noise
    y_multi_sine = (1.0 * np.sin(2 * np.pi * 5 * t) + 
                    0.5 * np.sin(2 * np.pi * 10 * t) + 
                    0.2 * np.sin(2 * np.pi * 20 * t) + 
                    np.random.normal(0, 0.1, len(t)))
    
    # Exponential decay with noise
    tau = 0.1  # Time constant
    A_exp = 1.0  # Initial amplitude
    offset_exp = 0.1  # Offset
    
    y_exp = A_exp * np.exp(-t / tau) + offset_exp + np.random.normal(0, 0.05, len(t))
    
    # Double exponential with noise
    y_double_exp = (1.0 * np.exp(-t / 0.1) + 
                    0.3 * np.exp(-t / 0.3) + 
                    np.random.normal(0, 0.05, len(t)))
    
    # Polynomial with noise
    y_poly = 0.1 * t**2 - 0.5 * t + 2 + np.random.normal(0, 0.2, len(t))
    
    # Generate data for circuit modeling
    # RC circuit step response
    R = 1000  # 1kΩ
    C = 1e-6  # 1μF
    tau_rc = R * C  # Time constant
    
    v_in_rc = np.ones_like(t)  # Step input
    v_out_rc = 1.0 * (1 - np.exp(-t / tau_rc)) + np.random.normal(0, 0.02, len(t))
    
    # RL circuit step response
    R = 100  # 100Ω
    L = 0.1  # 0.1H
    tau_rl = L / R  # Time constant
    
    i_in_rl = np.ones_like(t)  # Step input
    i_out_rl = 0.01 * (1 - np.exp(-t / tau_rl)) + np.random.normal(0, 0.0002, len(t))
    
    # RLC circuit step response
    R = 100  # 100Ω
    L = 0.1  # 0.1H
    C = 1e-6  # 1μF
    
    omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
    zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
    
    v_in_rlc = np.ones_like(t)  # Step input
    
    if zeta < 1:  # Underdamped
        omega_d = omega_0 * np.sqrt(1 - zeta**2)
        v_out_rlc = 1.0 * (1 - np.exp(-zeta * omega_0 * t) * (
            np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t)
        )) + np.random.normal(0, 0.02, len(t))
    elif zeta == 1:  # Critically damped
        v_out_rlc = 1.0 * (1 - np.exp(-omega_0 * t) * (1 + omega_0 * t)) + np.random.normal(0, 0.02, len(t))
    else:  # Overdamped
        s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
        s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
        v_out_rlc = 1.0 * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)) + np.random.normal(0, 0.02, len(t))
    
    # Diode circuit
    v_in_diode = np.linspace(0, 2, 1000)  # Ramp input from 0V to 2V
    
    # Diode parameters (simplified Shockley diode model)
    Is = 1e-12  # Reverse saturation current (A)
    n = 1.0  # Ideality factor
    Vt = 0.026  # Thermal voltage at room temperature (V)
    
    # Diode current
    i_out_diode = Is * (np.exp(v_in_diode / (n * Vt)) - 1) + np.random.normal(0, 1e-9, len(v_in_diode))
    
    # BJT amplifier
    v_in_bjt = 0.01 * np.sin(2 * np.pi * 1000 * t)  # 10mV, 1kHz sine wave
    
    # BJT parameters (simplified model)
    beta = 100  # Current gain
    Vbe = 0.7  # Base-emitter voltage (V)
    
    # Amplifier gain (simplified)
    Av = -10  # Voltage gain
    
    # Output voltage
    v_out_bjt = Av * v_in_bjt + np.random.normal(0, 0.001, len(v_in_bjt))
    
    # Generate data for system modeling
    # First-order system step response
    K1 = 1.0  # Gain
    tau1 = 0.1  # Time constant
    
    u1 = np.ones_like(t)  # Step input
    y1 = K1 * (1 - np.exp(-t / tau1)) + np.random.normal(0, 0.02, len(t))
    
    # Second-order system step response
    K2 = 1.0  # Gain
    zeta2 = 0.5  # Damping ratio
    omega_n2 = 10  # Natural frequency
    
    u2 = np.ones_like(t)  # Step input
    
    if zeta2 < 1:  # Underdamped
        omega_d2 = omega_n2 * np.sqrt(1 - zeta2**2)
        y2 = K2 * (1 - np.exp(-zeta2 * omega_n2 * t) * (
            np.cos(omega_d2 * t) + (zeta2 * omega_n2 / omega_d2) * np.sin(omega_d2 * t)
        )) + np.random.normal(0, 0.02, len(t))
    elif zeta2 == 1:  # Critically damped
    # data/modeling/sample_model_data.py (continued)
        y2 = K2 * (1 - np.exp(-omega_n2 * t) * (1 + omega_n2 * t)) + np.random.normal(0, 0.02, len(t))
    else:  # Overdamped
        s1 = -omega_n2 * (zeta2 - np.sqrt(zeta2**2 - 1))
        s2 = -omega_n2 * (zeta2 + np.sqrt(zeta2**2 - 1))
        y2 = K2 * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1)) + np.random.normal(0, 0.02, len(t))
    
    # PID controller response
    Kp = 1.0  # Proportional gain
    Ki = 0.1  # Integral gain
    Kd = 0.01  # Derivative gain
    
    # Simple first-order plant
    plant_tau = 0.1
    plant_K = 1.0
    
    # Setpoint
    r = np.ones_like(t)  # Step setpoint
    
    # Simulate PID controller
    dt = t[1] - t[0]
    integral = 0
    prev_error = 0
    y_pid = np.zeros_like(t)
    
    for i in range(1, len(t)):
        # Calculate error
        error = r[i] - y_pid[i-1]
        
        # PID controller
        proportional = Kp * error
        integral += Ki * error * dt
        derivative = Kd * (error - prev_error) / dt
        
        # Control signal
        u = proportional + integral + derivative
        
        # Plant response (first-order)
        y_pid[i] = y_pid[i-1] + (plant_K * u - y_pid[i-1]) * dt / plant_tau
        
        # Update previous error
        prev_error = error
    
    # Add noise to output
    y_pid += np.random.normal(0, 0.02, len(t))
    
    # State-space system
    # Define state-space matrices
    A = np.array([[-0.1, 0.2], [-0.3, -0.4]])
    B = np.array([[0.1], [0.2]])
    C = np.array([[1.0, 0.0]])
    D = np.array([[0.0]])
    
    # Simulate state-space system
    x = np.zeros((len(t), 2))
    y_ss = np.zeros_like(t)
    u_ss = np.ones_like(t)  # Step input
    
    for i in range(1, len(t)):
        # State equation: x_dot = A*x + B*u
        x_dot = A @ x[i-1] + B * u_ss[i-1]
        # Euler integration
        x[i] = x[i-1] + x_dot * dt
        # Output equation: y = C*x + D*u
        y_ss[i] = C @ x[i] + D * u_ss[i]
    
    # Add noise to output
    y_ss += np.random.normal(0, 0.02, len(t))
    
    # Transfer function system
    # Define transfer function: G(s) = 1 / (s^2 + s + 1)
    num = [1.0]
    den = [1.0, 1.0, 1.0]
    
    # Simulate transfer function
    t_tf, y_tf, _ = signal.lsim((num, den), u_ss, t)
    
    # Add noise to output
    y_tf += np.random.normal(0, 0.02, len(y_tf))
    
    # Return all data
    return {
        'signal_modeling': {
            't': t,
            'y_sine': y_sine,
            'y_multi_sine': y_multi_sine,
            'y_exp': y_exp,
            'y_double_exp': y_double_exp,
            'y_poly': y_poly
        },
        'circuit_modeling': {
            't': t,
            'v_in_rc': v_in_rc,
            'v_out_rc': v_out_rc,
            'i_in_rl': i_in_rl,
            'i_out_rl': i_out_rl,
            'v_in_rlc': v_in_rlc,
            'v_out_rlc': v_out_rlc,
            'v_in_diode': v_in_diode,
            'i_out_diode': i_out_diode,
            'v_in_bjt': v_in_bjt,
            'v_out_bjt': v_out_bjt
        },
        'system_modeling': {
            't': t,
            'u1': u1,
            'y1': y1,
            'u2': u2,
            'y2': y2,
            'r': r,
            'y_pid': y_pid,
            'u_ss': u_ss,
            'y_ss': y_ss,
            'y_tf': y_tf
        }
    }