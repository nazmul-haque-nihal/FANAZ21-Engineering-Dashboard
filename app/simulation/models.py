# app/simulation/models.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

def generate_simulation_data(simulation_type='default'):
    """Generate simulation data based on the simulation type"""
    np.random.seed(42)
    
    if simulation_type == 'vibration':
        return generate_vibration_data()
    elif simulation_type == 'thermal':
        return generate_thermal_data()
    elif simulation_type == 'fluid_dynamics':
        return generate_fluid_dynamics_data()
    elif simulation_type == 'structural':
        return generate_structural_data()
    else:
        return generate_default_simulation_data()

def generate_vibration_data():
    """Generate vibration simulation data"""
    t = np.linspace(0, 10, 1000)
    # Create a signal with multiple frequencies
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t) + 0.2 * np.sin(2 * np.pi * 30 * t)
    # Add some noise
    noise = 0.1 * np.random.normal(size=len(t))
    signal_with_noise = signal + noise
    
    df = pd.DataFrame({
        'Time': t,
        'Amplitude': signal_with_noise,
        'Frequency': [5] * len(t),
        'Damping': [0.05] * len(t)
    })
    
    return df

def generate_thermal_data():
    """Generate thermal simulation data"""
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a temperature distribution
    T = 100 * np.exp(-((X-5)**2 + (Y-5)**2) / 10)
    
    # Flatten for DataFrame
    data = []
    for i in range(len(x)):
        for j in range(len(y)):
            data.append([X[i, j], Y[i, j], T[i, j]])
    
    df = pd.DataFrame(data, columns=['X', 'Y', 'Temperature'])
    return df

def generate_fluid_dynamics_data():
    """Generate fluid dynamics simulation data"""
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple flow field
    U = -Y
    V = X
    speed = np.sqrt(U**2 + V**2)
    
    # Flatten for DataFrame
    data = []
    for i in range(len(x)):
        for j in range(len(y)):
            data.append([X[i, j], Y[i, j], U[i, j], V[i, j], speed[i, j]])
    
    df = pd.DataFrame(data, columns=['X', 'Y', 'U', 'V', 'Speed'])
    return df

def generate_structural_data():
    """Generate structural analysis simulation data"""
    # Create a simple beam with stress distribution
    x = np.linspace(0, 10, 100)
    
    # Stress distribution (simplified)
    stress = 1000 * (x - 5)**2
    
    # Displacement (simplified)
    displacement = 0.001 * x**3
    
    df = pd.DataFrame({
        'Position': x,
        'Stress': stress,
        'Displacement': displacement,
        'Load': [100] * len(x)
    })
    
    return df

def generate_default_simulation_data():
    """Generate default simulation data"""
    t = np.linspace(0, 10, 100)
    
    # Create a simple oscillation
    y = np.sin(2 * np.pi * t)
    
    df = pd.DataFrame({
        'Time': t,
        'Value': y,
        'Parameter1': [1.0] * len(t),
        'Parameter2': [0.5] * len(t)
    })
    
    return df

def run_simulation(simulation_type, parameters):
    """Run a simulation with given parameters"""
    # This is a simplified simulation runner
    # In a real application, this would interface with simulation software
    
    if simulation_type == 'vibration':
        return run_vibration_simulation(parameters)
    elif simulation_type == 'thermal':
        return run_thermal_simulation(parameters)
    elif simulation_type == 'fluid_dynamics':
        return run_fluid_dynamics_simulation(parameters)
    elif simulation_type == 'structural':
        return run_structural_simulation(parameters)
    else:
        return run_default_simulation(parameters)

def run_vibration_simulation(parameters):
    """Run vibration simulation"""
    frequency = parameters.get('frequency', 5)
    amplitude = parameters.get('amplitude', 1.0)
    damping = parameters.get('damping', 0.05)
    
    t = np.linspace(0, 10, 1000)
    signal = amplitude * np.sin(2 * np.pi * frequency * t) * np.exp(-damping * t)
    
    df = pd.DataFrame({
        'Time': t,
        'Amplitude': signal,
        'Frequency': [frequency] * len(t),
        'Damping': [damping] * len(t)
    })
    
    return df

def run_thermal_simulation(parameters):
    """Run thermal simulation"""
    source_temp = parameters.get('source_temp', 100)
    ambient_temp = parameters.get('ambient_temp', 20)
    thermal_conductivity = parameters.get('thermal_conductivity', 1.0)
    
    x = np.linspace(0, 10, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    
    # Create a temperature distribution
    T = ambient_temp + (source_temp - ambient_temp) * np.exp(-((X-5)**2 + (Y-5)**2) / (10 * thermal_conductivity))
    
    # Flatten for DataFrame
    data = []
    for i in range(len(x)):
        for j in range(len(y)):
            data.append([X[i, j], Y[i, j], T[i, j]])
    
    df = pd.DataFrame(data, columns=['X', 'Y', 'Temperature'])
    return df

def run_fluid_dynamics_simulation(parameters):
    """Run fluid dynamics simulation"""
    viscosity = parameters.get('viscosity', 1.0)
    flow_rate = parameters.get('flow_rate', 1.0)
    
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 10, 50)
    X, Y = np.meshgrid(x, y)
    
    # Create a simple flow field
    U = -Y * flow_rate / viscosity
    V = X * flow_rate / viscosity
    speed = np.sqrt(U**2 + V**2)
    
    # Flatten for DataFrame
    data = []
    for i in range(len(x)):
        for j in range(len(y)):
            data.append([X[i, j], Y[i, j], U[i, j], V[i, j], speed[i, j]])
    
    df = pd.DataFrame(data, columns=['X', 'Y', 'U', 'V', 'Speed'])
    return df

def run_structural_simulation(parameters):
    """Run structural simulation"""
    load = parameters.get('load', 100)
    youngs_modulus = parameters.get('youngs_modulus', 200000)
    moment_of_inertia = parameters.get('moment_of_inertia', 1.0)
    
    x = np.linspace(0, 10, 100)
    
    # Stress distribution (simplified beam bending)
    stress = load * x * (10 - x) / (2 * youngs_modulus * moment_of_inertia)
    
    # Displacement (simplified beam bending)
    displacement = load * x**2 * (10 - x)**2 / (24 * youngs_modulus * moment_of_inertia)
    
    df = pd.DataFrame({
        'Position': x,
        'Stress': stress,
        'Displacement': displacement,
        'Load': [load] * len(x)
    })
    
    return df

def run_default_simulation(parameters):
    """Run default simulation"""
    frequency = parameters.get('frequency', 1.0)
    amplitude = parameters.get('amplitude', 1.0)
    
    t = np.linspace(0, 10, 100)
    
    # Create a simple oscillation
    y = amplitude * np.sin(2 * np.pi * frequency * t)
    
    df = pd.DataFrame({
        'Time': t,
        'Value': y,
        'Parameter1': [frequency] * len(t),
        'Parameter2': [amplitude] * len(t)
    })
    
    return df