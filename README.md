# FANAZ21 Engineering Dashboard

A comprehensive, modern data visualization dashboard for simulation, modeling, and multi-purpose engineering analysis.

## Features

- **User Authentication**: Secure login system with role-based access control
- **Real-time Data Updates**: Dashboard updates every 3 seconds with new data
- **Simulation Module**:
  - Signal simulation (sine, square, sawtooth, noise, composite)
  - Circuit simulation (RC, RL, RLC, diode, BJT amplifier)
  - Control system simulation (first-order, second-order, PID, state-space)
- **Analysis Module**:
  - Statistical analysis (basic statistics, correlation, hypothesis testing, regression)
  - Time series analysis (stationarity, decomposition, autocorrelation, ARIMA)
  - Frequency analysis (FFT, power spectrum, spectrogram, coherence)
- **Modeling Module**:
  - Signal modeling (sinusoidal, exponential, polynomial, Fourier)
  - Circuit modeling (RC, RL, RLC, diode, BJT)
  - System modeling (first-order, second-order, PID, state-space, transfer function)
- **Data Upload**: Support for CSV, Excel, JSON, TXT, MAT, and HDF5 files
- **Export Capabilities**: Export data as CSV and charts as images
- **Modern Dark Theme**: Professional dark interface optimized for engineering applications

## Technology Stack

- **Frontend**: Dash by Plotly, Dash Bootstrap Components
- **Backend**: Flask
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Plotly
- **Authentication**: Flask-Login
- **System Monitoring**: psutil

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FANAZ21.git
   cd FANAZ21