# FANAZ21 Engineering Dashboard

A comprehensive engineering analysis dashboard built with Dash, Plotly, and Bootstrap that provides simulation, modeling, and analysis capabilities for various engineering disciplines including electrical engineering, electronics, control systems, power systems, and telecommunications.

<img width="1913" height="906" alt="Screenshot_20250906_122016" src="https://github.com/user-attachments/assets/0a338089-7464-48c9-a7db-1eb2733c1030" />

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=FANAZ21+Engineering+Dashboard+Screenshot)

## Features

### Dashboard
- Real-time system monitoring with customizable update intervals
- Project type-specific data visualization (Electrical, Electronics, Control Systems, Power Systems, Telecommunications)
- Performance metrics tracking
- Recent activities log

### Simulation Module
- **Signal Simulation**: Generate and visualize various signal types (sine, square, sawtooth, noise, composite)
- **Circuit Simulation**: Simulate RC, RL, RLC, diode, and BJT amplifier circuits
- **Control System Simulation**: Model first-order, second-order, PID, and state-space control systems

### Modeling Module
- **Signal Modeling**: Fit sinusoidal, exponential, polynomial, and Fourier models to data
- **Circuit Modeling**: Create detailed circuit models with temperature effects and parasitic elements
- **System Modeling**: Develop first-order, second-order, PID, state-space, and transfer function models

### Analysis Module
- **Statistical Analysis**: Basic statistics, correlation analysis, hypothesis testing, and regression analysis
- **Time Series Analysis**: Stationarity testing, decomposition, autocorrelation, and ARIMA modeling
- **Frequency Analysis**: FFT analysis, power spectrum, spectrogram, and coherence analysis

### Data Management
- Upload data files (CSV, Excel, JSON, TXT)
- Manual data input capability
- Data preview and validation
- Export data and charts functionality

## Technology Stack

- **Frontend**: Dash, Plotly, Dash Bootstrap Components
- **Backend**: Flask (Dash's underlying server)
- **Data Processing**: Pandas, NumPy, SciPy
- **Signal Processing**: SciPy Signal
- **Machine Learning**: Scikit-learn
- **Visualization**: Plotly Graph Objects, Plotly Express


##Configuration 

The application can be configured through the sidebar controls: 

     Theme: Switch between dark and light themes
     Update Interval: Set the refresh rate for real-time data (1-10 seconds)
     Project Type: Select the engineering discipline for specialized analysis
     

Data Formats 
Supported Upload Formats 

     CSV: Comma-separated values
     Excel: .xlsx and .xls files
     JSON: JavaScript Object Notation
     TXT: Tab-delimited text files
     

Manual Input Format 

Data can be entered manually using comma-separated or tab-delimited format: 
 
 
 
1
2
3
1.0, 2.5, 3.7
2.1, 3.2, 4.8
3.2, 4.1, 5.9
 
 
 
API Reference 
Callback Structure 

The application follows a modular callback structure: 

     Navigation Callbacks: Handle page routing
     Dashboard Callbacks: Update real-time data and metrics
     Simulation Callbacks: Process simulation parameters and generate visualizations
     Modeling Callbacks: Create and fit models to data
     Analysis Callbacks: Perform various types of data analysis
     Data Management Callbacks: Handle file uploads, data processing, and exports
     

Key Functions 

     update_dashboard(): Updates the main dashboard with real-time data
     update_signal_graph(): Generates signal visualizations based on parameters
     update_circuit_graph(): Simulates circuit behavior
     update_system_graph(): Models control system responses
     update_stat_analysis(): Performs statistical analysis
     update_time_analysis(): Conducts time series analysis
     update_freq_analysis(): Executes frequency domain analysis
     

Contributing 

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change. 
Development Guidelines 

     Follow the existing code style and structure
     Add appropriate comments and documentation
     Ensure all callbacks are properly tested
     Update the README as needed
     

Bug Reports 

When reporting bugs, please include: 

     A clear and descriptive title
     Steps to reproduce the issue
     Expected behavior
     Actual behavior
     Screenshots if applicable
     Your environment details (OS, Python version, etc.)
     

License 

This project is licensed under the MIT License - see the LICENSE  file for details. 
Acknowledgments 

     The Dash team for creating an excellent framework for building analytical web applications
     Plotly for powerful visualization capabilities
     The scientific Python community (NumPy, SciPy, Pandas) for robust data processing tools
     Bootstrap for responsive UI components
     

Contact 

For questions, suggestions, or collaboration opportunities, please contact: 

     Your Name
     your.email@example.com 
     Your LinkedIn Profile 
     Your GitHub Profile 
     

Roadmap 

     Add more circuit models (Op-Amps, Filters, etc.)
     Implement additional analysis techniques (Wavelet analysis, etc.)
     Create a user authentication system
     Add data persistence with a database backend
     Develop a report generation feature
     Add real-time data streaming capabilities
     Create a deployment guide for cloud platforms
     


## Installation


1. Clone the repository:
```bash
git clone https://github.com/yourusername/FANAZ21-Engineering-Dashboard.git
cd FANAZ21-Engineering-Dashboard

Project Structure
FANAZ21-Engineering-Dashboard/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── assets/               # Images and other static assets (if any)
