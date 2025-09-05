# app/callbacks.py
from dash import Input, Output, State, dcc, html, dash_table
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import json
import base64
import io
from flask_login import current_user
from app.auth import is_admin, is_engineer, is_analyst
from app.data_processor import DataProcessor
from app.simulation import SignalSimulator, CircuitSimulator, ControlSimulator
from app.analysis import StatisticalAnalyzer, TimeSeriesAnalyzer, FrequencyAnalyzer
from app.modeling import SignalModeler, CircuitModeler, SystemModeler
from app.simulation.visualizations import create_simulation_plots
from app.analysis.visualizations import create_analysis_plots
from app.modeling.visualizations import create_modeling_plots
from app import app
from app.config import Config
import os

# Initialize data processors and simulators
data_processor = DataProcessor(Config)
signal_simulator = SignalSimulator(Config)
circuit_simulator = CircuitSimulator(Config)
control_simulator = ControlSimulator(Config)
statistical_analyzer = StatisticalAnalyzer(Config)
time_series_analyzer = TimeSeriesAnalyzer(Config)
frequency_analyzer = FrequencyAnalyzer(Config)
signal_modeler = SignalModeler(Config)
circuit_modeler = CircuitModeler(Config)
system_modeler = SystemModeler(Config)

# Callback for page content
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/login':
        return html.Div()
    elif pathname == '/':
        if current_user.is_authenticated:
            from app.components import create_layout
            return create_layout()
        else:
            return html.Div([
                html.H1("Please log in to access the dashboard"),
                dcc.Link('Login', href='/login')
            ])
    elif pathname == '/simulation':
        if current_user.is_authenticated and is_engineer(current_user):
            from app.components import create_simulation_layout
            return create_simulation_layout()
        else:
            return html.Div([
                html.H1("Access Denied"),
                html.P("You don't have permission to access the simulation module.")
            ])
    elif pathname == '/analysis':
        if current_user.is_authenticated and is_analyst(current_user):
            from app.components import create_analysis_layout
            return create_analysis_layout()
        else:
            return html.Div([
                html.H1("Access Denied"),
                html.P("You don't have permission to access the analysis module.")
            ])
    elif pathname == '/modeling':
        if current_user.is_authenticated and is_engineer(current_user):
            from app.components import create_modeling_layout
            return create_modeling_layout()
        else:
            return html.Div([
                html.H1("Access Denied"),
                html.P("You don't have permission to access the modeling module.")
            ])
    else:
        return html.Div([
            html.H1("404: Not Found"),
            html.P("The page you requested does not exist.")
        ])

# Callback for system metrics
@app.callback(
    [Output('system-metrics', 'children'),
     Output('top-processes', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_system_metrics(n):
    metrics = data_processor.get_system_metrics()
    top_processes = data_processor.get_top_processes()
    
    # Create metrics display
    metrics_display = html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("CPU Usage"),
                        html.H2(f"{metrics['cpu']:.1f}%")
                    ])
                ], color="primary", inverse=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Memory Usage"),
                        html.H2(f"{metrics['memory']:.1f}%")
                    ])
                ], color="success", inverse=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Disk Usage"),
                        html.H2(f"{metrics['disk']:.1f}%")
                    ])
                ], color="warning", inverse=True)
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Network"),
                        html.H5(f"Sent: {metrics['network_sent'] / (1024*1024):.1f} MB"),
                        html.H5(f"Recv: {metrics['network_recv'] / (1024*1024):.1f} MB")
                    ])
                ], color="info", inverse=True)
            ], width=3)
        ])
    ])
    
    # Create processes table
    processes_table = html.Table([
        html.Thead([
            html.Tr([
                html.Th("PID"),
                html.Th("Name"),
                html.Th("CPU %")
            ])
        ]),
        html.Tbody([
            html.Tr([
                html.Td(proc['pid']),
                html.Td(proc['name']),
                html.Td(f"{proc['cpu_percent']:.1f}%")
            ]) for proc in top_processes
        ])
    ], className="table table-dark")
    
    return metrics_display, processes_table

# Callback for simulation upload
@app.callback(
    Output('upload-status-simulation', 'children'),
    [Input('upload-simulation', 'contents')],
    [State('upload-simulation', 'filename')]
)
def update_simulation_upload_status(contents, filename):
    if contents is not None:
        try:
            # Decode the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save the file
            upload_path = os.path.join(Config.UPLOAD_FOLDER, 'simulation')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, filename)
            
            with open(file_path, 'wb') as f:
                f.write(decoded)
            
            # Process the file
            file_ext = filename.split('.')[-1]
            processed_data = data_processor.process_uploaded_file(file_path, file_ext)
            
            if processed_data is not None:
                # Load data into simulator
                if isinstance(processed_data, pd.DataFrame):
                    # Use the first numeric column
                    numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        signal_data = processed_data[numeric_columns[0]].values
                        signal_simulator.load_data(signal_data, name='uploaded')
                        return html.Div([
                            html.P(f"File '{filename}' uploaded and loaded successfully!"),
                            html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}")
                        ], className="text-success")
                elif isinstance(processed_data, np.ndarray):
                    signal_simulator.load_data(processed_data, name='uploaded')
                    return html.Div([
                        html.P(f"File '{filename}' uploaded and loaded successfully!"),
                        html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}")
                    ], className="text-success")
            
            return html.Div([
                html.P(f"Failed to process file '{filename}'"),
                html.P("Unsupported file format or data structure")
            ], className="text-danger")
        
        except Exception as e:
            return html.Div([
                html.P(f"Error processing file: {str(e)}")
            ], className="text-danger")
    
    return html.Div()

# Callback for simulation manual input
@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    [Input('generate-button-simulation', 'n_clicks')],
    [State('manual-input-type-simulation', 'value'),
     State('sample-rate-simulation', 'value'),
     State('duration-simulation', 'value'),
     State('amplitude-simulation', 'value'),
     State('frequency-simulation', 'value'),
     State('phase-simulation', 'value')],
    prevent_initial_call=True
)
def generate_simulation_signal(n_clicks, input_type, sample_rate, duration, amplitude, frequency, phase):
    if n_clicks:
        try:
            # Set simulator parameters
            signal_simulator.set_parameters(sample_rate=sample_rate, duration=duration)
            
            # Generate signal
            signal_id, t, y = signal_simulator.generate_signal(
                signal_type=input_type,
                frequency=frequency,
                amplitude=amplitude,
                phase=phase
            )
            
            # Store in session
            session_data = {
                'simulation_signal_id': signal_id,
                'simulation_t': t.tolist(),
                'simulation_y': y.tolist()
            }
            
            return session_data
        
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return {}
    
    return {}

# Callback for simulation run
@app.callback(
    Output('simulation-results', 'children'),
    [Input('run-simulation-button', 'n_clicks')],
    [State('simulation-type', 'value'),
     State('analysis-type', 'value'),
     State('signal-type', 'value'),
     State('circuit-type', 'value'),
     State('control-type', 'value'),
     State('simulation-params', 'value'),
     State('session-store', 'data')],
    prevent_initial_call=True
)
def run_simulation(n_clicks, sim_type, analysis_type, signal_type, circuit_type, control_type, params_json, session_data):
    if n_clicks:
        try:
            # Parse parameters
            if params_json:
                params = json.loads(params_json)
            else:
                params = {}
            
            results = []
            
            if sim_type == 'signal':
                # Signal simulation
                if 'simulation_signal_id' in session_data:
                    signal_id = session_data['simulation_signal_id']
                    t = np.array(session_data['simulation_t'])
                    y = np.array(session_data['simulation_y'])
                else:
                    # Generate default signal
                    signal_id, t, y = signal_simulator.generate_signal(signal_type)
                
                if analysis_type == 'time':
                    # Time domain analysis
                    fig = create_simulation_plots({'t': t, 'y': y}, 'signal')
                    results.append(dcc.Graph(figure=fig))
                
                elif analysis_type == 'frequency':
                    # Frequency domain analysis
                    analysis = signal_simulator.analyze_signal(signal_id)
                    fig = create_simulation_plots({
                        'frequencies': analysis['fft']['frequencies'],
                        'magnitude': analysis['fft']['magnitude']
                    }, 'spectrum')
                    results.append(dcc.Graph(figure=fig))
                
                elif analysis_type == 'modulation':
                    # Modulation analysis
                    carrier_freq = params.get('carrier_freq', 50)
                    modulation_type = params.get('modulation_type', 'am')
                    modulation_index = params.get('modulation_index', 0.5)
                    
                    # Modulate signal
                    mod_id, t, y_mod = signal_simulator.modulate_signal(
                        signal_id, carrier_freq, modulation_type, modulation_index
                    )
                    
                    # Plot original and modulated signals
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t, y=y,
                        mode='lines',
                        name='Original Signal',
                        line=dict(color='#00ccff', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=t, y=y_mod,
                        mode='lines',
                        name='Modulated Signal',
                        line=dict(color='#ff9900', width=2)
                    ))
                    fig.update_layout(
                        title="Signal Modulation",
                        xaxis_title="Time",
                        yaxis_title="Amplitude",
                        template=Config.CHART_TEMPLATE,
                        height=Config.CHART_HEIGHT,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    results.append(dcc.Graph(figure=fig))
                
                elif analysis_type == 'filtering':
                    # Filtering analysis
                    filter_type = params.get('filter_type', 'lowpass')
                    cutoff_freq = params.get('cutoff_freq', 10)
                    filter_order = params.get('filter_order', 5)
                    
                    # Apply filter
                    filtered_id, t, y_filtered = signal_simulator.apply_filter(
                        signal_id, filter_type, cutoff_freq, filter_order
                    )
                    
                    # Plot original and filtered signals
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=t, y=y,
                        mode='lines',
                        name='Original Signal',
                        line=dict(color='#00ccff', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=t, y=y_filtered,
                        mode='lines',
                        name='Filtered Signal',
                        line=dict(color='#ff9900', width=2)
                    ))
                    fig.update_layout(
                        title="Signal Filtering",
                        xaxis_title="Time",
                        yaxis_title="Amplitude",
                        template=Config.CHART_TEMPLATE,
                        height=Config.CHART_HEIGHT,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    results.append(dcc.Graph(figure=fig))
                
                else:
                    results.append(html.P(f"Unknown analysis type: {analysis_type}"))
            
            elif sim_type == 'circuit':
                # Circuit simulation
                R = params.get('R', 1000)  # 1kΩ
                L = params.get('L', 0.1)    # 0.1H
                C = params.get('C', 1e-6)   # 1μF
                input_type = params.get('input_type', 'step')
                
                if circuit_type == 'rc':
                    circuit_id, t, u, y, metrics = circuit_simulator.simulate_rc_circuit(
                        R, C, input_type=input_type
                    )
                elif circuit_type == 'rl':
                    circuit_id, t, u, y, metrics = circuit_simulator.simulate_rl_circuit(
                        R, L, input_type=input_type
                    )
                elif circuit_type == 'rlc':
                    circuit_id, t, u, y, metrics = circuit_simulator.simulate_rlc_circuit(
                        R, L, C, input_type=input_type
                    )
                elif circuit_type == 'diode':
                    circuit_id, t, u, y = circuit_simulator.simulate_diode_circuit(
                        R, input_type=input_type
                    )
                    metrics = {}
                elif circuit_type == 'bjt':
                    R1 = params.get('R1', 10000)  # 10kΩ
                    R2 = params.get('R2', 10000)  # 10kΩ
                    Rc = params.get('Rc', 1000)   # 1kΩ
                    Re = params.get('Re', 1000)   # 1kΩ
                    Vcc = params.get('Vcc', 5)     # 5V
                    beta = params.get('beta', 100)  # Beta value
                    
                    circuit_id, t, u, y = circuit_simulator.simulate_transistor_amplifier(
                        R1, R2, Rc, Re, Vcc, beta, input_type=input_type
                    )
                    metrics = {}
                else:
                    results.append(html.P(f"Unknown circuit type: {circuit_type}"))
                    return html.Div(results)
                
                # Plot input and output
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=t, y=u,
                    mode='lines',
                    name='Input',
                    line=dict(color='#00ccff', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=t, y=y,
                    mode='lines',
                    name='Output',
                    line=dict(color='#ff9900', width=2)
                ))
                fig.update_layout(
                    title=f"{circuit_type.upper()} Circuit Simulation",
                    xaxis_title="Time",
                    yaxis_title="Amplitude",
                    template=Config.CHART_TEMPLATE,
                    height=Config.CHART_HEIGHT,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                results.append(dcc.Graph(figure=fig))
                
                # Display metrics if available
                if metrics:
                    metrics_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(key),
                                html.Td(f"{value:.4f}")
                            ]) for key, value in metrics.items()
                        ])
                    ], className="table table-dark")
                    results.append(metrics_table)
            
            elif sim_type == 'control':
                # Control system simulation
                if control_type == 'first_order':
                    K = params.get('K', 1.0)
                    tau = params.get('tau', 1.0)
                    input_type = params.get('input_type', 'step')
                    
                    system_id, t, u, y, metrics = control_simulator.simulate_first_order_system(
                        K, tau, input_type=input_type
                    )
                
                elif control_type == 'second_order':
                    K = params.get('K', 1.0)
                    zeta = params.get('zeta', 0.5)
                    omega_n = params.get('omega_n', 1.0)
                    input_type = params.get('input_type', 'step')
                    
                    system_id, t, u, y, metrics = control_simulator.simulate_second_order_system(
                        K, zeta, omega_n, input_type=input_type
                    )
                
                elif control_type == 'pid':
                    Kp = params.get('Kp', 1.0)
                    Ki = params.get('Ki', 0.1)
                    Kd = params.get('Kd', 0.01)
                    plant_num = params.get('plant_num', [1.0])
                    plant_den = params.get('plant_den', [1.0, 1.0])
                    setpoint = params.get('setpoint', 1.0)
                    
                    system_id, t, r, u, y, e, metrics = control_simulator.simulate_pid_controller(
                        Kp, Ki, Kd, plant_num, plant_den, setpoint=setpoint
                    )
                
                else:
                    results.append(html.P(f"Unknown control system type: {control_type}"))
                    return html.Div(results)
                
                # Plot input and output
                fig = go.Figure()
                if control_type == 'pid':
                    fig.add_trace(go.Scatter(
                        x=t, y=r,
                        mode='lines',
                        name='Setpoint',
                        line=dict(color='#00ccff', width=2, dash='dash')
                    ))
                    fig.add_trace(go.Scatter(
                        x=t, y=y,
                        mode='lines',
                        name='Output',
                        line=dict(color='#ff9900', width=2)
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=t, y=u,
                        mode='lines',
                        name='Input',
                        line=dict(color='#00ccff', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=t, y=y,
                        mode='lines',
                        name='Output',
                        line=dict(color='#ff9900', width=2)
                    ))
                
                fig.update_layout(
                    title=f"{control_type.replace('_', ' ').title()} Control System",
                    xaxis_title="Time",
                    yaxis_title="Amplitude",
                    template=Config.CHART_TEMPLATE,
                    height=Config.CHART_HEIGHT,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                results.append(dcc.Graph(figure=fig))
                
                # Display metrics if available
                if metrics:
                    metrics_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Metric"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(key),
                                html.Td(f"{value:.4f}")
                            ]) for key, value in metrics.items()
                        ])
                    ], className="table table-dark")
                    results.append(metrics_table)
            
            else:
                results.append(html.P(f"Unknown simulation type: {sim_type}"))
            
            return html.Div(results)
        
        except Exception as e:
            return html.Div([
                html.P(f"Error running simulation: {str(e)}"),
                html.Pre(str(e))
            ], className="text-danger")
    
    return html.P("Click 'Run Simulation' to see results")

# Callback for analysis upload
@app.callback(
    Output('upload-status-analysis', 'children'),
    [Input('upload-analysis', 'contents')],
    [State('upload-analysis', 'filename')]
)
def update_analysis_upload_status(contents, filename):
    if contents is not None:
        try:
            # Decode the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save the file
            upload_path = os.path.join(Config.UPLOAD_FOLDER, 'analysis')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, filename)
            
            with open(file_path, 'wb') as f:
                f.write(decoded)
            
            # Process the file
            file_ext = filename.split('.')[-1]
            processed_data = data_processor.process_uploaded_file(file_path, file_ext)
            
            if processed_data is not None:
                # Load data into analyzer
                if isinstance(processed_data, pd.DataFrame):
                    statistical_analyzer.load_data(processed_data, name='uploaded')
                    time_series_analyzer.load_data(processed_data, name='uploaded')
                    frequency_analyzer.load_data(processed_data, name='uploaded')
                    
                    # Update column options
                    columns = [{'label': col, 'value': col} for col in processed_data.columns]
                    
                    return html.Div([
                        html.P(f"File '{filename}' uploaded and loaded successfully!"),
                        html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}"),
                        dcc.Store(id='analysis-columns-data', data=columns)
                    ], className="text-success")
                elif isinstance(processed_data, np.ndarray):
                    statistical_analyzer.load_data(processed_data, name='uploaded')
                    frequency_analyzer.load_data(processed_data, name='uploaded')
                    
                    return html.Div([
                        html.P(f"File '{filename}' uploaded and loaded successfully!"),
                        html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}")
                    ], className="text-success")
            
            return html.Div([
                html.P(f"Failed to process file '{filename}'"),
                html.P("Unsupported file format or data structure")
            ], className="text-danger")
        
        except Exception as e:
            return html.Div([
                html.P(f"Error processing file: {str(e)}")
            ], className="text-danger")
    
    return html.Div()

# Callback for analysis manual input
@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    [Input('generate-button-analysis', 'n_clicks')],
    [State('manual-input-type-analysis', 'value'),
     State('sample-rate-analysis', 'value'),
     State('duration-analysis', 'value'),
     State('amplitude-analysis', 'value'),
     State('frequency-analysis', 'value'),
     State('phase-analysis', 'value')],
    prevent_initial_call=True
)
def generate_analysis_signal(n_clicks, input_type, sample_rate, duration, amplitude, frequency, phase):
    if n_clicks:
        try:
            # Generate signal
            from app.utils import generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise
            
            if input_type == 'sine':
                t, y = generate_sine_wave(frequency, duration, sample_rate, amplitude, phase)
            elif input_type == 'square':
                t, y = generate_square_wave(frequency, duration, sample_rate, amplitude)
            elif input_type == 'sawtooth':
                t, y = generate_sawtooth_wave(frequency, duration, sample_rate, amplitude)
            elif input_type == 'noise':
                t, y = generate_noise(duration, sample_rate, amplitude)
            else:
                return {}
            
            # Store in session
            session_data = {
                'analysis_signal_t': t.tolist(),
                'analysis_signal_y': y.tolist()
            }
            
            return session_data
        
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return {}
    
    return {}

# Callback for analysis run
@app.callback(
    Output('analysis-results', 'children'),
    [Input('run-analysis-button', 'n_clicks')],
    [State('analysis-type-analysis', 'value'),
     State('analysis-method', 'value'),
     State('data-column', 'value'),
     State('secondary-column', 'value'),
     State('analysis-params', 'value'),
     State('session-store', 'data')],
    prevent_initial_call=True
)
def run_analysis(n_clicks, analysis_type, method, data_column, secondary_column, params_json, session_data):
    if n_clicks:
        try:
            # Parse parameters
            if params_json:
                params = json.loads(params_json)
            else:
                params = {}
            
            results = []
            
            if analysis_type == 'statistical':
                # Statistical analysis
                if 'analysis_signal_t' in session_data and 'analysis_signal_y' in session_data:
                    # Use manually generated signal
                    data = np.array(session_data['analysis_signal_y'])
                    dataset_name = statistical_analyzer.load_data(data, name='manual')
                else:
                    # Use uploaded data
                    dataset_name = 'uploaded'
                
                if method == 'basic_stats':
                    stats = statistical_analyzer.basic_statistics(dataset_name)
                    
                    # Create statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Statistic"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(key),
                                html.Td(f"{value:.4f}")
                            ]) for key, value in stats.items()
                        ])
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'correlation':
                    corr_matrix = statistical_analyzer.correlation_analysis(dataset_name)
                    
                    # Create correlation heatmap
                    fig = create_analysis_plots({'corr_matrix': corr_matrix}, 'correlation')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'hypothesis':
                    test_type = params.get('test_type', 'ttest_1samp')
                    alpha = params.get('alpha', 0.05)
                    
                    result = statistical_analyzer.hypothesis_testing(
                        dataset_name, test_type, alpha=alpha
                    )
                    
                    # Create result table
                    result_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Property"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Test Type"), html.Td(result['test_type'])]),
                            html.Tr([html.Td("Statistic"), html.Td(f"{result['statistic']:.4f}")]),
                            html.Tr([html.Td("P-value"), html.Td(f"{result['pvalue']:.4f}")]),
                            html.Tr([html.Td("Conclusion"), html.Td(result['conclusion'])])
                        ])
                    ], className="table table-dark")
                    results.append(result_table)
                
                elif method == 'regression':
                    regression_type = params.get('regression_type', 'linear')
                    degree = params.get('degree', 2)
                    
                    result = statistical_analyzer.regression_analysis(
                        dataset_name, regression_type=regression_type, degree=degree
                    )
                    
                    # Create regression plot
                    fig = create_analysis_plots({
                        'x': result['x'],
                        'y': result['y'],
                        'y_pred': result['y_pred']
                    }, 'regression')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create regression statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Statistic"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{result['r_squared']:.4f}")]),
                            html.Tr([html.Td("Equation"), html.Td(result['equation'])])
                        ])
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'distribution':
                    result = statistical_analyzer.distribution_analysis(dataset_name)
                    
                    # Create distribution plot
                    fig = create_analysis_plots({
                        'sample': result['sample'],
                        'distributions': result['distributions']
                    }, 'distribution')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create best fit table
                    best_fit_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Property"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Best Fit"), html.Td(result['best_fit'])]),
                            html.Tr([html.Td("KS Statistic"), html.Td(f"{result['best_ks']:.4f}")])
                        ])
                    ], className="table table-dark")
                    results.append(best_fit_table)
                
                elif method == 'outlier':
                    method_type = params.get('outlier_method', 'iqr')
                    threshold = params.get('threshold', 1.5)
                    
                    result = statistical_analyzer.outlier_detection(
                        dataset_name, method=method_type, threshold=threshold
                    )
                    
                    # Create outlier plot
                    fig = create_analysis_plots({
                        'data': result['outliers']
                    }, 'histogram')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create outlier statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Statistic"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Method"), html.Td(result['method'])]),
                            html.Tr([html.Td("Threshold"), html.Td(result['threshold'])]),
                            html.Tr([html.Td("Outlier Count"), html.Td(result['outlier_count'])])
                        ])
                    ], className="table table-dark")
                    results.append(stats_table)
                
                else:
                    results.append(html.P(f"Unknown statistical method: {method}"))
            
            elif analysis_type == 'time_series':
                # Time series analysis
                if 'analysis_signal_t' in session_data and 'analysis_signal_y' in session_data:
                    # Use manually generated signal
                    t = np.array(session_data['analysis_signal_t'])
                    y = np.array(session_data['analysis_signal_y'])
                    dataset_name = time_series_analyzer.load_data(y, name='manual', time_index=pd.date_range(start='2020-01-01', periods=len(y), freq='D'))
                else:
                    # Use uploaded data
                    dataset_name = 'uploaded'
                
                if method == 'stationarity':
                    result = time_series_analyzer.check_stationarity(dataset_name)
                    
                    # Create result table
                    result_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Statistic"),
                                html.Td("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("ADF Statistic"), html.Td(f"{result['adf_statistic']:.4f}")]),
                            html.Tr([html.Td("P-value"), html.Td(f"{result['p_value']:.4f}")]),
                            html.Tr([html.Td("Conclusion"), html.Td(result['conclusion'])])
                        ])
                    ], className="table table-dark")
                    results.append(result_table)
                
                elif method == 'decomposition':
                    model = params.get('model', 'additive')
                    period = params.get('period', None)
                    
                    result = time_series_analyzer.decompose_time_series(
                        dataset_name, model=model, period=period
                    )
                    
                    # Create decomposition plot
                    fig = create_analysis_plots({
                        'original': result['original'],
                        'trend': result['trend'],
                        'seasonal': result['seasonal'],
                        'residual': result['residual']
                    }, 'decomposition')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'autocorrelation':
                    lags = params.get('lags', 40)
                    
                    result = time_series_analyzer.autocorrelation_analysis(dataset_name, lags=lags)
                    
                    # Create ACF plot
                    fig = create_analysis_plots({
                        'acf_values': result['acf']['values'],
                        'confint': result['acf']['confint']
                    }, 'autocorrelation')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create PACF plot
                    fig = create_analysis_plots({
                        'pacf_values': result['pacf']['values'],
                        'confint': result['pacf']['confint']
                    }, 'pacf')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'arima':
                    order = params.get('order', [1, 1, 1])
                    seasonal_order = params.get('seasonal_order', None)
                    
                    result = time_series_analyzer.fit_arima_model(
                        dataset_name, order=order, seasonal_order=seasonal_order
                    )
                    
                    # Create time series plot with predictions
                    fig = create_analysis_plots({
                        'series': result['original']
                    }, 'time_series')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model summary
                    summary_div = html.Div([
                        html.H5("ARIMA Model Summary"),
                        html.Pre(result['summary'])
                    ], className="mt-3")
                    results.append(summary_div)
                
                elif method == 'anomaly':
                    method_type = params.get('anomaly_method', 'zscore')
                    threshold = params.get('threshold', 3.0)
                    
                    result = time_series_analyzer.detect_anomalies(
                        dataset_name, method=method_type, threshold=threshold
                    )
                    
                    # Create time series plot with anomalies
                    fig = go.Figure()
                    dataset = time_series_analyzer.get_dataset(dataset_name)
                    series = dataset['data']
                    
                    fig.add_trace(go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode='lines',
                        name='Time Series',
                        line=dict(color='#00ccff', width=2)
                    ))
                    
                    if len(result['anomalies']) > 0:
                        fig.add_trace(go.Scatter(
                            x=result['anomaly_indices'],
                            y=result['anomalies'],
                            mode='markers',
                            name='Anomalies',
                            marker=dict(color='red', size=8)
                        ))
                    
                    fig.update_layout(
                        title="Time Series with Anomalies",
                        xaxis_title="Time",
                        yaxis_title="Value",
                        template=Config.CHART_TEMPLATE,
                        height=Config.CHART_HEIGHT,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create anomaly statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Statistic"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("Method"), html.Td(result['method'])]),
                            html.Tr([html.Td("Threshold"), html.Td(result['threshold'])]),
                            html.Tr([html.Td("Anomaly Count"), html.Td(result['anomaly_count'])])
                        ])
                    ], className="table table-dark")
                    results.append(stats_table)
                
                else:
                    results.append(html.P(f"Unknown time series method: {method}"))
            
            elif analysis_type == 'frequency':
                # Frequency analysis
                if 'analysis_signal_t' in session_data and 'analysis_signal_y' in session_data:
                    # Use manually generated signal
                    t = np.array(session_data['analysis_signal_t'])
                    y = np.array(session_data['analysis_signal_y'])
                    sample_rate = params.get('sample_rate', 1000)
                    dataset_name = frequency_analyzer.load_data(y, name='manual', sample_rate=sample_rate)
                else:
                    # Use uploaded data
                    dataset_name = 'uploaded'
                
                if method == 'fft':
                    window = params.get('window', 'hann')
                    detrend = params.get('detrend', 'constant')
                    
                    result = frequency_analyzer.fft_analysis(
                        dataset_name, window=window, detrend=detrend
                    )
                    
                    # Create FFT plot
                    fig = create_analysis_plots({
                        'frequencies': result['frequencies'],
                        'magnitude': result['magnitude']
                    }, 'spectrum')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create peak frequencies table
                    peaks_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Peak"),
                                html.Th("Frequency (Hz)"),
                                html.Th("Magnitude")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(i+1),
                                html.Td(f"{freq:.2f}"),
                                html.Td(f"{mag:.4f}")
                            ]) for i, (freq, mag) in enumerate(zip(
                                result['peak_frequencies'][:5],
                                result['peak_magnitudes'][:5]
                            ))
                        ])
                    ], className="table table-dark")
                    results.append(peaks_table)
                
                elif method == 'power_spectrum':
                    method_type = params.get('psd_method', 'welch')
                    window = params.get('window', 'hann')
                    nperseg = params.get('nperseg', None)
                    
                    result = frequency_analyzer.power_spectrum_analysis(
                        dataset_name, method=method_type, window=window, nperseg=nperseg
                    )
                    
                    # Create PSD plot
                    fig = create_analysis_plots({
                        'frequencies': result['frequencies'],
                        'psd': result['power_db']
                    }, 'psd')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'spectrogram':
                    window = params.get('window', 'hann')
                    nperseg = params.get('nperseg', None)
                    noverlap = params.get('noverlap', None)
                    
                    result = frequency_analyzer.spectrogram_analysis(
                        dataset_name, window=window, nperseg=nperseg, noverlap=noverlap
                    )
                    
                    # Create spectrogram plot
                    fig = create_analysis_plots({
                        'f': result['frequencies'],
                        't': result['times'],
                        'Sxx_db': result['spectrogram_db']
                    }, 'spectrogram')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'coherence':
                    window = params.get('window', 'hann')
                    nperseg = params.get('nperseg', None)
                    
                    result = frequency_analyzer.coherence_analysis(
                        dataset_name, window=window, nperseg=nperseg
                    )
                    
                    # Create coherence plot
                    fig = create_analysis_plots({
                        'frequencies': result['frequencies'],
                        'coherence': result['coherence']
                    }, 'coherence')
                    results.append(dcc.Graph(figure=fig))
                
                elif method == 'hilbert':
                    result = frequency_analyzer.hilbert_transform_analysis(dataset_name)
                    
                    # Create amplitude envelope plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result['time'],
                        y=result['original_signal'],
                        mode='lines',
                        name='Original Signal',
                        line=dict(color='#00ccff', width=2)
                    ))
                    fig.add_trace(go.Scatter(
                        x=result['time'],
                        y=result['amplitude_envelope'],
                        mode='lines',
                        name='Amplitude Envelope',
                        line=dict(color='#ff9900', width=2)
                    ))
                    fig.update_layout(
                        title="Hilbert Transform - Amplitude Envelope",
                        xaxis_title="Time",
                        yaxis_title="Amplitude",
                        template=Config.CHART_TEMPLATE,
                        height=Config.CHART_HEIGHT,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create instantaneous frequency plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=result['time'][1:],  # Skip first point as diff reduces length by 1
                        y=result['instantaneous_frequency'],
                        mode='lines',
                        name='Instantaneous Frequency',
                        line=dict(color='#ff3399', width=2)
                    ))
                    fig.update_layout(
                        title="Hilbert Transform - Instantaneous Frequency",
                        xaxis_title="Time",
                        yaxis_title="Frequency (Hz)",
                        template=Config.CHART_TEMPLATE,
                        height=Config.CHART_HEIGHT,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    results.append(dcc.Graph(figure=fig))
                
                else:
                    results.append(html.P(f"Unknown frequency analysis method: {method}"))
            
            else:
                results.append(html.P(f"Unknown analysis type: {analysis_type}"))
            
            return html.Div(results)
        
        except Exception as e:
            return html.Div([
                html.P(f"Error running analysis: {str(e)}"),
                html.Pre(str(e))
            ], className="text-danger")
    
    return html.P("Click 'Run Analysis' to see results")

# Callback for modeling upload
@app.callback(
    Output('upload-status-modeling', 'children'),
    [Input('upload-modeling', 'contents')],
    [State('upload-modeling', 'filename')]
)
def update_modeling_upload_status(contents, filename):
    if contents is not None:
        try:
            # Decode the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Save the file
            upload_path = os.path.join(Config.UPLOAD_FOLDER, 'modeling')
            os.makedirs(upload_path, exist_ok=True)
            file_path = os.path.join(upload_path, filename)
            
            with open(file_path, 'wb') as f:
                f.write(decoded)
            
            # Process the file
            file_ext = filename.split('.')[-1]
            processed_data = data_processor.process_uploaded_file(file_path, file_ext)
            
            if processed_data is not None:
                # Load data into modeler
                if isinstance(processed_data, pd.DataFrame):
                    signal_modeler.load_data(processed_data, name='uploaded')
                    circuit_modeler.load_data(processed_data, name='uploaded')
                    system_modeler.load_data(processed_data, name='uploaded')
                    
                    # Update column options
                    columns = [{'label': col, 'value': col} for col in processed_data.columns]
                    
                    return html.Div([
                        html.P(f"File '{filename}' uploaded and loaded successfully!"),
                        html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}"),
                        dcc.Store(id='modeling-columns-data', data=columns)
                    ], className="text-success")
                elif isinstance(processed_data, np.ndarray):
                    signal_modeler.load_data(processed_data, name='uploaded')
                    system_modeler.load_data(processed_data, name='uploaded')
                    
                    return html.Div([
                        html.P(f"File '{filename}' uploaded and loaded successfully!"),
                        html.P(f"Data type: {file_ext}, Shape: {processed_data.shape}")
                    ], className="text-success")
            
            return html.Div([
                html.P(f"Failed to process file '{filename}'"),
                html.P("Unsupported file format or data structure")
            ], className="text-danger")
        
        except Exception as e:
            return html.Div([
                html.P(f"Error processing file: {str(e)}")
            ], className="text-danger")
    
    return html.Div()

# Callback for modeling manual input
@app.callback(
    Output('session-store', 'data', allow_duplicate=True),
    [Input('generate-button-modeling', 'n_clicks')],
    [State('manual-input-type-modeling', 'value'),
     State('sample-rate-modeling', 'value'),
     State('duration-modeling', 'value'),
     State('amplitude-modeling', 'value'),
     State('frequency-modeling', 'value'),
     State('phase-modeling', 'value')],
    prevent_initial_call=True
)
def generate_modeling_signal(n_clicks, input_type, sample_rate, duration, amplitude, frequency, phase):
    if n_clicks:
        try:
            # Generate signal
            from app.utils import generate_sine_wave, generate_square_wave, generate_sawtooth_wave, generate_noise
            
            if input_type == 'sine':
                t, y = generate_sine_wave(frequency, duration, sample_rate, amplitude, phase)
            elif input_type == 'square':
                t, y = generate_square_wave(frequency, duration, sample_rate, amplitude)
            elif input_type == 'sawtooth':
                t, y = generate_sawtooth_wave(frequency, duration, sample_rate, amplitude)
            elif input_type == 'noise':
                t, y = generate_noise(duration, sample_rate, amplitude)
            else:
                return {}
            
            # Store in session
            session_data = {
                'modeling_signal_t': t.tolist(),
                'modeling_signal_y': y.tolist()
            }
            
            return session_data
        
        except Exception as e:
            print(f"Error generating signal: {str(e)}")
            return {}
    
    return {}

# Callback for modeling fit
@app.callback(
    Output('modeling-results', 'children'),
    [Input('fit-model-button', 'n_clicks')],
    [State('model-type', 'value'),
     State('model-method', 'value'),
     State('input-column', 'value'),
     State('output-column', 'value'),
     State('modeling-params', 'value'),
     State('session-store', 'data')],
    prevent_initial_call=True
)
def fit_model(n_clicks, model_type, method, input_column, output_column, params_json, session_data):
    if n_clicks:
        try:
            # Parse parameters
            if params_json:
                params = json.loads(params_json)
            else:
                params = {}
            
            results = []
            
            if model_type == 'signal':
                # Signal modeling
                if 'modeling_signal_t' in session_data and 'modeling_signal_y' in session_data:
                    # Use manually generated signal
                    t = np.array(session_data['modeling_signal_t'])
                    y = np.array(session_data['modeling_signal_y'])
                else:
                    # Use uploaded data
                    dataset = signal_modeler.get_dataset('uploaded')
                    if dataset['type'] == 'array':
                        t = np.arange(len(dataset['data']))
                        y = dataset['data']
                    elif dataset['type'] == 'dataframe':
                        if input_column is not None and output_column is not None:
                            t = dataset['data'][input_column].values
                            y = dataset['data'][output_column].values
                        else:
                            # Use first two columns
                            t = dataset['data'].iloc[:, 0].values
                            y = dataset['data'].iloc[:, 1].values
                    else:
                        raise ValueError("Unsupported data type for signal modeling")
                
                if method == 'sinusoidal':
                    model_type_sine = params.get('sinusoidal_type', 'single_sine')
                    
                    model_id, model_params, r_squared = signal_modeler.fit_sinusoidal_model(
                        t, y, model_type=model_type_sine
                    )
                    
                    # Get model
                    model = signal_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y_original'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'exponential':
                    model_type_exp = params.get('exponential_type', 'single_exp')
                    
                    model_id, model_params, r_squared = signal_modeler.fit_exponential_model(
                        t, y, model_type=model_type_exp
                    )
                    
                    # Get model
                    model = signal_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y_original'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'polynomial':
                    degree = params.get('degree', 2)
                    
                    model_id, model_params, r_squared = signal_modeler.fit_polynomial_model(
                        t, y, degree=degree
                    )
                    
                    # Get model
                    model = signal_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y_original'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'fourier':
                    n_harmonics = params.get('n_harmonics', 3)
                    
                    model_id, model_params, r_squared = signal_modeler.fit_fourier_model(
                        t, y, n_harmonics=n_harmonics
                    )
                    
                    # Get model
                    model = signal_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y_original'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                else:
                    results.append(html.P(f"Unknown signal modeling method: {method}"))
            
            elif model_type == 'circuit':
                # Circuit modeling
                if 'modeling_signal_t' in session_data and 'modeling_signal_y' in session_data:
                    # Use manually generated signal
                    t = np.array(session_data['modeling_signal_t'])
                    v_in = np.ones_like(t)  # Assume step input
                    v_out = np.array(session_data['modeling_signal_y'])
                else:
                    # Use uploaded data
                    dataset = circuit_modeler.get_dataset('uploaded')
                    if dataset['type'] == 'array':
                        t = np.arange(len(dataset['data']))
                        v_in = np.ones_like(t)  # Assume step input
                        v_out = dataset['data']
                    elif dataset['type'] == 'dataframe':
                        if input_column is not None and output_column is not None:
                            t = dataset['data'][input_column].values
                            v_in = np.ones_like(t)  # Assume step input
                            v_out = dataset['data'][output_column].values
                        else:
                            # Use first two columns
                            t = dataset['data'].iloc[:, 0].values
                            v_in = np.ones_like(t)  # Assume step input
                            v_out = dataset['data'].iloc[:, 1].values
                    else:
                        raise ValueError("Unsupported data type for circuit modeling")
                
                if method == 'rc':
                    model_id, model_params, r_squared = circuit_modeler.model_rc_circuit(t, v_in, v_out)
                    
                    # Get model
                    model = circuit_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['v_out'],
                        'y_fit': model['v_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'rl':
                    model_id, model_params, r_squared = circuit_modeler.model_rl_circuit(t, v_in, v_out)
                    
                    # Get model
                    model = circuit_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['i_out'],
                        'y_fit': model['i_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'rlc':
                    model_id, model_params, r_squared = circuit_modeler.model_rlc_circuit(t, v_in, v_out)
                    
                    # Get model
                    model = circuit_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['v_out'],
                        'y_fit': model['v_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'diode':
                    model_id, model_params, r_squared = circuit_modeler.model_diode_circuit(v_in, v_out)
                    
                    # Get model
                    model = circuit_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': np.arange(len(model['v_in'])),
                        'y_original': model['i_out'],
                        'y_fit': model['i_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4e}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'bjt':
                    model_id, model_params, r_squared = circuit_modeler.model_bjt_amplifier(v_in, v_out)
                    
                    # Get model
                    model = circuit_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': np.arange(len(model['v_in'])),
                        'y_original': model['v_out'],
                        'y_fit': model['v_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                else:
                    results.append(html.P(f"Unknown circuit modeling method: {method}"))
            
            elif model_type == 'system':
                # System modeling
                if 'modeling_signal_t' in session_data and 'modeling_signal_y' in session_data:
                    # Use manually generated signal
                    t = np.array(session_data['modeling_signal_t'])
                    u = np.ones_like(t)  # Assume step input
                    y = np.array(session_data['modeling_signal_y'])
                else:
                    # Use uploaded data
                    dataset = system_modeler.get_dataset('uploaded')
                    if dataset['type'] == 'array':
                        t = np.arange(len(dataset['data']))
                        u = np.ones_like(t)  # Assume step input
                        y = dataset['data']
                    elif dataset['type'] == 'dataframe':
                        if input_column is not None and output_column is not None:
                            t = dataset['data'][input_column].values
                            u = np.ones_like(t)  # Assume step input
                            y = dataset['data'][output_column].values
                        else:
                            # Use first two columns
                            t = dataset['data'].iloc[:, 0].values
                            u = np.ones_like(t)  # Assume step input
                            y = dataset['data'].iloc[:, 1].values
                    else:
                        raise ValueError("Unsupported data type for system modeling")
                
                if method == 'first_order':
                    model_id, model_params, r_squared = system_modeler.model_first_order_system(t, u, y)
                    
                    # Get model
                    model = system_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'second_order':
                    model_id, model_params, r_squared = system_modeler.model_second_order_system(t, u, y)
                    
                    # Get model
                    model = system_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'pid':
                    r = np.ones_like(t)  # Setpoint
                    model_id, model_params, r_squared = system_modeler.model_pid_controller(t, r, u, y)
                    
                    # Get model
                    model = system_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")])
                        ] + [
                            html.Tr([html.Td(key), html.Td(f"{value:.4f}")]) 
                            for key, value in model_params.items()
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'state_space':
                    n_states = params.get('n_states', 2)
                    
                    model_id, model_params, r_squared = system_modeler.model_state_space_system(t, u, y, n_states=n_states)
                    
                    # Get model
                    model = system_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create pole-zero plot
                    poles = np.array(model['system_params']['poles'])
                    zeros = np.array([0.0] * len(poles))  # No zeros for simplicity
                    
                    fig = create_modeling_plots({
                        'poles': poles,
                        'zeros': zeros
                    }, 'pole_zero')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")]),
                            html.Tr([html.Td("Is Stable"), html.Td("Yes" if model['system_params']['is_stable'] else "No")])
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                elif method == 'transfer_function':
                    order_num = params.get('order_num', 2)
                    order_den = params.get('order_den', 2)
                    
                    model_id, model_params, r_squared = system_modeler.model_transfer_function(t, u, y, order_num, order_den)
                    
                    # Get model
                    model = system_modeler.get_model(model_id)
                    
                    # Create model fit plot
                    fig = create_modeling_plots({
                        't': model['t'],
                        'y_original': model['y'],
                        'y_fit': model['y_fit']
                    }, 'model_fit')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create pole-zero plot
                    poles = np.array(model['system_params']['poles'])
                    zeros = np.array(model['system_params']['zeros'])
                    
                    fig = create_modeling_plots({
                        'poles': poles,
                        'zeros': zeros
                    }, 'pole_zero')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create Bode plot
                    # Generate frequency range
                    frequencies = np.logspace(-2, 2, 1000)
                    # Create transfer function
                    from scipy import signal
                    sys = signal.TransferFunction(model_params['num'], model_params['den'])
                    # Calculate frequency response
                    w, mag, phase = signal.bode(sys, frequencies)
                    
                    fig = create_modeling_plots({
                        'frequencies': w,
                        'magnitude': 10**(mag/20),  # Convert from dB to linear
                        'phase': phase
                    }, 'bode')
                    results.append(dcc.Graph(figure=fig))
                    
                    # Create model statistics table
                    stats_table = html.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Parameter"),
                                html.Th("Value")
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([html.Td("R-squared"), html.Td(f"{r_squared:.4f}")]),
                            html.Tr([html.Td("Is Stable"), html.Td("Yes" if model['system_params']['is_stable'] else "No")])
                        ]
                    ], className="table table-dark")
                    results.append(stats_table)
                
                else:
                    results.append(html.P(f"Unknown system modeling method: {method}"))
            
            else:
                results.append(html.P(f"Unknown model type: {model_type}"))
            
            return html.Div(results)
        
        except Exception as e:
            return html.Div([
                html.P(f"Error fitting model: {str(e)}"),
                html.Pre(str(e))
            ], className="text-danger")
    
    return html.P("Click 'Fit Model' to see results")

# Callback to update dropdown options based on uploaded data
@app.callback(
    [Output('data-column', 'options'),
     Output('secondary-column', 'options'),
     Output('input-column', 'options'),
     Output('output-column', 'options')],
    [Input('upload-status-analysis', 'children'),
     Input('upload-status-modeling', 'children')]
)
def update_dropdown_options(analysis_status, modeling_status):
    # Default empty options
    options = []
    
    # Check if we have uploaded data for analysis
    if analysis_status and 'analysis-columns-data' in analysis_status:
        # Extract column options from analysis status
        # This is a simplified approach - in a real app, you'd use dcc.Store
        pass
    
    # Check if we have uploaded data for modeling
    if modeling_status and 'modeling-columns-data' in modeling_status:
        # Extract column options from modeling status
        # This is a simplified approach - in a real app, you'd use dcc.Store
        pass
    
    return options, options, options, options