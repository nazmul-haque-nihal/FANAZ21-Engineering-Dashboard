import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import base64
import io
from datetime import datetime as dt
import json
from scipy import optimize, signal, stats
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Initialize the Dash app with a dark theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY, 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css'],
    meta_tags=[{'name': 'viewport', 'content': 'width=device-width, initial-scale=1.0'}],
    suppress_callback_exceptions=True
)
app.title = "FANAZ21 Engineering Dashboard"

# Create the layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    
    # Header
    html.Div([
        html.Div([
            html.H1("FANAZ21 Engineering Dashboard", className="display-4"),
            html.P("Simulation, Modeling & Multi-Purpose Engineering Analysis", className="lead")
        ], className="container"),
    ], className="bg-primary text-white p-4 mb-4"),
    
    # Main content
    dbc.Container([
        dbc.Row([
            # Sidebar
            dbc.Col([
                html.Div([
                    html.H4("Navigation", className="text-center mb-4"),
                    dbc.Nav([
                        dbc.NavItem(dbc.NavLink("Dashboard", href="/", active="exact", className="text-light")),
                        dbc.NavItem(dbc.NavLink("Simulation", href="/simulation", active="exact", className="text-light")),
                        dbc.NavItem(dbc.NavLink("Modeling", href="/modeling", active="exact", className="text-light")),
                        dbc.NavItem(dbc.NavLink("Analysis", href="/analysis", active="exact", className="text-light")),
                        dbc.NavItem(dbc.NavLink("Data Upload", href="/upload", active="exact", className="text-light")),
                    ], vertical=True, pills=True, className="bg-dark p-3 rounded"),
                    
                    html.Hr(className="my-4"),
                    
                    html.H4("Settings", className="text-center mb-4"),
                    html.Div([
                        html.Label("Theme", className="text-light"),
                        dcc.Dropdown(
                            id='theme-dropdown',
                            options=[
                                {'label': 'Dark', 'value': 'dark'},
                                {'label': 'Light', 'value': 'light'}
                            ],
                            value='dark',
                            className="text-success bg-dark"
                        ),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Update Interval (s)", className="text-light"),
                        dcc.Slider(
                            id='update-interval',
                            min=1,
                            max=10,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(1, 11)},
                            className="text-light"
                        ),
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Project Type", className="text-light"),
                        dcc.Dropdown(
                            id='project-type-dropdown',
                            options=[
                                {'label': 'Electrical Engineering', 'value': 'electrical'},
                                {'label': 'Electronics Engineering', 'value': 'electronics'},
                                {'label': 'Control Systems', 'value': 'control'},
                                {'label': 'Power Systems', 'value': 'power'},
                                {'label': 'Telecommunications', 'value': 'telecom'}
                            ],
                            value='electrical',
                            className="text-success bg-dark"
                        ),
                    ], className="mb-4"),
                    
                    html.Div([
                        dbc.Button("Export Data", id="export-data-btn", color="success", className="me-2"),
                        dbc.Button("Export Charts", id="export-charts-btn", color="success"),
                    ], className="d-grid gap-2"),
                ], className="sticky-top p-3 bg-secondary rounded")
            ], width=3),
            
            # Main content area
            dbc.Col([
                html.Div(id='page-content', className="p-3 bg-dark rounded")
            ], width=9),
        ]),
    ], fluid=True),
    
    # Interval component for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=3*1000,  # in milliseconds
        n_intervals=0
    ),
    
    # Hidden div to store data
    html.Div(id='hidden-data', style={'display': 'none'}),
    
    # Footer
    html.Footer([
        html.Div([
            html.P(f"© {dt.now().year} FANAZ21 Engineering Dashboard. All rights reserved.", className="text-center mb-0")
        ], className="container")
    ], className="bg-primary text-white p-3 mt-4")
], className="bg-dark")

# Function to create dashboard layout
def create_dashboard_layout():
    return html.Div([
        html.H2("Engineering Dashboard", className="text-center mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Status"),
                    dbc.CardBody([
                        html.Div(id='system-status')
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Real-time Data"),
                    dbc.CardBody([
                        dcc.Graph(id='real-time-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Performance Metrics"),
                    dbc.CardBody([
                        html.Div(id='performance-metrics')
                    ])
                ], color="dark", inverse=True)
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Recent Activities"),
                    dbc.CardBody([
                        html.Div(id='recent-activities')
                    ])
                ], color="dark", inverse=True)
            ], width=6),
        ], className="mb-4"),
    ])

# Function to create simulation layout
def create_simulation_layout():
    return html.Div([
        html.H2("Simulation Module", className="text-center mb-4"),
        
        dbc.Tabs([
            dbc.Tab(label="Signal Simulation", tab_id="signal-sim"),
            dbc.Tab(label="Circuit Simulation", tab_id="circuit-sim"),
            dbc.Tab(label="Control System Simulation", tab_id="control-sim"),
        ], id="simulation-tabs", active_tab="signal-sim", className="mb-4"),
        
        html.Div(id='simulation-content')
    ])

# Function to create modeling layout
def create_modeling_layout():
    return html.Div([
        html.H2("Modeling Module", className="text-center mb-4"),
        
        dbc.Tabs([
            dbc.Tab(label="Signal Modeling", tab_id="signal-mod"),
            dbc.Tab(label="Circuit Modeling", tab_id="circuit-mod"),
            dbc.Tab(label="System Modeling", tab_id="system-mod"),
        ], id="modeling-tabs", active_tab="signal-mod", className="mb-4"),
        
        html.Div(id='modeling-content')
    ])

# Function to create analysis layout
def create_analysis_layout():
    return html.Div([
        html.H2("Analysis Module", className="text-center mb-4"),
        
        dbc.Tabs([
            dbc.Tab(label="Statistical Analysis", tab_id="stat-analysis"),
            dbc.Tab(label="Time Series Analysis", tab_id="time-analysis"),
            dbc.Tab(label="Frequency Analysis", tab_id="freq-analysis"),
        ], id="analysis-tabs", active_tab="stat-analysis", className="mb-4"),
        
        html.Div(id='analysis-content')
    ])

# Function to create upload layout
def create_upload_layout():
    return html.Div([
        html.H2("Data Upload", className="text-center mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Upload Data File"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                html.A('Drag and Drop or ', style={'color': '#4CAF50'}),
                                html.A('Select Files', style={'color': '#4CAF50', 'text-decoration': 'underline'})
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px 0',
                                'backgroundColor': '#1E1E1E',
                                'color': 'white'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-output'),
                    ])
                ], color="dark", inverse=True)
            ], width=6),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Manual Data Input"),
                    dbc.CardBody([
                        dcc.Textarea(
                            id='manual-input',
                            placeholder='Enter data manually (comma separated values)',
                            style={'width': '100%', 'height': 200, 'backgroundColor': '#1E1E1E', 'color': 'white'}
                        ),
                        html.Br(),
                        dbc.Button("Process Data", id="process-manual-btn", color="success", className="me-2"),
                        dbc.Button("Clear", id="clear-manual-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Data Preview"),
                    dbc.CardBody([
                        html.Div(id='data-preview')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

# Navigation callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/simulation':
        return create_simulation_layout()
    elif pathname == '/modeling':
        return create_modeling_layout()
    elif pathname == '/analysis':
        return create_analysis_layout()
    elif pathname == '/upload':
        return create_upload_layout()
    else:
        return create_dashboard_layout()

# Dashboard callbacks
@app.callback(
    [Output('system-status', 'children'),
     Output('real-time-graph', 'figure'),
     Output('performance-metrics', 'children'),
     Output('recent-activities', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('project-type-dropdown', 'value')]
)
def update_dashboard(n, project_type):
    try:
        # System status
        status = dbc.ListGroup([
            dbc.ListGroupItem("CPU Usage: 45%", color="success"),
            dbc.ListGroupItem("Memory Usage: 60%", color="warning"),
            dbc.ListGroupItem("Disk Usage: 35%", color="success"),
            dbc.ListGroupItem("Network: Active", color="success"),
        ], flush=True)
        
        # Real-time graph
        np.random.seed(n)  # Change seed with each update for different data
        time = np.linspace(0, 10, 100)
        
        if project_type == 'electrical':
            # Simulate electrical power consumption
            base_value = 1000  # Base power in watts
            fluctuation = np.random.normal(0, 50, len(time))  # Random fluctuation
            daily_pattern = 200 * np.sin(2 * np.pi * np.arange(len(time)) / len(time))  # Daily pattern
            values = base_value + fluctuation + daily_pattern
            y_title = "Power (Watts)"
        elif project_type == 'electronics':
            # Simulate electronic device temperature
            base_value = 40  # Base temperature in Celsius
            fluctuation = np.random.normal(0, 2, len(time))  # Random fluctuation
            load_pattern = 10 * np.sin(2 * np.pi * np.arange(len(time)) / (len(time) / 5))  # Load pattern
            values = base_value + fluctuation + load_pattern
            y_title = "Temperature (°C)"
        elif project_type == 'control':
            # Simulate control system setpoint tracking
            setpoint = 5.0  # Desired value
            values = np.zeros(len(time))
            values[0] = 0  # Initial value
            
            # Simulate first-order system response
            tau = 10  # Time constant
            
            for i in range(1, len(time)):
                # Add some noise
                noise = np.random.normal(0, 0.1)
                # First-order response
                values[i] = values[i-1] + (setpoint - values[i-1]) / tau + noise
            y_title = "Control Signal"
        elif project_type == 'power':
            # Simulate power grid frequency
            nominal_freq = 50.0  # Nominal frequency in Hz
            fluctuation = np.random.normal(0, 0.01, len(time))  # Random fluctuation
            load_effect = 0.05 * np.sin(2 * np.pi * np.arange(len(time)) / (len(time) / 10))  # Load effect
            values = nominal_freq + fluctuation + load_effect
            y_title = "Frequency (Hz)"
        elif project_type == 'telecom':
            # Simulate network traffic
            base_value = 50  # Base traffic in Mbps
            fluctuation = np.random.normal(0, 5, len(time))  # Random fluctuation
            peak_hours = 30 * np.sin(2 * np.pi * np.arange(len(time)) / (len(time) / 4))  # Peak hours pattern
            values = base_value + fluctuation + peak_hours
            values = np.maximum(values, 0)  # Ensure non-negative
            y_title = "Network Traffic (Mbps)"
        else:
            # Default random data
            values = np.random.normal(50, 10, len(time))
            y_title = "Value"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time,
            y=values,
            mode='lines+markers',
            name='Real-time Data',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.update_layout(
            title=f'Real-time {project_type.title()} Engineering Data',
            xaxis_title='Time',
            yaxis_title=y_title,
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444')
        )
        
        # Performance metrics
        metrics = dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("85%", className="text-center"),
                    html.P("Efficiency", className="text-center")
                ], className="bg-success p-3 rounded text-white")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H4("92%", className="text-center"),
                    html.P("Accuracy", className="text-center")
                ], className="bg-success p-3 rounded text-white")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H4("78%", className="text-center"),
                    html.P("Stability", className="text-center")
                ], className="bg-warning p-3 rounded text-white")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.H4("96%", className="text-center"),
                    html.P("Reliability", className="text-center")
                ], className="bg-success p-3 rounded text-white")
            ], width=3),
        ])
        
        # Recent activities - Fixed datetime issue
        current_time = dt.now()
        activities = dbc.ListGroup([
            dbc.ListGroupItem(f"[{current_time.strftime('%H:%M')}] Signal simulation completed", color="dark"),
            dbc.ListGroupItem(f"[{(current_time.replace(minute=max(0, current_time.minute-5))).strftime('%H:%M')}] Circuit model updated", color="dark"),
            dbc.ListGroupItem(f"[{(current_time.replace(minute=max(0, current_time.minute-12))).strftime('%H:%M')}] Statistical analysis performed", color="dark"),
            dbc.ListGroupItem(f"[{(current_time.replace(minute=max(0, current_time.minute-18))).strftime('%H:%M')}] New data uploaded", color="dark"),
        ], flush=True)
        
        return status, fig, metrics, activities
    except Exception as e:
        # Return error message if something goes wrong
        error_msg = html.Div([
            html.H4("Error Loading Dashboard", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Loading Data",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return error_msg, empty_fig, error_msg, error_msg

# Simulation callbacks
@app.callback(
    Output('simulation-content', 'children'),
    [Input('simulation-tabs', 'active_tab'),
     Input('project-type-dropdown', 'value')]
)
def render_simulation_content(active_tab, project_type):
    if active_tab == "signal-sim":
        return signal_simulation_tab(project_type)
    elif active_tab == "circuit-sim":
        return circuit_simulation_tab(project_type)
    elif active_tab == "control-sim":
        return control_simulation_tab(project_type)
    return html.Div()

def signal_simulation_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Parameters"),
                    dbc.CardBody([
                        html.Label("Signal Type", className="text-light"),
                        dcc.Dropdown(
                            id='signal-type-dropdown',
                            options=[
                                {'label': 'Sine Wave', 'value': 'sine'},
                                {'label': 'Square Wave', 'value': 'square'},
                                {'label': 'Sawtooth Wave', 'value': 'sawtooth'},
                                {'label': 'Noise', 'value': 'noise'},
                                {'label': 'Composite', 'value': 'composite'}
                            ],
                            value='sine',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Amplitude", className="text-light"),
                        dcc.Slider(
                            id='amplitude-slider',
                            min=0.1,
                            max=10,
                            step=0.1,
                            value=5,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Frequency (Hz)", className="text-light"),
                        dcc.Slider(
                            id='frequency-slider',
                            min=0.1,
                            max=10,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Duration (s)", className="text-light"),
                        dcc.Slider(
                            id='duration-slider',
                            min=1,
                            max=10,
                            step=1,
                            value=5,
                            marks={i: str(i) for i in range(1, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Generate Signal", id="generate-signal-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-signal-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='signal-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Properties"),
                    dbc.CardBody([
                        html.Div(id='signal-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def circuit_simulation_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Circuit Parameters"),
                    dbc.CardBody([
                        html.Label("Circuit Type", className="text-light"),
                        dcc.Dropdown(
                            id='circuit-type-dropdown',
                            options=[
                                {'label': 'RC Circuit', 'value': 'rc'},
                                {'label': 'RL Circuit', 'value': 'rl'},
                                {'label': 'RLC Circuit', 'value': 'rlc'},
                                {'label': 'Diode Circuit', 'value': 'diode'},
                                {'label': 'BJT Amplifier', 'value': 'bjt'}
                            ],
                            value='rc',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Input Voltage (V)", className="text-light"),
                        dcc.Slider(
                            id='input-voltage-slider',
                            min=0,
                            max=12,
                            step=0.1,
                            value=5,
                            marks={i: str(i) for i in range(0, 13)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Resistance (Ω)", className="text-light"),
                        dcc.Slider(
                            id='resistance-slider',
                            min=100,
                            max=10000,
                            step=100,
                            value=1000,
                            marks={i: str(i) for i in range(0, 11000, 2000)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Capacitance (μF)", className="text-light"),
                        dcc.Slider(
                            id='capacitance-slider',
                            min=1,
                            max=1000,
                            step=1,
                            value=100,
                            marks={i: str(i) for i in range(0, 1100, 200)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Inductance (mH)", className="text-light"),
                        dcc.Slider(
                            id='inductance-slider',
                            min=1,
                            max=1000,
                            step=1,
                            value=100,
                            marks={i: str(i) for i in range(0, 1100, 200)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Simulate Circuit", id="simulate-circuit-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-circuit-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Circuit Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='circuit-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Circuit Properties"),
                    dbc.CardBody([
                        html.Div(id='circuit-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def control_simulation_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Control System Parameters"),
                    dbc.CardBody([
                        html.Label("System Type", className="text-light"),
                        dcc.Dropdown(
                            id='system-type-dropdown',
                            options=[
                                {'label': 'First-Order System', 'value': 'first_order'},
                                {'label': 'Second-Order System', 'value': 'second_order'},
                                {'label': 'PID Controller', 'value': 'pid'},
                                {'label': 'State-Space Model', 'value': 'state_space'}
                            ],
                            value='first_order',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Proportional Gain (Kp)", className="text-light"),
                        dcc.Slider(
                            id='kp-slider',
                            min=0,
                            max=10,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Integral Gain (Ki)", className="text-light"),
                        dcc.Slider(
                            id='ki-slider',
                            min=0,
                            max=10,
                            step=0.1,
                            value=0.1,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Derivative Gain (Kd)", className="text-light"),
                        dcc.Slider(
                            id='kd-slider',
                            min=0,
                            max=10,
                            step=0.1,
                            value=0.01,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Setpoint", className="text-light"),
                        dcc.Slider(
                            id='setpoint-slider',
                            min=0,
                            max=10,
                            step=0.1,
                            value=5,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Simulate System", id="simulate-system-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-system-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Response"),
                    dbc.CardBody([
                        dcc.Graph(id='system-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Properties"),
                    dbc.CardBody([
                        html.Div(id='system-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

# Signal simulation callback
@app.callback(
    [Output('signal-graph', 'figure'),
     Output('signal-properties', 'children')],
    [Input('generate-signal-btn', 'n_clicks')],
    [State('signal-type-dropdown', 'value'),
     State('amplitude-slider', 'value'),
     State('frequency-slider', 'value'),
     State('duration-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_signal_graph(n_clicks, signal_type, amplitude, frequency, duration, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Generate signal data
        sampling_rate = 1000  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        
        # Adjust signal parameters based on project type
        if project_type == 'electrical':
            # Electrical signals typically have lower frequencies
            frequency = frequency * 0.5
        elif project_type == 'electronics':
            # Electronics signals can have higher frequencies
            frequency = frequency * 2
        elif project_type == 'telecom':
            # Telecom signals have much higher frequencies
            frequency = frequency * 10
        
        # Generate signal based on type
        if signal_type == 'sine':
            signal = amplitude * np.sin(2 * np.pi * frequency * t)
        elif signal_type == 'square':
            signal = amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        elif signal_type == 'sawtooth':
            signal = amplitude * (2 * (t * frequency - np.floor(0.5 + t * frequency)))
        elif signal_type == 'noise':
            signal = amplitude * np.random.normal(size=len(t))
        elif signal_type == 'composite':
            # Composite signal with multiple frequency components
            signal = amplitude * (
                0.6 * np.sin(2 * np.pi * frequency * t) +
                0.3 * np.sin(2 * np.pi * 2 * frequency * t) +
                0.1 * np.sin(2 * np.pi * 3 * frequency * t)
            )
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=signal,
            mode='lines',
            name=signal_type.title() + ' Wave',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.update_layout(
            title=f'{signal_type.title()} Wave Signal ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444')
        )
        
        # Calculate signal properties
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"Signal Type: {signal_type.title()}", color="dark"),
            dbc.ListGroupItem(f"Amplitude: {amplitude} V", color="dark"),
            dbc.ListGroupItem(f"Frequency: {frequency} Hz", color="dark"),
            dbc.ListGroupItem(f"Duration: {duration} s", color="dark"),
            dbc.ListGroupItem(f"RMS Value: {rms:.2f} V", color="dark"),
            dbc.ListGroupItem(f"Peak Value: {peak:.2f} V", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Generating Signal", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Generating Signal",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Circuit simulation callback
@app.callback(
    [Output('circuit-graph', 'figure'),
     Output('circuit-properties', 'children')],
    [Input('simulate-circuit-btn', 'n_clicks')],
    [State('circuit-type-dropdown', 'value'),
     State('input-voltage-slider', 'value'),
     State('resistance-slider', 'value'),
     State('capacitance-slider', 'value'),
     State('inductance-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_circuit_graph(n_clicks, circuit_type, input_voltage, resistance, capacitance, inductance, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Convert units
        capacitance = capacitance * 1e-6  # Convert μF to F
        inductance = inductance * 1e-3    # Convert mH to H
        
        # Simulation parameters
        duration = 0.1  # seconds
        sampling_rate = 10000  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        
        # Adjust parameters based on project type
        if project_type == 'electronics':
            # Electronics typically use smaller components
            resistance = resistance * 0.1
            capacitance = capacitance * 0.1
            inductance = inductance * 0.01
        elif project_type == 'power':
            # Power systems use larger components
            resistance = resistance * 10
            capacitance = capacitance * 10
            inductance = inductance * 10
        
        if circuit_type == 'rc':
            # RC circuit simulation
            tau = resistance * capacitance  # Time constant
            voltage = input_voltage * (1 - np.exp(-t / tau))
            current = (input_voltage / resistance) * np.exp(-t / tau)
            
        elif circuit_type == 'rl':
            # RL circuit simulation
            tau = inductance / resistance  # Time constant
            voltage = input_voltage * np.exp(-t / tau)
            current = (input_voltage / resistance) * (1 - np.exp(-t / tau))
            
        elif circuit_type == 'rlc':
            # RLC circuit simulation
            # Calculate damping factor and resonant frequency
            omega_0 = 1 / np.sqrt(inductance * capacitance)  # Resonant frequency
            alpha = resistance / (2 * inductance)  # Damping factor
            
            if alpha < omega_0:  # Underdamped
                omega_d = np.sqrt(omega_0**2 - alpha**2)  # Damped frequency
                voltage = input_voltage * np.exp(-alpha * t) * np.cos(omega_d * t)
                current = (input_voltage / (omega_d * inductance)) * np.exp(-alpha * t) * np.sin(omega_d * t)
            elif alpha == omega_0:  # Critically damped
                voltage = input_voltage * (1 + alpha * t) * np.exp(-alpha * t)
                current = (input_voltage / inductance) * t * np.exp(-alpha * t)
            else:  # Overdamped
                s1 = -alpha + np.sqrt(alpha**2 - omega_0**2)
                s2 = -alpha - np.sqrt(alpha**2 - omega_0**2)
                A = input_voltage * s2 / (s2 - s1)
                B = -input_voltage * s1 / (s2 - s1)
                voltage = A * np.exp(s1 * t) + B * np.exp(s2 * t)
                current = (A * s1 * np.exp(s1 * t) + B * s2 * np.exp(s2 * t)) / resistance
        
        elif circuit_type == 'diode':
            # Diode circuit simulation
            # Simplified diode model: I = Is * (exp(V/Vt) - 1)
            Is = 1e-12  # Saturation current
            Vt = 0.026  # Thermal voltage at room temperature
            
            # Apply input voltage as a step
            voltage = np.ones_like(t) * input_voltage
            
            # Solve for current using diode equation
            # I = (V_input - V_diode) / R
            # V_diode = Vt * ln(I/Is + 1)
            # This is a transcendental equation, so we'll use an iterative approach
            current = np.zeros_like(t)
            for i in range(len(t)):
                # Initial guess
                I = (input_voltage - 0.7) / resistance  # Assume 0.7V diode drop
                
                # Newton-Raphson iteration
                for _ in range(10):  # Usually converges quickly
                    f = I * resistance + Vt * np.log(I/Is + 1) - input_voltage
                    df = resistance + Vt / (Is + I)
                    I_new = I - f / df
                    
                    if abs(I_new - I) < 1e-10:
                        break
                        
                    I = I_new
                
                current[i] = I
        
        elif circuit_type == 'bjt':
            # BJT amplifier circuit simulation
            # Simplified Ebers-Moll model
            Vbe = 0.7  # Base-emitter voltage
            beta = 100  # Current gain
            
            # Apply input voltage as a step
            voltage = np.ones_like(t) * input_voltage
            
            # Calculate base current
            Ib = (input_voltage - Vbe) / resistance if input_voltage > Vbe else 0
            
            # Calculate collector current
            Ic = beta * Ib
            
            # Calculate output voltage (assuming a simple common-emitter configuration)
            Rc = 1000  # Collector resistance
            Vcc = 12   # Supply voltage
            voltage_out = Vcc - Ic * Rc
            
            # For simplicity, we'll use the input voltage as the voltage across the transistor
            # and the collector current as the circuit current
            current = np.ones_like(t) * Ic
        
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=voltage,
            mode='lines',
            name='Voltage',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=current,
            mode='lines',
            name='Current',
            line=dict(color='#FF5722', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f'{circuit_type.upper()} Circuit Simulation ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Voltage (V)',
            yaxis2=dict(title='Current (A)', overlaying='y', side='right'),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            legend=dict(x=0, y=1)
        )
        
        # Calculate circuit properties
        power = np.mean(voltage * current)
        impedance = resistance  # Simplified for demonstration
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"Circuit Type: {circuit_type.upper()}", color="dark"),
            dbc.ListGroupItem(f"Input Voltage: {input_voltage} V", color="dark"),
            dbc.ListGroupItem(f"Resistance: {resistance} Ω", color="dark"),
            dbc.ListGroupItem(f"Capacitance: {capacitance*1e6} μF", color="dark"),
            dbc.ListGroupItem(f"Inductance: {inductance*1e3} mH", color="dark"),
            dbc.ListGroupItem(f"Average Power: {power:.2f} W", color="dark"),
            dbc.ListGroupItem(f"Impedance: {impedance:.2f} Ω", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Simulating Circuit", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Simulating Circuit",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Control system simulation callback
@app.callback(
    [Output('system-graph', 'figure'),
     Output('system-properties', 'children')],
    [Input('simulate-system-btn', 'n_clicks')],
    [State('system-type-dropdown', 'value'),
     State('kp-slider', 'value'),
     State('ki-slider', 'value'),
     State('kd-slider', 'value'),
     State('setpoint-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_system_graph(n_clicks, system_type, kp, ki, kd, setpoint, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Simulation parameters
        duration = 10.0  # seconds
        sampling_rate = 100  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        dt = 1.0 / sampling_rate
        
        # Adjust parameters based on project type
        if project_type == 'control':
            # Control systems may need different parameters
            kp = kp * 1.5
            ki = ki * 0.8
            kd = kd * 1.2
        
        if system_type == 'first_order':
            # First-order system: G(s) = K / (τs + 1)
            K = 1.0  # System gain
            tau = 1.0  # Time constant
            
            # Initialize
            output = np.zeros_like(t)
            error = np.ones_like(t) * setpoint
            
            # Simulate
            for i in range(1, len(t)):
                # Calculate error
                error[i] = setpoint - output[i-1]
                
                # Calculate control signal (P controller)
                control = kp * error[i]
                
                # Update output (first-order system response)
                output[i] = output[i-1] + (K * control - output[i-1]) * dt / tau
        
        elif system_type == 'second_order':
            # Second-order system: G(s) = K * ωn^2 / (s^2 + 2ζωn*s + ωn^2)
            K = 1.0  # System gain
            zeta = 0.7  # Damping ratio
            wn = 1.0  # Natural frequency
            
            # Initialize
            output = np.zeros_like(t)
            error = np.ones_like(t) * setpoint
            velocity = np.zeros_like(t)
            
            # Simulate
            for i in range(1, len(t)):
                # Calculate error
                error[i] = setpoint - output[i-1]
                
                # Calculate control signal (PD controller)
                control = kp * error[i] + kd * (error[i] - error[i-1]) / dt
                
                # Update velocity and position (second-order system response)
                acceleration = K * wn**2 * control - 2 * zeta * wn * velocity[i-1] - wn**2 * output[i-1]
                velocity[i] = velocity[i-1] + acceleration * dt
                output[i] = output[i-1] + velocity[i] * dt
        
        elif system_type == 'pid':
            # PID controller with first-order plant
            K = 1.0  # System gain
            tau = 1.0  # Time constant
            
            # Initialize
            output = np.zeros_like(t)
            error = np.ones_like(t) * setpoint
            integral = 0.0
            prev_error = setpoint
            
            # Simulate
            for i in range(1, len(t)):
                # Calculate error
                error[i] = setpoint - output[i-1]
                
                # Update integral
                integral += error[i] * dt
                
                # Calculate derivative
                derivative = (error[i] - prev_error) / dt
                
                # Calculate control signal (PID controller)
                control = kp * error[i] + ki * integral + kd * derivative
                
                # Update output (first-order system response)
                output[i] = output[i-1] + (K * control - output[i-1]) * dt / tau
                
                # Update previous error
                prev_error = error[i]
        
        elif system_type == 'state_space':
            # State-space model: ẋ = Ax + Bu, y = Cx + Du
            # Example: Mass-spring-damper system
            m = 1.0  # Mass
            c = 0.5  # Damping coefficient
            k = 2.0  # Spring constant
            
            # State-space matrices
            A = np.array([[0, 1], [-k/m, -c/m]])
            B = np.array([[0], [1/m]])
            C = np.array([[1, 0]])
            D = np.array([[0]])
            
            # Initialize
            x = np.array([[0.0], [0.0]])  # Initial state [position; velocity]
            output = np.zeros_like(t)
            error = np.ones_like(t) * setpoint
            
            # Simulate
            for i in range(1, len(t)):
                # Calculate error
                error[i] = setpoint - output[i-1]
                
                # Calculate control signal (state feedback controller)
                # For simplicity, we'll use a P controller
                control = kp * error[i]
                
                # Update state (Euler integration)
                x_dot = A @ x + B * control
                x = x + x_dot * dt
                
                # Calculate output
                output[i] = (C @ x + D * control)[0, 0]
        
        else:
            raise ValueError(f"Unknown system type: {system_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=[setpoint] * len(t),
            mode='lines',
            name='Setpoint',
            line=dict(color='#2196F3', width=2, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=output,
            mode='lines',
            name='System Output',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=error,
            mode='lines',
            name='Error',
            line=dict(color='#FF5722', width=1),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f'{system_type.replace("_", " ").title()} Control System ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Output',
            yaxis2=dict(title='Error', overlaying='y', side='right'),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            legend=dict(x=0, y=1)
        )
        
        # Calculate system properties
        rise_time = 0.5  # Simplified for demonstration
        settling_time = 2.0  # Simplified for demonstration
        overshoot = 10.0  # Simplified for demonstration
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"System Type: {system_type.replace('_', ' ').title()}", color="dark"),
            dbc.ListGroupItem(f"Proportional Gain (Kp): {kp}", color="dark"),
            dbc.ListGroupItem(f"Integral Gain (Ki): {ki}", color="dark"),
            dbc.ListGroupItem(f"Derivative Gain (Kd): {kd}", color="dark"),
            dbc.ListGroupItem(f"Setpoint: {setpoint}", color="dark"),
            dbc.ListGroupItem(f"Rise Time: {rise_time:.2f} s", color="dark"),
            dbc.ListGroupItem(f"Settling Time: {settling_time:.2f} s", color="dark"),
            dbc.ListGroupItem(f"Overshoot: {overshoot:.2f}%", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Simulating System", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Simulating System",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Modeling callbacks
@app.callback(
    Output('modeling-content', 'children'),
    [Input('modeling-tabs', 'active_tab'),
     Input('project-type-dropdown', 'value')]
)
def render_modeling_content(active_tab, project_type):
    if active_tab == "signal-mod":
        return signal_modeling_tab(project_type)
    elif active_tab == "circuit-mod":
        return circuit_modeling_tab(project_type)
    elif active_tab == "system-mod":
        return system_modeling_tab(project_type)
    return html.Div()

def signal_modeling_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Modeling Parameters"),
                    dbc.CardBody([
                        html.Label("Model Type", className="text-light"),
                        dcc.Dropdown(
                            id='signal-model-type-dropdown',
                            options=[
                                {'label': 'Sinusoidal Model', 'value': 'sinusoidal'},
                                {'label': 'Exponential Model', 'value': 'exponential'},
                                {'label': 'Polynomial Model', 'value': 'polynomial'},
                                {'label': 'Fourier Model', 'value': 'fourier'}
                            ],
                            value='sinusoidal',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Model Order", className="text-light"),
                        dcc.Slider(
                            id='model-order-slider',
                            min=1,
                            max=10,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(1, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Data Points", className="text-light"),
                        dcc.Slider(
                            id='data-points-slider',
                            min=10,
                            max=1000,
                            step=10,
                            value=100,
                            marks={i: str(i) for i in range(0, 1100, 200)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Generate Model", id="generate-signal-model-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-signal-model-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Signal Model Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='signal-model-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Properties"),
                    dbc.CardBody([
                        html.Div(id='signal-model-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def circuit_modeling_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Circuit Modeling Parameters"),
                    dbc.CardBody([
                        html.Label("Circuit Type", className="text-light"),
                        dcc.Dropdown(
                            id='circuit-model-type-dropdown',
                            options=[
                                {'label': 'RC Circuit Model', 'value': 'rc'},
                                {'label': 'RL Circuit Model', 'value': 'rl'},
                                {'label': 'RLC Circuit Model', 'value': 'rlc'},
                                {'label': 'Diode Circuit Model', 'value': 'diode'},
                                {'label': 'BJT Amplifier Model', 'value': 'bjt'}
                            ],
                            value='rc',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Model Complexity", className="text-light"),
                        dcc.Slider(
                            id='model-complexity-slider',
                            min=1,
                            max=5,
                            step=1,
                            value=3,
                            marks={i: str(i) for i in range(1, 6)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Temperature (°C)", className="text-light"),
                        dcc.Slider(
                            id='temperature-slider',
                            min=-40,
                            max=125,
                            step=5,
                            value=25,
                            marks={i: str(i) for i in range(-40, 130, 30)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Generate Model", id="generate-circuit-model-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-circuit-model-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Circuit Model Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='circuit-model-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Properties"),
                    dbc.CardBody([
                        html.Div(id='circuit-model-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def system_modeling_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Modeling Parameters"),
                    dbc.CardBody([
                        html.Label("System Type", className="text-light"),
                        dcc.Dropdown(
                            id='system-model-type-dropdown',
                            options=[
                                {'label': 'First-Order System', 'value': 'first_order'},
                                {'label': 'Second-Order System', 'value': 'second_order'},
                                {'label': 'PID Controller Model', 'value': 'pid'},
                                {'label': 'State-Space Model', 'value': 'state_space'},
                                {'label': 'Transfer Function', 'value': 'transfer_function'}
                            ],
                            value='first_order',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Model Order", className="text-light"),
                        dcc.Slider(
                            id='system-model-order-slider',
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                            marks={i: str(i) for i in range(1, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Damping Ratio", className="text-light"),
                        dcc.Slider(
                            id='damping-ratio-slider',
                            min=0,
                            max=2,
                            step=0.1,
                            value=0.7,
                            marks={i: str(i) for i in range(0, 3)},
                            className="text-light"
                        ),
                        html.Br(),
                        html.Label("Natural Frequency (rad/s)", className="text-light"),
                        dcc.Slider(
                            id='natural-frequency-slider',
                            min=0.1,
                            max=10,
                            step=0.1,
                            value=1,
                            marks={i: str(i) for i in range(0, 11)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Generate Model", id="generate-system-model-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-system-model-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("System Model Visualization"),
                    dbc.CardBody([
                        dcc.Graph(id='system-model-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Model Properties"),
                    dbc.CardBody([
                        html.Div(id='system-model-properties')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

# Signal modeling callback
@app.callback(
    [Output('signal-model-graph', 'figure'),
     Output('signal-model-properties', 'children')],
    [Input('generate-signal-model-btn', 'n_clicks')],
    [State('signal-model-type-dropdown', 'value'),
     State('model-order-slider', 'value'),
     State('data-points-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_signal_model_graph(n_clicks, model_type, model_order, data_points, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Generate time array
        t = np.linspace(0, 10, data_points)
        
        # Generate original signal with noise
        if model_type == 'sinusoidal':
            # Original signal: sum of sinusoids
            original = 2 * np.sin(2 * np.pi * 1 * t) + 1 * np.sin(2 * np.pi * 2.5 * t) + 0.5 * np.sin(2 * np.pi * 4 * t)
        elif model_type == 'exponential':
            # Original signal: exponential decay with oscillation
            original = 5 * np.exp(-0.5 * t) * np.cos(2 * np.pi * 1 * t)
        elif model_type == 'polynomial':
            # Original signal: polynomial function
            original = 0.1 * t**3 - 0.5 * t**2 + 2 * t + 1
        elif model_type == 'fourier':
            # Original signal: square wave
            original = np.sign(np.sin(2 * np.pi * 1 * t))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Add noise
        original += 0.2 * np.random.normal(size=data_points)
        
        # Adjust signal based on project type
        if project_type == 'electrical':
            # Electrical signals typically have lower frequencies
            original = original * 0.8
        elif project_type == 'electronics':
            # Electronics signals can have higher frequencies
            original = original * 1.2
        elif project_type == 'telecom':
            # Telecom signals have much higher frequencies
            original = original * 2.0
        
        # Fit model based on type
        if model_type == 'sinusoidal':
            # Fit sum of sinusoids
            def sinusoidal_model(t, *params):
                result = np.zeros_like(t)
                for i in range(model_order):
                    amp = params[2*i]
                    freq = params[2*i+1]
                    result += amp * np.sin(2 * np.pi * freq * t)
                return result
            
            # Initial guess for parameters
            initial_guess = []
            for i in range(model_order):
                initial_guess.extend([1.0, (i+1)])  # Amplitude, frequency
            
            # Fit the model
            popt, _ = optimize.curve_fit(sinusoidal_model, t, original, p0=initial_guess)
            modeled = sinusoidal_model(t, *popt)
            coefficients = popt
        
        elif model_type == 'exponential':
            # Fit exponential model
            def exponential_model(t, a, b, c, d):
                return a * np.exp(-b * t) * np.cos(c * t) + d
            
            # Initial guess for parameters
            initial_guess = [5.0, 0.5, 2*np.pi, 0.0]
            
            # Fit the model
            popt, _ = optimize.curve_fit(exponential_model, t, original, p0=initial_guess)
            modeled = exponential_model(t, *popt)
            coefficients = popt
        
        elif model_type == 'polynomial':
            # Fit polynomial model
            coefficients = np.polyfit(t, original, model_order)
            modeled = np.polyval(coefficients, t)
        
        elif model_type == 'fourier':
            # Fit Fourier series
            def fourier_model(t, *params):
                result = np.zeros_like(t)
                # DC component
                result += params[0]
                # Sinusoidal components
                for i in range(1, model_order+1):
                    result += params[2*i-1] * np.cos(2 * np.pi * i * t / 10)
                    result += params[2*i] * np.sin(2 * np.pi * i * t / 10)
                return result
            
            # Initial guess for parameters
            initial_guess = [0.0]  # DC component
            for i in range(1, model_order+1):
                initial_guess.extend([0.0, 0.0])  # Cosine and sine coefficients
            
            # Fit the model
            popt, _ = optimize.curve_fit(fourier_model, t, original, p0=initial_guess)
            modeled = fourier_model(t, *popt)
            coefficients = popt
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=original,
            mode='lines',
            name='Original Signal',
            line=dict(color='#2196F3', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=modeled,
            mode='lines',
            name='Modeled Signal',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.update_layout(
            title=f'{model_type.title()} Signal Model ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Amplitude',
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            legend=dict(x=0, y=1)
        )
        
        # Format coefficients for display
        coeff_str = ", ".join([f"{c:.4f}" for c in coefficients])
        
        # Calculate model accuracy
        mse = np.mean((original - modeled)**2)
        r2 = 1 - np.sum((original - modeled)**2) / np.sum((original - np.mean(original))**2)
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"Model Type: {model_type.title()}", color="dark"),
            dbc.ListGroupItem(f"Model Order: {model_order}", color="dark"),
            dbc.ListGroupItem(f"Data Points: {data_points}", color="dark"),
            dbc.ListGroupItem(f"Coefficients: [{coeff_str}]", color="dark"),
            dbc.ListGroupItem(f"Mean Squared Error: {mse:.4f}", color="dark"),
            dbc.ListGroupItem(f"R-squared: {r2:.4f}", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Generating Signal Model", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Generating Signal Model",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Circuit modeling callback
@app.callback(
    [Output('circuit-model-graph', 'figure'),
     Output('circuit-model-properties', 'children')],
    [Input('generate-circuit-model-btn', 'n_clicks')],
    [State('circuit-model-type-dropdown', 'value'),
     State('model-complexity-slider', 'value'),
     State('temperature-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_circuit_model_graph(n_clicks, circuit_type, complexity, temperature, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Simulation parameters
        duration = 0.1  # seconds
        sampling_rate = 10000  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        
        # Convert temperature to Kelvin
        T = temperature + 273.15
        
        # Boltzmann constant
        k = 1.38e-23  # J/K
        
        # Elementary charge
        q = 1.602e-19  # C
        
        # Thermal voltage
        Vt = k * T / q
        
        # Adjust parameters based on project type
        if project_type == 'electronics':
            # Electronics typically use smaller components
            resistance_factor = 0.1
            capacitance_factor = 0.1
            inductance_factor = 0.01
        elif project_type == 'power':
            # Power systems use larger components
            resistance_factor = 10
            capacitance_factor = 10
            inductance_factor = 10
        else:
            resistance_factor = 1
            capacitance_factor = 1
            inductance_factor = 1
        
        if circuit_type == 'rc':
            # RC circuit model
            # Parameters depend on complexity
            if complexity == 1:
                # Simple linear model
                R = 1000 * resistance_factor  # Resistance in ohms
                C = 100e-6 * capacitance_factor  # Capacitance in farads
                tau = R * C  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Output voltage
                Vout = Vin * (1 - np.exp(-t / tau))
                
                # Current
                I = (Vin - Vout) / R
                
                parameters = {
                    'Resistance': R,
                    'Capacitance': C,
                    'Time Constant': tau,
                    'Temperature': temperature
                }
            
            elif complexity == 2:
                # Include temperature effects
                R0 = 1000 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                C = 100e-6 * capacitance_factor  # Capacitance in farads
                tau = R * C  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Output voltage
                Vout = Vin * (1 - np.exp(-t / tau))
                
                # Current
                I = (Vin - Vout) / R
                
                parameters = {
                    'Resistance': R,
                    'Capacitance': C,
                    'Time Constant': tau,
                    'Temperature': temperature,
                    'Temp Coefficient': alpha
                }
            
            elif complexity >= 3:
                # Include parasitic effects
                R0 = 1000 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                C0 = 100e-6 * capacitance_factor  # Capacitance at reference temperature
                beta = 0.0002  # Temperature coefficient for capacitance
                
                # Temperature-dependent capacitance
                C = C0 * (1 + beta * (T - T0))
                
                # Parasitic elements
                Rs = 0.1 * R  # Series resistance
                Rp = 10 * R   # Parallel resistance
                L = 1e-6 * inductance_factor      # Parasitic inductance
                
                # More complex model
                tau = R * C  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Solve differential equation numerically
                Vout = np.zeros_like(t)
                I = np.zeros_like(t)
                
                for i in range(1, len(t)):
                    dt = t[i] - t[i-1]
                    
                    # Simplified model with parasitic elements
                    dVout = (Vin[i-1] - Vout[i-1] * (1 + R/Rp) - I[i-1] * Rs) / (R * C) * dt
                    Vout[i] = Vout[i-1] + dVout
                    I[i] = C * dVout / dt + Vout[i] / Rp
                
                parameters = {
                    'Resistance': R,
                    'Capacitance': C,
                    'Time Constant': tau,
                    'Temperature': temperature,
                    'Series Resistance': Rs,
                    'Parallel Resistance': Rp,
                    'Parasitic Inductance': L
                }
        
        elif circuit_type == 'rl':
            # RL circuit model
            # Parameters depend on complexity
            if complexity == 1:
                # Simple linear model
                R = 100 * resistance_factor  # Resistance in ohms
                L = 0.1 * inductance_factor  # Inductance in henries
                tau = L / R  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Current
                I = (Vin / R) * (1 - np.exp(-t / tau))
                
                # Output voltage
                Vout = I * R
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Time Constant': tau,
                    'Temperature': temperature
                }
            
            elif complexity == 2:
                # Include temperature effects
                R0 = 100 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                L = 0.1 * inductance_factor  # Inductance in henries
                tau = L / R  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Current
                I = (Vin / R) * (1 - np.exp(-t / tau))
                
                # Output voltage
                Vout = I * R
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Time Constant': tau,
                    'Temperature': temperature,
                    'Temp Coefficient': alpha
                }
            
            elif complexity >= 3:
                # Include parasitic effects
                R0 = 100 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                L0 = 0.1 * inductance_factor  # Inductance at reference temperature
                gamma = 0.0001  # Temperature coefficient for inductance
                
                # Temperature-dependent inductance
                L = L0 * (1 + gamma * (T - T0))
                
                # Parasitic elements
                Rs = 0.1 * R  # Series resistance
                Cp = 1e-9     # Parasitic capacitance
                
                # More complex model
                tau = L / R  # Time constant
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Solve differential equation numerically
                Vout = np.zeros_like(t)
                I = np.zeros_like(t)
                
                for i in range(1, len(t)):
                    dt = t[i] - t[i-1]
                    
                    # Simplified model with parasitic elements
                    dI = (Vin[i-1] - I[i-1] * (R + Rs) - Vout[i-1]) / L * dt
                    I[i] = I[i-1] + dI
                    Vout[i] = Vout[i-1] + I[i] * dt / Cp
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Time Constant': tau,
                    'Temperature': temperature,
                    'Series Resistance': Rs,
                    'Parasitic Capacitance': Cp
                }
        
        elif circuit_type == 'rlc':
            # RLC circuit model
            # Parameters depend on complexity
            if complexity == 1:
                # Simple linear model
                R = 100 * resistance_factor  # Resistance in ohms
                L = 0.1 * inductance_factor  # Inductance in henries
                C = 10e-6 * capacitance_factor  # Capacitance in farads
                
                # Calculate resonant frequency and damping
                omega_0 = 1 / np.sqrt(L * C)  # Resonant frequency
                zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Solve differential equation numerically
                Vout = np.zeros_like(t)
                I = np.zeros_like(t)
                
                for i in range(1, len(t)):
                    dt = t[i] - t[i-1]
                    
                    # Simplified model
                    dI = (Vin[i-1] - Vout[i-1] - I[i-1] * R) / L * dt
                    dVout = I[i-1] / C * dt
                    
                    I[i] = I[i-1] + dI
                    Vout[i] = Vout[i-1] + dVout
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Capacitance': C,
                    'Resonant Frequency': omega_0,
                    'Damping Ratio': zeta,
                    'Temperature': temperature
                }
            
            elif complexity == 2:
                # Include temperature effects
                R0 = 100 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                L = 0.1 * inductance_factor  # Inductance in henries
                C = 10e-6 * capacitance_factor  # Capacitance in farads
                
                # Calculate resonant frequency and damping
                omega_0 = 1 / np.sqrt(L * C)  # Resonant frequency
                zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Solve differential equation numerically
                Vout = np.zeros_like(t)
                I = np.zeros_like(t)
                
                for i in range(1, len(t)):
                    dt = t[i] - t[i-1]
                    
                    # Simplified model
                    dI = (Vin[i-1] - Vout[i-1] - I[i-1] * R) / L * dt
                    dVout = I[i-1] / C * dt
                    
                    I[i] = I[i-1] + dI
                    Vout[i] = Vout[i-1] + dVout
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Capacitance': C,
                    'Resonant Frequency': omega_0,
                    'Damping Ratio': zeta,
                    'Temperature': temperature,
                    'Temp Coefficient': alpha
                }
            
            elif complexity >= 3:
                # Include parasitic effects
                R0 = 100 * resistance_factor  # Resistance at reference temperature
                T0 = 298.15  # Reference temperature in K
                alpha = 0.0039  # Temperature coefficient
                
                # Temperature-dependent resistance
                R = R0 * (1 + alpha * (T - T0))
                
                L0 = 0.1 * inductance_factor  # Inductance at reference temperature
                gamma = 0.0001  # Temperature coefficient for inductance
                
                # Temperature-dependent inductance
                L = L0 * (1 + gamma * (T - T0))
                
                C0 = 10e-6 * capacitance_factor  # Capacitance at reference temperature
                beta = 0.0002  # Temperature coefficient for capacitance
                
                # Temperature-dependent capacitance
                C = C0 * (1 + beta * (T - T0))
                
                # Parasitic elements
                Rs = 0.1 * R  # Series resistance
                Rp = 10 * R   # Parallel resistance
                Lp = 0.01 * L # Parallel inductance
                Cs = 0.1 * C  # Series capacitance
                
                # Calculate resonant frequency and damping
                omega_0 = 1 / np.sqrt(L * C)  # Resonant frequency
                zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
                
                # Input voltage step
                Vin = np.ones_like(t) * 5
                
                # Solve differential equation numerically
                Vout = np.zeros_like(t)
                I = np.zeros_like(t)
                Vc = np.zeros_like(t)  # Voltage across capacitor
                
                for i in range(1, len(t)):
                    dt = t[i] - t[i-1]
                    
                    # More complex model with parasitic elements
                    dI = (Vin[i-1] - Vc[i-1] - I[i-1] * (R + Rs)) / (L + Lp) * dt
                    dVc = I[i-1] / (C + Cs) * dt
                    
                    I[i] = I[i-1] + dI
                    Vc[i] = Vc[i-1] + dVc
                    Vout[i] = Vc[i] * Rp / (Rp + R)  # Voltage divider with parallel resistance
                
                parameters = {
                    'Resistance': R,
                    'Inductance': L,
                    'Capacitance': C,
                    'Resonant Frequency': omega_0,
                    'Damping Ratio': zeta,
                    'Temperature': temperature,
                    'Series Resistance': Rs,
                    'Parallel Resistance': Rp,
                    'Parallel Inductance': Lp,
                    'Series Capacitance': Cs
                }
        
        elif circuit_type == 'diode':
            # Diode circuit model
            # Parameters depend on complexity
            if complexity == 1:
                # Simple Shockley diode model
                Is = 1e-12  # Saturation current
                n = 1.0     # Ideality factor
                
                # Input voltage ramp
                Vin = np.linspace(0, 1, len(t))
                
                # Solve for current using diode equation
                # I = Is * (exp(V/(n*Vt)) - 1)
                I = Is * (np.exp(Vin / (n * Vt)) - 1)
                
                # Output voltage (across a load resistor)
                R = 1000 * resistance_factor  # Load resistance
                Vout = I * R
                
                parameters = {
                    'Saturation Current': Is,
                    'Ideality Factor': n,
                    'Load Resistance': R,
                    'Temperature': temperature
                }
            
            elif complexity == 2:
                # Include temperature effects
                Is0 = 1e-12  # Saturation current at reference temperature
                T0 = 298.15  # Reference temperature in K
                Eg = 1.12    # Bandgap energy for silicon
                
                # Temperature-dependent saturation current
                Is = Is0 * (T / T0)**3 * np.exp(Eg * q / (k * T) - Eg * q / (k * T0))
                
                n = 1.0     # Ideality factor
                
                # Input voltage ramp
                Vin = np.linspace(0, 1, len(t))
                
                # Solve for current using diode equation
                # I = Is * (exp(V/(n*Vt)) - 1)
                I = Is * (np.exp(Vin / (n * Vt)) - 1)
                
                # Output voltage (across a load resistor)
                R = 1000 * resistance_factor  # Load resistance
                Vout = I * R
                
                parameters = {
                    'Saturation Current': Is,
                    'Ideality Factor': n,
                    'Load Resistance': R,
                    'Temperature': temperature,
                    'Bandgap Energy': Eg
                }
            
            elif complexity >= 3:
                # Include series resistance and other effects
                Is0 = 1e-12  # Saturation current at reference temperature
                T0 = 298.15  # Reference temperature in K
                Eg = 1.12    # Bandgap energy for silicon
                
                # Temperature-dependent saturation current
                Is = Is0 * (T / T0)**3 * np.exp(Eg * q / (k * T) - Eg * q / (k * T0))
                
                n = 1.0     # Ideality factor
                Rs = 1.0    # Series resistance
                
                # Input voltage ramp
                Vin = np.linspace(0, 1, len(t))
                
                # Solve for current using diode equation with series resistance
                # I = Is * (exp((V-I*Rs)/(n*Vt)) - 1)
                # This is a transcendental equation, so we'll use an iterative approach
                I = np.zeros_like(Vin)
                
                for i in range(len(Vin)):
                    # Initial guess
                    I[i] = Is * (np.exp(Vin[i] / (n * Vt)) - 1)
                    
                    # Newton-Raphson iteration
                    for _ in range(10):  # Usually converges quickly
                        f = I[i] - Is * (np.exp((Vin[i] - I[i] * Rs) / (n * Vt)) - 1)
                        df = 1 + Is * Rs / (n * Vt) * np.exp((Vin[i] - I[i] * Rs) / (n * Vt))
                        I_new = I[i] - f / df
                        
                        if abs(I_new - I[i]) < 1e-10:
                            break
                            
                        I[i] = I_new
                
                # Output voltage (across a load resistor)
                R = 1000 * resistance_factor  # Load resistance
                Vout = I * R
                
                parameters = {
                    'Saturation Current': Is,
                    'Ideality Factor': n,
                    'Series Resistance': Rs,
                    'Load Resistance': R,
                    'Temperature': temperature,
                    'Bandgap Energy': Eg
                }
        
        elif circuit_type == 'bjt':
            # BJT circuit model
            # Parameters depend on complexity
            if complexity == 1:
                # Simple Ebers-Moll model
                Is = 1e-15  # Saturation current
                beta = 100   # Current gain
                
                # Input voltage (base-emitter)
                Vbe = np.linspace(0.6, 0.8, len(t))
                
                # Calculate base current
                Ib = Is * (np.exp(Vbe / Vt) - 1)
                
                # Calculate collector current
                Ic = beta * Ib
                
                # Output voltage (across a load resistor)
                Rc = 1000 * resistance_factor  # Collector resistance
                Vcc = 12   # Supply voltage
                Vout = Vcc - Ic * Rc
                
                parameters = {
                    'Saturation Current': Is,
                    'Current Gain': beta,
                    'Collector Resistance': Rc,
                    'Supply Voltage': Vcc,
                    'Temperature': temperature
                }
            
            elif complexity == 2:
                # Include temperature effects
                Is0 = 1e-15  # Saturation current at reference temperature
                T0 = 298.15  # Reference temperature in K
                Eg = 1.12    # Bandgap energy for silicon
                
                # Temperature-dependent saturation current
                Is = Is0 * (T / T0)**3 * np.exp(Eg * q / (k * T) - Eg * q / (k * T0))
                
                beta0 = 100  # Current gain at reference temperature
                # Temperature-dependent current gain
                beta = beta0 * (T / T0)**1.5
                
                # Input voltage (base-emitter)
                Vbe = np.linspace(0.6, 0.8, len(t))
                
                # Calculate base current
                Ib = Is * (np.exp(Vbe / Vt) - 1)
                
                # Calculate collector current
                Ic = beta * Ib
                
                # Output voltage (across a load resistor)
                Rc = 1000 * resistance_factor  # Collector resistance
                Vcc = 12   # Supply voltage
                Vout = Vcc - Ic * Rc
                
                parameters = {
                    'Saturation Current': Is,
                    'Current Gain': beta,
                    'Collector Resistance': Rc,
                    'Supply Voltage': Vcc,
                    'Temperature': temperature,
                    'Bandgap Energy': Eg
                }
            
            elif complexity >= 3:
                # Include Early effect and other non-idealities
                Is0 = 1e-15  # Saturation current at reference temperature
                T0 = 298.15  # Reference temperature in K
                Eg = 1.12    # Bandgap energy for silicon
                
                # Temperature-dependent saturation current
                Is = Is0 * (T / T0)**3 * np.exp(Eg * q / (k * T) - Eg * q / (k * T0))
                
                beta0 = 100  # Current gain at reference temperature
                # Temperature-dependent current gain
                beta = beta0 * (T / T0)**1.5
                
                Va = 100     # Early voltage
                
                # Input voltage (base-emitter)
                Vbe = np.linspace(0.6, 0.8, len(t))
                
                # Calculate base current
                Ib = Is * (np.exp(Vbe / Vt) - 1)
                
                # Calculate collector current with Early effect
                Ic = beta * Ib * (1 + Vbe / Va)
                
                # Output voltage (across a load resistor)
                Rc = 1000 * resistance_factor  # Collector resistance
                Vcc = 12   # Supply voltage
                Vout = Vcc - Ic * Rc
                
                # Include base resistance
                Rb = 100 * resistance_factor   # Base resistance
                
                parameters = {
                    'Saturation Current': Is,
                    'Current Gain': beta,
                    'Collector Resistance': Rc,
                    'Supply Voltage': Vcc,
                    'Base Resistance': Rb,
                    'Early Voltage': Va,
                    'Temperature': temperature,
                    'Bandgap Energy': Eg
                }
        
        else:
            raise ValueError(f"Unknown circuit type: {circuit_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=Vout,
            mode='lines',
            name='Voltage',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=t,
            y=I,
            mode='lines',
            name='Current',
            line=dict(color='#FF5722', width=2),
            yaxis='y2'
        ))
        fig.update_layout(
            title=f'{circuit_type.upper()} Circuit Model ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Voltage (V)',
            yaxis2=dict(title='Current (A)', overlaying='y', side='right'),
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            legend=dict(x=0, y=1)
        )
        
        # Format parameters for display
        param_str = ", ".join([f"{k}: {v:.4f}" for k, v in parameters.items()])
        
        # Calculate model accuracy
        if circuit_type in ['rc', 'rl', 'rlc']:
            # For simple circuits, we can calculate theoretical values
            if circuit_type == 'rc':
                tau_theoretical = parameters.get('Time Constant', 0)
                Vout_theoretical = 5 * (1 - np.exp(-t / tau_theoretical))
            elif circuit_type == 'rl':
                tau_theoretical = parameters.get('Time Constant', 0)
                I_theoretical = (5 / parameters.get('Resistance', 1000)) * (1 - np.exp(-t / tau_theoretical))
                Vout_theoretical = I_theoretical * parameters.get('Resistance', 1000)
            elif circuit_type == 'rlc':
                # More complex, skip for now
                Vout_theoretical = Vout
            
            # Calculate MSE
            mse = np.mean((Vout - Vout_theoretical)**2)
        else:
            mse = 0.01  # Placeholder
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"Circuit Type: {circuit_type.upper()}", color="dark"),
            dbc.ListGroupItem(f"Model Complexity: {complexity}", color="dark"),
            dbc.ListGroupItem(f"Temperature: {temperature} °C", color="dark"),
            dbc.ListGroupItem(f"Parameters: {param_str}", color="dark"),
            dbc.ListGroupItem(f"Model MSE: {mse:.6f}", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Generating Circuit Model", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Generating Circuit Model",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# System modeling callback
@app.callback(
    [Output('system-model-graph', 'figure'),
     Output('system-model-properties', 'children')],
    [Input('generate-system-model-btn', 'n_clicks')],
    [State('system-model-type-dropdown', 'value'),
     State('system-model-order-slider', 'value'),
     State('damping-ratio-slider', 'value'),
     State('natural-frequency-slider', 'value'),
     State('project-type-dropdown', 'value')]
)
def update_system_model_graph(n_clicks, system_type, model_order, damping_ratio, natural_frequency, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Simulation parameters
        duration = 10.0  # seconds
        sampling_rate = 100  # Hz
        t = np.linspace(0, duration, int(duration * sampling_rate), endpoint=False)
        
        # Input signal (step input)
        u = np.ones_like(t)
        
        # Adjust parameters based on project type
        if project_type == 'control':
            # Control systems may need different parameters
            natural_frequency = natural_frequency * 1.2
            damping_ratio = damping_ratio * 0.9
        
        if system_type == 'first_order':
            # First-order system model
            # G(s) = K / (τs + 1)
            K = 1.0  # System gain
            tau = 1.0 / natural_frequency  # Time constant
            
            # State-space representation
            A = np.array([[-1/tau]])
            B = np.array([[K/tau]])
            C = np.array([[1.0]])
            D = np.array([[0.0]])
            
            # Simulate system
            y, x = signal.lsim((A, B, C, D), u, t)
            
            # State variables
            state_variables = [x[:, 0]]
            
            # Model parameters
            model_parameters = {
                'Gain': K,
                'Time Constant': tau,
                'System Type': 'First-Order'
            }
        
        elif system_type == 'second_order':
            # Second-order system model
            # G(s) = K * ωn^2 / (s^2 + 2ζωn*s + ωn^2)
            K = 1.0  # System gain
            wn = natural_frequency  # Natural frequency
            zeta = damping_ratio    # Damping ratio
            
            # State-space representation
            A = np.array([
                [0, 1],
                [-wn**2, -2*zeta*wn]
            ])
            B = np.array([[0], [K*wn**2]])
            C = np.array([[1, 0]])
            D = np.array([[0]])
            
            # Simulate system
            y, x = signal.lsim((A, B, C, D), u, t)
            
            # State variables
            state_variables = [x[:, 0], x[:, 1]]
            
            # Model parameters
            model_parameters = {
                'Gain': K,
                'Natural Frequency': wn,
                'Damping Ratio': zeta,
                'System Type': 'Second-Order'
            }
        
        elif system_type == 'pid':
            # PID controller model
            Kp = natural_frequency  # Proportional gain
            Ki = damping_ratio      # Integral gain
            Kd = model_order        # Derivative gain
            
            # PID controller in state-space form
            A = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]
            ])
            B = np.array([[0], [0], [1]])
            C = np.array([[Kp, Ki, Kd]])
            D = np.array([[0]])
            
            # Simulate system
            y, x = signal.lsim((A, B, C, D), u, t)
            
            # State variables
            state_variables = [x[:, 0], x[:, 1], x[:, 2]]
            
            # Model parameters
            model_parameters = {
                'Proportional Gain': Kp,
                'Integral Gain': Ki,
                'Derivative Gain': Kd,
                'System Type': 'PID Controller'
            }
        
        elif system_type == 'state_space':
            # General state-space model
            # Generate random but stable state-space matrices
            np.random.seed(42)  # For reproducibility
            
            # Generate random stable A matrix
            A = np.random.randn(model_order, model_order)
            A = A - np.eye(model_order) * (np.max(np.abs(np.linalg.eigvals(A))) + 0.1)
            
            # Random B, C, D matrices
            B = np.random.randn(model_order, 1)
            C = np.random.randn(1, model_order)
            D = np.zeros((1, 1))
            
            # Simulate system
            y, x = signal.lsim((A, B, C, D), u, t)
            
            # State variables
            state_variables = [x[:, i] for i in range(model_order)]
            
            # Model parameters
            model_parameters = {
                'System Order': model_order,
                'Eigenvalues': np.linalg.eigvals(A),
                'System Type': 'State-Space'
            }
        
        elif system_type == 'transfer_function':
            # Transfer function model
            # Generate random but stable transfer function
            np.random.seed(42)  # For reproducibility
            
            # Random denominator coefficients (stable)
            den = np.random.rand(model_order + 1)
            den[0] = 1  # Leading coefficient
            den = den / np.max(np.abs(np.roots(den)))  # Scale to ensure stability
            
            # Random numerator coefficients
            num = np.random.rand(model_order)
            
            # Create transfer function
            sys = signal.TransferFunction(num, den)
            
            # Simulate system
            y, x = signal.lsim(sys, u, t)
            
            # State variables (from state-space representation)
            A, B, C, D = signal.tf2ss(num, den)
            _, x = signal.lsim((A, B, C, D), u, t)
            state_variables = [x[:, i] for i in range(model_order)]
            
            # Model parameters
            model_parameters = {
                'Numerator': num,
                'Denominator': den,
                'Poles': np.roots(den),
                'Zeros': np.roots(num),
                'System Type': 'Transfer Function'
            }
        
        else:
            raise ValueError(f"Unknown system type: {system_type}")
        
        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=y,
            mode='lines',
            name='System Output',
            line=dict(color='#4CAF50', width=2)
        ))
        
        # Add state variables if available
        if len(state_variables) > 0:
            for i, state in enumerate(state_variables):
                fig.add_trace(go.Scatter(
                    x=t,
                    y=state,
                    mode='lines',
                    name=f'State Variable {i+1}',
                    line=dict(width=1),
                    visible='legendonly'
                ))
        
        fig.update_layout(
            title=f'{system_type.replace("_", " ").title()} System Model ({project_type.title()})',
            xaxis_title='Time (s)',
            yaxis_title='Output',
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white'),
            xaxis=dict(gridcolor='#444'),
            yaxis=dict(gridcolor='#444'),
            legend=dict(x=0, y=1)
        )
        
        # Format model parameters for display
        param_str = ", ".join([f"{k}: {v:.4f}" for k, v in model_parameters.items()])
        
        # Calculate model accuracy
        if system_type in ['first_order', 'second_order']:
            # For simple systems, we can calculate theoretical values
            if system_type == 'first_order':
                tau = model_parameters.get('Time Constant', 1.0)
                y_theoretical = 1.0 * (1 - np.exp(-t / tau))
            elif system_type == 'second_order':
                wn = model_parameters.get('Natural Frequency', 1.0)
                zeta = model_parameters.get('Damping Ratio', 0.7)
                
                if zeta < 1:  # Underdamped
                    wd = wn * np.sqrt(1 - zeta**2)
                    y_theoretical = 1.0 - np.exp(-zeta * wn * t) * (np.cos(wd * t) + (zeta * wn / wd) * np.sin(wd * t))
                elif zeta == 1:  # Critically damped
                    y_theoretical = 1.0 - np.exp(-wn * t) * (1 + wn * t)
                else:  # Overdamped
                    r1 = -wn * (zeta + np.sqrt(zeta**2 - 1))
                    r2 = -wn * (zeta - np.sqrt(zeta**2 - 1))
                    y_theoretical = 1.0 - (r2 * np.exp(r1 * t) - r1 * np.exp(r2 * t)) / (r2 - r1)
            
            # Calculate MSE
            mse = np.mean((y - y_theoretical)**2)
        else:
            mse = 0.01  # Placeholder
        
        properties = dbc.ListGroup([
            dbc.ListGroupItem(f"System Type: {system_type.replace('_', ' ').title()}", color="dark"),
            dbc.ListGroupItem(f"Model Order: {model_order}", color="dark"),
            dbc.ListGroupItem(f"Damping Ratio: {damping_ratio}", color="dark"),
            dbc.ListGroupItem(f"Natural Frequency: {natural_frequency} rad/s", color="dark"),
            dbc.ListGroupItem(f"Parameters: {param_str}", color="dark"),
            dbc.ListGroupItem(f"Model MSE: {mse:.6f}", color="dark"),
        ], flush=True)
        
        return fig, properties
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Generating System Model", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Generating System Model",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Analysis callbacks
@app.callback(
    Output('analysis-content', 'children'),
    [Input('analysis-tabs', 'active_tab'),
     Input('project-type-dropdown', 'value')]
)
def render_analysis_content(active_tab, project_type):
    if active_tab == "stat-analysis":
        return statistical_analysis_tab(project_type)
    elif active_tab == "time-analysis":
        return time_series_analysis_tab(project_type)
    elif active_tab == "freq-analysis":
        return frequency_analysis_tab(project_type)
    return html.Div()

def statistical_analysis_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistical Analysis Parameters"),
                    dbc.CardBody([
                        html.Label("Analysis Type", className="text-light"),
                        dcc.Dropdown(
                            id='stat-analysis-type-dropdown',
                            options=[
                                {'label': 'Basic Statistics', 'value': 'basic'},
                                {'label': 'Correlation Analysis', 'value': 'correlation'},
                                {'label': 'Hypothesis Testing', 'value': 'hypothesis'},
                                {'label': 'Regression Analysis', 'value': 'regression'}
                            ],
                            value='basic',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Data Source", className="text-light"),
                        dcc.Dropdown(
                            id='stat-data-source-dropdown',
                            options=[
                                {'label': 'Generated Data', 'value': 'generated'},
                                {'label': 'Uploaded Data', 'value': 'uploaded'},
                                {'label': 'Manual Input', 'value': 'manual'}
                            ],
                            value='generated',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Confidence Level (%)", className="text-light"),
                        dcc.Slider(
                            id='confidence-level-slider',
                            min=80,
                            max=99,
                            step=1,
                            value=95,
                            marks={i: str(i) for i in range(80, 100, 5)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Perform Analysis", id="perform-stat-analysis-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-stat-analysis-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Statistical Analysis Results"),
                    dbc.CardBody([
                        dcc.Graph(id='stat-analysis-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Summary"),
                    dbc.CardBody([
                        html.Div(id='stat-analysis-summary')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def time_series_analysis_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Time Series Analysis Parameters"),
                    dbc.CardBody([
                        html.Label("Analysis Type", className="text-light"),
                        dcc.Dropdown(
                            id='time-analysis-type-dropdown',
                            options=[
                                {'label': 'Stationarity Test', 'value': 'stationarity'},
                                {'label': 'Decomposition', 'value': 'decomposition'},
                                {'label': 'Autocorrelation', 'value': 'autocorrelation'},
                                {'label': 'ARIMA Modeling', 'value': 'arima'}
                            ],
                            value='stationarity',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Data Source", className="text-light"),
                        dcc.Dropdown(
                            id='time-data-source-dropdown',
                            options=[
                                {'label': 'Generated Data', 'value': 'generated'},
                                {'label': 'Uploaded Data', 'value': 'uploaded'},
                                {'label': 'Manual Input', 'value': 'manual'}
                            ],
                            value='generated',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Seasonality Period", className="text-light"),
                        dcc.Slider(
                            id='seasonality-slider',
                            min=1,
                            max=24,
                            step=1,
                            value=12,
                            marks={i: str(i) for i in range(0, 25, 4)},
                            className="text-light"
                        ),
                        html.Br(),
                        dbc.Button("Perform Analysis", id="perform-time-analysis-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-time-analysis-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Time Series Analysis Results"),
                    dbc.CardBody([
                        dcc.Graph(id='time-analysis-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Summary"),
                    dbc.CardBody([
                        html.Div(id='time-analysis-summary')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

def frequency_analysis_tab(project_type):
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Frequency Analysis Parameters"),
                    dbc.CardBody([
                        html.Label("Analysis Type", className="text-light"),
                        dcc.Dropdown(
                            id='freq-analysis-type-dropdown',
                            options=[
                                {'label': 'FFT Analysis', 'value': 'fft'},
                                {'label': 'Power Spectrum', 'value': 'power_spectrum'},
                                {'label': 'Spectrogram', 'value': 'spectrogram'},
                                {'label': 'Coherence Analysis', 'value': 'coherence'}
                            ],
                            value='fft',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Data Source", className="text-light"),
                        dcc.Dropdown(
                            id='freq-data-source-dropdown',
                            options=[
                                {'label': 'Generated Data', 'value': 'generated'},
                                {'label': 'Uploaded Data', 'value': 'uploaded'},
                                {'label': 'Manual Input', 'value': 'manual'}
                            ],
                            value='generated',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        html.Label("Window Function", className="text-light"),
                        dcc.Dropdown(
                            id='window-function-dropdown',
                            options=[
                                {'label': 'Rectangular', 'value': 'rectangular'},
                                {'label': 'Hanning', 'value': 'hanning'},
                                {'label': 'Hamming', 'value': 'hamming'},
                                {'label': 'Blackman', 'value': 'blackman'}
                            ],
                            value='hanning',
                            className="text-success bg-dark"
                        ),
                        html.Br(),
                        dbc.Button("Perform Analysis", id="perform-freq-analysis-btn", color="success", className="me-2"),
                        dbc.Button("Reset", id="reset-freq-analysis-btn", color="secondary"),
                    ])
                ], color="dark", inverse=True)
            ], width=4),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Frequency Analysis Results"),
                    dbc.CardBody([
                        dcc.Graph(id='freq-analysis-graph')
                    ])
                ], color="dark", inverse=True)
            ], width=8),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Analysis Summary"),
                    dbc.CardBody([
                        html.Div(id='freq-analysis-summary')
                    ])
                ], color="dark", inverse=True)
            ], width=12),
        ]),
    ])

# Statistical analysis callback
@app.callback(
    [Output('stat-analysis-graph', 'figure'),
     Output('stat-analysis-summary', 'children')],
    [Input('perform-stat-analysis-btn', 'n_clicks')],
    [State('stat-analysis-type-dropdown', 'value'),
     State('stat-data-source-dropdown', 'value'),
     State('confidence-level-slider', 'value'),
     State('hidden-data', 'children'),
     State('project-type-dropdown', 'value')]
)
def update_stat_analysis(n_clicks, analysis_type, data_source, confidence_level, hidden_data, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Generate or get data based on source
        if data_source == 'generated':
            # Generate sample data
            np.random.seed(42)
            n = 100
            
            # Adjust data based on project type
            if project_type == 'electrical':
                # Electrical data: voltage and current
                x = np.random.normal(220, 10, n)  # Voltage around 220V
                y = x / 100 + np.random.normal(0, 0.1, n)  # Current = V/R with noise
                z = x * y + np.random.normal(0, 50, n)  # Power = V*I with noise
            elif project_type == 'electronics':
                # Electronics data: small signals
                x = np.random.normal(3.3, 0.1, n)  # Voltage around 3.3V
                y = x / 1000 + np.random.normal(0, 0.001, n)  # Current in mA
                z = x * y * 1000 + np.random.normal(0, 0.5, n)  # Power in mW
            elif project_type == 'telecom':
                # Telecom data: signal strength and data rate
                x = np.random.normal(-70, 5, n)  # Signal strength in dBm
                y = 100 * (1 + (x + 70) / 30) + np.random.normal(0, 5, n)  # Data rate in Mbps
                z = y * (1 + np.random.normal(0, 0.1, n))  # Throughput with noise
            else:
                # Default data
                x = np.random.normal(size=n)
                y = 2 * x + np.random.normal(size=n)
                z = 0.5 * x - 0.5 * y + np.random.normal(size=n)
            
            data = pd.DataFrame({'x': x, 'y': y, 'z': z})
        elif data_source == 'uploaded':
            # In a real application, this would retrieve uploaded data
            # For now, we'll use sample data
            if hidden_data:
                try:
                    data = pd.read_json(hidden_data)
                except:
                    # Fallback to generated data
                    np.random.seed(42)
                    n = 100
                    x = np.random.normal(size=n)
                    y = 2 * x + np.random.normal(size=n)
                    z = 0.5 * x - 0.5 * y + np.random.normal(size=n)
                    data = pd.DataFrame({'x': x, 'y': y, 'z': z})
            else:
                # No uploaded data, use generated
                np.random.seed(42)
                n = 100
                x = np.random.normal(size=n)
                y = 2 * x + np.random.normal(size=n)
                z = 0.5 * x - 0.5 * y + np.random.normal(size=n)
                data = pd.DataFrame({'x': x, 'y': y, 'z': z})
        elif data_source == 'manual':
            # In a real application, this would retrieve manually entered data
            # For now, we'll use sample data
            np.random.seed(42)
            n = 100
            x = np.random.normal(size=n)
            y = 2 * x + np.random.normal(size=n)
            z = 0.5 * x - 0.5 * y + np.random.normal(size=n)
            data = pd.DataFrame({'x': x, 'y': y, 'z': z})
        else:
            return go.Figure(), html.Div()
        
        # Create figure based on analysis type
        fig = go.Figure()
        
        if analysis_type == 'basic':
            # Calculate basic statistics for each column
            statistics = []
            variables = []
            
            for col in data.columns:
                col_data = data[col]
                if np.issubdtype(col_data.dtype, np.number):
                    variables.append(col)
                    
                    # Calculate statistics
                    mean_val = np.mean(col_data)
                    median_val = np.median(col_data)
                    std_val = np.std(col_data)
                    min_val = np.min(col_data)
                    max_val = np.max(col_data)
                    
                    statistics.append([mean_val, median_val, std_val, min_val, max_val])
            
            # Create bar chart
            if len(statistics) > 0:
                fig.add_trace(go.Bar(
                    x=['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                    y=statistics[0],
                    name=variables[0],
                    marker_color='#4CAF50'
                ))
                
                if len(statistics) > 1:
                    fig.add_trace(go.Bar(
                        x=['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        y=statistics[1],
                        name=variables[1],
                        marker_color='#2196F3'
                    ))
                
                if len(statistics) > 2:
                    fig.add_trace(go.Bar(
                        x=['Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                        y=statistics[2],
                        name=variables[2],
                        marker_color='#FF5722'
                    ))
            
            fig.update_layout(
                title=f'Basic Statistics ({project_type.title()})',
                xaxis_title='Statistic',
                yaxis_title='Value',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444')
            )
            
            # Create summary
            if len(statistics) > 0:
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(data)}", color="dark"),
                    dbc.ListGroupItem(f"Variables: {', '.join(variables)}", color="dark"),
                    dbc.ListGroupItem(f"Mean of {variables[0]}: {statistics[0][0]:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Standard Deviation of {variables[0]}: {statistics[0][2]:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Confidence Level: {confidence_level}%", color="dark"),
                ], flush=True)
            else:
                summary = html.Div("No numeric data available for analysis", className="text-light")
        
        elif analysis_type == 'correlation':
            # Calculate correlation matrix
            corr_matrix = data.corr().values
            
            # Create heatmap
            fig.add_trace(go.Heatmap(
                z=corr_matrix,
                x=data.columns,
                y=data.columns,
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                title=f'Correlation Matrix ({project_type.title()})',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            
            # Find strongest and weakest correlations
            np.fill_diagonal(corr_matrix, np.nan)  # Ignore diagonal
            
            strongest_corr = np.nanmax(np.abs(corr_matrix))
            weakest_corr = np.nanmin(np.abs(corr_matrix))
            
            # Get indices of strongest correlation
            i, j = np.unravel_index(np.nanargmax(np.abs(corr_matrix)), corr_matrix.shape)
            strongest_pair = (data.columns[i], data.columns[j])
            
            # Get indices of weakest correlation
            i, j = np.unravel_index(np.nanargmin(np.abs(corr_matrix)), corr_matrix.shape)
            weakest_pair = (data.columns[i], data.columns[j])
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"Variables: {', '.join(data.columns)}", color="dark"),
                dbc.ListGroupItem(f"Strongest Correlation: {strongest_pair[0]} - {strongest_pair[1]}: {strongest_corr:.3f}", color="dark"),
                dbc.ListGroupItem(f"Weakest Correlation: {weakest_pair[0]} - {weakest_pair[1]}: {weakest_corr:.3f}", color="dark"),
                dbc.ListGroupItem(f"Confidence Level: {confidence_level}%", color="dark"),
            ], flush=True)
        
        elif analysis_type == 'hypothesis':
            # For simplicity, we'll perform a one-sample t-test
            # against a hypothesized mean of 0
            
            results = []
            
            for col in data.columns:
                if np.issubdtype(data[col].dtype, np.number):
                    # Perform one-sample t-test
                    t_stat, p_val = stats.ttest_1samp(data[col], 0)
                    
                    # Determine if we reject the null hypothesis
                    alpha = 1 - confidence_level / 100
                    reject_null = p_val < alpha
                    
                    results.append({
                        'variable': col,
                        'test_statistic': t_stat,
                        'p_value': p_val,
                        'reject_null': reject_null
                    })
            
            # Get the first result for display
            if results:
                result = results[0]
                
                # Create bar chart
                fig.add_trace(go.Bar(
                    x=['Test Statistic', 'Critical Value'],
                    y=[result['test_statistic'], stats.t.ppf(1 - alpha/2, len(data) - 1)],
                    name='Hypothesis Test',
                    marker_color=['#4CAF50', '#FF5722']
                ))
                
                fig.update_layout(
                    title=f'Hypothesis Test Results ({project_type.title()})',
                    xaxis_title='Value',
                    yaxis_title='Magnitude',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444')
                )
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Null Hypothesis: Mean of {result['variable']} is equal to 0", color="dark"),
                    dbc.ListGroupItem(f"Test Statistic: {result['test_statistic']:.4f}", color="dark"),
                    dbc.ListGroupItem(f"P-value: {result['p_value']:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Conclusion: {'Reject null hypothesis' if result['reject_null'] else 'Fail to reject null hypothesis'}", color="dark"),
                    dbc.ListGroupItem(f"Confidence Level: {confidence_level}%", color="dark"),
                ], flush=True)
            else:
                fig.update_layout(
                    title='Hypothesis Test Results',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white')
                )
                summary = html.Div("No numeric data available for analysis", className="text-light")
        
        elif analysis_type == 'regression':
            # For simplicity, we'll use the first column as X and the second as Y
            x_data = data.iloc[:, 0].values
            y_data = data.iloc[:, 1].values
            
            # Perform linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            
            # Calculate predicted values
            y_pred = slope * x_data + intercept
            
            # Create scatter plot with regression line
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_data,
                mode='markers',
                name='Data Points',
                marker_color='#2196F3'
            ))
            fig.add_trace(go.Scatter(
                x=x_data,
                y=y_pred,
                mode='lines',
                name='Regression Line',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title=f'Regression Analysis ({project_type.title()})',
                xaxis_title='X Variable',
                yaxis_title='Y Variable',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444'),
                legend=dict(x=0, y=1)
            )
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"R-squared: {r_value**2:.4f}", color="dark"),
                dbc.ListGroupItem(f"Slope: {slope:.4f}", color="dark"),
                dbc.ListGroupItem(f"Intercept: {intercept:.4f}", color="dark"),
                dbc.ListGroupItem(f"Standard Error: {std_err:.4f}", color="dark"),
                dbc.ListGroupItem(f"P-value: {p_value:.4f}", color="dark"),
                dbc.ListGroupItem(f"Confidence Level: {confidence_level}%", color="dark"),
            ], flush=True)
        
        else:
            fig.update_layout(
                title='Statistical Analysis',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            summary = html.Div("Unknown analysis type", className="text-light")
        
        return fig, summary
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Performing Statistical Analysis", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Performing Statistical Analysis",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Time series analysis callback
@app.callback(
    [Output('time-analysis-graph', 'figure'),
     Output('time-analysis-summary', 'children')],
    [Input('perform-time-analysis-btn', 'n_clicks')],
    [State('time-analysis-type-dropdown', 'value'),
     State('time-data-source-dropdown', 'value'),
     State('seasonality-slider', 'value'),
     State('hidden-data', 'children'),
     State('project-type-dropdown', 'value')]
)
def update_time_analysis(n_clicks, analysis_type, data_source, seasonality, hidden_data, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Generate or get data based on source
        if data_source == 'generated':
            # Generate sample time series data
            np.random.seed(42)
            n = 365  # One year of daily data
            dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
            
            # Adjust data based on project type
            if project_type == 'electrical':
                # Electrical data: power consumption with daily and weekly patterns
                trend = np.linspace(100, 120, n)  # Increasing trend
                seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Annual seasonality
                weekly = 5 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly seasonality
                noise = np.random.normal(0, 2, n)  # Random noise
                values = trend + seasonal + weekly + noise
            elif project_type == 'electronics':
                # Electronics data: device temperature
                trend = np.linspace(40, 45, n)  # Increasing trend
                seasonal = 3 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Annual seasonality
                daily = 2 * np.sin(2 * np.pi * np.arange(n) / 1)  # Daily seasonality
                noise = np.random.normal(0, 0.5, n)  # Random noise
                values = trend + seasonal + daily + noise
            elif project_type == 'telecom':
                # Telecom data: network traffic
                trend = np.linspace(50, 70, n)  # Increasing trend
                seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365.25)  # Annual seasonality
                weekly = 10 * np.sin(2 * np.pi * np.arange(n) / 7)  # Weekly seasonality
                daily = 5 * np.sin(2 * np.pi * np.arange(n) / 1)  # Daily seasonality
                noise = np.random.normal(0, 3, n)  # Random noise
                values = trend + seasonal + weekly + daily + noise
            else:
                # Default data
                trend = np.linspace(0, 10, n)
                seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
                weekly = 2 * np.sin(2 * np.pi * np.arange(n) / 7)
                noise = np.random.normal(0, 1, n)
                values = trend + seasonal + weekly + noise
            
            data = pd.DataFrame({'date': dates, 'value': values}).set_index('date')
        elif data_source == 'uploaded':
            # In a real application, this would retrieve uploaded data
            # For now, we'll use sample data
            if hidden_data:
                try:
                    data = pd.read_json(hidden_data)
                    # Try to convert to datetime index if possible
                    if 'date' in data.columns:
                        data['date'] = pd.to_datetime(data['date'])
                        data = data.set_index('date')
                except:
                    # Fallback to generated data
                    np.random.seed(42)
                    n = 365
                    dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
                    trend = np.linspace(0, 10, n)
                    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
                    weekly = 2 * np.sin(2 * np.pi * np.arange(n) / 7)
                    noise = np.random.normal(0, 1, n)
                    values = trend + seasonal + weekly + noise
                    data = pd.DataFrame({'date': dates, 'value': values}).set_index('date')
            else:
                # No uploaded data, use generated
                np.random.seed(42)
                n = 365
                dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
                trend = np.linspace(0, 10, n)
                seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
                weekly = 2 * np.sin(2 * np.pi * np.arange(n) / 7)
                noise = np.random.normal(0, 1, n)
                values = trend + seasonal + weekly + noise
                data = pd.DataFrame({'date': dates, 'value': values}).set_index('date')
        elif data_source == 'manual':
            # In a real application, this would retrieve manually entered data
            # For now, we'll use sample data
            np.random.seed(42)
            n = 365
            dates = pd.date_range(start='2020-01-01', periods=n, freq='D')
            trend = np.linspace(0, 10, n)
            seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
            weekly = 2 * np.sin(2 * np.pi * np.arange(n) / 7)
            noise = np.random.normal(0, 1, n)
            values = trend + seasonal + weekly + noise
            data = pd.DataFrame({'date': dates, 'value': values}).set_index('date')
        else:
            return go.Figure(), html.Div()
        
        # Extract values and time
        values = data.values.flatten()
        time = np.arange(len(values))
        
        # Create figure based on analysis type
        fig = go.Figure()
        
        if analysis_type == 'stationarity':
            # Plot the time series
            fig.add_trace(go.Scatter(
                x=time,
                y=values,
                mode='lines',
                name='Time Series',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title=f'Time Series Data ({project_type.title()})',
                xaxis_title='Time',
                yaxis_title='Value',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444')
            )
            
            # Perform Augmented Dickey-Fuller test
            try:
                from statsmodels.tsa.stattools import adfuller
                result = adfuller(values)
                
                # Extract test statistics and p-value
                adf_statistic = result[0]
                p_value = result[1]
                critical_values = result[4]
                
                # Determine conclusion
                is_stationary = p_value < 0.05  # Using 5% significance level
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"ADF Statistic: {adf_statistic:.4f}", color="dark"),
                    dbc.ListGroupItem(f"P-value: {p_value:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Critical Values: {critical_values}", color="dark"),
                    dbc.ListGroupItem(f"Conclusion: {'Stationary' if is_stationary else 'Non-stationary'}", color="dark"),
                ], flush=True)
            except ImportError:
                # If statsmodels is not available, use a simple test
                # Calculate mean and variance of first and second half
                n_half = len(values) // 2
                mean1 = np.mean(values[:n_half])
                mean2 = np.mean(values[n_half:])
                var1 = np.var(values[:n_half])
                var2 = np.var(values[n_half:])
                
                # Simple test for stationarity
                mean_diff = abs(mean1 - mean2) / (abs(mean1) + abs(mean2) + 1e-10)
                var_diff = abs(var1 - var2) / (abs(var1) + abs(var2) + 1e-10)
                is_stationary = mean_diff < 0.1 and var_diff < 0.1
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"Mean Difference: {mean_diff:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Variance Difference: {var_diff:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Conclusion: {'Stationary' if is_stationary else 'Non-stationary'}", color="dark"),
                ], flush=True)
        
        elif analysis_type == 'decomposition':
            # Perform seasonal decomposition
            try:
                from statsmodels.tsa.seasonal import seasonal_decompose
                
                # Perform decomposition
                decomposition = seasonal_decompose(values, model='additive', period=seasonality)
                
                # Extract components
                trend = decomposition.trend
                seasonal = decomposition.seasonal
                residual = decomposition.resid
                
                # Plot components
                fig.add_trace(go.Scatter(
                    x=time,
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#4CAF50', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=time,
                    y=seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='#2196F3', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=time,
                    y=residual,
                    mode='lines',
                    name='Residual',
                    line=dict(color='#FF5722', width=1)
                ))
                
                fig.update_layout(
                    title=f'Time Series Decomposition ({project_type.title()})',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444'),
                    legend=dict(x=0, y=1)
                )
                
                # Calculate strength of components
                trend_strength = 1 - np.var(residual[~np.isnan(residual)]) / np.var(trend[~np.isnan(trend)] + residual[~np.isnan(residual)])
                seasonal_strength = 1 - np.var(residual[~np.isnan(residual)]) / np.var(seasonal[~np.isnan(seasonal)] + residual[~np.isnan(residual)])
                residual_variance = np.var(residual[~np.isnan(residual)])
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"Seasonality Period: {seasonality}", color="dark"),
                    dbc.ListGroupItem(f"Trend Strength: {trend_strength:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Seasonality Strength: {seasonal_strength:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Residual Variance: {residual_variance:.4f}", color="dark"),
                ], flush=True)
            except ImportError:
                # If statsmodels is not available, use a simple decomposition
                # Simple moving average for trend
                window = seasonality
                trend = np.convolve(values, np.ones(window)/window, mode='same')
                
                # Simple seasonal component
                seasonal = np.zeros_like(values)
                for i in range(seasonality):
                    seasonal[i::seasonality] = np.mean(values[i::seasonality] - trend[i::seasonality])
                
                # Residual
                residual = values - trend - seasonal
                
                # Plot components
                fig.add_trace(go.Scatter(
                    x=time,
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#4CAF50', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=time,
                    y=seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='#2196F3', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=time,
                    y=residual,
                    mode='lines',
                    name='Residual',
                    line=dict(color='#FF5722', width=1)
                ))
                
                fig.update_layout(
                    title=f'Time Series Decomposition ({project_type.title()})',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444'),
                    legend=dict(x=0, y=1)
                )
                
                # Calculate strength of components
                trend_strength = 1 - np.var(residual) / (np.var(trend) + np.var(residual) + 1e-10)
                seasonal_strength = 1 - np.var(residual) / (np.var(seasonal) + np.var(residual) + 1e-10)
                residual_variance = np.var(residual)
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"Seasonality Period: {seasonality}", color="dark"),
                    dbc.ListGroupItem(f"Trend Strength: {trend_strength:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Seasonality Strength: {seasonal_strength:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Residual Variance: {residual_variance:.4f}", color="dark"),
                ], flush=True)
        
        elif analysis_type == 'autocorrelation':
            # Calculate autocorrelation function
            max_lag = min(40, len(values) // 4)  # Use a reasonable number of lags
            
            try:
                from statsmodels.tsa.stattools import acf, pacf
                
                # Calculate ACF and PACF
                autocorr = acf(values, nlags=max_lag, fft=True)
                partial_autocorr = pacf(values, nlags=max_lag)
                
                # Find significant lags
                # Using Bartlett's formula for standard error
                se = 1 / np.sqrt(len(values))
                significant_lags = np.where(np.abs(autocorr[1:]) > 1.96 * se)[0] + 1
                
                # Plot ACF
                fig.add_trace(go.Scatter(
                    x=np.arange(len(autocorr)),
                    y=autocorr,
                    mode='markers',
                    name='Autocorrelation',
                    marker_color='#4CAF50'
                ))
                
                # Add confidence intervals
                fig.add_trace(go.Scatter(
                    x=np.arange(len(autocorr)),
                    y=1.96 * se * np.ones_like(autocorr),
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='95% CI',
                    showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=np.arange(len(autocorr)),
                    y=-1.96 * se * np.ones_like(autocorr),
                    mode='lines',
                    line=dict(dash='dash', color='red'),
                    name='95% CI',
                    showlegend=False
                ))
                
                fig.update_layout(
                    title=f'Autocorrelation Function ({project_type.title()})',
                    xaxis_title='Lag',
                    yaxis_title='Autocorrelation',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444')
                )
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"Max Lag: {max_lag}", color="dark"),
                    dbc.ListGroupItem(f"Significant Lags: {significant_lags}", color="dark"),
                    dbc.ListGroupItem(f"Autocorrelation at Lag 1: {autocorr[1]:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Partial Autocorrelation at Lag 1: {partial_autocorr[1]:.4f}", color="dark"),
                ], flush=True)
            except ImportError:
                # If statsmodels is not available, use a simple autocorrelation calculation
                autocorr = np.zeros(max_lag + 1)
                
                for lag in range(max_lag + 1):
                    if lag == 0:
                        autocorr[lag] = 1.0
                    else:
                        autocorr[lag] = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                
                # Plot ACF
                fig.add_trace(go.Scatter(
                    x=np.arange(len(autocorr)),
                    y=autocorr,
                    mode='markers',
                    name='Autocorrelation',
                    marker_color='#4CAF50'
                ))
                
                fig.update_layout(
                    title=f'Autocorrelation Function ({project_type.title()})',
                    xaxis_title='Lag',
                    yaxis_title='Autocorrelation',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444')
                )
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                    dbc.ListGroupItem(f"Max Lag: {max_lag}", color="dark"),
                    dbc.ListGroupItem(f"Autocorrelation at Lag 1: {autocorr[1]:.4f}", color="dark"),
                ], flush=True)
        
        elif analysis_type == 'arima':
            # Fit ARIMA model
            try:
                from statsmodels.tsa.arima.model import ARIMA
                
                # Split data into train and test sets
                train_size = int(len(values) * 0.8)
                train, test = values[:train_size], values[train_size:]
                
                # Fit ARIMA model
                model = ARIMA(train, order=(1, 1, 1))
                model_fit = model.fit()
                
                # Make predictions
                forecast_steps = len(test)
                forecast = model_fit.forecast(steps=forecast_steps)
                
                # Get fitted values
                fitted = model_fit.fittedvalues
                
                # Calculate error metrics
                residuals = test - forecast
                mse = np.mean(residuals**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Get model parameters
                aic = model_fit.aic
                bic = model_fit.bic
                
                # Extract model order
                order = model_fit.model.order
                
                # Create time arrays
                train_time = np.arange(len(train))
                test_time = np.arange(len(train), len(train) + len(test))
                forecast_time = np.arange(len(train), len(train) + len(forecast))
                
                # Plot results
                fig.add_trace(go.Scatter(
                    x=train_time,
                    y=train,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='#4CAF50', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=test_time,
                    y=test,
                    mode='lines',
                    name='Test Data',
                    line=dict(color='#2196F3', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#FF5722', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'ARIMA Model ({project_type.title()})',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444'),
                    legend=dict(x=0, y=1)
                )
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"ARIMA Order: ({order[0]}, {order[1]}, {order[2]})", color="dark"),
                    dbc.ListGroupItem(f"AIC: {aic:.4f}", color="dark"),
                    dbc.ListGroupItem(f"BIC: {bic:.4f}", color="dark"),
                    dbc.ListGroupItem(f"RMSE: {rmse:.4f}", color="dark"),
                    dbc.ListGroupItem(f"MAE: {mae:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Forecast Period: {forecast_steps} steps", color="dark"),
                ], flush=True)
            except ImportError:
                # If statsmodels is not available, use a simple forecast
                # Split data into train and test sets
                train_size = int(len(values) * 0.8)
                train, test = values[:train_size], values[train_size:]
                
                # Simple exponential smoothing
                alpha = 0.3
                fitted = np.zeros_like(train)
                fitted[0] = train[0]
                
                for i in range(1, len(train)):
                    fitted[i] = alpha * train[i] + (1 - alpha) * fitted[i-1]
                
                # Forecast
                forecast_steps = len(test)
                forecast = np.ones(forecast_steps) * fitted[-1]
                
                # Calculate error metrics
                residuals = test - forecast
                mse = np.mean(residuals**2)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(residuals))
                
                # Create time arrays
                train_time = np.arange(len(train))
                test_time = np.arange(len(train), len(train) + len(test))
                forecast_time = np.arange(len(train), len(train) + len(forecast))
                
                # Plot results
                fig.add_trace(go.Scatter(
                    x=train_time,
                    y=train,
                    mode='lines',
                    name='Training Data',
                    line=dict(color='#4CAF50', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=test_time,
                    y=test,
                    mode='lines',
                    name='Test Data',
                    line=dict(color='#2196F3', width=2)
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_time,
                    y=forecast,
                    mode='lines',
                    name='Forecast',
                    line=dict(color='#FF5722', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Simple Forecast Model ({project_type.title()})',
                    xaxis_title='Time',
                    yaxis_title='Value',
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font=dict(color='white'),
                    xaxis=dict(gridcolor='#444'),
                    yaxis=dict(gridcolor='#444'),
                    legend=dict(x=0, y=1)
                )
                
                # Create summary
                summary = dbc.ListGroup([
                    dbc.ListGroupItem(f"Smoothing Parameter: {alpha:.4f}", color="dark"),
                    dbc.ListGroupItem(f"RMSE: {rmse:.4f}", color="dark"),
                    dbc.ListGroupItem(f"MAE: {mae:.4f}", color="dark"),
                    dbc.ListGroupItem(f"Forecast Period: {forecast_steps} steps", color="dark"),
                ], flush=True)
        
        else:
            fig.update_layout(
                title='Time Series Analysis',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            summary = html.Div("Unknown analysis type", className="text-light")
        
        return fig, summary
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Performing Time Series Analysis", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Performing Time Series Analysis",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Frequency analysis callback
@app.callback(
    [Output('freq-analysis-graph', 'figure'),
     Output('freq-analysis-summary', 'children')],
    [Input('perform-freq-analysis-btn', 'n_clicks')],
    [State('freq-analysis-type-dropdown', 'value'),
     State('freq-data-source-dropdown', 'value'),
     State('window-function-dropdown', 'value'),
     State('hidden-data', 'children'),
     State('project-type-dropdown', 'value')]
)
def update_freq_analysis(n_clicks, analysis_type, data_source, window_function, hidden_data, project_type):
    if n_clicks is None:
        return go.Figure(), html.Div()
    
    try:
        # Generate or get data based on source
        if data_source == 'generated':
            # Generate sample frequency data
            np.random.seed(42)
            sampling_rate = 1000  # Hz
            duration = 1  # second
            n = int(sampling_rate * duration)
            time = np.linspace(0, duration, n, endpoint=False)
            
            # Adjust signal based on project type
            if project_type == 'electrical':
                # Electrical signal: 50Hz power line with harmonics
                signal_values = (
                    1.0 * np.sin(2 * np.pi * 50 * time) +    # 50 Hz fundamental
                    0.2 * np.sin(2 * np.pi * 150 * time) +   # 3rd harmonic
                    0.1 * np.sin(2 * np.pi * 250 * time) +   # 5th harmonic
                    0.05 * np.sin(2 * np.pi * 350 * time) +  # 7th harmonic
                    0.1 * np.random.normal(size=n)           # Noise
                )
            elif project_type == 'electronics':
                # Electronics signal: digital communication signal
                carrier_freq = 100  # Hz
                bit_rate = 10       # bits per second
                bits = np.random.randint(0, 2, int(bit_rate * duration))
                
                # Create digital signal
                digital_signal = np.zeros_like(time)
                for i, bit in enumerate(bits):
                    start_idx = int(i * duration / len(bits) * sampling_rate)
                    end_idx = int((i + 1) * duration / len(bits) * sampling_rate)
                    digital_signal[start_idx:end_idx] = bit
                
                # Modulate with carrier
                signal_values = digital_signal * np.sin(2 * np.pi * carrier_freq * time)
                signal_values += 0.1 * np.random.normal(size=n)  # Add noise
            elif project_type == 'telecom':
                # Telecom signal: multi-frequency signal
                signal_values = (
                    1.0 * np.sin(2 * np.pi * 100 * time) +   # 100 Hz component
                    0.8 * np.sin(2 * np.pi * 200 * time) +   # 200 Hz component
                    0.6 * np.sin(2 * np.pi * 300 * time) +   # 300 Hz component
                    0.4 * np.sin(2 * np.pi * 400 * time) +   # 400 Hz component
                    0.2 * np.sin(2 * np.pi * 500 * time) +   # 500 Hz component
                    0.1 * np.random.normal(size=n)           # Noise
                )
            else:
                # Default signal
                signal_values = (
                    1.0 * np.sin(2 * np.pi * 10 * time) +   # 10 Hz component
                    0.5 * np.sin(2 * np.pi * 20 * time) +   # 20 Hz component
                    0.2 * np.sin(2 * np.pi * 50 * time) +   # 50 Hz component
                    0.1 * np.random.normal(size=n)           # Noise
                )
            
            data = pd.DataFrame({'time': time, 'value': signal_values})
        elif data_source == 'uploaded':
            # In a real application, this would retrieve uploaded data
            # For now, we'll use sample data
            if hidden_data:
                try:
                    data = pd.read_json(hidden_data)
                    # Try to extract time and value columns
                    if 'time' in data.columns and 'value' in data.columns:
                        time = data['time'].values
                        signal_values = data['value'].values
                    else:
                        # Use first column as time, second as value
                        time = data.iloc[:, 0].values
                        signal_values = data.iloc[:, 1].values
                except:
                    # Fallback to generated data
                    np.random.seed(42)
                    sampling_rate = 1000  # Hz
                    duration = 1  # second
                    n = int(sampling_rate * duration)
                    time = np.linspace(0, duration, n, endpoint=False)
                    signal_values = (
                        1.0 * np.sin(2 * np.pi * 10 * time) +   # 10 Hz component
                        0.5 * np.sin(2 * np.pi * 20 * time) +   # 20 Hz component
                        0.2 * np.sin(2 * np.pi * 50 * time) +   # 50 Hz component
                        0.1 * np.random.normal(size=n)           # Noise
                    )
                    data = pd.DataFrame({'time': time, 'value': signal_values})
            else:
                # No uploaded data, use generated
                np.random.seed(42)
                sampling_rate = 1000  # Hz
                duration = 1  # second
                n = int(sampling_rate * duration)
                time = np.linspace(0, duration, n, endpoint=False)
                signal_values = (
                    1.0 * np.sin(2 * np.pi * 10 * time) +   # 10 Hz component
                    0.5 * np.sin(2 * np.pi * 20 * time) +   # 20 Hz component
                    0.2 * np.sin(2 * np.pi * 50 * time) +   # 50 Hz component
                    0.1 * np.random.normal(size=n)           # Noise
                )
                data = pd.DataFrame({'time': time, 'value': signal_values})
        elif data_source == 'manual':
            # In a real application, this would retrieve manually entered data
            # For now, we'll use sample data
            np.random.seed(42)
            sampling_rate = 1000  # Hz
            duration = 1  # second
            n = int(sampling_rate * duration)
            time = np.linspace(0, duration, n, endpoint=False)
            signal_values = (
                1.0 * np.sin(2 * np.pi * 10 * time) +   # 10 Hz component
                0.5 * np.sin(2 * np.pi * 20 * time) +   # 20 Hz component
                0.2 * np.sin(2 * np.pi * 50 * time) +   # 50 Hz component
                0.1 * np.random.normal(size=n)           # Noise
            )
            data = pd.DataFrame({'time': time, 'value': signal_values})
        else:
            return go.Figure(), html.Div()
        
        # Extract time and values
        time = data['time'].values
        values = data['value'].values
        
        # Sampling rate
        sampling_rate = 1 / (time[1] - time[0])
        
        # Create figure based on analysis type
        fig = go.Figure()
        
        if analysis_type == 'fft':
            # Apply window function
            if window_function == 'rectangular':
                window = np.ones_like(values)
            elif window_function == 'hanning':
                window = np.hanning(len(values))
            elif window_function == 'hamming':
                window = np.hamming(len(values))
            elif window_function == 'blackman':
                window = np.blackman(len(values))
            else:
                window = np.ones_like(values)
            
            # Apply window
            windowed_values = values * window
            
            # Perform FFT
            fft_values = np.fft.rfft(windowed_values)
            
            # Calculate magnitude
            magnitude = np.abs(fft_values)
            
            # Calculate frequencies
            frequencies = np.fft.rfftfreq(len(values), 1/sampling_rate)
            
            # Plot magnitude spectrum
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=magnitude,
                mode='lines',
                name='FFT Magnitude',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title=f'FFT Analysis ({project_type.title()})',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Magnitude',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444')
            )
            
            # Find peak frequency and magnitude
            peak_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
            peak_freq = frequencies[peak_idx]
            peak_magnitude = magnitude[peak_idx]
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                dbc.ListGroupItem(f"Sampling Rate: {sampling_rate} Hz", color="dark"),
                dbc.ListGroupItem(f"Window Function: {window_function}", color="dark"),
                dbc.ListGroupItem(f"Peak Frequency: {peak_freq:.2f} Hz", color="dark"),
                dbc.ListGroupItem(f"Peak Magnitude: {peak_magnitude:.2f}", color="dark"),
            ], flush=True)
        
        elif analysis_type == 'power_spectrum':
            # Apply window function
            if window_function == 'rectangular':
                window = 'rectangular'
            elif window_function == 'hanning':
                window = 'hanning'
            elif window_function == 'hamming':
                window = 'hamming'
            elif window_function == 'blackman':
                window = 'blackman'
            else:
                window = 'hanning'
            
            # Calculate power spectrum using Welch's method
            frequencies, power = signal.welch(
                values, 
                fs=sampling_rate, 
                window=window_function,
                nperseg=min(256, len(values))
            )
            
            # Plot power spectrum
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=power,
                mode='lines',
                name='Power Spectrum',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title=f'Power Spectrum ({project_type.title()})',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Power',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444')
            )
            
            # Find peak frequency and power
            peak_idx = np.argmax(power)
            peak_freq = frequencies[peak_idx]
            peak_power = power[peak_idx]
            
            # Calculate total power
            total_power = np.sum(power)
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                dbc.ListGroupItem(f"Sampling Rate: {sampling_rate} Hz", color="dark"),
                dbc.ListGroupItem(f"Window Function: {window_function}", color="dark"),
                dbc.ListGroupItem(f"Peak Frequency: {peak_freq:.2f} Hz", color="dark"),
                dbc.ListGroupItem(f"Peak Power: {peak_power:.2f}", color="dark"),
                dbc.ListGroupItem(f"Total Power: {total_power:.2f}", color="dark"),
            ], flush=True)
        
        elif analysis_type == 'spectrogram':
            # Apply window function
            if window_function == 'rectangular':
                window = 'rectangular'
            elif window_function == 'hanning':
                window = 'hanning'
            elif window_function == 'hamming':
                window = 'hamming'
            elif window_function == 'blackman':
                window = 'blackman'
            else:
                window = 'hanning'
            
            # Parameters for spectrogram
            nperseg = 256  # Length of each segment
            noverlap = nperseg // 2  # Overlap between segments
            
            # Calculate spectrogram
            frequencies, times, Sxx = signal.spectrogram(
                values, 
                fs=sampling_rate, 
                window=window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Plot spectrogram
            fig.add_trace(go.Heatmap(
                x=times,
                y=frequencies,
                z=Sxx,
                colorscale='Viridis'
            ))
            
            fig.update_layout(
                title=f'Spectrogram ({project_type.title()})',
                xaxis_title='Time (s)',
                yaxis_title='Frequency (Hz)',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            
            # Calculate time and frequency resolution
            time_res = times[1] - times[0] if len(times) > 1 else 0
            freq_res = frequencies[1] - frequencies[0] if len(frequencies) > 1 else 0
            
            # Calculate overlap percentage
            overlap = noverlap / nperseg
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                dbc.ListGroupItem(f"Sampling Rate: {sampling_rate} Hz", color="dark"),
                dbc.ListGroupItem(f"Window Function: {window_function}", color="dark"),
                dbc.ListGroupItem(f"Time Resolution: {time_res:.3f} s", color="dark"),
                dbc.ListGroupItem(f"Frequency Resolution: {freq_res:.3f} Hz", color="dark"),
                dbc.ListGroupItem(f"Overlap: {overlap*100:.0f}%", color="dark"),
            ], flush=True)
        
        elif analysis_type == 'coherence':
            # Create two signals for coherence analysis
            # Signal 1: Original signal
            signal1 = values
            
            # Signal 2: Filtered version of the original signal
            # Apply a simple high-pass filter
            b, a = signal.butter(4, 0.2, btype='high')
            signal2 = signal.filtfilt(b, a, values)
            
            # Apply window function
            if window_function == 'rectangular':
                window = 'rectangular'
            elif window_function == 'hanning':
                window = 'hanning'
            elif window_function == 'hamming':
                window = 'hamming'
            elif window_function == 'blackman':
                window = 'blackman'
            else:
                window = 'hanning'
            
            # Calculate coherence
            frequencies, coherence = signal.coherence(
                signal1, signal2, 
                fs=sampling_rate, 
                window=window,
                nperseg=min(256, len(values))
            )
            
            # Plot coherence
            fig.add_trace(go.Scatter(
                x=frequencies,
                y=coherence,
                mode='lines',
                name='Coherence',
                line=dict(color='#4CAF50', width=2)
            ))
            
            fig.update_layout(
                title=f'Coherence Analysis ({project_type.title()})',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Coherence',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white'),
                xaxis=dict(gridcolor='#444'),
                yaxis=dict(gridcolor='#444')
            )
            
            # Calculate average coherence
            avg_coherence = np.mean(coherence)
            
            # Find peak coherence and frequency
            peak_idx = np.argmax(coherence)
            peak_coherence = coherence[peak_idx]
            peak_freq = frequencies[peak_idx]
            
            # Create summary
            summary = dbc.ListGroup([
                dbc.ListGroupItem(f"Data Points: {len(values)}", color="dark"),
                dbc.ListGroupItem(f"Sampling Rate: {sampling_rate} Hz", color="dark"),
                dbc.ListGroupItem(f"Window Function: {window_function}", color="dark"),
                dbc.ListGroupItem(f"Average Coherence: {avg_coherence:.3f}", color="dark"),
                dbc.ListGroupItem(f"Peak Coherence: {peak_coherence:.3f}", color="dark"),
                dbc.ListGroupItem(f"Peak Frequency: {peak_freq:.2f} Hz", color="dark"),
            ], flush=True)
        
        else:
            fig.update_layout(
                title='Frequency Analysis',
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font=dict(color='white')
            )
            summary = html.Div("Unknown analysis type", className="text-light")
        
        return fig, summary
    except Exception as e:
        error_msg = html.Div([
            html.H4("Error Performing Frequency Analysis", className="text-danger"),
            html.P(str(e), className="text-light")
        ])
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Error Performing Frequency Analysis",
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font=dict(color='white')
        )
        return empty_fig, error_msg

# Data upload callbacks
@app.callback(
    [Output('upload-output', 'children'),
     Output('hidden-data', 'children')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_upload_output(contents, filename, last_modified):
    if contents is None:
        return html.Div(), None
    
    try:
        # Process the uploaded file
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        # Determine file type and process accordingly
        if filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.xlsx') or filename.endswith('.xls'):
            df = pd.read_excel(io.BytesIO(decoded))
        elif filename.endswith('.json'):
            df = pd.read_json(io.StringIO(decoded.decode('utf-8')))
        elif filename.endswith('.txt'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')), delimiter='\t')
        else:
            return html.Div([
                html.H5(f"Unsupported file type: {filename}"),
                html.P("Please upload a CSV, Excel, JSON, or TXT file.")
            ], className="alert alert-danger"), None
        
        # Store the data in a hidden div for later use
        data_json = df.to_json(orient='records')
        
        return html.Div([
            html.H5(f"File {filename} uploaded successfully!"),
            html.P(f"File contains {len(df)} rows and {len(df.columns)} columns."),
            html.P(f"Last modified: {dt.fromtimestamp(last_modified)}"),
            html.Hr(),
            html.H6("Preview of the data:"),
            html.Div([
                dash_table.DataTable(
                    data=df.head(10).to_dict('records'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'auto'},
                    style_header={
                        'backgroundColor': '#1E1E1E',
                        'color': 'white',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'backgroundColor': '#1E1E1E',
                        'color': 'white',
                        'textAlign': 'left'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': '#2E2E2E'
                        }
                    ]
                )
            ]),
            dbc.Button("Use This Data", id="use-uploaded-data-btn", color="success", className="mt-2")
        ], className="alert alert-success"), data_json
        
    except Exception as e:
        return html.Div([
            html.H5("Error processing file"),
            html.P(str(e))
        ], className="alert alert-danger"), None

@app.callback(
    [Output('data-preview', 'children'),
     Output('hidden-data', 'children', allow_duplicate=True)],
    [Input('process-manual-btn', 'n_clicks'),
     Input('use-uploaded-data-btn', 'n_clicks')],
    [State('manual-input', 'value'),
     State('hidden-data', 'children')],
    prevent_initial_call=True
)
def update_data_preview(process_click, use_click, manual_input, hidden_data):
    ctx = dash.callback_context
    if not ctx.triggered:
        return html.Div(), hidden_data
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'process-manual-btn' and manual_input:
        try:
            # Process manual input
            df = process_manual_input(manual_input)
            data_json = df.to_json(orient='records')
            return create_data_preview(df), data_json
        except Exception as e:
            return html.Div([
                html.H5("Error processing manual input"),
                html.P(str(e))
            ], className="alert alert-danger"), hidden_data
    
    elif button_id == 'use-uploaded-data-btn' and hidden_data:
        try:
            df = pd.read_json(hidden_data)
            return create_data_preview(df), hidden_data
        except Exception as e:
            return html.Div([
                html.H5("Error using uploaded data"),
                html.P(str(e))
            ], className="alert alert-danger"), hidden_data
    
    return html.Div(), hidden_data

def create_data_preview(data):
    return html.Div([
        html.H5("Data Preview"),
        html.P(f"Data contains {len(data)} rows and {len(data.columns)} columns."),
        html.Div([
            dash_table.DataTable(
                data=data.head(10).to_dict('records'),
                columns=[{'name': i, 'id': i} for i in data.columns],
                style_table={'overflowX': 'auto'},
                style_header={
                    'backgroundColor': '#1E1E1E',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_cell={
                    'backgroundColor': '#1E1E1E',
                    'color': 'white',
                    'textAlign': 'left'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#2E2E2E'
                    }
                ]
            )
        ]),
        dbc.Button("Analyze This Data", id="analyze-data-btn", color="success", className="mt-2")
    ], className="alert alert-info")

def process_manual_input(input_text):
    """
    Process manually entered text data.
    
    Args:
        input_text: Text input from user
        
    Returns:
        DataFrame containing the processed data
    """
    try:
        # Split by lines and then by commas or tabs
        lines = input_text.strip().split('\n')
        data = []
        
        for line in lines:
            if ',' in line:
                values = line.split(',')
            else:
                values = line.split('\t')
            
            # Try to convert to numeric values
            row = []
            for value in values:
                try:
                    row.append(float(value))
                except ValueError:
                    row.append(value.strip())
            
            data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error processing manual input: {str(e)}")

# Export data callback
@app.callback(
    Output('hidden-data', 'children', allow_duplicate=True),
    [Input('export-data-btn', 'n_clicks')],
    [State('hidden-data', 'children')],
    prevent_initial_call=True
)
def export_data(n_clicks, hidden_data):
    if n_clicks is None:
        return hidden_data
    
    # In a real application, this would trigger a download of the data
    # For now, we'll just show a confirmation message
    return json.dumps({
        'message': 'Data exported successfully',
        'timestamp': dt.now().isoformat()
    })

# Export charts callback
@app.callback(
    Output('hidden-data', 'children', allow_duplicate=True),
    [Input('export-charts-btn', 'n_clicks')],
    [State('hidden-data', 'children')],
    prevent_initial_call=True
)
def export_charts(n_clicks, hidden_data):
    if n_clicks is None:
        return hidden_data
    
    # In a real application, this would trigger a download of the charts
    # For now, we'll just show a confirmation message
    return json.dumps({
        'message': 'Charts exported successfully',
        'timestamp': dt.now().isoformat()
    })

# Reset callbacks for all modules
@app.callback(
    [Output('amplitude-slider', 'value'),
     Output('frequency-slider', 'value'),
     Output('duration-slider', 'value')],
    [Input('reset-signal-btn', 'n_clicks')]
)
def reset_signal_parameters(n_clicks):
    if n_clicks:
        return 5, 1, 5
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('input-voltage-slider', 'value'),
     Output('resistance-slider', 'value'),
     Output('capacitance-slider', 'value'),
     Output('inductance-slider', 'value')],
    [Input('reset-circuit-btn', 'n_clicks')]
)
def reset_circuit_parameters(n_clicks):
    if n_clicks:
        return 5, 1000, 100, 100
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('kp-slider', 'value'),
     Output('ki-slider', 'value'),
     Output('kd-slider', 'value'),
     Output('setpoint-slider', 'value')],
    [Input('reset-system-btn', 'n_clicks')]
)
def reset_system_parameters(n_clicks):
    if n_clicks:
        return 1, 0.1, 0.01, 5
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    [Output('model-order-slider', 'value'),
     Output('data-points-slider', 'value')],
    [Input('reset-signal-model-btn', 'n_clicks')]
)
def reset_signal_model_parameters(n_clicks):
    if n_clicks:
        return 3, 100
    return dash.no_update, dash.no_update

@app.callback(
    [Output('model-complexity-slider', 'value'),
     Output('temperature-slider', 'value')],
    [Input('reset-circuit-model-btn', 'n_clicks')]
)
def reset_circuit_model_parameters(n_clicks):
    if n_clicks:
        return 3, 25
    return dash.no_update, dash.no_update

@app.callback(
    [Output('system-model-order-slider', 'value'),
     Output('damping-ratio-slider', 'value'),
     Output('natural-frequency-slider', 'value')],
    [Input('reset-system-model-btn', 'n_clicks')]
)
def reset_system_model_parameters(n_clicks):
    if n_clicks:
        return 2, 0.7, 1
    return dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output('manual-input', 'value'),
    [Input('clear-manual-btn', 'n_clicks')]
)
def clear_manual_input(n_clicks):
    if n_clicks:
        return ''
    return dash.no_update

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
