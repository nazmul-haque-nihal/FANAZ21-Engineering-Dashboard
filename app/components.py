# app/components.py
import dash_bootstrap_components as dbc
from dash import html, dcc, upload
from flask_login import current_user
from app.auth import is_admin, is_engineer, is_analyst
from app.config import Config

def create_header():
    return dbc.Navbar(
        dbc.Container([
            html.A(
                dbc.Row([
                    dbc.Col(html.Img(src="/assets/images/logo.png", height="30px")),
                    dbc.Col(dbc.NavbarBrand("FANAZ21 Engineering Dashboard", className="ms-2")),
                ], align="center", className="g-0"),
                href="/",
                style={"textDecoration": "none"}
            ),
            dbc.NavbarToggler(id="navbar-toggler"),
            dbc.Collapse(
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
                    dbc.NavItem(dbc.NavLink("Simulation", href="/simulation")),
                    dbc.NavItem(dbc.NavLink("Analysis", href="/analysis")),
                    dbc.NavItem(dbc.NavLink("Modeling", href="/modeling")),
                    dbc.NavItem(dbc.NavLink("Logout", href="/logout")),
                ], className="ms-auto", navbar=True),
                id="navbar-collapse",
                navbar=True,
            ),
        ]),
        color=Config.THEME_COLOR,
        dark=True,
        className="mb-4"
    )

def create_upload_card(module_type):
    return dbc.Card([
        dbc.CardHeader([
            html.H4(f"Upload Data for {module_type.capitalize()}"),
            dcc.Upload(
                id=f'upload-{module_type}',
                children=html.Div([
                    html.A('Select Files')
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
                    'color': 'white'
                },
                multiple=False
            ),
            html.Div(id=f'upload-status-{module_type}')
        ], className="text-center")
    ], className="mb-4")

def create_manual_input_card(module_type):
    return dbc.Card([
        dbc.CardHeader([
            html.H4(f"Manual Input for {module_type.capitalize()}"),
            dbc.Row([
                dbc.Col([
                    html.Label("Input Type"),
                    dcc.Dropdown(
                        id=f'manual-input-type-{module_type}',
                        options=[
                            {'label': 'Sine Wave', 'value': 'sine'},
                            {'label': 'Square Wave', 'value': 'square'},
                            {'label': 'Sawtooth Wave', 'value': 'sawtooth'},
                            {'label': 'Noise', 'value': 'noise'},
                            {'label': 'Composite', 'value': 'composite'}
                        ],
                        value='sine',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Sample Rate (Hz)"),
                    dbc.Input(
                        id=f'sample-rate-{module_type}',
                        type="number",
                        value=1000,
                        min=1,
                        max=10000,
                        className="mb-3"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Duration (s)"),
                    dbc.Input(
                        id=f'duration-{module_type}',
                        type="number",
                        value=1.0,
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Amplitude"),
                    dbc.Input(
                        id=f'amplitude-{module_type}',
                        type="number",
                        value=1.0,
                        min=0.1,
                        max=10.0,
                        step=0.1,
                        className="mb-3"
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Frequency (Hz)"),
                    dbc.Input(
                        id=f'frequency-{module_type}',
                        type="number",
                        value=5.0,
                        min=0.1,
                        max=100.0,
                        step=0.1,
                        className="mb-3"
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Phase (rad)"),
                    dbc.Input(
                        id=f'phase-{module_type}',
                        type="number",
                        value=0.0,
                        min=0,
                        max=6.28,
                        step=0.01,
                        className="mb-3"
                    )
                ], width=6)
            ]),
            dbc.Button(
                "Generate Signal",
                id=f'generate-button-{module_type}',
                color=Config.THEME_COLOR,
                className="mt-2"
            )
        ])
    ], className="mb-4")

def create_simulation_controls():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Simulation Controls"),
            dbc.Row([
                dbc.Col([
                    html.Label("Simulation Type"),
                    dcc.Dropdown(
                        id='simulation-type',
                        options=[
                            {'label': 'Signal Simulation', 'value': 'signal'},
                            {'label': 'Circuit Simulation', 'value': 'circuit'},
                            {'label': 'Control System Simulation', 'value': 'control'}
                        ],
                        value='signal',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Analysis Type"),
                    dcc.Dropdown(
                        id='analysis-type',
                        options=[
                            {'label': 'Time Domain', 'value': 'time'},
                            {'label': 'Frequency Domain', 'value': 'frequency'},
                            {'label': 'Modulation', 'value': 'modulation'},
                            {'label': 'Filtering', 'value': 'filtering'}
                        ],
                        value='time',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Signal Type"),
                    dcc.Dropdown(
                        id='signal-type',
                        options=[
                            {'label': 'Sine Wave', 'value': 'sine'},
                            {'label': 'Square Wave', 'value': 'square'},
                            {'label': 'Sawtooth Wave', 'value': 'sawtooth'},
                            {'label': 'Noise', 'value': 'noise'},
                            {'label': 'Composite', 'value': 'composite'}
                        ],
                        value='sine',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Circuit Type"),
                    dcc.Dropdown(
                        id='circuit-type',
                        options=[
                            {'label': 'RC Circuit', 'value': 'rc'},
                            {'label': 'RL Circuit', 'value': 'rl'},
                            {'label': 'RLC Circuit', 'value': 'rlc'},
                            {'label': 'Diode Circuit', 'value': 'diode'},
                            {'label': 'BJT Amplifier', 'value': 'bjt'}
                        ],
                        value='rc',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Control System Type"),
                    dcc.Dropdown(
                        id='control-type',
                        options=[
                            {'label': 'First Order System', 'value': 'first_order'},
                            {'label': 'Second Order System', 'value': 'second_order'},
                            {'label': 'PID Controller', 'value': 'pid'},
                            {'label': 'State Space', 'value': 'state_space'},
                            {'label': 'Lead-Lag Compensator', 'value': 'lead_lag'}
                        ],
                        value='first_order',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Parameters"),
                    dbc.Textarea(
                        id='simulation-params',
                        placeholder="Enter simulation parameters as JSON",
                        style={'height': '100px'},
                        className="mb-3"
                    )
                ], width=12)
            ]),
            dbc.Button(
                "Run Simulation",
                id='run-simulation-button',
                color=Config.THEME_COLOR,
                className="mt-2"
            )
        ])
    ], className="mb-4")

def create_analysis_controls():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Analysis Controls"),
            dbc.Row([
                dbc.Col([
                    html.Label("Analysis Type"),
                    dcc.Dropdown(
                        id='analysis-type-analysis',
                        options=[
                            {'label': 'Statistical Analysis', 'value': 'statistical'},
                            {'label': 'Time Series Analysis', 'value': 'time_series'},
                            {'label': 'Frequency Analysis', 'value': 'frequency'}
                        ],
                        value='statistical',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Method"),
                    dcc.Dropdown(
                        id='analysis-method',
                        options=[
                            {'label': 'Basic Statistics', 'value': 'basic_stats'},
                            {'label': 'Correlation Analysis', 'value': 'correlation'},
                            {'label': 'Hypothesis Testing', 'value': 'hypothesis'},
                            {'label': 'Regression Analysis', 'value': 'regression'},
                            {'label': 'Distribution Analysis', 'value': 'distribution'},
                            {'label': 'Outlier Detection', 'value': 'outlier'}
                        ],
                        value='basic_stats',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Data Column"),
                    dcc.Dropdown(
                        id='data-column',
                        options=[],
                        value=None,
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Secondary Column (if needed)"),
                    dcc.Dropdown(
                        id='secondary-column',
                        options=[],
                        value=None,
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Parameters"),
                    dbc.Textarea(
                        id='analysis-params',
                        placeholder="Enter analysis parameters as JSON",
                        style={'height': '100px'},
                        className="mb-3"
                    )
                ], width=12)
            ]),
            dbc.Button(
                "Run Analysis",
                id='run-analysis-button',
                color=Config.THEME_COLOR,
                className="mt-2"
            )
        ])
    ], className="mb-4")

def create_modeling_controls():
    return dbc.Card([
        dbc.CardHeader([
            html.H4("Modeling Controls"),
            dbc.Row([
                dbc.Col([
                    html.Label("Model Type"),
                    dcc.Dropdown(
                        id='model-type',
                        options=[
                            {'label': 'Signal Modeling', 'value': 'signal'},
                            {'label': 'Circuit Modeling', 'value': 'circuit'},
                            {'label': 'System Modeling', 'value': 'system'}
                        ],
                        value='signal',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Model Method"),
                    dcc.Dropdown(
                        id='model-method',
                        options=[
                            {'label': 'Sinusoidal Model', 'value': 'sinusoidal'},
                            {'label': 'Exponential Model', 'value': 'exponential'},
                            {'label': 'Polynomial Model', 'value': 'polynomial'},
                            {'label': 'Fourier Model', 'value': 'fourier'},
                            {'label': 'Custom Model', 'value': 'custom'}
                        ],
                        value='sinusoidal',
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Input Data Column"),
                    dcc.Dropdown(
                        id='input-column',
                        options=[],
                        value=None,
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6),
                dbc.Col([
                    html.Label("Output Data Column"),
                    dcc.Dropdown(
                        id='output-column',
                        options=[],
                        value=None,
                        className="mb-3",
                        style={'color': 'green'}
                    )
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.Label("Parameters"),
                    dbc.Textarea(
                        id='modeling-params',
                        placeholder="Enter modeling parameters as JSON",
                        style={'height': '100px'},
                        className="mb-3"
                    )
                ], width=12)
            ]),
            dbc.Button(
                "Fit Model",
                id='fit-model-button',
                color=Config.THEME_COLOR,
                className="mt-2"
            )
        ])
    ], className="mb-4")

def create_results_card(title, content_id):
    return dbc.Card([
        dbc.CardHeader(html.H4(title)),
        dbc.CardBody(html.Div(id=content_id))
    ], className="mb-4")

def create_charts_container():
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                create_results_card("Simulation Results", "simulation-results")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                create_results_card("Analysis Results", "analysis-results")
            ], width=12)
        ]),
        dbc.Row([
            dbc.Col([
                create_results_card("Modeling Results", "modeling-results")
            ], width=12)
        ])
    ], fluid=True)

def create_interval_component():
    return dcc.Interval(
        id='interval-component',
        interval=Config.UPDATE_INTERVAL,  # Update interval from config
        n_intervals=0
    )

def create_system_monitoring():
    return dbc.Card([
        dbc.CardHeader("System Monitoring"),
        dbc.CardBody([
            html.H5("System Metrics"),
            html.Div(id='system-metrics'),
            html.H5("Top Processes by CPU Usage"),
            html.Div(id='top-processes')
        ])
    ], className="mb-4")

def create_layout():
    return html.Div([
        create_header(),
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store', storage_type='session'),
        html.Div(id='page-content'),
        create_interval_component()
    ])

def create_simulation_layout():
    return html.Div([
        create_header(),
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store', storage_type='session'),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_upload_card('simulation'),
                    create_manual_input_card('simulation'),
                    create_simulation_controls()
                ], width=4),
                dbc.Col([
                    create_results_card("Simulation Results", "simulation-results"),
                    create_system_monitoring()
                ], width=8)
            ])
        ], fluid=True),
        create_interval_component()
    ])

def create_analysis_layout():
    return html.Div([
        create_header(),
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store', storage_type='session'),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_upload_card('analysis'),
                    create_manual_input_card('analysis'),
                    create_analysis_controls()
                ], width=4),
                dbc.Col([
                    create_results_card("Analysis Results", "analysis-results"),
                    create_system_monitoring()
                ], width=8)
            ])
        ], fluid=True),
        create_interval_component()
    ])

def create_modeling_layout():
    return html.Div([
        create_header(),
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store', storage_type='session'),
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_upload_card('modeling'),
                    create_manual_input_card('modeling'),
                    create_modeling_controls()
                ], width=4),
                dbc.Col([
                    create_results_card("Modeling Results", "modeling-results"),
                    create_system_monitoring()
                ], width=8)
            ])
        ], fluid=True),
        create_interval_component()
    ])