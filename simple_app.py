# simple_app.py
# Add these imports at the top
import base64
import io
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# Create the app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

# Define the layout
app.layout = dbc.Container([
    dbc.NavbarSimple(
        brand="FANAZ21 Engineering Dashboard",
        brand_href="#",
        dark=True,
        className="mb-4"
    ),
    
    dbc.Tabs([
        dbc.Tab(label="Simulation", tab_id="simulation-tab"),
        dbc.Tab(label="Analysis", tab_id="analysis-tab"),
        dbc.Tab(label="Modeling", tab_id="modeling-tab"),
    ], id="tabs", active_tab="simulation-tab"),
    
    html.Div(id="tab-content", className="p-4")
], fluid=True)

# Define callbacks for tab switching
@app.callback(
    Output("tab-content", "children"),
    [Input("tabs", "active_tab")]
)
def render_tab_content(active_tab):
    if active_tab == "simulation-tab":
        return simulation_tab()
    elif active_tab == "analysis-tab":
        return analysis_tab()
    elif active_tab == "modeling-tab":
        return modeling_tab()

def simulation_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Signal Generation"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Signal Type"),
                            dcc.Dropdown(
                                id='signal-type',
                                options=[
                                    {'label': 'Sine Wave', 'value': 'sine'},
                                    {'label': 'Square Wave', 'value': 'square'},
                                    {'label': 'Sawtooth Wave', 'value': 'sawtooth'},
                                    {'label': 'Noise', 'value': 'noise'}
                                ],
                                value='sine',
                                className="mb-3",
                                style={'color': 'green'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Frequency (Hz)"),
                            dcc.Input(
                                id='frequency',
                                type="number",
                                value=5,
                                min=0.1,
                                max=100,
                                step=0.1,
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    dbc.Button("Generate Signal", id="generate-button", color="primary", className="mb-3"),
                    dcc.Graph(id='signal-plot')
                ])
            ])
        ], width=12)
    ])

def analysis_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Statistical Analysis"),
                dbc.CardBody([
                    html.P("Upload a CSV file for analysis:"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
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
                    html.Div(id='output-data-upload'),
                    dcc.Graph(id='histogram-plot')
                ])
            ])
        ], width=12)
    ])

def modeling_tab():
    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Fitting"),
                dbc.CardBody([
                    html.P("Upload a CSV file for modeling:"),
                    dcc.Upload(
                        id='upload-model-data',
                        children=html.Div([
                            'Drag and Drop or ',
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
                    html.Div(id='output-model-upload'),
                    dcc.Graph(id='model-fit-plot')
                ])
            ])
        ], width=12)
    ])

# Define callback for signal generation
@app.callback(
    Output('signal-plot', 'figure'),
    [Input('generate-button', 'n_clicks')],
    [State('signal-type', 'value'),
     State('frequency', 'value')]
)
def update_signal_plot(n_clicks, signal_type, frequency):
    if n_clicks:
        # Generate time vector
        t = np.linspace(0, 1, 1000)
        
        # Generate signal based on type
        if signal_type == 'sine':
            y = np.sin(2 * np.pi * frequency * t)
        elif signal_type == 'square':
            from scipy import signal
            y = signal.square(2 * np.pi * frequency * t)
        elif signal_type == 'sawtooth':
            from scipy import signal
            y = signal.sawtooth(2 * np.pi * frequency * t)
        elif signal_type == 'noise':
            y = np.random.normal(0, 1, len(t))
        else:
            y = np.zeros_like(t)
        
        # Create plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=t,
            y=y,
            mode='lines',
            name=signal_type.capitalize(),
            line=dict(color='#00ccff', width=2)
        ))
        fig.update_layout(
            title=f"{signal_type.capitalize()} Wave (f = {frequency} Hz)",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    return go.Figure()

# Define callback for data upload (analysis)
@app.callback(
    Output('output-data-upload', 'children'),
    Output('histogram-plot', 'figure'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        try:
            # Decode the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Process the file
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                
                # Create histogram of the first numeric column
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(
                        x=df[numeric_columns[0]],
                        name='Histogram',
                        marker=dict(color='#00ccff')
                    ))
                    fig.update_layout(
                        title=f"Histogram of {numeric_columns[0]}",
                        xaxis_title=numeric_columns[0],
                        yaxis_title="Count",
                        template="plotly_dark",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    return html.Div([
                        html.P(f"Successfully processed {filename}"),
                        html.P(f"Shape: {df.shape}")
                    ]), fig
                else:
                    return html.Div([
                        html.P(f"No numeric columns found in {filename}")
                    ]), go.Figure()
            else:
                return html.Div([
                    html.P("Only CSV files are supported for analysis")
                ]), go.Figure()
        
        except Exception as e:
            return html.Div([
                html.P(f"There was an error processing this file: {str(e)}")
            ]), go.Figure()
    
    return html.Div(), go.Figure()

# Define callback for data upload (modeling)
@app.callback(
    Output('output-model-upload', 'children'),
    Output('model-fit-plot', 'figure'),
    [Input('upload-model-data', 'contents')],
    [State('upload-model-data', 'filename')]
)
def update_model_output(contents, filename):
    if contents is not None:
        try:
            # Decode the file
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            
            # Process the file
            if 'csv' in filename:
                # Assume that the user uploaded a CSV file
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
                
                # Get the first two numeric columns
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) >= 2:
                    x = df[numeric_columns[0]].values
                    y = df[numeric_columns[1]].values
                    
                    # Fit a linear model
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                    
                    # Calculate fitted values
                    y_fit = slope * x + intercept
                    
                    # Create plot
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y,
                        mode='markers',
                        name='Data',
                        marker=dict(color='#00ccff', size=8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y_fit,
                        mode='lines',
                        name='Fit',
                        line=dict(color='#ff9900', width=2)
                    ))
                    fig.update_layout(
                        title=f"Linear Fit: y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})",
                        xaxis_title=numeric_columns[0],
                        yaxis_title=numeric_columns[1],
                        template="plotly_dark",
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )
                    
                    return html.Div([
                        html.P(f"Successfully processed {filename}"),
                        html.P(f"Shape: {df.shape}")
                    ]), fig
                else:
                    return html.Div([
                        html.P(f"At least two numeric columns are required for modeling")
                    ]), go.Figure()
            else:
                return html.Div([
                    html.P("Only CSV files are supported for modeling")
                ]), go.Figure()
        
        except Exception as e:
            return html.Div([
                html.P(f"There was an error processing this file: {str(e)}")
            ]), go.Figure()
    
    return html.Div(), go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)