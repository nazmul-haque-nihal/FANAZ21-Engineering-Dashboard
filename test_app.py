# test_app.py
# Add these imports at the top
import base64
import io
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from datetime import datetime

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
    
    dbc.Row([
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
                    dbc.Button("Generate Signal", id="generate-button", color="primary", className="mb-3")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Signal Plot"),
                dbc.CardBody([
                    dcc.Graph(id='signal-plot')
                ])
            ])
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Circuit Simulation"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Circuit Type"),
                            dcc.Dropdown(
                                id='circuit-type',
                                options=[
                                    {'label': 'RC Circuit', 'value': 'rc'},
                                    {'label': 'RL Circuit', 'value': 'rl'},
                                    {'label': 'RLC Circuit', 'value': 'rlc'}
                                ],
                                value='rc',
                                className="mb-3",
                                style={'color': 'green'}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Component Value"),
                            dcc.Input(
                                id='component-value',
                                type="number",
                                value=1000,
                                min=1,
                                max=10000,
                                step=1,
                                className="mb-3"
                            )
                        ], width=6)
                    ]),
                    dbc.Button("Simulate Circuit", id="simulate-button", color="success", className="mb-3")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Circuit Response"),
                dbc.CardBody([
                    dcc.Graph(id='circuit-plot')
                ])
            ])
        ], width=8)
    ])
], fluid=True)

# Define callbacks
@app.callback(
    dash.dependencies.Output('signal-plot', 'figure'),
    [dash.dependencies.Input('generate-button', 'n_clicks')],
    [dash.dependencies.State('signal-type', 'value'),
     dash.dependencies.State('frequency', 'value')]
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

@app.callback(
    dash.dependencies.Output('circuit-plot', 'figure'),
    [dash.dependencies.Input('simulate-button', 'n_clicks')],
    [dash.dependencies.State('circuit-type', 'value'),
     dash.dependencies.State('component-value', 'value')]
)
def update_circuit_plot(n_clicks, circuit_type, component_value):
    if n_clicks:
        # Generate time vector
        t = np.linspace(0, 0.01, 1000)
        
        # Generate circuit response based on type
        if circuit_type == 'rc':
            # RC circuit step response
            R = component_value  # Resistance in ohms
            C = 1e-6  # Capacitance in farads
            tau = R * C  # Time constant
            
            v_in = np.ones_like(t)  # Step input
            v_out = 1.0 * (1 - np.exp(-t / tau))  # Step response
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t,
                y=v_in,
                mode='lines',
                name='Input',
                line=dict(color='#00ccff', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=t,
                y=v_out,
                mode='lines',
                name='Output',
                line=dict(color='#ff9900', width=2)
            ))
            fig.update_layout(
                title=f"RC Circuit Response (R = {R}Ω, C = {C*1e6}μF)",
                xaxis_title="Time (s)",
                yaxis_title="Voltage (V)",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        elif circuit_type == 'rl':
            # RL circuit step response
            R = component_value  # Resistance in ohms
            L = 0.1  # Inductance in henrys
            tau = L / R  # Time constant
            
            i_in = np.ones_like(t)  # Step input
            i_out = 0.01 * (1 - np.exp(-t / tau))  # Step response
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t,
                y=i_in,
                mode='lines',
                name='Input',
                line=dict(color='#00ccff', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=t,
                y=i_out,
                mode='lines',
                name='Output',
                line=dict(color='#ff9900', width=2)
            ))
            fig.update_layout(
                title=f"RL Circuit Response (R = {R}Ω, L = {L}H)",
                xaxis_title="Time (s)",
                yaxis_title="Current (A)",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
        
        elif circuit_type == 'rlc':
            # RLC circuit step response
            R = component_value  # Resistance in ohms
            L = 0.1  # Inductance in henrys
            C = 1e-6  # Capacitance in farads
            
            omega_0 = 1 / np.sqrt(L * C)  # Natural frequency
            zeta = R / (2 * np.sqrt(L / C))  # Damping ratio
            
            v_in = np.ones_like(t)  # Step input
            
            if zeta < 1:  # Underdamped
                omega_d = omega_0 * np.sqrt(1 - zeta**2)
                v_out = 1.0 * (1 - np.exp(-zeta * omega_0 * t) * (
                    np.cos(omega_d * t) + (zeta * omega_0 / omega_d) * np.sin(omega_d * t)
                ))
            elif zeta == 1:  # Critically damped
                v_out = 1.0 * (1 - np.exp(-omega_0 * t) * (1 + omega_0 * t))
            else:  # Overdamped
                s1 = -omega_0 * (zeta - np.sqrt(zeta**2 - 1))
                s2 = -omega_0 * (zeta + np.sqrt(zeta**2 - 1))
                v_out = 1.0 * (1 - (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s2 - s1))
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=t,
                y=v_in,
                mode='lines',
                name='Input',
                line=dict(color='#00ccff', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=t,
                y=v_out,
                mode='lines',
                name='Output',
                line=dict(color='#ff9900', width=2)
            ))
            fig.update_layout(
                title=f"RLC Circuit Response (R = {R}Ω, L = {L}H, C = {C*1e6}μF)",
                xaxis_title="Time (s)",
                yaxis_title="Voltage (V)",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )
            
            return fig
    
    return go.Figure()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)