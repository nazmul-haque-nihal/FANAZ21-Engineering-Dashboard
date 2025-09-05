# app/simulation/visualizations.py
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from app.config import Config

def create_signal_plot(t, y, title="Signal Plot", xlabel="Time", ylabel="Amplitude", color='#00ccff'):
    """Create a signal plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, 
        y=y, 
        mode='lines',
        name='Signal',
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_spectrum_plot(frequencies, magnitude, title="Frequency Spectrum", color='#00ccff'):
    """Create a frequency spectrum plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequencies, 
        y=magnitude, 
        mode='lines',
        name='Spectrum',
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            type='log',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            type='log',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_psd_plot(frequencies, psd, title="Power Spectral Density", color='#00ccff'):
    """Create a PSD plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frequencies, 
        y=psd, 
        mode='lines',
        name='PSD',
        line=dict(color=color, width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Power/Frequency (dB/Hz)',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            type='log',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            type='log',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_phase_plot(y1, y2, title="Phase Plot", color1='#00ccff', color2='#ff9900'):
    """Create a phase plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y1, 
        y=y2, 
        mode='lines',
        name='Phase',
        line=dict(color=color1, width=2)
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Signal 1',
        yaxis_title='Signal 2',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_bode_plot(frequencies, magnitude, phase, title="Bode Plot"):
    """Create a Bode plot with dark theme"""
    fig = go.Figure()
    
    # Magnitude plot
    fig.add_trace(go.Scatter(
        x=frequencies, 
        y=20 * np.log10(magnitude),
        mode='lines',
        name='Magnitude',
        line=dict(color='#00ccff', width=2),
        yaxis='y'
    ))
    
    # Phase plot
    fig.add_trace(go.Scatter(
        x=frequencies, 
        y=phase,
        mode='lines',
        name='Phase',
        line=dict(color='#ff9900', width=2),
        yaxis='y2'
    ))
    
    # Update layout with two y-axes
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        xaxis=dict(
            type='log',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            title='Magnitude (dB)',
            side='left',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis2=dict(
            title='Phase (degrees)',
            overlaying='y',
            side='right',
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

def create_nyquist_plot(real, imag, title="Nyquist Plot", color='#00ccff'):
    """Create a Nyquist plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=real, 
        y=imag, 
        mode='lines',
        name='Nyquist',
        line=dict(color=color, width=2)
    ))
    
    # Add a marker at the origin
    fig.add_trace(go.Scatter(
        x=[0], 
        y=[0], 
        mode='markers',
        name='Origin',
        marker=dict(color='red', size=8)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Real Part',
        yaxis_title='Imaginary Part',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_root_locus_plot(real_parts, imag_parts, K_values, title="Root Locus"):
    """Create a root locus plot with dark theme"""
    fig = go.Figure()
    
    # Plot root locus
    for i in range(len(K_values)):
        fig.add_trace(go.Scatter(
            x=real_parts[i], 
            y=imag_parts[i], 
            mode='markers',
            marker=dict(
                size=4,
                color=K_values[i],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Gain K")
            ),
            name=f'K={K_values[i]:.2f}',
            showlegend=False
        ))
    
    # Add x-axis
    fig.add_trace(go.Scatter(
        x=[-10, 10], 
        y=[0, 0], 
        mode='lines',
        line=dict(color='white', width=1, dash='dash'),
        name='Real Axis',
        showlegend=False
    ))
    
    # Add y-axis
    fig.add_trace(go.Scatter(
        x=[0, 0], 
        y=[-10, 10], 
        mode='lines',
        line=dict(color='white', width=1, dash='dash'),
        name='Imaginary Axis',
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Real Part',
        yaxis_title='Imaginary Part',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[-10, 10]
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[-10, 10],
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    return fig

def create_polar_plot(theta, r, title="Polar Plot", color='#00ccff'):
    """Create a polar plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=r,
        theta=theta,
        mode='lines',
        name='Polar',
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                linecolor='rgba(255,255,255,0.2)'
            )
        ),
        font=dict(color='white')
    )
    
    return fig

def create_step_response_plot(t, y, u, title="Step Response", color1='#00ccff', color2='#ff9900'):
    """Create a step response plot with dark theme"""
    fig = go.Figure()
    
    # Input
    fig.add_trace(go.Scatter(
        x=t, 
        y=u, 
        mode='lines',
        name='Input',
        line=dict(color=color2, width=2, dash='dash')
    ))
    
    # Output
    fig.add_trace(go.Scatter(
        x=t, 
        y=y, 
        mode='lines',
        name='Output',
        line=dict(color=color1, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_impulse_response_plot(t, y, title="Impulse Response", color='#00ccff'):
    """Create an impulse response plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, 
        y=y, 
        mode='lines',
        name='Impulse Response',
        line=dict(color=color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Amplitude',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_simulation_plots(simulation_data, plot_type='signal'):
    """Create appropriate plots based on simulation type"""
    if plot_type == 'signal':
        return create_signal_plot(
            simulation_data['t'], 
            simulation_data['y'], 
            title="Signal Simulation"
        )
    elif plot_type == 'spectrum':
        return create_spectrum_plot(
            simulation_data['frequencies'], 
            simulation_data['magnitude'], 
            title="Frequency Spectrum"
        )
    elif plot_type == 'psd':
        return create_psd_plot(
            simulation_data['frequencies'], 
            simulation_data['psd'], 
            title="Power Spectral Density"
        )
    elif plot_type == 'phase':
        return create_phase_plot(
            simulation_data['y1'], 
            simulation_data['y2'], 
            title="Phase Plot"
        )
    elif plot_type == 'bode':
        return create_bode_plot(
            simulation_data['frequencies'], 
            simulation_data['magnitude'],
            simulation_data['phase'],
            title="Bode Plot"
        )
    elif plot_type == 'nyquist':
        return create_nyquist_plot(
            simulation_data['real'], 
            simulation_data['imag'], 
            title="Nyquist Plot"
        )
    elif plot_type == 'root_locus':
        return create_root_locus_plot(
            simulation_data['real_parts'], 
            simulation_data['imag_parts'],
            simulation_data['K_values'],
            title="Root Locus"
        )
    elif plot_type == 'polar':
        return create_polar_plot(
            simulation_data['theta'], 
            simulation_data['r'], 
            title="Polar Plot"
        )
    elif plot_type == 'step_response':
        return create_step_response_plot(
            simulation_data['t'], 
            simulation_data['y'],
            simulation_data['u'],
            title="Step Response"
        )
    elif plot_type == 'impulse_response':
        return create_impulse_response_plot(
            simulation_data['t'], 
            simulation_data['y'], 
            title="Impulse Response"
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")