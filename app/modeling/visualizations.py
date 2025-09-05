# app/modeling/visualizations.py
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
from app.config import Config

def create_model_fit_plot(t, y_original, y_fit, title="Model Fit", ylabel="Value", 
                         data_color='#00ccff', fit_color='#ff9900'):
    """Create a model fit plot with dark theme"""
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(
        x=t,
        y=y_original,
        mode='markers',
        name='Original Data',
        marker=dict(color=data_color, size=8)
    ))
    
    # Fitted model
    fig.add_trace(go.Scatter(
        x=t,
        y=y_fit,
        mode='lines',
        name='Fitted Model',
        line=dict(color=fit_color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
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

def create_residual_plot(t, residuals, title="Residuals", ylabel="Residuals", color='#00ccff'):
    """Create a residual plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t,
        y=residuals,
        mode='markers',
        name='Residuals',
        marker=dict(color=color, size=8)
    ))
    
    # Add zero line
    fig.add_trace(go.Scatter(
        x=[t[0], t[-1]],
        y=[0, 0],
        mode='lines',
        name='Zero',
        line=dict(color='white', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
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

def create_prediction_plot(t_original, y_original, t_new, y_pred, title="Prediction", 
                         data_color='#00ccff', pred_color='#ff9900'):
    """Create a prediction plot with dark theme"""
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(
        x=t_original,
        y=y_original,
        mode='markers',
        name='Original Data',
        marker=dict(color=data_color, size=8)
    ))
    
    # Prediction
    fig.add_trace(go.Scatter(
        x=t_new,
        y=y_pred,
        mode='lines',
        name='Prediction',
        line=dict(color=pred_color, width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
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

def create_pole_zero_plot(poles, zeros, title="Pole-Zero Plot"):
    """Create a pole-zero plot with dark theme"""
    fig = go.Figure()
    
    # Plot poles
    fig.add_trace(go.Scatter(
        x=np.real(poles),
        y=np.imag(poles),
        mode='markers',
        name='Poles',
        marker=dict(color='red', size=12, symbol='x')
    ))
    
    # Plot zeros
    fig.add_trace(go.Scatter(
        x=np.real(zeros),
        y=np.imag(zeros),
        mode='markers',
        name='Zeros',
        marker=dict(color='blue', size=12, symbol='circle')
    ))
    
    # Add axes
    fig.add_trace(go.Scatter(
        x=[-10, 10],
        y=[0, 0],
        mode='lines',
        name='Real Axis',
        line=dict(color='white', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[-10, 10],
        mode='lines',
        name='Imaginary Axis',
        line=dict(color='white', width=1, dash='dash'),
        showlegend=False
    ))
    
    # Add unit circle for discrete systems
    theta = np.linspace(0, 2*np.pi, 100)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)
    fig.add_trace(go.Scatter(
        x=x_circle,
        y=y_circle,
        mode='lines',
        name='Unit Circle',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dot'),
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
            scaleanchor="y",
            scaleratio=1,
            range=[-2, 2]
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)',
            range=[-2, 2]
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
    
    # Add negative real axis
    fig.add_trace(go.Scatter(
        x=[-2, 0],
        y=[0, 0],
        mode='lines',
        name='Negative Real Axis',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
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
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    
    return fig

def create_step_response_plot(t, y, title="Step Response", color='#00ccff'):
    """Create a step response plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, 
        y=y, 
        mode='lines',
        name='Step Response',
        line=dict(color=color, width=2)
    ))
    
    # Add reference line at final value
    fig.add_trace(go.Scatter(
        x=[t[0], t[-1]],
        y=[y[-1], y[-1]],
        mode='lines',
        name='Final Value',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        showlegend=False
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

def create_modeling_plots(modeling_data, plot_type='model_fit'):
    """Create appropriate plots based on modeling type"""
    if plot_type == 'model_fit':
        return create_model_fit_plot(
            modeling_data['t'],
            modeling_data['y_original'],
            modeling_data['y_fit'],
            title="Model Fit Analysis"
        )
    elif plot_type == 'residuals':
        return create_residual_plot(
            modeling_data['t'],
            modeling_data['residuals'],
            title="Residual Analysis"
        )
    elif plot_type == 'prediction':
        return create_prediction_plot(
            modeling_data['t_original'],
            modeling_data['y_original'],
            modeling_data['t_new'],
            modeling_data['y_pred'],
            title="Prediction Analysis"
        )
    elif plot_type == 'pole_zero':
        return create_pole_zero_plot(
            modeling_data['poles'],
            modeling_data['zeros'],
            title="Pole-Zero Analysis"
        )
    elif plot_type == 'bode':
        return create_bode_plot(
            modeling_data['frequencies'],
            modeling_data['magnitude'],
            modeling_data['phase'],
            title="Bode Plot Analysis"
        )
    elif plot_type == 'nyquist':
        return create_nyquist_plot(
            modeling_data['real'],
            modeling_data['imag'],
            title="Nyquist Plot Analysis"
        )
    elif plot_type == 'step_response':
        return create_step_response_plot(
            modeling_data['t'],
            modeling_data['y'],
            title="Step Response Analysis"
        )
    elif plot_type == 'impulse_response':
        return create_impulse_response_plot(
            modeling_data['t'],
            modeling_data['y'],
            title="Impulse Response Analysis"
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")