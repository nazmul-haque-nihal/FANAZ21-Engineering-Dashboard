# app/analysis/visualizations.py
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import pandas as pd
from app.config import Config

def create_histogram_plot(data, title="Histogram", xlabel="Value", ylabel="Frequency", color='#00ccff'):
    """Create a histogram plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=data,
        name='Histogram',
        marker=dict(color=color, line=dict(color='white', width=1))
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

def create_box_plot(data, title="Box Plot", ylabel="Value", color='#00ccff'):
    """Create a box plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=data,
        name='Box Plot',
        marker=dict(color=color)
    ))
    fig.update_layout(
        title=title,
        yaxis_title=ylabel,
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_scatter_plot(x, y, title="Scatter Plot", xlabel="X", ylabel="Y", color='#00ccff'):
    """Create a scatter plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Scatter',
        marker=dict(color=color, size=8)
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

def create_line_plot(x, y, title="Line Plot", xlabel="X", ylabel="Y", color='#00ccff'):
    """Create a line plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Line',
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

def create_correlation_heatmap(corr_matrix, title="Correlation Heatmap"):
    """Create a correlation heatmap with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='Viridis',
        zmid=0,
        text=corr_matrix.values,
        texttemplate="%{text:.2f}",
        textfont={"size": 10},
        hoverinfo='text'
    ))
    fig.update_layout(
        title=title,
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_regression_plot(x, y, y_pred, title="Regression Plot", xlabel="X", ylabel="Y", 
                          data_color='#00ccff', regression_color='#ff9900'):
    """Create a regression plot with dark theme"""
    fig = go.Figure()
    
    # Original data
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Data',
        marker=dict(color=data_color, size=8)
    ))
    
    # Regression line
    fig.add_trace(go.Scatter(
        x=x,
        y=y_pred,
        mode='lines',
        name='Regression',
        line=dict(color=regression_color, width=2)
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

def create_distribution_plot(data, distributions, title="Distribution Analysis"):
    """Create a distribution plot with dark theme"""
    fig = go.Figure()
    
    # Histogram of original data
    fig.add_trace(go.Histogram(
        x=data,
        name='Data',
        histnorm='probability density',
        marker=dict(color='#00ccff', line=dict(color='white', width=1)),
        opacity=0.7
    ))
    
    # Fitted distributions
    colors = ['#ff9900', '#ff3399', '#33ff99', '#9933ff', '#ffff33']
    for i, (dist_name, dist_result) in enumerate(distributions.items()):
        if 'error' not in dist_result:
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(
                x=dist_result['x'],
                y=dist_result['pdf'],
                mode='lines',
                name=dist_name.capitalize(),
                line=dict(color=color, width=2)
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Value',
        yaxis_title='Probability Density',
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

def create_qq_plot(data, dist='norm', title="Q-Q Plot"):
    """Create a Q-Q plot with dark theme"""
    from scipy import stats
    
    # Calculate theoretical quantiles
    theoretical_quantiles = stats.probplot(data, dist=dist, fit=False)[0]
    
    # Calculate sample quantiles
    sample_quantiles = np.sort(data)
    
    # Fit a line
    slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, sample_quantiles)
    line_y = slope * theoretical_quantiles + intercept
    
    fig = go.Figure()
    
    # Q-Q points
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Data',
        marker=dict(color='#00ccff', size=8)
    ))
    
    # Reference line
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=line_y,
        mode='lines',
        name='Reference Line',
        line=dict(color='#ff9900', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Theoretical Quantiles',
        yaxis_title='Sample Quantiles',
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

def create_time_series_plot(series, title="Time Series", ylabel="Value", color='#00ccff'):
    """Create a time series plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=series.index,
        y=series.values,
        mode='lines',
        name='Time Series',
        line=dict(color=color, width=2)
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

def create_decomposition_plot(original, trend, seasonal, residual, title="Time Series Decomposition"):
    """Create a time series decomposition plot with dark theme"""
    fig = go.Figure()
    
    # Original
    fig.add_trace(go.Scatter(
        x=original.index,
        y=original.values,
        mode='lines',
        name='Original',
        line=dict(color='#00ccff', width=1)
    ))
    
    # Trend
    fig.add_trace(go.Scatter(
        x=trend.index,
        y=trend.values,
        mode='lines',
        name='Trend',
        line=dict(color='#ff9900', width=1)
    ))
    
    # Seasonal
    fig.add_trace(go.Scatter(
        x=seasonal.index,
        y=seasonal.values,
        mode='lines',
        name='Seasonal',
        line=dict(color='#ff3399', width=1)
    ))
    
    # Residual
    fig.add_trace(go.Scatter(
        x=residual.index,
        y=residual.values,
        mode='lines',
        name='Residual',
        line=dict(color='#33ff99', width=1)
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

def create_autocorrelation_plot(acf_values, confint, title="Autocorrelation Function"):
    """Create an autocorrelation plot with dark theme"""
    lags = np.arange(len(acf_values))
    
    fig = go.Figure()
    
    # ACF values
    fig.add_trace(go.Scatter(
        x=lags,
        y=acf_values,
        mode='markers',
        name='ACF',
        marker=dict(color='#00ccff', size=8)
    ))
    
    # Confidence intervals
    upper_confint = confint[:, 1]
    lower_confint = confint[:, 0]
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=upper_confint,
        mode='lines',
        name='Upper CI',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=lower_confint,
        mode='lines',
        name='Lower CI',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.1)',
        showlegend=False
    ))
    
    # Zero line
    fig.add_trace(go.Scatter(
        x=[lags[0], lags[-1]],
        y=[0, 0],
        mode='lines',
        name='Zero',
        line=dict(color='white', width=1),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Lag',
        yaxis_title='Correlation',
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

def create_pacf_plot(pacf_values, confint, title="Partial Autocorrelation Function"):
    """Create a partial autocorrelation plot with dark theme"""
    lags = np.arange(len(pacf_values))
    
    fig = go.Figure()
    
    # PACF values
    fig.add_trace(go.Scatter(
        x=lags,
        y=pacf_values,
        mode='markers',
        name='PACF',
        marker=dict(color='#00ccff', size=8)
    ))
    
    # Confidence intervals
    upper_confint = confint[:, 1]
    lower_confint = confint[:, 0]
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=upper_confint,
        mode='lines',
        name='Upper CI',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=lags,
        y=lower_confint,
        mode='lines',
        name='Lower CI',
        line=dict(color='rgba(255,255,255,0.3)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(255,255,255,0.1)',
        showlegend=False
    ))
    
    # Zero line
    fig.add_trace(go.Scatter(
        x=[lags[0], lags[-1]],
        y=[0, 0],
        mode='lines',
        name='Zero',
        line=dict(color='white', width=1),
        showlegend=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Lag',
        yaxis_title='Partial Correlation',
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

def create_spectrogram_plot(f, t, Sxx_db, title="Spectrogram"):
    """Create a spectrogram plot with dark theme"""
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        x=t,
        y=f,
        z=Sxx_db,
        colorscale='Viridis',
        colorbar=dict(title="Power (dB)")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (s)',
        yaxis_title='Frequency (Hz)',
        template=Config.CHART_TEMPLATE,
        height=Config.CHART_HEIGHT,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def create_coherence_plot(f, Cxy, title="Coherence"):
    """Create a coherence plot with dark theme"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=f,
        y=Cxy,
        mode='lines',
        name='Coherence',
        line=dict(color='#00ccff', width=2)
    ))
    
    # Add a line at 0.8 (common threshold for significant coherence)
    fig.add_trace(go.Scatter(
        x=[f[0], f[-1]],
        y=[0.8, 0.8],
        mode='lines',
        name='Threshold',
        line=dict(color='#ff9900', width=1, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Frequency (Hz)',
        yaxis_title='Coherence',
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
            range=[0, 1.05],
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.2)'
        )
    )
    return fig

def create_analysis_plots(analysis_data, plot_type='histogram'):
    """Create appropriate plots based on analysis type"""
    if plot_type == 'histogram':
        return create_histogram_plot(
            analysis_data['data'], 
            title="Histogram Analysis"
        )
    elif plot_type == 'box':
        return create_box_plot(
            analysis_data['data'], 
            title="Box Plot Analysis"
        )
    elif plot_type == 'scatter':
        return create_scatter_plot(
            analysis_data['x'], 
            analysis_data['y'], 
            title="Scatter Plot Analysis"
        )
    elif plot_type == 'line':
        return create_line_plot(
            analysis_data['x'], 
            analysis_data['y'], 
            title="Line Plot Analysis"
        )
    elif plot_type == 'correlation':
        return create_correlation_heatmap(
            analysis_data['corr_matrix'], 
            title="Correlation Analysis"
        )
    elif plot_type == 'regression':
        return create_regression_plot(
            analysis_data['x'], 
            analysis_data['y'],
            analysis_data['y_pred'],
            title="Regression Analysis"
        )
    elif plot_type == 'distribution':
        return create_distribution_plot(
            analysis_data['sample'],
            analysis_data['distributions'],
            title="Distribution Analysis"
        )
    elif plot_type == 'qq':
        return create_qq_plot(
            analysis_data['data'],
            analysis_data.get('dist', 'norm'),
            title="Q-Q Plot Analysis"
        )
    elif plot_type == 'time_series':
        return create_time_series_plot(
            analysis_data['series'],
            title="Time Series Analysis"
        )
    elif plot_type == 'decomposition':
        return create_decomposition_plot(
            analysis_data['original'],
            analysis_data['trend'],
            analysis_data['seasonal'],
            analysis_data['residual'],
            title="Time Series Decomposition"
        )
    elif plot_type == 'autocorrelation':
        return create_autocorrelation_plot(
            analysis_data['acf_values'],
            analysis_data['confint'],
            title="Autocorrelation Analysis"
        )
    elif plot_type == 'pacf':
        return create_pacf_plot(
            analysis_data['pacf_values'],
            analysis_data['confint'],
            title="Partial Autocorrelation Analysis"
        )
    elif plot_type == 'spectrogram':
        return create_spectrogram_plot(
            analysis_data['f'],
            analysis_data['t'],
            analysis_data['spectrogram_db'],
            title="Spectrogram Analysis"
        )
    elif plot_type == 'coherence':
        return create_coherence_plot(
            analysis_data['frequencies'],
            analysis_data['coherence'],
            title="Coherence Analysis"
        )
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")