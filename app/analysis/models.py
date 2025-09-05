# app/analysis/models.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64

def generate_analysis_data(analysis_type='default'):
    """Generate analysis data based on the analysis type"""
    np.random.seed(42)
    
    if analysis_type == 'statistical':
        return generate_statistical_analysis_data()
    elif analysis_type == 'signal_processing':
        return generate_signal_processing_data()
    elif analysis_type == 'clustering':
        return generate_clustering_data()
    elif analysis_type == 'dimensionality_reduction':
        return generate_dimensionality_reduction_data()
    else:
        return generate_default_analysis_data()

def generate_statistical_analysis_data():
    """Generate statistical analysis data"""
    # Generate data from different distributions
    normal_data = np.random.normal(0, 1, 1000)
    uniform_data = np.random.uniform(-3, 3, 1000)
    exponential_data = np.random.exponential(1, 1000)
    
    df = pd.DataFrame({
        'Normal': normal_data,
        'Uniform': uniform_data,
        'Exponential': exponential_data
    })
    
    return df

def generate_signal_processing_data():
    """Generate signal processing data"""
    t = np.linspace(0, 10, 1000)
    
    # Create a signal with multiple frequencies
    signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t) + 0.2 * np.sin(2 * np.pi * 30 * t)
    
    # Add some noise
    noise = 0.1 * np.random.normal(size=len(t))
    signal_with_noise = signal + noise
    
    df = pd.DataFrame({
        'Time': t,
        'Signal': signal,
        'Signal with Noise': signal_with_noise
    })
    
    return df

def generate_clustering_data():
    """Generate clustering data"""
    # Generate three clusters
    cluster1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 100)
    cluster2 = np.random.multivariate_normal([5, 5], [[1, -0.5], [-0.5, 1]], 100)
    cluster3 = np.random.multivariate_normal([-5, 5], [[1, 0], [0, 2]], 100)
    
    data = np.vstack([cluster1, cluster2, cluster3])
    
    df = pd.DataFrame(data, columns=['X', 'Y'])
    
    return df

def generate_dimensionality_reduction_data():
    """Generate dimensionality reduction data"""
    # Generate data with 10 features
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Create correlated features
    base = np.random.normal(0, 1, n_samples)
    features = []
    
    for i in range(n_features):
        # Each feature is a combination of the base and some noise
        feature = base + np.random.normal(0, 0.5, n_samples)
        features.append(feature)
    
    data = np.column_stack(features)
    
    df = pd.DataFrame(data, columns=[f'Feature {i+1}' for i in range(n_features)])
    
    return df

def generate_default_analysis_data():
    """Generate default analysis data"""
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    
    df = pd.DataFrame({
        'X': x,
        'Y': y
    })
    
    return df

def perform_statistical_analysis(df):
    """Perform statistical analysis on the data"""
    results = {}
    
    for column in df.columns:
        # Basic statistics
        results[column] = {
            'mean': df[column].mean(),
            'median': df[column].median(),
            'std': df[column].std(),
            'min': df[column].min(),
            'max': df[column].max(),
            'skewness': stats.skew(df[column]),
            'kurtosis': stats.kurtosis(df[column])
        }
        
        # Normality test
        _, p_value = stats.normaltest(df[column])
        results[column]['normality_p_value'] = p_value
        results[column]['is_normal'] = p_value > 0.05
    
    return results

def perform_signal_processing_analysis(df):
    """Perform signal processing analysis on the data"""
    results = {}
    
    # Get the signal
    signal = df['Signal with Noise'].values
    
    # Find peaks
    peaks, _ = find_peaks(signal, height=0)
    
    # FFT
    fft_vals = np.abs(np.fft.rfft(signal))
    fft_freq = np.fft.rfftfreq(len(signal), d=df['Time'][1]-df['Time'][0])
    
    # Find dominant frequencies
    dominant_freq_idx = np.argsort(fft_vals)[-3:][::-1]
    dominant_freqs = fft_freq[dominant_freq_idx]
    dominant_amps = fft_vals[dominant_freq_idx]
    
    results = {
        'peaks': peaks,
        'fft': fft_vals,
        'fft_freq': fft_freq,
        'dominant_frequencies': dominant_freqs,
        'dominant_amplitudes': dominant_amps
    }
    
    return results

def perform_clustering_analysis(df, n_clusters=3):
    """Perform clustering analysis on the data"""
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Get cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    silhouette = silhouette_score(scaled_data, clusters)
    
    results = {
        'clusters': clusters,
        'centers': centers,
        'silhouette_score': silhouette,
        'n_clusters': n_clusters
    }
    
    return results

def perform_dimensionality_reduction_analysis(df, n_components=2):
    """Perform dimensionality reduction analysis on the data"""
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Get explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    results = {
        'principal_components': principal_components,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'n_components': n_components
    }
    
    return results

def perform_analysis(df, analysis_type='statistical', **kwargs):
    """Perform analysis on the data based on the analysis type"""
    if analysis_type == 'statistical':
        return perform_statistical_analysis(df)
    elif analysis_type == 'signal_processing':
        return perform_signal_processing_analysis(df)
    elif analysis_type == 'clustering':
        n_clusters = kwargs.get('n_clusters', 3)
        return perform_clustering_analysis(df, n_clusters)
    elif analysis_type == 'dimensionality_reduction':
        n_components = kwargs.get('n_components', 2)
        return perform_dimensionality_reduction_analysis(df, n_components)
    else:
        return perform_statistical_analysis(df)