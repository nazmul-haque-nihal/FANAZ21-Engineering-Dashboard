# app/analysis/statistical_analysis.py
import numpy as np
import pandas as pd
from scipy import stats
import plotly.graph_objs as go
from app.config import Config

class StatisticalAnalyzer:
    def __init__(self, config):
        self.config = config
        self.datasets = {}  # Dictionary to store datasets
    
    def load_data(self, data, name='dataset'):
        """Load data for analysis"""
        if isinstance(data, np.ndarray):
            self.datasets[name] = {'data': data, 'type': 'array'}
        elif isinstance(data, pd.DataFrame):
            self.datasets[name] = {'data': data, 'type': 'dataframe'}
        elif isinstance(data, dict):
            self.datasets[name] = {'data': data, 'type': 'dict'}
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        return name
    
    def basic_statistics(self, dataset_name):
        """Calculate basic statistics for a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        if data_type == 'array':
            stats_dict = {
                'mean': np.mean(data),
                'median': np.median(data),
                'mode': stats.mode(data)[0][0],
                'std': np.std(data),
                'var': np.var(data),
                'min': np.min(data),
                'max': np.max(data),
                'range': np.max(data) - np.min(data),
                'skewness': stats.skew(data),
                'kurtosis': stats.kurtosis(data),
                'percentiles': {
                    '25%': np.percentile(data, 25),
                    '50%': np.percentile(data, 50),
                    '75%': np.percentile(data, 75),
                    '95%': np.percentile(data, 95)
                }
            }
        elif data_type == 'dataframe':
            stats_dict = {}
            for column in data.columns:
                if pd.api.types.is_numeric_dtype(data[column]):
                    stats_dict[column] = {
                        'mean': data[column].mean(),
                        'median': data[column].median(),
                        'mode': data[column].mode().iloc[0],
                        'std': data[column].std(),
                        'var': data[column].var(),
                        'min': data[column].min(),
                        'max': data[column].max(),
                        'range': data[column].max() - data[column].min(),
                        'skewness': data[column].skew(),
                        'kurtosis': data[column].kurtosis(),
                        'percentiles': {
                            '25%': data[column].quantile(0.25),
                            '50%': data[column].quantile(0.5),
                            '75%': data[column].quantile(0.75),
                            '95%': data[column].quantile(0.95)
                        }
                    }
        elif data_type == 'dict':
            stats_dict = {}
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)) and len(value) > 0:
                    arr = np.array(value)
                    stats_dict[key] = {
                        'mean': np.mean(arr),
                        'median': np.median(arr),
                        'mode': stats.mode(arr)[0][0],
                        'std': np.std(arr),
                        'var': np.var(arr),
                        'min': np.min(arr),
                        'max': np.max(arr),
                        'range': np.max(arr) - np.min(arr),
                        'skewness': stats.skew(arr),
                        'kurtosis': stats.kurtosis(arr),
                        'percentiles': {
                            '25%': np.percentile(arr, 25),
                            '50%': np.percentile(arr, 50),
                            '75%': np.percentile(arr, 75),
                            '95%': np.percentile(arr, 95)
                        }
                    }
        else:
            raise ValueError(f"Unsupported data type for statistics: {data_type}")
        
        return stats_dict
    
    def correlation_analysis(self, dataset_name, method='pearson'):
        """Perform correlation analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        if data_type == 'array':
            if data.ndim == 1:
                raise ValueError("Correlation analysis requires at least 2D data")
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(data)
        elif data_type == 'dataframe':
            df = data.copy()
        elif data_type == 'dict':
            # Convert dict to DataFrame
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type for correlation: {data_type}")
        
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = df.corr(method='pearson')
        elif method == 'spearman':
            corr_matrix = df.corr(method='spearman')
        elif method == 'kendall':
            corr_matrix = df.corr(method='kendall')
        else:
            raise ValueError(f"Unknown correlation method: {method}")
        
        return corr_matrix
    
    def hypothesis_testing(self, dataset_name, test_type, **kwargs):
        """Perform hypothesis testing on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Convert to numpy array if needed
        if data_type == 'dataframe':
            data = data.values
        elif data_type == 'dict':
            # For dict, we need to extract arrays
            arrays = []
            for key, value in data.items():
                if isinstance(value, (list, np.ndarray)):
                    arrays.append(np.array(value))
            data = np.array(arrays)
        
        result = {}
        
        if test_type == 'ttest_1samp':
            # One-sample t-test
            popmean = kwargs.get('popmean', 0)
            statistic, pvalue = stats.ttest_1samp(data, popmean)
            result = {
                'test_type': 'One-sample t-test',
                'statistic': statistic,
                'pvalue': pvalue,
                'popmean': popmean
            }
        
        elif test_type == 'ttest_ind':
            # Independent two-sample t-test
            if len(data.shape) < 2 or data.shape[0] < 2:
                raise ValueError("Independent t-test requires at least two samples")
            
            sample1 = data[0]
            sample2 = data[1]
            equal_var = kwargs.get('equal_var', True)
            
            statistic, pvalue = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
            result = {
                'test_type': 'Independent two-sample t-test',
                'statistic': statistic,
                'pvalue': pvalue,
                'equal_var': equal_var
            }
        
        elif test_type == 'ttest_rel':
            # Related/paired t-test
            if len(data.shape) < 2 or data.shape[0] < 2:
                raise ValueError("Paired t-test requires at least two samples")
            
            sample1 = data[0]
            sample2 = data[1]
            
            statistic, pvalue = stats.ttest_rel(sample1, sample2)
            result = {
                'test_type': 'Paired t-test',
                'statistic': statistic,
                'pvalue': pvalue
            }
        
        elif test_type == 'chi2_contingency':
            # Chi-square test of independence
            observed = kwargs.get('observed', data)
            chi2, pvalue, dof, expected = stats.chi2_contingency(observed)
            result = {
                'test_type': 'Chi-square test of independence',
                'statistic': chi2,
                'pvalue': pvalue,
                'dof': dof,
                'expected': expected.tolist()
            }
        
        elif test_type == 'anova':
            # One-way ANOVA
            if len(data.shape) < 2 or data.shape[0] < 2:
                raise ValueError("ANOVA requires at least two samples")
            
            statistic, pvalue = stats.f_oneway(*data)
            result = {
                'test_type': 'One-way ANOVA',
                'statistic': statistic,
                'pvalue': pvalue
            }
        
        elif test_type == 'normaltest':
            # Test for normality
            statistic, pvalue = stats.normaltest(data)
            result = {
                'test_type': "D'Agostino's normality test",
                'statistic': statistic,
                'pvalue': pvalue
            }
        
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Add interpretation
        alpha = kwargs.get('alpha', 0.05)
        if result['pvalue'] < alpha:
            result['conclusion'] = f"Reject the null hypothesis at the {alpha} significance level"
        else:
            result['conclusion'] = f"Fail to reject the null hypothesis at the {alpha} significance level"
        
        return result
    
    def regression_analysis(self, dataset_name, x_column=None, y_column=None, regression_type='linear'):
        """Perform regression analysis on a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        if data_type == 'array':
            if data.ndim == 1:
                raise ValueError("Regression analysis requires at least 2D data")
            # Convert to DataFrame for easier handling
            df = pd.DataFrame(data)
        elif data_type == 'dataframe':
            df = data.copy()
        elif data_type == 'dict':
            # Convert dict to DataFrame
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type for regression: {data_type}")
        
        # Extract x and y data
        if x_column is None:
            x = df.iloc[:, :-1].values
        else:
            x = df[x_column].values.reshape(-1, 1)
        
        if y_column is None:
            y = df.iloc[:, -1].values
        else:
            y = df[y_column].values
        
        result = {}
        
        if regression_type == 'linear':
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(x.flatten(), y)
            
            # Calculate predicted values
            y_pred = slope * x.flatten() + intercept
            
            # Calculate residuals
            residuals = y - y_pred
            
            result = {
                'type': 'Linear Regression',
                'slope': slope,
                'intercept': intercept,
                'r_value': r_value,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_err': std_err,
                'equation': f'y = {slope:.4f}x + {intercept:.4f}',
                'x': x.flatten(),
                'y': y,
                'y_pred': y_pred,
                'residuals': residuals
            }
        
        elif regression_type == 'polynomial':
            # Polynomial regression
            degree = kwargs.get('degree', 2)
            coeffs = np.polyfit(x.flatten(), y, degree)
            
            # Calculate predicted values
            poly_func = np.poly1d(coeffs)
            y_pred = poly_func(x.flatten())
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Calculate R-squared
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Format equation
            equation = "y = "
            for i, coef in enumerate(coeffs):
                power = degree - i
                if power == 0:
                    equation += f"{coef:.4f}"
                elif power == 1:
                    equation += f"{coef:.4f}x + "
                else:
                    equation += f"{coef:.4f}x^{power} + "
            
            result = {
                'type': f'Polynomial Regression (degree={degree})',
                'coefficients': coeffs.tolist(),
                'r_squared': r_squared,
                'equation': equation,
                'x': x.flatten(),
                'y': y,
                'y_pred': y_pred,
                'residuals': residuals
            }
        
        else:
            raise ValueError(f"Unknown regression type: {regression_type}")
        
        return result
    
    def distribution_analysis(self, dataset_name, column=None):
        """Analyze the distribution of data"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    sample = data[:, column]
                else:
                    sample = data.flatten()
            else:
                sample = data
        elif data_type == 'dataframe':
            if column is not None:
                sample = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    sample = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                sample = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        sample = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for distribution analysis: {data_type}")
        
        # Fit various distributions
        distributions = [
            ('norm', stats.norm),
            ('lognorm', stats.lognorm),
            ('expon', stats.expon),
            ('gamma', stats.gamma),
            ('beta', stats.beta),
            ('weibull_min', stats.weibull_min)
        ]
        
        results = {}
        
        for dist_name, dist in distributions:
            try:
                # Fit distribution
                params = dist.fit(sample)
                
                # Separate parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]
                
                # Calculate PDF
                x = np.linspace(min(sample), max(sample), 100)
                pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                
                # Calculate Kolmogorov-Smirnov test
                ks_stat, ks_p = stats.kstest(sample, dist_name, params)
                
                results[dist_name] = {
                    'params': params,
                    'arg': arg,
                    'loc': loc,
                    'scale': scale,
                    'x': x,
                    'pdf': pdf,
                    'ks_stat': ks_stat,
                    'ks_p': ks_p
                }
            except Exception as e:
                results[dist_name] = {'error': str(e)}
        
        # Find best fitting distribution (lowest KS statistic)
        best_fit = None
        best_ks = float('inf')
        
        for dist_name, result in results.items():
            if 'error' not in result and result['ks_stat'] < best_ks:
                best_fit = dist_name
                best_ks = result['ks_stat']
        
        return {
            'sample': sample,
            'distributions': results,
            'best_fit': best_fit,
            'best_ks': best_ks
        }
    
    def outlier_detection(self, dataset_name, column=None, method='iqr', threshold=1.5):
        """Detect outliers in a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        
        dataset = self.datasets[dataset_name]
        data = dataset['data']
        data_type = dataset['type']
        
        # Extract data
        if data_type == 'array':
            if data.ndim > 1:
                if column is not None:
                    sample = data[:, column]
                else:
                    sample = data.flatten()
            else:
                sample = data
        elif data_type == 'dataframe':
            if column is not None:
                sample = data[column].values
            else:
                # Use the first numeric column
                numeric_columns = data.select_dtypes(include=[np.number]).columns
                if len(numeric_columns) > 0:
                    sample = data[numeric_columns[0]].values
                else:
                    raise ValueError("No numeric columns found in the DataFrame")
        elif data_type == 'dict':
            if column is not None and column in data:
                sample = np.array(data[column])
            else:
                # Use the first array in the dict
                for key, value in data.items():
                    if isinstance(value, (list, np.ndarray)):
                        sample = np.array(value)
                        break
                else:
                    raise ValueError("No arrays found in the dictionary")
        else:
            raise ValueError(f"Unsupported data type for outlier detection: {data_type}")
        
        outliers = None
        outlier_indices = None
        
        if method == 'iqr':
            # IQR method
            q1 = np.percentile(sample, 25)
            q3 = np.percentile(sample, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (sample < lower_bound) | (sample > upper_bound)
            outliers = sample[outlier_mask]
            outlier_indices = np.where(outlier_mask)[0]
            
            result = {
                'method': 'IQR',
                'threshold': threshold,
                'q1': q1,
                'q3': q3,
                'iqr': iqr,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers': outliers.tolist(),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outliers)
            }
        
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs(stats.zscore(sample))
            outlier_mask = z_scores > threshold
            outliers = sample[outlier_mask]
            outlier_indices = np.where(outlier_mask)[0]
            
            result = {
                'method': 'Z-score',
                'threshold': threshold,
                'z_scores': z_scores.tolist(),
                'outliers': outliers.tolist(),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outliers)
            }
        
        elif method == 'modified_zscore':
            # Modified Z-score method (based on median and MAD)
            median = np.median(sample)
            mad = np.median(np.abs(sample - median))
            modified_z_scores = 0.6745 * (sample - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold
            outliers = sample[outlier_mask]
            outlier_indices = np.where(outlier_mask)[0]
            
            result = {
                'method': 'Modified Z-score',
                'threshold': threshold,
                'median': median,
                'mad': mad,
                'modified_z_scores': modified_z_scores.tolist(),
                'outliers': outliers.tolist(),
                'outlier_indices': outlier_indices.tolist(),
                'outlier_count': len(outliers)
            }
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return result
    
    def get_dataset(self, dataset_name):
        """Get a dataset by name"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset not found: {dataset_name}")
        return self.datasets[dataset_name]
    
    def list_datasets(self):
        """List all available datasets"""
        return list(self.datasets.keys())
    
    def remove_dataset(self, dataset_name):
        """Remove a dataset by name"""
        if dataset_name in self.datasets:
            del self.datasets[dataset_name]
            return True
        return False