# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'engineering-dashboard-secret-key'
    UPDATE_INTERVAL = 3000  # 3 seconds in milliseconds for real-time updates
    
    # File upload settings
    UPLOAD_FOLDER = 'data/uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'json', 'txt', 'mat', 'h5'}
    
    # Theme settings
    THEME_COLOR = '#0d6efd'  # Primary blue color
    SECONDARY_COLOR = '#6c757d'  # Secondary gray color
    SUCCESS_COLOR = '#198754'  # Success green color
    DANGER_COLOR = '#dc3545'  # Danger red color
    WARNING_COLOR = '#ffc107'  # Warning yellow color
    INFO_COLOR = '#0dcaf0'  # Info cyan color
    
    # Chart settings
    CHART_HEIGHT = 400
    CHART_TEMPLATE = 'plotly_dark'
    
    # Data settings
    SAMPLE_DATA_PATHS = {
        'simulation': 'data/simulation/sample_simulation_data.py',
        'analysis': 'data/analysis/sample_analysis_data.py',
        'modeling': 'data/modeling/sample_model_data.py'
    }