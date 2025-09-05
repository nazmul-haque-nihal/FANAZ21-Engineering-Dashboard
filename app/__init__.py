# app/__init__.py
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from flask import Flask
from flask_login import LoginManager, current_user
import os
from .config import Config

def create_app():
    # Create Flask server
    server = Flask(__name__)
    server.config.from_object(Config)
    server.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    server.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
    
    # Ensure upload folder exists
    os.makedirs(server.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Initialize Flask-Login
    login_manager = LoginManager()
    login_manager.init_app(server)
    login_manager.login_view = '/login'
    
    # Create Dash app
    app = dash.Dash(
        __name__,
        server=server,
        external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
        meta_tags=[
            {"name": "viewport", "content": "width=device-width, initial-scale=1"},
            {"http-equiv": "X-UA-Compatible", "content": "IE=edge"}
        ]
    )
    
    app.title = 'FANAZ21 Engineering Dashboard'
    server.config['app'] = app
    
    # Define the layout
    app.layout = html.Div([
        dcc.Location(id='url', refresh=False),
        dcc.Store(id='session-store', storage_type='session'),
        html.Div(id='page-content')
    ])
    
    return app, server, login_manager

@login_manager.user_loader
def load_user(user_id):
    from app.auth import User
    return User(user_id)