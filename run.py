# run.py
import sys
import os

# Add the current directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now import the app
from app import create_app
from app.auth import logout, is_admin, is_engineer, is_analyst
from flask import request, redirect, url_for, render_template, send_from_directory, flash
from flask_login import login_required, current_user
import os
from werkzeug.utils import secure_filename

app, server, login_manager = create_app()

# Import callbacks after app creation to avoid circular imports
with server.app_context():
    from app import callbacks

# Helper function to check allowed file extensions
def allowed_file(filename):
    from app.config import Config
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

# Login route
@server.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        from app.auth import authenticate
        if authenticate(username, password):
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error='Invalid credentials')
    return render_template('login.html')

# Logout route
@server.route('/logout')
@login_required
def logout_route():
    logout()
    return redirect('/login')

# Index route (protected)
@server.route('/')
@login_required
def index():
    return app.index()

# File upload route
@server.route('/upload/<module_type>', methods=['POST'])
@login_required
def upload_file(module_type):
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.referrer)
    
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.referrer)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(server.config['UPLOAD_FOLDER'], module_type)
        os.makedirs(upload_path, exist_ok=True)
        file_path = os.path.join(upload_path, filename)
        file.save(file_path)
        flash(f'File {filename} uploaded successfully for {module_type} module')
        return redirect(request.referrer)
    
    flash('File type not allowed')
    return redirect(request.referrer)

# Serve uploaded files
@server.route('/uploads/<module_type>/<filename>')
@login_required
def uploaded_file(module_type, filename):
    return send_from_directory(os.path.join(server.config['UPLOAD_FOLDER'], module_type), filename)

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)