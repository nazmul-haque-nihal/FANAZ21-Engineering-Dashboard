from flask_login import UserMixin, login_user, logout_user
from werkzeug.security import generate_password_hash, check_password_hash

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        self.name = id
        self.role = 'admin' if id == 'admin' else 'user'

# Hardcoded user credentials (in production, use a database)
users = {
    'admin': generate_password_hash('fanaz21'),
    'engineer': generate_password_hash('eng123'),
    'analyst': generate_password_hash('ana123')
}

def authenticate(username, password):
    if username in users and check_password_hash(users[username], password):
        user = User(username)
        login_user(user)
        return True
    return False

def logout():
    logout_user()

def is_admin(user):
    return user.role == 'admin' if user else False

def is_engineer(user):
    return user.role in ['admin', 'engineer'] if user else False

def is_analyst(user):
    return user.role in ['admin', 'engineer', 'analyst'] if user else False
