# check_imports.py
import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("Python path:")
for p in sys.path:
    print(f"  {p}")

print("\nTrying to import modules:")

try:
    from app.config import Config
    print("✓ Successfully imported app.config")
except ImportError as e:
    print(f"✗ Failed to import app.config: {e}")

try:
    from app import create_app
    print("✓ Successfully imported app.create_app")
except ImportError as e:
    print(f"✗ Failed to import app.create_app: {e}")