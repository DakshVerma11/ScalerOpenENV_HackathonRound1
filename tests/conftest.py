import os
import sys

# Add the project root to sys.path so tests can import modules like 'models', 'server', etc.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
