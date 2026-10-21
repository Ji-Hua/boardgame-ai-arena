"""Entry point for running the Training Dashboard backend.

Run from repo root:
    python -m training_dashboard_backend
Or:
    uvicorn training_dashboard_backend:app --port 8740
"""

import sys
from pathlib import Path

# Add the training-dashboard directory to the Python path
_td_dir = Path(__file__).parent
sys.path.insert(0, str(_td_dir))

from backend.app import app  # noqa: E402, F401
