import os
import sys

# Ensure backend root is on Python path for imports like `from app.main import app`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app
