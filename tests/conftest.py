"""Make `src/` importable so `import scml` works when running tests from root."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
