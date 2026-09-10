import os
import sys

# Make the `georecon` package importable when pytest runs from backend/.
sys.path.insert(0, os.path.dirname(__file__))
