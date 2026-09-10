import os
import sys

_here = os.path.dirname(__file__)
# Make the `georecon` package and the repo `scripts/` importable when pytest
# runs from backend/.
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(os.path.dirname(_here), "scripts"))
