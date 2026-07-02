"""
NeuroIgniter AI Recruiter — Streamlit Cloud entrypoint.
Streamlit Community Cloud expects the main app file at the repo root
(or a configurable path). This thin shim sets up the Python path and
delegates to the actual dashboard implementation.

Deployment: https://share.streamlit.io
  Repository: NeuroIgniter/ai-recruiter-ranker
  Branch:     main
  Main file:  streamlit_app.py

The dashboard reads the pre-computed ranking CSV from output/ —
the submission CSV is bundled in the repo so judges can run the
demo without needing the 487MB candidates.jsonl file.
"""

import sys
import os

# Ensure src/ is importable when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Execute the dashboard module
exec(open(os.path.join(os.path.dirname(__file__), 'dashboard', 'app.py')).read())
