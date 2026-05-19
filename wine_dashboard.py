"""
wine_dashboard.py · Streamlit Cloud entry point (root-level shim)
================================================================

This file is a *thin delegate* — it adds the sibling ``app/`` directory
to ``sys.path`` and runs the real dashboard from ``app/wine_dashboard.py``.

Why it exists
-------------
The dashboard's actual implementation, plus its 5 supporting modules,
live in ``app/`` (single-responsibility folder, clean separation from
data, deliverables, notebooks and docs). Streamlit Community Cloud,
however, looks for the entry script at the repository root by default
and changing the configured "Main file path" after deployment is
awkward — so we keep the entry name (``wine_dashboard.py``) where the
cloud expects it and delegate one level deep.

Local usage (either is fine):

    streamlit run wine_dashboard.py            # via this shim
    streamlit run app/wine_dashboard.py        # directly
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Make the support modules in app/ importable. We insert at the front of
# sys.path so 'from wine_data import ...' resolves to app/wine_data.py
# rather than to anything that happens to be lying around in site-packages.
APP_DIR = Path(__file__).resolve().parent / "app"
sys.path.insert(0, str(APP_DIR))

# Run the real dashboard with __name__ set to "__main__", so the
# `if __name__ == "__main__": main()` guard at the bottom of the real
# file still fires the way streamlit expects.
runpy.run_path(str(APP_DIR / "wine_dashboard.py"), run_name="__main__")
