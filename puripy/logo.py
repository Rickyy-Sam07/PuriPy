import os
from pathlib import Path


def get_logo_path():
    """Return the absolute path to the CleansiPy logo file."""
    return os.path.join(os.path.dirname(__file__), 'assets', 'logo.png')


def show_logo():
    """Display the CleansiPy logo if possible (Jupyter/IPython or print path in terminal)."""
    logo_path = get_logo_path()
    try:
        from IPython.display import Image, display
        display(Image(filename=logo_path))
    except ImportError:
        print(f"[CleansiPy logo] {logo_path}")
