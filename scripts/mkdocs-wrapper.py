#!/usr/bin/env python3
"""
Wrapper for mkdocs serve that patches os.getcwd() before mkdocs loads.
The Claude Preview runner's sandbox has an inaccessible working directory,
which breaks os.path.abspath() calls inside mkdocs at startup.
"""
import os, sys

PROJECT_ROOT = "/Users/anvit/Documents/Projects/Textbooks/Graph-Neural-Networks-Textbook"
_real_getcwd = os.getcwd

def _patched_getcwd():
    try:
        return _real_getcwd()
    except (PermissionError, OSError):
        return PROJECT_ROOT

os.getcwd = _patched_getcwd
os.chdir(PROJECT_ROOT)

sys.argv = ["mkdocs", "serve", "--dev-addr", "127.0.0.1:8000"]
from mkdocs.__main__ import cli
cli()
