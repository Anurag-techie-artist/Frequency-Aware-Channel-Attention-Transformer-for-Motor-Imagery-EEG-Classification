"""
Notebook Helper Utilities for Phase Walkthrough Notebooks.

Provides helper routines for path setup, plot styling, and HTML table formatting.
Does NOT duplicate production code.
"""

import os
import sys
import matplotlib.pyplot as plt
from IPython.display import display, HTML


def setup_notebook_env():
    """Ensure project root directory is added to sys.path."""
    nb_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(nb_dir, "..", ".."))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root


def set_notebook_style():
    """Configure matplotlib defaults for publication-style inline rendering."""
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#D0D7DE"
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["grid.color"] = "#E1E4E8"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.6


def display_html_table(headers, data, title=None):
    """Render a clean HTML table inside Jupyter Notebook cell outputs."""
    html = ""
    if title:
        html += f"<h4 style='color: #1F2328; margin-top: 10px; margin-bottom: 8px;'>{title}</h4>"
    html += "<table style='border-collapse: collapse; width: 100%; font-family: sans-serif; font-size: 13px;'>"
    html += "<thead><tr style='background-color: #F6F8FA; border-bottom: 2px solid #D0D7DE; text-align: left;'>"
    for h in headers:
        html += f"<th style='padding: 8px 12px; color: #24292F;'>{h}</th>"
    html += "</tr></thead><tbody>"
    for r_idx, row in enumerate(data):
        bg = "#FFFFFF" if r_idx % 2 == 0 else "#F6F8FA"
        html += f"<tr style='background-color: {bg}; border-bottom: 1px solid #E1E4E8;'>"
        for col in row:
            html += f"<td style='padding: 8px 12px; color: #57606A;'>{col}</td>"
        html += "</tr>"
    html += "</tbody></table>"
    display(HTML(html))
