# ---
# description: |
#   Core notebook setup utility for CrSBr analysis. Configures matplotlib
#   with publication-quality RC params and the Okabe-Ito 8-color colorblind-safe
#   palette, enables autoreload, and returns common library handles.
# entry_point: from scripts.utils import setup_notebook; PROJECT_ROOT, np, pd, plt, Path = setup_notebook()
# dependencies:
#   - matplotlib
#   - numpy
#   - pandas
#   - IPython (optional, for autoreload)
# input: |
#   No CLI arguments. Called from Jupyter notebooks. Automatically locates
#   project root by searching for a 'scripts' directory in the path hierarchy.
# process: |
#   Enables IPython autoreload, imports scientific libraries, sets matplotlib
#   RC params (Okabe-Ito 8-color palette, large fonts, thick spines/ticks),
#   and resolves the project root path.
# output: |
#   Returns (PROJECT_ROOT, np, pd, plt, Path). Side effect: matplotlib global
#   RC params are updated for all subsequent plots in the session.
# last_updated: 2026-04-09
# ---


# ---------------------------------------------------------------------------
# Okabe & Ito (2008) colorblind-safe palette — 8 colors
# Source: Okabe, M. & Ito, K. (2008). Color Universal Design (CUD).
#         https://jfly.uni-koeln.de/color/
# Named dict for use when selecting specific colors by role.
# ---------------------------------------------------------------------------
OKABE_ITO = {
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'soft_violet': '#7B5EA7',
}

# Ordered list for matplotlib prop_cycle — yellow last (low contrast on white)
OKABE_ITO_CYCLE = [
    '#E69F00',  # orange
    '#56B4E9',  # sky blue
    '#009E73',  # bluish green
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#7B5EA7',  # soft violet
]

# Convenient aliases for common plot elements (Okabe-Ito colors)
COLOR_B_AXIS = '#0072B2'   # blue
COLOR_C_AXIS = '#D55E00'   # vermillion
COLOR_G_AXIS = '#009E73'   # bluish green

import sys
from pathlib import Path


def setup_notebook():
    """
    Set up notebook environment for CrSBr analysis.

    This function:
    1. Enables IPython autoreload for automatic module reloading
    2. Imports and returns common scientific computing libraries
    3. Configures matplotlib plotting defaults
    4. Determines and returns project root path

    Returns:
        tuple: (PROJECT_ROOT, np, pd, plt, Path)
            - PROJECT_ROOT: Path to project root directory
            - np: numpy module
            - pd: pandas module
            - plt: matplotlib.pyplot module
            - Path: pathlib.Path class

    Usage:
        from scripts.utils import setup_notebook
        PROJECT_ROOT, np, pd, plt, Path = setup_notebook()
    """

    # Configure automatic reloading of modules
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic('load_ext', 'autoreload')
            ipython.run_line_magic('autoreload', '2')
            print("[OK] Autoreload enabled - modules will reload automatically")
            print("     Scripts will update immediately when you edit them!")
    except Exception as e:
        print(f"[WARNING] Could not enable autoreload: {e}")

    # Standard scientific computing imports
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Project path setup
    def _find_project_root():
        """Find project root by locating scripts package"""
        try:
            import scripts
            if hasattr(scripts, '__file__') and scripts.__file__ is not None:
                project_root = Path(scripts.__file__).parent.parent
                print(f"[OK] Using installed 'scripts' package from: {project_root}")
                return project_root
            else:
                print("[OK] 'scripts' package found, locating project root...")
                raise ImportError("Finding root manually")
        except (ImportError, AttributeError, TypeError):
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                if (parent / 'scripts').exists():
                    if str(parent) not in sys.path:
                        sys.path.insert(0, str(parent))
                    print(f"[OK] Added project root to path: {parent}")
                    return parent
            raise RuntimeError("Could not find project root (no 'scripts' directory found)")

    PROJECT_ROOT = _find_project_root()

    # Configure matplotlib for publication-quality plots
    configure_plot_style()
    print("[OK] Matplotlib configured with publication style (Okabe-Ito 8 colors)")

    print("=" * 60)
    print("Notebook setup complete! Ready for analysis.")
    print("=" * 60)

    return PROJECT_ROOT, np, pd, plt, Path


def configure_plot_style():
    """
    Configure matplotlib RC parameters for publication-quality, colorblind-friendly plots.

    Style features:
    - High DPI (150 screen / 300 saved figures)
    - Thick spines and tick marks (linewidth=2)
    - Large, readable fonts (20pt labels/title, 16pt ticks/legend)
    - Okabe-Ito 8-color colorblind-safe cycle
    - 'viridis' default colormap (colorblind-safe, perceptually uniform)
    - Ticks on all four sides, pointing inward (physics convention)

    Color palette source:
        Okabe, M. & Ito, K. (2008). Color Universal Design (CUD).
        https://jfly.uni-koeln.de/color/
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        # --- Figure ---
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'figure.figsize': (10, 7),
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'savefig.format': 'png',

        # --- Font (STIX Two — serif, Times-like, full math support) ---
        'font.family': 'STIXGeneral',
        'mathtext.fontset': 'stix',
        'font.size': 16,
        'axes.labelsize': 22,
        'axes.titlesize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 15,
        'legend.title_fontsize': 15,

        # --- Axes frame (thick outlines) ---
        'axes.linewidth': 2,
        'axes.edgecolor': 'black',
        'axes.labelcolor': 'black',
        'axes.grid': False,
        'axes.axisbelow': True,

        # --- Ticks (all four sides, inward, thick) ---
        'xtick.major.width': 2,
        'ytick.major.width': 2,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'xtick.bottom': True,
        'ytick.left': True,
        'ytick.right': True,

        # --- Lines and markers ---
        'lines.linewidth': 2,
        'lines.markersize': 7,

        # --- Legend ---
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': False,

        # --- Okabe-Ito 8-color colorblind-safe cycle ---
        # Source: Okabe & Ito (2008), https://jfly.uni-koeln.de/color/
        'axes.prop_cycle': mpl.cycler(color=OKABE_ITO_CYCLE),

        # --- Default colormap: viridis (colorblind-safe, perceptually uniform) ---
        'image.cmap': 'viridis',
        'image.aspect': 'auto',
    })


def set_high_dpi(dpi=1200):
    """Set high DPI for publication-quality figures."""
    import matplotlib.pyplot as plt
    plt.rcParams['savefig.dpi'] = dpi


def apply_thick_spines(ax=None, linewidth=2):
    """Apply thick spines to axes."""
    import matplotlib.pyplot as plt
    if ax is None:
        ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)


def create_publication_figure(dpi=600):
    """Create a figure configured for publication."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(dpi=dpi)
    apply_thick_spines(ax, linewidth=2)
    ax.tick_params(width=2)
    return fig, ax


# Color schemes
TEMP_COLORMAP = 'brg'
FIELD_COLORMAP = 'coolwarm'
ERROR_COLOR = 'm'

# Recommended sizes
MARKER_SIZE_SMALL = 3
MARKER_SIZE_MEDIUM = 6
MARKER_SIZE_LARGE = 50
LINE_WIDTH_THIN = 1
LINE_WIDTH_MEDIUM = 1.5
LINE_WIDTH_THICK = 2

__all__ = ['setup_notebook', 'configure_plot_style', 'set_high_dpi',
           'apply_thick_spines', 'create_publication_figure']
