"""Shared figure style for everything in figures/.

Palette is an Okabe-Ito subset. Purple is left out because it collides with the
green under deuteranopia.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from fnirs_glmhmm import config

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GREY = "#595959"
LIGHT = "#BBBBBB"

# Fixed assignment, never cycled: each measure keeps its colour across every figure.
MEASURE_COLORS = {
    "HMM (raw VTC)": BLUE,
    "HMM (smoothed)": GREEN,
    "Median split": VERMILLION,
}
ERROR_COLORS = {"omission": BLUE, "commission": VERMILLION}

TRIAL_S = 0.8  # one gradCPT trial


def set_style() -> None:
    sns.set_theme(style="ticks", context="paper")
    mpl.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.titleweight": "normal",
            "axes.titlelocation": "left",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "legend.frameon": False,
            "svg.fonttype": "none",  # keep text as text in the SVG
            "pdf.fonttype": 42,
        }
    )


def save(fig, name: str, outdir: Path | None = None) -> list[Path]:
    """Write both formats side by side and return the paths."""
    outdir = Path(outdir or config.FIGURES_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("svg", "png"):
        p = outdir / f"{name}.{ext}"
        fig.savefig(p, format=ext)
        paths.append(p)
    plt.close(fig)
    return paths
