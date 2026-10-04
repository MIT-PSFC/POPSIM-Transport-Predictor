"""Shared dark-theme styling for the study figures."""

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"

# Font sizes of the study comparison figures and the case report pages
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 9
LEGEND_FONTSIZE = 10

# Legend colors, passed as ax.legend(..., **LEGEND_STYLE)
LEGEND_STYLE = {"labelcolor": TEXT_COLOR, "facecolor": BACKGROUND_COLOR, "edgecolor": TEXT_COLOR}


def style_axis(ax, tick_fontsize: int = TICK_FONTSIZE):
    """Apply the dark background, tick, spine, and grid styling to one axis."""
    ax.set_facecolor(FACE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=TEXT_COLOR)
