"""Shared dark-theme styling for the study figures."""

BACKGROUND_COLOR = "#2F2F2F"
FACE_COLOR = "#1A1A1A"
TEXT_COLOR = "white"


def style_axis(ax, tick_fontsize: int = 9):
    """Apply the dark background, tick, spine, and grid styling to one axis."""
    ax.set_facecolor(FACE_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=tick_fontsize)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEXT_COLOR)
    ax.grid(True, alpha=0.2, color=TEXT_COLOR)
