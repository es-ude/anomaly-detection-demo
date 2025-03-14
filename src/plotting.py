import numpy as np
from matplotlib.colors import LinearSegmentedColormap


def hot_cmap(use_alpha: bool = False) -> LinearSegmentedColormap:
    colors = ["black", "red", "orange", "yellow", "white"]
    fractions = [0.0, 0.2, 0.6, 0.8, 1.0]

    if use_alpha:
        alphas = np.emath.logn(1000, np.linspace(1, 1000, len(colors)))
    else:
        alphas = np.ones(len(colors))

    anchor_points = list(zip(fractions, zip(colors, alphas)))

    return LinearSegmentedColormap.from_list("alpha_hot", anchor_points, N=512)
