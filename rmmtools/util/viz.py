import random


def generate_hex_colors(n=100, seed=None):
    """Generates a list of n random hex colors."""

    if seed is not None:
        random.seed(seed)
    colors = []
    for i in range(n):
        colors.append(
            "#" + "".join(random.choice("0123456789ABCDEF") for j in range(6))
        )
    return colors
