from pathlib import Path
import matplotlib as mpl

ROOTDIR = Path(__file__).parent.parent.parents[0]


def verboseprint(inp, v=False):
    print(inp) if v else lambda *a, **k: None


quantitative_1 = mpl.cycler(
    color=["#b4222c", "#ebac23", "#2660a4", "f49fbc", "496c6e"]
)

quantitative_2 = mpl.cycler(
    color=['#016F01', '#F6511D', '#1F9EFF', '#624154', '#165865']
)

dark_theme_colors = mpl.cycler(
    color=["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65",
           "#beb9db", "#fdcce5", "#8bd3c7"]
)

cm = 1 / 2.54  # centimeters in inches