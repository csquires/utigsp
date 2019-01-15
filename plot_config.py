import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ALGS2COLORS = dict(zip(['utigsp', 'igsp', 'gies', ''], sns.color_palette()))
LINESTYLES = ['-', '--', 'dotted']
ALG_HANDLES = [Patch(color=c, label=alg) for alg, c in ALGS2COLORS.items()]


def create_line_handles(lst):
    return [Line2D([], [], color='k', linestyle=ls, label=el) for el, ls in zip(lst, LINESTYLES)]
