import matplotlib
matplotlib.use('TkAgg')
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ALGS2COLORS = dict(zip(['utigsp', 'igsp', 'gies', 'jcigsp', 'igsp_r', 'igsp_pool', 'utigsp_pool', 'icp'], sns.color_palette()))
ALGS2MARKERS = {'utigsp': '*', 'igsp': 'o', 'gies': '+', 'icp': 's'}
LINESTYLES = ['-', '--', 'dotted']
MARKERS = ['x', 'o', '.', 's', '*', '+']
ALG_HANDLES = [Patch(color=c, label=alg) for alg, c in ALGS2COLORS.items()]
ALG_HANDLES_MARKER = [Line2D([], [], color='k', marker=m, label=alg) for alg, m in ALGS2MARKERS.items()]


def create_line_handles(lst):
    return [Line2D([], [], color='k', linestyle=ls, label=el) for el, ls in zip(lst, LINESTYLES)]


def create_color_handles(lst):
    return [Patch(color=c, label=el) for el, c in zip(lst, sns.color_palette())]


def create_marker_handles(lst):
    return [Line2D([], [], color='k', marker=m, label=el) for el, m in zip(lst, MARKERS)]
