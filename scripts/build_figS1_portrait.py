#!/usr/bin/env python3
"""Rebuild Figure S1 (complete nine-metric comparison) as a portrait figure.

Self-contained: reads only source_data/all_nine_metrics_both_thresholds.csv.
Canvas sized close to final placement (SI text width ~17cm) so nominal font
sizes hold in print. Replaces the sideways (landscape) version.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

OUT = Path(__file__).resolve().parent
DATA = OUT / 'source_data'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.titlesize': 8.5, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9aa9b4', 'text.color': '#203347',
    'axes.labelcolor': '#203347', 'savefig.facecolor': 'white',
    'figure.facecolor': 'white'})
MET = ['AUROC', 'AP', 'Accuracy', 'F1', 'Precision', 'Sensitivity',
       'Specificity', 'Balanced accuracy', 'MCC']
NAMES = ['OS-Carcinogenicity', 'Single-task GAT', 'MolBERT', 'CarcGC',
         'DCAMCP', 'XGBoost',
         'Random forest', 'Structure + supervised', 'CarcPNU']
tbl = pd.read_csv(DATA / 'all_nine_metrics_both_thresholds.csv')
v45 = tbl[(tbl.threshold == .45) & (tbl.organ == 'Macro')].set_index('model').loc[NAMES, MET].to_numpy(float)
v50 = tbl[(tbl.threshold == .50) & (tbl.organ == 'Macro')].set_index('model').loc[NAMES, MET].to_numpy(float)
TEAL, SLATE = '#008b7f', '#4a6274'

f = plt.figure(figsize=(6.5, 5.3))
gs = f.add_gridspec(3, 3, left=.19, right=.985, bottom=.115, top=.93,
                    hspace=.75, wspace=.42)
yy = np.arange(len(NAMES))[::-1]
for k, m in enumerate(MET):
    ax = f.add_subplot(gs[k // 3, k % 3])
    lo = min(v45[:, k].min(), v50[:, k].min())
    hi = max(v45[:, k].max(), v50[:, k].max())
    pad = .10 * (hi - lo) if hi > lo else .05
    ax.set_xlim(lo - pad, hi + pad)
    for j, name in enumerate(NAMES):
        c = TEAL if name == 'CarcPNU' else '#9aa9b4'
        ax.plot([v50[j, k], v45[j, k]], [yy[j]] * 2, color=c, lw=.9,
                alpha=.85, zorder=2)
    dotc = [TEAL if n == 'CarcPNU' else SLATE for n in NAMES]
    ax.scatter(v45[:, k], yy, s=14, c=dotc, zorder=3)
    ax.scatter(v50[:, k], yy, s=16, marker='x', linewidths=1.0,
               c=SLATE, zorder=3)
    ax.set_ylim(-.7, len(NAMES) - .3)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.grid(axis='x', color='#e7edf2', lw=.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=3)
    if k % 3 == 0:
        ax.set_yticks(yy, NAMES)
        ax.get_yticklabels()[-1].set_color(TEAL)
        ax.get_yticklabels()[-1].set_fontweight('bold')
    else:
        ax.set_yticks(yy, [''] * len(NAMES))
    ax.set_title('ABCDEFGHI'[k] + '  ' + m, loc='left', fontsize=8.5,
                 fontweight='bold', pad=6)
handles = [Line2D([], [], marker='o', linestyle='', color=SLATE, ms=5,
                  label='Threshold 0.45'),
           Line2D([], [], marker='x', linestyle='', color=SLATE, ms=5.5,
                  label='Threshold 0.50')]
f.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .008),
         ncol=2, frameon=False, fontsize=7, columnspacing=1.0,
         handletextpad=.3)
for ext in ['pdf', 'svg', 'png']:
    name = 'figS1_nine_metrics' + ('_manuscript' if ext == 'pdf' else '') + '.' + ext
    f.savefig(OUT / name, dpi=300)
plt.close(f)
print('Saved figS1 portrait from packaged source_data only.')
