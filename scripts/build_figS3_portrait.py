#!/usr/bin/env python3
"""Rebuild Figure S3 (organ-specific ROC/PR curves) as a portrait figure.

Self-contained: reads only source_data/curve_coordinates.json.
Canvas sized close to final placement so nominal font sizes hold in print.
Replaces the sideways (landscape) version.
"""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
DATA = OUT / 'source_data'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5,
    'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9aa9b4', 'text.color': '#203347',
    'axes.labelcolor': '#203347', 'savefig.facecolor': 'white',
    'figure.facecolor': 'white'})
C = {'sup': '#347daf', 'pnu': '#008b7f', 'os': '#778694'}
cc = json.load(open(DATA / 'curve_coordinates.json'))
bkey = 'Breast' if any(k.startswith('Breast_') for k in cc) else 'Mammary'
ORG = [('Liver', 'Liver', 311, 293, 18), ('Lung', 'Lung', 211, 161, 50),
       ('Stomach', 'Stomach', 36, 25, 11), ('Breast', bkey, 89, 58, 31)]

f = plt.figure(figsize=(6.5, 6.6))
gs = f.add_gridspec(4, 2, left=.115, right=.985, bottom=.06, top=.955,
                    hspace=.42, wspace=.18)
for r, (disp, key, n, pos, neg) in enumerate(ORG):
    for c, (pr, letter) in enumerate([(0, 'ABCD'[r]), (1, 'EFGH'[r])]):
        ax = f.add_subplot(gs[r, c])
        def seeds(model):
            out = []
            for s in range(5):
                kk = f'{key}_{model}_{pr}_{s}'
                if kk in cc: out.append(kk)
            return out
        mean_m = lambda model: float(np.mean([cc[k]['metric'] for k in seeds(model)]))
        for model, col in [('Supervised', C['sup']), ('PNU', C['pnu'])]:
            for kk in seeds(model):
                cv = cc[kk]
                ax.plot(cv['x'], cv['y'], color=col, lw=.6, alpha=.4, zorder=2)
        osk = f'{key}_Released OS_{pr}_0'
        ax.plot(cc[osk]['x'], cc[osk]['y'], color=C['os'], lw=1.3,
                zorder=3)
        if pr == 0:
            ax.plot([0, 1], [0, 1], ls=':', color='#9aa9b4', lw=.9, zorder=1)
            loc = 'lower right'
        else:
            ax.axhline(pos / n, ls=':', color='#9aa9b4', lw=.9, zorder=1)
            loc = 'lower left'
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
        ax.grid(color='#e7edf2', lw=.5); ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=2)
        labels = [(f'Released OS {cc[osk]["metric"]:.3f}', C['os']),
                  (f'Supervised {mean_m("Supervised"):.3f}', C['sup']),
                  (f'PNU {mean_m("PNU"):.3f}', C['pnu'])]
        x = .97 if pr == 0 else .03
        ha = 'right' if pr == 0 else 'left'
        for j, (label, color) in enumerate(labels):
            ax.text(x, .20 - .07 * j, label, transform=ax.transAxes,
                    ha=ha, va='bottom', fontsize=5.8, color=color)
        sign = '+' if pr == 0 else '+'
        ax.set_title(f'{letter}  {disp} (n={n}; {sign}{pos}/-{neg})',
                     fontsize=8, loc='left', fontweight='bold', pad=4)
        ax.set_ylabel('True-positive rate' if pr == 0 else 'Precision', fontsize=7)
        if r == 3:
            ax.set_xlabel('False-positive rate' if pr == 0 else 'Recall',
                          fontsize=7)
for ext in ['pdf', 'svg', 'png']:
    name = 'fig3_organ_curves' + ('_manuscript' if ext == 'pdf' else '') + '.' + ext
    f.savefig(OUT / name, dpi=300)
plt.close(f)
print('Saved fig3_organ_curves portrait from packaged source_data only.')
