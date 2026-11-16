#!/usr/bin/env python3
"""Recompose archived results into the merged main figure and focused SI figures.

Reads exported numerical tables only. No model fitting or threshold selection.
The retained benchmark, molecular-case, and sensitivity figure files are untouched.
"""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
DATA = OUT / 'source_data'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10.5,
    'axes.titlesize': 12, 'axes.labelsize': 10.5,
    'xtick.labelsize': 10, 'ytick.labelsize': 10.5,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#9aa9b4', 'text.color': '#203347',
    'axes.labelcolor': '#203347', 'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
})
C = {'sup': '#347daf', 'pnu': '#008b7f', 'pn': '#c28b36',
     'pu': '#9776b6', 'graph': '#7c8b99', 'train': '#b4bfca', 'bad': '#c45b58'}
ORG = ['Liver', 'Lung', 'Stomach', 'Mammary']
MET = ['AUROC', 'AP', 'Accuracy', 'F1', 'Precision', 'Sensitivity',
       'Specificity', 'Balanced accuracy', 'MCC']
COL = ['AUROC', 'AP', 'accuracy', 'F1', 'precision', 'sensitivity',
       'specificity', 'balanced_accuracy', 'MCC']
inputs = ['representation_metrics.csv', 'archived_macro_metrics.csv',
          'archived_paired_differences.csv', 'chemical_space.csv',
          'training_space.csv', 'threshold_transitions.csv',
          'all_prediction_changes.csv']
rep = pd.read_csv(DATA / inputs[0])
abl = pd.read_csv(DATA / inputs[1])
abl = abl[abl.cohort.eq('original')].set_index('method')
pairs = pd.read_csv(DATA / inputs[2])
space = pd.read_csv(DATA / inputs[3])
train = pd.read_csv(DATA / inputs[4])
trans = pd.read_csv(DATA / inputs[5]).set_index('organ').loc[ORG].reset_index()
changes = pd.read_csv(DATA / inputs[6])
branches = ['Local graph', 'FP supervised', 'Fusion supervised', 'FP PNU', 'Fusion PNU']
macro = rep.groupby('branch')[MET].mean().loc[branches]
for name, method in [('FP supervised', 'fingerprint_supervised'),
                     ('Fusion supervised', 'fusion_supervised'),
                     ('FP PNU', 'fingerprint_PNU'), ('Fusion PNU', 'fusion_PNU')]:
    np.testing.assert_allclose(macro.loc[name].to_numpy(),
                               abl.loc[method, COL].to_numpy(dtype=float), atol=1e-12)
fusion_delta = np.array([
    rep[rep.branch.eq('Fusion ' + loss)].set_index('organ').loc[ORG, 'AUROC'].to_numpy()
    - rep[rep.branch.eq('FP ' + loss)].set_index('organ').loc[ORG, 'AUROC'].to_numpy()
    for loss in ['supervised', 'PNU']
])


def save(fig, name):
    for extension in ['pdf', 'png', 'svg']:
        suffix = '_manuscript' if extension == 'pdf' else ''
        fig.savefig(OUT / (name + suffix + '.' + extension), dpi=240,
                    bbox_inches='tight', pad_inches=.09)
    plt.close(fig)
    print('Saved', name, flush=True)


def heading(fig, x, y, letter, title):
    fig.text(x, y, f'{letter}  {title}', fontsize=12.5,
             fontweight='bold', va='bottom')


def radar(ax, branch):
    theta = np.linspace(0, 2*np.pi, len(MET), endpoint=False)
    closed = np.r_[theta, theta[0]]
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    labels = [s.replace('Balanced accuracy', 'Balanced\naccuracy') for s in MET]
    ax.set_xticks(theta, labels)
    ax.tick_params(axis='x', labelsize=9.5, pad=8)
    for angle, tick in zip(theta, ax.get_xticklabels()):
        side = np.sin(angle)
        tick.set_horizontalalignment('left' if side > .2 else 'right' if side < -.2 else 'center')
    ax.set_ylim(0, 1)
    ax.set_yticks([.25, .5, .75, 1], ['0.25', '0.50', '0.75', '1.00'],
                 fontsize=7, color='#778b9c')
    ax.set_rlabel_position(20)
    ax.grid(color='#d6e0e7', linewidth=.7)
    ax.spines['polar'].set_color('#c4d1db')
    ax.set_facecolor('#fafcfd')
    for label, loss, color, style in [
        ('Supervised', 'supervised', C['sup'], '--'),
        ('PN-only', 'PN_only', C['pn'], '-.'),
        ('PU-only', 'PU_only', C['pu'], ':'),
        ('PNU', 'PNU', C['pnu'], '-'),
    ]:
        vals = abl.loc[branch + '_' + loss, COL].to_numpy(dtype=float)
        assert np.isfinite(vals).all() and (vals >= 0).all() and (vals <= 1).all()
        v = np.r_[vals, vals[0]]
        ax.plot(closed, v, color=color, linestyle=style, marker='o',
                markersize=3, linewidth=2.2 if loss == 'PNU' else 1.5, label=label)
        ax.fill(closed, v, color=color, alpha=.035)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.22), frameon=False,
              ncol=2, fontsize=9.5, columnspacing=1.4, handlelength=2.5)


def intervals(ax, metric):
    ax.axvline(0, linestyle=':', color=C['graph'], linewidth=1.2)
    for i, method in enumerate(['supervised', 'PN_only', 'PU_only']):
        r = pairs[(pairs.cohort == 'original') & (pairs.branch == 'fusion')
                  & (pairs.method_A == 'PNU') & (pairs.method_B == method)
                  & (pairs.organ == 'macro') & (pairs.metric == metric)].iloc[0]
        expected = abl.loc['fusion_PNU', metric] - abl.loc['fusion_' + method, metric]
        np.testing.assert_allclose(r.delta, expected, atol=1e-12)
        y = 2 - i
        ax.hlines(y, r.CI95_low, r.CI95_high, color=C['pnu'], linewidth=2.5)
        ax.scatter(r.delta, y, s=50, color=C['pnu'], zorder=3)
        ax.text(r.delta, y + .22, f'{r.delta:+.4f}', ha='center', fontsize=10.5)
    ax.set_yticks([2, 1, 0], ['vs supervised', 'vs PN-only', 'vs PU-only'])
    ax.set_ylim(-.5, 2.65)
    ax.set_xlim(*((-.02, .108) if metric == 'AUROC' else (-.007, .054)))
    ax.set_xlabel('PNU minus comparator')
    ax.grid(axis='x', color='#e7edf2')
    ax.set_axisbelow(True)


# Main Figure 3: A-C are structural contributions; D-F are matched loss ablations.
f = plt.figure(figsize=(12.4, 11.6))
point_colors = [C['graph'], C['sup'], C['sup'], C['pnu'], C['pnu']]
for j, x in enumerate([.21, .64]):
    ax = f.add_axes([x, .755, .30, .18])
    vals = macro.iloc[:, j].to_numpy()
    yy = np.arange(5)[::-1]
    ax.scatter(vals, yy, c=point_colors, s=52, zorder=3)
    for value, y in zip(vals, yy):
        ax.annotate(f'{value:.4f}', (value, y), xytext=(7, 0),
                    textcoords='offset points', va='center', fontsize=11)
    ax.set_yticks(yy, branches if j == 0 else ['']*5)
    ax.tick_params(axis='y', length=0, pad=8)
    ax.set_ylim(-.6, 4.6)
    ax.set_xlim(*((.60, .86) if j == 0 else (.80, .975)))
    ax.set_xlabel('Macro ' + MET[j])
    ax.grid(axis='x', color='#e7edf2')
    ax.set_axisbelow(True)
    heading(f, x, .96, 'AB'[j], 'Structural branches: ' + MET[j])

ax = f.add_axes([.21, .425, .30, .18])
im = ax.imshow(fusion_delta, vmin=-.015, vmax=.015, cmap='BrBG', aspect='auto')
ax.set_xticks(range(4), [('Breast' if x == 'Mammary' else x) for x in ORG])
ax.set_yticks([0, 1], ['Supervised', 'PNU'])
ax.tick_params(length=0, pad=8)
for spine in ax.spines.values():
    spine.set_visible(False)
for i in range(2):
    for j in range(4):
        ax.text(j, i, f'{fusion_delta[i,j]:+.4f}', ha='center', va='center', fontsize=11)
ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
ax.grid(which='minor', color='white', linewidth=1)
ax.tick_params(which='minor', length=0)
cb = f.colorbar(im, cax=f.add_axes([.21, .35, .30, .012]),
                orientation='horizontal', ticks=[-.015, 0, .015])
cb.set_label('Fusion minus fingerprint-only AUROC', fontsize=10)
cb.outline.set_visible(False)
heading(f, .21, .685, 'C', 'Organ-level fusion contribution')

diameter = 3.0
ax = f.add_axes([.79-diameter/12.4/2, .39, diameter/12.4, diameter/11.6], projection='polar')
radar(ax, 'fusion')
heading(f, .64, .685, 'D', 'Matched training objectives')
for j, x in enumerate([.21, .64]):
    intervals(f.add_axes([x, .075, .30, .16]), MET[j])
    heading(f, x, .26, 'EF'[j], MET[j] + ': paired 95% intervals')
save(f, 'fig3_component_ablation')

# Figure S4: only chemical-space panels moved from the old structural figure.
f = plt.figure(figsize=(11.8, 8.8))
gs = f.add_gridspec(2, 2, left=.09, right=.94, bottom=.10, top=.94,
                   wspace=.36, hspace=.49)
ax = f.add_subplot(gs[0, 0])
ax.scatter(train.pc1, train.pc2, s=16, color=C['train'], alpha=.7, label='Training parents (435)')
ax.scatter(space.pc1, space.pc2, s=18, facecolors='none', edgecolors=C['sup'],
           linewidth=.7, label='Test records (454)')
ax.set_xlabel('PC1 (6.3% variance)'); ax.set_ylabel('PC2 (4.3% variance)')
ax.set_title('A  Training and test chemical space', loc='left', fontweight='bold', pad=14)
ax.legend(fontsize=9)
limits = (ax.get_xlim(), ax.get_ylim())
ax = f.add_subplot(gs[0, 1])
v = space.mean_label_aligned_score_change.to_numpy()
limit = np.max(np.abs(v))
im = ax.scatter(space.pc1, space.pc2, s=23, c=v, cmap='BrBG',
                vmin=-limit, vmax=limit, alpha=.85)
ax.set_xlim(limits[0]); ax.set_ylim(limits[1])
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.set_title('B  PNU-related prediction changes', loc='left', fontweight='bold', pad=14)
f.colorbar(im, ax=ax, shrink=.8, pad=.025, label='Mean label-aligned score change')
ax = f.add_subplot(gs[1, :])
ax.hist(space.nearest_Tanimoto, bins=np.linspace(0, 1, 21),
        color=C['sup'], edgecolor='white', alpha=.85)
ax.set_xlabel('Maximum Tanimoto similarity to the fingerprint-training pool')
ax.set_ylabel('Test records')
ax.set_title('C  Nearest-neighbor structural coverage', loc='left', fontweight='bold', pad=14)
ax.grid(axis='y', color='#e7edf2'); ax.set_axisbelow(True)
save(f, 'figS4_chemical_space')

# Figure S5: fingerprint-only loss profiles and aggregate prediction behavior.
f = plt.figure(figsize=(11.8, 10.4))
diameter = 3.7
radar(f.add_axes([.5-diameter/11.8/2, .55, diameter/11.8, diameter/10.4],
                projection='polar'), 'fingerprint')
heading(f, .12, .96, 'A', 'Fingerprint-only training objectives')
ax = f.add_axes([.12, .085, .34, .24])
left = np.zeros(4)
for key, label, color in [
    ('correct_both', 'Correct in both', '#a4c8bb'),
    ('corrected', 'Corrected', C['pnu']),
    ('regressed', 'Regressed', C['bad']),
    ('wrong_both', 'Wrong in both', '#d5dce2'),
]:
    vals = trans[key].to_numpy() / trans.n.to_numpy()
    ax.barh(range(4), vals, left=left, height=.66, color=color, label=label)
    for i, value in enumerate(vals):
        if value > .04:
            ax.text(left[i]+value/2, i, str(trans.iloc[i][key]), ha='center', va='center', fontsize=9.5)
    left += vals
ax.set_yticks(range(4), [f"{'Breast' if r.organ == 'Mammary' else r.organ} (n={r.n})" for r in trans.itertuples()])
ax.invert_yaxis(); ax.set_xlim(0, 1)
ax.set_xlabel('Fraction of evaluable organ records')
ax.legend(loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, frameon=False, fontsize=9)
heading(f, .12, .36, 'B', 'Classification transitions at 0.45')
ax = f.add_axes([.62, .085, .33, .24])
v = ax.violinplot([changes.loc[changes.organ.eq(o), 'signed_score_change'] for o in ORG],
                  showmeans=True, showextrema=False)
for body in v['bodies']:
    body.set_facecolor(C['pnu']); body.set_alpha(.45)
ax.axhline(0, linestyle=':', color=C['graph'])
ax.set_xticks(range(1, 5), [('Breast' if x == 'Mammary' else x) for x in ORG])
ax.set_ylabel('Label-aligned score change')
heading(f, .62, .36, 'C', 'Continuous prediction changes')
save(f, 'figS5_loss_behavior')

manifest = {
    'operation': 'Recomposition of existing figures; no model fitting',
    'main_panel_mapping': {
        'A': 'old Figure 3A', 'B': 'old Figure 3B', 'C': 'old Figure 3F',
        'D': 'old Figure 5B', 'E': 'old Figure 5C', 'F': 'old Figure 5D',
    },
    'supplementary_mapping': {
        'S3': 'old Figure 4, unchanged organ curves',
        'S4A-C': 'old Figure 3C-E',
        'S5A-C': 'old Figure 5A/E/F',
    },
    'checks': {'representation_matches_archived_metrics': True,
               'paired_deltas_match_macro_differences': True},
    'source_sha256': {s: hashlib.sha256((DATA/s).read_bytes()).hexdigest() for s in inputs},
}
(OUT/'merged_ablation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print('All plotted branch metrics and paired differences match archived tables.')
