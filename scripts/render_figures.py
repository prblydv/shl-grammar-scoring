"""Render publication charts from aggregate-only evidence, without private records."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'docs/assets'
INK, MUTED, BLUE, TEAL, ORANGE = '#17243b', '#68758a', '#5c65dd', '#168c87', '#c47b35'
COLORS = [BLUE, TEAL, ORANGE]


def style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'text.color': INK, 'axes.labelcolor': MUTED,
                         'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.spines.bottom': False,
                         'axes.grid': True, 'grid.color': '#e9edf3', 'grid.linewidth': .7,
                         'axes.axisbelow': True, 'svg.fonttype': 'none',
                         'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff'})


def save(fig, name, title, subtitle, note):
    fig.suptitle(title, x=.08, y=.965, ha='left', fontsize=23, fontweight='bold')
    fig.text(.08, .895, subtitle, color=MUTED, fontsize=10)
    fig.text(.08, .035, note, color=MUTED, fontsize=9)
    fig.subplots_adjust(left=.09, right=.95, top=.79, bottom=.19, wspace=.35)
    for suffix in ('svg', 'png'):
        fig.savefig(DEST / f'{name}.{suffix}', dpi=180, facecolor='white', bbox_inches='tight')
    plt.close(fig)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    style()
    d = json.loads((ROOT / 'reports/chart_data.json').read_text())
    rows = d['score_comparison']
    # Separate panels prevent the public score being mistaken for an OOF estimate.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, key, title, color in zip(axes, ['local', 'public'],
                                     ['Local validation / development', 'Recorded public leaderboard'], [BLUE, TEAL]):
        ys = np.arange(len(rows))
        ax.barh(ys, [r[key] for r in rows], color=[color]*6+[ORANGE], height=.5)
        ax.set_yticks(ys, [r['name'] for r in rows], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 1.34 if key == 'local' else .83)
        ax.set_title(title, loc='left', fontsize=12, pad=15); ax.set_xlabel('RMSE · lower is better')
        for y, r in zip(ys, rows):ax.text(r[key]+.015, y, f'{r[key]:.4f}', va='center', fontsize=9)
        ax.grid(axis='y', visible=False)
    fig.subplots_adjust(wspace=.65)
    save(fig, 'score_comparison', 'Progress, with the evaluation attached',
         'Early baselines: 6-fold OOF. AST: 3-fold grouped OOF. Ensembles: reused meta-development.',
         'Different evaluation populations. Orange marks the experimental hurdle; private performance is unknown.')

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    t=d['hurdle_trajectory']
    for ax, key, title, color in zip(axes, ['development','public'],
                                     ['Meta-development RMSE', 'Recorded public RMSE'], [BLUE,TEAL]):
        ax.plot(t['weights'], t[key], marker='o', markersize=7, linewidth=2.4, color=color)
        for x,y in zip(t['weights'],t[key]):ax.annotate(f'{y:.5f}',(x,y),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
        ax.set_xlabel('Hurdle weight (%)');ax.set_xticks(t['weights']);ax.set_title(title,loc='left',fontsize=12)
        ax.margins(y=.4);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    save(fig,'hurdle_tradeoff','A better public score is not a generalization test',
         '35% had the lower development error; 60% had the lower observed public error.',
         'Weights were explored after public feedback. The zero-case correction may depend on dataset composition.')

    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    for fold,color in zip(d['fold_history'],COLORS):
        h=fold['history'];epochs=[x['epoch'] for x in h]
        axes[0].plot(epochs,[x['train_loss'] for x in h],color=color,lw=2,label=f"Fold {fold['fold']+1}")
        axes[1].plot(epochs,[x['validation_rmse'] for x in h],color=color,lw=2)
        best=next(x for x in h if x['epoch']==fold['best_epoch'])
        axes[1].scatter(best['epoch'],best['validation_rmse'],s=65,color=color,edgecolors='white',zorder=5)
    axes[0].set_title('Training MSE during optimizer updates',loc='left',fontsize=12)
    axes[1].set_title('Validation RMSE · dots mark selected epochs',loc='left',fontsize=12)
    axes[0].legend(frameon=False)
    for ax in axes:ax.set_xlabel('Epoch')
    save(fig,'learning_curves','Fit the training data. Watch the held-out error.',
         'Audited DeBERTa + gated audio MLP + AST run. Best epochs: 9, 11, 16.',
         'Training MSE and evaluation RMSE are different quantities; the two panels intentionally use separate scales.')

    fig,axes=plt.subplots(1,2,figsize=(13,5.8))
    h=d['residual_histogram'];edges=np.array(h['edges'])
    axes[0].bar((edges[:-1]+edges[1:])/2,h['counts'],width=np.diff(edges)*.86,color=BLUE)
    axes[0].axvline(0,color=INK,lw=1);axes[0].set_xlabel('Prediction − true score');axes[0].set_ylabel('Recordings')
    axes[0].set_title('OOF residual distribution',loc='left',fontsize=12)
    cal=d['calibration'];axes[1].plot([0,5],[0,5],color='#bac2d0',linestyle='--',lw=1)
    axes[1].plot([x['prediction_mean'] for x in cal],[x['label_mean'] for x in cal],color=TEAL,marker='o',lw=2)
    axes[1].set_xlabel('Mean predicted score per decile');axes[1].set_ylabel('Mean true score per decile')
    axes[1].set_title('Prediction-bin calibration',loc='left',fontsize=12)
    save(fig,'error_diagnostics','Inspect the errors before expanding the model',
         'Aggregate held-out diagnostics for the audited AST fusion model.',
         'A near-diagonal calibration curve can coexist with large individual errors. No per-record competition data is published.')

    fig,axes=plt.subplots(1,2,figsize=(13,5.8));s=d['score_slices'];x=np.arange(len(s))
    axes[0].bar(x,[r['rmse'] for r in s],color=BLUE,width=.52)
    axes[1].bar(x,[r['bias'] for r in s],color=[TEAL if r['bias']>=0 else ORANGE for r in s],width=.52)
    axes[1].axhline(0,color=INK,lw=1)
    for ax in axes:ax.set_xticks(x,[f"{r['label']}\nn={r['rows']}" for r in s]);ax.set_xlabel('True-score slice')
    axes[0].set_title('Slice RMSE',loc='left',fontsize=12);axes[1].set_title('Mean prediction − true score',loc='left',fontsize=12)
    save(fig,'score_slices','Different score ranges, different failure patterns',
         'Original labels retained. Zero cases are shown separately rather than hidden in the average.',
         'Target-conditioned bias is descriptive; it does not, by itself, prove global miscalibration or justify score expansion.')
    print('Rendered five figures in SVG and PNG from aggregate evidence.')


if __name__ == '__main__':
    main()
