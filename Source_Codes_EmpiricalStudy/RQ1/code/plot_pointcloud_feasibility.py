"""Plot the audited development measurements; never synthesizes point clouds."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
ORDER = ['stereobm', 'sgm', 'psmnet', 'gwcnet', 'raft_stereo', 'igev_stereo', 'foundationstereo']
NAMES = ['StereoBM', 'SGM', 'PSMNet', 'GwcNet', 'RAFT-Stereo', 'IGEV-Stereo', 'FoundationStereo']


def plot(label):
    source = STUDY / 'RQ1/data' / label / 'per_scene.csv'
    result = STUDY / 'RQ1/results' / label
    rows = list(csv.DictReader(source.open()))
    assert len(rows) == 504 and all(r['split'] == 'development' for r in rows)
    groups = [[r for r in rows if r['method'] == method] for method in ORDER]
    assert all(len(g) == 72 for g in groups)
    fig = plt.figure(figsize=(14, 9), layout='constrained')
    grid = fig.add_gridspec(2, 2)
    for ax, field, title, ylabel in [
        (fig.add_subplot(grid[0, 0]), 'target_points', '(a) Number of target-ROI points', 'Points (log scale)'),
        (fig.add_subplot(grid[0, 1]), 'surface_mae_mm', '(b) Predicted points to target mesh', 'Per-scene mean distance, mm (log scale)')]:
        ax.boxplot([[float(r[field]) for r in g] for g in groups], showfliers=True,
                   medianprops={'color': '#d55e00', 'linewidth': 2},
                   flierprops={'markersize': 3, 'markeredgecolor': '#666666'})
        ax.set_xticks(range(1, 8), NAMES, rotation=25, ha='right')
        ax.set_yscale('log')
        ax.set_title(title, loc='left')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', alpha=.2)
    ax = fig.add_subplot(grid[1, :])
    x = np.arange(7)
    for offset, field, title, color in [
        (-.25, 'target_source_pixel_fraction', 'Points originating from target pixels', '#0072b2'),
        (0, 'precision_2mm', 'Points within 2 mm of target surface', '#e69f00'),
        (.25, 'coverage_2mm', 'Left/right visible-surface coverage at 2 mm', '#009e73')]:
        values = [100 * np.mean([float(r[field]) for r in g]) for g in groups]
        bars = ax.bar(x + offset, values, width=.24, label=title, color=color)
        ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=3)
    ax.set_xticks(x, NAMES)
    ax.set_ylim(0, 112)
    ax.set_ylabel('Scene-equal mean (%)')
    ax.set_title('(c) Point count, geometric accuracy and coverage measure different properties', loc='left')
    ax.legend(loc='upper left', bbox_to_anchor=(0, -.13), ncol=1, frameon=False)
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.suptitle('Saved development stereo point clouds: 72 scenes per method; 504 clouds', fontsize=15)
    for suffix in ['png', 'svg']:
        fig.savefig(result / ('pointcloud_comparison.' + suffix), dpi=180)
    plt.close(fig)
    record = dict(source=str(source.relative_to(STUDY)),
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  plot_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  measured_data_only=True, scene_count=72, method_count=7,
                  error_bars='Boxplots: median, quartiles, 1.5 IQR whiskers and observed outliers',
                  source_pixel_membership='Evaluation only; GT not supplied to inference')
    (result / 'figure_provenance.json').write_text(json.dumps(record, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='development_pointclouds_20261012_v2')
    plot(parser.parse_args().label)
