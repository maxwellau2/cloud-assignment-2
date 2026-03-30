#!/usr/bin/env python3
"""Generate plots for the PageRank report."""

import csv
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_convergence(csv_path, output_path):
    """Plot L1 diff vs iteration number (log scale)."""
    iterations, diffs = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            iterations.append(int(row['iteration']))
            diffs.append(float(row['l1_diff']))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(iterations, diffs, 'b-', linewidth=1.5, label='$L_1$ difference')
    ax.axhline(y=1e-10, color='r', linestyle='--', linewidth=1, label='$\\epsilon = 10^{-10}$')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('$\\|\\mathbf{r}^{(t+1)} - \\mathbf{r}^{(t)}\\|_1$', fontsize=12)
    ax.set_title('Convergence of the Power Method ($p = 0.15$)', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.annotate(f'Converged at iteration {iterations[-1]}',
                xy=(iterations[-1], diffs[-1]),
                xytext=(iterations[-1] * 0.5, diffs[-1] * 100),
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {output_path}')


def plot_vary_p(csv_path, output_path):
    """Plot PageRank scores of top nodes vs teleport probability p."""
    # Parse CSV: p,node_id,rank,score
    data = {}  # node_id -> {p: score}
    all_p = set()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = float(row['p'])
            node_id = int(row['node_id'])
            score = float(row['score'])
            rank = int(row['rank'])
            all_p.add(p)
            if rank <= 5:
                if node_id not in data:
                    data[node_id] = {}
                data[node_id][p] = score

    p_values = sorted(all_p)

    # Find the top-5 nodes at p=0.15 (or smallest p) to track consistently
    ref_p = 0.15 if 0.15 in all_p else min(all_p)
    top_nodes = []
    for nid, scores in data.items():
        if ref_p in scores:
            top_nodes.append((nid, scores[ref_p]))
    top_nodes.sort(key=lambda x: -x[1])
    top_nodes = [n[0] for n in top_nodes[:5]]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    markers = ['o', 's', '^', 'D', 'v']
    for i, nid in enumerate(top_nodes):
        scores = data[nid]
        ps = sorted(scores.keys())
        vals = [scores[p] for p in ps]
        ax.plot(ps, vals, marker=markers[i % len(markers)], linewidth=1.5,
                markersize=5, label=f'Node {nid}')

    # Uniform baseline
    N = 10000  # approximate
    ax.axhline(y=1.0/N, color='gray', linestyle=':', linewidth=1, label=f'Uniform ($1/N$)')

    ax.set_xlabel('Teleport probability $p$', fontsize=12)
    ax.set_ylabel('PageRank score', fontsize=12)
    ax.set_title('Effect of Teleport Probability on Top-5 PageRank Scores', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {output_path}')


def plot_addt(csv_path, output_path):
    """Plot stacked horizontal bar chart of ADDT score breakdown."""
    names, authority, demand, diversity, trust = [], [], [], [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Shorten URL for display
            name = row['name'].replace('https://', '').replace('http://', '')
            names.append(name)
            authority.append(float(row['authority']))
            demand.append(float(row['demand']))
            diversity.append(float(row['diversity']))
            trust.append(float(row['trust']))

    names.reverse()
    authority.reverse()
    demand.reverse()
    diversity.reverse()
    trust.reverse()

    fig, ax = plt.subplots(figsize=(8, 4))
    y = range(len(names))

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#9C27B0']
    bars1 = ax.barh(y, authority, color=colors[0], label='Authority (PageRank)')
    left = authority[:]
    bars2 = ax.barh(y, demand, left=left, color=colors[1], label='Demand (In-degree)')
    left = [a + d for a, d in zip(left, demand)]
    bars3 = ax.barh(y, diversity, left=left, color=colors[2], label='Diversity (Out-degree)')
    left = [l + d for l, d in zip(left, diversity)]
    bars4 = ax.barh(y, trust, left=left, color=colors[3], label='Domain Trust')

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('ADDT Score', fontsize=12)
    ax.set_title('ADDT Score Breakdown for Top Crawl Targets', fontsize=13)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3, axis='x')
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Generate plots for PageRank report')
    parser.add_argument('--convergence-csv', required=True)
    parser.add_argument('--vary-p-csv', required=True)
    parser.add_argument('--addt-csv', required=True)
    parser.add_argument('--output-dir', default='.')
    args = parser.parse_args()

    import os
    out = args.output_dir
    os.makedirs(out, exist_ok=True)

    plot_convergence(args.convergence_csv, os.path.join(out, 'convergence.pdf'))
    plot_vary_p(args.vary_p_csv, os.path.join(out, 'pagerank_vs_p.pdf'))
    plot_addt(args.addt_csv, os.path.join(out, 'addt_breakdown.pdf'))


if __name__ == '__main__':
    main()
