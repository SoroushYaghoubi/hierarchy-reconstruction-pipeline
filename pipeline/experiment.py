import os
import subprocess
import numpy as np
import plotly.graph_objects as go
from tree_io import read_closure
from closure_comp import compare_closures

ARTEFACTS = 'pipeline/artefacts'
RESULTS = 'pipeline/results'
RATES = [r / 100 for r in range(0, 31, 2)]
SEED = 61
os.makedirs(RESULTS, exist_ok=True)


def run(
    name,
    depth, branch_fact, stop_prob, chain_prob,
    corr_type, rate,
    dim, lr, epochs, burnin, negs, batchsize,
    seed=42,
):
    corrupted_stem = f'{name}_{corr_type}'

    subprocess.run(['python', 'pipeline/tree_gen.py',
        '-name', name,
        '-depth', str(depth),
        '-branch_fact', str(branch_fact),
        '-stop_prob', str(stop_prob),
        '-chain_prob', str(chain_prob),
        '-seed', str(seed),
    ], check=True, capture_output=True)

    subprocess.run(['python', 'pipeline/tree_corrupt.py', name,
        '-type', corr_type,
        '-rate', str(rate),
        '-seed', str(seed),
    ], check=True, capture_output=True)

    subprocess.run(['python', 'embed.py',
        '-dim', str(dim),
        '-lr', str(lr),
        '-epochs', str(epochs),
        '-negs', str(negs),
        '-burnin', str(burnin),
        '-ndproc', '1',
        '-model', 'distance',
        '-manifold', 'poincare',
        '-dset', f'{ARTEFACTS}/{corrupted_stem}.csv',
        '-checkpoint', f'{ARTEFACTS}/{corrupted_stem}.pth',
        '-batchsize', str(batchsize),
        '-eval_each', '1',
        '-fresh',
        '-sparse',
        '-train_threads', '1',
        '-gpu', '-1',
    ], check=True, capture_output=True)

    for method in ['poincare', 'angular']:
        subprocess.run(['python', 'pipeline/tree_rec.py', corrupted_stem,
            '-method', method,
        ], check=True, capture_output=True)

    original  = read_closure(name)
    corrupted = read_closure(corrupted_stem)

    results = {}
    results['corrupted'] = compare_closures(original, corrupted)
    for method in ['poincare', 'angular']:
        stem = f'{corrupted_stem}_recovered_{method}'
        results[method] = compare_closures(original, read_closure(stem))

    for method in ['corrupted', 'poincare', 'angular']:
        print(f"  {method}: precision={results[method]['precision']:.3f} recall={results[method]['recall']:.3f} f1={results[method]['f1']:.3f}")

    return results


def collect_best_annotations(x_vals, y_vals, z_dict):
    lines = []
    baseline = np.array(z_dict['corrupted baseline'])
    for label, Z in z_dict.items():
        if label == 'corrupted baseline':
            continue
        Z = np.array(Z)
        diff = Z - baseline
        idx = np.unravel_index(np.argmax(diff), diff.shape)
        best_x = x_vals[idx[1]]
        best_y = y_vals[idx[0]]
        best_diff = diff[idx]
        lines.append(f'<b>{label}</b>: max gain={best_diff:+.3f} at rate={best_x}%, param={best_y}')
    return '<br>'.join(lines)

def plot_3d(x_vals, y_vals, z_dict, xlabel, ylabel, zlabel, title, filename):
    colors = {'corrupted baseline': 'blues', 'poincare': 'oranges', 'angular': 'greens'}
    fig = go.Figure()

    for label, Z in z_dict.items():
        fig.add_trace(go.Surface(
            x=x_vals, y=y_vals, z=np.array(Z),
            name=label,
            colorscale=colors[label],
            opacity=0.7,
            showscale=False,
        ))

    annotations = collect_best_annotations(x_vals, y_vals, z_dict)

    fig.update_layout(
        title=dict(
            text=f'{title}<br><sup>{annotations}</sup>',
            x=0.5,
            xanchor='center',
        ),
        scene=dict(
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            zaxis_title=zlabel,
        ),
        legend=dict(x=0, y=1),
    )

    path = f'{RESULTS}/{filename}'
    fig.write_html(path)
    print(f"Saved to {path}")

##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### 
##### #####
##### #####             EXPERIMENTS
##### #####
##### ##### ##### ##### ##### ##### 
##### ##### ##### ##### ##### ##### 

def experiment_chain_prob(corr_type, chain_probs=None, rates=None):
    if rates is None:
        rates = RATES
    if chain_probs is None:
        chain_probs = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    all_data = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

    for chain_prob in chain_probs:
        seed = SEED
        tree_tag = f'chain{round(chain_prob*100)}'
        row = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

        for rate in rates:
            name = f'mytree_{tree_tag}_r{round(rate*100)}_s{seed}'
            print(f"\n=== {corr_type} rate={rate:.2f} chain={chain_prob:.2f} seed={seed} ===")
            results = run(
                name=name,
                depth=6, branch_fact=2, stop_prob=0.0, chain_prob=chain_prob,
                corr_type=corr_type, rate=rate,
                dim=5, lr=0.3, epochs=50, burnin=10, negs=50, batchsize=10,
                seed=seed,
            )
            for metric in ['f1', 'precision', 'recall']:
                for method in ['corrupted', 'poincare', 'angular']:
                    row[metric][method].append(results[method][metric])

        for metric in ['f1', 'precision', 'recall']:
            for method in ['corrupted', 'poincare', 'angular']:
                all_data[metric][method].append(row[metric][method])

    x_vals = [round(r * 100) for r in rates]
    y_vals = [round(c * 100) for c in chain_probs]

    for metric in ['f1', 'precision', 'recall']:
        plot_3d(
            x_vals=x_vals,
            y_vals=y_vals,
            z_dict={
                'corrupted baseline': all_data[metric]['corrupted'],
                'poincare':           all_data[metric]['poincare'],
                'angular':            all_data[metric]['angular'],
            },
            xlabel='corruption rate (%)',
            ylabel='chain prob (%)',
            zlabel=metric,
            title=f'{metric} vs corruption rate and chain prob ({corr_type})',
            filename=f'chain_{corr_type}_{metric}.html',
        )


def experiment_depth(corr_type, depths=None, rates=None):
    if rates is None:
        rates = RATES
    if depths is None:
        depths = list(range(3, 10))

    all_data = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

    for depth in depths:
        seed = SEED
        tree_tag = f'depth{depth}'
        row = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

        for rate in rates:
            name = f'mytree_{tree_tag}_r{round(rate*100)}_s{seed}'
            print(f"\n=== depth={depth} vs corruption type={corr_type} rate={rate:.2f} seed={seed} ===")
            results = run(
                name=name,
                depth=depth, branch_fact=2, stop_prob=0.0, chain_prob=0.0,
                corr_type=corr_type, rate=rate,
                dim=5, lr=0.3, epochs=50, burnin=10, negs=50, batchsize=10,
                seed=seed,
            )
            for metric in ['f1', 'precision', 'recall']:
                for method in ['corrupted', 'poincare', 'angular']:
                    row[metric][method].append(results[method][metric])

        for metric in ['f1', 'precision', 'recall']:
            for method in ['corrupted', 'poincare', 'angular']:
                all_data[metric][method].append(row[metric][method])

    x_vals = [round(r * 100) for r in rates]
    y_vals = depths

    for metric in ['f1', 'precision', 'recall']:
        plot_3d(
            x_vals=x_vals,
            y_vals=y_vals,
            z_dict={
                'corrupted baseline': all_data[metric]['corrupted'],
                'poincare':           all_data[metric]['poincare'],
                'angular':            all_data[metric]['angular'],
            },
            xlabel='corruption rate (%)',
            ylabel='depth',
            zlabel=metric,
            title=f'{metric} vs corruption rate and depth ({corr_type})',
            filename=f'depth_{corr_type}_{metric}.html',
        )


def experiment_branch_fact(corr_type, branch_facts=None, rates=None):
    if rates is None:
        rates = RATES
    if branch_facts is None:
        branch_facts = list(range(2, 7))

    all_data = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

    for branch_fact in branch_facts:
        seed = SEED
        tree_tag = f'branch{branch_fact}'
        row = {metric: {'corrupted': [], 'poincare': [], 'angular': []} for metric in ['f1', 'precision', 'recall']}

        for rate in rates:
            name = f'mytree_{tree_tag}_r{round(rate*100)}_s{seed}'
            print(f"\n=== branch_fact={branch_fact} vs corruption type={corr_type} rate={rate:.2f} seed={seed} ===")
            results = run(
                name=name,
                depth=4, branch_fact=branch_fact, stop_prob=0.0, chain_prob=0.0,
                corr_type=corr_type, rate=rate,
                dim=5, lr=0.3, epochs=50, burnin=10, negs=50, batchsize=10,
                seed=seed,
            )
            for metric in ['f1', 'precision', 'recall']:
                for method in ['corrupted', 'poincare', 'angular']:
                    row[metric][method].append(results[method][metric])

        for metric in ['f1', 'precision', 'recall']:
            for method in ['corrupted', 'poincare', 'angular']:
                all_data[metric][method].append(row[metric][method])

    x_vals = [round(r * 100) for r in rates]
    y_vals = branch_facts

    for metric in ['f1', 'precision', 'recall']:
        plot_3d(
            x_vals=x_vals,
            y_vals=y_vals,
            z_dict={
                'corrupted baseline': all_data[metric]['corrupted'],
                'poincare':           all_data[metric]['poincare'],
                'angular':            all_data[metric]['angular'],
            },
            xlabel='corruption rate (%)',
            ylabel='branch factor',
            zlabel=metric,
            title=f'{metric} vs corruption rate and branch factor ({corr_type})',
            filename=f'branch_{corr_type}_{metric}.html',
        )


if __name__ == '__main__':
    for corr_type in ['missing', 'shuffled', 'false']:
        experiment_chain_prob(corr_type)    
        experiment_depth(corr_type) 
        experiment_branch_fact(corr_type)   