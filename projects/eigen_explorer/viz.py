import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_convergence(results_csv: str, outpath: str):
    df = pd.read_csv(results_csv)
    fig = plt.figure()
    plt.semilogy(df['k'], df['residual_pm'])
    plt.xlabel('Iteration k')
    plt.ylabel('Residual ||A x_k - lambda_k x_k||')
    plt.title('Power Method Convergence')
    plt.grid(True, which='both')
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)

def plot_invariant_axes(A, eigs_csv: str, outpath: str):
    # 2D-only geometry for this plot
    A = np.array(A, dtype=float)
    assert A.shape == (2,2), "Invariant-axes plot implemented for 2D matrices."
    eigs = pd.read_csv(eigs_csv)
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 400)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    ellipse = A @ circle
    # Eigenvectors from eigs.csv not carrying vectors; recompute for arrows
    vals, vecs = np.linalg.eig(A)
    v1 = vecs[:, np.argmax(np.abs(vals))].real
    v2 = vecs[:, np.argmin(np.abs(vals))].real
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    fig = plt.figure()
    plt.plot(circle[0], circle[1], label='Unit circle')
    plt.plot(ellipse[0], ellipse[1], label='A·(unit circle)')
    # Eigenvector rays
    for v, lab in [(v1, 'eigvec 1'), (v2, 'eigvec 2')]:
        plt.plot([0, v[0]], [0, v[1]], linewidth=2, label=lab)
        plt.plot([0, -v[0]], [0, -v[1]], linewidth=2)
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.title('Invariant Axes and Linear Image of Unit Circle')
    plt.grid(True)
    fig.savefig(outpath, dpi=160, bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=str, default='results.csv')
    parser.add_argument('--eigs', type=str, default='eigs.csv')
    parser.add_argument('--summary', type=str, default='summary.json')
    parser.add_argument('--out_convergence', type=str, default='figs/convergence.png')
    parser.add_argument('--out_invariant', type=str, default='figs/invariant_axes.png')
    args = parser.parse_args()
    with open('config.yaml','r') as f:
        import yaml
        A = yaml.safe_load(f)['matrix']['A']
    plot_convergence(args.results, args.out_convergence)
    plot_invariant_axes(A, args.eigs, args.out_invariant)
    print("Saved figures:", args.out_convergence, args.out_invariant)

if __name__ == '__main__':
    main()