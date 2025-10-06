# viz_helpers.py
import matplotlib.pyplot as plt
import numpy as np


def plot_analysis(result, title: str = "Function Explorer"):
    X, Y, segs = result.samples["X"], result.samples["Y"], result.samples["segments"]
    plt.figure()
    ax = plt.gca()
    for s in segs:
        xs, ys = X[s], Y[s]
        ax.plot(xs, ys, label="f(x)")
    for s in result.singularities:
        ax.axvline(s, linestyle="--", alpha=0.6, label="singularity")
    for z in result.x_intercepts:
        ax.scatter([z], [0], marker="o", label="x-intercept")
    if result.y_intercept is not None:
        ax.scatter([0], [result.y_intercept], marker="D", label="y-intercept")
    ax.set_xlabel("x (unitless)")
    ax.set_ylabel("f(x) (unitless)")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys())
    plt.show()


def plot_value_hist(result, bins: int = 60):
    Y = result.samples["Y"]
    Y = Y[np.isfinite(Y)]
    plt.figure()
    ax = plt.gca()
    ax.hist(Y, bins=bins)
    ax.set_xlabel("f(x) value")
    ax.set_ylabel("frequency")
    ax.set_title("Distribution of sampled f(x)")
    plt.show()
