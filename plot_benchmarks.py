import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from asv_to_pandas import create_benchmark_dataframe

from benchmarks.config import N_SAMPLES


params = {
    "axes.titlesize": 24,
    "axes.labelsize": 20,
    "lines.linewidth": 3,
    "lines.markersize": 10,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.size": 16,
}
mpl.rcParams.update(params)

PLOT = True

df = create_benchmark_dataframe(group_by="class")

all_benchmarks = {}
for name, res in df.items():
    if "Mixin" in name:
        # My Mixin classes are not abstracts for now so asv will try to run the
        # benchmarks
        continue

    if "MultiTaskElasticNetCV" in name:
        continue

    bench_regex = re.compile(r"(\w+)(SingleFit|MultipleFit)Bench")
    estimator, bench_type = re.findall(bench_regex, name)[0]
    n_samples = N_SAMPLES[estimator]
    all_benchmarks[(estimator, bench_type, n_samples)] = res
all_benchmarks = pd.concat(
    all_benchmarks,
    names=["estimator", "parallel_level", "effective_n_samples"],
)

# plot the distributions of the benchmarks
if PLOT:
    f, ax = plt.subplots(figsize=(13, 8))
    # plot threading vs loky
    sequential_vs_parallel = True
    N_JOBS = 4
    if sequential_vs_parallel:
        ax.set_xlabel("n_jobs = 1")
        ax.set_ylabel("n_jobs = 4")

        plot_df = all_benchmarks.unstack("n_jobs")
        max_ = plot_df.max().max()
        x = plot_df[1]
        y = plot_df[4]
        shapes = np.log10(
            plot_df.index.get_level_values("effective_n_samples")
        )
        colors = [
            1 if c == "loky" else 0
            for c in plot_df.index.get_level_values("backend")
        ]
        pts = ax.scatter(x, y, s=shapes ** 3, c=colors, marker="o")
        ax.set_xlim(left=0, right=max_)
        ax.set_ylim(bottom=0)

        # plot the identity function, and the best possible
        # speedup in reference
        ax.plot([0, max_], [0, max_], linewidth=1, c="red")
        ax.plot([0, max_], [0, max_ / N_JOBS], linewidth=1, c="green")

        ax.fill_between(
            [0, max_], [0, 0], [0, max_ / N_JOBS], facecolor="grey", alpha=0.2
        )
        # legend
        kw = dict(
            prop="sizes",
            num=[1000, 10000, 100000, 1000000],
            func=lambda x: 10 ** (x ** (1 / 3)),
            color=pts.cmap(0.7),
        )
        legend = ax.legend(
            *pts.legend_elements(**kw),
            loc="lower right",
            title="data length",
            prop={"size": 10}
        )
        pts.set_visible(False)
        legend.set_visible(False)
        f.savefig(
            "plots/benchmark_plots/seq_vs_parallel_bounds_only.png", dpi=f.dpi
        )

        pts.set_visible(True)
        legend.set_visible(True)
        f.savefig(
            "plots/benchmark_plots/seq_vs_parallel_bounds_and_scatter.png",
            dpi=f.dpi,
        )

        # show gil contention
        gil_zone = mpatches.Ellipse(
            (15, 27), 10, 30, angle=-45, alpha=0.2, color="red"
        )
        ax.add_patch(gil_zone)
        ax.text(15, 27, "GIL contention", size=20, rotation=30)

        good_enough_zone = mpatches.Ellipse(
            (25, 8.5), 3, 47, angle=-75, alpha=0.2, color="green"
        )
        ax.add_patch(good_enough_zone)
        ax.text(32, 10, "Good enough?", size=18, rotation=10)
        f.savefig(
            "plots/benchmark_plots/seq_vs_parallel_bounds_and_scatter_and_patches.png",  # noqa
            dpi=f.dpi,
        )
