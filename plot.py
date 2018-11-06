import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as colors
import matplotlib.cm as cmx


from sum_up_results import create_benchmark_dataframe


def plot_results(df,
                 title=None,
                 x_label=None,
                 y_label=None,
                 hue_label=None,
                 row_label=None,
                 col_label=None,
                 fig_label=None,
                 save=False,
                 path=None):
    if save and (path is None):
        raise ValueError("must provide a valid path to save the figures")

    if fig_label is not None:
        # loop over all fig_label index values and create a figure each time
        for n, g in df.groupby(fig_label):
            _title = "{} ({}: {})".format(title, fig_label, n)
            if path is not None:
                filename, ext = os.path.splitext(path)
                _path = "{}_{}{}".format(filename, n, ext)
            else:
                _path = None
            plot_results(g, _title, x_label, y_label, hue_label,
                         row_label, col_label, fig_label=None,
                         save=save, path=_path)

    df = df.to_frame("time").reset_index()

    # replace empty string with default pickler (used to be pickle, not anymore
    # but I don't use empty string ever since, this is simply for back
    # compatibility)
    df = df.replace("", "pickle")

    df = df[[x_label, y_label, row_label, col_label, hue_label]]
    df = df.sort_values(by=x_label)
    n_values = df.nunique()

    use_errorbars = False

    # create a subplot grid with the correct shape
    f, axs = plt.subplots(
        nrows=n_values[row_label],
        ncols=n_values[col_label],
        squeeze=False,
        figsize=(12, 8),
        sharex=False,
        sharey='row')
    f.suptitle(title, x=0.5, y=0.92)

    # create appropriate cmap objects to loop over colors of a colormap
    tab_cm = plt.get_cmap('tab10')
    c_norm = colors.Normalize(vmin=0, vmax=n_values[hue_label])
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=tab_cm)

    labels = np.sort(df[hue_label].unique())
    hue_colors = {l: scalar_map.to_rgba(i) for i, l in enumerate(labels)}

    # create the legend
    f.legend(
        handles=[Patch(color=c, label=l) for l, c in hue_colors.items()],
        loc='upper center',
        ncol=2)

    # loop over all groups
    by_rowsandcol = df.groupby([row_label, col_label])
    targets = zip(by_rowsandcol, axs.flatten())
    for group, ax in targets:
        n, g = group

        infos = zip((row_label, col_label), n)
        text = '\n'.join(['{}: {}'.format(*vals) for vals in infos])
        ax.text(0.1, 0.7, text, transform=ax.transAxes, bbox=dict(alpha=0.2))
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # re compute n_values for the specific group
        n_values = g.nunique()
        x_ticks = np.arange(n_values[x_label])
        ax.set_xticklabels(g[x_label].unique())
        ax.set_xticks(x_ticks)
        width = 0.2
        hue_no = 0
        for hue, subgroup in g.groupby(hue_label):
            by_xlabel = subgroup.groupby(x_label)

            if use_errorbars:
                ax.fill_between(
                    subgroup[x_label].unique(),
                    by_xlabel[y_label].min(),
                    by_xlabel[y_label].max(),
                    color=hue_colors[hue],
                    alpha=0.3)
            else:
                bars = by_xlabel.mean()
                ax.bar(
                    x_ticks + hue_no * width,
                    bars[y_label],
                    width,
                    alpha=0.5,
                    color=hue_colors[hue])
                # show the points for each bar
                xtick = 0
                for n, g in by_xlabel:
                    ax.scatter(
                        [xtick + hue_no * width] * len(g),
                        g[y_label],
                        c='black',
                        s=20,
                        marker='_',
                        alpha=0.5)
                    xtick += 1
            hue_no += 1

    if save:
        f.savefig(path)
        plt.close(f)
    else:
        return f


if __name__ == "__main__":
    all_dfs = create_benchmark_dataframe(group_by='class')
    labels = {
        'MakeRegressionDataBench':
        dict(
            y_label='time',
            x_label='n_jobs',
            hue_label='pickler',
            col_label='backend',
            row_label='name'),
        'CaliforniaHousingBench':
        dict(
            y_label='time',
            x_label='n_jobs',
            hue_label='pickler',
            col_label='backend',
            row_label='name'),
        'TwentyDataBench':
        dict(
            y_label='time',
            x_label='n_jobs',
            hue_label='pickler',
            col_label='backend',
            row_label='name'),
        'RegressionBench':
        dict(
            y_label='time',
            x_label='n_jobs',
            hue_label='backend',
            col_label='n_samples',
            row_label='name',
            fig_label='estimator_name'),
    }

    all_dfs['MakeRegressionDataBench'] = all_dfs['MakeRegressionDataBench'].xs(
        '10000', axis=0, level='n_samples', drop_level=False)

    # subset only few combinations of pickler+backend
    all_dfs['RegressionBench'] = pd.concat(
        [all_dfs['RegressionBench'].xs(
             ("cloudpickle", "loky"), level=["pickler", "backend"],
             drop_level=False),
         all_dfs["RegressionBench"].xs(
             ("pickle", "threading"), level=['pickler', 'backend'],
             drop_level=False)],
        axis=0)

    for benchmark_name, benchmark_df in all_dfs.items():
        path = 'plots/{}.png'.format(benchmark_name)
        f = plot_results(
            benchmark_df, title=benchmark_name, **labels[benchmark_name],
            path=path, save=True)
