"""
plotting functions
"""


from typing import Union, List, Tuple, Dict
import os.path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ecombine.eprocesses import (
    eprocess_exch_universal,
    eprocess_exch_conformal,
)


# globals
PLOT_DEFAULT_ARGS = dict(
    alpha=0.7,
    linewidth=2,
    height=4,
    aspect=2,
)
PLOT_DEFAULT_COLORS = sns.color_palette("colorblind")


def set_theme(style="whitegrid", palette="colorblind", font="Avenir", font_scale=1.5):
    """Set the default plotting themes for the package.

    A wrapper around `seaborn.set_theme()` with custom defaults.
    """
    return sns.set_theme(
        style=style,
        palette=palette,
        font=font,
        font_scale=font_scale,
    )


def plot_eprocess(
        e: Union[np.ndarray, List[np.ndarray]],
        e_label: Union[str, List[str]] = None,
        time_index: np.ndarray = None,
        threshold: float = 10,
        title: str = "E-process",
) -> sns.FacetGrid:
    """Plot e-processes and return the FacetGrid object for further modifications."""

    # each row is an e-process
    e = np.array(e)
    if len(e.shape) == 1:
        e = e[np.newaxis, :]
        e_label = [e_label]
    T = e.shape[1]
    df = pd.DataFrame({"Time": time_index if time_index else range(1, T + 1)})
    for e_proc, e_lab in zip(e, e_label):
        df[e_lab] = e_proc

    # plot e-processes as lines
    df = pd.melt(df, id_vars=["Time"], var_name="E-process", value_name="Value")
    fg = sns.relplot(
        x="Time",
        y="Value",
        hue="E-process",
        kind="line",
        data=df,
        **PLOT_DEFAULT_ARGS
    )
    fg.ax.set(
        title=title,
        ylabel="E-process (log-scale)",
        yscale="log",
    )
    if threshold > 0:
        fg.ax.axhline(
            y=threshold, color="gray", alpha=PLOT_DEFAULT_ARGS["alpha"],
            linestyle="dashed", linewidth=PLOT_DEFAULT_ARGS["linewidth"],
        )
    return fg


def plot_eprocesses_exch(
    x: np.ndarray,
    jumps: Tuple[float] = (0.1, 0.01, 0.001),
    jumper_weights: Tuple[float] = (1/3, 1/3, 1/3),
    title: str = "E-processes for testing exchangeability",
    rng: np.random.Generator = None,  # for conformal
):
    """Plot various e-processes for testing exchangeability."""
    e_ui = eprocess_exch_universal(x)
    e_confs = [
        eprocess_exch_conformal(
            x,
            jump=jump,
            jumper_weights=jumper_weights,
            rng=rng,
        )
        for jump in jumps
    ]
    eprocesses = [e_ui] + e_confs
    names = ["UI"] + [f"Conformal-j{jump:g}" for jump in jumps]
    fg = plot_eprocess(eprocesses, names, title=title)
    return fg, eprocesses


def plot_stopped_e_values(
        dfs: Dict[str, pd.DataFrame],
        estop_dfs: Dict[str, pd.DataFrame],
        hue_order: List[str],
        xlim_hist: Tuple = (0, 4),
        n_samples: int = 100,
        no_title: bool = False,
        plots_dir: str = "./plots",
) -> Tuple[sns.FacetGrid, sns.FacetGrid]:
    """Plot a histogram of stopped e-values.

    Input is a pair of list of data frames obtained by calling
    `ecombine.diagnostics.compute_e_and_stopping()`.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # define a consistent palette across plots
    palette = [PLOT_DEFAULT_COLORS[hue_order.index(method)] for method in dfs]

    # 1. Histogram
    combined_edf = pd.concat(
        [estop_df for estop_df in estop_dfs.values()],
        ignore_index=True,
    )
    fg_hist = sns.displot(
        x="e_stopped",
        kind="hist",
        hue="E-process",
        palette=palette,
        binwidth=0.25,
        data=combined_edf,
        aspect=1.25,
        facet_kws=dict(legend_out=False)
    )
    for i, (method, estop_df) in enumerate(estop_dfs.items()):
        fg_hist.ax.axvline(
            x=estop_df.e_stopped.mean(),
            c=palette[i],
            linestyle="solid",
            linewidth=3,
        )
    fg_hist.ax.axvline(x=1, c="gray", linestyle="dotted", linewidth=2)
    fg_hist.ax.set(
        xlabel="Stopped e-value",
        xlim=xlim_hist,
    )
    if not no_title:
        fg_hist.ax.set_title(
            "Histogram of stopped e-values at " 
            r"$\tau^{\mathbb{F}}$"
        )
    fg_hist.savefig(os.path.join(plots_dir, "stopped_e_histogram"))

    # 2. Sample e-processes and stopping times
    combined_df = pd.concat(
        [df for df in dfs.values()],
        ignore_index=True,
    )
    fg_e = sns.relplot(
        x="Time",
        y="e",
        style="id",
        hue="E-process",
        col="E-process",
        kind="line",
        linewidth=PLOT_DEFAULT_ARGS["linewidth"],
        palette=palette,
        alpha=0.5,
        aspect=1.25,
        legend=False,
        data=combined_df.loc[combined_df.id < n_samples].loc[~combined_df.stopped],
    )
    # highlight stopped values
    for i, (method, estop_df) in enumerate(estop_dfs.items()):
        ax = fg_e.axes[0][i]
        sns.scatterplot(
            x="tau",
            y="e_stopped",
            hue="E-process",
            palette=[palette[i]],
            data=estop_df.loc[estop_df.id < n_samples],
            ax=ax,
            legend=False,
        )
        ax.set_title("" if no_title else f"E-process: {method}")
        ax.axhline(y=1, color="gray", linestyle="dashed")
        ax.set(yscale="log", ylabel="E-process (log-scale)")
    fg_e.savefig(os.path.join(plots_dir, "stopped_e_samples"))

    return fg_hist, fg_e
