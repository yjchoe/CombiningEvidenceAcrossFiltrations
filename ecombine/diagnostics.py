"""
diagnostic functions for e-processes (validity & growth)
"""


from typing import Callable, Tuple
import numpy as np
import pandas as pd
from tqdm import trange


def compute_e_and_stopping(
        data_generator: Callable[[int], np.ndarray],      # accepts T
        eprocess_fn: Callable[[np.ndarray], np.ndarray],  # accepts x
        stopping_fn: Callable[[np.ndarray], int],         # accepts x
        n_repeats: int = 100,
        T: int = 10000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute e-processes across repeated data samples and apply the stopping function for each e-process.

    Results are stored in a "tall" data frame.
    Additionally returns the list of stopping times and the stopped e-values.
    """

    # for mp
    # def _single_run():
    #     x = data_generator(T)
    #     e = eprocess_fn(x)
    #     tau = stopping_fn(x)
    #     stopped = np.concatenate([np.zeros(tau), np.ones(T - tau)]).astype(bool)
    #     return x, e, stopped

    all_x, all_e, all_stopped = [], [], []  # list of lists
    all_tau, all_e_stopped = [], []         # list of numbers
    for _ in trange(n_repeats, desc="compute_e_and_stopping repeated trials"):
        x = data_generator(T)
        e = eprocess_fn(x)
        tau = stopping_fn(x)            # int
        # stopped is a binary vector indicating whether the process is stopped
        stopped = np.concatenate([np.zeros(tau), np.ones(T - tau)])
        # (it is zero at tau for plotting purposes)
        if tau < T:
            stopped[tau] = 0
        e_stopped = e[min(tau, T - 1)]    # float

        all_x.append(x)
        all_e.append(e)
        all_stopped.append(stopped)
        all_tau.append(tau)
        all_e_stopped.append(e_stopped)

    # all_x, all_e, all_stopped = [np.concatenate(all_res) for all_res in zip(*all_results)]
    all_times = np.tile(np.arange(T), n_repeats)    # [0, 1, 2, ..., T, ..., 0, 1, 2, ..., T]
    all_ids = np.repeat(np.arange(n_repeats), T)    # [0, 0, 0, 1, 1, 1, ..., n, n, n]

    full_df = pd.DataFrame({
        "Time": all_times,
        "id": all_ids,
        "x": np.concatenate(all_x),
        "e": np.concatenate(all_e),
        "stopped": np.concatenate(all_stopped),
    }).astype({"Time": int, "id": int, "stopped": bool})
    estop_df = pd.DataFrame({
        "id": np.arange(n_repeats),
        "tau": all_tau,
        "e_stopped": all_e_stopped,
    }).astype({"id": int, "tau": int})
    return full_df, estop_df
