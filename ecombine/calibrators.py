"""
calibrators (and adjusters) of evidence processes that allow "lifting"
"""

import numpy as np


def e_to_p_calibrator(e: np.ndarray):
    """Calibrate an e-process into a p-process via f(e) = min(1/e, 1).

    This is the unique admissible e-to-p (deterministic) calibrator.
    """
    return np.minimum(np.divide(1, e), 1)


def p_to_e_calibrator(p: np.ndarray, kappa: float = None):
    """Calibrate a p-process into an e-process.

    Different admissible calibrators exist. The default option is given by

        f(p) = (1 - p + p * log(p)) / (p * log(1/p) ** 2).

    When kappa is specified (between 0 and 1), the function is

        f(p; kappa) = kappa * p ** (kappa - 1).
    """
    if kappa is not None and not (0 < kappa < 1):
        raise ValueError("invalid value for kappa: must be between 0 and 1 if specified")

    # handle zeros
    p = np.array(p)
    p = np.where(p > 0, p, 1e-16)
    if kappa:
        return kappa * p ** (kappa - 1)
    else:
        return (1 - p + p * np.log(p)) / (p * np.log(p) ** 2 + 1e-8)


def adjuster(e: np.ndarray, use_maximum: bool = True, kappa: float = None):
    """Adjust the running maximum of an e-process into an e-process.

    Equivalent to:

        F(e_t*) = f(min(1, 1/e_t*))

    where e* is the running maximum up to t and f is any p-to-e calibrator.
    """
    if use_maximum:
        e = np.maximum.accumulate(e)

    return p_to_e_calibrator(np.minimum(1, np.divide(1, e)), kappa=kappa)
