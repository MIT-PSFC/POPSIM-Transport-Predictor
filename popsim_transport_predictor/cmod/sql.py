"""
Module for SQL queries. Snagged from https://github.com/MIT-PSFC/disruption-efit/blob/main/disruption_efit/sql.py
"""

import numpy as np
from disruption_py.machine.tokamak import Tokamak
from disruption_py.workflow import get_database
from loguru import logger

summary_table = "summary"

ipmax = 100e3  # [A]
pulse_length = 0.1  # [s]


def summary(
    ipmax: float = ipmax,
    pulse_length: float = pulse_length,
    min_shot: int = 1050204013,
    max_shot: int = 1160930043,
    shots: list[int] | bool = False,
) -> np.ndarray:
    """
    Perform a SELECT query on the `summary` table to find shots
    with high enough current and long enough pulse length.
    Optionally select shots from a given list.

    Parameters
    ----------
    ipmax : float, default = config.ipmax
        threshold that maximum plasma current must exceed [A]
    pulse_length : float, default = config.pulse_length
        threshold that pulse length must exceed [s]
    min_shot : int, optional, default = config.min_shot
        disregard shots below this number.
    max_shot : int, optional, default = config.max_shot
        disregard shots above this number.
    shots : list[int] | bool, optional, default = False
        list of shots to be queried.

    Returns
    -------
    np.ndarray
        Nx2 array of [shot_id, pulse_length] for the N shots which exceed the thresholds
    """

    # database
    db = get_database(tokamak=Tokamak.CMOD)

    # query
    query = [
        f"select distinct(shot), pulse_length from {summary_table} ",
        f"where ipmax > {ipmax} and pulse_length > {pulse_length}",
    ]
    if min_shot > 0:
        query += [f"and shot >= {min_shot}"]
    if max_shot > 0:
        query += [f"and shot <= {max_shot}"]
    if hasattr(shots, "__iter__"):
        query += [f"and shot in ({', '.join(str(s) for s in shots)})"]
    query += ["order by shot"]
    logger.trace("> {query}", query=" ".join(query))

    # results
    data = db.query(" ".join(query), use_pandas=True).values

    logger.trace("= {shape}", shape=data.shape)
    return data
