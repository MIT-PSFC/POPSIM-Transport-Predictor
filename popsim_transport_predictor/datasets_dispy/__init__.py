import numpy as np

from disruption_py.machine.tokamak import Tokamak
from disruption_py.settings import TimeSetting, TimeSettingParams


class Uniform1kHzTimeSetting(TimeSetting):
    """
    Time setting for creating a uniform timebase at 1 kHz, based on the maximum EFIT time.
    """

    def _get_times(self, params: TimeSettingParams) -> np.ndarray:
        """
        Parameters
        ----------
        params : TimeSettingParams
            Parameters needed to retrieve the timebase.

        Returns
        -------
        np.ndarray
            Array of times in the timebase.
        """
        (efit_time,) = params.mds_conn.get_dims(
            r"\efit_aeqdsk:ali", tree_name="_efit_tree"
        )

        max_time = np.max(efit_time)
        if params.tokamak == Tokamak.CMOD:
            times = np.round(np.arange(0, max_time + 1e-3, 1e-3), 3)
        if params.tokamak == Tokamak.D3D:
            times = np.round(np.arange(0, max_time + 1, 1), 0)
            times = times * 1e-3  # Convert to seconds

        times = np.unique(times).astype("float32")
        return times
