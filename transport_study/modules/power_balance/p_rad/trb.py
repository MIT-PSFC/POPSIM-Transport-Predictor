from transport_study.modules.power_balance.p_rad.module import RadiatedPower
from transport_study.modules.power_balance.scalar_power_trb import ScalarPowerTRB


class RadiatedPowerTRB(ScalarPowerTRB):
    """TrainRunBuilder for the radiated power predictor used in transfer learning"""

    SIGNAL = "power_radiated_MW"
    MODULE_CLS = RadiatedPower
