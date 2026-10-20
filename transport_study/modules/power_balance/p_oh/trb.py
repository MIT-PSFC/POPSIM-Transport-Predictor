from transport_study.modules.power_balance.p_oh.module import OhmicPower
from transport_study.modules.power_balance.scalar_power_trb import ScalarPowerTRB


class OhmicPowerTRB(ScalarPowerTRB):
    """TrainRunBuilder for the ohmic power predictor used in transfer learning"""

    SIGNAL = "power_ohm_MW"
    MODULE_CLS = OhmicPower
