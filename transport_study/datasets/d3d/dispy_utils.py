from functools import partialmethod

from disruption_py.settings import LogSettings
from loguru import logger

VERBOSE_LEVEL_NO = 15


def register_verbose_level():
    """Add loguru's VERBOSE level and bind logger.verbose(), idempotently.

    disruption_py does this inside LogSettings.setup_logging(), which passive_log_settings skips,
    but its own modules still call logger.verbose().
    """
    try:
        logger.level("VERBOSE", color="<dim>")
    except ValueError:
        logger.level("VERBOSE", color="<dim>", no=VERBOSE_LEVEL_NO)
    if not hasattr(logger.__class__, "verbose"):
        logger.__class__.verbose = partialmethod(logger.__class__.log, "VERBOSE")


def passive_log_settings() -> LogSettings:
    """LogSettings that leave this process's loguru sinks alone.

    disruption_py's setup_logging()/reset_handlers() call logger.remove(),
    which wipes every sink on the global loguru logger - including the
    dataset CLI's raw_data_{pid}.log file sink, silently emptying the run
    log. Pre-setting the setup flag and a console level skips both reset
    paths in disruption_py's get_shots_data, so its messages flow through
    whatever sinks the caller already configured.
    """
    register_verbose_level()
    return LogSettings(file_path=None, console_level="VERBOSE", _logging_has_been_setup=True)
