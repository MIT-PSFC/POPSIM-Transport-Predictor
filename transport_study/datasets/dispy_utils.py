from disruption_py.settings import LogSettings


def passive_log_settings() -> LogSettings:
    """LogSettings that leave this process's loguru sinks alone.

    disruption_py's setup_logging()/reset_handlers() call logger.remove(),
    which wipes every sink on the global loguru logger - including the
    dataset CLI's raw_data_{pid}.log file sink, silently emptying the run
    log. Pre-setting the setup flag and a console level skips both reset
    paths in disruption_py's get_shots_data, so its messages flow through
    whatever sinks the caller already configured.
    """
    return LogSettings(file_path=None, console_level="VERBOSE", _logging_has_been_setup=True)
