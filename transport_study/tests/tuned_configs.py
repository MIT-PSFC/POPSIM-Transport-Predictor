"""Tuned configs for the tests that build the train config of a case that is not a hyperparam case.

Production writes a hyperparam case's tuned config when its sweep finishes, or from its base config under skip_tuning,
and every other case merges that file strictly (Study._apply_tuned_config), so such a test writes them first.
"""

from transport_study.orchestration.study import Study


def write_hyperparam_tuned_configs(study: Study) -> None:
    """Write the base train config of every local hyperparam case of the study as its tuned config, as skip_tuning does.

    A hyperparam case whose prereqs are hyperparam cases (the submodule ones) gets theirs first,
    since its anchor terms read their tuned deltas. Cases that already have a tuned config are left alone.
    """
    for case in study.cases:
        if case.is_hyperparam_case() and not study.is_borrowed(case):
            _write_with_prereqs(study, case)


def _write_with_prereqs(study: Study, case: Study.Case) -> None:
    if study.tuned_config_path(case).exists():
        return
    for prereq in case.prereqs or []:
        if prereq.is_hyperparam_case():
            _write_with_prereqs(study, prereq)
    study.write_tuned_config(case, study.make_train_config(case))
