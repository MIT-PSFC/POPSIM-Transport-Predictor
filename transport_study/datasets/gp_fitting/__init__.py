"""GP profile fitting that can run in-process or be dispatched to a SLURM cluster.

fit_worker.py is the single source of truth for the fitting math and the batch
file format. It must stay importable with only numpy + gptools installed, since
it is shipped alone to remote clusters.
"""
