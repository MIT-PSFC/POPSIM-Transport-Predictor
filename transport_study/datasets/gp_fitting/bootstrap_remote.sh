#!/usr/bin/env bash
# One-time setup of the GP fitting environment on a SLURM cluster.
#
# The fitting jobs need only python + numpy + scipy + mkgp, so instead of
# replicating the full repo environment remotely, this builds a minimal venv
# on the cluster scratch space. Pass the venv path it prints to the dataset
# CLI as --cluster_venv.
#
# Usage:
#   bash bootstrap_remote.sh <ssh-host> <remote-workdir> [python-version]
#
#   ssh-host        Host alias from ~/.ssh/config (same one the srunx profile uses)
#   remote-workdir  Cluster scratch dir for fitting files, e.g. /pool001/$USER/gpfit
#   python-version  Optional, default 3.12

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh-host> <remote-workdir> [python-version]" >&2
    exit 1
fi

HOST="$1"
WORKDIR="$2"
PYVER="${3:-3.12}"

# Keep in sync with the mkgp pin in pyproject.toml.
MKGP_SPEC="mkgp>=3.1.4"

echo "==> Creating $WORKDIR on $HOST"
ssh "$HOST" "mkdir -p '$WORKDIR/logs'"

echo "==> Building venv with python $PYVER (installs uv if missing)"
ssh "$HOST" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd '$WORKDIR'
uv venv --python '$PYVER' .venv
uv pip install --python .venv/bin/python numpy scipy '$MKGP_SPEC'
.venv/bin/python -c "import mkgp; print('mkgp OK:', mkgp.__file__)"
EOF

echo
echo "Remote environment ready."
echo "  venv: $WORKDIR/.venv"
echo "  mkgp: $MKGP_SPEC (from PyPI)"
echo
echo "Pass to the dataset CLI:"
echo "  --cluster_remote_workdir '$WORKDIR' --cluster_venv '$WORKDIR/.venv'"
echo
echo "If you have not yet, register the srunx profile:"
echo "  srunx ssh profile add <profile-name> --ssh-host $HOST"
