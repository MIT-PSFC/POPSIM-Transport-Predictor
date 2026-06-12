#!/usr/bin/env bash
# One-time setup of the GP fitting environment on a SLURM cluster.
#
# The fitting jobs need only python + numpy + scipy + gptools, so instead of
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
GPTOOLS_DIR="$REPO_ROOT/submodules/gptools"

if [ ! -f "$GPTOOLS_DIR/pyproject.toml" ]; then
    echo "gptools submodule not found at $GPTOOLS_DIR (run: git submodule update --init)" >&2
    exit 1
fi

echo "==> Creating $WORKDIR on $HOST"
ssh "$HOST" "mkdir -p '$WORKDIR/logs'"

echo "==> Syncing gptools source"
rsync -a --delete --exclude .git --exclude __pycache__ "$GPTOOLS_DIR/" "$HOST:$WORKDIR/gptools/"

echo "==> Building venv with python $PYVER (installs uv if missing)"
ssh "$HOST" bash -s <<EOF
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd '$WORKDIR'
uv venv --python '$PYVER' .venv
uv pip install --python .venv/bin/python numpy scipy ./gptools
.venv/bin/python -c "import gptools; print('gptools OK:', gptools.__file__)"
EOF

# Record which gptools commit the remote env was built from, so a stale env
# can be detected by comparing against the local submodule commit.
GPTOOLS_COMMIT="$(git -C "$GPTOOLS_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
ssh "$HOST" "echo '$GPTOOLS_COMMIT' > '$WORKDIR/gptools_commit'"

echo
echo "Remote environment ready."
echo "  venv:    $WORKDIR/.venv"
echo "  gptools: $GPTOOLS_COMMIT"
echo
echo "Pass to the dataset CLI:"
echo "  --cluster_remote_workdir '$WORKDIR' --cluster_venv '$WORKDIR/.venv'"
echo
echo "If you have not yet, register the srunx profile:"
echo "  srunx ssh profile add <profile-name> --ssh-host $HOST"
