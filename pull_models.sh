#!/bin/bash
# pull_models.sh — fetch champion model checkpoints from HPC to local backup.
#
# Run from your LOCAL machine (not HPC). Pulls into models_backup/.
# See BACKUP.md for the full procedure and rationale.
#
# Usage:
#   ./pull_models.sh           # priority 1+2 (~30 MB)
#   ./pull_models.sh --all     # priority 1+2+3 (~120 MB; includes the
#                              # drawness curriculum dataset, ~57 MB)

set -euo pipefail

REMOTE="s235437@transfer.gbar.dtu.dk"
HPC_HOME="~/Petra-ver2/models"
HPC_DATA="/dtu/blackhole/0b/206091"
DEST="$(dirname "$0")/models_backup"

mkdir -p "${DEST}"

# Priority 1 — champions
PRIO1=(
    "drawness_full_natural_init"
    "natural"
)

# Priority 2 — supporting / predecessors
PRIO2=(
    "2021_06_all"
    "drawness_full"
    "feb_sf"
)

# Priority 3 — alternatives + the dataset
PRIO3_MODELS=(
    "natural_v2"
    "natural_with_drawness"
    "drawness_head_v2"
)
PRIO3_DATA=(
    "${HPC_DATA}/dataset_drawness_curriculum.pt"
)

pull_model() {
    local name="$1"
    local src="${HPC_HOME}/${name}/best.pt"
    local dst="${DEST}/${name}.best.pt"
    if [ -f "${dst}" ]; then
        echo "  [skip] ${name}.best.pt — already present"
        return
    fi
    echo "  [pull] ${name}.best.pt"
    scp -p "${REMOTE}:${src}" "${dst}"
}

pull_data() {
    local src="$1"
    local name=$(basename "${src}")
    local dst="${DEST}/${name}"
    if [ -f "${dst}" ]; then
        echo "  [skip] ${name} — already present"
        return
    fi
    echo "  [pull] ${name}"
    scp -p "${REMOTE}:${src}" "${dst}"
}

echo "Pulling priority 1 (champions)..."
for m in "${PRIO1[@]}"; do pull_model "$m"; done

echo "Pulling priority 2 (supporting)..."
for m in "${PRIO2[@]}"; do pull_model "$m"; done

if [ "${1:-}" = "--all" ]; then
    echo "Pulling priority 3 models..."
    for m in "${PRIO3_MODELS[@]}"; do pull_model "$m"; done

    echo "Pulling priority 3 data..."
    for d in "${PRIO3_DATA[@]}"; do pull_data "$d"; done
fi

echo
echo "Done. Local backup at: ${DEST}/"
echo "Update models_backup/MANIFEST.md checkboxes to mark which are verified."
