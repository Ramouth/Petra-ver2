#!/bin/bash
#BSUB -J train_phase15_${EXP:-mid_only}
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=32GB] span[hosts=1]"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_phase15_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_phase15_%J.err

# Phase 1.5 ablation training — one script for all 6 conditions.
# EXP controls the dataset combination:
#
#   mid_only        — mid-band data only (baseline)
#   special_only    — endgame + near_mate + near_mate_losing + material (no mid)
#   mid_full        — mid + all special datasets
#   mid_no_endgame  — mid + near_mate + near_mate_losing + material
#   mid_no_nearmate — mid + endgame + material
#   mid_no_material — mid + endgame + near_mate + near_mate_losing
#
#   bsub -env "EXP=mid_only"        < jobs/train_phase15_exp.sh
#   bsub -env "EXP=special_only"    < jobs/train_phase15_exp.sh
#   bsub -env "EXP=mid_full"        < jobs/train_phase15_exp.sh
#   bsub -env "EXP=mid_no_endgame"  < jobs/train_phase15_exp.sh
#   bsub -env "EXP=mid_no_nearmate" < jobs/train_phase15_exp.sh
#   bsub -env "EXP=mid_no_material" < jobs/train_phase15_exp.sh

EXP="${EXP:-mid_only}"

BLACKHOLE="/dtu/blackhole/0b/206091"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

MID="${BLACKHOLE}/dataset_phase15_mid_sf18.pt"
SPECIAL_FULL="${BLACKHOLE}/dataset_special_full.pt"
SPECIAL_NO_ENDGAME="${BLACKHOLE}/dataset_special_no_endgame.pt"
SPECIAL_NO_NEARMATE="${BLACKHOLE}/dataset_special_no_nearmate.pt"
SPECIAL_NO_MATERIAL="${BLACKHOLE}/dataset_special_no_material.pt"

INIT_MODEL="${INIT_MODEL:-${HOME_DIR}/Petra-ver2/models/feb_sf/best.pt}"
OUT_DIR="${HOME_DIR}/Petra-ver2/models/phase15_${EXP}"

case "${EXP}" in
    mid_only)
        DATASET="${MID}"
        EXTRA=""
        ;;
    special_only)
        DATASET="${SPECIAL_FULL}"
        EXTRA=""
        ;;
    mid_full)
        DATASET="${MID}"
        EXTRA="${SPECIAL_FULL}"
        ;;
    mid_no_endgame)
        DATASET="${MID}"
        EXTRA="${SPECIAL_NO_ENDGAME}"
        ;;
    mid_no_nearmate)
        DATASET="${MID}"
        EXTRA="${SPECIAL_NO_NEARMATE}"
        ;;
    mid_no_material)
        DATASET="${MID}"
        EXTRA="${SPECIAL_NO_MATERIAL}"
        ;;
    *)
        echo "ERROR: unknown EXP=${EXP}"
        echo "Valid: mid_only | special_only | mid_full | mid_no_endgame | mid_no_nearmate | mid_no_material"
        exit 1
        ;;
esac

echo "=== Phase 1.5 training: exp=${EXP} ==="
echo "Dataset:    ${DATASET}"
echo "Extra:      ${EXTRA:-none}"
echo "Init model: ${INIT_MODEL}"
echo "Out:        ${OUT_DIR}"
echo

for F in "${DATASET}" "${INIT_MODEL}"; do
    if [ ! -f "${F}" ]; then echo "ERROR: not found: ${F}"; exit 1; fi
done
if [ -n "${EXTRA}" ] && [ ! -f "${EXTRA}" ]; then
    echo "ERROR: extra dataset not found: ${EXTRA}"; exit 1
fi

source "${HOME_DIR}/petra-env/bin/activate"
module load gcc/13.4.0-binutils-2.44
module load cuda/12.1

EXTRA_ARG=""
if [ -n "${EXTRA}" ]; then
    EXTRA_ARG="--extra-dataset ${EXTRA}"
fi

python3 -u "${SRC}/train.py" \
    --dataset       "${DATASET}" \
    ${EXTRA_ARG} \
    --init-model    "${INIT_MODEL}" \
    --out           "${OUT_DIR}" \
    --policy-weight 0.0 \
    --rank-reg      0.1 \
    --draw-reg      0.0 \
    --weight-decay  5e-4 \
    --lr            3e-4 \
    --epochs        30 \
    --geo-patience  5 \
    --num-workers   0

echo
echo "Done. Checkpoint: ${OUT_DIR}/best.pt"
echo "Next: bsub -env \"EXP=${EXP}\" < jobs/eval_phase15_exp.sh"
