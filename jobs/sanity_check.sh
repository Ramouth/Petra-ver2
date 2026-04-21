#!/bin/bash
#BSUB -J sanity_check
#BSUB -q hpc
#BSUB -n 1
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 0:10
#BSUB -o /zhome/81/b/206091/logs/sanity_check_%J.out
#BSUB -e /zhome/81/b/206091/logs/sanity_check_%J.err

# Run post-training sanity checks on any saved model without re-training.
# Usage:
#   bsub -env "MODEL=models/feb_sf/best.pt" < jobs/sanity_check.sh
#   bsub -env "MODEL=models/lichess_2023_03_e1/best.pt" < jobs/sanity_check.sh

MODEL="${MODEL:-models/feb_sf/best.pt}"
HOME_DIR="/zhome/81/b/206091"
SRC="${HOME_DIR}/Petra-ver2/src"

echo "=== Sanity check: ${HOME_DIR}/Petra-ver2/${MODEL} ==="
echo

source "${HOME_DIR}/petra-env/bin/activate"

python3 -c "
import sys, torch, chess
sys.path.insert(0, '${SRC}')
from model import PetraNet

device = torch.device('cpu')
model = PetraNet().to(device)
model.load_state_dict(torch.load('${HOME_DIR}/Petra-ver2/${MODEL}', map_location=device, weights_only=True))
model.eval()

DRAW_THRESHOLD = 0.35
tests = [
    ('Start position',         chess.Board(),                                    None),
    ('White up queen',         chess.Board('4k3/8/8/8/8/8/8/Q3K3 w - - 0 1'),  1.0),
    ('Black up queen',         chess.Board('4K3/8/8/8/8/8/8/q3k3 w - - 0 1'), -1.0),
    ('KQ vs K, White to move', chess.Board('8/8/8/8/4k3/8/8/3QK3 w - - 0 1'),  1.0),
    ('KQ vs K, Black to move', chess.Board('8/8/8/8/4k3/8/8/3QK3 b - - 0 1'), -1.0),
    ('KR vs KR (drawn)',       chess.Board('8/3k4/8/r7/8/8/3K4/7R w - - 0 1'), 'draw'),
]

all_pass = True
for name, board, expected in tests:
    val = model.value(board, device)
    if expected is None:
        mark, note = '~', ''
    elif expected == 'draw':
        ok = abs(val) < DRAW_THRESHOLD
        mark = '✓' if ok else '✗'
        note = f'  (expected |value| < {DRAW_THRESHOLD}, got {abs(val):.3f})'
        if not ok: all_pass = False
    else:
        ok = (val * expected) > 0
        mark = '✓' if ok else '✗'
        note = f\"  (expected sign {'+'  if expected > 0 else '-'})\"
        if not ok: all_pass = False
    print(f'  {mark} {name:35s}  value={val:+.3f}{note}')

print()
print('All checks passed.' if all_pass else 'WARNING: some checks failed.')
"
