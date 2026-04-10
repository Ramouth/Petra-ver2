# HPC Guide — Petra on DTU gbar

## Environment

- **Cluster**: DTU HPC (gbar login nodes: `gbarlogin1`, `gbarlogin2`)
- **Scheduler**: LSF (`bsub`, `bjobs`, `bkill`)
- **Access**: CPU-only confirmed. GPU available via `gpuv100` queue (24h wall limit).
- **Home**: `/zhome/81/b/206091/`
- **Project**: `~/Petra-ver2/`
- **Logs**: `~/logs/`
- **Python env**: `~/petra-env/`
- **Stockfish**: `~/bin/stockfish`

---

## Before Running Anything

Always activate the virtualenv:

```bash
source ~/petra-env/bin/activate
```

Stockfish requires a newer `libstdc++` than the default system provides. Any job that calls Stockfish must load the GCC module:

```bash
module load gcc/13.4.0-binutils-2.44
```

This must be in every job script that uses `reeval_stockfish.py`. Forgetting it causes:
```
RuntimeError: Stockfish process exited before sending 'uciok'
stderr: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.30' not found
```

---

## Submitting Jobs

```bash
bsub < jobs/<script>.sh          # submit a job
bjobs                            # list running/pending jobs
bjobs -a                         # include recently finished jobs
bkill <JOBID>                    # cancel a job
```

Jobs are in `jobs/`. Each writes stdout to `~/logs/<name>_<JOBID>.out` and stderr to `~/logs/<name>_<JOBID>.err`.

---

## Monitoring a Running Job

```bash
# Find job ID
bjobs

# Watch live output (replace JOBID)
tail -f ~/logs/<name>_<JOBID>.out

# Check for errors
cat ~/logs/<name>_<JOBID>.err

# Check output file exists (e.g. after parse)
ls -lh ~/Petra-ver2/data/dataset.pt
```

---

## Current Pipeline (Session 8 — SF Lichess)

### Step 1 — Parse Lichess PGN → dataset.pt

**Status: completed** (Apr 9, 1.1G output)

```bash
bsub < jobs/parse_lichess.sh
```

Produces: `~/Petra-ver2/data/dataset.pt` (~1.1M positions, 150k games, min ELO 1500)

### Step 2 — SF Reeval → dataset_sf.pt

```bash
bsub < jobs/reeval_sf.sh
```

Settings: depth 15, 200k positions, 4 Stockfish threads, 12h wall.
Produces: `~/Petra-ver2/data/dataset_sf.pt`

Watch progress:
```bash
tail -f ~/logs/reeval_sf_<JOBID>.out
# Prints every 5000 positions: count, rate (pos/s), ETA
```

Expected rate with 4 threads at depth 15: ~4–8 pos/s → 200k positions in ~7–14h.

### Step 3 — GPU Training → models/sf_gpu/best.pt

Job script not yet written. Use `gpuv100` queue.

Template:
```bash
#!/bin/bash
#BSUB -J train_sf_gpu
#BSUB -q gpuv100
#BSUB -n 4
#BSUB -R "rusage[mem=16GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00
#BSUB -o /zhome/81/b/206091/logs/train_sf_gpu_%J.out
#BSUB -e /zhome/81/b/206091/logs/train_sf_gpu_%J.err

source /zhome/81/b/206091/petra-env/bin/activate

python3 /zhome/81/b/206091/Petra-ver2/src/train.py \
    --dataset /zhome/81/b/206091/Petra-ver2/data/dataset_sf.pt \
    --anchor-dataset /zhome/81/b/206091/Petra-ver2/data/endgame_stage12.pt \
    --anchor-fraction 0.15 \
    --out /zhome/81/b/206091/Petra-ver2/models/sf_gpu/
```

Run 3 seeds in parallel for A/B geometry selection (see FIX.md).

### Step 4 — Geometry Probe

After training:
```bash
python3 src/generate_endgame.py --positions 5000 --stages 1 2 \
    --out ~/Petra-ver2/data/endgame_probe.pt

python3 src/probe_geometry.py \
    --model ~/Petra-ver2/models/sf_gpu/best.pt \
    --dataset ~/Petra-ver2/data/endgame_probe.pt \
    --n 5000
```

**Gate criterion: effective rank > 30.**

---

## Job Scripts Summary

| Script | Queue | Wall | CPUs | What it does |
|--------|-------|------|------|--------------|
| `parse_lichess.sh` | hpc | 4h | 4 | PGN → dataset.pt (150k games) |
| `reeval_sf.sh` | hpc | 12h | 4 | SF depth 15 reeval → dataset_sf.pt |
| `endgame_stage12.sh` | hpc | 4h | 4 | Endgame curriculum training (done) |
| `train_sf_gpu.sh` | gpuv100 | 24h | 4+GPU | SF Lichess training (to write) |

---

## Common Failures

### Stockfish won't start — libstdc++ error
```
RuntimeError: Stockfish process exited before sending 'uciok'
```
Fix: add `module load gcc/13.4.0-binutils-2.44` to the job script before the python call.

### Job killed — wall time exceeded
Check `~/logs/<name>_<JOBID>.err` for `TERM_RUNLIMIT`. Increase `#BSUB -W` and resubmit.

### ModuleNotFoundError: torch
Not in system Python. Every job must start with:
```bash
source /zhome/81/b/206091/petra-env/bin/activate
```

### GPU job stuck in queue
Check available GPU slots:
```bash
bjobs -q gpuv100
bqueues -q gpuv100
```

---

## Useful One-Liners

```bash
# Check what's running
bjobs

# Tail the most recent log for a job name
ls -t ~/logs/reeval_sf_*.out | head -1 | xargs tail -f

# Check dataset size
ls -lh ~/Petra-ver2/data/

# Check model directory
ls -lh ~/Petra-ver2/models/

# Interactive session for debugging (CPU, 1h)
bsub -Is -q hpc -n 2 -R "rusage[mem=4GB]" -W 1:00 bash
# Then activate env + run scripts manually
```

---

## Notes

- **Depth vs decisiveness**: SF depth 10 on Lichess middlegames gives ~43% decisive labels (std 0.565). Depth 15 is the minimum for useful geometry training signal. See project memory for rationale.
- **Stockfish threads**: `reeval_stockfish.py` sends `setoption name Threads value 4` at init. Requesting 4 CPUs in the job script (`#BSUB -n 4`) is required to match.
- **GPU queue**: `gpuv100` has 24h wall limit. Request with `#BSUB -gpu "num=1:mode=exclusive_process"`.
