# BACKUP.md — Backing up the Petra champion models

The two champion models are the project's load-bearing artifacts. The
HPC home directory is not strictly redundant, and the `dataset_*.pt`
files in `/dtu/blackhole/0b/206091/` will be **deleted after summer
2026** per the scratch-space policy.

## What to back up (priority order)

| Priority | What | Why | Where (HPC) |
|---|---|---|---|
| 1 | `models/drawness_full_natural_init/best.pt` | petra-drawness champion; novel result; paper-worthy | `~/Petra-ver2/models/drawness_full_natural_init/best.pt` |
| 1 | `models/natural/best.pt` | petra-elo champion; +143 ELO baseline | `~/Petra-ver2/models/natural/best.pt` |
| 2 | `models/2021_06_all/best.pt` | base init for natural lineage; needed for reproduction | `~/Petra-ver2/models/2021_06_all/best.pt` |
| 2 | `models/drawness_full/best.pt` | predecessor; useful for ablation history | `~/Petra-ver2/models/drawness_full/best.pt` |
| 2 | `models/feb_sf/best.pt` | earlier playing baseline | `~/Petra-ver2/models/feb_sf/best.pt` |
| 3 | `dataset_drawness_curriculum.pt` | training data for both champions; reproducible from Lichess but expensive | `/dtu/blackhole/0b/206091/dataset_drawness_curriculum.pt` |
| 3 | `models/natural_v2/best.pt` | alternative track; included for completeness | `~/Petra-ver2/models/natural_v2/best.pt` |
| 3 | `models/natural_with_drawness/best.pt` | Path A negative result; useful documentation | `~/Petra-ver2/models/natural_with_drawness/best.pt` |
| 3 | `models/drawness_head_v2/best.pt` | post-hoc-fit precedent | `~/Petra-ver2/models/drawness_head_v2/best.pt` |

Each `best.pt` is ~5.5 MB (1.36M params × float32). Total priority-1+2
checkpoints: ~30 MB. With the curriculum dataset (~57 MB), priority-1+2+3
fits in ~120 MB.

## Pulling the backup

The HPC's `transfer.gbar.dtu.dk` is sftp-only at the moment. From your
local machine:

```bash
# Set up SSH key auth on HPC first if not already (one-time):
#   ssh-copy-id s235437@login.gbar.dtu.dk
# Or use sftp interactively (password each time):

# Priority 1 — champions
scp s235437@transfer.gbar.dtu.dk:~/Petra-ver2/models/drawness_full_natural_init/best.pt \
    models_backup/drawness_full_natural_init.best.pt
scp s235437@transfer.gbar.dtu.dk:~/Petra-ver2/models/natural/best.pt \
    models_backup/natural.best.pt

# Priority 2 — supporting
scp s235437@transfer.gbar.dtu.dk:~/Petra-ver2/models/2021_06_all/best.pt \
    models_backup/2021_06_all.best.pt
scp s235437@transfer.gbar.dtu.dk:~/Petra-ver2/models/drawness_full/best.pt \
    models_backup/drawness_full.best.pt
scp s235437@transfer.gbar.dtu.dk:~/Petra-ver2/models/feb_sf/best.pt \
    models_backup/feb_sf.best.pt
```

Or use the convenience script:
```bash
./pull_models.sh           # priority 1+2 only
./pull_models.sh --all     # priority 1+2+3
```

## Storing the backup

The `.pt` files are gitignored by default — too large to live in git
history without LFS. Pick one or more:

1. **GitHub via Git LFS** (recommended for the paper):
   ```bash
   git lfs install
   git lfs track "models_backup/*.pt"
   git add .gitattributes models_backup/*.pt
   git commit -m "Add champion model checkpoints"
   git push
   ```
   GitHub free tier gives 1 GB LFS storage / 1 GB bandwidth/month — plenty for these.

2. **External cloud** (Google Drive, Dropbox, S3, IDA storage) — pick one
   and document the location in `models_backup/MANIFEST.md`.

3. **Both.** Belt and suspenders for paper artifacts.

## What is NOT backed up here

- Training logs (`hpc_logs/`) — already pulled and tracked in git
- `epoch_NN.pt` per-epoch checkpoints — discardable; only `best.pt` is canonical
- `latest.pt` / `resume.pt` — training-state checkpoints, useful for resume
  but not the production artifact
- Larger datasets in `/dtu/blackhole/` — reproducible from Lichess archives;
  document the parse parameters in `MANIFEST.md`
