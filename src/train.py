"""
Supervised pretraining for PetraNet.

Trains value head and policy head jointly on Lichess game data:
  value  loss: MSE  (predicted value vs game outcome)
  policy loss: cross-entropy (predicted move distribution vs move played)
  total  loss: value_loss + policy_loss  (equal weighting, AlphaZero convention)

Validation
----------
Runs after every epoch on the held-out val set (split at game level in data.py).
Reports: total loss, value MSE, value R², policy top-1 accuracy, policy top-5 accuracy.

Early stopping on geometry patience: stops when effective rank and win·draw cosine
both stop improving for --geo-patience consecutive epochs (default 3).
Best checkpoint saved when geometry improves, not on val loss.

Usage
-----
    python3 train.py --dataset dataset.pt --out models/
    python3 train.py --dataset dataset.pt --out models/ --epochs 20 --lr 3e-4
"""

import argparse
import os
import sys
import time

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import PetraNet
from config import device
from topology_monitor import (topological_health_check,
                               should_abort_early,
                               format_topology_line)


# ---------------------------------------------------------------------------
# Inline geometry metrics — called every epoch
# ---------------------------------------------------------------------------

def _compute_geometry_metrics(model: "PetraNet", val_loader, n: int = 300) -> dict:
    """
    Compute effective rank and strict win·draw centroid cosine from val geometry.

    Fast enough to call every epoch: extracts n geometry vectors, runs one
    SVD of the 128×128 covariance matrix.

    Returns:
        eff_rank     (float)       — (Σλ)²/Σλ² on mean-centred covariance
        win_draw_cos (float|None)  — cosine of win/draw centroids using strict
                                     thresholds (win >0.7, draw <0.3); None if
                                     fewer than 10 samples in either bucket
    """
    model.eval()
    vecs, vals = [], []
    count = 0
    with torch.no_grad():
        for batch in val_loader:
            if count >= n:
                break
            tensors = batch[0].float().to(device)
            take    = min(len(tensors), n - count)
            g = model.geometry(tensors[:take])
            vecs.append(g.cpu().numpy())
            vals.append(batch[1][:take].cpu().numpy())
            count += take

    vecs = np.concatenate(vecs, axis=0)
    vals = np.concatenate(vals, axis=0)

    # Effective rank on mean-centred covariance
    centred  = vecs - vecs.mean(axis=0)
    cov      = np.cov(centred.T)
    eigvals  = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eff_rank = float((eigvals.sum() ** 2) / ((eigvals ** 2).sum() + 1e-8))

    # Strict win·draw centroid cosine (win >0.7, draw <0.3 STM-relative)
    win_mask  = vals >  0.7
    draw_mask = np.abs(vals) < 0.3
    win_draw_cos = None
    if win_mask.sum() >= 10 and draw_mask.sum() >= 10:
        c_win  = vecs[win_mask].mean(axis=0)
        c_draw = vecs[draw_mask].mean(axis=0)
        denom  = (np.linalg.norm(c_win) + 1e-8) * (np.linalg.norm(c_draw) + 1e-8)
        win_draw_cos = float(np.dot(c_win, c_draw) / denom)

    return {"eff_rank": eff_rank, "win_draw_cos": win_draw_cos}


def _compute_drawness_sanity(model: "PetraNet") -> dict:
    """
    Probe whether the auxiliary drawness head separates structural draws from
    sharp balanced positions. Used only when --draw-reg is active.
    """
    import chess

    model.eval()
    krkr = chess.Board("8/3k4/8/r7/8/8/3K4/7R w - - 0 1")
    sharp_equal = chess.Board(
        "rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"
    )
    with torch.no_grad():
        d_structural = model.drawness(krkr, device)
        d_balanced = model.drawness(sharp_equal, device)
    return {
        "structural": d_structural,
        "balanced": d_balanced,
        "gap": d_structural - d_balanced,
    }


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset(path: str):
    """
    Load a dataset saved by data.py or selfplay.py.

    Supervised datasets have 'move_idxs' (int64 one-hot target).
    Self-play datasets additionally have 'visit_dists' (float32 dense target).
    Always returns 4-tuple loaders — supervised positions get a one-hot visit_dist
    so they can be mixed with self-play data in a single DataLoader.
    """
    print(f"Loading dataset from {path} ...")
    data = torch.load(path, map_location="cpu", weights_only=False)

    has_visit_dists = "visit_dists" in data.get("train", {})
    if has_visit_dists:
        print("  Self-play dataset detected — using dense policy loss (visit distributions)")
    else:
        print("  Supervised dataset detected — one-hot visit distributions will be created")

    def make_loader(split, batch_size, shuffle):
        d = data[split]
        tensors_list = [d["tensors"], d["values"], d["move_idxs"], _ensure_visit_dists(d)]
        if "legal_masks" in d:
            tensors_list.append(d["legal_masks"])
        ds = TensorDataset(*tensors_list)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=(device.type == "cuda"))

    meta = data.get("meta", {})
    print(f"  train: {meta.get('n_train', '?'):,}  val: {meta.get('n_val', '?'):,}  "
          f"source: {meta.get('source', 'supervised')}")
    return make_loader, data, True   # always dense_policy=True now


def _ensure_visit_dists(d: dict) -> torch.Tensor:
    """Return visit_dists tensor; create one-hot from move_idxs if absent."""
    if "visit_dists" in d:
        return d["visit_dists"]
    n = len(d["move_idxs"])
    vd = torch.zeros(n, 4096, dtype=torch.float32)
    vd[torch.arange(n), d["move_idxs"]] = 1.0
    return vd


def merge_datasets(primary_data: dict, extra_path: str) -> dict:
    """
    Merge a second full dataset into the primary for training.

    Both train AND val splits are merged so early stopping monitors loss
    on both distributions — if the model memorises primary positions, the
    extra val loss will catch it.

    Use this to combine datasets from different Lichess months (different
    position distributions) rather than mix_anchor which is for small
    anchor fractions.
    """
    print(f"\nMerging extra dataset: {extra_path}")
    extra = torch.load(extra_path, map_location="cpu", weights_only=False)

    def _cat_split(a: dict, b: dict) -> dict:
        merged = {
            "tensors":   torch.cat([a["tensors"],   b["tensors"]]),
            "values":    torch.cat([a["values"],     b["values"]]),
            "move_idxs": torch.cat([a["move_idxs"],  b["move_idxs"]]),
            "visit_dists": torch.cat([_ensure_visit_dists(a),
                                      _ensure_visit_dists(b)]),
        }
        if "legal_masks" in a or "legal_masks" in b:
            # If one side is missing masks, substitute all-ones (0xFF = all bits
            # set = all moves legal = no masking effect for those positions).
            a_masks = a.get("legal_masks",
                            torch.full((len(a["tensors"]), 512), 255, dtype=torch.uint8))
            b_masks = b.get("legal_masks",
                            torch.full((len(b["tensors"]), 512), 255, dtype=torch.uint8))
            merged["legal_masks"] = torch.cat([a_masks, b_masks])
        # Shuffle so batches contain positions from both datasets
        perm = torch.randperm(len(merged["tensors"]))
        return {k: v[perm] for k, v in merged.items()}

    result = dict(primary_data)
    result["train"] = _cat_split(primary_data["train"], extra["train"])
    result["val"]   = _cat_split(primary_data["val"],   extra["val"])

    n_train = len(result["train"]["tensors"])
    n_val   = len(result["val"]["tensors"])
    print(f"  Merged train: {n_train:,}  val: {n_val:,}")
    return result


def mix_anchor(primary_data: dict, anchor_path: str, anchor_frac: float) -> dict:
    """
    Mix a fraction of anchor (supervised SF) positions into the primary training split.

    anchor_frac : fraction of primary train size to sample from anchor.
                  e.g. 0.15 → 15% extra positions from the SF oracle dataset.

    Val split is kept pure (primary only) so validation measures self-play generalisation.
    """
    print(f"\nMixing anchor dataset: {anchor_path}  (frac={anchor_frac})")
    anchor = torch.load(anchor_path, map_location="cpu", weights_only=False)

    n_primary = len(primary_data["train"]["tensors"])
    n_sample  = max(1, int(n_primary * anchor_frac))
    n_anchor  = len(anchor["train"]["tensors"])
    n_sample  = min(n_sample, n_anchor)

    idx = torch.randperm(n_anchor)[:n_sample]
    a   = anchor["train"]

    # Use anchor's precomputed visit_dists when available (e.g. endgame anchors
    # have uniform distributions over legal moves — use them as-is).
    # For supervised SF anchors (no visit_dists), build one-hot from move_idxs.
    # This avoids injecting arbitrary one-hot policy targets into endgame positions
    # where move_idxs is just the first legal move (meaningless for policy training).
    anchor_has_vd = "visit_dists" in a
    if anchor_has_vd:
        vd = a["visit_dists"][idx]
    else:
        a_sampled_idxs = a["move_idxs"][idx]
        vd = torch.zeros(n_sample, 4096, dtype=torch.float32)
        vd[torch.arange(n_sample), a_sampled_idxs] = 1.0

    mixed = {
        "tensors":     torch.cat([primary_data["train"]["tensors"],   a["tensors"][idx]]),
        "values":      torch.cat([primary_data["train"]["values"],    a["values"][idx]]),
        "move_idxs":   torch.cat([primary_data["train"]["move_idxs"], a["move_idxs"][idx]]),
    }
    if "visit_dists" in primary_data["train"]:
        mixed["visit_dists"] = torch.cat([primary_data["train"]["visit_dists"], vd])
    elif anchor_has_vd:
        # Keep anchor dense policy targets separate so large supervised primaries
        # do not require materialising a full (N, 4096) one-hot tensor in RAM.
        mixed["anchor_visit_dists"] = vd
        mixed["anchor_offset"] = n_primary

    # Preserve legal masks when available from either side.
    # If one side is missing masks, substitute all-ones (0xFF = all bits
    # set = all moves legal = no masking effect).
    if "legal_masks" in primary_data["train"] or "legal_masks" in a:
        primary_masks = primary_data["train"].get(
            "legal_masks",
            torch.full((n_primary, 512), 255, dtype=torch.uint8),
        )
        if "legal_masks" in a:
            anchor_masks = a["legal_masks"][idx]
        else:
            anchor_masks = torch.full((n_sample, 512), 255, dtype=torch.uint8)
        mixed["legal_masks"] = torch.cat([primary_masks, anchor_masks])

    # Track which positions came from the structural draw anchor pool.
    # Used by the drawness head loss: anchor positions with value ≈ 0 are
    # structural draws (positives); decisive positions anywhere are negatives.
    mixed["is_anchor"] = torch.cat([
        torch.zeros(n_primary, dtype=torch.bool),
        torch.ones(n_sample,  dtype=torch.bool),
    ])

    print(f"  anchor positions sampled: {n_sample:,} / {n_anchor:,}")
    print(f"  mixed train size: {len(mixed['tensors']):,}  "
          f"(was {n_primary:,}, +{n_sample:,} anchor)")

    result = dict(primary_data)
    result["train"] = mixed
    return result


class _IndexedSplitDataset(Dataset):
    """Dataset wrapper that preserves row indices for batch-time policy target assembly."""

    def __init__(self, split: dict):
        self.split = split

    def __len__(self):
        return len(self.split["tensors"])

    def __getitem__(self, idx: int):
        d = self.split
        item = [idx, d["tensors"][idx], d["values"][idx], d["move_idxs"][idx]]
        if "legal_masks" in d:
            item.append(d["legal_masks"][idx])
        return tuple(item)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, optimizer, is_train: bool, dense_policy: bool = False,
              policy_weight: float = 1.0, rank_reg: float = 0.0, draw_reg: float = 0.0,
              has_stored_vd: bool = True):
    model.train(is_train)
    total_loss = total_vloss = total_ploss = total_rloss = total_dloss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    all_value_preds, all_value_targets = [], []
    policy_correct_top1 = policy_correct_top5 = policy_total = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            # Batch layout — is_anchor is always the last element:
            #   dense+vd, no masks:  (tensors, values, move_idxs, visit_dists, is_anchor)              5
            #   dense+vd, masks:     (tensors, values, move_idxs, visit_dists, packed_masks, is_anchor) 6
            #   dense+no-vd, no masks: (tensors, values, move_idxs, is_anchor)                         4
            #   dense+no-vd, masks:    (tensors, values, move_idxs, packed_masks, is_anchor)            5
            if dense_policy and has_stored_vd:
                if len(batch) == 6:
                    tensors, values, move_idxs, visit_dists, packed_masks, is_anchor = batch
                else:  # 5
                    tensors, values, move_idxs, visit_dists, is_anchor = batch
                    packed_masks = None
                visit_dists = visit_dists.to(device)
            else:
                if len(batch) == 5:
                    tensors, values, move_idxs, packed_masks, is_anchor = batch
                else:  # 4
                    tensors, values, move_idxs, is_anchor = batch
                    packed_masks = None

            # Unpack legal masks per batch
            if packed_masks is not None:
                bs = packed_masks.shape[0]
                legal_bool = torch.from_numpy(
                    np.unpackbits(packed_masks.numpy().reshape(-1))
                    .reshape(bs, 4096)
                ).bool().to(device)
            else:
                legal_bool = None

            tensors   = tensors.float().to(device)   # uint8 → float32
            values    = values.to(device)
            move_idxs = move_idxs.to(device)

            # For SF supervised data without stored visit_dists, build one-hot per batch.
            # This avoids materialising a (N, 4096) float32 tensor at load time
            # (~5.8 GiB for 380k positions).
            if dense_policy and not has_stored_vd:
                visit_dists = F.one_hot(move_idxs, 4096).float()

            # Route through _geometry_fwd when any geometry-based loss is active,
            # so we get the bottleneck tensor without a second forward pass.
            if rank_reg > 0 or draw_reg > 0:
                g             = model._geometry_fwd(tensors)
                value_pred    = model.value_head(g)
                policy_logits = model.policy_head(g)
            else:
                value_pred, policy_logits = model(tensors)
                g = None
            value_pred = value_pred.squeeze(1)

            if legal_bool is not None:
                eff_logits = policy_logits.masked_fill(~legal_bool, float("-inf"))
                # Guard: if a row is all-illegal (e.g. terminal position in dataset),
                # -inf everywhere → NaN from log_softmax. Fall back to unmasked logits.
                all_masked = ~legal_bool.any(dim=1)
                if all_masked.any():
                    eff_logits = eff_logits.clone()
                    eff_logits[all_masked] = policy_logits[all_masked]
            else:
                eff_logits = policy_logits

            vloss = F.mse_loss(value_pred, values)
            if dense_policy:
                # KL(visit_dist || softmax(logits)) — masked to legal moves when available.
                # Clamp to -1e9 before multiplying: 0 * (-inf) = NaN in IEEE 754, but
                # 0 * (-1e9) = 0.  This occurs whenever visit_dists is zero for an
                # illegal move whose logit was masked to -inf.
                log_probs = F.log_softmax(eff_logits, dim=-1).clamp(min=-1e9)
                ploss = -(visit_dists * log_probs).sum(dim=-1).mean()
            else:
                ploss = F.cross_entropy(eff_logits, move_idxs)
            loss  = vloss + policy_weight * ploss

            # Rank regularisation — penalise eigenvalue concentration.
            # Vectors are L2-normalised so tr(C) = 1 always.
            # Loss = tr(C²) = ||C||_F²  ∈ [1/128, 1].
            # Minimising this maximises effective rank (= 1/tr(C²)).
            # λ=0.1 adds ~0.007 to loss when rank=7 — small but directional.
            if rank_reg > 0 and g is not None:
                C = g.T @ g / g.shape[0]
                rloss = (C * C).sum()
                loss  = loss + rank_reg * rloss
            else:
                rloss = torch.zeros(1, device=device)

            # Drawness loss — BCE on structural draw positions.
            # Positives: anchor positions with value ≈ 0.0 (endgame pool draws)
            # Negatives: decisive positions (|v| > 0.5) from anywhere
            # Balanced middlegames (|v| ≤ 0.5, non-anchor) are skipped — ambiguous.
            if draw_reg > 0 and g is not None:
                is_anchor_dev = is_anchor.to(device)
                d_pred    = model.drawness_fwd(g)
                draw_pos  = is_anchor_dev & (values.abs() < 0.1)
                draw_neg  = values.abs() > 0.5
                draw_mask = draw_pos | draw_neg
                if draw_mask.sum() >= 4:
                    draw_targets = draw_pos[draw_mask].float()
                    dloss = F.binary_cross_entropy(d_pred[draw_mask], draw_targets)
                    loss  = loss + draw_reg * dloss
                else:
                    dloss = torch.zeros(1, device=device)
            else:
                dloss = torch.zeros(1, device=device)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                total_grad_norm += float(grad_norm)
                optimizer.step()

            total_loss  += loss.item()
            total_vloss += vloss.item()
            total_ploss += ploss.item()
            total_rloss += rloss.item()
            total_dloss += dloss.item()
            n_batches   += 1

            # Accumulate for R² and accuracy
            all_value_preds.append(value_pred.detach().cpu())
            all_value_targets.append(values.detach().cpu())

            topk = eff_logits.topk(5, dim=1).indices
            policy_correct_top1 += (topk[:, 0] == move_idxs).sum().item()
            policy_correct_top5 += (topk == move_idxs.unsqueeze(1)).any(dim=1).sum().item()
            policy_total        += len(move_idxs)

    preds   = torch.cat(all_value_preds)
    targets = torch.cat(all_value_targets)
    ss_res  = ((preds - targets) ** 2).sum().item()
    ss_tot  = ((targets - targets.mean()) ** 2).sum().item() + 1e-8
    r2      = 1.0 - ss_res / ss_tot

    return {
        "loss":       total_loss  / n_batches,
        "value_loss": total_vloss / n_batches,
        "policy_loss":total_ploss / n_batches,
        "rank_loss":  total_rloss / n_batches,
        "draw_loss":  total_dloss / n_batches,
        "value_r2":   r2,
        "top1":       policy_correct_top1 / policy_total,
        "top5":       policy_correct_top5 / policy_total,
        "grad_norm":  total_grad_norm / n_batches if is_train else 0.0,
    }


def train(dataset_path: str = None,
          out_dir: str = "models",
          epochs: int = 15,
          batch_size: int = 512,
          lr: float = 1e-3,
          weight_decay: float = 1e-4,
          geo_patience: int = 3,
          seed: int = 42,
          init_model: str = None,
          resume: str = None,
          extra_dataset: str = None,
          anchor_dataset: str = None,
          anchor_frac: float = 0.15,
          policy_weight: float = 1.0,
          endgame_positions: int = 0,
          endgame_stages=None,
          num_workers: int = 0,
          deterministic: bool = False,
          rank_reg: float = 0.0,
          draw_reg: float = 0.0):

    from generate_endgame import generate_positions, build_dataset as _build_eg

    _stages = endgame_stages if endgame_stages is not None else [1]
    if isinstance(_stages, int):
        _stages = [_stages]

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Deterministic mode enabled (seed={seed})")
    os.makedirs(out_dir, exist_ok=True)

    def _fresh_endgame_data():
        positions = generate_positions(endgame_positions, include_mirrors=True,
                                       stages=_stages)
        return _build_eg(positions)

    regenerate = endgame_positions > 0

    if regenerate:
        stage_str = "+".join(f"stage{s}" for s in _stages)
        print(f"Endgame curriculum: {stage_str}  "
              f"positions/epoch={endgame_positions:,} + mirrors")
        data = _fresh_endgame_data()
        dense_policy = True
    else:
        if dataset_path is None:
            raise ValueError("--dataset is required unless --endgame-positions > 0")
        _, data, dense_policy = load_dataset(dataset_path)
        if extra_dataset:
            data = merge_datasets(data, extra_dataset)
        if anchor_dataset:
            data = mix_anchor(data, anchor_dataset, anchor_frac)

    def _make_loader(data, split, shuffle):
        d = data[split]
        has_batch_vd = ("visit_dists" in d) or ("anchor_visit_dists" in d)

        def _collate(batch):
            idxs = torch.tensor([row[0] for row in batch], dtype=torch.long)
            tensors = torch.stack([row[1] for row in batch])
            values = torch.stack([row[2] for row in batch])
            move_idxs = torch.stack([row[3] for row in batch])

            packed_masks = None
            if "legal_masks" in d:
                packed_masks = torch.stack([row[4] for row in batch])

            if "visit_dists" in d:
                visit_dists = d["visit_dists"][idxs]
            elif "anchor_visit_dists" in d:
                visit_dists = F.one_hot(move_idxs, 4096).float()
                anchor_rows = idxs >= d["anchor_offset"]
                if anchor_rows.any():
                    anchor_local = idxs[anchor_rows] - d["anchor_offset"]
                    visit_dists[anchor_rows] = d["anchor_visit_dists"][anchor_local]
            else:
                visit_dists = None

            # is_anchor: True for positions from the structural draw anchor pool.
            # Always appended last so batch format is unambiguous.
            if "is_anchor" in d:
                is_anchor = d["is_anchor"][idxs]
            else:
                is_anchor = torch.zeros(len(idxs), dtype=torch.bool)

            out = [tensors, values, move_idxs]
            if visit_dists is not None:
                out.append(visit_dists)
            if packed_masks is not None:
                out.append(packed_masks)
            out.append(is_anchor)
            return tuple(out)

        ds = _IndexedSplitDataset(d)
        # _collate is a closure — cannot be pickled for multiprocessing workers.
        # All data is already in RAM so num_workers=0 has no cost here.
        loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                            num_workers=0,
                            pin_memory=(device.type == "cuda"),
                            collate_fn=_collate)
        return loader, has_batch_vd

    train_loader, has_stored_vd = _make_loader(data, "train", shuffle=True)
    val_loader, _               = _make_loader(data, "val",   shuffle=False)

    model = PetraNet().to(device)
    if init_model:
        sd = torch.load(init_model, map_location=device, weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  New parameters (randomly initialized): {missing}")
        if unexpected:
            print(f"  Unexpected keys ignored: {unexpected}")
        print(f"Loaded starting weights from {init_model}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PetraNet: {n_params:,} parameters  |  device: {device}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5, min_lr=1e-5
        # patience=2: LR halves after 2 epochs no improvement
        # With early_stop patience=5, allows up to 2 LR reductions before stopping
    )

    best_val_loss  = float("inf")   # logged only; not used for stopping
    best_rank      = 0.0
    best_cos       = float("inf")   # lower is better; inf = no measurement yet
    best_draw_gap  = float("-inf")
    geo_no_improve = 0
    topo_trajectory  = []
    topo_check_every = 2
    start_epoch      = 1

    # Resume from full training-state checkpoint (model + optimizer + scheduler)
    if resume:
        ckpt = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch    = ckpt["epoch"] + 1
        best_val_loss  = ckpt.get("best_val_loss", float("inf"))
        best_rank      = ckpt.get("best_rank",      0.0)
        best_cos       = ckpt.get("best_cos",       float("inf"))
        best_draw_gap  = ckpt.get("best_draw_gap",  float("-inf"))
        geo_no_improve = ckpt.get("geo_no_improve", 0)
        print(f"Resumed from {resume}  (epoch {ckpt['epoch']}, "
              f"best_rank={best_rank:.1f}, best_wdcos={best_cos:.4f})")

    if rank_reg > 0:
        print(f"Rank regularisation: λ={rank_reg}  "
              f"(tr(C²) penalty — maximises effective rank)")
    if draw_reg > 0:
        if not anchor_dataset:
            print(f"WARNING: --draw-reg set but no --anchor-dataset. "
                  f"No structural draw positives — drawness loss will only train on negatives.")
        print(f"Drawness regularisation: λ={draw_reg}  "
              f"(BCE — structural draws from anchor pool vs decisive positions)")
    print(f"Geometry patience: {geo_patience} epochs  "
          f"(stops when rank and win·draw cosine both plateau)\n")
    rank_col = f"  {'RankL':>6}" if rank_reg > 0 else ""
    draw_col = f"  {'DrawL':>6}" if draw_reg > 0 else ""
    print(f"{'Epoch':>5}  {'T-loss':>7}  {'V-loss':>7}  {'V-MSE':>6}  "
          f"{'V-R²':>6}  {'Top1':>5}  {'Top5':>5}  {'LR':>8}  {'GNorm':>6}"
          + rank_col + draw_col)
    print("-" * (73 + (9 if rank_reg > 0 else 0) + (9 if draw_reg > 0 else 0)))

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()

        # Regenerate endgame positions each epoch — prevents memorisation
        if regenerate and epoch > 1:
            data = _fresh_endgame_data()
            train_loader, has_stored_vd = _make_loader(data, "train", shuffle=True)
            val_loader, _               = _make_loader(data, "val",   shuffle=False)

        train_m = run_epoch(model, train_loader, optimizer, is_train=True,
                            dense_policy=dense_policy, policy_weight=policy_weight,
                            rank_reg=rank_reg, draw_reg=draw_reg,
                            has_stored_vd=has_stored_vd)
        val_m   = run_epoch(model, val_loader,   optimizer, is_train=False,
                            dense_policy=dense_policy, policy_weight=policy_weight,
                            rank_reg=rank_reg, draw_reg=draw_reg,
                            has_stored_vd=has_stored_vd)

        scheduler.step(val_m["loss"])
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]

        rank_str = f"  {val_m['rank_loss']:>6.4f}" if rank_reg > 0 else ""
        draw_str = f"  {val_m['draw_loss']:>6.4f}" if draw_reg > 0 else ""
        print(f"{epoch:>5}  "
              f"{train_m['loss']:>7.4f}  "
              f"{val_m['loss']:>7.4f}  "
              f"{val_m['value_loss']:>6.4f}  "
              f"{val_m['value_r2']:>6.3f}  "
              f"{val_m['top1']:>5.3f}  "
              f"{val_m['top5']:>5.3f}  "
              f"{lr_now:>8.2e}  "
              f"{train_m['grad_norm']:>6.3f}"
              + rank_str + draw_str +
              f"  ({elapsed:.0f}s)")

        # --- Geometry patience (primary stopping criterion) ---
        geo = _compute_geometry_metrics(model, val_loader)
        rank = geo["eff_rank"]
        cos  = geo["win_draw_cos"]
        cos_str = f"{cos:.4f}" if cos is not None else "   n/a"

        rank_improved = rank > best_rank + 0.1
        cos_improved  = (cos is not None) and (cos < best_cos - 0.005)
        draw_improved = False
        draw = None
        if draw_reg > 0:
            draw = _compute_drawness_sanity(model)
            draw_improved = draw["gap"] > best_draw_gap + 0.02

        if rank_improved or cos_improved or draw_improved:
            geo_no_improve = 0
            if rank_improved:
                best_rank = rank
            if cos_improved:
                best_cos = cos
            if draw_improved:
                best_draw_gap = draw["gap"]
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"  Geometry: rank={rank:>5.1f}  wdcos={cos_str}  ↑ new best")
        else:
            geo_no_improve += 1
            print(f"  Geometry: rank={rank:>5.1f}  wdcos={cos_str}"
                  f"  [{geo_no_improve}/{geo_patience}]")
        if draw_reg > 0:
            print(f"  Drawness: structural={draw['structural']:.3f}  "
                  f"balanced={draw['balanced']:.3f}  gap={draw['gap']:+.3f}  "
                  f"best_gap={best_draw_gap:+.3f}")

        # --- Topological health check every 2 epochs (background crash guard) ---
        if epoch % topo_check_every == 1:
            topo = topological_health_check(model, val_loader, epoch,
                                            device=device)
            topo_trajectory.append(topo)
            print(format_topology_line(topo))

            abort, reason = should_abort_early(topo_trajectory)
            if abort:
                print(f"\nTopological early abort at epoch {epoch}: {reason}")
                print(f"  Try a different seed or learning rate.")
                sys.exit(3)  # 3 = topological abort (distinct from other exits)

        torch.save(model.state_dict(), os.path.join(out_dir, "latest.pt"))

        # Full training-state checkpoint — enables true resume on HPC preemption
        torch.save({
            "epoch":          epoch,
            "model":          model.state_dict(),
            "optimizer":      optimizer.state_dict(),
            "scheduler":      scheduler.state_dict(),
            "best_val_loss":  best_val_loss,
            "best_rank":      best_rank,
            "best_cos":       best_cos,
            "best_draw_gap":  best_draw_gap,
            "geo_no_improve": geo_no_improve,
        }, os.path.join(out_dir, "resume.pt"))

        if geo_no_improve >= geo_patience:
            print(f"\nGeometry patience exhausted ({geo_patience} epochs without improvement).")
            break

    best_cos_str = f"{best_cos:.4f}" if best_cos < float("inf") else "n/a"
    draw_gap_str = (f"  best draw_gap: {best_draw_gap:+.3f}"
                    if draw_reg > 0 and best_draw_gap > float("-inf") else "")
    print(f"\nTraining complete. Best rank: {best_rank:.1f}  "
          f"best wdcos: {best_cos_str}{draw_gap_str}")
    print(f"Best model → {os.path.join(out_dir, 'best.pt')}")

    # Final sanity check on value range
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"),
                                     map_location=device, weights_only=True))
    _sanity_check(model)


# ---------------------------------------------------------------------------
# Sanity check — runs after training, not a substitute for ELO testing
# ---------------------------------------------------------------------------

def _sanity_check(model: PetraNet):
    import chess
    print("\nPost-training sanity checks:")
    model.eval()

    DRAW_THRESHOLD = 0.35  # equal endgame should be within this of 0

    tests = [
        # (name, board, expected)
        # expected=None → unchecked (just print), positive/negative → sign check,
        # "draw" → |value| must be < DRAW_THRESHOLD
        ("Start position",          chess.Board(),                                    None),
        ("White up queen",          chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1"),  1.0),
        ("Black up queen",          chess.Board("4K3/8/8/8/8/8/8/q3k3 w - - 0 1"), -1.0),
        ("KQ vs K, White to move",  chess.Board("8/8/8/8/4k3/8/8/3QK3 w - - 0 1"),  1.0),
        ("KQ vs K, Black to move",  chess.Board("8/8/8/8/4k3/8/8/3QK3 b - - 0 1"), -1.0),
        # Equal endgame — tests whether draw dimension is open (dataset_feb_sf target)
        # Rooks on different ranks AND files so no immediate capture is possible.
        ("KR vs KR (drawn)",        chess.Board("8/3k4/8/r7/8/8/3K4/7R w - - 0 1"), "draw"),
    ]

    all_pass = True
    for name, board, expected in tests:
        val = model.value(board, device)
        if expected is None:
            ok = True
            mark = "~"
            note = ""
        elif expected == "draw":
            ok = abs(val) < DRAW_THRESHOLD
            mark = "✓" if ok else "✗"
            note = f"  (expected |value| < {DRAW_THRESHOLD}, got {abs(val):.3f})"
        else:
            ok = (val * expected) > 0   # correct sign
            mark = "✓" if ok else "✗"
            note = f"  (expected sign {'+'  if expected > 0 else '-'})"
        if not ok:
            all_pass = False
        print(f"  {mark} {name:35s}  value={val:+.3f}{note}")

    if all_pass:
        print("  All sign checks passed.")
    else:
        print("  WARNING: some sign checks failed — review training data.")

    # Drawness sanity: structural draw should score high, balanced middlegame low.
    # Gate: KR vs KR > 0.7, sharp balanced < 0.3.
    # Only meaningful after draw_reg training — newly initialised head will be random.
    print("\n  Drawness head:")
    drawness_tests = [
        ("KR vs KR (structural draw)",
         chess.Board("8/3k4/8/r7/8/8/3K4/7R w - - 0 1"),
         ">0.7"),
        ("Sicilian 1.d4 c5 (balanced, sharp)",
         chess.Board("rnbqkbnr/pp1ppppp/8/2p5/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2"),
         "<0.3"),
    ]
    for name, board, gate in drawness_tests:
        d_val = model.drawness(board, device)
        if gate == ">0.7":
            ok = d_val > 0.7
        else:
            ok = d_val < 0.3
        mark = "✓" if ok else "~"  # ~ = not yet trained, not a hard failure
        print(f"    {mark} {name:45s}  drawness={d_val:.3f}  (gate: {gate})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default=None,
                    help="Pre-built dataset (.pt). Required unless --endgame-positions > 0.")
    ap.add_argument("--out",         default="models")
    ap.add_argument("--epochs",      type=int,   default=15)
    ap.add_argument("--batch-size",  type=int,   default=512)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 regularisation (default: 1e-4). Increase to 5e-4 when merging datasets.")
    ap.add_argument("--extra-dataset", default=None,
                    help="Second dataset to merge into train+val before training. "
                         "Use for combining different Lichess months.")
    ap.add_argument("--geo-patience", type=int, default=3,
                    help="Stop after this many epochs with no improvement in "
                         "effective rank or win·draw cosine (default: 3)")
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--init-model",  default=None,
                    help="Load these weights before training (zigzag fine-tuning)")
    ap.add_argument("--anchor-dataset", default=None,
                    help="Path to supervised SF dataset to mix in as anchor.")
    ap.add_argument("--anchor-frac", type=float, default=0.15,
                    help="Fraction of primary train size to sample from anchor (default: 0.15)")
    ap.add_argument("--policy-weight", type=float, default=1.0,
                    help="Weight for policy loss. Set to 0 for value-only (endgame curriculum).")
    ap.add_argument("--endgame-positions", type=int, default=0,
                    help="If > 0, regenerate this many endgame positions each epoch. "
                         "Replaces --dataset for the endgame curriculum.")
    ap.add_argument("--endgame-stages", type=int, nargs="+", default=[1],
                    help="Endgame stages: 1=KQK 2=KRK 3=KPK 4=KQvKR ... (default: 1)")
    ap.add_argument("--num-workers",   type=int, default=0,
                    help="DataLoader worker processes (default: 0). "
                         "Set to 2-4 on GPU nodes to overlap data loading with compute.")
    ap.add_argument("--resume",        default=None,
                    help="Path to resume.pt to continue a previous run "
                         "(restores model, optimizer, scheduler, epoch, best loss).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Enable cuDNN deterministic mode for reproducibility "
                         "(slightly slower). Also seeds numpy and random.")
    ap.add_argument("--rank-reg", type=float, default=0.0,
                    help="Rank regularisation weight λ (default: 0). "
                         "Adds λ·tr(C²) to the loss to penalise eigenvalue "
                         "concentration and push effective rank higher. "
                         "Try 0.05–0.2. tr(C²) ≈ 1/eff_rank so at rank=7 "
                         "the penalty adds ~λ·0.14.")
    ap.add_argument("--draw-reg", type=float, default=0.0,
                    help="Drawness regularisation weight λ (default: 0). "
                         "Trains the auxiliary drawness head with BCE loss. "
                         "Requires --anchor-dataset: anchor positions with |v|<0.1 "
                         "are structural draw positives; |v|>0.5 positions are negatives. "
                         "Try 0.01–0.05. Balanced middlegames (|v|≤0.5, non-anchor) are "
                         "skipped — they are ambiguous, not draws.")
    args = ap.parse_args()

    if args.dataset is None and args.endgame_positions == 0:
        ap.error("--dataset is required unless --endgame-positions > 0")

    train(
        dataset_path=args.dataset,
        out_dir=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        extra_dataset=args.extra_dataset,
        geo_patience=args.geo_patience,
        seed=args.seed,
        init_model=args.init_model,
        anchor_dataset=args.anchor_dataset,
        anchor_frac=args.anchor_frac,
        policy_weight=args.policy_weight,
        endgame_positions=args.endgame_positions,
        endgame_stages=args.endgame_stages,
        num_workers=args.num_workers,
        resume=args.resume,
        deterministic=args.deterministic,
        rank_reg=args.rank_reg,
        draw_reg=args.draw_reg,
    )


if __name__ == "__main__":
    main()
