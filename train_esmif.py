# ----------------------------- train_esminv.py ---------------------------

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from esm import pretrained

from data_utils import get_dataloaders
from util import freeze_module, spearman_corrcoef, dpo_loss
from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
import torch.nn.functional as F 
from random import shuffle
import time


# ----------------------------- Model -------------------------------------
class ESMIFScore(nn.Module):

    def __init__(self, model_name: str = "esm_if1_gvp4_t16_142M"):
        super().__init__()
        self.backbone, self.alphabet = esm_if1_gvp4_t16_142M_UR50()
        freeze_module(self.backbone.encoder)


    def forward(self, coords, mask, seqs):
        device = next(self.parameters()).device
        coords = coords.to(device)
        mask   = mask.to(device)
        
        coords = F.pad(coords, (0, 0, 0, 0, 1, 1), value=float('inf'))
        mask = F.pad(mask, (1, 1), value=False)

        # tokenization ensures consistent length
        seqs = [("", s) for s in seqs]  # Add an empty label for each sequence
        # print("seqs:", seqs[1])
        batch_tokens = self.alphabet.get_batch_converter()(seqs)[2].to(device) #torch.Size([16, 761]), padded one position at the beginning
        
        # Fix: Manually append <eos> token if missing (ESM-IF batch_converter might not add it)
        if batch_tokens.shape[1] > 0 and batch_tokens[0, -1] != self.alphabet.eos_idx:
            eos = torch.full((batch_tokens.shape[0], 1), self.alphabet.eos_idx, device=device)
            batch_tokens = torch.cat([batch_tokens, eos], dim=1)

        # print("batch_tokens:", batch_tokens.shape)

        # confidence: 1.0 for valid residues, -1.0 for padding/invalid (consistent with official esm-if)
        confidence = mask.to(dtype=coords.dtype)
        confidence[~mask] = -1.0
        
        logits, _ = self.backbone(
            coords=coords,
            padding_mask=~mask,                  # pad==1
            prev_output_tokens=batch_tokens[:, :-1],
            confidence=confidence,
            # features_only=False,
        )

        # -------- Compute the log probability of each token given the preceding tokens. -----------------
        log_probs = F.log_softmax(logits, dim=1)                 # (B,V,L)
        ll_token  = log_probs.gather(1, batch_tokens[:,1:].unsqueeze(1)).squeeze(1)
        
        # Use mask[:, 1:] to match ll_token (B, L+1)
        loss_mask = mask[:, 1:]
        ll_token  = ll_token * loss_mask.to(ll_token.dtype)           
        ll_seq    = ll_token.sum(-1) / loss_mask.sum(-1)               # (B,)

        return ll_seq 
       


def train_epoch(model, loader, optimizer, device, beta: float, margin: float, higher_is_better: bool, val_loader=None, best_val_corr=-math.inf, save_path=None):

    # Fix: Reset _future_mask to avoid "Inference tensors cannot be saved for backward" error
    if hasattr(model.backbone.decoder, '_future_mask'):
        model.backbone.decoder._future_mask = torch.empty(0)

    model.train()
    running_loss, n_steps = 0.0, 0
    eval_every = 100

    for batch in loader:
        preds   = model(batch["coords"], batch["mask"], batch["seqs"])   # (B,)
        labels  = batch["labels"].to(preds.device)                       # (B,)
        antigens = batch["antigen"]                                      # list[str]

        # -------- ① Group sample indices by antigen -------------------
        ag2idx = {}
        for idx, ag in enumerate(antigens):
            ag2idx.setdefault(ag, []).append(idx)

        chosen_s, rejected_s = [], []

        # -------- ② Shuffle within each group, then pair "offsets" -----------------------
        #    And select (chosen, rejected) based on binding score

        for idxs in ag2idx.values():           # idxs: List of indices for the same antigen
            if len(idxs) < 2:                  # Only 1 sample -> cannot pair
                continue

            idxs = idxs.copy()
            shuffle(idxs)                      # Shuffle in place

            # Let the partner be a cyclic right shift of the shuffled sequence to ensure not comparing with itself
            partner = idxs[1:] + idxs[:1]

            idxs_t     = torch.tensor(idxs,     device=preds.device)
            partner_t  = torch.tensor(partner,  device=preds.device)

            score_i = labels[idxs_t]
            score_j = labels[partner_t]

            # Determine which is better based on user specified direction
            better = score_i >= score_j if higher_is_better else score_i <= score_j

            # Assemble chosen / rejected scores
            chosen_s.append( torch.where(better, preds[idxs_t],  preds[partner_t]) )
            rejected_s.append(torch.where(better, preds[partner_t], preds[idxs_t]))

        # If no pairs are formed in the entire batch (rare), skip directly
        if not chosen_s:
            continue

        chosen_scores   = torch.cat(chosen_s)    # (N_pair,)
        rejected_scores = torch.cat(rejected_s)  # (N_pair,)

        loss = dpo_loss(chosen_scores, rejected_scores, beta, margin)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_steps += 1

        # Validate every N batches, save the best model based on the highest average correlation coefficient on the validation set
        if val_loader is not None and n_steps % eval_every == 0:
            val_corr, corr_dict = eval_epoch(model, val_loader, device)
            print(f"[Step {n_steps}] ρ {val_corr:.3f}")
            print(corr_dict)
            # print('Current training loss:', loss.item())

            if val_corr > best_val_corr and save_path is not None:
                print(f"New best model found (ρ={val_corr:.3f}), saving to {save_path}")
                torch.save(model.state_dict(), save_path)
                best_val_corr = val_corr

            # Fix: Reset _future_mask to avoid "Inference tensors cannot be saved for backward" error
            if hasattr(model.backbone.decoder, '_future_mask'):
                model.backbone.decoder._future_mask = torch.empty(0)

            model.train()


    return running_loss / max(n_steps, 1), best_val_corr


@torch.inference_mode()
def eval_epoch(model, loader, device):
    
    model.eval()
    all_preds, all_labels, all_antigens = [], [], []

    total_loss = 0.0
    for batch in loader:
        labels = batch["labels"].to(device)
        preds  = model(batch["coords"], batch["mask"], batch["seqs"])

        # # Accumulate loss (weighted by number of samples)
        # total_loss += criterion(preds, labels).item() * labels.size(0)

        # Collect results, group later
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_antigens.extend(batch["antigen"])

    preds   = torch.cat(all_preds)
    labels  = torch.cat(all_labels)

    # ---- Calculate Spearman correlation grouped by antigen ----
    from collections import defaultdict
    group = defaultdict(list)
    for p, y, ag in zip(preds, labels, all_antigens):
        group[ag].append((p, y))

    # corr_list = []
    corr_dict = {}
    for ag, pairs in group.items():
        if len(pairs) < 2:        # Skip if there is only 1 mutant, correlation coefficient is meaningless
            continue
        p_stack = torch.stack([p for p, _ in pairs])
        y_stack = torch.stack([y for _, y in pairs])
        corr = spearman_corrcoef(p_stack, y_stack)
        # corr_list.append(corr)
        corr_dict[ag] = corr     

    # mean_corr = sum(corr_dict) / len(corr_dict) if corr_dict else 0.0
    mean_corr = sum(corr_dict.values()) / len(corr_dict) if corr_dict else 0.0
    # mean_loss = total_loss / len(preds)

    return mean_corr, corr_dict


# ------------------------------ Main -------------------------------------
def main(args):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = get_dataloaders(
        args.meta_json, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=777
    )

    model = ESMIFScore(args.model).to(device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "runs"))

    best_val_corr = -math.inf
    for epoch in range(1, args.epochs + 1):
        save_path = out_dir / "best_model.pt"
        start = time.time()
        tr_loss, best_val_corr = train_epoch(
            model, train_loader, optimizer, device,
            beta=args.beta,
            margin=args.margin,
            higher_is_better=args.higher_is_better,
            val_loader=val_loader,
            best_val_corr=best_val_corr,
            save_path=save_path
        )
        end = time.time()
        print(f"Running time: {end - start:.4f} seconds")
        val_corr, val_corr_dict = eval_epoch(model, val_loader, device)


        print(
            f"[Epoch {epoch:03d}] "
            f"Train MSE {tr_loss:.4f}| "
            f"Val ρ {val_corr:.3f}"
        )
        # Print correlation coefficients for each antigen (optional)
        for ag, r in sorted(val_corr_dict.items()):
            print(f"    {ag:<12s}:  ρ = {r: .3f}")

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), out_dir / "best_model.pt")

    # ------------------------- Final Test -------------------------
    model.load_state_dict(torch.load(out_dir / "best_model.pt"))
    te_corr, te_corr_dict = eval_epoch(model, test_loader, device)
    print(f"\n[Test] Spearman ρ = {te_corr:.3f}")
    # Print correlation coefficients for each antigen (optional)
    for ag, r in sorted(te_corr_dict.items()):
        print(f"    {ag:<12s}:  ρ = {r: .3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--higher_is_better", action="store_true", default=True,
                    help="If higher binding score indicates better, keep default True; "
                         "If lower score indicates better (e.g. ΔG), pass --higher_is_better False")
    parser.add_argument("--meta_json", type=str, default="../data/metadata.json", help="metadata.json (contains pdb paths & affinity csvs)")
    parser.add_argument("--pdb_dir", type=str, default="../data/complex_structure", help="PDB folder")
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--model",   type=str, default="esm_if1_gvp4_t16_142M")
    parser.add_argument("--epochs",  type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr",      type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature β")
    parser.add_argument("--margin", type=float, default=0.1, help="SimPO target reward margin δ")

    
    args = parser.parse_args()

    main(args)
# -------------------------------------------------------------------------
