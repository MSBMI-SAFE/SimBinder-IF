# --------------------------- data_utils_full.py --------------------------
"""
Self-contained Data Processing:
1. Read metadata.json → Cache coordinates/sequences for each antigen
2. Reuse ESM-IF official _concatenate_coords / _concatenate_seqs, insert 10 × NaN residues between chains; place heavy chain first
3. CSV contains per-example heavy_chain, light_chain, binding_score
4. Return coords / mask / full_seq directly usable by ESM-IF
"""

# ===== Python std / 3rd-party =================================================
import json, os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import biotite.structure as bst
from biotite.structure.io import pdb, pdbx
from biotite.structure.residues import get_residues
from biotite.sequence import ProteinSequence
import pickle
# ==============================================================================


def get_atom_coords_residuewise(atoms, struct: bst.AtomArray):
    """Extract backbone coordinates and align them per residue into shape (L, 3, 3)."""
    def filt(s, axis=None):
        filt_mat = np.stack([s.atom_name == name for name in atoms], axis=1)
        idx = filt_mat.argmax(0)
        coords = s[idx].coord
        coords[filt_mat.sum(0) == 0] = np.nan
        return coords
    return bst.apply_residue_wise(struct, struct, filt)


def load_structure(fpath, chains=None):
    """Read PDB/CIF and retain only the backbone atoms and specified chains."""
    if fpath.endswith(".cif"):
        with open(fpath) as fh:
            pdbxf = pdbx.PDBxFile.read(fh)
        struct = pdbx.get_structure(pdbxf, model=1)
    else:               # .pdb
        with open(fpath) as fh:
            pdbf = pdb.PDBFile.read(fh)
        struct = pdb.get_structure(pdbf, model=1)

    struct = struct[bst.filter_backbone(struct)]
    all_c = bst.get_chains(struct)
    if chains is None: chains = all_c
    if isinstance(chains, str): chains = [chains]
    for c in chains:
        if c not in all_c:
            raise ValueError(f"chain {c} not found in {fpath}")
    struct = struct[[a.chain_id in chains for a in struct]]
    return struct


def extract_coords_from_structure(struct: bst.AtomArray):
    coords = get_atom_coords_residuewise(["N", "CA", "C"], struct)
    res_ids = get_residues(struct)[1]
    seq = "".join([ProteinSequence.convert_letter_3to1(r) for r in res_ids])
    return coords, seq


def load_complex_coords(fpath: str, chains: List[str]):
    """Returns dict(chain_id → coords/seq)."""
    struct = load_structure(fpath, chains)
    coords, seqs = {}, {}
    for cid in chains:
        chain_struct = struct[struct.chain_id == cid]
        coords[cid], seqs[cid] = extract_coords_from_structure(chain_struct)
    return coords, seqs
# -----------------------------------------------------------------------------


# ------------- multichain_util.py -- Copy the official implementation. ----------------------------
def _concatenate_coords(coords, target_chain_id, padding_length=10, order=None):
    pad = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    if order is None:
        order = [target_chain_id] + [c for c in coords if c != target_chain_id]
    coords_out, tags_out = [], []
    for idx, cid in enumerate(order):
        if idx > 0:
            coords_out.append(pad)
            tags_out.append(["pad"] * padding_length)
        coords_out.append(coords[cid])
        tags_out.append([cid] * coords[cid].shape[0])
    return np.concatenate(coords_out, 0), np.concatenate(tags_out, 0).ravel()


def _concatenate_seqs(native_seqs, target_seq, target_chain_id,
                      padding_length=10, order=None):
    if order is None:
        order = [target_chain_id] + [c for c in native_seqs if c != target_chain_id]
    seqs_out = []
    for idx, cid in enumerate(order):
        if idx > 0:
            seqs_out.append(["<mask>"] * (padding_length - 1) + ["<cath>"])
        seqs_out.append(list(target_seq if cid == target_chain_id else native_seqs[cid]))
    return "".join(np.concatenate(seqs_out, 0))
# -----------------------------------------------------------------------------


# --------------------- The data pipeline of this project.----------------------------------------
# Each CSV has three columns: heavy_chain_seq, light_chain_seq, binding_score
COL_MUT_HEAVY = "heavy_chain_seq"
COL_MUT_LIGHT = "light_chain_seq"
COL_SCORE = "binding_score"


def build_antigen_cache(meta_json: str):
    meta = json.load(open(meta_json))
    cache = {}
    for ag, info in meta.items():
        chains = [info["heavy_chain"], info["light_chain"], *info["antigen_chains"]]
        coords_d, seqs_d = load_complex_coords(info["pdb_path"], chains)

        coords_cat, tags = _concatenate_coords(
            coords_d, target_chain_id=info["heavy_chain"], padding_length=10
        )
        cache[ag] = dict(
            coords=torch.tensor(coords_cat, dtype=torch.float32),
            # mask=~torch.isnan(coords_cat[:, 0, 0]),
            mask=torch.tensor(~np.isnan(coords_cat[:, 0, 0]), dtype=torch.bool),
            native_seqs=seqs_d,
            heavy_id=info["heavy_chain"],
            light_id=info["light_chain"],
        )
    return meta, cache


class Example:
    __slots__ = ("antigen", "cache_key", "mut_heavy", "mut_light", "score")

    def __init__(self, antigen: str, cache_key: str,
                 mut_heavy: str, mut_light: str, score: float):
        # antigen: Used for train/val/test split & group evaluation (e.g., 1mlc_LC, 5a12_ang2, etc.)
        # cache_key: Used to look up original coordinates in metadata.json structure cache (e.g., 1mlc, 5a12_ang2, etc.)
        self.antigen = antigen
        self.cache_key = cache_key
        self.mut_heavy = mut_heavy
        self.mut_light = mut_light
        self.score = score


class AffinityDS(Dataset):
    def __init__(self, samples: List[Example], cache):
        self.s = samples; self.c = cache
        df = pd.DataFrame({"ag": [x.antigen for x in samples],
                           "y":  [x.score   for x in samples]})
        st = df.groupby("ag")["y"].agg(["mean", "std"]).replace({"std": {0:1}})
        self.stats = st.to_dict("index")

    def __len__(self): return len(self.s)

    def __getitem__(self, i):
        ex = self.s[i]
        c  = self.c[ex.cache_key]  # Coordinates are from the original antigen.
        native_seqs = dict(c["native_seqs"])  # copy a sequence dictionary

        # Sequences for both heavy / light chains come from the mutant sequences in the CSV
        heavy_id = c["heavy_id"]
        light_id = self._get_light_chain_id(c)
        native_seqs[heavy_id] = ex.mut_heavy
        native_seqs[light_id] = ex.mut_light

        # Still use heavy chain as target_chain_id, other logic remains the same
        full_seq = _concatenate_seqs(
            native_seqs, native_seqs[heavy_id], heavy_id, padding_length=10
        )
        mu, sig = self.stats[ex.antigen]["mean"], self.stats[ex.antigen]["std"]
        return dict(
            coords=c["coords"],
            mask=c["mask"],
            seq=full_seq,
            label=torch.tensor(ex.score, dtype=torch.float32),
            antigen=ex.antigen,
        )

    @staticmethod
    def _get_light_chain_id(cache_entry):
        """Directly use the light_chain ID given in metadata."""
        return cache_entry["light_id"]


def _pad(t, L, pad_val):
    if t.shape[0] == L: return t
    pad = t.new_full((L - t.shape[0],) + t.shape[1:], pad_val)
    return torch.cat([t, pad], 0)


def collate(batch):
    Lmax = max(b["coords"].shape[0] for b in batch)
    coords = torch.stack([_pad(b["coords"], Lmax, np.nan)  for b in batch])
    masks  = torch.stack([_pad(b["mask"],  Lmax, False)    for b in batch])
    seqs   = [b["seq"] for b in batch]
    labels = torch.stack([b["label"] for b in batch])
    antigens = [b["antigen"] for b in batch]
    return dict(coords=coords, mask=masks, seqs=seqs, labels=labels, antigen=antigens)



def get_dataloaders(meta_json: str, batch_size=4, num_workers=4, seed=42):
    """
    Split data by antigen name (manual train/val/test) and save/load split from file.
    """
    meta, cache = build_antigen_cache(meta_json)
    samples: List[Example] = []

    # Define antigen sets
    train_antigens = {"4fqi_h1_benchmarking_data", "3gbn_h1_benchmarking_data", "aayl51", "2fjg", "1mlc", "1mlc_LC", "aayl52_LC", "g6_LC"}
    test_antigens  = {"4fqi_h3_benchmarking_data", "3gbn_h9_benchmarking_data", "aayl49", "aayl49_ML", "1n8z", "5a12_ang2", "aayl50_LC"}

    # Gather all valid examples
    for ag, info in meta.items():
        multi_csv = len(info["affinity_data"]) > 1
        for csv_fp in info["affinity_data"]:
            df = pd.read_csv(csv_fp)
            dataset_id = Path(csv_fp).stem if multi_csv else ag
            for _, r in df.iterrows():
                antigen_id = dataset_id
                if antigen_id in train_antigens or antigen_id in test_antigens:
                    samples.append(
                        Example(
                            antigen_id,
                            ag,
                            str(r[COL_MUT_HEAVY]),
                            str(r[COL_MUT_LIGHT]),
                            float(r[COL_SCORE]),
                        )
                    )

    # Determine saved split path
    split_dir = Path(meta_json).parent
    split_file = split_dir / "antigen_split_manual.pkl"

    # Load cached split if it exists
    if split_file.exists():
        with open(split_file, "rb") as f:
            split = pickle.load(f)
        tr_ids = set(split["train"])
        va_ids = set(split["val"])
        te_ids = set(split["test"])
        print(f"[Info] Loaded cached antigen split from: {split_file}")
    else:
        # Create new split
        from collections import defaultdict
        grouped = defaultdict(list)
        for i, s in enumerate(samples):
            if s.antigen in train_antigens:
                grouped[s.antigen].append(i)

        rng = np.random.default_rng(seed)
        tr_ids, va_ids, te_ids = set(), set(), set()

        for ag, idxs in grouped.items():
            idxs = np.array(idxs)
            rng.shuffle(idxs)
            n_val = max(1, int(0.2 * len(idxs)))
            va_ids.update(idxs[:n_val])
            tr_ids.update(idxs[n_val:])

        for i, s in enumerate(samples):
            if s.antigen in test_antigens:
                te_ids.add(i)

        # Save to disk
        with open(split_file, "wb") as f:
            pickle.dump({"train": list(tr_ids), "val": list(va_ids), "test": list(te_ids)}, f)
        print(f"[Info] Saved new antigen split to: {split_file}")

    # Construct DataLoaders
    dl_kw = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
    return (
        DataLoader(AffinityDS([samples[i] for i in tr_ids], cache), shuffle=True,  **dl_kw),
        DataLoader(AffinityDS([samples[i] for i in va_ids], cache), shuffle=False, **dl_kw),
        DataLoader(AffinityDS([samples[i] for i in te_ids], cache), shuffle=False, **dl_kw),
    )

# --------------------------------------------------------------------------
