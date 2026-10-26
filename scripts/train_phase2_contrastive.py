"""
Phase 2: Supervised Contrastive Embedding + TC Number Retrieval  (v12)
===========================================================================
训练 + 续训 + 同步 + post-train (用 df_tr 避免 data leakage)
"""

import os
import argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import random
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import esm

# ── Paths ─────────────────────────────────────────────────────────────────────
ESM2_PATH   = os.environ.get("ESM2_PATH", "/data2/chao/subcellular/new_project/models/esm2/checkpoints/esm2_t33_650M_UR50D.pt")  # env var override for 3B
PHASE1_CKPT = "/data2/chao/transport/phase1_family/best_phase1.pth"
DATA_CSV    = "/data2/chao/transport/data/stage2_tc.csv"  # TCDB protein CSV with sequence + tc_number
OUTPUT_DIR  = Path(os.environ.get("PHASE2_OUTPUT_DIR", "/data2/chao/transport/phase2_contrastive_3b"))  # env var override; default 3B dir (NOT the 650M dir to avoid overwrite)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESUME_CKPT = OUTPUT_DIR / "latest_phase2.pth"
BEST_CKPT   = OUTPUT_DIR / "best_phase2.pth"


# ── Auto GPU selection ────────────────────────────────────────────────────────
def select_idle_gpus(n: int = 2) -> str:
    """
    Use nvidia-smi to find the `n` least-loaded GPUs and return their
    indices as a comma-separated string suitable for CUDA_VISIBLE_DEVICES.
    If nvidia-smi is unavailable or fails, returns "" (caller falls back to default).
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return ""
        # Sort by memory.used ascending → least loaded first
        gpu_usage = sorted(
            (line.split(",") for line in result.stdout.strip().split("\n") if line),
            key=lambda x: int(x[1]),
        )
        if len(gpu_usage) < n:
            return ""
        selected = ",".join(g[0] for g in gpu_usage[:n])
        return selected
    except Exception as e:
        print(f"[Auto-GPU] nvidia-smi query failed: {e}")
        return ""

# ── Hyperparameters ───────────────────────────────────────────────────────────
MAX_LEN           = 800
BATCH_SIZE        = 12
N_VIEWS           = 2
EPOCHS            = 25
ESM_LR            = 1e-6
PROJ_LR           = 1e-4
WEIGHT_DECAY      = 0.01
TEMPERATURE       = 0.07
EMBED_DIM         = 512
UNFREEZE_LAYERS   = 6
SEED              = 42
VALID_AA          = set("ACDEFGHIKLMNPQRSTVWXY")
NUM_WORKERS       = 8
PREFETCH_FACTOR   = 4


# ── Dataset ───────────────────────────────────────────────────────────────────
class ContrastiveDataset(Dataset):
    def __init__(self, seqs, tc_numbers, tc_families, family_to_int,
                 alphabet, max_len=MAX_LEN, n_views=N_VIEWS, augment=True):
        self.tc_numbers    = list(tc_numbers)
        self.tc_families   = list(tc_families)
        self.family_to_int = family_to_int
        self.n_views       = n_views
        self.augment       = augment
        self.x_idx         = alphabet.get_idx("X")
        self.eos_idx       = alphabet.eos_idx
        self.pad_idx       = alphabet.padding_idx

        self.tokens = []
        for seq in seqs:
            seq = str(seq)[: max_len - 2]
            toks = [alphabet.cls_idx]
            for aa in seq.upper():
                t = alphabet.get_idx(aa)
                toks.append(t if t != alphabet.unk_idx else self.x_idx)
            toks.append(alphabet.eos_idx)
            toks = toks + [self.pad_idx] * (max_len - len(toks))
            self.tokens.append(toks[:max_len])

    def __len__(self):
        return len(self.tokens)

    def _augment(self, toks, mask_prob=0.10):
        out = list(toks)
        for i in range(1, len(out)):
            t = out[i]
            if t == self.eos_idx or t == self.pad_idx:
                break
            if random.random() < mask_prob:
                out[i] = self.x_idx
        return out

    def __getitem__(self, idx):
        base = self.tokens[idx]
        if self.augment:
            views = [torch.tensor(self._augment(base), dtype=torch.long)
                     for _ in range(self.n_views)]
        else:
            views = [torch.tensor(base, dtype=torch.long)
                     for _ in range(self.n_views)]
        return {
            "views":     torch.stack(views, dim=0),
            "fam_label": torch.tensor(
                self.family_to_int.get(self.tc_families[idx], 0),
                dtype=torch.long),
            "tc_number": self.tc_numbers[idx],
        }


def contrastive_collate(batch):
    views      = torch.stack([b["views"] for b in batch], dim=0)
    labels     = torch.tensor([b["fam_label"] for b in batch])
    tc_numbers = [b["tc_number"] for b in batch]
    return {"views": views, "fam_label": labels, "tc_number": tc_numbers}


# ── Model ─────────────────────────────────────────────────────────────────────
class ContrastiveEncoder(nn.Module):
    def __init__(self, esm_model, embed_dim=EMBED_DIM,
                 hidden=None, unfreeze_n=UNFREEZE_LAYERS):
        # Default `hidden` to ESM-2's actual embed dim (1280 for 650M, 2560 for 3B)
        if hidden is None:
            hidden = esm_model.embed_dim
        super().__init__()
        self.esm = esm_model

        for p in self.esm.parameters():
            p.requires_grad = False
        for layer in self.esm.layers[-unfreeze_n:]:
            for p in layer.parameters():
                p.requires_grad = True
        for p in self.esm.emb_layer_norm_after.parameters():
            p.requires_grad = True

        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _pool(self, tok_emb, input_ids):
        pad_idx = 1
        mask = (input_ids != pad_idx)
        mask[:, 0] = False
        mf   = mask.unsqueeze(-1).float()
        mean = (tok_emb * mf).sum(1) / mf.sum(1).clamp(min=1)
        cls  = tok_emb[:, 0, :]
        return torch.cat([cls, mean], dim=-1)

    def forward(self, input_ids):
        # Use the last layer of the ESM-2 model (works for both 650M and 3B)
        last_layer = self.esm.num_layers
        out    = self.esm(input_ids, repr_layers=[last_layer], return_contacts=False)
        pooled = self._pool(out["representations"][last_layer], input_ids)
        z      = self.proj(pooled)
        return F.normalize(z, dim=-1)

    def encode_views(self, views):
        B, V, L = views.shape
        flat    = views.view(B * V, L)
        z_flat  = self.forward(flat)
        return z_flat.view(B, V, -1)


# ── SupCon Loss ───────────────────────────────────────────────────────────────
class SupConLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.T = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        n      = embeddings.size(0)
        sim = torch.mm(embeddings, embeddings.T) / self.T

        labels   = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        self_mask = torch.eye(n, device=device)
        pos_mask  = pos_mask - self_mask

        sim_exp   = torch.exp(sim) * (1 - self_mask)
        log_denom = torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
        log_pos   = sim - log_denom
        n_pos     = pos_mask.sum(dim=1).clamp(min=1)
        loss      = -(pos_mask * log_pos).sum(dim=1) / n_pos
        return loss.mean()


# ── Data prep ─────────────────────────────────────────────────────────────────
def prepare_data():
    df = pd.read_csv(DATA_CSV)
    df = df.dropna(subset=["sequence", "tc_number"])
    df["sequence"]  = df["sequence"].str.upper()
    df["tc_family"] = df["tc_number"].str.extract(r"^(\d+\.[A-Z]+\.\d+)")

    def ok(s):
        s = str(s)
        return (50 <= len(s) <= 5000 and
                sum(c not in VALID_AA for c in s) / len(s) <= 0.05)
    df = df[df["sequence"].apply(ok)].copy()

    families      = sorted(df["tc_family"].dropna().unique())
    family_to_int = {f: i for i, f in enumerate(families)}
    int_to_family = {i: f for f, i in family_to_int.items()}

    try:
        tr_idx, va_idx = train_test_split(
            df.index, test_size=0.15,
            stratify=df["tc_family"].fillna("UNK"), random_state=SEED)
    except Exception:
        tr_idx, va_idx = train_test_split(df.index, test_size=0.15, random_state=SEED)

    df_tr = df.loc[tr_idx].copy()
    df_va = df.loc[va_idx].copy()

    print(f"[Data] Train: {len(df_tr)}  Val: {len(df_va)}")
    print(f"[Data] Unique TC numbers: {df['tc_number'].nunique()}")
    print(f"[Data] Unique TC families: {len(families)}")
    return df_tr, df_va, family_to_int, int_to_family


# ── Build retrieval index ────────────────────────────────────────────────────
@torch.no_grad()
def build_retrieval_index(model, df, alphabet, device, batch_size=16,
                          use_bf16=True):
    try:
        import faiss
    except ImportError:
        faiss = None

    model.eval()
    seqs        = df["sequence"].values
    tc_numbers  = df["tc_number"].values
    tc_families = df["tc_family"].fillna("UNK").values

    amp_ctx = autocast(dtype=torch.bfloat16) if use_bf16 else autocast()
    all_embs = []
    for start in tqdm(range(0, len(seqs), batch_size), desc="Building index"):
        batch_seqs = seqs[start: start + batch_size]
        batch_tokens = []
        for seq in batch_seqs:
            seq = str(seq)[: MAX_LEN - 2]
            tokens = [alphabet.cls_idx]
            for aa in seq.upper():
                t = alphabet.get_idx(aa)
                tokens.append(t if t != alphabet.unk_idx else alphabet.get_idx("X"))
            tokens.append(alphabet.eos_idx)
            pad = MAX_LEN - len(tokens)
            tokens = tokens + [alphabet.padding_idx] * pad
            batch_tokens.append(tokens[:MAX_LEN])
        input_ids = torch.tensor(batch_tokens, dtype=torch.long).to(device)
        with amp_ctx:
            emb = model(input_ids)
        all_embs.append(emb.cpu().float().numpy())

    all_embs = np.concatenate(all_embs, axis=0)
    np.save(OUTPUT_DIR / "tcdb_embeddings.npy", all_embs)
    pd.DataFrame({
        "tc_number": tc_numbers, "tc_family": tc_families, "sequence": seqs,
    }).to_csv(OUTPUT_DIR / "tcdb_index_metadata.csv", index=False)
    print(f"[Index] Saved {len(all_embs)} embeddings → tcdb_embeddings.npy")

    if faiss is not None:
        dim   = all_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.IndexIDMap(index)
        index.add_with_ids(all_embs.astype(np.float32),
                           np.arange(len(all_embs), dtype=np.int64))
        faiss.write_index(index, str(OUTPUT_DIR / "tcdb_faiss.index"))
        print("[Index] FAISS index saved")
    return all_embs, tc_numbers, tc_families


def evaluate_retrieval(query_embs, query_tc, ref_embs, ref_tc,
                       k_list=(1, 3, 5, 10)):
    sim     = query_embs @ ref_embs.T
    max_k   = max(k_list)
    topk_idx = np.argsort(-sim, axis=1)[:, :max_k]

    n = len(query_tc)
    results = {}
    for k in k_list:
        hits = family_hits = 0
        for i, tc in enumerate(query_tc):
            top_tc    = [ref_tc[j] for j in topk_idx[i, :k]]
            top_fam   = [t.rsplit(".", 2)[0] for t in top_tc]
            query_fam = tc.rsplit(".", 2)[0]
            if tc in top_tc:         hits += 1
            if query_fam in top_fam: family_hits += 1
        results[k] = (hits / n, family_hits / n)
    return results


# ── Save full checkpoint ──────────────────────────────────────────────────────
def save_checkpoint(path, epoch, best_loss, model, optimizer, scheduler, scaler,
                    family_to_int, device):
    state = {
        "epoch":           epoch,
        "best_loss":       best_loss,
        "model_state":     model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state":    scaler.state_dict() if scaler is not None else None,
        "rng_state":       torch.get_rng_state(),
        "cuda_rng_state":  torch.cuda.get_rng_state(device),
        "family_to_int":   family_to_int,
    }
    torch.save(state, path)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # ── Parse args (override hardcoded paths) ──
    global DATA_CSV, OUTPUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", default=DATA_CSV,
                    help=f"Path to training CSV (default: {DATA_CSV})")
    ap.add_argument("--output_dir", default=str(OUTPUT_DIR),
                    help=f"Output dir for trained model (default: {OUTPUT_DIR}). NOTE: default is 3B dir; pass --output_dir explicitly to override.")
    ap.add_argument("--esm2_path", default=ESM2_PATH,
                    help=f"Path to ESM-2 checkpoint (default: {ESM2_PATH}). Override with ESM-2 3B for larger model.")
    args, _ = ap.parse_known_args()
    DATA_CSV = args.data_csv
    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Args] data_csv = {DATA_CSV}")
    print(f"[Args] output_dir = {OUTPUT_DIR}")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.allow_tf32       = True

    # ★ 30 分钟 timeout, 够 build_retrieval_index 跑完
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device  = torch.device(f"cuda:{local_rank}")
    is_main = (local_rank == 0)

    torch.manual_seed(SEED + local_rank)
    random.seed(SEED + local_rank)
    np.random.seed(SEED + local_rank)

    use_bf16 = torch.cuda.is_bf16_supported()
    if is_main:
        print(f"[Config] AMP dtype: {'bf16' if use_bf16 else 'fp16'}")
        print(f"[Config] MAX_LEN={MAX_LEN}, BATCH_SIZE={BATCH_SIZE} per GPU")
        print(f"[Config] EPOCHS={EPOCHS}, NCCL timeout=30min")
        print(f"[Config] ESM: 直接调 native forward")
        print(f"[Config] Workers={NUM_WORKERS}, prefetch={PREFETCH_FACTOR}")

    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.esm2_path)
    print(f"[Config] ESM-2 embed_dim={esm_model.embed_dim}, num_layers={esm_model.num_layers}")

    df_tr, df_va, family_to_int, int_to_family = prepare_data()
    dist.barrier()

    def make_loader(df, augment):
        ds = ContrastiveDataset(
            df["sequence"].values, df["tc_number"].values,
            df["tc_family"].fillna("UNK").values,
            family_to_int, alphabet, augment=augment,
        )
        sampler = DistributedSampler(ds, shuffle=augment)
        return DataLoader(
            ds, batch_size=BATCH_SIZE, sampler=sampler,
            collate_fn=contrastive_collate,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=PREFETCH_FACTOR,
        )

    tr_loader = make_loader(df_tr, augment=True)
    va_loader = make_loader(df_va, augment=False)

    model = ContrastiveEncoder(esm_model).to(device)
    if Path(PHASE1_CKPT).exists():
        try:
            ckpt = torch.load(PHASE1_CKPT, map_location=device)
            esm_state = {k.replace("esm.", ""): v
                         for k, v in ckpt["model_state"].items()
                         if k.startswith("esm.")}
            model.esm.load_state_dict(esm_state, strict=False)
            if is_main:
                print(f"[Init] Loaded Phase 1 ESM weights from {PHASE1_CKPT}")
        except (RuntimeError, KeyError) as e:
            # ESM-2 dim mismatch (e.g. Phase 1 ckpt was 650M but current model is 3B)
            if is_main:
                print(f"[Init] Phase 1 ckpt incompatible ({type(e).__name__}: {str(e)[:200]})")
                print(f"[Init] Training Phase 2 from scratch (ESM-2 dim changed: 650M → 3B)")
    else:
        if is_main:
            print("[Init] Phase 1 checkpoint not found, training from scratch")

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    optimizer = optim.AdamW([
        {"params": model.module.esm.parameters(),  "lr": ESM_LR,
         "weight_decay": 0.0},
        {"params": model.module.proj.parameters(), "lr": PROJ_LR,
         "weight_decay": WEIGHT_DECAY},
    ])

    total_steps  = len(tr_loader) * EPOCHS
    warmup_steps = total_steps // 10
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * prog)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    criterion = SupConLoss(temperature=TEMPERATURE)
    scaler    = None if use_bf16 else GradScaler()

    start_epoch = 0
    best_loss   = float("inf")
    has_resume  = False

    if is_main and Path(RESUME_CKPT).exists():
        meta = torch.load(RESUME_CKPT, map_location="cpu")
        has_resume = True
        print(f"[Resume] {RESUME_CKPT.name} found, last_epoch={meta['epoch']}, "
              f"best_loss={meta['best_loss']:.4f}")

    flag = [int(has_resume)]
    dist.broadcast_object_list(flag, src=0)
    has_resume = bool(flag[0])

    if has_resume:
        ckpt = torch.load(RESUME_CKPT, map_location=device)
        start_epoch = ckpt["epoch"]
        best_loss   = ckpt["best_loss"]
        model.module.load_state_dict(ckpt["model_state"], strict=False)
        try:
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and ckpt.get("scaler_state"):
                scaler.load_state_dict(ckpt["scaler_state"])
            if ckpt.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state(ckpt["cuda_rng_state"], device=device)
            if ckpt.get("rng_state") is not None:
                torch.set_rng_state(ckpt["rng_state"])
            if is_main:
                print(f"[Resume] Full state restored, resuming from epoch {start_epoch}")
        except Exception as e:
            if is_main:
                print(f"[Resume] optimizer/scheduler load warning: {e}")
                print("[Resume] Continuing with model weights only")

    for epoch in range(start_epoch, EPOCHS):
        tr_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        amp_ctx = autocast(dtype=torch.bfloat16) if use_bf16 else autocast()
        pbar    = tqdm(tr_loader, desc=f"Ep{epoch+1:02d}",
                       disable=not is_main)

        for batch in pbar:
            views  = batch["views"].to(device, non_blocking=True)
            labels = batch["fam_label"].to(device, non_blocking=True)
            B, V, L = views.shape

            with amp_ctx:
                flat_views = views.view(B * V, L)
                embs       = model.module.forward(flat_views)
                rep_labels = labels.unsqueeze(1).expand(-1, V).reshape(-1)
                loss       = criterion(embs, rep_labels)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        avg_loss = epoch_loss / len(tr_loader)

        if is_main:
            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
                torch.save({
                    "epoch": epoch + 1, "loss": avg_loss,
                    "model_state": model.module.state_dict(),
                    "family_to_int": family_to_int,
                }, BEST_CKPT)
            print(f"\n[Ep{epoch+1:02d}] avg_loss={avg_loss:.4f}  "
                  f"{'✅ new best' if improved else ''}")

            save_checkpoint(RESUME_CKPT, epoch + 1, best_loss,
                            model.module, optimizer, scheduler, scaler,
                            family_to_int, device)
            print(f"  💾 Saved {RESUME_CKPT.name}")

        torch.cuda.synchronize()
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()

    # ★ Post-train (only rank 0)
    if is_main:
        print("\n[Post-train] Building TCDB retrieval index...")
        ckpt = torch.load(BEST_CKPT, map_location=device)
        model.module.load_state_dict(ckpt["model_state"])

        # ★ 关键: ref 用训练集 (不含 val), 避免 data leakage
        ref_embs, ref_tc, ref_fam = build_retrieval_index(
            model.module, df_tr, alphabet, device, use_bf16=use_bf16)

        print("\n[Eval] Computing retrieval metrics on val set...")
        val_embs_list, val_tc_list = [], []
        model.module.eval()
        amp_ctx = autocast(dtype=torch.bfloat16) if use_bf16 else autocast()
        for batch in tqdm(va_loader, desc="Encoding val"):
            inp = batch["views"][:, 0, :].to(device, non_blocking=True)
            with torch.no_grad(), amp_ctx:
                emb = model.module(inp)
            val_embs_list.append(emb.cpu().float().numpy())
            val_tc_list.extend(batch["tc_number"])
        val_embs = np.concatenate(val_embs_list, axis=0)

        results = evaluate_retrieval(val_embs, val_tc_list, ref_embs, ref_tc)
        print("  k   TC     Family")
        for k, (r_tc, r_fam) in results.items():
            marker = "  ✅" if r_tc >= 0.5 else ""
            print(f"  {k:2d}  {r_tc:.4f}  {r_fam:.4f}{marker}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()