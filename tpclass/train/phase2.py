"""Phase 2 training: contrastive TC-family retrieval.

The Phase 2 head is trained with a supervised contrastive objective (SupCon):
two augmented views of every protein are encoded, and proteins that share a
TC family are pulled together while different families are pushed apart.  The
resulting encoder maps proteins into a space in which TCDB references can be
retrieved by cosine similarity.

Hyper-parameters follow the released training script
(``train_phase2_contrastive.py``): 25 epochs, batch size 12, 2 views with 10 %
residue masking, temperature 0.07, embedding size 512, the last 6 ESM-2 layers
unfrozen with a 10x smaller learning rate than the projection head.

The checkpoint written here is the one
:func:`tpclass.core.embeddings.load_contrastive_head` expects: a dict with a
``model_state`` entry holding the encoder weights.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_EPOCHS = 25
DEFAULT_BATCH_SIZE = 12
DEFAULT_N_VIEWS = 2
DEFAULT_MASK_PROB = 0.10
DEFAULT_TEMPERATURE = 0.07
DEFAULT_EMBED_DIM = 512
DEFAULT_UNFREEZE_LAYERS = 6
DEFAULT_ESM_LR = 1e-6
DEFAULT_PROJ_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_LEN = 800
DEFAULT_SEED = 42


class SupConLoss(nn.Module):
    """Supervised contrastive loss (Khosla et al. style, as used in training)."""

    def __init__(self, temperature: float = DEFAULT_TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = embeddings.device
        n = embeddings.size(0)
        sim = torch.mm(embeddings, embeddings.T) / self.temperature

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T).float()
        self_mask = torch.eye(n, device=device)
        pos_mask = pos_mask - self_mask

        sim_exp = torch.exp(sim) * (1 - self_mask)
        log_denom = torch.log(sim_exp.sum(dim=1, keepdim=True) + 1e-8)
        log_pos = sim - log_denom
        n_pos = pos_mask.sum(dim=1).clamp(min=1)
        loss = -(pos_mask * log_pos).sum(dim=1) / n_pos
        return loss.mean()


class ContrastiveEncoder(nn.Module):
    """ESM-2 backbone + projection head used for Phase 2.

    The head consumes the concatenation of the ``cls`` vector and the
    mean-pooled token vector (pads and ``cls`` excluded from the mean) and
    returns an L2-normalised embedding.
    """

    def __init__(self, esm_model, embed_dim: int = DEFAULT_EMBED_DIM, unfreeze_n: int = DEFAULT_UNFREEZE_LAYERS):
        super().__init__()
        hidden = int(getattr(esm_model, "embed_dim", 0)) or int(
            esm_model.config.hidden_size
        )
        self.esm = esm_model
        for p in self.esm.parameters():
            p.requires_grad = False
        layers = getattr(self.esm, "layers", None)
        if layers is not None:
            for layer in layers[-unfreeze_n:]:
                for p in layer.parameters():
                    p.requires_grad = True
        norm = getattr(self.esm, "emb_layer_norm_after", None)
        if norm is not None:
            for p in norm.parameters():
                p.requires_grad = True
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, token_emb: torch.Tensor, input_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
        mask = (input_ids != pad_id).unsqueeze(-1).float()
        mask = mask.clone()
        mask[:, 0, :] = 0.0
        mean = (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        cls = token_emb[:, 0, :]
        pooled = torch.cat([cls, mean], dim=-1)
        return F.normalize(self.proj(pooled), dim=-1)


class ContrastiveDataset(torch.utils.data.Dataset):
    """Tokenised proteins with two randomly masked views each."""

    def __init__(
        self,
        sequences: Sequence[str],
        families: Sequence[str],
        alphabet,
        max_len: int = DEFAULT_MAX_LEN,
        n_views: int = DEFAULT_N_VIEWS,
        mask_prob: float = DEFAULT_MASK_PROB,
        augment: bool = True,
    ):
        self.n_views = n_views
        self.mask_prob = mask_prob
        self.augment = augment
        self.x_idx = alphabet.get_idx("X")
        self.eos_idx = alphabet.eos_idx
        self.pad_idx = alphabet.padding_idx
        self.families = list(families)
        self.family_to_int: Dict[str, int] = {}
        for fam in self.families:
            if fam not in self.family_to_int:
                self.family_to_int[fam] = len(self.family_to_int)

        self.tokens: List[List[int]] = []
        for seq in sequences:
            seq = str(seq).upper()[: max_len - 2]
            toks = [alphabet.cls_idx]
            for aa in seq:
                t = alphabet.get_idx(aa)
                toks.append(t if t != alphabet.unk_idx else self.x_idx)
            toks.append(alphabet.eos_idx)
            toks = toks + [self.pad_idx] * (max_len - len(toks))
            self.tokens.append(toks[:max_len])

    def __len__(self) -> int:
        return len(self.tokens)

    def _augment(self, toks: List[int]) -> List[int]:
        out = list(toks)
        for i in range(1, len(out)):
            if out[i] in (self.eos_idx, self.pad_idx):
                break
            if random.random() < self.mask_prob:
                out[i] = self.x_idx
        return out

    def __getitem__(self, idx: int) -> dict:
        base = self.tokens[idx]
        if self.augment:
            views = [torch.tensor(self._augment(base), dtype=torch.long) for _ in range(self.n_views)]
        else:
            views = [torch.tensor(base, dtype=torch.long) for _ in range(self.n_views)]
        return {
            "views": torch.stack(views, dim=0),
            "fam_label": torch.tensor(self.family_to_int.get(self.families[idx], 0), dtype=torch.long),
        }


def _collate(batch: List[dict]) -> dict:
    return {
        "views": torch.stack([b["views"] for b in batch], dim=0),
        "fam_label": torch.tensor([b["fam_label"] for b in batch]),
    }


def train_contrastive(
    sequences: Sequence[str],
    families: Sequence[str],
    esm2_path: str | Path,
    output_dir: str | Path,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_views: int = DEFAULT_N_VIEWS,
    temperature: float = DEFAULT_TEMPERATURE,
    embed_dim: int = DEFAULT_EMBED_DIM,
    unfreeze_layers: int = DEFAULT_UNFREEZE_LAYERS,
    esm_lr: float = DEFAULT_ESM_LR,
    proj_lr: float = DEFAULT_PROJ_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    max_len: int = DEFAULT_MAX_LEN,
    device: Optional[str] = None,
    seed: int = DEFAULT_SEED,
    num_workers: int = 4,
) -> Path:
    """Train the Phase 2 contrastive encoder.

    Parameters
    ----------
    sequences : sequence of str
        Protein sequences.
    families : sequence of str
        TC family of each protein (the contrastive label).
    esm2_path : str or Path
        ESM-2 checkpoint (``.pt`` or HuggingFace directory).
    output_dir : str or Path
        Directory that receives ``best_phase2.pth``.
    device : str, optional
        Torch device.
    seed : int
        Random seed.

    Returns
    -------
    Path
        Path of the best checkpoint.
    """
    from torch.utils.data import DataLoader

    from tpclass.core.esm2 import ESM2Backbone

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    backbone = ESM2Backbone(path=esm2_path, device=device, max_len=max_len).load()
    alphabet = backbone.alphabet
    if alphabet is None:
        raise NotImplementedError(
            "Contrastive training requires a fair-esm checkpoint (.pt); the "
            "HuggingFace backbone is supported for inference only."
        )

    dataset = ContrastiveDataset(sequences, families, alphabet, max_len=max_len, n_views=n_views)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        drop_last=True,
    )
    logger.info(f"Phase 2 training on {len(dataset)} proteins, {len(dataset.family_to_int)} families")

    encoder = ContrastiveEncoder(
        backbone.model, embed_dim=embed_dim, unfreeze_n=unfreeze_layers
    ).to(device)
    criterion = SupConLoss(temperature=temperature)

    esm_params, proj_params = [], []
    for name, param in encoder.named_parameters():
        (proj_params if name.startswith("proj.") else esm_params).append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": esm_params, "lr": esm_lr},
            {"params": proj_params, "lr": proj_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_phase2.pth"
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        encoder.train()
        running, n_batches = 0.0, 0
        for batch in loader:
            views = batch["views"].to(device)  # (B, V, L)
            labels = batch["fam_label"].to(device)
            b, v, length = views.shape
            flat = views.reshape(b * v, length)
            # ESM2Backbone.encode returns (input_ids, token_emb).
            input_ids, token_emb = backbone.encode(flat)
            z = encoder(token_emb, input_ids, backbone.pad_id)
            z = z.reshape(b, v, -1)
            # SupCon over all views of all proteins in the batch.
            z = torch.cat([z[:, i, :] for i in range(v)], dim=0)
            labels = labels.repeat(v)
            loss = criterion(z, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
        scheduler.step()
        mean_loss = running / max(n_batches, 1)
        logger.info(f"epoch {epoch:3d}/{epochs}  supcon_loss={mean_loss:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(
                {
                    "model_state": encoder.state_dict(),
                    "epoch": epoch,
                    "loss": mean_loss,
                    "embed_dim": embed_dim,
                    "family_to_int": dataset.family_to_int,
                },
                best_path,
            )
            logger.info(f"  new best loss -> {best_path}")

    logger.info(f"Phase 2 training done. Best loss {best_loss:.4f} -> {best_path}")
    return best_path
