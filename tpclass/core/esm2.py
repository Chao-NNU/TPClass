"""ESM-2 backbone wrapper shared by Phase 1 and Phase 2.

TPClass supports the two ESM-2 checkpoint layouts that occur in practice:

* a **fair-esm** ``.pt`` file, e.g. ``models/esm2_t33_650M_UR50D.pt`` -- this is
  the layout the published Phase 1 / Phase 2 checkpoints were trained with;
* a **HuggingFace** ``transformers`` directory, e.g. ``models/esm2_3b/``.

Both layouts are wrapped in the same :class:`ESM2Backbone` so that Phase 1
(mean pooling) and Phase 2 (contrastive projection) can share a single
backbone instance and therefore a single forward pass over the input.

The two tokenizers produce interchangeable token ids: the ESM-2 byte-pair
vocabulary is identical, and ``cls``/``eos``/``pad`` are ids 0/2/1 in both
layouts.  This is what allows a Phase 1 backbone (e.g. 3B) and a Phase 2
backbone (e.g. 650M) to be mixed safely.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DEFAULT_MAX_LEN = 800
DEFAULT_BATCH_SIZE = 8

#: Dimensionality of the Phase 2 contrastive embedding.
EMBED_DIM = 512
#: Number of ESM-2 layers that were unfrozen when the contrastive head was trained.
UNFREEZE_LAYERS = 6


class ESM2Backbone:
    """Uniform ESM-2 interface over fair-esm and HuggingFace checkpoints.

    Parameters
    ----------
    path : str or Path
        Either a fair-esm ``.pt`` file or a HuggingFace model directory.
    device : str
        Torch device, e.g. ``"cuda"`` or ``"cpu"``.
    repr_layer : int, optional
        Transformer layer whose output is used.  Defaults to the **last** layer.
        The published 650M Phase 2 checkpoint was trained with
        ``repr_layers=[33]``, which is the last layer of the 33-layer 650M
        model, so the default is the correct choice for it.
    max_len : int
        Maximum residue count fed to the model.
    batch_size : int
        Sequences per forward pass.
    """

    def __init__(
        self,
        path: str | Path,
        device: str = "cuda",
        repr_layer: Optional[int] = None,
        max_len: int = DEFAULT_MAX_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        self.path = Path(path)
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.repr_layer = repr_layer
        self.is_hf = self.path.is_dir()

        self._model = None
        self._alphabet = None
        self._tokenizer = None
        self._embed_dim = None
        self._num_layers = None
        self._pad_id = None
        self._eos_id = None
        self._cls_id = None

    # ------------------------------------------------------------------ load
    def load(self) -> "ESM2Backbone":
        """Load the underlying model (idempotent)."""
        if self._model is not None:
            return self

        if self.is_hf:
            from transformers import AutoModel, AutoTokenizer

            logger.info(f"Loading ESM-2 (HuggingFace) from {self.path}")
            self._tokenizer = AutoTokenizer.from_pretrained(self.path)
            self._model = AutoModel.from_pretrained(
                self.path, torch_dtype=torch.float32
            ).to(self.device).eval()
            self._embed_dim = int(self._model.config.hidden_size)
            self._num_layers = int(self._model.config.num_hidden_layers)
            self._pad_id = int(self._tokenizer.pad_token_id)
            self._eos_id = int(self._tokenizer.eos_token_id)
            self._cls_id = int(self._tokenizer.cls_token_id)
        else:
            import esm

            logger.info(f"Loading ESM-2 (fair-esm) from {self.path}")
            model, alphabet = esm.pretrained.load_model_and_alphabet_local(str(self.path))
            self._model = model.to(self.device).eval()
            self._alphabet = alphabet
            self._embed_dim = int(model.embed_dim)
            self._num_layers = int(model.num_layers)
            self._pad_id = int(alphabet.padding_idx)
            self._eos_id = int(alphabet.eos_idx)
            self._cls_id = int(alphabet.cls_idx)

        if self.repr_layer is None:
            self.repr_layer = self._num_layers
        logger.info(
            f"ESM-2 ready: embed_dim={self.embed_dim}, num_layers={self.num_layers}, "
            f"repr_layer={self.repr_layer}, device={self.device}"
        )
        return self

    # ------------------------------------------------------------ properties
    @property
    def model(self):
        """The underlying torch module (exposed for fine-tuning)."""
        self.load()
        return self._model

    @property
    def embed_dim(self) -> int:
        self.load()
        return self._embed_dim

    @property
    def num_layers(self) -> int:
        self.load()
        return self._num_layers

    @property
    def pad_id(self) -> int:
        self.load()
        return self._pad_id

    @property
    def eos_id(self) -> int:
        self.load()
        return self._eos_id

    @property
    def cls_id(self) -> int:
        self.load()
        return self._cls_id

    @property
    def alphabet(self):
        """The fair-esm alphabet (for tokenising without the HF tokenizer).

        ``None`` for a HuggingFace backbone.
        """
        self.load()
        return self._alphabet

    # ------------------------------------------------------------ tokenizing
    def tokenize(self, seqs: Sequence[str]) -> torch.Tensor:
        """Tokenize a batch of sequences into padded id tensors.

        Returns
        -------
        torch.Tensor
            Shape ``(B, L)`` with ``L`` equal to ``max_len`` (fair-esm) or the
            longest sequence in the batch (HuggingFace).
        """
        self.load()
        if self.is_hf:
            enc = self._tokenizer(
                list(seqs),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len + 2,
            )
            return enc["input_ids"].to(self.device)

        alphabet = self._alphabet
        rows = []
        for seq in seqs:
            seq = seq[: self.max_len - 2]
            tokens = [alphabet.cls_idx]
            for aa in seq:
                idx = alphabet.get_idx(aa)
                tokens.append(idx if idx != alphabet.unk_idx else alphabet.get_idx("X"))
            tokens.append(alphabet.eos_idx)
            tokens = tokens + [alphabet.padding_idx] * (self.max_len - len(tokens))
            rows.append(tokens[: self.max_len])
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    # --------------------------------------------------------------- encoding
    def encode(self, seqs: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the transformer and return token-level embeddings.

        Returns
        -------
        (input_ids, token_emb)
            ``input_ids`` has shape ``(B, L)``; ``token_emb`` has shape
            ``(B, L, embed_dim)``.
        """
        self.load()
        input_ids = self.tokenize(seqs)
        if self.is_hf:
            attn_mask = (input_ids != self._pad_id).long()
            out = self._model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            token_emb = out.hidden_states[self.repr_layer]
        else:
            out = self._model(
                input_ids, repr_layers=[self.repr_layer], return_contacts=False
            )
            token_emb = out["representations"][self.repr_layer]
        return input_ids, token_emb

    def encode_all(
        self,
        seqs: Sequence[str],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """Encode every sequence, returning ids, token embeddings and input ids.

        Returns
        -------
        (names, token_emb, input_ids)
            ``names`` are the accessions in input order, ``token_emb`` has shape
            ``(N, L, D)`` and ``input_ids`` has shape ``(N, L)``.
        """
        self.load()
        bs = batch_size or self.batch_size
        embs: List[torch.Tensor] = []
        ids: List[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(seqs), bs):
                batch = list(seqs[start : start + bs])
                batch_ids, batch_emb = self.encode(batch)
                ids.append(batch_ids)
                embs.append(batch_emb.float().cpu())
        if not embs:
            empty = torch.zeros((0, 0, self.embed_dim))
            return [], empty, torch.zeros((0, 0), dtype=torch.long)
        # Different fair-esm / HF batches can have different widths; pad to max.
        max_len = max(t.shape[1] for t in embs)
        padded_emb, padded_ids = [], []
        for t_emb, t_ids in zip(embs, ids):
            if t_emb.shape[1] < max_len:
                pad = max_len - t_emb.shape[1]
                t_emb = F.pad(t_emb, (0, 0, 0, pad))
                t_ids = F.pad(t_ids, (0, pad), value=self.pad_id)
            padded_emb.append(t_emb)
            padded_ids.append(t_ids)
        return list(seqs), torch.cat(padded_emb, dim=0), torch.cat(padded_ids, dim=0)

    # ---------------------------------------------------------------- pooling
    @staticmethod
    def phase1_mean_pool(
        token_emb: torch.Tensor,
        input_ids: torch.Tensor,
        pad_id: int,
        eos_id: int,
    ) -> torch.Tensor:
        """Mean pooling used for the Phase 1 features.

        Pads and the ``eos`` token are excluded; the ``cls`` token is kept.
        This reproduces the released pipeline exactly.
        """
        mask = ((input_ids != pad_id) & (input_ids != eos_id)).unsqueeze(-1).float()
        return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def phase2_mean_pool(
        token_emb: torch.Tensor,
        input_ids: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        """Mean pooling used by the Phase 2 contrastive encoder.

        Pads and the ``cls`` token are excluded; the ``eos`` token is kept.
        This reproduces the released pipeline exactly.
        """
        mask = (input_ids != pad_id).unsqueeze(-1).float()
        mask = mask.clone()
        mask[:, 0, :] = 0.0
        return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1)

    @staticmethod
    def cls_vector(token_emb: torch.Tensor) -> torch.Tensor:
        """The ``cls`` token vector (first position)."""
        return token_emb[:, 0, :]


class ContrastiveProjection(nn.Module):
    """The Phase 2 contrastive projection head.

    Mirrors ``ContrastiveEncoder.proj`` of the released pipeline:
    ``Linear(2*hidden) -> LayerNorm -> GELU -> Linear(embed_dim) -> LayerNorm``.
    The head consumes the concatenation of the ``cls`` vector and the mean-pooled
    token vector and returns an L2-normalised embedding.
    """

    def __init__(self, hidden: int, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.proj = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(pooled), dim=-1)


def build_phase2_pooled(
    token_emb: torch.Tensor,
    input_ids: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """Concatenate the ``cls`` vector and the Phase 2 mean pool."""
    cls_vec = ESM2Backbone.cls_vector(token_emb)
    mean_vec = ESM2Backbone.phase2_mean_pool(token_emb, input_ids, pad_id)
    return torch.cat([cls_vec, mean_vec], dim=-1)


def load_esm2_backbone(
    path: str | Path,
    device: str = "cuda",
    repr_layer: Optional[int] = None,
    max_len: int = DEFAULT_MAX_LEN,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> ESM2Backbone:
    """Convenience factory returning a loaded :class:`ESM2Backbone`."""
    return ESM2Backbone(
        path=path,
        device=device,
        repr_layer=repr_layer,
        max_len=max_len,
        batch_size=batch_size,
    ).load()
