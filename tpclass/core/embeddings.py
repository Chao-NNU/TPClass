"""Feature extraction utilities for TPClass.

This module owns the three steps that turn protein sequences into model
inputs:

* :func:`read_fasta_ordered` -- parse a FASTA file while preserving input
  order (the order is the contract that keeps the ESM-2 features, the s4pred
  features and the output table aligned);
* :func:`extract_s4pred_features` -- the three s4pred secondary-structure
  fractions, with a warning-driven fallback to the released pipeline's
  ``(0.5, 0.2, 0.3)`` constant when s4pred is unavailable;
* :func:`extract_query_embeddings` -- Phase 2 contrastive embeddings for query
  proteins.

All of them follow the released pipeline (v3) exactly, including which tokens
are excluded from each mean-pooling operation.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from tpclass.core.esm2 import (
    ContrastiveProjection,
    ESM2Backbone,
    build_phase2_pooled,
)

logger = logging.getLogger(__name__)

#: Sequences outside this length range are rejected before encoding.
MIN_SEQ_LEN = 30
MAX_SEQ_LEN = 5000
#: Maximum tolerated fraction of non-standard residues.
MAX_BAD_FRACTION = 0.05
#: Valid amino-acid alphabet.
#: Matches the released pipeline: ``X`` (unknown) counts as a standard residue
#: there, so a protein is only rejected when more than 5 % of its residues are
#: outside this set.  The tokeniser maps any unknown residue to ``X`` anyway.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWXY")

#: Fallback secondary-structure feature used when s4pred is unavailable.
#: Matches the released pipeline, which substitutes this constant instead of
#: failing when s4pred cannot be run.
S4PRED_FALLBACK = (0.5, 0.2, 0.3)

_UNIPROT_HEADER_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|")


def read_fasta_ordered(fasta_path: str | Path) -> Tuple[List[str], Dict[str, str]]:
    """Read a FASTA file, preserving the input order.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the FASTA file.

    Returns
    -------
    (names, sequences)
        ``names`` is the list of accessions in file order; ``sequences`` maps
        accession -> residue string.  Duplicate accessions keep their first
        occurrence so that both lists stay in sync.
    """
    names: List[str] = []
    sequences: Dict[str, str] = {}
    acc: Optional[str] = None
    parts: List[str] = []

    def flush() -> None:
        if acc is not None and acc not in sequences:
            names.append(acc)
            sequences[acc] = "".join(parts)

    with open(fasta_path) as handle:
        for line in handle:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                flush()
                m = _UNIPROT_HEADER_RE.match(line)
                if m:
                    acc = m.group(1)
                else:
                    parts_hdr = line[1:].split()
                    acc = parts_hdr[0] if parts_hdr else f"seq_{len(sequences)}"
                parts = []
            else:
                parts.append(line)
    flush()
    return names, sequences


def read_fasta(fasta_path: str | Path) -> Dict[str, str]:
    """Read a FASTA file and return ``{accession: sequence}``."""
    return read_fasta_ordered(fasta_path)[1]


def is_valid_sequence(seq: str, min_len: int = MIN_SEQ_LEN, max_len: int = MAX_SEQ_LEN) -> bool:
    """Reject sequences that are too short, too long or mostly non-standard."""
    if not (min_len <= len(seq) <= max_len):
        return False
    bad = sum(1 for c in seq if c not in VALID_AA) / max(1, len(seq))
    return bad <= MAX_BAD_FRACTION


def parse_s4pred_horiz(filepath: str | Path) -> Optional[np.ndarray]:
    """Parse one s4pred ``.horiz`` file into H/E/C fractions.

    The released pipeline runs s4pred as an external program that writes one
    ``<name>.horiz`` file per sequence; the file holds a ``Pred:`` line with
    the per-residue secondary structure.  Reading the same files lets this
    package reuse the exact secondary structure that the released run used,
    instead of recomputing it with a possibly different build.

    Parameters
    ----------
    filepath : str or Path
        Path to a ``.horiz`` file.

    Returns
    -------
    np.ndarray or None
        ``(h, e, c)`` as float32, or ``None`` when the file carries no
        prediction.
    """
    chars: List[str] = []
    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("Pred:"):
                chars.extend(line[5:].strip().upper())
    if not chars:
        return None
    n = len(chars)
    h = sum(1 for c in chars if c == "H") / n
    e = sum(1 for c in chars if c == "E") / n
    c = sum(1 for c in chars if c == "C") / n
    return np.array([h, e, c], dtype=np.float32)


def scan_horiz_cache(
    cache_dir: str | Path | None,
    query_names: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Collect cached s4pred features for the given sequences.

    Each sequence is looked up as ``<cache_dir>/<name>.horiz``.  The
    ``pos__`` / ``neg__`` prefixes that the balanced held-out preparation adds
    to the FASTA headers are stripped for the lookup, so the same cache can be
    reused across runs with different header conventions -- this is the same
    normalisation the released pipeline performs.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory holding the ``.horiz`` files.  A missing directory yields an
        empty mapping.
    query_names : sequence of str
        Names to look up.

    Returns
    -------
    dict
        Mapping from name to its ``(h, e, c)`` feature vector.
    """
    if not cache_dir or not query_names:
        return {}
    outdir = Path(cache_dir)
    if not outdir.is_dir():
        return {}
    found: Dict[str, np.ndarray] = {}
    for name in query_names:
        cache_name = name
        if cache_name.startswith(("pos__", "neg__")) and "__" in cache_name:
            cache_name = cache_name.split("__", 1)[1]
        horiz = outdir / f"{cache_name}.horiz"
        if not horiz.exists():
            continue
        try:
            feat = parse_s4pred_horiz(horiz)
        except Exception:  # pragma: no cover - defensive, mirrors the release
            continue
        if feat is not None:
            found[name] = feat
    return found


def extract_s4pred_features(
    sequences: Dict[str, str],
    names: Optional[Sequence[str]] = None,
    strict: bool = False,
    cache_dir: Optional[str | Path] = None,
) -> np.ndarray:
    """Extract s4pred secondary-structure fractions (H/E/C) per sequence.

    Parameters
    ----------
    sequences : dict
        ``{accession: sequence}``.
    names : sequence of str, optional
        Order of the returned rows.  Defaults to the order of ``sequences``.
    strict : bool
        If ``True``, a missing s4pred installation raises instead of falling
        back to the constant used by the released pipeline.
    cache_dir : str or Path, optional
        Directory of cached s4pred ``.horiz`` files.  When given, the cached
        secondary structure is used for the sequences it covers -- the same
        source the released pipeline reads -- so both pipelines see identical
        features.  Sequences without a cache entry are predicted with s4pred,
        or fall back to the constant.

    Returns
    -------
    np.ndarray
        Shape ``(n, 3)``, float32.
    """
    order = list(names) if names is not None else list(sequences.keys())

    cached = scan_horiz_cache(cache_dir, order)
    if cached:
        logger.info(
            f"s4pred cache hit for {len(cached)}/{len(order)} sequences in {cache_dir}"
        )
    missing = [acc for acc in order if acc not in cached]

    predictor = None
    if missing:
        try:
            from s4pred.pred import Predictor as S4Pred

            predictor = S4Pred()
        except Exception as exc:  # pragma: no cover - depends on optional dep
            message = (
                "s4pred is not available; secondary-structure features fall back to "
                f"the constant {S4PRED_FALLBACK} used by the released pipeline"
            )
            if strict:
                raise RuntimeError(message) from exc
            logger.warning(message)

    features = np.zeros((len(order), 3), dtype=np.float32)
    for i, acc in enumerate(order):
        if acc in cached:
            features[i] = cached[acc]
            continue
        seq = sequences.get(acc, "")
        if predictor is None:
            features[i] = S4PRED_FALLBACK
            continue
        try:
            ss = predictor.predict(seq)
        except Exception:
            ss = ""
        n = len(ss)
        if n == 0:
            features[i] = S4PRED_FALLBACK
        else:
            features[i] = (
                sum(1 for s in ss if s == "H") / n,
                sum(1 for s in ss if s == "E") / n,
                sum(1 for s in ss if s == "C") / n,
            )
    return features


def extract_query_embeddings(
    sequences: Dict[str, str] | Sequence[str],
    backbone: ESM2Backbone,
    projection: ContrastiveProjection,
    names: Optional[Sequence[str]] = None,
    token_batch: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    batch_size: int = 8,
) -> Dict[str, np.ndarray]:
    """Compute Phase 2 contrastive embeddings for query proteins.

    Parameters
    ----------
    sequences : dict or sequence of str
        ``{accession: sequence}`` or a plain list of sequences.
    backbone : ESM2Backbone
        The **Phase 2** backbone.  It must match the backbone the contrastive
        head was trained with.
    projection : ContrastiveProjection
        The trained projection head.
    names : sequence of str, optional
        Accessions to use.  Required when ``sequences`` is a plain list.
    token_batch : (token_emb, input_ids), optional
        Pre-computed token embeddings.  Supplying these avoids a second forward
        pass when Phase 1 and Phase 2 share a backbone.
    batch_size : int
        Sequences per forward pass when ``token_batch`` is not supplied.

    Returns
    -------
    dict
        ``{accession: L2-normalised embedding}``.
    """
    if isinstance(sequences, dict):
        seq_map = dict(sequences)
        order = list(names) if names is not None else list(seq_map.keys())
    else:
        seq_list = list(sequences)
        if names is None:
            raise ValueError("`names` is required when `sequences` is a list")
        order = list(names)
        if len(order) != len(seq_list):
            raise ValueError(
                f"len(names)={len(order)} does not match len(sequences)={len(seq_list)}"
            )
        seq_map = dict(zip(order, seq_list))

    if token_batch is not None:
        token_emb, input_ids = token_batch
        token_emb = token_emb.to(backbone.device)
        input_ids = input_ids.to(backbone.device)
    else:
        _, token_emb, input_ids = backbone.encode_all(
            [seq_map[n] for n in order], batch_size=batch_size
        )
        token_emb = token_emb.to(backbone.device)
        input_ids = input_ids.to(backbone.device)

    if token_emb.shape[0] != len(order):
        raise ValueError(
            f"token batch has {token_emb.shape[0]} rows but {len(order)} queries were requested"
        )

    with torch.no_grad():
        pooled = build_phase2_pooled(token_emb, input_ids, backbone.pad_id)
        z = projection(pooled)
    z = z.detach().float().cpu().numpy()
    return {name: z[i] for i, name in enumerate(order)}


def load_contrastive_head(
    checkpoint: str | Path,
    backbone: ESM2Backbone | str,
    device: str = "cuda",
    embed_dim: int = 512,
    load_backbone_weights: bool = True,
) -> ContrastiveProjection:
    """Load the Phase 2 encoder from a training checkpoint.

    The released checkpoints store the whole contrastive encoder (ESM-2 plus
    the projection) under ``model_state``.  Both parts are restored here:

    * ``proj.*`` -- the projection head, loaded into a fresh
      :class:`~tpclass.core.esm2.ContrastiveProjection`;
    * ``esm.*`` -- the **fine-tuned** ESM-2 weights.  The contrastive training
      unfroze the last six transformer layers plus the final layer norm, so the
      published checkpoint contains a partially fine-tuned ESM-2.  The TCDB
      reference index was built with that fine-tuned encoder; using the plain
      pretrained ESM-2 instead would put the queries and the reference in two
      different embedding spaces and silently degrade the TC family calls.

    Parameters
    ----------
    checkpoint : str or Path
        Path to the Phase 2 checkpoint (``.pth``).
    backbone : ESM2Backbone or str
        The Phase 2 backbone (or its path).
    device : str
        Torch device.
    embed_dim : int
        Dimensionality of the contrastive embedding.
    load_backbone_weights : bool
        Whether to restore the fine-tuned ``esm.*`` weights into the backbone.
        Set to ``False`` to keep the pretrained ESM-2 (only sensible for a
        from-scratch reference index).

    Returns
    -------
    ContrastiveProjection
        Loaded head in ``eval`` mode.
    """
    if isinstance(backbone, str):
        backbone = ESM2Backbone(backbone, device=device)
    backbone.load()
    hidden = backbone.embed_dim

    head = ContrastiveProjection(hidden, embed_dim=embed_dim).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]

    if load_backbone_weights:
        esm_state = {
            k[len("esm.") :]: v for k, v in state.items() if k.startswith("esm.")
        }
        if esm_state:
            _restore_finetuned_esm(backbone, esm_state, checkpoint)

    proj_state = {
        k[len("proj.") :]: v for k, v in state.items() if k.startswith("proj.")
    }
    if not proj_state:
        raise RuntimeError(
            f"No 'proj.*' weights found in {checkpoint}. The checkpoint was not "
            "produced by the TPClass Phase 2 trainer."
        )
    missing, unexpected = head.proj.load_state_dict(proj_state, strict=False)
    if missing:
        raise RuntimeError(
            f"Phase 2 head is missing {len(missing)} weights (e.g. {missing[:3]}). "
            "The checkpoint does not match this backbone."
        )
    if unexpected:
        logger.warning(f"Phase 2 head: ignored {len(unexpected)} unexpected weights")
    head.eval()
    logger.info(f"Loaded Phase 2 projection head from {checkpoint} (hidden={hidden}, embed_dim={embed_dim})")
    return head


def _restore_finetuned_esm(
    backbone: ESM2Backbone,
    esm_state: Dict[str, torch.Tensor],
    checkpoint: str | Path,
) -> None:
    """Restore the fine-tuned ``esm.*`` weights of a Phase 2 checkpoint.

    The released pipeline restores the whole ``ContrastiveEncoder`` state dict
    (``strict=False``), so the partially fine-tuned ESM-2 is part of the
    published model.  This helper does the same for the backbone only.
    """
    model = backbone.model
    missing, unexpected = model.load_state_dict(esm_state, strict=False)
    matched = len(esm_state) - len(unexpected)
    if matched == 0:
        logger.warning(
            f"None of the {len(esm_state)} fine-tuned ESM-2 tensors in {checkpoint} matched "
            f"the {type(model).__name__} backbone. Phase 2 needs the fair-esm checkpoint "
            "the head was trained with; queries and reference index would otherwise live "
            "in different embedding spaces."
        )
        return
    if missing or unexpected:
        logger.warning(
            f"Restored {matched}/{len(esm_state)} fine-tuned ESM-2 tensors from {checkpoint} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)."
        )
    else:
        logger.info(f"Restored all {matched} fine-tuned ESM-2 tensors from {checkpoint}")
