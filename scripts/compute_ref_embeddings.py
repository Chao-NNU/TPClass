#!/usr/bin/env python3
"""
compute_ref_embeddings.py
Encode known reference transporter proteins with the SAME ESM-2 3B model
that v3 uses, save mean-pool embeddings for downstream cosine-similarity filtering.

Two modes:
  1. Default (no args): encode hardcoded HMO_REFS and SQUALENE_REFS
     (UniProt accessions), pulled from inputs/uniprot_sprot.fasta
     Output:
       /data2/chao/transport/models/ref_embeddings_hmo.npz
       /data2/chao/transport/models/ref_embeddings_squalene.npz

  2. Custom (--fasta X --out Y): encode any FASTA file
     Header format: >ID  description...
     First whitespace-separated token is the accession/id.
     Output:
       Y (default: models/ref_embeddings_custom.npz)

Usage:
    # 默认：编码 HMO 和 squalene 硬编码 ref
    python compute_ref_embeddings.py

    # 自定义 FASTA（文档里收集的 HMO 转运蛋白）
    python compute_ref_embeddings.py \
        --fasta /data2/chao/transport/models/collected_HMO_transporters.fasta \
        --out /data2/chao/transport/models/ref_embeddings_hmo_extra.npz
"""
import argparse
import re
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


# === 默认配置 ===
ESM2_PATH     = Path("/data2/chao/transport/models/esm2_3b")  # HF transformers 3B
FASTA_PATH    = Path("/data2/chao/transport/inputs/uniprot_sprot.fasta")
OUT_HMO      = Path("/data2/chao/transport/models/ref_embeddings_hmo.npz")
OUT_SQUALENE = Path("/data2/chao/transport/models/ref_embeddings_squalene.npz")
MAX_LEN      = 800
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# === 硬编码参考集（保留原行为，向后兼容） ===
HMO_REFS = {
    # accession : (gene_name, substrate, tc_family) — for display only
    "P02920": ("lacY",     "lactose",        "2.A.1.5.1"),
    "P11551": ("fucP",     "L-fucose",       "2.A.1.7.2"),
    "P39352": ("nanX",     "sialic acid",    "2.A.1.15.4"),
    "P23200": ("mglC",     "galactose",      "3.A.1.2.3"),
    "P23889": ("kpsM",     "polysialic acid","3.A.1.101.1"),
    "P44776": ("fucP_HAE", "L-fucose",       "2.A.1.7.7"),  # H. influenzae ortholog
    "P18817": ("lacY_KLE", "lactose",        "2.A.1.5.4"),  # K. oxytoca ortholog
    "P59832": ("lacY_KPN", "lactose",        "2.A.1.7.18"), # K. pneumoniae
    "Q9ZK41": ("gluP_HP",  "glucose/galactose","2.A.1.7.7"),
    "Q24JQ0": ("SLC35D4",  "UDP-GlcNAc",     "2.A.7.12.4"),  # human
}

SQUALENE_REFS = {
    "B3QHB6":  ("HpnN_RHOPT", "hopanoid transport",  "2.A.6.7.5"),   # Rhodopseudomonas
    "A0A0H3KP92": ("HpnN_BURM1", "hopanoid transport", "2.A.6.2.23"), # Burkholderia
    "Q2T2R2":  ("HpnN_BURTA", "hopanoid transport",  "2.A.6.2.23"),
    "Q2FZY8":  ("POPT_STAA8", "polyprenyl-P transport","9.B.217.2.10"),
    "Q9KNC5":  ("POPT_VIBCH", "polyprenyl-P transport","9.B.217.2.9"),
    "B0R3W4":  ("CoxX_HALS3", "protoheme farnesyltransferase","9.B.241"),
    # 2026-09-12: 删 3 个代谢酶（PimE 多萜醇甘露糖基转移酶、ERG1×2 角鲨烯环氧化酶），
    # 它们是小疏水蛋白，跟 V-ATPase 16 kDa c 亚基在 ESM-2 空间结构相似，导致
    # V-ATPase c 亚基大批假阳性召回（608/955）。改用纯转运蛋白参考集。
    # "P9WN00":  ("PimE_MYCTO", "polyprenol mannosyltransferase","9.B.142"),
    # "P32476":  ("ERG1_YEAST", "squalene epoxidase",  "3.D.10.1.2"),
    # "Q9C1W3":  ("ERG1_SCHPO","squalene epoxidase",  "1.A.115.1.1"),
}


# === 加载 ESM-2 3B（两种模式都需要） ===
def load_esm2():
    print(f"Loading ESM-2 3B from {ESM2_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(ESM2_PATH))
    model = AutoModel.from_pretrained(str(ESM2_PATH), torch_dtype=torch.float32).to(DEVICE)
    model.eval()
    print(f"  loaded (hidden_size={model.config.hidden_size}, "
          f"num_layers={model.config.num_hidden_layers}, device={DEVICE})")
    return tokenizer, model


def encode_sequences(seqs, tokenizer, model, max_len=MAX_LEN, batch_size=4):
    """Mean-pool the final hidden state (same as v3 pipeline)."""
    embs = []
    for i in range(0, len(seqs), batch_size):
        batch = seqs[i:i+batch_size]
        enc = tokenizer(batch, return_tensors='pt', padding=True,
                        truncation=True, max_length=max_len + 2)
        input_ids = enc['input_ids'].to(DEVICE)
        attn_mask = enc['attention_mask'].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attn_mask,
                            output_hidden_states=True)
        tok_emb = outputs.hidden_states[-1]
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        mask = ((input_ids != pad_id) & (input_ids != eos_id)).unsqueeze(-1).float()
        mean_pool = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1)
        embs.append(mean_pool.cpu().numpy().astype(np.float32))
    return np.concatenate(embs, axis=0)


# === 模式 1：从 uniprot_sprot.fasta 抽 accession，编码硬编码 ref ===
def load_uniprot_fasta(path):
    """Load Swiss-Prot FASTA → {accession: (sequence, description)}."""
    print(f"Loading {path} ...")
    acc_to_seq, acc_to_desc = {}, {}
    UNIPROT_RE = re.compile(r"^>(?:sp|tr)\|([A-Z][A-Z0-9]+(?:\-\d+)?)\|(\S+)\s*(.*)$")
    TRAILING_TAG_RE = re.compile(r"\s+(GN=\S+|PE=\S+|SV=\S+)\s*")
    header, buf = None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if header:
                    m = UNIPROT_RE.match(header)
                    if m:
                        acc_to_seq[m.group(1)] = "".join(buf).upper()
                        acc_to_desc[m.group(1)] = TRAILING_TAG_RE.sub(" ", m.group(3)).strip()
                header = line
                buf = []
            else:
                buf.append(line)
        if header:
            m = UNIPROT_RE.match(header)
            if m:
                acc_to_seq[m.group(1)] = "".join(buf).upper()
                acc_to_desc[m.group(1)] = TRAILING_TAG_RE.sub(" ", m.group(3)).strip()
    print(f"  {len(acc_to_seq)} entries loaded")
    return acc_to_seq, acc_to_desc


def build_hardcoded_refs(tokenizer, model, acc_to_seq, acc_to_desc):
    """Encode hardcoded HMO_REFS and SQUALENE_REFS."""
    def build_one(refs, out_path):
        accs, genes, descs, subs, tcs, seqs = [], [], [], [], [], []
        skipped = []
        for acc, (gene, substrate, tc) in refs.items():
            if acc not in acc_to_seq:
                skipped.append(acc)
                continue
            accs.append(acc)
            genes.append(gene)
            descs.append(acc_to_desc[acc])
            subs.append(substrate)
            tcs.append(tc)
            seqs.append(acc_to_seq[acc])
        if skipped:
            print(f"  WARNING: skipped (not in FASTA): {skipped}")
        print(f"  encoding {len(seqs)} proteins ...")
        embs = encode_sequences(seqs, tokenizer, model)
        np.savez(out_path, accessions=np.array(accs), genes=np.array(genes),
                 descriptions=np.array(descs), substrates=np.array(subs),
                 tc_numbers=np.array(tcs), embeddings=embs)
        print(f"  [Save] {out_path}  ({embs.shape})")
        for acc, gene, sub, tc in zip(accs, genes, subs, tcs):
            print(f"    {acc} {gene:18s} {sub:25s} TC {tc}")

    print(f"\n[Build HMO ref embeddings]")
    build_one(HMO_REFS, OUT_HMO)
    print(f"\n[Build squalene ref embeddings]")
    build_one(SQUALENE_REFS, OUT_SQUALENE)


# === 模式 2：从任意 FASTA 编码 ===
def load_any_fasta(path):
    """
    Load arbitrary FASTA → (accession_list, description_list, sequence_list)
    Header format: >ID  description...
    First whitespace-separated token is the ID; rest is description.
    Lines starting with # inside the file are treated as comments (skipped).
    """
    print(f"Loading {path} ...")
    accs, descs, seqs = [], [], []
    cur_id, cur_desc, cur_buf = None, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if cur_id:
                    accs.append(cur_id)
                    descs.append(cur_desc or "")
                    seqs.append("".join(cur_buf).upper())
                header = line[1:].strip()
                if not header or header.startswith("#"):
                    cur_id, cur_desc = None, None
                else:
                    parts = header.split(None, 1)
                    cur_id = parts[0]
                    cur_desc = parts[1] if len(parts) > 1 else ""
                cur_buf = []
            elif not line.startswith("#"):
                if cur_id is not None:
                    cur_buf.append(line)
        if cur_id:
            accs.append(cur_id)
            descs.append(cur_desc or "")
            seqs.append("".join(cur_buf).upper())
    print(f"  {len(accs)} entries loaded")
    return accs, descs, seqs


def build_custom_refs(fasta_path, out_path, tokenizer, model):
    accs, descs, seqs = load_any_fasta(fasta_path)
    if not accs:
        print(f"  ERROR: no entries in {fasta_path}, skipping")
        return
    print(f"  encoding {len(seqs)} proteins ...")
    embs = encode_sequences(seqs, tokenizer, model)
    np.savez(out_path, accessions=np.array(accs),
             descriptions=np.array(descs),
             embeddings=embs)
    print(f"  [Save] {out_path}  ({embs.shape})")
    for acc, desc in zip(accs, descs):
        print(f"    {acc:15s} {desc[:80]}")


def main():
    ap = argparse.ArgumentParser(description="Encode reference protein sequences with ESM-2 3B.")
    ap.add_argument("--fasta", default=None,
                    help="Custom FASTA file to encode (bypasses hardcoded HMO/squalene refs).")
    ap.add_argument("--out", default=None,
                    help="Output .npz path (required if --fasta is set).")
    args = ap.parse_args()

    if args.fasta and not args.out:
        ap.error("--out is required when --fasta is set")

    tokenizer, model = load_esm2()

    if args.fasta:
        # 模式 2：编码用户提供的 FASTA
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[Build custom ref embeddings from {args.fasta}]")
        build_custom_refs(Path(args.fasta), out_path, tokenizer, model)
    else:
        # 模式 1：默认行为 — 编码 HMO_REFS 和 SQUALENE_REFS
        acc_to_seq, acc_to_desc = load_uniprot_fasta(FASTA_PATH)
        build_hardcoded_refs(tokenizer, model, acc_to_seq, acc_to_desc)

    print(f"\nDone.")


if __name__ == "__main__":
    main()