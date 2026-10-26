#!/usr/bin/env bash
# ============================================================
# run_ablation_train_val.sh (v4 — balanced held-out)
# Full retrain pipeline: TCDB 80/10 split + UniProt random neg →
#   extract features (pos + neg) → retrain Phase 1 (MLP + XGB)
#   → 5 ablation variants on balanced val (pos + neg, 1:1)
#
# Total time: ~3-5 hours (ESM-2 extraction is the bottleneck, 30-50 min)
# ============================================================

SCRIPTS=/data2/chao/transport/scripts
ROOT=/data2/chao/transport
DATA=$ROOT/data
FEAT_DIR=$DATA/features_v3_retrain
MODELS=$ROOT/models/v3_retrained
OUT=$ROOT/outputs/ablation
mkdir -p $DATA $MODELS $OUT

ts() { date '+%H:%M:%S'; }

# ── Step 1: Stratified train/val/test split of TCDB positives ──
echo "[$(ts)] [1/6] Splitting TCDB 80/10/10 ..."
if [ ! -f $DATA/tcdb_train.fasta ]; then
    python $SCRIPTS/split_tcdb.py \
        --input $DATA/tcdb.fa \
        --out_dir $DATA \
        --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 --seed 42
fi

# ── Step 2: Extract 1283-dim features (TCDB pos + UniProt neg, balanced) ──
echo "[$(ts)] [2/6] Extracting features (ESM-2 + s4pred, pos+neg) — bottleneck, 30-50 min ..."
if [ ! -f $FEAT_DIR/X_train.npy ] || [ ! -f $FEAT_DIR/X_val.npy ]; then
    python $SCRIPTS/extract_features_for_retrain.py \
        --pos_train_fasta $DATA/tcdb_train.fasta \
        --pos_val_fasta   $DATA/tcdb_val.fasta \
        --neg_pool_fasta  $DATA/uniprot_negative_c90.fasta \
        --output_dir      $FEAT_DIR \
        --gpu 0
fi

# ── Step 3: Retrain Phase 1 MLP ──
echo "[$(ts)] [3/6] Retraining Phase 1 MLP ..."
if [ ! -f $MODELS/phase1/transport_classifier_best.pth ]; then
    python $SCRIPTS/train_mlp.py \
        --data_dir $FEAT_DIR \
        --output_path $MODELS/phase1/transport_classifier_best.pth \
        --batch_size 1024 --epochs 100 --lr 1e-3
fi

# ── Step 4: Retrain Phase 1 XGBoost ──
echo "[$(ts)] [4/6] Retraining Phase 1 XGBoost ..."
if [ ! -f $MODELS/phase1/transport_xgb_best.json ]; then
    python $SCRIPTS/train_xgb.py \
        --data_dir $FEAT_DIR \
        --output_path $MODELS/phase1/transport_xgb_best.json
fi

# ── Step 5: Build balanced val fasta (TCDB pos + UniProt neg, 1:1) ──
echo "[$(ts)] [5/6] Building balanced val fasta + running 5 ablation variants ..."
BAL=$DATA/tcdb_val_balanced.fasta
if [ ! -f $BAL ]; then
    if [ ! -f $DATA/uniprot_negative_c90.val.fasta ]; then
        echo "[$(ts)] [5/6] Splitting UniProt neg pool (first N=tcdb_val_size for val, next N=pos_train_size for train) ..."
        python -c "
import random
from pathlib import Path
pos_val_count = sum(1 for line in open('$DATA/tcdb_val.fasta') if line.startswith('>'))
pos_train_count = sum(1 for line in open('$DATA/tcdb_train.fasta') if line.startswith('>'))
queries = []
header = None
seq_chunks = []
with open('$DATA/uniprot_negative_c90.fasta') as f:
    for line in f:
        line = line.rstrip()
        if line.startswith('>'):
            if header is not None:
                queries.append((header, ''.join(seq_chunks)))
            header = line[1:]
            seq_chunks = []
        else:
            seq_chunks.append(line)
    if header is not None:
        queries.append((header, ''.join(seq_chunks)))
print(f'[NegPool] {len(queries)} sequences before shuffle')
rnd = random.Random(42)
rnd.shuffle(queries)
val_block = queries[:pos_val_count]
after_val = queries[pos_val_count:]
train_block = after_val[:pos_train_count]
print(f'[NegPool] train_neg={len(train_block)}, val_neg={len(val_block)}')
def write_fasta(path, items, tag):
    with open(path, 'w') as f:
        for n, s in items:
            f.write(f'>{tag}__{n}\n{s}\n')
write_fasta('$DATA/uniprot_negative_c90.train.fasta', train_block, 'neg')
write_fasta('$DATA/uniprot_negative_c90.val.fasta', val_block, 'neg')
print('[NegPool] Wrote train and val fastas')
"
    fi
    # Build balanced val: TCDB val (already has pos__ prefix? no, it's raw TCDB) + neg__ val
    # Need to also prefix TCDB val headers with 'pos__'
    python -c "
from pathlib import Path
with open('$DATA/tcdb_val.fasta') as f_in, open('$BAL', 'w') as f_out:
    for line in f_in:
        if line.startswith('>'):
            f_out.write('>pos__' + line[1:])
        else:
            f_out.write(line)
with open('$DATA/uniprot_negative_c90.val.fasta') as f_in, open('$BAL', 'a') as f_out:
    for line in f_in:
        f_out.write(line)
print(f'[BalVal] Wrote $BAL')
"
fi

# Step 5b: Run 5 ablation variants on the balanced val fasta
VARIANTS=("full_v3" "no_phase1" "no_s4pred" "pure_mlp" "pure_xgb")
for v in "${VARIANTS[@]}"; do
    case $v in
        full_v3)   flag="" ;;
        no_phase1) flag="--no_phase1" ;;
        no_s4pred) flag="--no_s4pred" ;;
        pure_mlp)  flag="--phase1_weights 1.0,0.0" ;;
        pure_xgb)  flag="--phase1_weights 0.0,1.0" ;;
    esac

    VOUT=$OUT/val_$v
    mkdir -p $VOUT
    FINAL=$VOUT/final_report.csv
    if [ -f "$FINAL" ]; then
        echo "  [$(ts)] [Skip] $v already done"
        continue
    fi

    echo "  [$(ts)] [Run] $v  (flag: ${flag:-<default>})"
    MLP_ASSET=$MODELS/phase1/transport_classifier_best.pth \
    XGB_ASSET=$MODELS/phase1/transport_xgb_best.json \
    python $SCRIPTS/predict_transport_family_v3.py \
        -in $BAL \
        -out $VOUT \
        --tcdb_annot $DATA/tcdb_annot.clean.tsv \
        --no_plot $flag
done

# ── Step 6: Evaluate all 5 variants against the balanced val fasta ──
echo "[$(ts)] [6/6] Evaluating 5 variants on balanced val ..."
ARGS=""
for v in "${VARIANTS[@]}"; do
    ARGS="$ARGS $v|$OUT/val_$v/final_report.csv"
done
python $SCRIPTS/evaluate_heldout.py \
    --heldout_fasta $BAL \
    --variants $ARGS \
    --out $OUT/val_table.csv

echo ""
echo "[$(ts)] [Done] Table:"
cat $OUT/val_table.csv
echo ""
cat $OUT/val_table.md 2>/dev/null
