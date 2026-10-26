#!/bin/bash
# run_tclass_hmo_pipeline.sh
# 一键跑：拆 FASTA → 双卡 TClass → 合并 → HMO 过滤
# 用法: bash run_tclass_hmo_pipeline.sh

set -e
cd /data2/chao/transport

# === 配置 ===
INPUT_FASTA="inputs/uniprot_sprot.fasta"
CHUNK_PREFIX="inputs/uniprot_sprot_chunk"
CHUNK_SIZE=287752  # 575504 / 2 = 287752，向上取整 575503/2
GPUS=(2 7)
OUT_P0="outputs/uniprot_sprot_p0"
OUT_P1="outputs/uniprot_sprot_p1"
OUT_FINAL="outputs/uniprot_sprot"
LOG_DIR="logs"
SCRIPT="scripts/predict_transport_family_v3.py"

mkdir -p "$LOG_DIR"

echo "=== [1/5] Split FASTA ==="
TOTAL=$(grep -c "^>" "$INPUT_FASTA")
echo "  total: $TOTAL"
python scripts/split_fasta.py "$INPUT_FASTA" $CHUNK_SIZE "$CHUNK_PREFIX"
echo "  chunks:"
for f in ${CHUNK_PREFIX}_*.fasta; do
    n=$(grep -c "^>" "$f")
    echo "    $f: $n"
done

echo ""
echo "=== [2/5] Start TClass on GPU 2 and GPU 7 (background) ==="
nohup env CUDA_VISIBLE_DEVICES=${GPUS[0]} python "$SCRIPT" \
    -in "${CHUNK_PREFIX}_000.fasta" \
    -out "$OUT_P0/" \
    --gpu 0 \
    --no_plot \
    > "$LOG_DIR/sprot_p0.log" 2>&1 &
P0_PID=$!
echo "  p0 PID: $P0_PID  (GPU ${GPUS[0]})"

nohup env CUDA_VISIBLE_DEVICES=${GPUS[1]} python "$SCRIPT" \
    -in "${CHUNK_PREFIX}_001.fasta" \
    -out "$OUT_P1/" \
    --gpu 0 \
    --no_plot \
    > "$LOG_DIR/sprot_p1.log" 2>&1 &
P1_PID=$!
echo "  p1 PID: $P1_PID  (GPU ${GPUS[1]})"

# 等 30 秒看是否启动成功
sleep 30
echo ""
echo "=== [3/5] Process check ==="
ps -p $P0_PID -o pid,etime,cmd 2>/dev/null || echo "  p0 died"
ps -p $P1_PID -o pid,etime,cmd 2>/dev/null || echo "  p1 died"
echo ""
echo "GPU 占用:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | head -8
echo ""
echo "p0 log tail:"
tail -15 "$LOG_DIR/sprot_p0.log"
echo ""
echo "p1 log tail:"
tail -15 "$LOG_DIR/sprot_p1.log"

echo ""
echo "=== [4/5] Wait for completion (this blocks until both done) ==="
echo " 监控方式: watch -n 30 'tail -5 $LOG_DIR/sprot_p0.log; tail -5 $LOG_DIR/sprot_p1.log; nvidia-smi | head -12'"
echo ""
# 轮询 wait
while kill -0 $P0_PID 2>/dev/null || kill -0 $P1_PID 2>/dev/null; do
    sleep 60
    P0_ALIVE=$([ -e /proc/$P0_PID ] && echo Y || echo N)
    P1_ALIVE=$([ -e /proc/$P1_PID ] && echo Y || echo N)
    echo "  [$(date +%H:%M:%S)] p0:$P0_ALIVE p1:$P1_ALIVE"
    if [ ! -e /proc/$P0_PID ] && [ ! -e /proc/$P1_PID ]; then
        break
    fi
done

echo ""
echo "=== [5/5] Merge + filter ==="
# 确认两个 final_report.csv 都生成了
for d in "$OUT_P0" "$OUT_P1"; do
    if [ ! -f "$d/final_report.csv" ]; then
        echo "ERROR: $d/final_report.csv not found, exit"
        echo "p0 log tail:"; tail -30 "$LOG_DIR/sprot_p0.log"
        echo "p1 log tail:"; tail -30 "$LOG_DIR/sprot_p1.log"
        exit 1
    fi
done

# 合并
python scripts/merge_final_reports.py "$OUT_FINAL" "$OUT_P0" "$OUT_P1"

# HMO 过滤
python scripts/filter_hmo_candidates.py

echo ""
echo "=== Done ==="
echo "HMO 候选 top 50: $OUT_FINAL/hmo_top_candidates.csv"
echo "完整候选:       $OUT_FINAL/hmo_candidates.csv"
echo "统计摘要:       $OUT_FINAL/hmo_summary.txt"
