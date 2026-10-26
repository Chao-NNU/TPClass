#!/bin/bash
# run_tclass_hmo_one_shot.sh
# 一条龙：启 p0 (GPU 2) + p1 (GPU 7) → 等完成 → merge → HMO filter
# 用法:  nohup bash run_tclass_hmo_one_shot.sh > logs/oneshot.log 2>&1 &

set -e
cd /data2/chao/transport

# === 配置 ===
GPUS=(2 7)
CHUNK_PREFIX="inputs/uniprot_sprot_chunk"
CHUNK_SIZE=287752
OUT_P0="outputs/uniprot_sprot_p0"
OUT_P1="outputs/uniprot_sprot_p1"
OUT_FINAL="outputs/uniprot_sprot"
SCRIPT="scripts/predict_transport_family_v3.py"

# === 准备 ===
mkdir -p logs "$OUT_P0" "$OUT_P1"

# 杀掉可能残留的旧进程（保险）
pkill -f predict_transport_family_v3 2>/dev/null || true
sleep 3

# 删之前跑出来的部分输出
rm -rf "$OUT_P0" "$OUT_P1" "$OUT_FINAL"
mkdir -p "$OUT_P0" "$OUT_P1"

# 验证 chunk 存在
TOTAL=$(grep -c "^>" inputs/uniprot_sprot.fasta)
echo "[$(date +%H:%M:%S)] Total proteins: $TOTAL"
if [ ! -f "${CHUNK_PREFIX}_000.fasta" ]; then
    echo "[$(date +%H:%M:%S)] Splitting FASTA..."
    python scripts/split_fasta.py inputs/uniprot_sprot.fasta $CHUNK_SIZE "$CHUNK_PREFIX"
fi

# === 启动 p0 (chunk 000, GPU 2) ===
echo "[$(date +%H:%M:%S)] Starting p0 (GPU ${GPUS[0]}, chunk 000)..."
nohup python "$SCRIPT" \
    -in "${CHUNK_PREFIX}_000.fasta" \
    -out "$OUT_P0/" \
    --gpu ${GPUS[0]} \
    --no_plot \
    > logs/sprot_p0.log 2>&1 &
P0_PID=$!
echo "  p0 PID: $P0_PID"

# === 启动 p1 (chunk 001, GPU 7) ===
echo "[$(date +%H:%M:%S)] Starting p1 (GPU ${GPUS[1]}, chunk 001)..."
nohup python "$SCRIPT" \
    -in "${CHUNK_PREFIX}_001.fasta" \
    -out "$OUT_P1/" \
    --gpu ${GPUS[1]} \
    --no_plot \
    > logs/sprot_p1.log 2>&1 &
P1_PID=$!
echo "  p1 PID: $P1_PID"

# 30 秒后看是否启动成功
sleep 30
echo ""
echo "[$(date +%H:%M:%S)] === 启动 30 秒后状态 ==="
ps -p $P0_PID -o pid,etime,cmd 2>/dev/null || echo "  p0 已退出"
ps -p $P1_PID -o pid,etime,cmd 2>/dev/null || echo "  p1 已退出"
echo ""
echo "GPU 占用:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits | head -8

# === 等待 p0 + p1 都完成（轮询）===
echo ""
echo "[$(date +%H:%M:%S)] === 等待 p0 + p1 完成 ==="
while kill -0 $P0_PID 2>/dev/null || kill -0 $P1_PID 2>/dev/null; do
    sleep 120
    P0_ALIVE=$([ -e /proc/$P0_PID ] && echo "Y" || echo "N")
    P1_ALIVE=$([ -e /proc/$P1_PID ] && echo "Y" || echo "N")
    P0_HAS_OUT=$([ -f "$OUT_P0/final_report.csv" ] && echo "Y" || echo "N")
    P1_HAS_OUT=$([ -f "$OUT_P1/final_report.csv" ] && echo "Y" || echo "N")
    echo "  [$(date +%H:%M:%S)] p0:$P0_ALIVE/OUT:$P0_HAS_OUT  p1:$P1_ALIVE/OUT:$P1_HAS_OUT"
done

# === 检查两个 final_report.csv 都生成了 ===
echo ""
echo "[$(date +%H:%M:%S)] === 检查 final_report.csv ==="
for d in "$OUT_P0" "$OUT_P1"; do
    if [ -f "$d/final_report.csv" ]; then
        SIZE=$(du -h "$d/final_report.csv" | cut -f1)
        echo "  OK $d/final_report.csv  ($SIZE)"
    else
        echo "  ERROR: $d/final_report.csv not found!"
        echo "  log tail p0:"; tail -30 logs/sprot_p0.log
        echo "  log tail p1:"; tail -30 logs/sprot_p1.log
        exit 1
    fi
done

# === Merge ===
echo ""
echo "[$(date +%H:%M:%S)] === 合并 final_report.csv ==="
mkdir -p "$OUT_FINAL"
python scripts/merge_final_reports.py "$OUT_FINAL" "$OUT_P0" "$OUT_P1"

# === HMO 过滤 ===
echo ""
echo "[$(date +%H:%M:%S)] === HMO 候选过滤 ==="
python scripts/filter_hmo_candidates.py

# === 收尾 ===
echo ""
echo "[$(date +%H:%M:%S)] === 完成 ==="
echo "  合并结果:   $OUT_FINAL/final_report.csv"
echo "  HMO 候选:   $OUT_FINAL/hmo_candidates.csv"
echo "  top 50:     $OUT_FINAL/hmo_top_candidates.csv"
echo "  统计:       $OUT_FINAL/hmo_summary.txt"
echo ""
echo "总耗时:"
ps -p $P0_PID -o etime 2>/dev/null || true
