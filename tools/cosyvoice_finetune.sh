#!/usr/bin/env bash
# Fine-tune CosyVoice 3 (LLM part: text → speech tokens, where pronunciation and stress live) on the
# narrator's dataset from `factory voice-dataset`. Follows examples/libritts/cosyvoice3/run.sh.
#
#   source /workspace/factory.env
#   bash tools/cosyvoice_finetune.sh work/voice_dataset narrator_v1 [EPOCHS=10]
#
# Result: $WS/cache/models/narrator_v1 = base model dir with the fine-tuned llm.pt.
# Use it: tts.cosyvoice.model_dir in config/pod.yaml (or COSYVOICE_MODEL_DIR).
set -euo pipefail
DATA=$(realpath "$1"); NAME=$2; EPOCHS=${3:-10}
WS=${WORKSPACE:-/workspace}
CV=${COSYVOICE_DIR:?run tools/cosyvoice_setup.sh and source factory.env}
PY=${COSYVOICE_PYTHON:?}
BASE=${COSYVOICE_BASE_MODEL_DIR:-$WS/cache/models/Fun-CosyVoice3-0.5B-2512}
EXP=${FT_EXP_DIR:-/root/ft/$NAME}          # local disk: checkpoints are written often
OUT=$WS/cache/models/$NAME
export PYTHONPATH=$CV:$CV/third_party/Matcha-TTS:${PYTHONPATH:-} PYTHONIOENCODING=UTF-8
mkdir -p "$EXP"

"$PY" -c "import deepspeed" 2>/dev/null || VIRTUAL_ENV=$(dirname "$(dirname "$PY")") uv pip install -q deepspeed==0.15.1

echo "== 1/5 speaker embeddings (campplus)"
"$PY" "$CV/tools/extract_embedding.py" --dir "$DATA" --onnx_path "$BASE/campplus.onnx"
echo "== 2/5 speech tokens"
"$PY" "$CV/tools/extract_speech_token.py" --dir "$DATA" --onnx_path "$BASE/speech_tokenizer_v3.onnx"
echo "== 3/5 parquet"
rm -rf "$DATA/parquet"; mkdir -p "$DATA/parquet"
"$PY" "$CV/tools/make_parquet_list.py" --num_utts_per_parquet 40 --num_processes 4 \
    --src_dir "$DATA" --des_dir "$DATA/parquet"
# hold out the last parquet shard (~40 lines) for validation
n=$(wc -l < "$DATA/parquet/data.list")
if [ "$n" -ge 2 ]; then
    head -n $((n - 1)) "$DATA/parquet/data.list" > "$EXP/train.list"; tail -n 1 "$DATA/parquet/data.list" > "$EXP/dev.list"
else
    cp "$DATA/parquet/data.list" "$EXP/train.list"; cp "$DATA/parquet/data.list" "$EXP/dev.list"
fi

echo "== 4/5 train llm ($EPOCHS epochs, SFT lr 1e-5)"
sed -e "s/max_epoch: 200/max_epoch: $EPOCHS/" -e "s/log_interval: 100/log_interval: 10/" \
    "$CV/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml" > "$EXP/cosyvoice3.yaml"
cd "$CV/examples/libritts/cosyvoice3"
"$(dirname "$PY")/torchrun" --nnodes=1 --nproc_per_node=1 --rdzv_id=1986 --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:1234 "$CV/cosyvoice/bin/train.py" \
    --train_engine torch_ddp --config "$EXP/cosyvoice3.yaml" \
    --train_data "$EXP/train.list" --cv_data "$EXP/dev.list" \
    --qwen_pretrain_path "$BASE/CosyVoice-BlankEN" --onnx_path "$BASE" \
    --model llm --checkpoint "$BASE/llm.pt" \
    --model_dir "$EXP/llm" --tensorboard_dir "$EXP/tensorboard" \
    --ddp.dist_backend nccl --num_workers 2 --prefetch 100 --pin_memory --use_amp

echo "== 5/5 average best checkpoints → $OUT"
"$PY" "$CV/cosyvoice/bin/average_model.py" --dst_model "$EXP/llm.pt" --src_path "$EXP/llm" \
    --num "$(( EPOCHS < 3 ? EPOCHS : 3 ))" --val_best
mkdir -p "$OUT"
for f in "$BASE"/*; do
    b=$(basename "$f")
    [ "$b" = llm.pt ] && continue
    [ -e "$OUT/$b" ] || cp -r "$f" "$OUT/$b"
done
cp "$EXP/llm.pt" "$OUT/llm.pt"
echo "Done: $OUT"
echo "Use it:  COSYVOICE_MODEL_DIR=$OUT factory build … --config config/pod.yaml"
