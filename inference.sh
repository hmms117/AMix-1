#!/bin/bash
cd ./src

# ===== Default =====
output_dir="./output"
num_seq=10
time=0.8
ckpt_path="./ckpt/AMix-1-1.7b.ckpt"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --input_seq) input_seq="$2"; shift ;;
        --output_dir) output_dir="$2"; shift ;;
        --num_seq) num_seq="$2"; shift ;;
        --time) time="$2"; shift ;;
        --ckpt_path) ckpt_path="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Input Sequence  : $input_seq"
echo "Output Directory: $output_dir"
echo "Number of Seq   : $num_seq"
echo "Noise Factor    : $time"
echo "Checkpoint Path : $ckpt_path"

python inference.py \
    --input_seq "$input_seq" \
    --output_dir "$output_dir" \
    --num_seq "$num_seq" \
    --time "$time" \
    --ckpt_path "$ckpt_path"
