# !bin/bash

model_opts=(parallel_wavegan parallel_wavegan wavernn)
models=(melgan pwg wavernn)
device=1
dataset="opencpop"
tgt_dir=infer/$dataset

mkdir -p $tgt_dir

for i in "${!models[@]}"; do
    model_opt=${model_opts[$i]}
    model=${models[$i]}
    mkdir -p $tgt_dir/$model
    echo "CUDA ${device}: infer ${model}"
    CUDA_VISIBLE_DEVICES="$device" python cli.py $model_opt evaluate \
        --dataset ./datasets/$dataset \
        --path ./models/$dataset/$model \
        --checkpoint ./models/opencpop/$model/checkpoints/00400000.ckpt
done