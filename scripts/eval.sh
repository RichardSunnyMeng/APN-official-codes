time=$(date "+%Y%m%d-%H%M%S")
NAME=${0%\.*}

export PYTHONPATH=$PYTHONPATH:~/APN-official-codes

CUDA_VISIBLE_DEVICES=$n python ${ROOT}/eval.py \
    --eval_cfg ./configs/eval.json
