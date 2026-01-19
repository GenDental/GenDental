export CUDA_VISIBLE_DEVICES=1
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/stage_one.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/old_gpt \
 --ckpt_path /data3/leics/dataset/checkpoints/ToothWise/old_gpt_segment/ckpt/last.ckpt \
 --epochs 1000 \
 --num_gpus $num_gpus \
 --base_lr 1e-5\
 --test \
 --fast \
