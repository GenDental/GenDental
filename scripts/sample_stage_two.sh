export CUDA_VISIBLE_DEVICES=2
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/stage_two_sample.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/test \
 --ckpt_path /data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth3/ckpt/last.ckpt \
 --epochs 1000 \
 --num_gpus $num_gpus \
 --base_lr 1e-3\
 --fast \
 --test \