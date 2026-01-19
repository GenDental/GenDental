export CUDA_VISIBLE_DEVICES=3
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/stage_two.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth3 \
 --ckpt_path /data3/leics/dataset/checkpoints/ToothWise/motion_transfer_smooth2/ckpt/last.ckpt \
 --epochs 2000 \
 --num_gpus $num_gpus \
 --base_lr 1e-5\
 --fast \
