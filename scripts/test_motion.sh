export CUDA_VISIBLE_DEVICES=4
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/stage_prediction.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/tooth_motion \
 --ckpt_path /data3/leics/dataset/checkpoints/ToothWise/tooth_motion_nonsmoothed_20steps_5xpure_synthetic_newckpt/ckpt/ckpt-epoch=968-val_total_loss=59.3741.ckpt \
 --epochs 1000 \
 --num_gpus $num_gpus \
 --base_lr 1e-4\
 --fast \
 --test
