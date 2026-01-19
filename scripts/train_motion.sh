export CUDA_VISIBLE_DEVICES=2
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/stage_prediction.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/tooth_motion_nonsmoothed_20steps_3xpure_synthetic_newckpt \
 --epochs 1000 \
 --num_gpus $num_gpus \
 --base_lr 1e-4\
 --fast \
