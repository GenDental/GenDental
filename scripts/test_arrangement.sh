export CUDA_VISIBLE_DEVICES=6
num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
python main.py --config configs/arrangement.yaml \
 --output_dir /data3/leics/dataset/checkpoints/ToothWise/alignment5 \
 --ckpt_path /data3/leics/dataset/checkpoints/ToothWise/alignment_zj_mix0.8synthetic_TANet/ckpt/ckpt-epoch=918-val_total_loss=61.6593.ckpt \
 --epochs 1000 \
 --num_gpus $num_gpus \
 --base_lr 1e-4\
 --fast \
 --test
