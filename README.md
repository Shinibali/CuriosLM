
python train.py config/train_shakespeare.py --device=mps --compile=True --position_encoding=rope --max_iters=100 --eval_interval=50 --eval_iters=5 --mlp_type=geglu --norm_placement=sandwich

python sample.py --out_dir=out-shakespeare --device=mps --num_samples=2 --max_new_tokens=100