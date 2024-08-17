python -m torch.distributed.launch --nproc_per_node=4 --master_port 29501 main.py  --lr 0.002 -warmup-lr 1e-6  --drop-path 0.1 --clip-grad 2  --workers 8  --amp --sync-bn 

