python example_noshare.py --dataset_type='yidong-22' --epochs=50 --train_type='example_noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=0519 --embed_dim=64
python example_noshare.py --dataset_type='yidong-22' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=0522 --embed_dim=64
python example_noshare.py --dataset_type='water-distribution' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=0522 --embed_dim=64 
python example_noshare.py --dataset_type='secure-water-treatment' --epochs=15 --train_type='效率实验' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=0522 --embed_dim=64  --batch_size=64


python example_noshare.py --dataset_type='yidong-22' --epochs=100 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-4 --topk=19 --out_dir=0522 --embed_dim=64
python example_noshare.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --epochs=100 --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=7e-4 --topk=19 --out_dir=0606 --embed_dim=64


python get_eval_result.py --dataset_type='yidong-22' --epochs=100 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-4 --topk=19 --out_dir=1123_3 --embed_dim=64

python example_noshare.py --dataset_type='PSM' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1208 --embed_dim=64
python get_eval_result.py --dataset_type='PSM' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1208 --embed_dim=64

python example_noshare.py --dataset_type='hai-23' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1213 --embed_dim=64
python get_eval_result.py --dataset_type='hai-23' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1213 --embed_dim=64

python example_noshare.py --dataset_type='hai-22' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1213 --embed_dim=64
python get_eval_result.py --dataset_type='hai-22' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1213 --embed_dim=64

python example_noshare.py --dataset_type='hai-21' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1215 --embed_dim=64
python get_eval_result.py --dataset_type='hai-21' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-3 --topk=19 --out_dir=1215 --embed_dim=64

python get_eval_result.py --dataset_type='yidong-22' --epochs=50 --train_type='noshare' --seed=2021 --gpu_id=0 --base_model_dir=test --lr=5e-4 --topk=19 --out_dir=0111 --embed_dim=64