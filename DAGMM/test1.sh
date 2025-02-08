python example_noshare.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python example_noshare.py --dataset_type='application-server-dataset' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python example_noshare.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409

python get_eval_result.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python get_eval_result.py --dataset_type='application-server-dataset' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python get_eval_result.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409

python arrange.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python arrange.py --dataset_type='application-server-dataset' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409
python arrange.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-3 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0409


python example_noshare.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-4 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0422
python example_noshare.py --dataset_type='water-distribution' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-4 --epoch=500 --dropout_r=0.1 --origin_samples=1024 --out_dir=0422

python example_noshare.py --dataset_type='yidong-22' --train_type='noshare' --seed=2021 --gpu_id=1 --base_model_dir=test --lr=1e-4 --epoch=50 --dropout_r=0.1 --origin_samples=1024 --out_dir=1123