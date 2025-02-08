# python example_noshare.py --dataset_type='yidong-22' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100
python example_noshare.py --dataset_type='server-machine-dataset' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100 --batch_size=64
python example_noshare.py --dataset_type='soil-moisture-active-passive' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100
python example_noshare.py --dataset_type='mars-science-laboratory' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100
python example_noshare.py --dataset_type='application-server-dataset' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100
# python example_noshare.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=50 --lr=1e-4 --global_window_size=100
python example_noshare.py --dataset_type='secure-water-treatment' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=5 --lr=1e-4 --global_window_size=100
python example_noshare.py --dataset_type='water-distribution' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=5 --lr=1e-4 --global_window_size=100 --batch_size=64



python example_noshare_test_time.py --dataset_type='server-machine-dataset' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100 --batch_size=64
python example_noshare_test_time.py --dataset_type='secure-water-treatment' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=0  --epochs=100 --lr=1e-4 --global_window_size=100 --batch_size=64
python example_noshare_test_time.py --dataset_type='water-distribution' --train_type='效率实验' --seed=2021 --out_dir=out_0423 --gpu_id=1  --epochs=100 --lr=1e-4 --global_window_size=100 --batch_size=64


python example_noshare.py --dataset_type='CTF_OmniClusterSelected_th48_26cluster' --train_type='noshare' --seed=2021 --out_dir=0606 --gpu_id=0  --epochs=250 --lr=1e-3 --global_window_size=60
python example_noshare.py --dataset_type='soil-moisture-active-passive' --train_type='noshare' --seed=2021 --out_dir=0606 --gpu_id=1  --epochs=250 --lr=5e-4 --global_window_size=60
python example_noshare.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --out_dir=0606 --gpu_id=0  --epochs=250 --lr=5e-4 --global_window_size=60

python example_noshare.py --dataset_type='mars-science-laboratory' --train_type='noshare' --seed=2021 --out_dir=1206 --gpu_id=1  --epochs=20 --lr=5e-4 --global_window_size=60
