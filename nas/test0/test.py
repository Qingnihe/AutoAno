import nni
import subprocess
import os
def run_model(model_name, params):
    if model_name == "sdfvae":
        os.chdir('../../SDFVAE_pytorch')
        command = f"python example_noshare.py --out_dir='out_test' --dataset_type='application-server-dataset' --entity=1 --global_window_size=60 --T=5 --batch_size={params['batch_size']} --s_dim={params['s_dim']} --d_dim={params['d_dim']} --model_dim={params['model_dim']} --lr={params['lr']}"
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("sdfvae Command executed successfully")
            print(result)
        else:
            print("sdfvae Command execution failed")
            print(result)
    elif model_name == "omnianomaly":
        os.chdir('../../OmniAnomaly')
        print(params['seed'])
        command = f"python example_noshare.py --out_dir='out_test' --dataset_type='application-server-dataset' --entity=2 --seed={params['seed']}"
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("omnianomaly Command executed successfully")
            print(result)
        else:
            print("omnianomaly Command execution failed")
            print(result)
    elif model_name == "usad":
        os.chdir('../../USAD')
        print(params['seed'])
        command = f"python example_noshare.py --out_dir='out_test' --dataset_type='application-server-dataset' --entity=2 --seed={params['seed']}"
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("usad Command executed successfully")
            print(result)
        else:
            print("usad Command execution failed")
            print(result)
        pass
    elif model_name == "interfusion":
        os.chdir('../../IF_pytorch')
        print(params['seed'])
        command = f"python example_noshare.py --out_dir='out_test' --dataset_type='application-server-dataset' --entity=2 --seed={params['seed']}"
        print(command)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("interfusion Command executed successfully")
            print(result)
        else:
            print("interfusion Command execution failed")
            print(result)
        


def main():
    params = nni.get_next_parameter()
    model_name = params["model"]["_name"]
    run_model(model_name, params["model"])
    
def test():
    params = nni.get_next_parameter()
    model_name = params["model"]["_name"]
    os.chdir('../../SDFVAE_pytorch')
    command = f"python example_noshare.py --out_dir='out_test' --dataset_type='application-server-dataset' --entity=1 --global_window_size=60 --T=5 --batch_size=64 --s_dim=10 --d_dim=10 --model_dim=10 --lr=0.0001"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    print(result)
    

if __name__ == "__main__":
    main()
    # test()
