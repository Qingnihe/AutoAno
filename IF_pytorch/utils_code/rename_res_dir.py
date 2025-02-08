import pathlib
import shutil

import os

res_dir = pathlib.Path('/home/chengdaguo/code/Compared_zhujun/USAD/out')
for f in os.listdir(res_dir):
    new_f = f.replace('_ws20_ELU', '')
    new_f = new_f.replace('alpha_0.5_beta_0.5_', '')
    # os.rename(res_dir / f , res_dir/ new_f)
    print(new_f)