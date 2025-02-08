import pandas as pd

# exp_list = ['noshare_ctf_earlystop10', 'finetune_all_ctf_earlystop10', 'use_center_ctf_earlystop10']
exp_list = ['noshare22_409']

for exp in exp_list:
    print(exp, '-------------')
    df = pd.read_csv(f'/home/chengdaguo/code/Compared_zhujun/USAD/out/{exp}/evaluation_result/bf_machine_best_f1.csv')

    df = df.sort_values(by='f1')

    low_m_list = {}
    for index, row in df.iterrows():
        # print(f"m:{int(row['machine_id'])} f1:{round(row['f1'], 4)}")
        if row['f1'] < 0.7 :
            low_m_list[int(row['machine_id'])] =round( row['f1'], 4)

    print(low_m_list)