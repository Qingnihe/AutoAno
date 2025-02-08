import os
import matplotlib.pyplot as plt
# from data_config import *
# from models.nni_model import TrainGDN

# class Config:
#     train_config = {
#         'batch': global_batch_size,
#         'epoch': global_epochs,
#         'slide_win': global_window_size,
#         'dim': global_embed_dim,
#         'slide_stride': global_slide_stride,
#         'comment': '',
#         'seed': seed,
#         'out_layer_num': global_out_layer_num,
#         'out_layer_inter_dim': global_out_layer_inter_dim,
#         'decay': learning_rate_decay_by_epoch,
#         'val_ratio': global_val_ratio,
#         'topk': global_topk,
#         'loss_fn': global_loss_fn
#     }

#     env_config={
#         'save_path': exp_dir,
#         'device': global_device,
#         'load_model_path': exp_dir,

#         'dataset': 'None',
#         'report': 'best'
#     }

def get_last_loss(log_file_path):
    """
    从日志文件中读取最后一个 Loss
    """

    # 读取日志文件的每一行
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 找到最后一个包含 train_loss 的行
    last_loss = None
    for line in reversed(lines):
        loss_strings = line.split(', ')
        if 'Loss' in loss_strings[0]:
            last_loss = float(loss_strings[0].split('Loss:')[1].strip())
            break

    # 输出最后一个 train_loss
    if last_loss is not None:
        # print("最后一个 train_loss:", last_train_loss)
        # print(last_loss)
        return last_loss
    else: 
        # print("未找到 train_loss")
        return 0.0
    
def get_f1(f1_file_path):
    try:
        with open(f1_file_path, 'r') as log_file:
            for line in log_file:
                if 'f1:' in line:
                    # 以空格分割，获取 f1 后面的数值部分
                    f1_value = float(line.split('f1:')[1].split()[0])
                    return f1_value
    except FileNotFoundError:
        pass
        # print(f"文件 '{f1_file_path}' 未找到")

    except Exception as e:
        pass
        # print(f"读取文件 '{f1_file_path}' 时出现错误: {str(e)}")

    return 0.0

# def get_mse_nf(model_path):
#     entity = int(model_path.split('/')[-2])
#     print(entity)

#     for i,(tr,te,lab) in enumerate(zip(train_data,test_data,label)):   
#         if i != entity:
#             continue
#         te=np.array(te)
#         test_data_item = te.T

#         feature_dim = test_data_item.shape[1]
#         config.train_config["feature_dim"] = feature_dim
#         model = TrainGDN(config.train_config, config.env_config)
#         model.restore(model_path)

#         avg_loss, test_predicted_list, test_ground_list = model.predict(test_values=test_data_item)
#         # score = get_err_scores(test_predicted_list, test_ground_list)
#         print(test_ground_list.shape, test_predicted_list.shape)

#         return

def main(path):
    min_loss = float('inf')
    min_loss_f1 = 0.0
    max_f1 = 0.0
    max_f1_loss = float('inf')
    min_loss_point = None
    max_f1_point = None

    dirs = []
    train_loss_list = []
    f1_list = []
    for item in os.scandir(path):
        if item.is_dir():
        #   dirs.append(item.path)
            log_file_path = item.path + "/model/1/log.txt"
            last_train_loss = get_last_loss(log_file_path)
            # window_size = int(item.path[-2:])
            # # print(window_size)
            # last_train_loss = last_train_loss / window_size
            f1_file_path = item.path + "/evaluation_result/bf_all_res.txt"
            f1 = get_f1(f1_file_path)
            if f1 == 0.0:
                continue
            dirs.append(os.path.basename(item.path))
            train_loss_list.append(last_train_loss)
            f1_list.append(f1)

            # 更新最小loss值和对应F1分数
            if last_train_loss < min_loss:
                min_loss = last_train_loss
                min_loss_f1 = f1
                min_loss_point = (last_train_loss, f1)
            # 更新最大F1分数和对应loss值
            if f1 > max_f1:
                max_f1 = f1
                max_f1_loss = last_train_loss
                max_f1_point = (last_train_loss, f1)
            
    fig, ax = plt.subplots(figsize=(40, 20))
    print(min_loss_point,max_f1_point)

    # 画散点图
    ax.scatter(train_loss_list, f1_list, label='Data Points', color='blue')
    # 如果找到了特殊点，则绘制它们
    if min_loss_point:
        ax.scatter(*min_loss_point, color='red', label='最小loss值对应的F1分数')
    if max_f1_point:
        ax.scatter(*max_f1_point, color='red', label='F1分数最大时对应的loss值')

    ax.set_xlabel('Train Loss')
    ax.set_ylabel('F1')

    # 添加标签，你可以根据需要进行调整
    # for i, txt in enumerate(dirs):
    #     ax.annotate(txt[8:], (train_loss_list[i], f1_list[i]), fontsize=8)

    # 保存为 PDF 文件
    plt.savefig("imgs/gdn-asd-2-w300.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)

    # 显示图形
    plt.show()

# def main_nf(path):
#     dirs = []
#     mse_nf_list = []
#     f1_list = []
#     for item in os.scandir(path):
#         if item.is_dir():
#         #   dirs.append(item.path)
#             model_path = item.path + "/model/1/model.pth"
#             mse_nf = get_mse_nf(model_path)
#             return
#             # window_size = int(item.path[-2:])
#             # # print(window_size)
#             # last_train_loss = last_train_loss / window_size
#             f1_file_path = item.path + "/evaluation_result/bf_all_res.txt"
#             f1 = get_f1(f1_file_path)
#             if f1 == 0.0:
#                 continue
#             dirs.append(os.path.basename(item.path))
#             mse_nf_list.append(mse_nf)
#             f1_list.append(f1)
            
#     fig, ax = plt.subplots(figsize=(40, 20))

#     # 画散点图
#     ax.scatter(mse_nf_list, f1_list, label='Data Points', color='blue')

#     ax.set_xlabel('Train Loss')
#     ax.set_ylabel('F1')

#     # 添加标签，你可以根据需要进行调整
#     # for i, txt in enumerate(dirs):
#     #     ax.annotate(txt[8:], (train_loss_list[i], f1_list[i]), fontsize=8)

#     # 保存为 PDF 文件
#     plt.savefig("gdn-asd-2-w300.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)

#     # 显示图形
#     plt.show()
            

if __name__ == "__main__":
    # config = Config()
    main("/home/sunyongqian/chenshiqi/mts-ano2/GDN/out/out_asd_2_w300/application-server-dataset")
    # main("/home/zhangshenglin/chenshiqi/lipeng/GDN/out_0131_lp_9/application-server-dataset")

    