import os
import matplotlib.pyplot as plt


def get_last_train_loss(log_file_path):
    """
    从日志文件中读取最后一个 train_loss
    """

    # 读取日志文件的每一行
    with open(log_file_path, 'r') as file:
        lines = file.readlines()

    # 找到最后一个包含 train_loss 的行
    last_train_loss = None
    for line in reversed(lines):
        if 'train_loss' in line:
            last_train_loss = float(line.split('train_loss:')[1].strip())
            break

    # 输出最后一个 train_loss
    if last_train_loss is not None:
        # print("最后一个 train_loss:", last_train_loss)
        return last_train_loss
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
            log_file_path = item.path + "/model/data_1/log.txt"
            last_train_loss = get_last_train_loss(log_file_path)
            # window_size = int(item.path[-2:])
            # print(window_size)
            # last_train_loss = last_train_loss / window_size
            f1_file_path = item.path + "/evaluation_result/bf_res.txt"
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
    plt.savefig("imgs/sdfvae-asd-2-w300.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)

    # 显示图形
    plt.show()
            

if __name__ == "__main__":
    main("/home/zhangshenglin/chenshiqi/SDFVAE_pytorch/out_asd_2_w300/application-server-dataset")

    