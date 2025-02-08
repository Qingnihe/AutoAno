import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_npy_file(file_path):
    # 从.npy文件加载数据
    data = np.load(file_path)

    # 获取数据的维度
    data_length, num_dimensions = data.shape

    # 创建一个PDF文件
    pdf_file = PdfPages('output_plots.pdf')

    # 遍历每个维度并绘制图形
    for dim in range(num_dimensions):
        plt.figure(figsize=(12, 2))  # 设置图形大小
        plt.plot(data[:,dim])  # 绘制数据
        plt.title(f'Dimension {dim + 1}')  # 添加标题
        plt.xlabel('Data Index')  # x轴标签
        plt.ylabel('Value')  # y轴标签
        plt.tight_layout()

        # 将图形保存到PDF文件
        pdf_file.savefig()
        plt.close()

    # 关闭PDF文件
    pdf_file.close()

if __name__ == "__main__":
    npy_file_path = "/home/zhangshenglin/chenshiqi/GDN/out_nni_0118/yidong-22/noshare_409sd_64ws_0.001lr_32ed_5tk/result/11/test_score.npy"  # 替换成你的.npy文件路径
    plot_npy_file(npy_file_path)
