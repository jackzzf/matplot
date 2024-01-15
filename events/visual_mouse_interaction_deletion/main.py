import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import Button
import matplotlib.path as mplPath


class removeAnomalyPoints:
    def __init__(self, fig, ax, dataset):
        self.fig = fig
        self.ax = ax
        self.dataset_orig = dataset  # 原始数据集
        self.dataset_last = dataset  # 保存上一步的数据集，便于stepback操作
        self.dataset_now = dataset  # 用于操作的数据集，实时维护
        self.dataset_now_withLabel = []
        self.roi = []
        self.press = False
        self.ax.scatter(dataset[:, 0], dataset[:, 1])
        self.button1_axes = plt.axes([0.5, 0.9, 0.1, 0.075])
        self.button1 = Button(self.button1_axes, 'clean')
        self.button2_axes = plt.axes([0.8, 0.9, 0.1, 0.075])
        self.button2 = Button(self.button2_axes, 'repaint')
        self.button3_axes = plt.axes([0.65, 0.9, 0.1, 0.075])
        self.button3 = Button(self.button3_axes, 'stepback')

    def on_press(self, event):
        if event.inaxes:  # 判断鼠标是否在axes内
            if event.button == MouseButton.LEFT:  # 判断按下的是否为鼠标左键
                # print("Start drawing")
                self.press = True

    def on_move(self, event):
        if event.inaxes:
            if self.press == True:
                x = event.xdata
                y = event.ydata
                self.roi.append([x, y])
                self.ax.plot(x, y, '.', c='r')  # 画点
                self.fig.canvas.draw()  # 更新画布

    def on_release(self, event):
        if self.press == True:
            self.press = False

    def get_dataset_now(self):
        return self.dataset_now

    def get_roi_now(self):
        return self.roi

    def get_deleted_dataset_now(self):  # 获取删除区域点后的dataset_now
        label_now = np.zeros(len(self.dataset_now))  # label_now长度始终与self.dataset_now保持一致
        self.roi = self.get_roi_now()
        poly_path = mplPath.Path(self.roi)

        for i in range(len(self.dataset_now)):
            if poly_path.contains_point(self.dataset_now[i]):
                label_now[i] = 1
            else:
                label_now[i] = 0

        self.dataset_now_withLabel = np.c_[self.dataset_now, label_now]
        deleted_dataset_now_withLabel = np.delete(self.dataset_now_withLabel,
                                                  np.where(self.dataset_now_withLabel[:, 2] == 1), axis=0)
        self.dataset_last = self.dataset_now
        self.dataset_now = deleted_dataset_now_withLabel[:, :2]
        return self.dataset_now

    def renew_img(self, event):
        try:
            self.get_deleted_dataset_now()  # 执行此函数，完成对self.dataset_now的更新
            self.ax.cla()
            self.ax.scatter(self.dataset_now[:, 0], self.dataset_now[:, 1])
            self.roi = []
            self.fig.canvas.draw()  # 更新画布
        except:
            print("Haven't selected area yet!")  # 异常处理

    def clean_to_orig(self, event):
        self.ax.cla()
        self.dataset_now = self.dataset_orig
        self.dataset_last = self.dataset_orig
        self.ax.scatter(self.dataset_now[:, 0], self.dataset_now[:, 1])
        self.roi = []
        self.fig.canvas.draw()  # 更新画布

    def stepback(self, event):
        self.dataset_now = self.dataset_last
        self.dataset_last = self.dataset_now
        self.ax.cla()
        self.ax.scatter(self.dataset_now[:, 0], self.dataset_now[:, 1])
        self.roi = []
        self.fig.canvas.draw()  # 更新画布

    def connect(self):
        self.fig.canvas.mpl_connect("button_press_event", self.on_press)
        self.fig.canvas.mpl_connect("button_release_event", self.on_release)
        self.fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.fig.canvas.mpl_connect("key_press_event", self.clean_to_orig)
        self.button1.on_clicked(self.renew_img)
        self.button2.on_clicked(self.clean_to_orig)
        self.button3.on_clicked(self.stepback)


def generate_data():
    # 生成二次函数数据点
    num_quadratic = 800  # 二次函数数据点数量
    x_quadratic = np.linspace(-10, 10, num_quadratic)
    y_quadratic = x_quadratic ** 2 + np.random.randn(num_quadratic) * 10

    # 生成异常点
    num_outliers = 200  # 异常点数量
    x_outliers = np.random.uniform(-10, 10, num_outliers)
    y_outliers = np.random.uniform(-300, 300, num_outliers)

    # 合并数据点
    x_data = np.concatenate((x_quadratic, x_outliers))
    y_data = np.concatenate((y_quadratic, y_outliers))

    # 打乱数据点顺序
    indices = np.random.permutation(x_data.shape[0])
    x_data = x_data[indices]
    y_data = y_data[indices]

    return x_data, y_data


if __name__ == '__main__':
    ##1.以下部分进行异常点去除
    data_x, data_y = generate_data()  # 生成一个含异常点的二次函数数据集
    data = np.c_[data_x, data_y]
    dataset_orig = np.array(data)  # dataset_orig是n*2, array类
    fig1, ax1 = plt.subplots()
    img = removeAnomalyPoints(fig1, ax1, dataset_orig)
    img.connect()
    plt.show()
    print("The removal of the anomaly has been completed!")
