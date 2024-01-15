import matplotlib.pyplot as plt
import numpy as np


def onclick(event):
    print(f'你点击的位置是：({event.xdata}, {event.ydata})')


if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.random.rand(10))

    fig.canvas.mpl_connect('button_press_event', onclick)

    plt.show()