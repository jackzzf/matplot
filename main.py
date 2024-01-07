# -*-coding:utf-8-*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np


def simpleUse():
    # 50 points taken evenly from (-1, 1)
    x = np.linspace(-1, 1, 50)
    y = 2 * x
    plt.plot(x, y)
    plt.show()


def figureObject():
    x = np.linspace(-1, 1, 50)
    y1 = x ** 2
    y2 = x * 2
    # this is first figure object, all of the
    # following are displayed in the first figure
    plt.figure()
    plt.plot(x, y1)
    # this is the second figure object, you can set
    # figure num(will be displayed in the window title), and figure size
    plt.figure(num=3, figsize=(10, 5))
    # you can set line color or width or style
    plt.plot(x, y2, color='red', linewidth=3, linestyle='--')
    plt.show()


def setAxes01():
    # 50 points taken evenly from (-1, 1)
    x = np.linspace(-1, 1, 50)
    y = 2 * x
    # set the range of x/y axes
    plt.xlim((0, 2))
    plt.ylim((-2, 2))
    # set the name of x/y axes
    plt.xlabel("x label")
    plt.ylabel("y label")
    plt.plot(x, y, color='g')
    plt.show()


def setAxes02():
    # 50 points taken evenly from (-1, 1)
    x = np.linspace(-1, 1, 50)
    y = 2 * x
    # create a figure object
    plt.figure(figsize=(10, 8))
    # set the range of x/y axes
    plt.xlim((0, 2))
    plt.ylim((-2, 2))
    # set the name of x/y axes
    plt.xlabel("x label")
    plt.ylabel("y label")
    # change the axes to different units
    new_ticks = np.linspace(-1, 2, 5)
    plt.xticks(new_ticks)
    # change the name at the corresponding coordinates
    # standard font
    plt.yticks([-2, -1, 0, 1, 2], ['really bad', 'b', 'c', 'd', 'good'])
    # math font
    plt.yticks([-2, -1, 0, 1, 2], [r'$really\ bad$', r'$b$', r'$c\ \alpha$', 'd', 'good'])
    plt.plot(x, y, color='g')
    plt.show()


def setAxes03():
    # 50 points taken evenly from (-1, 1)
    x = np.linspace(-1, 1, 50)
    y = 2 * x
    # set the name of x/y axes
    # plt.xlabel("x label")
    # plt.ylabel("y label")
    plt.plot(x, y, color='g')
    # gca = 'get current axis', get current four axis
    ax = plt.gca()
    # set the color of the four axis(top/bottom/left/right)
    ax.spines['right'].set_color('r')
    ax.spines['top'].set_color('none')
    # 将底坐标轴作为x轴
    ax.xaxis.set_ticks_position('bottom')
    # 设置x轴的位置(依据的是y轴)
    # set_position, the 1st is in 'outward' |'axes' | 'data'
    # axes : precentage of y axis
    # data : depend on y data
    ax.spines['bottom'].set_position(('data', 0))
    # 将左坐标轴作为y轴
    ax.yaxis.set_ticks_position('left')
    # 设置y轴的位置(依据的是x轴)
    ax.spines['left'].set_position(('data', 0))
    plt.show()


def figureLegend():
    # 50 points taken evenly from (-1, 1)
    x = np.linspace(-1, 1, 50)
    y1 = x ** 2
    y2 = x * 2
    # set the name of x/y axes
    plt.xlabel("x label")
    plt.ylabel("y label")
    l1, = plt.plot(x, y1, color='g', label='linear line')
    l2, = plt.plot(x, y2, color='r', label='square line')
    # simple use, set location
    '''
        The *loc* location codes are::
              'best' : 0,         
              'upper right'  : 1,
              'upper left'   : 2,
              'lower left'   : 3,
              'lower right'  : 4,
              'right'        : 5,
              'center left'  : 6,
              'center right' : 7,
              'lower center' : 8,
              'upper center' : 9,
              'center'       : 10,
    '''
    plt.legend(loc='upper right')
    # advanced use(set labels/shadow)
    plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best', shadow=True)
    # setting the legend style
    legend = plt.legend(handles=[l1, l2], labels=['up', 'down'], loc='best', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('w')
    plt.show()


def addAnnotation():
    x = np.linspace(-3, 3, 50)
    y = 2 * x + 1

    plt.figure(num=1, figsize=(8, 5))
    plt.plot(x, y)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # 将底下的作为x轴
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    # 将左边的作为y轴
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    print("-----方式一-----")
    x0 = 1
    y0 = 2 * x0 + 1
    plt.plot([x0, x0], [0, y0], 'k--', linewidth=2.5, color='r')
    plt.scatter([x0], [y0], s=50, color='b')
    '''
        1. xy就是需要进行注释的点的横纵坐标；
        2. xycoords = 'data' 说明的是要注释点的xy的坐标是以横纵坐标轴为基准的；
        3. xytext = (+30,-30) 和 textcoords = 'offset points' 说明了这里的文字是
           基于标注的点的x坐标的偏移+30以及标注点y坐标-30位置，就是我们要进行注释文字的位置；
        4. fontsize = 16 就说明字体的大小；
        5. arrowprops = dict() 这个是对于这个箭头的描述，arrowstyle = '->' 这个是箭头的类型，
           connectionstyle = "arc3,rad=.2" 这两个是描述我们的箭头的弧度以及角度的。
    '''
    plt.annotate(r'$2x+1 = %s$' % y0, xy=(x0, y0), xycoords='data',
                 xytext=(+30, -30), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))

    print("-----方式二-----")
    plt.text(-3.7, 3, r'$this\ is\ the\ some\ text. \mu\ \sigma_i\ \alpha_t$',
             fontdict={'size': 16, 'color': 'r'})
    plt.show()


def drawingOrder():
    x = np.linspace(-3, 3, 50)
    y1 = 0.1 * x
    y2 = x ** 2

    plt.figure()
    # zorder 控制绘图顺序
    plt.plot(x, y1, linewidth=10, zorder=0, label=r'$y_1\ =\ 0.1*x$')
    plt.plot(x, y2, linewidth=10, zorder=1, label=r'$y_2\ =\ x^{2}$')

    plt.legend(loc='lower right')
    plt.show()


def drawingOrderCover():
    x = np.linspace(-3, 3, 50)
    y1 = 0.1 * x
    y2 = x ** 2

    plt.figure()
    # zorder 控制绘图顺序
    plt.plot(x, y1, linewidth=10, zorder=0, label=r'$y_1\ =\ 0.1*x$')
    plt.plot(x, y2, linewidth=10, zorder=1, label=r'$y_2\ =\ x^{2}$')

    plt.ylim(-2, 2)

    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    print(ax.get_xticklabels())
    print(ax.get_yticklabels())

    '''
        1. ax.get_xticklabels()获取得到就是坐标轴上的数字；
        2. set_bbox()这个bbox就是那坐标轴上的数字的那一小块区域，从结果我们可以很明显的看出来；
        3. facecolor = 'white', edgecolor='none，第一个参数表示的这个box的前面的背景，边上的颜色
    '''
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(12)
        label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, zorder=2))

    plt.legend(loc='lower right')
    plt.show()


def scatterFigure():
    n = 1024
    X = np.random.normal(0, 1, n)
    Y = np.random.normal(0, 1, n)
    T = np.arctan2(Y, X)  # for color later on

    plt.scatter(X, Y, s=75, c=T, alpha=.5)

    plt.xlim((-1.5, 1.5))
    plt.xticks([])  # ignore xticks
    plt.ylim((-1.5, 1.5))
    plt.yticks([])  # ignore yticks
    plt.show()


def barFigure():
    n = 12
    X = np.arange(n)
    Y1 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    Y2 = (1 - X / float(n)) * np.random.uniform(0.5, 1.0, n)
    # facecolor:表面的颜色;edgecolor:边框的颜色
    plt.bar(X, +Y1, facecolor='#9999ff', edgecolor='white')
    plt.bar(X, -Y2, facecolor='#ff9999', edgecolor='white')
    # 描绘text在图表上
    # plt.text(0 + 0.4, 0 + 0.05, "huhu")
    for x, y in zip(X, Y1):
        # ha : horizontal alignment
        # va : vertical alignment
        plt.text(x + 0.01, y + 0.05, '%.2f' % y, ha='center', va='bottom')

    for x, y in zip(X, Y2):
        # ha : horizontal alignment
        # va : vertical alignment
        plt.text(x + 0.01, -y - 0.05, '%.2f' % (-y), ha='center', va='top')

    plt.xlim(-.5, n)
    plt.xticks([])  # ignore xticks
    plt.ylim(-1.25, 1.25)
    plt.yticks([])  # ignore yticks
    plt.show()


def imageFigure():
    # image data
    a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
                  0.365348418405, 0.439599930621, 0.525083754405,
                  0.423733120134, 0.525083754405, 0.651536351379]).reshape(3, 3)

    '''
    for the value of "interpolation"，check this:
    http://matplotlib.org/examples/images_contours_and_fields/interpolation_methods.html
    for the value of "origin"= ['upper', 'lower'], check this:
    http://matplotlib.org/examples/pylab_examples/image_origin.html
    '''
    # 显示图像
    # 这里的cmap='bone'等价于plt.cm.bone
    plt.imshow(a, interpolation='nearest', cmap='bone', origin='up')
    # 显示右边的栏
    plt.colorbar(shrink=.92)

    # ignore ticks
    plt.xticks([])
    plt.yticks([])
    plt.show()


def threedFigure():
    fig = plt.figure()
    ax = Axes3D(fig)
    # X Y value
    X = np.arange(-4, 4, 0.25)
    Y = np.arange(-4, 4, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    # hight value
    Z = np.sin(R)

    """
    ============= ================================================
            Argument      Description
            ============= ================================================
            *X*, *Y*, *Z* Data values as 2D arrays
            *rstride*     Array row stride (step size), defaults to 10
            *cstride*     Array column stride (step size), defaults to 10
            *color*       Color of the surface patches
            *cmap*        A colormap for the surface patches.
            *facecolors*  Face colors for the individual patches
            *norm*        An instance of Normalize to map values to colors
            *vmin*        Minimum value to map
            *vmax*        Maximum value to map
            *shade*       Whether to shade the facecolors
            ============= ================================================
    """
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))

    # I think this is different from plt12_contours
    """
    ==========  ================================================
            Argument    Description
            ==========  ================================================
            *X*, *Y*,   Data values as numpy.arrays
            *Z*
            *zdir*      The direction to use: x, y or z (default)
            *offset*    If specified plot a projection of the filled contour
                        on this position in plane normal to zdir
            ==========  ================================================
    """
    ax.contourf(X, Y, Z, zdir='z', offset=-2, cmap=plt.get_cmap('rainbow'))
    ax.set_zlim(-2, 2)
    plt.show()


def useSubplot01():
    plt.figure(figsize=(6, 5))

    ax1 = plt.subplot(3, 1, 1)
    ax1.set_title("ax1 title")
    plt.plot([0, 1], [0, 1])

    # 这种情况下如果再数的话以3,3,4为标准了，
    # 把上面的第一行看成是3个列
    ax2 = plt.subplot(3, 3, 4)
    ax2.set_title("ax2 title")
    ax3 = plt.subplot(3, 3, 5)
    ax4 = plt.subplot(3, 3, 6)
    # 把上面的第一,二行分别看成是2个列
    ax5 = plt.subplot(3, 2, 5)
    ax6 = plt.subplot(3, 2, 6)

    plt.show()


def useSubplot02():
    plt.figure(figsize=(6, 4))
    # plt.subplot(n_rows,n_cols,plot_num)
    plt.subplot(2, 1, 1)
    # figure splits into 2 rows, 1 col, plot to the 1st sub-fig
    plt.plot([0, 1], [0, 1])

    plt.subplot(2, 3, 4)
    # figure splits into 2 rows, 3 col, plot to the 4th sub-fig
    plt.plot([0, 1], [0, 2])

    plt.subplot(2, 3, 5)
    # figure splits into 2 rows, 3 col, plot to the 5th sub-fig
    plt.plot([0, 1], [0, 3])

    plt.subplot(2, 3, 6)
    # figure splits into 2 rows, 3 col, plot to the 6th sub-fig
    plt.plot([0, 1], [0, 4])

    plt.tight_layout()
    plt.show()


def useSubplot2Grid():
    plt.figure()
    # 第一个参数shape也就是我们网格的形状
    # 第二个参数loc,位置,这里需要注意位置是从0开始索引的
    # 第三个参数colspan跨多少列,默认是1
    # 第四个参数rowspan跨多少行,默认是1
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
    ax1.set_title("ax1_title")

    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2, rowspan=1)
    ax3 = plt.subplot2grid((3, 3), (1, 2), colspan=1, rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=1, rowspan=1)
    ax5 = plt.subplot2grid((3, 3), (2, 1), colspan=1, rowspan=1)

    plt.show()


def useGridSpec():
    plt.figure()
    gs = gridspec.GridSpec(3, 3)
    # use index from 0
    # 第0行，第0-2列
    ax1 = plt.subplot(gs[0, :])
    ax1.set_title("ax1 title")
    # 第1行，第0-1列
    ax2 = plt.subplot(gs[1, :2])
    ax2.plot([1, 2], [3, 4], 'r')
    # 第1-2行，第2列
    ax3 = plt.subplot(gs[1:, 2:])
    # 负值怎么理解？？？
    ax4 = plt.subplot(gs[-1, 0])
    ax5 = plt.subplot(gs[-1, -2])

    plt.show()


def useDefStructure():
    # 这种方式不能生成指定跨行列的那种
    # (ax11,ax12),(ax13,ax14)代表了两行
    # f就是figure对象,
    # sharex: 是否共享x轴
    # sharey: 是否共享y轴
    f, ((ax11, ax12), (ax13, ax14)) = plt.subplots(2, 2, sharex=False, sharey=True)
    ax11.set_title("a11 title")
    ax12.scatter([1, 2], [1, 2])

    plt.show()


def figInFigure():
    fig = plt.figure()
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [1, 3, 4, 2, 5, 8, 6]

    # below are all percentage
    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    # 使用plt.figure()显示的是一个空的figure
    # 如果使用fig.add_axes会添加轴
    ax1 = fig.add_axes([left, bottom, width, height])  # main axes
    ax1.plot(x, y, 'r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('title')

    # inside axes
    ax2 = fig.add_axes([0.2, 0.6, 0.25, 0.25])
    ax2.plot(y, x, 'b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('title inside 1')

    # different method to add axes
    plt.axes([0.6, 0.2, 0.25, 0.25])
    plt.plot(y[::-1], x, 'g')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('title inside 2')

    plt.show()


def secondaryAxis():
    x = np.arange(0, 10, 0.1)
    y1 = 0.05 * x ** 2
    y2 = -1 * y1

    fig, ax1 = plt.subplots()

    ax2 = ax1.twinx()
    ax1.plot(x, y1, 'g-')
    ax2.plot(x, y2, 'b-')

    ax1.set_xlabel('X data')
    ax1.set_ylabel('Y1 data', color='g')
    ax2.set_ylabel('Y2 data', color='b')

    plt.show()


def useAnimation():
    fig, ax = plt.subplots()

    x = np.arange(0, 2 * np.pi, 0.01)
    # 因为这里返回的是一个列表，但是我们只想要第一个值
    # 所以这里需要加,号
    line, = ax.plot(x, np.sin(x))

    def animate(i):
        line.set_ydata(np.sin(x + i / 10.0))  # updata the data
        return line,

    def init():
        line.set_ydata(np.sin(x))
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    # blit=True dose not work on Mac, set blit=False
    # interval= update frequency
    # frames帧数
    ani = animation.FuncAnimation(fig=fig, func=animate, frames=100, init_func=init,
                                  interval=20, blit=False)

    plt.show()


if __name__ == '__main__':
    # 2. simple use
    # simpleUse()

    # 3. figure object
    # figureObject()

    # 4. set the axes
    # setAxes01()
    # setAxes02()
    # setAxes03()

    # 5. figure legend
    # figureLegend()

    # 6. add annotation
    # addAnnotation()

    # 7. control of drawing order
    # drawingOrder()
    # drawingOrderCover()

    # 8. types of drawings
    # scatterFigure()
    # barFigure()
    # imageFigure()
    # threedFigure()

    # 9. subplot
    # useSubplot01()
    # useSubplot02()
    # useSubplot2Grid()
    # useGridSpec()
    # useDefStructure()

    # 10. figure in figure
    # figInFigure()
    # secondaryAxis()

    # 11. animation
    useAnimation()
