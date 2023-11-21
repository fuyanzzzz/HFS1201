# # 当前选用matplotlib版本3.3.2
# import matplotlib.pyplot as plt
# import random
#
# # 创建一个折线图
# fig = plt.figure()
#
# # 设置中文语言
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
# plt.rcParams['axes.unicode_minus'] = False
#
# # 创建四个表格，411代表创建4行1列，当前在1的位置
# ax = fig.add_subplot(2, 1, 1)
# bx = fig.add_subplot(2, 1, 2)
#
#
# # 给表的Y轴位置加上标签，rotation代表让文字横着展示，labelpad代表文字距表格多远了
# ax.set_ylabel('表一', rotation=0, fontsize=16, labelpad=20)
# bx.set_ylabel('表二', rotation=0, fontsize=16, labelpad=20)
#
# # 给定一个参数，用来标识是不是第一次创建
# line = None
#
# # 给定一个X轴和Y轴的参数列表，用作后面承载数据
# obsX = []
# obsY = []
#
# # 再给定一个X轴的开始位置，用作后面累加
# i = 1
#
# while True:
#     # 往列表插入展示的点的坐标
#     obsX.append(i)
#     # Y轴的话，由于没有实际数据，这里就用随机数代替
#     obsY.append(random.randrange(100, 200))
#
#     # 如果图还没有画，则创建一个画图
#     if line is None:
#         # -代表用横线画，g代表线的颜色是绿色，.代表，画图的关键点，用点代替。也可以用*，代表关键点为五角星
#         line = bx.plot(obsX, obsY, '-g', marker='.')[0]
#
#     # 这里插入需要画图的参数，由于图线，是由很多个点组成的，所以这里需要的是一个列表
#     line.set_xdata(obsX)
#     line.set_ydata(obsY)
#
#     # 我这里设计了一种方法，当X轴跑了100次的时候，则让X坐标的原点动起来
#     if len(obsX) < 100:
#         bx.set_xlim([min(obsX), max(obsX) + 30])
#     else:
#         bx.set_xlim([obsX[-80], max(obsX) * 1.2])
#
#     # Y轴的话我就没让他动了，然后加一个10，防止最高的订单顶到天花板
#     bx.set_ylim([min(obsY), max(obsY) + 10])
#
#     # 这个就是表的刷新时间了，以秒为单位
#     plt.pause(0.5)
#
#     # 画完一次了，i的数据加1，让X轴可以一直往前走。
#     i += 1
'''
动态折线图演示示例
'''

import numpy as np
import matplotlib.pyplot as plt
import random

plt.ion()
plt.figure(1)
t_list = []
result_list = []
t = 0

while True:
    # if t >= 10 * np.pi:
    #     plt.clf()
    #     t = 0
    #     t_list.clear()
    #     result_list.clear()
    # else:
    t += 1
    t_list.append(t)
    result_list.append(random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]))
    plt.plot(t_list, result_list, c='g', ls='-', mec='b', mfc='w')  # 保存历史数据
    # plt.plot(t_list, result_list)  ## 保存历史数据
    # plt.plot(t, np.sin(t), 'o')
    plt.pause(0.1)