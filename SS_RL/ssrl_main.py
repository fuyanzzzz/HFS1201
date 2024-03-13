'''
-------------------------------------------------
File Name: ssrl_main.py
Author: LRS
Create Time: 2023/6/5 21:47
-------------------------------------------------
'''
import os
import time

import pandas as pd
import numpy as np
# from torchvision.utils import save_image

from SS_RL import neighbo_search
from SS_RL.Schedule import Schedule_Instance
from SS_RL.diagram import job_diagram

path = r'C:\paper_code_0501\HFS1201\useful0424\data'
from config import DataInfo
from SS_RL.RL_ import RL_Q
from SS_RL.inital_solution import HFS
# from SS_RL.schedule_cal import ScheduleCal
import matplotlib.pyplot as plt


N_STATES = 12
ACTIONS = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
# 1. 生成初始解，这个没有问题
actions = range(6)
max_iter = 3
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# # q_table = pd.DataFrame(np.random.rand(12, 11),columns=actions)
q_table = pd.DataFrame(
            np.random.uniform(-1, 1, size=(N_STATES, len(actions))),columns=actions)
# q_table = pd.DataFrame(
#             np.zeros((N_STATES, len(actions))),columns=actions)

# q_table = pd.DataFrame(
#             np.zeros((N_STATES, len(actions))),columns=actions)

data_folder = "data"  # 数据文件夹的路径
# 获取data文件夹下所有的文件名

# 新建一个变量，用于存储每个动作是否有改进，以及改进多少
use_action_dict = {}
action_space = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
for i_action in action_space:
    use_action_dict[i_action] = [0,0]

train = False
text = True
trial_list = [10]



# stop_iter_list = [20]
a_list = [0]
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
iter = 0
i_iter = 0

jingying_i = [1/3,1/4,1/5]
i_discount_rate = [0.8,0.9,1]
stop_iter_list = [20,30,50]
lr_rate = [0.01,0.05,0.1]

i_key = [i for i in range(10)]
i_value = [(1/5,1/2,0.01,0.8),
           (1/5,1,0.05,0.9),
           (1/5,3/2,0.1,1),

           (1/4,1/2,0.05,1),
           (1/4,1,0.1,0.8),
           (1/4,3/2,0.01,0.9),

           (1/3,1/2,0.1,0.9),
           (1/3,1,0.01,1.0),
           (1/3,3/2,0.05,0.8),

           (1/3,3/2,0.1,0.8)
           ]

tiankou_table = dict(zip(i_key, i_value))


if train is True:
    # for i_stop_iter in a_list:
    # for a in a_list:
    a = 1
    i_trial = 10
    # for i_trial in trial_list:
    for i_tiankou in range(9,10):
        # if i_tiankou == 3 or i_tiankou == 4 or i_tiankou == 5 or i_tiankou == 6 or i_tiankou == 7:
        #     continue
        q_table = pd.DataFrame(
            np.random.uniform(-1, 1, size=(N_STATES, len(actions))), columns=actions)
        item = tiankou_table[i_tiankou]
        jingying = item[0]


        lr = item[2]
        discount_rate = item[3]

        q_value_changes = []
        CUM_REWARD = []
        case_CUM_REWARD = []
        case_CUM_obj = []
        INDEX = []
        # try:
        for i_iter in range(4):
            for index, file_name in enumerate(txt_files):
                split_list = file_name.split('_')
                if (int(split_list[0]) - i_iter) % 5 != 0:
                    continue
                if (int(split_list[0]) - 4) % 5 == 0:
                    continue


                time_cost = 0
                start_time = time.time()
                # file_name = '1258_Instance_20_2_3_0,6_1_20_Rep3.txt'
                # if index == 0:
                #     continue
                print('换数据集啦{0}'.format(file_name))
                print('第{0}幕'.format(index))
                job_num = int(split_list[2])
                machine_num_per_stage = int(split_list[4])
                i_stop_iter = int(job_num * machine_num_per_stage * item[1])
                hfs = HFS(file_name,int(job_num *2 *jingying))

                hfs.initial_solu()

                rl_ = RL_Q(N_STATES,ACTIONS,hfs.inital_refset,q_table,file_name,len(INDEX),i_stop_iter,a,i_trial,lr,jingying,discount_rate,hfs.population_refset)
                hfs.inital_refset,q_table,delta, REWARD = rl_.rl()
                print(hfs.inital_refset[0][1])
                opt_item = hfs.inital_refset[0]
                schedule = opt_item[0]
                obj = opt_item[1]
                job_execute_time = opt_item[2]

                end_time = time.time()

                # 计算函数运行时长
                duration = end_time - start_time

                q_value_changes.append(delta)
                CUM_REWARD.append(REWARD)

                INDEX.append(len(CUM_REWARD))

                fp = open('./time_cost.txt', 'a+')
                if index == 0:
                    print('索引   文件名   耗时  数据最优解   实验最优解 ', file=fp)
                print('{0}   {1}   {2}   {3}   {4} '.format(len(INDEX),file_name,round(duration,2),rl_.config.ture_opt,rl_.best_opt), file=fp)
                fp.close()

                with open('./time_cost.txt', 'a+') as fp:
                    # 设置显示选项
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)

                    # 将 DataFrame 写入文件
                    # print(index, q_table, file=fp)

                    if index == 0:
                        print('索引   文件名   耗时  数据最优解   实验最优解 ', file=fp)
                    print()
                    print('{0}   {1}   {2}   {3}   {4} '.format(len(txt_files) * iter + index, file_name, round(duration, 2),
                                                                rl_.config.ture_opt, rl_.best_opt), file=fp)
                    # sort_time = 0
                    # effe_time = 0
                    # rand_time = 0
                    # for item in rl_.use_actions.keys():
                    #     # print(item, rl_.use_actions[item],round(rl_.use_actions[item][1]/max(rl_.use_actions[item][0],1),3), file=fp)
                    #     if item[:4] == 'sort':
                    #         sort_time += rl_.use_actions[item][0]
                    #     elif item[:4] == 'effe':
                    #         effe_time += rl_.use_actions[item][0]
                    #     else:
                    #         rand_time += rl_.use_actions[item][0]
                    # print('sort_time    ',sort_time, file=fp)
                    # print('effe_time    ',effe_time, file=fp)
                    # print('rand_time    ',rand_time, file=fp)
                    # print('total_time    ',rand_time+sort_time+effe_time, file=fp)
                    # print('effe_time占比    ',round(effe_time/(rand_time+sort_time+effe_time),2), file=fp)
                    # print('effe_time + rand_time 占比    ',round((effe_time+rand_time)/(rand_time+sort_time+effe_time),2), file=fp)
                    # 重置显示选项为默认值
                    pd.reset_option('display.max_rows')
                    pd.reset_option('display.max_columns')

                # from SS_RL.diagram import job_diagram
                # import matplotlib.pyplot as plt

                # dia = job_diagram(schedule, job_execute_time, file_name, len(txt_files)*iter +index)
                # dia.pre()
                # plt.savefig('./img1203/pic-{}.png'.format(len(txt_files)*iter +index))
                # plt.savefig('./img1203/pic-{}.png'.format(len(txt_files)*iter +index))


                # 每过一幕验证一下奖励
                time_cost = 0
                start_time = time.time()
                # case_file_name = '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt'
                # # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
                # hfs = HFS(case_file_name)
                #
                # hfs.initial_solu()
                #
                # print('当前目标值：{0}'.format(hfs.inital_refset[0][1]))
                #
                # # dia = job_diagram(hfs.inital_refset[0][0], hfs.inital_refset[0][2], case_file_name, index)
                # # dia.pre()
                # # plt.savefig('./img1203/pic-{}.png'.format(index))
                # # plt.show()
                #
                # rl_ = RL_Q(N_STATES,ACTIONS,hfs.inital_refset,q_table,case_file_name,len(INDEX),i_stop_iter,a,i_trial,lr,jingying,discount_rate)
                # best_opt_execute ,CUM_REWARD_case= rl_.rl_execute()
                # with open('./MDP.txt', 'a+') as fp:
                #     print('幕：{2},    目标值:{0},   奖励:{1},'.format(best_opt_execute, CUM_REWARD_case,len(txt_files)*iter +index), file=fp)
                #     print('', file=fp)
                #     print('', file=fp)
                #
                # case_CUM_obj.append(best_opt_execute)
                # case_CUM_REWARD.append(CUM_REWARD_case)
                end_time = time.time()
                duration = end_time - start_time

                with open('./time_cost.txt', 'a+') as fp:
                    # 设置显示选项
                    pd.set_option('display.max_rows', None)
                    pd.set_option('display.max_columns', None)

                    # 将 DataFrame 写入文件
                    # print(index, q_table, file=fp)

                    # if index == 0:
                    #     print('索引   文件名   耗时  数据最优解   实验最优解 ', file=fp)
                    # print()
                    # print('{0}   {1}   {2}   {3}   {4} '.format(len(txt_files) * iter + index, case_file_name, round(duration, 2),
                    #                                             rl_.config.ture_opt, rl_.best_opt), file=fp)
                    # sort_time = 0
                    # effe_time = 0
                    # rand_time = 0
                    # for item in rl_.use_actions.keys():
                    #     print(item, rl_.use_actions[item],round(rl_.use_actions[item][1]/max(rl_.use_actions[item][0],1),3), file=fp)
                    #     if item[:4] == 'sort':
                    #         sort_time += rl_.use_actions[item][0]
                    #     elif item[:4] == 'effe':
                    #         effe_time += rl_.use_actions[item][0]
                    #     else:
                    #         rand_time += rl_.use_actions[item][0]

                # from SS_RL.diagram import job_diagram
                # import matplotlib.pyplot as plt
                #
                # dia = job_diagram(rl_.schedule, rl_.job_execute_time, rl_.file_name, index)
                # dia.pre()
                # plt.savefig('./img1203/pic-{}.png'.format(index))




                # plt.plot(q_value_changes)
                # plt.xlabel('训练轮次')
                # plt.ylabel('Q值变化')
                # plt.title('Q值变化随训练轮次的变化')
                # plt.pause(0.1)  # 用于动态展示图像

                if (len(INDEX)) % 20 == 0:
                    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)
                    #
                    # ax1.plot(INDEX, case_CUM_obj, label='', color='yellow')
                    # ax1.set_ylabel('实验案例目标值')
                    # ax1.legend()
                    #
                    # ax2.plot(INDEX, CUM_REWARD, label='', color='red')
                    # ax2.set_ylabel('累计奖励')
                    # ax2.legend()
                    #
                    # ax3.plot(INDEX, case_CUM_REWARD, label='', color='green')
                    # ax3.set_ylabel('实验案例')
                    # ax3.legend()
                    #
                    # ax4.plot(INDEX, q_value_changes, label='', color='blue')
                    # ax4.set_ylabel('Q值变化程度')
                    # ax4.legend()

                    fig, (ax1) = plt.subplots(1, 1, sharex=True)

                    ax1.plot(INDEX, q_value_changes, label='', color='blue')
                    ax1.set_ylabel('Q值变化程度')
                    ax1.legend()

                    # 调整子图之间的垂直间距
                    plt.tight_layout()
                    # plt.pause(0.1)  # 用于动态展示图像
                    plt.savefig('./tiankou0304_termination/img0304_{}/pic-{}.png'.format(i_tiankou,len(CUM_REWARD)))

                    with open('./tiankou0304_termination/0304_q_{}.txt'.format(i_tiankou), 'a+') as fp:
                        # 设置显示选项
                        pd.set_option('display.max_rows', None)
                        pd.set_option('display.max_columns', None)

                        # 将 DataFrame 写入文件
                        print(len(INDEX), q_table, file=fp)
                        # for item in rl_.use_actions.keys():
                        #     print(item,rl_.use_actions[item], file=fp)

                        # 重置显示选项为默认值
                        pd.reset_option('display.max_rows')
                        pd.reset_option('display.max_columns')


                # fp = open('./0.05_0.9_1207_2.txt', 'a+')
                # print(len(txt_files)*iter +index,rl_.q_table, file=fp)
                # fp.close()
        iter += 1
        # except:
        #     continue


def mian_mult(file_name):
    for tiankou_index in range(9, 10):
        if tiankou_index == 0:
            q_table.loc[0, :] = [-0.938371, 0.601020, 0.500611, 0.864727, 0.594710, -0.313630]
            q_table.loc[1, :] = [0.006826, -0.182103, -0.631414, -0.409230, 0.092903, -0.434484]
            q_table.loc[2, :] = [0.154912, -0.203984, 0.015967, -0.659772, 0.309395, -0.389682]
            q_table.loc[3, :] = [-0.098166, 0.256405, -0.943949, 0.331508, -0.539271, -0.980787]
            q_table.loc[4, :] = [0.285390, 0.272946, 0.296852, -0.430623, 0.310168, 0.245958]
            q_table.loc[5, :] = [0.360592, 0.280863, 0.316082, 0.247771, 0.362280, 0.285784]
            q_table.loc[6, :] = [-0.195092, 0.911389, 0.172999, -0.053111, -0.842334, -0.700462]
            q_table.loc[7, :] = [0.200727, 0.186540, -0.717979, 0.186205, 0.210538, 0.190497]
            q_table.loc[8, :] = [-0.628102, -0.243074, 0.222937, 0.208056, 0.265296, 0.237309]
            q_table.loc[9, :] = [0.790262, 0.689505, 0.439669, -0.832898, 0.755999, -0.334612]
            q_table.loc[10, :] = [0.478973, 0.433207, 0.508669, -0.159163, -0.170423, 0.145674]
            q_table.loc[11, :] = [-0.368779, -0.423740, 0.033933, -0.274561, 0.637626, -0.135194]
        elif tiankou_index == 1:
            q_table.loc[0, :] = [0.290213, 0.781718, 0.932620, -0.509302, -0.250566, 0.736934]
            q_table.loc[1, :] = [-0.356790, -0.456773, -0.407873, -0.403042, -0.306490, -0.446384]
            q_table.loc[2, :] = [-0.093485, -0.364620, -0.313007, -0.340677, -0.022081, -0.343261]
            q_table.loc[3, :] = [0.358468, -0.133309, 0.156880, -0.855869, 0.564713, 0.252220]
            q_table.loc[4, :] = [0.202231, 0.197209, 0.181775, 0.194742, 0.193059, 0.164528]
            q_table.loc[5, :] = [0.085031, 0.040049, 0.143566, 0.079575, 0.121373, 0.072231]
            q_table.loc[6, :] = [0.326706, -0.013143, 0.508957, 0.405012, 0.399481, 0.427739]
            q_table.loc[7, :] = [0.037289, 0.033914, 0.045465, 0.051796, 0.049162, 0.052272]
            q_table.loc[8, :] = [0.095985, 0.087603, 0.094072, 0.092787, 0.098821, 0.091657]
            q_table.loc[9, :] = [-0.696659, 0.456816, 0.834546, 0.916215, 0.645481, 0.462936]
            q_table.loc[10, :] = [0.733520, 0.897248, 0.707564, -0.022689, 0.046451, 0.959779]
            q_table.loc[11, :] = [0.246563, 0.015000, 0.288835, -0.821822, -0.231140, 0.083465]
        elif tiankou_index == 2:
            q_table.loc[0, :] = [-0.989139, -0.048401, 0.066099, 1.028513, -0.010294, 0.132813]
            q_table.loc[1, :] = [4.977766, 4.709525, 4.763428, -0.073024, 5.044245, 4.585672]
            q_table.loc[2, :] = [3.600836, 3.325947, 3.318326, 3.049944, 3.381454, -0.178734]
            q_table.loc[3, :] = [1.198978, 0.820822, 0.977698, -0.172480, -0.095274, 0.607618]
            q_table.loc[4, :] = [5.386371, 5.362784, 5.398083, 5.380670, 5.370161, 5.377495]
            q_table.loc[5, :] = [4.098199, 3.978090, 3.915621, 4.037348, 3.974627, 3.966494]
            q_table.loc[6, :] = [0.987393, 0.897035, 0.991886, 0.880388, 0.969252, 0.90662]
            q_table.loc[7, :] = [5.126257, 5.131961, 4.609369, 5.048285, 4.859344, 5.159743]
            q_table.loc[8, :] = [3.822270, 3.822062, 3.784294, 3.820533, 3.477910, 3.432783]
            q_table.loc[9, :] = [-0.114816, -0.642625, -0.864023, 0.028326, 0.672632, -0.367072]
            q_table.loc[10, :] = [0.332579, -0.984244, -0.805196, 0.774695, 0.358116, -0.801771]
            q_table.loc[11, :] = [-0.920222, 0.724798, -0.140493, -0.637320, 0.369087, -0.092888]
        elif tiankou_index == 3:
            q_table.loc[0, :] = [0.274096, -0.118370, 0.520648, -0.262838, 0.480391, 0.518670]
            q_table.loc[1, :] = [2.492862, 2.300515, 2.438407, 1.975442, 2.560490, 1.936369]
            q_table.loc[2, :] = [2.518901, 2.115278, 2.199465, 1.599073, 2.445078, 1.648880]
            q_table.loc[3, :] = [0.547742, 0.294825, 0.541405, -0.519497, 0.543496, 0.71028]
            q_table.loc[4, :] = [2.719419, 2.780019, 2.755309, 2.751212, 2.885222, 2.813951]
            q_table.loc[5, :] = [2.714071, 2.697190, 2.497272, 2.646961, 2.746935, 2.723779]
            q_table.loc[6, :] = [-0.971764, -0.667426, 0.010838, 0.529272, 0.285134, -0.455346]
            q_table.loc[7, :] = [1.653497, 1.943276, 1.887048, 1.879692, 2.082153, -0.542944]
            q_table.loc[8, :] = [1.853017, 1.563552, 1.814598, 1.641716, 2.004640, 1.563169]
            q_table.loc[9, :] = [0.776324, 0.078662, -0.367723, 0.555037, -0.288179, -0.030205]
            q_table.loc[10, :] = [-0.092628, 0.772135, -0.585268, -0.747303, 0.526260, 0.970319]
            q_table.loc[11, :] = [0.226827, 0.096366, -0.712502, 0.501877, -0.662874, -0.969642]
        elif tiankou_index == 4:
            q_table.loc[0, :] = [0.477134, 0.316209, 0.410613, -0.454672, 0.277682, -0.618554]
            q_table.loc[1, :] = [-0.304350, -0.509220, -0.524477, -0.639730, -0.222643, -0.63856]
            q_table.loc[2, :] = [0.025752, -0.675120, -0.238971, -0.674417, 0.175185, -0.674669]
            q_table.loc[3, :] = [0.170985, -0.565752, -0.096403, -0.052960, 0.042134, 0.101518]
            q_table.loc[4, :] = [0.133997, 0.112579, 0.172088, 0.111345, 0.136896, 0.107334]
            q_table.loc[5, :] = [0.179458, 0.191248, 0.134498, -0.217738, 0.191119, 0.046771]
            q_table.loc[6, :] = [-0.451474, 0.144647, 0.205122, 0.135775, 0.199105, 0.176849]
            q_table.loc[7, :] = [-0.071113, 0.052450, 0.043294, 0.049294, 0.047618, 0.043858]
            q_table.loc[8, :] = [0.022404, 0.020079, 0.023805, 0.022640, -0.010805, 0.023605]
            q_table.loc[9, :] = [0.643394, 0.611641, -0.747095, -0.806911, -0.160150, -0.151743]
            q_table.loc[10, :] = [-0.302981, 0.572053, -0.508141, -0.444518, -0.128679, 0.143945]
            q_table.loc[11, :] = [-0.285382, -0.276600, 0.742320, 0.796914, -0.801778, -0.082559]
        elif tiankou_index == 5:
            q_table.loc[0, :] = [0.730153, 0.052891, -0.812951, -0.700529, -0.510331, -0.257622]
            q_table.loc[1, :] = [-0.031761, -0.255184, -0.250401, -0.743387, -0.070438, -0.485420]
            q_table.loc[2, :] = [0.051394, -0.195850, -0.088283, -0.797316, 0.143359, -0.409664]
            q_table.loc[3, :] = [0.580589, -0.889899, 0.875300, -0.411947, -0.843799, -0.16920]
            q_table.loc[4, :] = [0.243537, 0.232681, 0.225340, -0.241132, 0.243558, 0.218245]
            q_table.loc[5, :] = [0.261853, -0.004128, 0.245400, 0.253192, 0.325730, 0.280346]
            q_table.loc[6, :] = [-0.907549, -0.304897, 0.121416, 0.001726, 0.190901, 0.585869]
            q_table.loc[7, :] = [0.065815, 0.061019, 0.058222, -0.143220, 0.076692, 0.063469]
            q_table.loc[8, :] = [0.205312, 0.201767, 0.206270, 0.204152, 0.212234, 0.204021]
            q_table.loc[9, :] = [0.158233, 0.124685, -0.583542, 0.386826, 0.833968, -0.634750]
            q_table.loc[10, :] = [-0.083092, -0.260037, 0.030933, 0.873379, -0.785274, 0.541584]
            q_table.loc[11, :] = [0.756085, 0.307227, -0.222658, -0.242105, 0.338671, 0.775300]
        elif tiankou_index == 6:
            q_table.loc[0, :] = [-0.316831, -0.089409, 0.475088, 0.814135, -0.338100, 0.387182]
            q_table.loc[1, :] = [0.419376, 0.110977, 0.391983, -0.385638, 0.719306, 0.073414]
            q_table.loc[2, :] = [-0.021651, -0.150876, 0.042247, -0.184714, 0.112860, -0.186892]
            q_table.loc[3, :] = [0.588560, 0.458287, -0.460333, 0.335189, 0.546789, 0.214826]
            q_table.loc[4, :] = [0.750542, 0.561953, 0.684096, 0.633767, 0.625977, 0.624605]
            q_table.loc[5, :] = [0.520929, 0.490062, 0.370871, 0.310072, 0.321438, 0.4620485]
            q_table.loc[6, :] = [0.715471, 0.450896, 0.640099, -0.674601, 0.102116, 0.781731]
            q_table.loc[7, :] = [0.371757, 0.386189, 0.326671, 0.391035, 0.420085, 0.329884]
            q_table.loc[8, :] = [0.348764, 0.333418, 0.300349, 0.333606, 0.383981, 0.314639]
            q_table.loc[9, :] = [0.262151, 0.899805, 0.217892, -0.697352, 0.537068, 0.273302]
            q_table.loc[10, :] = [0.187786, 0.322218, 0.442598, 0.217145, 0.531765, -0.449027]
            q_table.loc[11, :] = [0.663377, 0.962284, -0.582866, -0.358000, -0.175679, -0.318108]
        elif tiankou_index == 7:
            q_table.loc[0, :] = [-0.537794, -0.108564, 0.157248, 0.683655, 0.690799, 0.882108]
            q_table.loc[1, :] = [0.852745, 0.603412, 0.575980, -0.591932, 0.721203, 0.231196]
            q_table.loc[2, :] = [0.235509, -0.092506, 0.079785, -0.299042, 0.262779, -0.811113]
            q_table.loc[3, :] = [-0.545851, 0.315133, -0.903139, -0.151533, 0.208022, -0.298129]
            q_table.loc[4, :] = [1.084104, -0.047638, 1.069573, 1.068253, 1.094649, 1.065648]
            q_table.loc[5, :] = [-0.025726, 0.512810, 0.478482, 0.472018, 0.552276, 0.504623]
            q_table.loc[6, :] = [-0.526976, -0.472709, -0.380786, 0.573609, -0.747990, 0.540424]
            q_table.loc[7, :] = [-0.417660, 0.721945, 0.746669, 0.729204, 0.754077, 0.729135]
            q_table.loc[8, :] = [0.637051, 0.658198, 0.646790, 0.667197, 0.642190, 0.665636]
            q_table.loc[9, :] = [0.837573, 0.101113, -0.060353, -0.333417, -0.088503, 0.120987]
            q_table.loc[10, :] = [0.483464, 0.867902, 0.130103, -0.584855, -0.849504, 0.921563]
            q_table.loc[11, :] = [-0.109952, -0.260939, 0.215350, -0.731100, -0.248946, -0.662203]
        elif tiankou_index == 8:
            q_table.loc[0, :] = [-0.104442, 0.695218, -0.037584, 0.511392, -0.458377, -0.224905]
            q_table.loc[1, :] = [-0.008872, -0.333773, -0.204810, -0.537485, -0.151556, -0.534913]
            q_table.loc[2, :] = [-0.095015, -0.396089, -0.208480, -0.589965, 0.216262, -0.595305]
            q_table.loc[3, :] = [0.399035, 0.431396, 0.295135, 0.389296, -0.449267, -0.470957]
            q_table.loc[4, :] = [0.169241, 0.148808, 0.152820, 0.152799, -0.010524, 0.155598]
            q_table.loc[5, :] = [0.217709, 0.075098, 0.149944, -0.157190, 0.213449, 0.122432]
            q_table.loc[6, :] = [0.279991, -0.444078, 0.228934, 0.344015, 0.384843, 0.352090]
            q_table.loc[7, :] = [0.050632, 0.048696, 0.058343, 0.049385, -0.062897, 0.047171]
            q_table.loc[8, :] = [0.038988, 0.035075, 0.035936, 0.034636, -0.005874, 0.032390]
            q_table.loc[9, :] = [-0.308785, -0.106199, 0.845129, -0.795929, -0.171762, 0.304020]
            q_table.loc[10, :] = [0.715505, 0.797486, -0.825996, -0.700185, -0.150285, -0.540733]
            q_table.loc[11, :] = [0.738975, 0.633874, 0.011621, 0.591509, -0.964596, -0.671729]

        elif tiankou_index == 9:
            q_table.loc[0, :] = [-0.779127, 0.208019, -0.483981, 0.110223, 0.831561, -0.655010]
            q_table.loc[1, :] = [-0.055884, -0.421797, -0.628905, -0.639318, -0.252555, -0.630156]
            q_table.loc[2, :] = [-0.183467, -0.374010, -0.366441, -0.728201, -0.089234, -0.735401]
            q_table.loc[3, :] = [-0.059829, -0.628596, -0.017911, -0.233676, -0.816663, -0.144225]
            q_table.loc[4, :] = [0.149273, 0.159635, 0.141756, 0.168458, -0.021664, 0.138485]
            q_table.loc[5, :] = [0.020408, 0.074517, 0.066116, 0.098335, 0.118077, -0.243028]
            q_table.loc[6, :] = [0.189784, 0.203534, -0.618323, 0.194173, 0.194259, 0.206369]
            q_table.loc[7, :] = [0.033297, 0.027480, -0.145820, 0.026435, 0.028810, 0.023228]
            q_table.loc[8, :] = [0.009993, 0.010543, 0.010447, 0.009552, -0.064133, 0.010393]
            q_table.loc[9, :] = [-0.820178, 0.065312, -0.863097, 0.754807, 0.696587, -0.973677]
            q_table.loc[10, :] = [-0.605510, -0.036617, -0.756280, -0.281979, -0.888620, 0.759758]
            q_table.loc[11, :] = [-0.507974, -0.762860, -0.265094, 0.489176, 0.432315, 0.799927]

        item = tiankou_table[tiankou_index]
        jingying = item[0]
        lr = item[2]
        discount_rate = item[3]
        i_iter_file_name = 0
        # for index, file_name in enumerate(txt_files):
        # file_name = '1094_Instance_20_2_2_0,2_0,6_10_Rep4.txt'
        split_list = file_name.split('_')
        job_num = int(split_list[2])
        machine_num_per_stage = int(split_list[4])
        i_stop_iter = int(job_num * machine_num_per_stage * item[1])
        # if (int(split_list[0]) - 4) % 5 != 0:
        # if (int(split_list[0])-1) % 5 != 0:
        #     continue
        chosen_file_index = [72	,127,137,152,157,162,167,172,177,572,582,602,607,612,622,627,637,667,672,677,682,687,692,
                             697,702,707,712,717,1112,1127,1132,1142,1147,1152,1157,1162,1167,1202,1207,1212,1217,1222,
                             1227,1232,1237,1242,1247,1252,1257,73,  128, 138, 153, 158 , 163 , 168 ,  173 , 178 , 573  , 583 ,  603 ,608,
                             613,  623,    628,    638,    668 ,   673 ,   678 ,   683 ,   688 ,   693 ,   698 ,   703  ,  708,
                             713,   718,   1113,    1128,    1133 ,   1143  ,  1148  ,  1153  ,  1158   , 1163,    1168  ,  1203,
                             1208,   1213,   1218,    1223 ,   1228  ,  1233  ,  1238 ,   1243   , 1248   , 1253 ,   1258]
        if (int(split_list[0])-2) % 5 != 0 and (int(split_list[0])-3) % 5 != 0:
            continue

        if int(split_list[0]) not in chosen_file_index:
            continue
        # if int(split_list[0]) % 5 != 0:
        #     continue
        i_iter_file_name += 1
        time_cost = 0
        start_time = time.time()
        # 初始化
        a = 1
        i_trial = 10  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!修改
        best_opt_list = []
        cur_best_opt = np.inf
        cur_best_schedule = None
        cur_best_job_execute_time = None
        for i in range(20):
            hfs = HFS(file_name, int(job_num * 2 * jingying))
            hfs.initial_solu()
            inital_obj = hfs.inital_refset[0][1]
            # if i >= 1:
            #     hfs.inital_refset.append(cur_best_item)
            #     hfs.inital_refset = sorted(hfs.inital_refset, key=lambda x: x[1])
            #     hfs.inital_refset.remove(hfs.inital_refset[-1])
            rl_ = RL_Q(N_STATES, ACTIONS, hfs.inital_refset, q_table, file_name, i_iter_file_name, i_stop_iter, a,
                       i_trial, lr, jingying, discount_rate,
                       hfs.population_refset)  # ！！！！！！倒数第二个参数修改，对应的动作不应该再有随机了！！！！！！！！！
            best_opt_execute, CUM_REWARD_case, best_item = rl_.rl_execute()
            if best_item[1] < cur_best_opt:
                cur_best_item = best_item
                cur_best_schedule = cur_best_item[0]
                cur_best_job_execute_time = cur_best_item[2]
                cur_best_opt = cur_best_item[1]

                # 画图记录一下

            best_opt_list.append(cur_best_opt)
            with open('./tiankou0304_termination/real_date_MDP_real_para_0311_2233.txt', 'a+') as fp:
                print(best_opt_execute, file=fp)
                print(file_name, file=fp)
                if hfs.population_refset:
                    if best_item[1] < cur_best_opt:
                        cur_best_item = best_item
                        cur_best_opt = best_item[1]
                if i == 19:
                    print('', file=fp)
                    print('', file=fp)
                    best_opt_list.sort()
                    opt_list = best_opt_list[:10]
                    print('文件名：', file_name, file=fp)
                    print('目标值列表：', best_opt_list, file=fp)
                    print('计算目标值：', opt_list, file=fp)
                    print('田口序列：{}'.format(tiankou_index), file=fp)

                    split_list = file_name.split('_')
                    dia = job_diagram(cur_best_schedule, cur_best_job_execute_time, file_name, split_list[0])
                    dia.pre()
                    plt.savefig('./tiankou0304_termination/img0311_2233/pic-{}.png'.format(split_list[0]))
                print('', file=fp)
                print('', file=fp)

        # best_opt_execute, CUM_REWARD_case = rl_.rl_excuse_case(hfs.inital_refset, file_name, len(case_CUM_REWARD),
        #                                                        inital_obj)
        # best_opt_list.append(best_opt_execute)
        # with open('./MDP.txt', 'a+') as fp:
        #     print(best_opt_execute, file=fp)
        #     print(file_name, file=fp)
        #     print('', file=fp)
        #     print('', file=fp)

        # from SS_RL.diagram import job_diagram
        # import matplotlib.pyplot as plt
        #
        # schedule = rl_.env.inital_refset[0][0]
        # job_execute_time = rl_.env.inital_refset[0][2]
        # split_list = file_name.split('_')
        # dia = job_diagram(schedule, job_execute_time, file_name, split_list[0])
        # dia.pre()
        # plt.savefig('./img0121/pic-{}.png'.format(file_name))

        end_time = time.time()
        duration = end_time - start_time

        with open('./tiankou0304_termination/real_data_result_para_0311_2233.txt', 'a+') as fp:
            print(file_name, rl_.config.ture_opt, round(sum(opt_list) / len(opt_list), 2), min(opt_list),
                  round(duration / len(best_opt_list), 2), file=fp)

if __name__ == "__main__":
    if text is True:
        # 以下参数需要确认
        # try:
        # for index, file_name in enumerate(txt_files):
        #     mian_mult(file_name)
        import multiprocessing
        from multiprocessing import Pool
        import time

        with Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(mian_mult, txt_files)



    # except:
    #     pass


    # epoch = 1
    # diag = job_diagram(schedule,job_execute_time,file_name,epoch)
    # diag.pre()
    # print(1)
    #
    #
    # before_schedule = {(0, 0): [3, 8, 9, 1, 6],
    #  (0, 1): [0, 4, 2, 5, 7],
    #  (1, 0): [3, 8, 4, 2],
    #  (1, 1): [0, 9, 5, 7, 1, 6]}
    #
    # schedule = {(0, 0): [3, 8, 9, 1, 6],
    #  (0, 1): [0, 4, 2, 5, 7],
    #  (1, 0): [8, 3, 4, 2],
    #  (1, 1): [0, 9, 5, 7, 1, 6]}
    #
    # recal_schedule = ScheduleCal(schedule,file_name)
    # diag = job_diagram(schedule, job_execute_time, file_name, epoch)
    # diag.pre()

    # 413
    # before_schedule = {(0, 0): [3, 8, 9, 1, 6],
    #  (0, 1): [0, 4, 2, 5, 7],
    #  (1, 0): [3, 8, 4, 2],
    #  (1, 1): [0, 9, 5, 7, 1, 6]}
    # schedule = {(0, 0): [3, 8, 9, 1, 6],
    #  (0, 1): [0, 4, 2, 5, 7],
    #  (1, 0): [8, 3, 4, 2],
    #  (1, 1): [0, 9, 5, 7, 1, 6]}
    # neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, file_name)
    # neig_search.re_cal(schedule)    # 更新了工件的开始和完工时间，以及目标值
    # neig_search.update_schedule, neig_search.update_job_execute_time, neig_search.update_obj = neig_search.hfs.idle_time_insertion(
    #     schedule, neig_search.update_job_execute_time, neig_search.update_obj)
    # diag = job_diagram(schedule, neig_search.update_job_execute_time, file_name, 2)
    # diag.pre()

    # neig_search.update_schedule, neig_search.update_job_execute_time, neig_search.update_obj = neig_search.hfs.idle_time_insertion(
    #     schedule, neig_search.update_job_execute_time, neig_search.update_obj)
    # neig_search.hfs.cal(neig_search.update_job_execute_time)






# 3. 解的结合


# 4.


# # 主代码
# hfs = HFS(machine_num_on_stage,job_process_time,ect_weight,ddl_weight,ddl_windows,ect_windows,jobs_num,stages_num)
# hfs.initial_solu()
#
# # print(hfs.inital_refset)
# for key in hfs.inital_refset.keys():
#     for item in hfs.inital_refset[key]:
#         print(item[0],item[1])
#
# for key in hfs.inital_refset.keys():
#     new_list = hfs.inital_refset[key]
#     for index,item in enumerate(hfs.inital_refset[key]):
#         schedule = item[0]
#         job_execute_time = item[2]
#         obj = item[1]
#         action_space = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
#         for opea_name in action_space:
#             # 根据邻域搜索方式进行搜索
#             neig_search = neighbo_search.Neighbo_Search(opea_name, schedule, job_execute_time, obj,
#                                                         machine_num_on_stage, job_process_time, ect_weight, ddl_weight,
#                                                         ddl_windows, ect_windows, jobs_num, stages_num)
#             update_schedule,update_obj,update_job_execute_time = neig_search.search_opea()
#             if update_obj < obj:
#                 print(0, 'self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))
#                 improve_flag = 0
#                 hfs.inital_refset[key][index] = (update_schedule, update_obj, update_job_execute_time)
#             else:
#                 print(1, 'self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))
#
#             schedule = hfs.inital_refset[key][index][0]
#             job_execute_time = hfs.inital_refset[key][index][2]
#             obj = hfs.inital_refset[key][index][1]
#             print('success')
