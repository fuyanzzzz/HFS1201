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
from SS_RL.diagram import job_diagram

path = r'C:\paper_code_0501\HFS1201\useful0424\data'
from config import DataInfo
from SS_RL.RL_ import RL_Q
from SS_RL.inital_solution import HFS
# from SS_RL.schedule_cal import ScheduleCal
import matplotlib.pyplot as plt


N_STATES = 9
ACTIONS = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
# 1. 生成初始解，这个没有问题
actions = range(7)
max_iter = 3
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.reset_option('display.max_rows')
pd.reset_option('display.max_columns')

# q_table = pd.DataFrame(np.random.rand(12, 11),columns=actions)
q_table = pd.DataFrame(
            np.zeros((N_STATES, len(actions))),columns=actions)

data_folder = "data"  # 数据文件夹的路径
# 获取data文件夹下所有的文件名

# 新建一个变量，用于存储每个动作是否有改进，以及改进多少
use_action_dict = {}
action_space = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
for i_action in action_space:
    use_action_dict[i_action] = [0,0]
q_value_changes = []
CUM_REWARD = []
case_CUM_REWARD = []
case_CUM_obj = []
INDEX = []


txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
iter = 0
while True:
    if iter >= max_iter:
        break
    for index, file_name in enumerate(txt_files):
        time_cost = 0
        start_time = time.time()
        # file_name = '1258_Instance_20_2_3_0,6_1_20_Rep3.txt'
        # if index == 0:
        #     continue
        print('换数据集啦{0}'.format(file_name))
        print('第{0}幕'.format(index))
        hfs = HFS(file_name)

        hfs.initial_solu()

        rl_ = RL_Q(N_STATES,ACTIONS,hfs.inital_refset,q_table,file_name,len(txt_files)*iter +index)
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

        INDEX.append(len(txt_files)*iter +index)

        fp = open('./time_cost.txt', 'a+')
        if index == 0:
            print('索引   文件名   耗时  数据最优解   实验最优解 ', file=fp)
        print('{0}   {1}   {2}   {3}   {4} '.format(len(txt_files)*iter +index,file_name,round(duration,2),rl_.config.ture_opt,rl_.best_opt), file=fp)
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
        #
        # dia = job_diagram(schedule, job_execute_time, file_name, len(txt_files)*iter +index)
        # dia.pre()
        # plt.savefig('./img1203/pic-{}.png'.format(len(txt_files)*iter +index))
        # plt.savefig('./img1203/pic-{}.png'.format(len(txt_files)*iter +index))

        # 每过一幕验证一下奖励
        time_cost = 0
        start_time = time.time()
        case_file_name = '1258_Instance_20_2_3_0,6_1_20_Rep3.txt'
        # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
        hfs = HFS(case_file_name)

        hfs.initial_solu()

        print('当前目标值：{0}'.format(hfs.inital_refset[0][1]))

        # dia = job_diagram(hfs.inital_refset[0][0], hfs.inital_refset[0][2], case_file_name, index)
        # dia.pre()
        # plt.savefig('./img1203/pic-{}.png'.format(index))
        # plt.show()

        rl_ = RL_Q(N_STATES,ACTIONS,hfs.inital_refset,q_table,case_file_name,len(txt_files)*iter +index)
        best_opt_execute ,CUM_REWARD_case= rl_.rl_execute()
        with open('./MDP.txt', 'a+') as fp:
            print('目标值:{0},   奖励:{1},'.format(best_opt_execute, CUM_REWARD_case), file=fp)
            print('', file=fp)
            print('', file=fp)

        case_CUM_obj.append(best_opt_execute)
        case_CUM_REWARD.append(CUM_REWARD_case)
        end_time = time.time()
        duration = end_time - start_time

        with open('./time_cost.txt', 'a+') as fp:
            # 设置显示选项
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)

            # 将 DataFrame 写入文件
            # print(index, q_table, file=fp)

            if index == 0:
                print('索引   文件名   耗时  数据最优解   实验最优解 ', file=fp)
            print()
            print('{0}   {1}   {2}   {3}   {4} '.format(len(txt_files) * iter + index, case_file_name, round(duration, 2),
                                                        rl_.config.ture_opt, rl_.best_opt), file=fp)
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

        if (len(txt_files)*iter +index) % 20 == 0:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True)

            ax1.plot(INDEX, q_value_changes, label='子图1', color='blue')
            ax1.set_ylabel('Q值变化程度')
            ax1.legend()

            ax2.plot(INDEX, CUM_REWARD, label='子图2', color='red')
            ax2.set_ylabel('累计奖励')
            ax2.legend()

            ax3.plot(INDEX, case_CUM_REWARD, label='子图3', color='green')
            ax3.set_ylabel('实验案例')
            ax3.legend()

            ax4.plot(INDEX, case_CUM_obj, label='子图4', color='yellow')
            ax4.set_ylabel('实验案例目标值')
            ax4.legend()

            # 调整子图之间的垂直间距
            plt.tight_layout()
            # plt.pause(0.1)  # 用于动态展示图像
            plt.savefig('./img0.02_0.9_0112/pic-{}.png'.format(int(len(txt_files)*iter +index)))

            with open('./0.02_0.9_0112.txt', 'a+') as fp:
                # 设置显示选项
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)

                # 将 DataFrame 写入文件
                print(index, q_table, file=fp)
                # for item in rl_.use_actions.keys():
                #     print(item,rl_.use_actions[item], file=fp)

                # 重置显示选项为默认值
                pd.reset_option('display.max_rows')
                pd.reset_option('display.max_columns')


            # fp = open('./0.05_0.9_1207_2.txt', 'a+')
            # print(len(txt_files)*iter +index,rl_.q_table, file=fp)
            # fp.close()
    iter += 1



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
