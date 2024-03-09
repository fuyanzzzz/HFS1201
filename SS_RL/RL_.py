'''
-------------------------------------------------
File Name: RL_.py
Author: LRS
Create Time: 2023/6/6 11:43
-------------------------------------------------
'''
import copy
import math
import time
import random

import numpy as np
import pandas as pd

from SS_RL import neighbo_search
from SS_RL import inital_solution
# from SS_RL import diagram
from SS_RL.diagram import job_diagram
from SS_RL.public import AllConfig
from config import DataInfo
import matplotlib.pyplot as plt
from collections import deque
from itertools import product
from Schedule import Schedule_Instance
'''
如果此次最优解没有更新，trial += 1
平均目标值进行更新

状态划分：
1. trail = 0， 平均目标值>0  :解还在优化，且最优解更新了，  好
2. trail 【0，20】，平均目标值>0,  解在优化，但是最优解不更新，Hao
3.
'''

N_STATES = 8
ACTIONS =['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
action_set = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
EPSILON = 0.9
# ALPHA = 0.1
# ALPHA = 0.01
# GAMMA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.3
TerminalFlag = "terminal"




class RL_Q():
    def __init__(self,n_states,n_actions,inital_refset,q_table,file_name,iter,stop_iter,a,i_trial,lr,jingying,discount_rate):

        self.trial = 0
        self.ori_mean_obj = 9999999

        self.best_schedule = None
        self.inital_refset = inital_refset
        self.config = AllConfig.get_config(file_name)
        self.population = self.config.jobs_num *2
        self.jingying_num = int(self.population * jingying)
        self.ini = inital_solution.HFS(file_name,self.jingying_num)
        self.inital_refset = sorted(self.inital_refset, key=lambda x: x[1])
        self.inital_obj = self.inital_refset[0][1]
        self.best_opt = self.inital_obj
        self.file_name = file_name
        self.max_iter = 0


        self.gen_action_space()
        # self.q_table = self.build_q_table(n_states,range(7))
        self.q_table = q_table
        # 以下的只是为了统计
        self.use_actions = {}   # 要放在主函数外，不然会重置掉
        # self.action_space_1 = ['effeinsert0', 'effeinsert1', 'randinsert0', 'randinsert1', 'effeswap0', 'effeswap1',
        #                 'randswap0', 'randswap1']
        self.iter_index = iter
        self.not_opt_iter = 0
        self.stop_iter = stop_iter

        self.from_now = 0
        self.delay_early = []
        self.update_jingying = 0
        self.a = a
        self.i_trial = i_trial
        self.GAMMA = discount_rate
        self.jingying_num = int(self.population* jingying)
        self.ALPHA = lr

        '''
        动作空间：
        1. 对于第二阶段卡住的工件，将该工件在第一阶段往前insert/swap,同一/不同机器
        2. 选择第二阶段往前/往后增加的目标值最少的工件块，选择权重最大/第一个/加工时间最短的工件进行insert/swap,对于同一机器/不同机器
        3. 随机选择一个工件，根据属性，进行insert/swap
        4. 找到延误工件/早到工件块，按照一定规则【加工时间规则/ddl规则/ect规则/权重规则】进行排序
        '''

        self.action_space_1 = ['effe_insert_same_DM_1', 'effe_swap_same_DM_1', 'effe_insert_other_DM_1', 'effe_swap_other_DM_1',
        'effe_insert_same_EM_1', 'effe_swap_same_EM_1', 'effe_insert_other_EM_1', 'effe_swap_other_EM_1',
        'rand_insert_same_M_1', 'rand_swap_same_M_1', 'rand_insert_other_M_1', 'rand_swap_other_M_1',
        'effe_insert_same_M_0', 'effe_swap_same_M_0', 'effe_insert_other_M_0', 'effe_swap_other_M_0',
        'dire_insert_same_dweight_1', 'dire_insert_other_dweight_1',
        'dire_insert_same_eweight_1', 'dire_insert_other_eweight_1',
        'effe_insert_same_DRM_1', 'effe_swap_same_DRM_1', 'effe_insert_other_DRM_1', 'effe_swap_other_DRM_1',
        'effe_insert_same_ERM_1', 'effe_swap_same_ERM_1', 'effe_insert_other_ERM_1', 'effe_swap_other_ERM_1',
                               'effe_insert_same_stuck_0','effe_swap_same_stuck_0','effe_insert_other_stuck_0','effe_swap_other_stuck_0']

        for i_action in self.action_space_1:
            self.use_actions[i_action] =[0, 0, 0]

        self.oper_same_machine = deque(maxlen=3)
        a_values = range(0, 2)
        b_values = range(0, 3)
        c_values = range(0, 2)
        combinations = list(product(a_values, b_values, c_values))


        # 创建字典
        # self.state_space = dict(zip(combinations, list(range(len(combinations)))))
        # print(1)




    def build_q_table(self,n_states, actions):
        return pd.DataFrame(
            np.zeros((n_states, len(actions))),columns=actions)

    def gen_action_space(self):
        self.action_space = {}
        # self.action_space[0] = ['effeinsert0','effeinsert1','effeswap0','effeswap1']
        # self.action_space[1] = ['randinsert0','randinsert1','randswap0','randswap1']
        # self.action_space[2] = ['effeinsert0','randinsert0','effeswap0','randswap0']
        # self.action_space[3] = ['effeinsert1','randinsert1','effeswap1','randswap1']
        # self.action_space[4] = ['effeinsert0','effeinsert1','randinsert0','randinsert1']
        # self.action_space[5] = ['effeswap0','effeswap1','randswap0','randswap1']
        # self.action_space[6] = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
        '''
        建立新的动作搜索空间

        '''
        # 在第二阶段选择目标值最大的工件之一，进行insert/swap操作effe_insert_same_AF_1
        # self.action_space[0] = ['effe_insert_same_EM_1', 'effe_swap_same_EM_1']     # 同一个机器
        # self.action_space[1] = ['effe_insert_other_EM_1', 'effe_swap_other_EM_1']     # 不同机器
        # 在第一阶段选择松弛最小的工件之一，进行insert/swap操作
        # self.action_space[2] = ['effe_insert_same_M_0','effe_swap_same_M_0']      # 同一个机器
        # self.action_space[3] = ['effe_insert_other_M_0','effe_swap_other_M_0']      # 同一个机器

        # # 在第二阶段选择目标值最大的工件之一，进行insert/swap操作effe_insert_same_AF_1
        # self.action_space[0] = ['effe_insert_same_DM_1', 'effe_swap_same_DM_1']     # 同一个机器
        # self.action_space[1] = ['effe_insert_other_DM_1', 'effe_swap_other_DM_1']     # 同一个机器
        #
        # self.action_space[2] = ['effe_insert_same_EM_1', 'effe_swap_same_EM_1']     # 同一个机器
        # self.action_space[3] = ['effe_insert_other_EM_1', 'effe_swap_other_EM_1']     # 同一个机器
        #
        # # 在第二阶段随机选择工件进行insert/swap:
        # self.action_space[4] = ['rand_insert_other_M_1','rand_swap_other_M_1']      # 同一个机器
        # self.action_space[5] = ['rand_insert_same_M_1','rand_swap_same_M_1',]      # 同一个机器
        # # self.action_space[4] = ['effe_insert_same_M_0','effe_swap_same_M_0','effe_insert_other_M_0','effe_swap_other_M_0']      # 同一个机器
        #
        #
        # # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        # # self.action_space[10] = ['sort_stuck_A_S0_1']
        # self.action_space[6] = ['dire_insert_same_dweight_1',]
        # self.action_space[7] = ['dire_insert_other_dweight_1']
        #
        # self.action_space[8] = ['dire_insert_same_eweight_1',]
        # self.action_space[9] = ['dire_insert_other_eweight_1']
        #
        # self.action_space[10] = ['effe_insert_same_DRM_1','effe_swap_same_DRM_1']      # 同一个机器
        # self.action_space[11] = ['effe_insert_other_DRM_1','effe_swap_other_DRM_1']      # 同一个机器
        #
        #
        # self.action_space[12] = ['effe_insert_same_ERM_1','effe_swap_same_ERM_1']      # 同一个机器
        # self.action_space[13] = ['effe_insert_other_ERM_1','effe_swap_other_ERM_1']      # 同一个机器




        # # 第0阶段，卡住的工件在同一机器/其他机器，进行insert/swap
        # self.action_space[0] = ['effe_insert_same_stuck_0','effe_swap_same_stuck_0']
        # self.action_space[1] = ['effe_insert_other_stuck_0','effe_swap_other_stuck_0']
        #
        # # 第1阶段，单位增加目标值最多的方向的第一个工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[2] = ['sort_delay_A_P_1']
        # self.action_space[3] = ['sort_delay_A_D_1']
        #
        # # 第1阶段，单位可改善最多的方向的权重贡献最大的工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[4] = ['sort_early_D_P_1']
        # self.action_space[5] = ['sort_early_A_D_1']
        #
        # # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[6] = ['sort_stuck_A_S0_1']
        #
        # # 第1阶段，单位可改善最多的方向的第一个工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[8] = ['effe_insert_same_AF_1', 'effe_swap_same_AF_1']
        # self.action_space[9] = ['effe_insert_other_AF_1', 'effe_swap_other_AF_1']
        #
        # # 第1阶段，单位可改善最多的方向的权重贡献最大的工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[10] = ['effe_insert_same_AW_1','effe_swap_same_AW_1']
        # self.action_space[11] = ['effe_insert_other_AW_1','effe_swap_other_AW_1']
        #
        # # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[12] = ['effe_insert_same_AP_1', 'effe_swap_same_AP_1']
        # self.action_space[13] = ['effe_insert_other_AP_1', 'effe_swap_other_AP_1']
        #
        # # 第一阶段，随机
        # self.action_space[14] = ['rand_insert_same_R_1','rand_swap_same_R_1']
        # self.action_space[7] = ['rand_insert_other_R_1', 'rand_swap_other_R_1']

        # # 以延误为单位的insert，
        # self.action_space[0] = ['dire_insert_same_dweight_1', 'dire_insert_other_dweight_1', 'effe_insert_same_DRM_1',
        #                         'effe_insert_other_DRM_1']
        #
        # # 以延误为单位的swap
        # self.action_space[1] = ['effe_swap_same_DRM_1', 'effe_swap_other_DRM_1']
        # # 以早到为单位的insert
        # self.action_space[2] = ['dire_insert_same_eweight_1', 'dire_insert_other_eweight_1', 'effe_insert_same_ERM_1',
        #                         'effe_insert_other_ERM_1']
        # # 以早到为单位的sawp
        # self.action_space[3] = ['effe_swap_same_ERM_1', 'effe_swap_other_ERM_1']
        # # 以卡住为单位，移动第一阶段
        #
        # # 完全随机工件 insert
        # self.action_space[4] = ['rand_insert_other_M_1', 'rand_insert_same_M_1']
        # # 完全随机工件 swap
        # self.action_space[5] = ['rand_swap_other_M_1', 'rand_swap_same_M_1']
        #
        # self.action_space[0] = ['effe_insert_same_stuck_0','effe_swap_same_stuck_0']
        # self.action_space[1] = ['effe_insert_other_stuck_0','effef_swap_other_stuck_0']

        # # 以延误为单位的insert，
        self.action_space[0] = [ 'effe_insert_same_DRM_1','effe_insert_other_DRM_1','effe_swap_same_DRM_1', 'effe_swap_other_DRM_1']

        # 以延误为单位的swap
        self.action_space[1] = ['dire_insert_same_dweight_1', 'dire_insert_other_dweight_1',]
        # 以早到为单位的insert
        self.action_space[2] = [ 'effe_insert_same_ERM_1','effe_insert_other_ERM_1','effe_swap_same_ERM_1', 'effe_swap_other_ERM_1']
        # 以早到为单位的sawp
        self.action_space[3] = ['dire_insert_same_eweight_1', 'dire_insert_other_eweight_1',]
        # 以卡住为单位，移动第一阶段

        # 完全随机工件 insert
        self.action_space[4] = ['rand_insert_other_M_1', 'rand_insert_same_M_1','rand_swap_other_M_1', 'rand_swap_same_M_1']
        # 完全随机工件 swap
        self.action_space[5] = ['effe_insert_same_stuck_0','effe_swap_same_stuck_0','effe_insert_other_stuck_0','effe_swap_other_stuck_0']




    def choose_action(self,state, q_table):
        state_table = q_table.loc[state, :]
        # if self.file_name[0] == '0':
        #     action_name = np.random.choice(range(len(self.action_space)))
        # else:
        #     if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        #         action_name = np.random.choice(range(len(self.action_space)))
        #     else:
        #         action_name = state_table.idxmax()

        # if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        #     action_name = np.random.choice(range(len(self.action_space)))
        # else:
        #     action_name = state_table.idxmax()

        # if ((state_table == 0).all()) or self.iter_index < 20:
        #     action_name = np.random.choice(range(len(self.action_space)))
        # else:
        #     new_list = copy.deepcopy(state_table)
        #     min_value = min(new_list)
        #     if min_value <0:
        #         for i in range(len(new_list)):
        #             new_list[i] += -min_value
        #
        #
        #     total_weight = sum(new_list)
        #     cumulative_weights = [sum(new_list[:i + 1]) for i in range(len(new_list))]  # 计算累积权重
        #     rand_val = random.uniform(0, total_weight)  # 生成一个随机值
        #
        #     # 根据随机值选择元素
        #     for item, cumulative_weight in zip(range(len(new_list)), cumulative_weights):
        #         if rand_val <= cumulative_weight:
        #             action_name = item
        #             break
        action_name = state_table.idxmax()
        # action_name = np.random.choice(range(len(self.action_space)))
        return action_name


    def get_state(self,old_inital_refset):
        # 根据传入的参数确定转移状态
        # state_space = {}

        ect_value = 0
        ddl_value = 0
        for opt_item in self.inital_refset[:self.jingying_num]:
        # opt_item = self.inital_refset[0]
            job_execute_time = opt_item[2]


            for job in range(self.config.jobs_num):
                job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
                if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                    ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
                elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                    ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]



        '''
        第二种状态设置方式：
        1. 机器数 > 分段数
        2. 几代不优
        3. 修复的是否同一个机器
        '''
        # self.schedule_ins.idle_time_insertion(self.inital_refset[0][0], self.inital_refset[0][2], self.inital_refset[0][1])
        # job_block = []
        # for i in self.schedule_ins.schedule_job_block.keys():
        #     for j in self.schedule_ins.schedule_job_block[i]:
        #         job_block.append(j)

        # from SS_RL.diagram import job_diagram
        # import matplotlib.pyplot as plt
        # dia = job_diagram(self.ini.schedule, self.ini.job_execute_time, self.file_name, 1)
        # dia.pre()
        # plt.savefig('./img1203/pic-{}.png'.format(int(1)))
        # plt.show()

        #
        # # 判断第一阶段和第二阶段的松弛度：
        # # 若第二阶段的延误工件1/3位置的工件的松弛度 < 第二阶段的平均加工时间，则表明第二阶段松弛度不够
        # delay_job = []
        # early_job = []
        # job_execute_time = self.inital_refset[0][2]
        # aver_job_process_1 = sum([self.config.job_process_time[1][job] for job in range(self.config.jobs_num)]) / self.config.jobs_num
        # for job in range(self.config.jobs_num):
        #     if job_execute_time[(1,job)] > self.config.ddl_windows[job]:
        #         slackness = job_execute_time[(1,job)] - self.config.job_process_time[1][job] - job_execute_time[(0,job)]
        #         delay_job.append((job,slackness))
        #     if job_execute_time[(1,job)] < self.config.ect_windows[job]:
        #         early_job.append(job)
        # # 默认优先第二阶段
        # priority_2 = 1
        # # 列表不为空
        # if delay_job:
        #     delay_job = sorted(delay_job, key=lambda x: x[-1], reverse=True)  # 降序
        #     # 获得1/3位置的工件的松弛度：
        #     if round(len(delay_job)//3) < aver_job_process_1:
        #         priority_2 = 0
        #     else:
        #         priority_2 = 1
        # # 若早到工件个数 > 延误工件个数
        # priority_early = 0
        # if len(delay_job) > len(early_job):
        #     priority_early = 1
        # else:
        #     priority_early = 0
        # # 是否超过10次
        # rand = 0
        # if self.trial == 0:
        #     rand = 0
        # elif self.trial < 10:
        #     rand = 1
        # else:
        #     rand = 2
        # if self.trial <= 10:
        #     rand = 1
        # else:
        #     rand = 0






        # if self.config.machine_num_on_stage[0] < len(job_block):        # 若机器数 < 分段数：说明第一阶段被卡了
        #     a = 0
        # else:
        #     a = 1
        #
        # if self.trial == 0:
        #     b = 0
        # elif self.trial <= 5:
        #     b = 1
        # else:
        #     b = 2
        #
        # # 修复的是否同一个机器，
        # if len(self.oper_same_machine) == 3 and len(set(self.oper_same_machine)) == 1:
        #     c = 0
        # else:
        #     c = 1

        '''
        判断延误目标值和早到目标值
        '''
        # ect_value = 0
        # ddl_value = 0
        # opt_item = self.inital_refset[0]
        # job_execute_time = opt_item[2]
        #
        # for job in range(self.config.jobs_num):
        #     job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
        #     if job_makespan < self.config.ect_windows[job]:  # 早前权重值
        #         ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
        #     elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
        #         ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
        #
        # self.state_space = {}
        # # ect_or_ddl = None
        #
        # new_delay_early = []
        # for i in range(int(self.population / 5)):
        #     for i_item in self.inital_refset:
        #         ect_value = 0
        #         ddl_value = 0
        #         opt_item = i_item
        #         job_execute_time = opt_item[2]
        #
        #         for job in range(self.config.jobs_num):
        #             job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
        #             if job_makespan < self.config.ect_windows[job]:  # 早前权重值
        #                 ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
        #             elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
        #                 ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
        #
        #         new_delay_early.append((ddl_value,ect_value))
        #
        # old_delay_early = []
        # for i in range(int(self.population / 5)):
        #     for i_item in old_inital_refset:
        #         ect_value = 0
        #         ddl_value = 0
        #         opt_item = i_item
        #         job_execute_time = opt_item[2]
        #
        #         for job in range(self.config.jobs_num):
        #             job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
        #             if job_makespan < self.config.ect_windows[job]:  # 早前权重值
        #                 ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
        #             elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
        #                 ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
        #
        #         old_delay_early.append((ddl_value,ect_value))
        #
        # sum_delay_change = 0
        # sum_early_change = 0
        # for i in range(len(old_delay_early)):
        #     sum_delay_change += old_delay_early[i][0] - new_delay_early[i][0]
        #     sum_early_change += old_delay_early[i][1] - new_delay_early[i][1]

        # average_delay_change = sum_delay_change / len(old_delay_early)
        # average_early_change = sum_early_change / len(old_delay_early)


        if self.trial == 0:
            if ect_value > 3* ddl_value:
                next_state = 0
            elif ddl_value > 3* ect_value:
                next_state = 1
            else:
                next_state = 2

        elif self.trial <= self.i_trial:
            if ect_value > 3* ddl_value:
                next_state = 3
            elif ddl_value > 3* ect_value:
                next_state = 4
            else:
                next_state = 5


        else:
            if ect_value > 3* ddl_value:
                next_state = 6
            elif ddl_value > 3* ect_value:
                next_state = 7
            else:
                next_state = 8

        # # 0229隐
        # sum_delay_change = 0
        # sum_early_change = 0
        # for i in range(len(self.delay_early)):
        #     sum_delay_change += self.delay_early[i][0]
        #     sum_early_change += self.delay_early[i][1]
        # average_delay_change = sum_delay_change
        # average_early_change = sum_early_change
        #
        # if self.trial == 0:
        #     if average_delay_change > 0 and average_early_change > 0:
        #         next_state = 0
        #     elif average_delay_change > 0 and average_early_change <= 0:
        #         next_state = 1
        #     elif average_delay_change <= 0 and average_early_change > 0:
        #         next_state = 2
        #     else:
        #         next_state = 3
        #
        # elif self.trial <= self.i_trial:
        #     if average_delay_change > 0 and average_early_change > 0:
        #         next_state = 4
        #     elif average_delay_change > 0 and average_early_change <= 0:
        #         next_state = 5
        #     elif average_delay_change <= 0 and average_early_change > 0:
        #         next_state = 6
        #     else:
        #         next_state = 7
        #
        #
        # else:
        #     if average_delay_change > 0 and average_early_change > 0:
        #         next_state = 8
        #     elif average_delay_change > 0 and average_early_change <= 0:
        #         next_state = 9
        #     elif average_delay_change <= 0 and average_early_change > 0:
        #         next_state = 10
        #     else:
        #         next_state = 11


        # self.state_space[next_state] = {self.trial, ect_or_ddl}

        return next_state

    def get_reward(self,cur_best_opt,state,next_state):
        # reward = 0
        # if (self.best_opt - cur_best_opt) > 0:
        #     reward = (self.best_opt - cur_best_opt) / self.inital_obj
        #     if reward > 0.2:
        #         reward = 0.2
        # else:
        #     reward = -0.001

        # fp = open('./time_cost.txt', 'a+')
        # print('{0}   {1}   {2}    {3}     {4}'.format(self.file_name,self.best_opt,cur_best_opt,self.inital_obj,self.config.ture_opt), file=fp)
        # fp.close()


        # if self.best_opt == 0:
        #     impor = (self.best_opt - cur_best_opt) / 1
        # else:
        #     impor = (self.best_opt - cur_best_opt) / self.best_opt
        #
        # if self.trial == 0:
        #     reward = math.exp((impor + impro_degree + diversity_degree)/3)
        # else:
        #     reward = -math.exp((impro_degree + diversity_degree)/2)
        # if np.isnan(reward):
        #     print(True)
        # if self.trial > 50:
        #     reward = -math.exp(1)
        # if self.inital_obj == 0:
        #     reward = 0
        # elif self.best_opt >= cur_best_opt:
        #     reward = (self.best_opt - cur_best_opt) / (self.inital_obj - self.config.ture_opt)
        # else:
        #     reward = 0
        # if reward < 0:
        #     reward = (self.best_opt - cur_best_opt) / self.config.ture_opt
        #     fp = open('./time_cost.txt', 'a+')
        #     print('{0}   {1}   {2}    {3}     {4}'.format(self.file_name,self.best_opt,cur_best_opt,self.inital_obj,self.config.ture_opt), file=fp)
        #     fp.close()

            # 这是一个注释
            # diag = job_diagram(self.inital_refset[0][0], self.inital_refset[0][2], self.file_name, 2)
            # diag.pre()
            # print(self.file_name,self.best_opt,cur_best_opt,self.inital_obj,self.config.ture_opt)

        # 改进了几个解，就更新几个


        # # 获取当下的平均目标值
        # update_num = 0
        # for i in range(int(len(self.inital_refset) / 2)):
        #     if old_inital_refset[i][0] != self.inital_refset[i][0] and self.inital_refset[i][1] < \
        #             old_inital_refset[i][1]:
        #         update_num += 1
        #
        # reward = update_num
        if self.a == 0:
            if state in [0,1,2,3] and next_state in [4,5,6,7]:
                reward = -(self.jingying_num - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [8,9,10,11]:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [8,9,10,11] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2,3] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 11 and next_state in [8,9,10]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 3 and next_state in [0,1,2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 7 and next_state in [4,5,6]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2] and next_state == 3:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6] and next_state == 7:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [8,9,10] and next_state == 11:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            else:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
        elif self.a == 1:
            if state in [0, 1, 2, 3] and next_state in [4, 5, 6, 7]:
                reward = -(self.jingying_num - self.update_jingying) / self.jingying_num
            elif state in [4, 5, 6, 7] and next_state in [8, 9, 10, 11]:
                reward = -(self.jingying_num - self.update_jingying)/ self.jingying_num
            elif state in [4, 5, 6, 7] and next_state in [0, 1, 2, 3]:
                reward = self.update_jingying/ self.jingying_num
            elif state in [8, 9, 10, 11] and next_state in [0, 1, 2, 3]:
                reward = self.update_jingying/ self.jingying_num
            elif state in [0, 1, 2, 3] and next_state in [0, 1, 2, 3]:
                reward = self.update_jingying/ self.jingying_num
            else:
                reward = 0
        elif self.a == 2:
            if state in [0,1,2,3] and next_state in [4,5,6,7]:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [8,9,10,11]:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [8,9,10,11] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2,3] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 11 and next_state in [8,9,10]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 3 and next_state in [0,1,2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 7 and next_state in [4,5,6]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2] and next_state == 3:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6] and next_state == 7:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [8,9,10] and next_state == 11:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            else:
                reward = 0
        elif self.a == 4:
            if state in [0,1,2,3] and next_state in [4,5,6,7]:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [8,9,10,11]:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [8,9,10,11] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2,3] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 11 and next_state in [8,9,10]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 3 and next_state in [0,1,2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 7 and next_state in [4,5,6]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2] and next_state == 3:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [4,5,6] and next_state == 7:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            elif state in [8,9,10] and next_state == 11:
                reward = -(int(self.population/5) - self.update_jingying)/ int(self.population / 5)
            else:
                reward = 0
        elif self.a == 3:
            if state in [0,1,2,3] and next_state in [4,5,6,7]:
                reward = -self.update_jingying/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [8,9,10,11]:
                reward = -self.update_jingying/ int(self.population / 5)
            elif state in [4,5,6,7] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [8,9,10,11] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2,3] and next_state in [0,1,2,3]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 11 and next_state in [8,9,10]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 3 and next_state in [0,1,2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state == 7 and next_state in [4,5,6]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0,1,2] and next_state == 3:
                reward = -self.update_jingying/ int(self.population / 5)
            elif state in [4,5,6] and next_state == 7:
                reward = -self.update_jingying/ int(self.population / 5)
            elif state in [8,9,10] and next_state == 11:
                reward = -self.update_jingying/ int(self.population / 5)
            else:
                reward = -self.update_jingying/ int(self.population / 5)
        elif self.a == 5:
            if self.trial == 0:
                reward = math.exp(self.update_jingying/ int(self.population / 5))
            else:
                reward = -math.exp((int(self.population/5) - self.update_jingying)/ int(self.population / 5))
        elif self.a == 6:
            if state in [0, 1, 2, 3] and next_state in [4, 5, 6, 7]:
                reward = -math.exp((int(self.population/5) - self.update_jingying)/ int(self.population / 5))
            elif state in [4, 5, 6, 7] and next_state in [8, 9, 10, 11]:
                reward = -math.exp((int(self.population/5) - self.update_jingying)/ int(self.population / 5))
            elif state in [4, 5, 6, 7] and next_state in [0, 1, 2, 3]:
                reward = math.exp(self.update_jingying/ int(self.population / 5))
            elif state in [8, 9, 10, 11] and next_state in [0, 1, 2, 3]:
                reward = math.exp(self.update_jingying/ int(self.population / 5))
            elif state in [0, 1, 2, 3] and next_state in [0, 1, 2, 3]:
                reward = math.exp(self.update_jingying/ int(self.population / 5))
            else:
                reward = 0
        elif self.a == 7:
            if state in [0, 1, 2] and next_state in [3, 4, 5]:
                reward = -(int(self.population / 5) - self.update_jingying) / int(self.population / 5)
            elif state in [3, 4, 5] and next_state in [6,7,8]:
                reward = -(int(self.population / 5) - self.update_jingying)/ int(self.population / 5)
            elif state in [6,7,8] and next_state in [0, 1, 2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [3, 4, 5]and next_state in [0, 1, 2]:
                reward = self.update_jingying/ int(self.population / 5)
            elif state in [0, 1, 2] and next_state in [0, 1, 2]:
                reward = self.update_jingying/ int(self.population / 5)
            else:
                reward = 0


        return reward

        # 有可能是一步调已经调不limit == 50，直接接动了，如果受下一次的解，重新去调整
        # 但这样有一个问题，就是分成两个子种群就已经没有意义了

    def excuse_action(self,action):
        new_list = []
        count = 0
        i_count = 10    # 注释
        self.upadate_num = 0
        self.delay_early = []
        self.update_jingying = 0

        for index, item in enumerate(self.inital_refset):
            i = 0

            update = False
            schedule = item[0]
            job_execute_time = item[2]
            obj = item[1]

            if index < self.jingying_num:
                ect_value_1 = 0
                ddl_value_1 = 0
                for job in range(self.config.jobs_num):
                    job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
                    if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                        ect_value_1 += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
                    elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                        ddl_value_1 += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
            else:
                i_action = np.random.choice(range(len(self.action_space)))
                action = i_action



            # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
            # dia = job_diagram(schedule, job_execute_time, self.file_name, 11)
            # dia.pre()
            # plt.savefig('./img1203/pic-{}.png'.format(11))


            while i < len(self.action_space[action]):
                exceuse_search = self.action_space[action][i]
                # schedule = copy.deepcopy(schedule)
                # job_execute_time = copy.deepcopy(job_execute_time)
                # obj = copy.deepcopy(obj)
                # print(0,schedule,obj)
                # self.diagram = diagram.job_diagram(schedule, job_execute_time, count)
                # self.diagram.pre()
                count += 1
                # 根据邻域搜索方式进行搜索
                # neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj,self.file_name)

                # 确定领域搜索的相关信息
                split_list = exceuse_search.split('_')
                # effe_insert_same_IW_1
                stage = int(split_list[-1])
                oper_method = split_list[1]
                search_method_1 = split_list[0]
                search_method_2 = split_list[3]
                config_same_machine = split_list[2]
                if config_same_machine == 'same':
                    config_same_machine = True
                if config_same_machine == 'other':
                    config_same_machine = False



                # stage = int(opea_name[-1])      # 确定操作的阶段
                # oper_method = opea_name[4]      # 确定是insert/swap
                # search_method = opea_name[:4]   # 确定是有效搜索/随机搜索

                # 确定每个工件的信息特征，用于后续选择需要操作的工件
                self.schedule_ins = Schedule_Instance(schedule, job_execute_time, self.file_name)
                self.job_info = self.schedule_ins.get_job_info(job_execute_time)
                self.schedule_ins.idle_time_insertion(schedule, job_execute_time, obj)

                # 根据上面的工件信息特征，去确定进行领域搜索涉及的工件信息stage
                # loca_machine, selected_job, oper_machine, oper_job = self.chosen_job(search_method_1, search_method_2,config_same_machine,stage,oper_method)
                neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name,self.jingying_num)
                if search_method_1 != 'sort':
                    break_info = False

                    loca_machine, selected_job, oper_job_list = neig_search.chosen_job(search_method_1, search_method_2,
                                                                                       config_same_machine, stage,
                                                                                       oper_method)
                    if selected_job is None:
                        i+=1
                        continue
                    update_obj = obj
                    for key in oper_job_list.keys():
                        oper_machine = key
                        for job in oper_job_list[key]:

                            oper_job = job
                            neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name,self.jingying_num)
                            update_schedule, update_obj, update_job_execute_time = neig_search.search_opea(
                                oper_method,obj,stage, loca_machine, selected_job, oper_machine, oper_job,search_method_1,True)
                            # print(0, update_schedule, update_obj)
                            # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
                            # dia = job_diagram(update_schedule, update_job_execute_time,
                            # self.file_name, 21)
                            # dia.pre()
                            # plt.savefig('./img1203/pic-{}.png'.format(21))

                            # if update_obj == obj and stage == 0 and search_method_1 == 'effe':      # 间接实现两步调
                            #     break_info = True
                            #     break
                            if update_obj < obj:
                                break_info = True
                                break
                            else:
                                neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj,
                                                                            self.file_name, self.jingying_num)
                                update_schedule, update_obj, update_job_execute_time = neig_search.search_opea(
                                    oper_method, obj, stage, loca_machine, selected_job, oper_machine, oper_job,
                                    search_method_1, False)
                                if update_obj < obj:
                                    break_info = True
                                    break
                        if break_info:
                            break
                stage = int(split_list[-1])
                job_block_rule = split_list[1]
                search_method_1 = split_list[0]
                sort_rule = split_list[3]
                A_or_D = split_list[2]
                if search_method_1 == 'sort':
                     update_schedule, update_obj, update_job_execute_time = neig_search.sort_(search_method_1, job_block_rule,A_or_D, stage,sort_rule)

                # loca_machine, selected_job, oper_job_list = neig_search.chosen_job(search_method_1, search_method_2,
                #                                                             config_same_machine, stage, oper_method)


                # print(1, update_schedule,update_obj)
                self.use_actions[exceuse_search][0] += 1        # 计算该搜索因子使用了几次，有效的有几次，具体改进了多少
                # 如果更优，则更新相关参数


                if update_obj < obj:
                    update = True
                    # print(0,update_schedule,update_obj)
                    # print('成功更新-----self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))
                    self.use_actions[exceuse_search][1] += 1
                    self.use_actions[exceuse_search][2] += (obj - update_obj)

                    # if obj == 0 or update_obj == 0:
                    #     print(1)

                    # new_list[index] = (update_schedule, update_obj, update_job_execute_time)
                # 如果没有更优，则保留
                    schedule = update_schedule
                    job_execute_time = update_job_execute_time
                    obj = update_obj
                    # if update_obj == obj and stage == 0 and search_method_1 == 'effe':
                    #     i += 1
                    # else:
                    #     i += 1
                else:
                    i += 1


                    # print(1, 'self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))

            if index < self.jingying_num and update:
                self.update_jingying += 1
                ect_value_2 = 0
                ddl_value_2 = 0
                for job in range(self.config.jobs_num):
                    job_makespan = update_job_execute_time[(self.config.stages_num - 1, job)]
                    if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                        ect_value_2 += (self.config.ect_windows[job] - job_makespan) * \
                                       self.config.ect_weight[job]
                    elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                        ddl_value_2 += (job_makespan - self.config.ddl_windows[job]) * \
                                       self.config.ddl_weight[job]

                self.delay_early.append((ddl_value_1 - ddl_value_2, ect_value_1 - ect_value_2))



            new_list.append([copy.deepcopy(schedule), copy.deepcopy(obj), copy.deepcopy(job_execute_time)])
            # new_list[index] = [final_schedule, final_obj, final_job_execute_time]
            if update:
                self.upadate_num += 1
        self.oper_same_machine.append((stage, config_same_machine))
        new_list = sorted(new_list, key=lambda x: x[1])
        self.inital_refset = copy.deepcopy(new_list)
        print(1)
            # self.diagram = diagram.job_diagram(new_list[key][index][0], new_list[key][index][2], i_count)
            # i_count += 1
            # self.diagram.pre()

            # schedule = self.inital_refset[key][index][0]
            # job_execute_time = self.inital_refset[key][index][2]
            # obj = self.inital_refset[key][index][1]
            # print('success')

    def schedule_split(self,schedule):

        job_on_machine = np.zeros((self.config.stages_num, self.config.jobs_num), dtype=int)
        job_seqence = np.zeros((self.config.stages_num, self.config.jobs_num), dtype=float)

        for item in schedule.keys():
            random_list = sorted([round(random.uniform(0, 1), 2) for _ in range(len(schedule[item]))])
            for index, job in enumerate(schedule[item]):
                job_on_machine[item[0]][job] = item[1]
                job_seqence[item[0]][job] = random_list[index]

        # for item in schedule.keys():
        #     for i_job in schedule[item]:
        #         job_on_machine[item[0]][i_job] = item[1]    # 0：代表在第0台机器上，1：代表在第1台机器上
        #         job_seqence[item[0]][i_job] =               #


        return job_on_machine, job_seqence



    def refer(self,schedule_1, schedule_2):
        # 生成每个解
        new_schedule = {}
        new_job_seqence =  {}
        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[stage]):
                new_schedule[(stage,machine)] = []
                new_job_seqence[(stage,machine)] = []

        gen_job_on_machine = np.zeros((self.config.stages_num, self.config.jobs_num), dtype=int)
        job_on_machine_1, job_seqence_1 = self.schedule_split(schedule_1)
        job_on_machine_2, job_seqence_2 = self.schedule_split(schedule_2)
        # 随机生成两个随机元素的序列，一个用于决定工件所在的机器，一个用于决定第0阶段，一个用于决定第1阶段
        random_list = [round(random.uniform(0, 1), 2) for _ in range(self.config.jobs_num)]
        for stage in range(self.config.stages_num):
            for job_index,i in enumerate(random_list):
                if i < 0.5:
                    machine = job_on_machine_1[stage][job_index]
                    pro = job_seqence_1[stage][job_index]
                else:
                    machine = job_on_machine_2[stage][job_index]
                    pro = job_seqence_2[stage][job_index]
                new_schedule[(stage, machine)].append(job_index)
                new_job_seqence[(stage, machine)].append(pro)

        sort_schedule = {}
        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[stage]):
                sort_schedule[(stage,machine)] = [x for _, x in sorted(zip(new_job_seqence[(stage, machine)], new_schedule[(stage, machine)]))]

        return sort_schedule

    def step(self,state, action):

        # 执行动作，得到优化后的值，
        # 根据动作，执行：这里就是调用搜索的主函数去调用去搞
        # ⭐⭐⭐
        # 返回trial，impro_mean_obj

        old_inital_refset = copy.deepcopy(self.inital_refset)
        # 获取当下的平均目标值
        self.excuse_action(action)
        action_name = self.action_space[action]
        # impro_degree = self.upadate_num / 20
        # for i_item in range(len(self.inital_refset)):
        #     for j_item in self.inital_refset[i_item][0]:
        #         if len(j_item) == 0:
        #             print('报错了！！！')



        obj_list = [item[1] for item in self.inital_refset]
        # diversity_degree = len(set(obj_list)) / len(obj_list)
        diversity_degree = (len(obj_list) - len(set(obj_list))) / len(obj_list)
        # cur_mean_obj = sum(item[1] for item in self.inital_refset) / 20   # 需要传入的参数
        cur_best_opt = self.inital_refset[0][1]    # 需要传入的参数
        # impro_mean_obj = self.ori_mean_obj - cur_mean_obj

        if cur_best_opt < self.best_opt:
            self.trial = 0
            self.from_now = 0
        else:
            self.trial += 1
            self.from_now += 1

        next_state = self.get_state(old_inital_refset)
        reward = self.get_reward(cur_best_opt,state,next_state)


        if cur_best_opt < self.best_opt:
            self.best_opt = cur_best_opt
            self.not_opt_iter = 0

        else:
            self.not_opt_iter +=1


        # 状态转移函数
        # state_name = self.state_space[state]

        # next_state_name = self.state_space[next_state]
        # if self.file_name == '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt':
        with open('./MDP.txt', 'a+') as fp:
            print('s:{0},   r:{2},    a:{1},    s_:{3}'.format(state, action_name, reward, next_state), file=fp)


        new_inital_refset = copy.deepcopy(self.inital_refset)
        # new_inital_refset = []

        # if next_state == 7:

        # if self.trial > 7:
        if self.from_now > self.config.jobs_num:
            # self.max_iter += 1
            self.from_now = 0
            jinying_i = 0
            for index in range(self.jingying_num,self.population-self.jingying_num):
                if jinying_i >= self.jingying_num:
                    jinying_i = 0
                schedule_1 = self.inital_refset[jinying_i][0]
                schedule_2 = self.inital_refset[index][0]
                sort_schedule = self.refer(schedule_1,schedule_2)
                neig_search = neighbo_search.Neighbo_Search(sort_schedule, None, None,self.file_name,self.jingying_num)
                neig_search.re_cal(sort_schedule)
                self.schedule_ins = Schedule_Instance(sort_schedule, neig_search.update_job_execute_time, self.file_name)
                new_obj = self.schedule_ins.cal(neig_search.update_job_execute_time)

                new_inital_refset[index] =  \
                    [copy.deepcopy(sort_schedule), copy.deepcopy(new_obj), copy.deepcopy(neig_search.update_job_execute_time)]
                # new_inital_refset.append([copy.deepcopy(sort_schedule), copy.deepcopy(new_obj), copy.deepcopy(neig_search.update_job_execute_time)])
                jinying_i += 1

            self.inital_refset = copy.deepcopy(new_inital_refset)
            # self.inital_refset = self.inital_refset[:10] + new_inital_refset

                # 计算一下


        return next_state, reward


    def rl(self):
        # q_table = self.build_q_table(N_STATES, ACTIONS)
        ori_impro_mean_obj = 0
        ori_trial = 0
        # for episode in range(MAX_EPISODES):
        step_counter = 0

        S = 1      # 初始状态为0
        CUM_REWARD = 0

        is_terminated = False
        # update_env(S, episode, step_counter)
        self.max_iter = 0
        a = 0
        history_R = []
        # while self.max_iter < 3:
        with open('./MDP.txt', 'a+') as fp:
            print('', file=fp)
        delta_list = []
        while self.not_opt_iter <= self.stop_iter:
            print('数据集的名字：{0}'.format(self.file_name))
            A = self.choose_action(S, self.q_table)
            S_, R = self.step(S, A)
            CUM_REWARD += R
            history_R.append(R)
            q_predict = self.q_table.loc[S, A]

            if self.not_opt_iter == self.stop_iter:
                q_target = R
            else:
                q_target = R + self.GAMMA * self.q_table.loc[S_, :].max()
            # if S_ != TerminalFlag:
            #     q_target = R + GAMMA * self.q_table.loc[S_, :].max()
            # else:
            #     q_target = R
            #     is_terminated = True
            # if np.isnan(q_target - q_predict):
            #     print(True)
            #     print(q_target)
            #     print(q_predict)
            # if self.max_iter == 0:
            self.q_table.loc[S, A] += self.ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1

            delta_list.append(np.linalg.norm(q_target - q_predict))
        delta = sum(delta_list)/len(delta_list)
        # if self.inital_obj - self.config.ture_opt < 0:
        #     CUM_REWARD += 1

        return copy.deepcopy(self.inital_refset), self.q_table,delta,CUM_REWARD

    def rl_execute(self):
        step_counter = 0
        S = 1  # 初始状态为0
        CUM_REWARD = 0

        self.max_iter = 0
        while self.not_opt_iter <= self.stop_iter:
            A = self.choose_action(S, self.q_table)
            S_, R = self.step(S, A)
            CUM_REWARD += R
            S = S_
            step_counter += 1

        return min([self.inital_refset[i][1] for i in range(len(self.inital_refset))]),CUM_REWARD

# if __name__ == '__main__':
#     rlll = RL_Q(8, ACTIONS)
#     q_table = rlll.rl()
#     print(q_table)

