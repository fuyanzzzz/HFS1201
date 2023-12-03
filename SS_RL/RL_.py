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
ALPHA = 0.05
GAMMA = 0.9
MAX_EPISODES = 15
FRESH_TIME = 0.3
TerminalFlag = "terminal"




class RL_Q():
    def __init__(self,n_states,n_actions,inital_refset,q_table,file_name):

        self.trial = 0
        self.ori_mean_obj = 9999999

        self.best_schedule = None
        self.inital_refset = inital_refset
        self.ini = inital_solution.HFS(file_name)
        self.inital_refset = sorted(self.inital_refset, key=lambda x: x[1])
        self.inital_obj = self.inital_refset[0][1]
        self.best_opt = self.inital_obj
        self.file_name = file_name
        self.max_iter = 0
        self.config = AllConfig.get_config(file_name)

        self.gen_action_space()
        # self.q_table = self.build_q_table(n_states,range(7))
        self.q_table = q_table
        # 以下的只是为了统计
        self.use_actions = {}
        # self.action_space_1 = ['effeinsert0', 'effeinsert1', 'randinsert0', 'randinsert1', 'effeswap0', 'effeswap1',
        #                 'randswap0', 'randswap1']

        self.action_space_1 = ['effe_insert_same_stuck_0','effe_swap_same_stuck_0','effe_insert_other_stuck_0','effe_swap_other_stuck_0',
                        'effe_insert_same_IF_1','effe_swap_same_IF_1','effe_insert_other_IF_1','effe_swap_other_IF_1',
                        'effe_insert_same_IW_1','effe_swap_same_IW_1','effe_insert_other_IW_1','effe_swap_other_IW_1',
                        'effe_insert_same_IP_1','effe_swap_same_IP_1','effe_insert_other_IP_1','effe_swap_other_IP_1',
                        'effe_insert_same_AF_1', 'effe_swap_same_AF_1','effe_insert_other_AF_1', 'effe_swap_other_AF_1',
                        'effe_insert_same_AW_1','effe_swap_same_AW_1','effe_insert_other_AW_1','effe_swap_other_AW_1',
                        'effe_insert_same_AP_1', 'effe_swap_same_AP_1','effe_insert_other_APe_1', 'effe_swap_other_AP_1',
                        'rand_insert_same_R_1','rand_swap_same_R_1','rand_insert_other_R_1', 'rand_swap_other_R_1']
        for i_action in self.action_space_1:
            self.use_actions[i_action] =[0, 0, 0]

        self.oper_same_machine = deque(maxlen=3)
        a_values = range(0, 2)
        b_values = range(0, 3)
        c_values = range(0, 2)
        combinations = list(product(a_values, b_values, c_values))


        # 创建字典
        self.state_space = dict(zip(combinations, list(range(len(combinations)))))
        print(1)




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
        # 第0阶段，卡住的工件在同一机器/其他机器，进行insert/swap
        self.action_space[0] = ['effe_insert_same_stuck_0','effe_swap_same_stuck_0']
        self.action_space[1] = ['effe_insert_other_stuck_0','effe_swap_other_stuck_0']

        # 第1阶段，单位增加目标值最多的方向的第一个工件，在同一机器/其他机器，进行insert/swap
        self.action_space[2] = ['effe_insert_same_IF_1','effe_swap_same_IF_1']
        self.action_space[3] = ['effe_insert_other_IF_1','effe_swap_other_IF_1']

        # 第1阶段，单位可改善最多的方向的权重贡献最大的工件，在同一机器/其他机器，进行insert/swap
        self.action_space[4] = ['effe_insert_same_IW_1','effe_swap_same_IW_1']
        self.action_space[5] = ['effe_insert_other_IW_1','effe_swap_other_IW_1']

        # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        self.action_space[6] = ['effe_insert_same_IP_1','effe_swap_same_IP_1']
        self.action_space[7] = ['effe_insert_other_IP_1','effe_swap_other_IP_1']

        # 第1阶段，单位可改善最多的方向的第一个工件，在同一机器/其他机器，进行insert/swap
        self.action_space[8] = ['effe_insert_same_AF_1', 'effe_swap_same_AF_1']
        self.action_space[9] = ['effe_insert_other_AF_1', 'effe_swap_other_AF_1']

        # 第1阶段，单位可改善最多的方向的权重贡献最大的工件，在同一机器/其他机器，进行insert/swap
        self.action_space[10] = ['effe_insert_same_AW_1','effe_swap_same_AW_1']
        self.action_space[11] = ['effe_insert_other_AW_1','effe_swap_other_AW_1']

        # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        self.action_space[12] = ['effe_insert_same_AP_1', 'effe_swap_same_AP_1']
        self.action_space[13] = ['effe_insert_other_APe_1', 'effe_swap_other_AP_1']

        # 第一阶段，随机
        self.action_space[14] = ['rand_insert_same_R_1','rand_swap_same_R_1']
        self.action_space[15] = ['rand_insert_other_R_1', 'rand_swap_other_R_1']



    def choose_action(self,state, q_table):
        state_table = q_table.loc[state, :]
        # if self.file_name[0] == '0':
        #     action_name = np.random.choice(range(len(self.action_space)))
        # else:
        #     if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        #         action_name = np.random.choice(range(len(self.action_space)))
        #     else:
        #         action_name = state_table.idxmax()

        if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
            action_name = np.random.choice(range(len(self.action_space)))
        else:
            action_name = state_table.idxmax()

        return action_name


    def get_state(self,trial, impro_degree, diversity_degree):
        # 根据传入的参数确定转移状态
        # state_space = {}
        #
        # if impro_degree >= diversity_degree:
        #     if trial == 0:
        #         state = 0
        #     elif trial > 0 and trial <= 20:
        #         state = 1
        #     elif trial > 20 and trial <= 50:
        #         state = 2
        #     else:
        #         state = 3
        # else:
        #     if trial == 0:
        #         state = 4
        #     elif trial > 0 and trial <= 20:
        #         state = 5
        #     elif trial > 20 and trial <= 50:
        #         state = 6
        #     else:
        #         state = 7
        # if self.trial > 100:
        #     state = 7
        # # if trial > 50:
        # #     state = 8

        '''
        第二种状态设置方式：
        1. 机器数 > 分段数
        2. 几代不优
        3. 修复的是否同一个机器
        '''
        job_block = []
        for i in self.ini.schedule_job_block.keys():
            for j in self.ini.schedule_job_block[i]:
                job_block.append(j)

        if self.config.machine_num_on_stage[0] < len(job_block):        # 若机器数 < 分段数：说明第一阶段被卡了
            a = 0
        else:
            a = 1

        if self.trial == 0:
            b = 0
        elif self.trial <= 5:
            b = 1
        else:
            b = 2

        # 修复的是否同一个机器，
        if len(self.oper_same_machine) == 3 and len(set(self.oper_same_machine)) == 1:
            c = 0
        else:
            c = 1

        return self.state_space[(a,b,c)]

    def get_reward(self, cur_best_opt,impro_degree,diversity_degree):
        if self.best_opt == 0:
            impor = (self.best_opt - cur_best_opt) / 1
        else:
            impor = (self.best_opt - cur_best_opt) / self.best_opt

        if self.trial == 0:
            reward = math.exp((impor + impro_degree + diversity_degree)/2)
        else:
            reward = -math.exp((impro_degree + diversity_degree)/2)
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
        return reward

        # 有可能是一步调已经调不limit == 50，直接接动了，如果受下一次的解，重新去调整
        # 但这样有一个问题，就是分成两个子种群就已经没有意义了

    def excuse_action(self,action):
        new_list = []
        count = 0
        i_count = 10    # 注释
        self.upadate_num = 0

        for index, item in enumerate(self.inital_refset):
            i = 0

            update = False
            schedule = item[0]
            job_execute_time = item[2]
            obj = item[1]


            while i < len(self.action_space[action]):
                exceuse_search = self.action_space[action][i]
                # schedule = copy.deepcopy(schedule)2
                # job_execute_time = copy.deepcopy(job_execute_time)
                # obj = copy.deepcopy(obj)
                print(0,schedule,obj)
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

                # stage = int(opea_name[-1])      # 确定操作的阶段
                # oper_method = opea_name[4]      # 确定是insert/swap
                # search_method = opea_name[:4]   # 确定是有效搜索/随机搜索

                # 确定每个工件的信息特征，用于后续选择需要操作的工件
                self.schedule_ins = Schedule_Instance(schedule, job_execute_time, self.file_name)
                self.job_info = self.schedule_ins.get_job_info(job_execute_time)
                self.schedule_ins.idle_time_insertion(schedule, job_execute_time, obj)

                # 根据上面的工件信息特征，去确定进行领域搜索涉及的工件信息stage
                # loca_machine, selected_job, oper_machine, oper_job = self.chosen_job(search_method_1, search_method_2,config_same_machine,stage,oper_method)
                neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name)
                loca_machine, selected_job, oper_job_list = neig_search.chosen_job(search_method_1, search_method_2,
                                                                            config_same_machine, stage, oper_method)
                update_obj = obj
                for key in oper_job_list.keys():
                    oper_machine = key
                    for job in oper_job_list[key]:
                        oper_job = job
                        neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name)
                        update_schedule, update_obj, update_job_execute_time = neig_search.search_opea(oper_method,obj,stage, loca_machine, selected_job, oper_machine, oper_job)
                        print(0, update_schedule, update_obj)
                        if update_obj < obj:
                            break

                # print(1, update_schedule,update_obj)
                self.use_actions[exceuse_search][0] += 1        # 计算该搜索因子使用了几次，有效的有几次，具体改进了多少
                # 如果更优，则更新相关参数
                if update_obj < obj:
                    update = True
                    # print(0,update_schedule,update_obj)
                    print('成功更新-----self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))
                    self.use_actions[exceuse_search][1] += 1
                    self.use_actions[exceuse_search][2] += (obj - update_obj)

                    if obj == 0 or update_obj == 0:
                        print(1)

                    # new_list[index] = (update_schedule, update_obj, update_job_execute_time)
                # 如果没有更优，则保留
                    schedule = update_schedule
                    job_execute_time = update_job_execute_time
                    obj = update_obj

                else:
                    i+=1

                    print(1, 'self.obj:{0},self.update_obj:{1}'.format(obj, update_obj))

            new_list.append([copy.deepcopy(schedule), copy.deepcopy(obj), copy.deepcopy(job_execute_time)])
            # new_list[index] = [final_schedule, final_obj, final_job_execute_time]
            if update:
                self.upadate_num += 1
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
        #
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


        # 获取当下的平均目标值
        self.excuse_action(action)
        impro_degree = self.upadate_num / 20
        for i_item in range(len(self.inital_refset)):
            for j_item in self.inital_refset[i_item][0]:
                if len(j_item) == 0:
                    print('报错了！！！')

        obj_list = [item[1] for item in self.inital_refset]
        # diversity_degree = len(set(obj_list)) / len(obj_list)
        diversity_degree = (len(obj_list) - len(set(obj_list))) / len(obj_list)
        # cur_mean_obj = sum(item[1] for item in self.inital_refset) / 20   # 需要传入的参数
        cur_best_opt = self.inital_refset[0][1]    # 需要传入的参数
        # impro_mean_obj = self.ori_mean_obj - cur_mean_obj

        reward = self.get_reward(cur_best_opt,impro_degree,diversity_degree)
        if cur_best_opt < self.best_opt:
            self.trial = 0
            self.best_opt = cur_best_opt

        else:
            self.trial += 1


        # 状态转移函数
        next_state = self.get_state(self.trial, impro_degree, diversity_degree)

        new_inital_refset = copy.deepcopy(self.inital_refset)
        # new_inital_refset = []

        if next_state == 7:

            self.max_iter += 1
            self.trial = 0
            for index in range(int(len(self.inital_refset) / 2)):
                schedule_1 = self.inital_refset[index][0]
                schedule_2 = self.inital_refset[int((len(self.inital_refset) / 2))+index][0]
                sort_schedule = self.refer(schedule_1,schedule_2)
                neig_search = neighbo_search.Neighbo_Search(sort_schedule, None, None,self.file_name)
                neig_search.re_cal(sort_schedule)
                self.schedule_ins = Schedule_Instance(sort_schedule, neig_search.update_job_execute_time, self.file_name)
                new_obj = self.schedule_ins.cal(neig_search.update_job_execute_time)

                new_inital_refset[int((len(self.inital_refset) / 2)) + index] =  \
                    [copy.deepcopy(sort_schedule), copy.deepcopy(new_obj), copy.deepcopy(neig_search.update_job_execute_time)]
                # new_inital_refset.append([copy.deepcopy(sort_schedule), copy.deepcopy(new_obj), copy.deepcopy(neig_search.update_job_execute_time)])

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
        S = 7       # 初始状态为0
        CUM_REWARD = 0

        is_terminated = False
        # update_env(S, episode, step_counter)
        self.max_iter = 0
        a = 0
        while self.max_iter < 2:
            print('数据集的名字：{0}'.format(self.file_name))
            A = self.choose_action(S, self.q_table)
            S_, R = self.step(S, A)
            CUM_REWARD += R
            q_predict = self.q_table.loc[S, A]

            if S_ == 7 and self.max_iter >= 3:
                q_target = R
            else:
                q_target = R + GAMMA * self.q_table.loc[S_, :].max()
            # if S_ != TerminalFlag:
            #     q_target = R + GAMMA * self.q_table.loc[S_, :].max()
            # else:
            #     q_target = R
            #     is_terminated = True
            if np.isnan(q_target - q_predict):
                print(True)
                print(q_target)
                print(q_predict)
            self.q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            step_counter += 1
        delta = np.linalg.norm(q_target - q_predict)
        # if self.inital_obj - self.config.ture_opt < 0:
        #     CUM_REWARD += 1

        return copy.deepcopy(self.inital_refset), self.q_table,delta,CUM_REWARD

    def rl_execute(self):
        step_counter = 0
        S = 0  # 初始状态为0
        CUM_REWARD = 0

        self.max_iter = 0
        while self.max_iter < 2:
            A = self.choose_action(S, self.q_table)
            S_, R = self.step(S, A)
            CUM_REWARD += R
            S = S_
            step_counter += 1

        return min([self.inital_refset[i][1] for i in range(len(self.inital_refset))])

# if __name__ == '__main__':
#     rlll = RL_Q(8, ACTIONS)
#     q_table = rlll.rl()
#     print(q_table)

