'''
-------------------------------------------------
File Name: d3qnsecond0316.py
Author: LRS
Create Time: 2023/3/16 00:47
-------------------------------------------------
'''
import random
from itertools import count
# from tensorboardX import SummaryWriter
# import gym
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
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
-------------------------------------------------
File Name: hfs.py
Author: LRS
Create Time: 2023/2/10 15:45
-------------------------------------------------
'''
import time

import torch
import numpy as np
import random
import copy




class Dueling_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.f1 = nn.Linear(state_dim, 512)
        self.f2 = nn.Linear(512, 256)

        self.val_hidden = nn.Linear(256, 128)
        self.adv_hidden = nn.Linear(256, 128)

        self.val = nn.Linear(128, 1)

        self.adv = nn.Linear(128, action_dim)

    def forward(self, x):
        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)

        adv = self.adv(adv_hidden)

        # 用平均值代替最大值，能够获得更好的稳定性
        adv_ave = torch.mean(adv, dim=1, keepdim=True)

        x = adv + val - adv_ave

        return x

    def select_action(self, state):
        with torch.no_grad():
            # print(state)
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()


class Memory(object):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > self.size():
            batch_size = self.size()
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

class Envior():
    def __init__(self,inital_refset,file_name,iter):

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
        # 以下的只是为了统计
        self.use_actions = {}   # 要放在主函数外，不然会重置掉

        self.iter_index = iter

        self.action_space_1 = ['effe_insert_same_DM_1', 'effe_swap_same_DM_1', 'effe_insert_other_DM_1', 'effe_swap_other_DM_1',
        'effe_insert_same_EM_1', 'effe_swap_same_EM_1', 'effe_insert_other_EM_1', 'effe_swap_other_EM_1',
        'rand_insert_same_M_1', 'rand_swap_same_M_1', 'rand_insert_other_M_1', 'rand_swap_other_M_1',
        'effe_insert_same_M_0', 'effe_swap_same_M_0', 'effe_insert_other_M_0', 'effe_swap_other_M_0',
        'dire_insert_same_dweight_1', 'dire_insert_other_dweight_1',
        'dire_insert_same_eweight_1', 'dire_insert_other_eweight_1',
        'effe_insert_same_DRM_1', 'effe_swap_same_DRM_1', 'effe_insert_other_DRM_1', 'effe_swap_other_DRM_1',
        'effe_insert_same_ERM_1', 'effe_swap_same_ERM_1', 'effe_insert_other_ERM_1', 'effe_swap_other_ERM_1']

        for i_action in self.action_space_1:
            self.use_actions[i_action] =[0, 0, 0]


    def gen_action_space(self):
        self.action_space = {}


        # 在第二阶段选择目标值最大的工件之一，进行insert/swap操作effe_insert_same_AF_1
        self.action_space[0] = ['effe_insert_same_DM_1', 'effe_swap_same_DM_1']     # 同一个机器
        self.action_space[1] = ['effe_insert_other_DM_1', 'effe_swap_other_DM_1']     # 同一个机器

        self.action_space[2] = ['effe_insert_same_EM_1', 'effe_swap_same_EM_1']     # 同一个机器
        self.action_space[3] = ['effe_insert_other_EM_1', 'effe_swap_other_EM_1']     # 同一个机器

        # 在第二阶段随机选择工件进行insert/swap:
        self.action_space[4] = ['rand_insert_other_M_1','rand_swap_other_M_1']      # 同一个机器
        self.action_space[5] = ['rand_insert_same_M_1','rand_swap_same_M_1',]      # 同一个机器
        # self.action_space[4] = ['effe_insert_same_M_0','effe_swap_same_M_0','effe_insert_other_M_0','effe_swap_other_M_0']      # 同一个机器


        # 第1阶段，单位可改善最多的方向的加工时间最小的工件，在同一机器/其他机器，进行insert/swap
        # self.action_space[10] = ['sort_stuck_A_S0_1']
        self.action_space[6] = ['dire_insert_same_dweight_1',]
        self.action_space[7] = ['dire_insert_other_dweight_1']

        self.action_space[8] = ['dire_insert_same_eweight_1',]
        self.action_space[9] = ['dire_insert_other_eweight_1']

        self.action_space[10] = ['effe_insert_same_DRM_1','effe_swap_same_DRM_1']      # 同一个机器
        self.action_space[11] = ['effe_insert_other_DRM_1','effe_swap_other_DRM_1']      # 同一个机器


        self.action_space[12] = ['effe_insert_same_ERM_1','effe_swap_same_ERM_1']      # 同一个机器
        self.action_space[13] = ['effe_insert_other_ERM_1','effe_swap_other_ERM_1']      # 同一个机器

    def get_reward(self,old_inital_refset):
      # update_num = 0
      # for i in range(int(len(self.inital_refset) / 2)):
      #     if old_inital_refset[i][0] != self.inital_refset[i][0] and self.inital_refset[i][1] < \
      #             old_inital_refset[i][1]:
      #         update_num += 1
      #
      # reward = update_num

      reward = old_inital_refset[0][1] - self.inital_refset[0][1]

      return reward
    def get_state(self,old_inital_refset,step_counter):
        self.reward = 0
        next_state = torch.zeros(5)
        ect_value = 0
        ddl_value = 0
        opt_item = self.inital_refset[0]
        job_execute_time = opt_item[2]

        for job in range(self.config.jobs_num):
            job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
            if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
            elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]

        # 状态：【工件数、机器数、TRW、早到和延误的倍数、self.trial】
        # next_state[0] = self.config.jobs_num
        # next_state[1] = self.config.machine_num_on_stage[0]
        # next_state[2] = self.config.T
        # next_state[3] = self.config.R
        # next_state[4] = self.config.W
        # next_state[5] = ddl_value / ect_value if ect_value!=0 else ddl_value
        # next_state[6] = self.trial
        # next_state[7] = step_counter

        # 遍历所有的工件，用工件的完工时间，减去早到或者延误的时间点
        early_error_2 = 0
        delay_error_2 = 0
        for job in range(self.config.jobs_num):
            if job_execute_time[(1,job)] > self.config.ddl_windows[job]:
                delay_error_2 += (job_execute_time[(1,job)] - self.config.ddl_windows[job])**2
            elif job_execute_time[(1,job)] < self.config.ect_windows[job]:
                early_error_2 += (self.config.ect_windows[job] - job_execute_time[(1,job)])**2



        # next_state[0] = self.config.T
        # next_state[1] = self.config.R
        next_state[0] = math.sqrt(delay_error_2/self.config.jobs_num)/100
        next_state[1] = math.sqrt(early_error_2/self.config.jobs_num)/100
        next_state[2] = min(ddl_value / ect_value if ect_value!=0 else ddl_value,100)/100
        next_state[3] = self.trial
        next_state[4] = step_counter


        # with open('./state.txt', 'a+') as fp:
        #     print('{0}'.format(next_state[0]), file=fp)
        #     print('{0}'.format(next_state[1]), file=fp)
        #     print('{0}'.format(next_state[2]), file=fp)
        #     print('{0}'.format(next_state[3]), file=fp)
        #     print('{0}'.format(next_state[4]), file=fp)
        #     print('', file=fp)

        reward = self.get_reward(old_inital_refset)

        return next_state,reward

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

            # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
            # dia = job_diagram(schedule, job_execute_time, self.file_name, 11)
            # dia.pre()
            # plt.savefig('./img1203/pic-{}.png'.format(11))


            while i < len(self.action_space[action]):
                exceuse_search = self.action_space[action][i]
                # schedule = copy.deepcopy(schedule)
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
                neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name)
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
                            neig_search = neighbo_search.Neighbo_Search(schedule, job_execute_time, obj, self.file_name)
                            update_schedule, update_obj, update_job_execute_time = neig_search.search_opea(oper_method,obj,stage, loca_machine, selected_job, oper_machine, oper_job,search_method_1)
                            print(0, update_schedule, update_obj)
                            # case_file_name = '1259_Instance_20_2_3_0,6_1_20_Rep4.txt'
                            # dia = job_diagram(update_schedule, update_job_execute_time,
                            # self.file_name, 21)
                            # dia.pre()
                            # plt.savefig('./img1203/pic-{}.png'.format(21))

                            if update_obj == obj and stage == 0 and search_method_1 == 'effe':      # 间接实现两步调
                                break_info = True
                                break
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


                if update_obj < obj or (update_obj == obj and stage == 0 and search_method_1 == 'effe'):
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
                    if update_obj == obj and stage == 0 and search_method_1 == 'effe':
                        i += 1
                    else:
                        i += 1

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



    def step(self,state, action,step_counter):

        # 执行动作，得到优化后的值，
        # 根据动作，执行：这里就是调用搜索的主函数去调用去搞
        # ⭐⭐⭐
        # 返回trial，impro_mean_obj

        old_inital_refset = copy.deepcopy(self.inital_refset)
        # 获取当下的平均目标值
        self.excuse_action(action)
        action_name = self.action_space[action]
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




        if cur_best_opt < self.best_opt:
            self.trial = 0
            self.best_opt = cur_best_opt

        else:
            self.trial += 1


        # 状态转移函数
        next_state,reward = self.get_state(old_inital_refset,step_counter)

        # if self.file_name == '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt':
        #     with open('./MDP.txt', 'a+') as fp:
        #         print('s:{0},   r:{2},    a:{1},    s_:{3}'.format(state_name, action_name, reward, next_state_name), file=fp)


        new_inital_refset = copy.deepcopy(self.inital_refset)


        if self.file_name == '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt':
            with open('./MDP.txt', 'a+') as fp:
                print('s:{0},   r:{2},    a:{1}'.format(state, action_name, reward), file=fp)

        # new_inital_refset = []

        # if next_state == 7:

        if self.trial > 7:
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








# writer = SummaryWriter('logs/dueling_DQN2')

n_action = 14
n_state = 5
class rl_main():
    def __init__(self):
        self.n_action = n_action
        self.gen_nn()


    def gen_nn(self):
        self.init_paras()
        self.target_network = Dueling_DQN(n_state, n_action)
        self.network = Dueling_DQN(n_state, n_action)
        self.target_network.load_state_dict(self.network.state_dict())
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0001)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0001)
        self.memory = Memory(self.REPLAY_MEMORY)



    def init_paras(self):
        self.GAMMA = 0.99
        # self.BATH = 256  # 批量训练256
        self.BATH = 256  # 批量训练256
        self.EXPLORE = 2000000
        self.REPLAY_MEMORY = 50000  # 经验池容量5W
        self.BEGIN_LEARN_SIZE = 1024
        # self.UPDATA_TAGESTEP = 200  # 目标网络的更新频次
        self.UPDATA_TAGESTEP = 10  # 目标网络的更新频次
        self.learn_step = 0
        # writer = SummaryWriter('logs/dueling_DQN2')
        self.FINAL_EPSILON = 0.00001
        self.epsilon = 1
        self.min_epsilon = 0.1


    def rl_excuse(self,inital_refset, file_name, iter,inital_obj):
        # 初始化环境
        self.env = Envior(inital_refset, file_name, iter)
        step_counter = 0
        # 初始化状态
        state,_ = self.env.get_state(inital_refset,step_counter)
        # 初始化当前幕的数据
        episode_reward = 0
        episode_step = []

        # 设置终止状态
        while step_counter < self.env.config.jobs_num * 2:

            p = random.random()
            # 动作选择
            self.epsilon = max(self.epsilon - iter * 0.01, self.min_epsilon)
            if p < self.epsilon:
                action = random.randint(0, n_action - 1)
            else:
                state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
                action = self.network.select_action(state_tensor)



            # 根据（状态，动作）得到step序列
            next_state, reward = self.env.step(state,action,step_counter)
            done = False
            if step_counter + 1 == self.env.config.jobs_num * 2:
                done = True
            # print(next_state,reward,done)


            # 将得到序列添加到经验池
            episode_step.append([state, next_state, action, reward, done])


            # 只有当经验池中的样本数量大于批量训练的数量，才会执行训练
            if self.memory.size() > self.BEGIN_LEARN_SIZE:
                self.learn_step += 1

                # 每隔一段时间，更新目标网络
                if self.learn_step % self.UPDATA_TAGESTEP:
                    self.target_network.load_state_dict(self.network.state_dict())

                # 训练的时候，从经验池中进行采样
                batch = self.memory.sample(self.BATH, False)
                batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                batch_state = torch.as_tensor([item.cpu().detach().numpy() for item in batch_state], dtype=torch.float)
                batch_next_state = torch.as_tensor([item.cpu().detach().numpy() for item in batch_next_state],
                                                   dtype=torch.float)
                batch_action = torch.as_tensor(batch_action, dtype=torch.long).unsqueeze(1)
                batch_reward = torch.as_tensor(batch_reward, dtype=torch.float).unsqueeze(1)
                batch_done = torch.as_tensor(batch_done, dtype=torch.long).unsqueeze(1)

                with torch.no_grad():
                    target_Q_next = self.target_network(batch_next_state)  # 从目标网络中

                    # 将next_state输入预测网络，并选择最优的状态动作价值函数
                    Q_next = self.network(batch_next_state)
                    Q_max_action = torch.argmax(Q_next, dim=1, keepdim=True)

                    # 再将这个输入目标网络中
                    y = batch_reward + target_Q_next.gather(1, Q_max_action)

                loss = F.mse_loss(self.network(batch_state).gather(1, batch_action), y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # writer.add_scalar('loss', loss.item(), global_step=learn_step)

                # if epsilon > FINAL_EPSILON: ## 减小探索
                #     epsilon -= (0.1 - FINAL_EPSILON) / EXPLORE
                loss_item = loss.item()
            else:
                loss_item = 0


            state = next_state
            step_counter += 1
        gap_opt = inital_obj - self.env.inital_refset[0][1]
        for item in episode_step:
            episode_reward += item[3]/gap_opt
            self.memory.add((item[0],item[1],item[2],item[3]/gap_opt,item[4]))

        return copy.deepcopy(self.env.inital_refset),loss_item, episode_reward

    def rl_excuse_case(self, inital_refset, file_name, iter):
        # 初始化环境
        self.env = Envior(inital_refset, file_name, iter)
        # 初始化状态
        step_counter = 0
        state,_ = self.env.get_state(inital_refset,step_counter)
        # 初始化当前幕的数据
        episode_reward = 0
        # 设置终止状态
        step_counter = 0
        while step_counter < self.env.config.jobs_num:

            p = random.random()
            # 动作选择
            self.epsilon = max(self.epsilon - iter * 0.01, self.min_epsilon)
            if p < self.epsilon:
                action = random.randint(0, n_action - 1)
            else:
                state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
                action = self.network.select_action(state_tensor)

            # 根据（状态，动作）得到step序列
            next_state, reward = self.env.step(state, action,step_counter)
            done = False
            if step_counter + 1 == self.env.config.jobs_num:
                done = True
            # print(next_state,reward,done)
            episode_reward += reward

            state = next_state
            step_counter += 1

        return min([self.env.inital_refset[i][1] for i in range(len(self.env.inital_refset))]), episode_reward
