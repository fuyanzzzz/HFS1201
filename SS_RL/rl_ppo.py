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
'''
@Author  ：Yan JP
@Created on Date：2023/4/19 17:31 
'''

# 代码用于离散环境的模型
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ----------------------------------- #
# 构建策略网络--actor
# ----------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_actions)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b, n_actions]
        x = F.softmax(x, dim=1)  # [b, n_actions]  计算每个动作的概率
        return x


# ----------------------------------- #
# 构建价值网络--critic
# ----------------------------------- #

class ValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, 1)

    def forward(self, x):
        x = self.fc1(x)  # [b,n_states]-->[b,n_hiddens]
        x = F.relu(x)
        x = self.fc2(x)  # [b,n_hiddens]-->[b,1]  评价当前的状态价值state_value
        return x


# ----------------------------------- #
# 构建模型
# ----------------------------------- #

class PPO:
    def __init__(self, n_states, n_hiddens, n_actions,
                 actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        # 实例化策略网络
        self.actor = PolicyNet(n_states, n_hiddens, n_actions).to(device)
        # 实例化价值网络
        self.critic = ValueNet(n_states, n_hiddens).to(device)
        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  # 折扣因子
        self.lmbda = lmbda  # GAE优势函数的缩放系数
        self.epochs = epochs  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数
        self.device = device

    # 动作选择
    def take_action(self, state):
        # 维度变换 [n_state]-->tensor[1,n_states]
        state = torch.tensor(state[np.newaxis, :]).to(self.device)
        # 当前状态下，每个动作的概率分布 [1,n_states]
        probs = self.actor(state)
        # 创建以probs为标准的概率分布
        action_list = torch.distributions.Categorical(probs)
        # 依据其概率随机挑选一个动作
        action = action_list.sample().item()
        return action

    # 训练
    def learn(self, transition_dict):
        # 提取数据集
        states = torch.tensor([item.numpy() for item in transition_dict['states']],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).to(self.device).view(-1, 1)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).to(self.device).view(-1, 1)
        next_states = torch.tensor([item.numpy() for item in transition_dict['next_states']], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device).view(-1, 1)


        # 目标，下一个状态的state_value  [b,1]
        next_q_target = self.critic(next_states)
        # 目标，当前状态的state_value  [b,1]
        td_target = rewards + self.gamma * next_q_target * (1 - dones)
        # 预测，当前状态的state_value  [b,1]
        td_value = self.critic(states)
        # 目标值和预测值state_value之差  [b,1]
        td_delta = td_target - td_value

        # 时序差分值 tensor-->numpy  [b,1]
        td_delta = td_delta.cpu().detach().numpy()
        advantage = 0  # 优势函数初始化
        advantage_list = []

        # 计算优势函数
        for delta in td_delta[::-1]:  # 逆序时序差分值 axis=1轴上倒着取 [], [], []
            # 优势函数GAE的公式
            advantage = self.gamma * self.lmbda * advantage + delta
            advantage_list.append(advantage)
        # 正序
        advantage_list.reverse()
        # numpy --> tensor [b,1]
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)

        # 策略网络给出每个动作的概率，根据action得到当前时刻下该动作的概率
        old_log_probs = torch.log(self.actor(states).gather(1, actions)).detach()

        # 一组数据训练 epochs 轮
        for _ in range(self.epochs):
            # 每一轮更新一次策略网络预测的状态
            log_probs = torch.log(self.actor(states).gather(1, actions))
            # 新旧策略之间的比例
            ratio = torch.exp(log_probs - old_log_probs)
            # 近端策略优化裁剪目标函数公式的左侧项
            surr1 = ratio * advantage
            # 公式的右侧项，ratio小于1-eps就输出1-eps，大于1+eps就输出1+eps
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage

            # 策略网络的损失函数
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            # 价值网络的损失函数，当前时刻的state_value - 下一时刻的state_value
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

            # 梯度清0
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 梯度更新
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        return actor_loss.item(),critic_loss.item()




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

        self.iprovement_time = 0


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
        next_state = torch.zeros(6)
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
        early_a = math.sqrt(early_error_2/self.config.jobs_num)
        delay_b = math.sqrt(delay_error_2/self.config.jobs_num)
        # next_state[0] = min(delay_b/100,1)
        # next_state[1] = min(early_a/100,1)
        # next_state[2] = min(delay_b / early_a if early_a!=0 else delay_b,100)/100
        # next_state[3] = min(ddl_value / ect_value if ect_value!=0 else ddl_value,100)/100
        # next_state[4] = self.trial/7
        # next_state[5] = step_counter/40

        next_state[0] = delay_b
        next_state[1] = early_a
        next_state[2] = delay_b / early_a if early_a!=0 else delay_b
        next_state[3] = ddl_value / ect_value if ect_value!=0 else ddl_value
        next_state[4] = self.trial
        next_state[5] = step_counter


        # next_state[0] = min(delay_b/100,1)
        # next_state[1] = min(early_a/100,1)
        # next_state[0] = min(ddl_value / ect_value if ect_value!=0 else ddl_value,100)/100
        # next_state[1] = self.trial/7
        # next_state[2] = step_counter/40


        # with open('./state.txt', 'a+') as fp:
        #     print('{0}'.format(next_state[0]), file=fp)
        #     print('{0}'.format(next_state[1]), file=fp)
        #     print('{0}'.format(next_state[2]), file=fp)
        #     print('{0}'.format(next_state[3]), file=fp)
        #     print('{0}'.format(next_state[4]), file=fp)
        #     print('', file=fp)

        reward = self.get_reward(old_inital_refset)
        result = torch.any(torch.isnan(next_state))
        # with open('./state.txt', 'a+') as fp:
        #
        #     print(result, file=fp)
        #     print(next_state, file=fp)
        #     print('',file=fp)


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
            self.iprovement_time += 1

        else:
            self.trial += 1


        # 状态转移函数
        next_state,reward = self.get_state(old_inital_refset,step_counter)

        # if self.file_name == '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt':
        #     with open('./MDP.txt', 'a+') as fp:
        #         print('s:{0},   r:{2},    a:{1},    s_:{3}'.format(state_name, action_name, reward, next_state_name), file=fp)


        new_inital_refset = copy.deepcopy(self.inital_refset)


        # if self.file_name == '1236_Instance_20_2_3_0,6_0,2_20_Rep1.txt':
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


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ----------------------------------------- #
# 参数设置
# ----------------------------------------- #

num_episodes = 100  # 总迭代次数
gamma = 0.9  # 折扣因子
actor_lr = 1e-3  # 策略网络的学习率
critic_lr = 1e-2  # 价值网络的学习率
n_hiddens = 16  # 隐含层神经元个数
env_name = 'CartPole-v1'
return_list = []  # 保存每个回合的return

# writer = SummaryWriter('logs/dueling_DQN2')

n_actions = 14
n_states = 6
class rl_main():
    def __init__(self,i_text):
        self.n_action = n_actions
        self.gen_nn()
        self.i_text = i_text


    def gen_nn(self):
        # self.init_paras()

        self.agent = PPO(n_states=n_states,  # 状态数
                    n_hiddens=n_hiddens,  # 隐含层数
                    n_actions=n_actions,  # 动作数
                    actor_lr=actor_lr,  # 策略网络学习率
                    critic_lr=critic_lr,  # 价值网络学习率
                    lmbda=0.95,  # 优势函数的缩放因子
                    epochs=10,  # 一组序列训练的轮次
                    eps=0.2,  # PPO中截断范围的参数
                    gamma=gamma,  # 折扣因子
                    device=device
                    )

    # def init_paras(self):
        # self.GAMMA = 0.99
        # self.BATH = 256  # 批量训练256
        # self.EXPLORE = 2000000
        # self.REPLAY_MEMORY = 50000  # 经验池容量5W
        # self.BEGIN_LEARN_SIZE = 1024
        # # self.UPDATA_TAGESTEP = 200  # 目标网络的更新频次
        # self.UPDATA_TAGESTEP = 20  # 目标网络的更新频次
        # self.learn_step = 0
        # # writer = SummaryWriter('logs/dueling_DQN2')
        # self.epsilon = 1
        # self.min_epsilon = 0.5


    def rl_excuse(self,inital_refset, file_name, iter,inital_obj):
        # 初始化环境
        self.env = Envior(inital_refset, file_name, iter)
        step_counter = 0
        # 初始化状态
        state, _ = self.env.get_state(inital_refset, step_counter)
        # 初始化当前幕的数据
        episode_reward = 0

        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }

        episode_step = []

        while step_counter < self.env.config.jobs_num * 2:

            # 动作选择
            action = self.agent.take_action(state)


            # 根据（状态，动作）得到step序列
            next_state, reward = self.env.step(state, action, step_counter)
            done = False
            if step_counter + 1 == self.env.config.jobs_num * 2:
                done = True
            # print(next_state,reward,done)
            episode_step.append([state, next_state, action, reward, done])

            gap_opt = inital_obj - self.env.inital_refset[0][1]

            state = next_state
            # episode_reward += 1
            step_counter += 1

        gap_opt = inital_obj - self.env.inital_refset[0][1]
        for i_index, item in enumerate(episode_step):
            if gap_opt == 0:
                gap_opt = 1
            if item[3]/gap_opt >0 :
                episode_reward += 1/self.env.config.jobs_num
            # else:
            #     episode_reward += -0.01
            episode_reward += min(item[3] / gap_opt,self.i_text)
            transition_dict['states'].append(item[0])
            transition_dict['actions'].append(item[2])
            transition_dict['next_states'].append(item[1])
            # if item[3]/gap_opt >0 :
            transition_dict['rewards'].append(min(item[3] / gap_opt,self.i_text)+ 1/self.env.config.jobs_num)
            # else:
            #     transition_dict['rewards'].append(-0.01)
            transition_dict['dones'].append(item[4])


        # 保存每个回合的return
        return_list.append(episode_reward)
        # 模型训练
        actor_loss,critic_loss = self.agent.learn(transition_dict)

        return copy.deepcopy(self.env.inital_refset),episode_reward,actor_loss,critic_loss



    def rl_excuse_case(self, inital_refset, file_name, iter,inital_obj):
        # 初始化环境
        self.env = Envior(inital_refset, file_name, iter)
        step_counter = 0
        # 初始化状态
        state, _ = self.env.get_state(inital_refset, step_counter)
        # 初始化当前幕的数据
        episode_reward = 0

        episode_step = []

        while step_counter < self.env.config.jobs_num * 2:

            # 动作选择
            action = self.agent.take_action(state)

            # 根据（状态，动作）得到step序列
            next_state, reward = self.env.step(state, action, step_counter)
            done = False
            if step_counter + 1 == self.env.config.jobs_num * 2:
                done = True

            episode_step.append([state, next_state, action, reward, done])
            state = next_state
            # episode_reward += 1
            step_counter += 1

        gap_opt = inital_obj - self.env.inital_refset[0][1]
        for i_index, item in enumerate(episode_step):
            if gap_opt == 0:
                gap_opt = 1
            if item[3] / gap_opt > 0:
                episode_reward += 1 / self.env.config.jobs_num
            episode_reward += min(item[3] / gap_opt,self.i_text)


        return min([self.env.inital_refset[i][1] for i in range(len(self.env.inital_refset))]), episode_reward


