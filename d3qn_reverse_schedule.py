'''
-------------------------------------------------
File Name: d3qn_reverse_schedule.py
Author: LRS
Create Time: 2023/3/16 09:19
-------------------------------------------------
'''
'''
-------------------------------------------------
File Name: d3qnsecond0316.py
Author: LRS
Create Time: 2023/3/16 00:47
-------------------------------------------------
'''
import random
from itertools import count
from tensorboardX import SummaryWriter
# import gym
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

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
job_process_time = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=2, max_rows=10,
                              usecols=(1, 3), unpack=True, dtype=int)
ect_delay_wight = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=14, max_rows=10,
                             usecols=(2, 3), unpack=True, dtype=int)
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = (np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=25, max_rows=10,
                              dtype=int))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

setup_time = {}
setup_time[0] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=37, max_rows=10,
                           dtype=int)
setup_time[1] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=48, max_rows=10,
                           dtype=int)

jobs_num = int(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(0),
                          dtype=int))
jobs = list(range(jobs_num))

stages_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(2),
                        dtype=int)
total_mahcine_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1,
                               usecols=(1), dtype=int)


# 　读取数据
# i = 0
# job_process_time = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=2, max_rows=20,
#                               usecols=(1, 3,5,7), unpack=True, dtype=int))
# ect_delay_wight = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=24, max_rows=20,
#                              usecols=(2, 3), unpack=True, dtype=int))
# ect_weight = ect_delay_wight[0]
# ddl_weight = ect_delay_wight[1]
# due_date_windows = torch.tensor((np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=45, max_rows=20,
#                               dtype=int)))
# ddl_windows = [i[1] for i in due_date_windows]
# ect_windows = [i[0] for i in due_date_windows]
#
# setup_time = {}
# setup_time[0] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=67, max_rows=20,
#                            dtype=int))
# setup_time[1] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=88, max_rows=20,
#                            dtype=int))
# setup_time[2] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=109, max_rows=20,
#                            dtype=int))
# setup_time[3] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=130, max_rows=20,
#                            dtype=int))
#
# jobs_num = torch.tensor(int(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1, usecols=(0),
#                           dtype=int)))
# jobs = torch.tensor(list(range(jobs_num)))
#
# stages_num = (np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1, usecols=(2),
#                         dtype=int))
# total_mahcine_num = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1,
#                                usecols=(1), dtype=int))

machine_num_on_stage = []
for job in range(stages_num):
    machine_num_on_stage.append(int(total_mahcine_num / stages_num))

# 　声明schedule的字典变量
schedule = {}
for stage in range(stages_num):
    for machine in range(machine_num_on_stage[stage]):
        schedule[(stage, machine)] = []

# 声明一个空的每个工件的完工时间
machine_completion_time = {}
for stage in range(stages_num):
    for machine in range(machine_num_on_stage[stage]):
        machine_completion_time[(stage, machine)] = [-1, -1]

# 声明一个阶段的完工时间的字典
job_completion_time = {}
for stage in range(stages_num):
    job_completion_time[stage] = np.zeros(jobs_num, dtype=int)

#　声明机器上的第一个工件的开始加工时间
machine_first_job_start_time = {}
for stage in range(stages_num):
    for machine in range(machine_num_on_stage[stage]):
        machine_first_job_start_time[(stage, machine)] = [-1, -1]



job_execute_time = {}
for stage in range(stages_num):
    for job in range(jobs_num):
        job_execute_time[(stage,job)] = [0,0]

machine_first_job_execute_time = {}
for stage in range(stages_num):
    for machine in range(machine_num_on_stage[stage]):
        machine_first_job_execute_time[(stage,machine)] = [-1,-1,-1]




def get_reset():
    # 　声明schedule的字典变量
    schedule = {}
    for stage in range(stages_num):
        for machine in range(machine_num_on_stage[stage]):
            schedule[(stage, machine)] = []

    job_execute_time = {}
    for stage in range(stages_num):
        for job in range(jobs_num):
            job_execute_time[(stage, job)] = [0, 0]

    machine_first_job_execute_time = {}
    for stage in range(stages_num):
        for machine in range(machine_num_on_stage[stage]):
            machine_first_job_execute_time[(stage, machine)] = [-1, -1, -1]

    return schedule, job_execute_time

# 声明一个变量，用于存储每个阶段的工件调度顺序【用于第二阶段的local search】
job_sort_on_satges = {}


def intal_variable():
    # 　声明schedule的字典变量
    schedule = {}
    for stage in range(stages_num):
        for machine in range(machine_num_on_stage[stage]):
            schedule[(stage, machine)] = []

    # 声明一个空的每个工件的完工时间
    machine_completion_time = {}
    for stage in range(stages_num):
        for machine in range(machine_num_on_stage[stage]):
            machine_completion_time[(stage, machine)] = [-1, -1]

    # 声明一个阶段的完工时间的字典
    job_completion_time = {}
    for stage in range(stages_num):
        job_completion_time[stage] = np.zeros(jobs_num, dtype=int)

    # 声明一个每个阶段的工件排序

    return schedule,machine_completion_time,job_completion_time


def job_assignment(stage,job,machine,schedule,job_completion_time,machine_completion_time):
    '''
    传入的参数：6个
    1. 第一阶段的工件加工顺序
    2. 开始调度阶段，结束调度阶段
    3. 所有阶段上的工件加工顺序

    传出的参数：4个
    1. 三个更新的变量
    2. 当前的目标值【未进行空闲插入程序】
    '''

    if stage == 0:
        pro_job = machine_completion_time[(stage,machine)][0]
        # 如果是第一个阶段的第一个工件
        if pro_job == -1:
            # ect = 该机器上最后一个工件的完工时间 +　切换时间　　【第一阶段的第一个工件：０＋０】
            ect_value = machine_completion_time[(stage,machine)][1] + 0

        else:   # 第一阶段，非第一个工件　　ect = 该机器上最后一个工件的完工时间 +　切换时间
            ect_value = machine_completion_time[(stage,machine)][1] + setup_time[stage][pro_job][job]

    # 如果非第一阶段
    else:
        pro_job = machine_completion_time[(stage,machine)][0]
        # 如果非第一阶段，但是该机器上的第一个工件
        if pro_job == -1:
            # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b = 0】
            job_on_pro_machine = job_completion_time[stage - 1][job]
            ect_value = max(machine_completion_time[(stage,machine)][1],job_on_pro_machine) + 0

        else:   # 非第一阶段，且非该机器上的第一个工件
            # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b ！＝ 0】
            job_on_pro_machine = job_completion_time[stage-1][job]
            ect_value = max(job_on_pro_machine,machine_completion_time[(stage,machine)][1]) + setup_time[stage][pro_job][
                job]  # 机器最早可开始加工时间 = 上个工件的完工时间 + 切换时间



    job_completion_time[stage][job] = max(0,ect_value) + job_process_time[stage][job]
    schedule[(stage,machine)].append(job)
    # schedule[machine, list(schedule[machine]).index(-1)] = job
    machine_completion_time[(stage,machine)][0] = job
    machine_completion_time[(stage,machine)][1] = max(0,ect_value) + job_process_time[stage][job]


    return schedule, job_completion_time, machine_completion_time



process_time_all_stages = []
for i in range(jobs_num):
    process_time_all_stages.append(sum([j[i] for j in job_process_time]))
b = np.array(ddl_windows) - np.array(process_time_all_stages)
job_sort_osl = np.argsort(b)  # 根据ddl或者工件的调度顺序

class Envior():
    def __init__(self,job_execute_time,machine_first_job_execute_time,schedule,job_sort_osl):
        self.job_execute_time = job_execute_time
        # self.machine_first_job_execute_time = machine_first_job_execute_time
        self.schedule = schedule
        self.reward = 0
        self.jobs_sort = job_sort_osl
        self.job_index = jobs_num-1
        self.stage_index = stages_num-1


    def get_state(self):
        n_state = torch.zeros(15)
        remanin_process_time = torch.zeros(10)
        renmain_setup_time = torch.zeros(10)
        ect_start_time = torch.zeros(10)
        ddl_ten = torch.zeros(10)
        ect_ten = torch.zeros(10)
        ect_machine_statrt_time = []

        # for index, job in enumerate(self.jobs_sort[max(0,self.job_index-9):self.job_index+1]):
        for index, job in enumerate(self.jobs_sort[self.job_index:max(-1,self.job_index-10):-1]):
            ddl_ten[index] = ddl_windows[job]
            ect_ten[index] = ect_windows[job]
            for stage in range(self.stage_index, -1,-1):
                remanin_process_time[index] += job_process_time[stage][job]      # 剩余的加工时间的加总
                renmain_setup_time[index] += sum(setup_time[stage][job])        # 剩余的平均切换时间
                # ect_start_time [index] = max(self.job_completion_time[stage-1][job] , min_ect_machine_statrt_time)

        renmain_setup_time = renmain_setup_time / jobs_num
        # 下一个阶段该工件的开始加工时间 - 剩余待加工的时间 - 剩余待切换的平均值
        for index, job in enumerate(self.jobs_sort[self.job_index:max(-1, self.job_index - 10):-1]):
            if self.stage_index == stages_num-1:
                n_state[index] = (ddl_ten[index] - remanin_process_time[index] - renmain_setup_time[index]) / ddl_weight[job]
            else:
                n_state[index] = (self.job_execute_time[(self.stage_index+1,job)][0] - remanin_process_time[index] - renmain_setup_time[index])/ddl_weight[job]

        for machine in range(machine_num_on_stage[self.stage_index]):
            # 根据schuedule和工件完工时间去确定
            if self.schedule[(self.stage_index,machine)]:
                job = self.schedule[(self.stage_index,machine)][0]
                n_state[10 + machine] = self.job_execute_time[(self.stage_index,job)][0]
            else:
                n_state[10 + machine] = -1
            # n_state[10 + machine] = self.machine_first_job_execute_time[(self.stage_index, machine)][1]

        n_state[13] = self.job_index / (jobs_num - 1)
        n_state[14] = self.stage_index / (stages_num - 1)
        self.state = n_state

        return self.state

    def reverse_job_assignment(self,stage, job, machine):
        '''
        传入的参数：6个
        1. 第一阶段的工件加工顺序
        2. 开始调度阶段，结束调度阶段
        3. 所有阶段上的工件加工顺序

        传出的参数：4个
        1. 三个更新的变量
        2. 当前的目标值【未进行空闲插入程序】
        '''

        # ect_value值为该机器上的第一个工件的开始加工时间

        '''
        维护变量：
        1. job_execute_time：工件在每个阶段的【开始加工时间,结束加工时间】
        2. machine_first_job_execute_time : 每个机器上第一个工件的【开始加工时间,结束加工时间】
        3. schedule
        '''

        if self.stage_index == stages_num - 1:
            if self.schedule[(self.stage_index,machine)]:
                later_job = self.schedule[(self.stage_index,machine)][0]
            else:
                later_job = -1
            # later_job = machine_first_job_execute_time[(stage, machine)][0]
            # 如果是最后一个阶段的最后一个工件【此时该机器上还没有调度工件】
            if later_job == -1:
                # 该机器上还没有工件，第一个工件就应该贴着他的逾期进行生产
                est_avail_time = ddl_windows[job] - 0

            else:  # 第一阶段，非第一个工件　　ect = min（该机器上第一个工件的开始加工时间 + 切换时间 + 工件的加工时间 ，该工件自己的due——date）
                est_avail_time = min(
                    (self.job_execute_time[(self.stage_index,later_job)][0] - setup_time[self.stage_index][job][later_job]),
                    ddl_windows[job])

        # 如果非第一阶段
        else:
            if self.schedule[(self.stage_index, machine)]:
                later_job = self.schedule[(self.stage_index, machine)][0]
            else:
                later_job = -1
                # 如果非最后一个阶段，但是该机器上的第一个工件
            if later_job == -1:
                # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b = 0】
                job_on_later_stage_start_time = self.job_execute_time[(self.stage_index + 1, job)][0]
                est_avail_time = job_on_later_stage_start_time - 0

            else:  # 非第一阶段，且非该机器上的第一个工件
                # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b ！＝ 0】
                job_on_later_stage_start_time = self.job_execute_time[(self.stage_index + 1, job)][0]
                est_avail_time = min(
                    (self.job_execute_time[(self.stage_index,later_job)][0] - setup_time[self.stage_index][job][later_job]),
                    job_on_later_stage_start_time)

        # self.machine_first_job_execute_time[(stage, machine)][0] = job
        # self.machine_first_job_execute_time[(stage, machine)][1] = est_avail_time - job_process_time[stage][job]
        # self.machine_first_job_execute_time[(stage, machine)][2] = est_avail_time

        self.job_execute_time[(self.stage_index, job)][0] = est_avail_time - job_process_time[self.stage_index][job]
        self.job_execute_time[(self.stage_index, job)][1] = est_avail_time

        self.schedule[(self.stage_index  , machine)].insert(0, job)

        # return self.schedule, self.job_execute_time, self.machine_first_job_execute_time
        return self.schedule, self.job_execute_time

    def reset(self):
        self.jobs_sort = job_sort_osl
        self.reward = 0
        self.job_index = jobs_num-1
        self.stage_index = stages_num-1
        self.schedule,self.job_execute_time = get_reset()
        self.state = self.get_state()

        return self.state


    def update(self, cur_job, machine):

        # new——update
        # 确保所有工件不会在时间0之前开始
        # 当前调度的这个工件是否在时间0之前开始，为了减少可能的调度：当前阶段的加工时间-剩余阶段的该工件的加工时间 - 剩余平均切换时间
        # 判断需要该工件如果要符合规则，需要移动的距离
        for stage in range(self.stage_index,stages_num-1):
            for job in self.schedule[(stage,machine)]:
                job_index = self.schedule[(stage,machine)].index(job)
                if self.job_execute_time[(stage,job)][0] >= self.job_execute_time[(stage,job)][1]:
                    print(True)
                if job_index + 1 < len(self.schedule[(stage,machine)]):
                    later_job = self.schedule[(stage,machine)][job_index + 1]
                    if self.job_execute_time[(stage,job)][1] + setup_time[stage][job][later_job] > self.job_execute_time[(stage,later_job)][0]:
                        print(True)
                if stage+1 < stages_num:
                    if self.job_execute_time[(stage,job)][1] > self.job_execute_time[(stage+1,job)][0]:
                        print(True)

        pre_job = None
        # 当前工件调度的最早阶段的机器上
        influence_jobs = {}
        for stage in range(self.stage_index, stages_num):
            influence_jobs[stage] = []

        for index, job in enumerate(self.schedule[(self.stage_index, machine)]):
            # 当前阶段开始，
            if job == cur_job:  # 如果是当前阶段的第一个工件
                pre_job = cur_job
                overlap_time = copy.copy(0 - self.job_execute_time[(self.stage_index, cur_job)][0])
                # 　当前这个工件就要开始往后推迟了
                self.job_execute_time[(self.stage_index, cur_job)][0] += overlap_time
                self.job_execute_time[(self.stage_index, cur_job)][1] += overlap_time
                influence_jobs[self.stage_index].append(job)
            else:
                overlap_time = copy.copy(self.job_execute_time[(self.stage_index, job)][0] - self.job_execute_time[(self.stage_index, pre_job)][
                    1] - setup_time[self.stage_index][pre_job][job])
                # 　如果当前这个工件需要推迟【idle_time是大于需要推迟的时间】
                if overlap_time < 0:
                    need_move_time = -overlap_time
                    self.job_execute_time[(self.stage_index, job)][0] += need_move_time
                    self.job_execute_time[(self.stage_index, job)][1] += need_move_time
                    influence_jobs[self.stage_index].append(job)
                pre_job = job

        if self.stage_index != stages_num-1:
            for stage in range(self.stage_index+1, stages_num):
                # print(1,stage)
                for influ_job in influence_jobs[stage-1]:
                    for machine in range(machine_num_on_stage[stage]):
                        if influ_job in self.schedule[(stage,machine)]:
                            influ_job_index = self.schedule[(stage,machine)].index(influ_job)
                            # print(2,stage)
                            for index,job in enumerate(self.schedule[(stage,machine)][influ_job_index:]):
                                # 遍历该阶段该job之后的所有工件，如果刚好是该job，直接用上一阶段的工件结束时间和下一阶段的开始时间比较
                                if job == influ_job:
                                    pre_job = influ_job
                                    # print(3,stage)
                                    overlap_time = copy.copy(self.job_execute_time[(stage, job)][0] - self.job_execute_time[(stage-1, job)][1])
                                    # 　如果当前这个工件需要推迟【idle_time是大于需要推迟的时间】
                                    if overlap_time < 0:
                                        need_move_time = -overlap_time
                                        self.job_execute_time[(stage, job)][0] += need_move_time
                                        self.job_execute_time[(stage, job)][1] += need_move_time

                                        if job in influence_jobs[stage]:
                                            continue
                                        influence_jobs[stage].append(job)
                                # 如果不是当前的阶段，那么就只需要查看该机器上一个工件和这个工件是否有重叠，有重叠再推迟
                                else:
                                    overlap_time = copy.copy(self.job_execute_time[(stage, job)][0] - \
                                                   self.job_execute_time[(stage, pre_job)][
                                                       1] - setup_time[stage][pre_job][job])
                                    # 　如果当前这个工件需要推迟【idle_time是大于需要推迟的时间】
                                    if overlap_time < 0:
                                        need_move_time = -overlap_time
                                        self.job_execute_time[(stage, job)][0] += need_move_time
                                        self.job_execute_time[(stage, job)][1] += need_move_time
                                        if job in influence_jobs[stage]:
                                            continue
                                        influence_jobs[stage].append(job)
                                    pre_job = job

        # print(1)
        for stage in range(self.stage_index,stages_num-1):
            for job in self.schedule[(stage,machine)]:
                job_index = self.schedule[(stage,machine)].index(job)
                if self.job_execute_time[(stage,job)][0] >= self.job_execute_time[(stage,job)][1]:
                    print(True)
                if job_index + 1 < len(self.schedule[(stage,machine)]):
                    later_job = self.schedule[(stage,machine)][job_index + 1]
                    if self.job_execute_time[(stage,job)][1] + setup_time[stage][job][later_job] > self.job_execute_time[(stage,later_job)][0]:
                        print(True)
                if stage+1 < stages_num:
                    if self.job_execute_time[(stage,job)][1] > self.job_execute_time[(stage+1,job)][0]:
                        print(True)


    def get_reward(self):
        # 遍历最后一个阶段的完工时间：
        # 判断工件的完工时间是否在交付期窗口内
        penalty_value = 0
        if self.stage_index == stages_num - 1:
            sorted_jobs = []
            for machine in range(machine_num_on_stage[self.stage_index]):
                sorted_jobs.extend(self.schedule[(self.stage_index, machine)])
        else:
            sorted_jobs = jobs
        for job in sorted_jobs:
            if self.job_execute_time[(stages_num-1,int(job))][1] == 0:
                continue
            if self.job_execute_time[(stages_num-1,int(job))][1] < ect_windows[int(job)]:
                penalty_value += (ect_windows[job] - self.job_execute_time[(stages_num-1,int(job))][1])* ect_weight[job]
            elif self.job_execute_time[(stages_num-1,int(job))][1] > ddl_windows[job]:
                penalty_value += (self.job_execute_time[(stages_num-1,int(job))][1] - ddl_windows[job]) * ddl_weight[job]
            else:
                penalty_value += 0

        return penalty_value


    def step(self,action):
        # 执行动作，更新三个变量
        self.done = False
        ori_penalty_value = self.get_reward()
        # print('ori_penalty_value:{0}'.format(ori_penalty_value))
        self.schedule, self.job_execute_time = self.reverse_job_assignment(self.stage_index,self.jobs_sort[self.job_index],action)
        job = self.jobs_sort[self.job_index]
        # for j in jobs:
        #     if self.job_execute_time[(3,int(j))][0] < 0:
        #         # print(True)
        if self.job_execute_time[self.stage_index,job][0] < 0 :
            self.update(job,action)
        cur_penalty_value = self.get_reward()
        # print('cur_penalty_value:{0}'.format(cur_penalty_value))

        self.reward = ori_penalty_value - cur_penalty_value
        if self.stage_index == 0:
            if self.job_index == 0:
                self.done = True
                print(cur_penalty_value)
                print(self.schedule)
            else:
                self.job_index -= 1
        else:
            if self.job_index == 0:
                # 工件的due_date 和最早可开始加工时间
                remanin_process_time = np.zeros(jobs_num)
                renmain_setup_time = np.zeros(jobs_num)
                job_state = np.zeros(jobs_num)
                for job in self.jobs_sort:
                    for stage in range(self.stage_index-1, -1, -1):
                        remanin_process_time[job] += job_process_time[stage][job]  # 剩余的加工时间的加总
                        renmain_setup_time[job] += sum(setup_time[stage][job])  # 剩余的平均切换时间
                        # ect_start_time [index] = max(self.job_completion_time[stage-1][job] , min_ect_machine_statrt_time)

                renmain_setup_time = renmain_setup_time / jobs_num
                for job in self.jobs_sort:
                # 下一个阶段该工件的开始加工时间 - 剩余待加工的时间 - 剩余待切换的平均值
                    job_state[job] = (self.job_execute_time[(self.stage_index, job)][0] - remanin_process_time[job] - renmain_setup_time[job]) / ddl_weight[job]

                self.jobs_sort = np.argsort(job_state)
                self.stage_index -=1
                self.job_index = jobs_num-1
            else:
                self.job_index -= 1

        next_state = self.get_state()

        return next_state, self.reward, self.done


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


def plot_cost(losst,episode_rewardt,C):  # 展示学习曲线

    plt.ion()
    plt.subplot(3,1,1)
    plt.plot(np.arange(C), losst)
    plt.ylabel('losst')
    plt.xlabel('episode')

    plt.subplot(3,1,2)
    plt.plot(np.arange(C), episode_rewardt)
    plt.ylabel('makespan')
    plt.xlabel('episode')

    plt.show()
    plt.pause(0.01)


GAMMA = 0.99
BATH = 256  # 批量训练256
EXPLORE = 2000000
REPLAY_MEMORY = 50000  # 经验池容量5W
BEGIN_LEARN_SIZE = 1024
memory = Memory(REPLAY_MEMORY)
UPDATA_TAGESTEP = 200  # 目标网络的更新频次
learn_step = 0
epsilon = 0.2
writer = SummaryWriter('logs/dueling_DQN2')
FINAL_EPSILON = 0.00001
epsilon = 1
min_epsilon = 0.05

env = Envior(job_execute_time,machine_first_job_execute_time,schedule,job_sort_osl)

# 自定义的！！重要！！！
# n_state = 15
# stage_sum = 4      # !!!!!!!!!!!!!!阶段数
# n_action = 3
n_state = 15
stage_sum = 2      # !!!!!!!!!!!!!!阶段数
n_action = 2


target_network = Dueling_DQN(n_state, n_action)
network = Dueling_DQN(n_state, n_action)
target_network.load_state_dict(network.state_dict())
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
r = 0
c = 0
LOSST = []
REWARDT = []
for epoch in count():
    state = env.reset()
    episode_reward = 0
    c += 1
    while True:
        # env.render()

        p = random.random()
        # 动作选择
        epsilon = max(epsilon - c*0.002, min_epsilon)
        if p < epsilon:
            action = random.randint(0, n_action - 1)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
            action = network.select_action(state_tensor)

        # 根据（状态，动作）得到step序列
        next_state, reward, done = env.step(action)
        # print(reward)
        # print(next_state,reward,done)
        episode_reward += reward

        # 将得到序列添加到经验池
        memory.add((state, next_state, action, reward, done))

        # 只有当经验池中的样本数量大于批量训练的数量，才会执行训练
        if memory.size() > BEGIN_LEARN_SIZE:
            learn_step += 1

            # 每隔一段时间，更新目标网络
            if learn_step % UPDATA_TAGESTEP:
                target_network.load_state_dict(network.state_dict())

            # 训练的时候，从经验池中进行采样
            batch = memory.sample(BATH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.as_tensor([item.cpu().detach().numpy() for item in batch_state], dtype=torch.float)
            batch_next_state = torch.as_tensor([item.cpu().detach().numpy() for item in batch_next_state], dtype=torch.float)
            batch_action = torch.as_tensor(batch_action, dtype=torch.long).unsqueeze(1)
            batch_reward = torch.as_tensor(batch_reward, dtype=torch.float).unsqueeze(1)
            batch_done = torch.as_tensor(batch_done, dtype=torch.long).unsqueeze(1)

            with torch.no_grad():
                target_Q_next = target_network(batch_next_state)  # 从目标网络中

                # 将next_state输入预测网络，并选择最优的状态动作价值函数
                Q_next = network(batch_next_state)
                Q_max_action = torch.argmax(Q_next, dim=1, keepdim=True)

                # 再将这个输入目标网络中
                y = batch_reward + target_Q_next.gather(1, Q_max_action)

            loss = F.mse_loss(network(batch_state).gather(1, batch_action), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_step)

            # if epsilon > FINAL_EPSILON: ## 减小探索
            #     epsilon -= (0.1 - FINAL_EPSILON) / EXPLORE
        if done:
            print('第{0}幕'.format(c))
            print('obj:{0}'.format(episode_reward))
            if memory.size() > BEGIN_LEARN_SIZE:
                LOSST.append(loss.item())
                print(loss.item())
                REWARDT.append(episode_reward)
                # plot_cost(LOSST,REWARDT,len(REWARDT))
                fig, ax = plt.subplots()
                ax.plot(range(len(REWARDT)), REWARDT, linewidth=3)
                ax.set_xlabel('epoch')
                ax.set_ylabel('testreward')
                plt.show()
            break
        state = next_state
    r += episode_reward
    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 100 == 0:
        print(f"第{epoch / 100}个100epoch的reward为{r / 100}", epsilon)
        r = 0
    # if epoch % 10 == 0:
    #     #     torch.save(network.state_dict(), 'modelnetwark{}.pt'.format("dueling"))