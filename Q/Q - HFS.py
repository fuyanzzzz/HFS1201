'''
-------------------------------------------------
File Name: Q - HFS.py
Author: LRS
Create Time: 2023/4/11 10:47
-------------------------------------------------
'''
import time

import numpy as np
import pandas as pd


import time

import torch
import numpy as np
import random
import copy

# 　读取数据
i = 0
job_process_time = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=2, max_rows=20,
                              usecols=(1, 3,5,7), unpack=True, dtype=int))
ect_delay_wight = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=24, max_rows=20,
                             usecols=(2, 3), unpack=True, dtype=int))
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = torch.tensor((np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=45, max_rows=20,
                              dtype=int)))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

setup_time = {}
setup_time[0] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=67, max_rows=20,
                           dtype=int))
setup_time[1] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=88, max_rows=20,
                           dtype=int))
setup_time[2] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=109, max_rows=20,
                           dtype=int))
setup_time[3] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=130, max_rows=20,
                           dtype=int))

jobs_num = torch.tensor(int(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(0),
                          dtype=int)))
jobs = torch.tensor(list(range(jobs_num)))

stages_num = (np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(2),
                        dtype=int))
total_mahcine_num = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1,
                               usecols=(1), dtype=int))

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


# job_completion_time = np.zeros((stages_num, jobs_num), dtype=int)
class Envior():
    def __init__(self,job_completion_time,machine_completion_time,schedule,job_sort_osl):
        self.job_completion_time = job_completion_time
        self.machine_completion_time = machine_completion_time
        self.schedule = schedule

        self.reward = 0
        self.jobs_sort = job_sort_osl
        self.job_index = jobs_num-1
        self.stage_index = stages_num-1

    def get_reset(self):
        # 　声明schedule的字典变量
        schedule = {}
        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):
                schedule[(stage, machine)] = []

        # 声明一个空的每个工件的完工时间
        self.machine_completion_time = {}
        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):
                self.machine_completion_time[(stage, machine)] = [-1, -1]

        # 声明一个阶段的完工时间的字典
        self.job_completion_time = {}
        for stage in range(stages_num):
            self.job_completion_time[stage] = np.zeros(jobs_num, dtype=int)
        # job_completion_time = np.zeros((stages_num, jobs_num), dtype=int)

    # 声明一个变量，用于存储每个阶段的工件调度顺序【用于第二阶段的local search】
    job_sort_on_satges = {}

    def cal(self):

        # 考虑每个工件的最早可开始加工时间： max(当前机器上的完工时间，工件在上一个阶段的完工时间)
        #　完工时间 = 最早可开始加工时间 + 切换时间 + 加工时间
        #　从第一个阶段开始，逐阶段进行加工
        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):  # 从第一个阶段开始
                pre_job = None
                # for job in schedule[machine]:
                for job in schedule[(stage,machine)]:
                    # 如果是第一个阶段
                    # if job == -1:
                    #     continue
                    if stage == 0:
                        # 机器上有工件，且是第一个工件，该阶段工件的完工时间 = 工件的加工时间
                        if schedule[(stage,machine)].index(job) == 0:
                            pre_job = job
                            self.job_completion_time[stage][job] = job_process_time[stage][job]
                        else:  # 如果是工件不是第一个工件，则直接用上一个工件的完工时间 +　切换时间 + 工件的加工时间
                            self.job_completion_time[stage][job] = self.job_completion_time[stage][pre_job] + setup_time[stage][pre_job][job] +job_process_time[stage][job]
                            pre_job = job
                    else:   # 第二到n个阶段
                        # 如果该工件是机器上的第一个工件，工件的完工时间 = 该工件在上一个阶段的完工时间 + 工件的加工时间
                        if schedule[(stage,machine)].index(job) == 0:
                            pre_job = job
                            self.job_completion_time[stage][job] = self.job_completion_time[stage-1][job] + job_process_time[stage][job]
                        else: # 如果工件不是第一个工件，则完工时间 = max(上一个阶段该工件的完工时间, 该机器上上一个工件的完工时间) + 切换时间 + 工件的加工时间
                            self.job_completion_time[stage][job] = max(self.job_completion_time[stage][pre_job],self.job_completion_time[stage-1][job]) + setup_time[stage][pre_job][job] + job_process_time[stage][job]
                            pre_job = job

                # machine_completion_time[(stage,machine)][0] = job


        job_makespan = self.job_completion_time[stages_num - 1]
        ect_value = 0
        ddl_value = 0
        for i in range(len(job_makespan)):
            if job_makespan[i] < ect_windows[i]:  # 早前权重值
                ect_value += max(ect_windows[i] - job_makespan[i], 0) * ect_weight[i]
            elif job_makespan[i] > ddl_windows[i]:  # 延误权重值
                ddl_value += max(job_makespan[i] - ddl_windows[i], 0) * ddl_weight[i]

        obj = ect_value + ddl_value


    def job_assignment(self,job_sort):
        '''
        传入的参数：6个
        1. 第一阶段的工件加工顺序
        2. 开始调度阶段，结束调度阶段
        3. 所有阶段上的工件加工顺序

        传出的参数：4个
        1. 三个更新的变量
        2. 当前的目标值【未进行空闲插入程序】
        '''
        terminate = False
        stage = 0
        while terminate:
            for job in job_sort:
                if stage == 0:
                    pro_job = self.machine_completion_time[(stage,machine)][0]
                    # 如果是第一个阶段的第一个工件
                    if pro_job == -1:
                        # ect = 该机器上最后一个工件的完工时间 +　切换时间　　【第一阶段的第一个工件：０＋０】
                        ect_value = self.machine_completion_time[(stage,machine)][1] + 0

                    else:   # 第一阶段，非第一个工件　　ect = 该机器上最后一个工件的完工时间 +　切换时间
                        ect_value = self.machine_completion_time[(stage,machine)][1] + setup_time[stage][pro_job][job]

                # 如果非第一阶段
                else:
                    pro_job = self.machine_completion_time[(stage,machine)][0]
                    # 如果非第一阶段，但是该机器上的第一个工件
                    if pro_job == -1:
                        # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b = 0】
                        job_on_pro_machine = self.job_completion_time[stage - 1][job]
                        ect_value = max(self.machine_completion_time[(stage,machine)][1],job_on_pro_machine) + 0

                    else:   # 非第一阶段，且非该机器上的第一个工件
                        # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b ！＝ 0】
                        job_on_pro_machine = job_completion_time[stage-1][job]
                        ect_value = max(job_on_pro_machine,self.machine_completion_time[(stage,machine)][1]) + setup_time[stage][pro_job][
                            job]  # 机器最早可开始加工时间 = 上个工件的完工时间 + 切换时间

                job_completion_time[stage][job] = max(0,ect_value) + job_process_time[stage][job]
                schedule[(stage,machine)].append(job)
                # schedule[machine, list(schedule[machine]).index(-1)] = job
                self.machine_completion_time[(stage,machine)][0] = job
                self.machine_completion_time[(stage,machine)][1] = max(0,ect_value) + job_process_time[stage][job]

            if stage == 1:
                terminate = True

            job_sort = np.argsort(self.job_completion_time[stage])
            stage += 1


    def idle_time_insertion(schedule):
        #　外部循环，遍历最后一个阶段上的所有机器
        obj,job_completion_time,machine_completion_time = self.cal(schedule)
        job_makespan = job_completion_time[stage]
        all_job_block = []
        for machine in range(machine_num_on_stage[-1]):
            # 内部循环，从机器上最后一个工件开始往前遍历
            job_block = []      # 声明一个空的工件块
            all_job_block = []
            delay_job = []
            early_job = []
            on_time_job = []

            job_list_machine = schedule[int((stages_num-1)),machine].copy()
            job_num_machine = len(job_list_machine) # 判断该机器上有几个工件
            while job_num_machine > 0:
                job = job_list_machine[job_num_machine-1]
                if job_num_machine == len(job_list_machine):        # 如果是倒数第一个工件
                    later_job = None
                else:
                    later_job = job_list_machine[job_num_machine]

                # 判断是否该工件和后面的工件块合并在一起，导致重新运算

                # 判断这个工件和下一个工件有没有并在一块
                # 如果工件有并在有一块，那么直接插入这个工件块中
                # if job_makespan[job] == (job_makespan[later_job] - setup_time[stages_num][job][later_job] - job_process_time[stages_num-1][later_job]):
                # 如果是最后一个工件,或者第一次查看的时候，这个工件就是紧挨着下一个工件的
                if (job == job_list_machine[-1] or (job_makespan[job] == (job_makespan[later_job] - setup_time[stages_num-1][job][later_job] - job_process_time[stages_num-1][later_job]))) and job not in job_block:
                    job_block.insert(0,job)           # 构建工件块

                elif job not in job_block:   # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
                    all_job_block.insert(0,job_block)
                    job_block = []      # 再声明一个新的工件块
                    job_block.insert(0,job)     # 重新插入新的工件块中


                job_before_idle = job_block[-1]
                if len(all_job_block) != 0:        # 如果当前工件块右侧存在工件
                    job_after_idle = all_job_block[0][0]
                    job_completion_time = job_makespan[job_before_idle]  # 当前工件块最后一个工件的完工时间
                    later_block_start_time = job_makespan[job_after_idle] - setup_time[stages_num - 1][job_before_idle][job_after_idle] - job_process_time[stages_num - 1][job_after_idle]
                    idle_2 = later_block_start_time - job_completion_time
                else:
                    idle_2 = 999999         # 如果右边没有工件了，赋值无穷大

                # 根据当前工件块生成三个子集
                early_job.clear()
                delay_job.clear()
                on_time_job.clear()
                for job in job_block:
                    if job_makespan[job] < due_date_windows[job][0]:
                        early_job.append(job)
                    elif job_makespan[job] >= due_date_windows[job][1]:
                        delay_job.append(job)
                    else:
                        on_time_job.append(job)

                early_job_weight = sum([ect_weight[job] for job in early_job])
                delay_job_weight = sum([ddl_weight[job] for job in delay_job])

                if early_job_weight > delay_job_weight:
                    early = []      # 计算距离准时早到的空闲时间
                    delay = []      # 计算超过准时的延误的空闲时间

                    # 计算距离准时的最小早到的空闲时间
                    for job in early_job:
                        early.append(ect_windows[job] - job_makespan[job])               # !!!!!!!!这个变量很重要，要实时更新

                    # 计算超过准时的最小延误的空闲时间
                    for job in on_time_job:
                        # delay.append(job_makespan[job] - ddl_windows[job])                # !!!!!!!!这个变量很重要，要实时更新
                        delay.append(ddl_windows[job] - job_makespan[job])
                    if len(early) == 0 and len(delay) != 0:
                        idle_1 = min(delay)
                    elif len(delay) == 0 and len(early) != 0:
                        idle_1 = min(early)
                    else:
                        idle_1 = min(min(early),min(delay))
                        print('断点')
                    print(idle_1)
                    print((idle_2))
                    print(delay)
                    print(early)
                    insert_idle_time = min(idle_1,idle_2)       # 确定需要插入的工件块

                    for job in job_block:
                        job_makespan[job] += insert_idle_time
                    improvement_obj = (early_job_weight - delay_job_weight)* insert_idle_time       # 获得改进的目标值
                    obj -= improvement_obj      # 重新计算目标值

                    # 判断插入工件块之后，是否会和后面的工件块进行合并
                    if insert_idle_time == idle_2:
                        job_block.extend(all_job_block[0])
                        all_job_block.remove(all_job_block[0])

                    # 更新工件块内的工件完工时间


                else:
                    job_num_machine -= 1    # 而且对于工件块有合并的选项

            all_job_block.insert(0,job_block)

        return obj,all_job_block



N_STATES = 12
ACTIONS = ["NS1", "NS2","NS3", "NS4","NS5", "NS6"]
EPSILON = 0.9
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 10000
FRESH_TIME = 0.3
TerminalFlag = "terminal"


def build_q_table(n_states, actions):
    return pd.DataFrame(
        np.zeros((n_states, len(actions))),
        columns=actions
    )


def choose_action(state, q_table):
    state_table = q_table.loc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_table == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = state_table.idxmax()
    return action_name


def get_env_feedback(S, A):
    if A == "right":
        if S == N_STATES - 2:
            S_, R = TerminalFlag, 1
        else:
            S_, R = S + 1, 0
    else:
        S_, R = max(0, S - 1), 0
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ["-"] * (N_STATES - 1) + ["T"]
    if S == TerminalFlag:
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print(interaction)
        time.sleep(2)
    else:
        env_list[S] = '0'
        interaction = ''.join(env_list)
        print(interaction)
        time.sleep(FRESH_TIME)

def judge_state(state):
    state_space = np.matrix([
        [0,3,6,9],
        [1,4,7,10],
        [2,5,8,11]])

    trial = state[0]
    improve = state[1]

    if trial == 0:
        index_i = 0
    elif trial <= 20:
        index_i = 1
    elif trial <= 50:
        index_i = 2
    else:
        index_i = 3

    if improve == 0:
        index_j = 0
    elif improve < 0:
        index_j = 1
    else:
        index_j =2

    S = state_space[index_j][index_i]


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0   # 初始状态
        is_terminated = False
        # update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_predict = q_table.loc[S, A]

            if S_ != TerminalFlag:
                q_target = R + GAMMA * q_table.loc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_predict)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print(q_table)
