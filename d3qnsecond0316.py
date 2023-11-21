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

# 　读取数据
i = 0
job_process_time = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=2, max_rows=20,
                              usecols=(1, 3,5,7), unpack=True, dtype=int))
ect_delay_wight = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=24, max_rows=20,
                             usecols=(2, 3), unpack=True, dtype=int))
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = torch.tensor((np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=45, max_rows=20,
                              dtype=int)))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

setup_time = {}
setup_time[0] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=67, max_rows=20,
                           dtype=int))
setup_time[1] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=88, max_rows=20,
                           dtype=int))
setup_time[2] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=109, max_rows=20,
                           dtype=int))
setup_time[3] = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=130, max_rows=20,
                           dtype=int))

jobs_num = torch.tensor(int(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1, usecols=(0),
                          dtype=int)))
jobs = torch.tensor(list(range(jobs_num)))

stages_num = (np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1, usecols=(2),
                        dtype=int))
total_mahcine_num = torch.tensor(np.loadtxt("1619_Instance_20_4_3_0,6_1_20_Rep4.txt", skiprows=1, max_rows=1,
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


def get_reset():
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

    return schedule,machine_completion_time,job_completion_time

# 声明一个变量，用于存储每个阶段的工件调度顺序【用于第二阶段的local search】
job_sort_on_satges = {}

def cal(schedule):
    job_completion_time = np.zeros((stages_num, jobs_num), dtype=int)

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
                        job_completion_time[stage][job] = job_process_time[stage][job]
                    else:  # 如果是工件不是第一个工件，则直接用上一个工件的完工时间 +　切换时间 + 工件的加工时间
                        job_completion_time[stage][job] = job_completion_time[stage][pre_job] + setup_time[stage][pre_job][job] +job_process_time[stage][job]
                        pre_job = job
                else:   # 第二到n个阶段
                    # 如果该工件是机器上的第一个工件，工件的完工时间 = 该工件在上一个阶段的完工时间 + 工件的加工时间
                    if schedule[(stage,machine)].index(job) == 0:
                        pre_job = job
                        job_completion_time[stage][job] = job_completion_time[stage-1][job] + job_process_time[stage][job]
                    else: # 如果工件不是第一个工件，则完工时间 = max(上一个阶段该工件的完工时间, 该机器上上一个工件的完工时间) + 切换时间 + 工件的加工时间
                        job_completion_time[stage][job] = max(job_completion_time[stage][pre_job],job_completion_time[stage-1][job]) + setup_time[stage][pre_job][job] + job_process_time[stage][job]
                        pre_job = job

            # machine_completion_time[(stage,machine)][0] = job


    job_makespan = job_completion_time[stages_num - 1]
    ect_value = 0
    ddl_value = 0
    for i in range(len(job_makespan)):
        if job_makespan[i] < ect_windows[i]:  # 早前权重值
            ect_value += max(ect_windows[i] - job_makespan[i], 0) * ect_weight[i]
        elif job_makespan[i] > ddl_windows[i]:  # 延误权重值
            ddl_value += max(job_makespan[i] - ddl_windows[i], 0) * ddl_weight[i]

    obj = ect_value + ddl_value

    return obj,job_completion_time,machine_completion_time


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




def idle_time_insertion(schedule):
    #　外部循环，遍历最后一个阶段上的所有机器
    obj,job_completion_time,machine_completion_time = cal(schedule)
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



process_time_all_stages = []
for i in range(jobs_num):
    process_time_all_stages.append(sum([j[i] for j in job_process_time]))
b = np.array(ddl_windows) - np.array(process_time_all_stages)
job_sort_osl = np.argsort(b)  # 根据ddl或者工件的调度顺序

class Envior():
    def __init__(self,job_completion_time,machine_completion_time,schedule,job_sort_osl):
        self.job_completion_time = job_completion_time
        self.machine_completion_time = machine_completion_time
        self.schedule = schedule
        self.reward = 0
        self.jobs_sort = job_sort_osl
        self.job_index = 0
        self.stage_index = 0


    def get_state(self):
        self.reward = 0
        n_state = torch.zeros(15)
        remanin_process_time = torch.zeros(10)
        renmain_setup_time = torch.zeros(10)
        ect_start_time = torch.zeros(10)
        ddl_ten = torch.zeros(10)
        ect_ten = torch.zeros(10)
        ect_machine_statrt_time = []
        for machine in range(machine_num_on_stage[self.stage_index]):
            ect_machine_statrt_time.append(self.machine_completion_time[(self.stage_index,machine)][1])
        min_ect_machine_statrt_time = min(ect_machine_statrt_time)
        for index, job in enumerate(self.jobs_sort[self.job_index:min(jobs_num-1,self.job_index+10)]):
            ddl_ten[index] = ddl_windows[job]
            ect_ten[index] = ect_windows[job]
            for stage in range(self.stage_index, stages_num):
                remanin_process_time[index] += job_process_time[stage][job]
                renmain_setup_time[index] += sum(setup_time[stage][job])
                ect_start_time [index] = max(self.job_completion_time[stage-1][job] , min_ect_machine_statrt_time)

        renmain_setup_time = renmain_setup_time / jobs_num
        a = ddl_ten - remanin_process_time - renmain_setup_time - ect_start_time
        b = ect_ten - remanin_process_time - renmain_setup_time - ect_start_time

        for index, job in enumerate(self.jobs_sort[self.job_index:min(jobs_num-1,self.job_index+10)]):
            if a[index] < 0:
                n_state[index] = a[index] * ddl_weight[job]
                self.reward += -a[index] * ddl_weight[job] * -2
            elif b[index] > 0:
                n_state[index] = b[index] * ect_weight[job]
                self.reward += b[index] * ddl_weight[job] * -1
            else:
                n_state[index] = 0
                self.reward += 0

        for machine in range(machine_num_on_stage[self.stage_index]):
            if self.stage_index == 0:
                n_state[10 + machine] = self.machine_completion_time[(self.stage_index, machine)][1]
            else:
                n_state[10 + machine] = self.job_completion_time[self.stage_index - 1][self.jobs_sort[self.job_index]] - \
                                  self.machine_completion_time[(self.stage_index, machine)][1]

        n_state[13] = self.job_index / (jobs_num - 1)
        n_state[14] = self.stage_index / (stages_num - 1)
        self.state = n_state

        return self.state



    def reset(self):
        self.state = self.get_state()
        self.schedule, self.machine_completion_time, self.job_completion_time = get_reset()
        self.jobs_sort = job_sort_osl
        self.job_index = 0
        self.stage_index = 0
        return self.state


    def step(self,action):
        # 执行动作，更新三个变量
        self.done = False
        # 多次连续action == 3,会陷入死循环
        # if action == 3:
        #     if self.job_index < jobs_num-1:
        #         self.jobs_sort[self.job_index], self.jobs_sort[self.job_index + 1] = self.jobs_sort[self.job_index + 1], self.jobs_sort[self.job_index]
        # else:
            # self.schedule[(self.stage_index, action)].append(self.jobs_sort[self.job_index])
            # self.job_completion_time,self.machine_completion_time = cal(self.schedule)

        self.schedule, self.job_completion_time, self.machine_completion_time = job_assignment(self.stage_index,self.jobs_sort[self.job_index],action,self.schedule,self.job_completion_time,self.machine_completion_time)

        if self.stage_index < stages_num-1:
            if self.job_index == jobs_num-1:
                self.jobs_sort = np.argsort(self.job_completion_time[self.stage_index])
                self.stage_index += 1
                self.job_index = 0
            else:
                self.job_index += 1
        else:
            if self.job_index == jobs_num-1:
                self.done = True
                self.reward, block = idle_time_insertion(self.schedule)
                print(self.reward)
                print(self.schedule, self.reward)
            else:
                self.job_index += 1
        # if self.stage_index == stages_num-1 and self.job_index == jobs_num-1:
        #     self.done = True
        #     self.reward, block = idle_time_insertion(self.schedule)
        #     print(self.schedule,self.reward)
        # if self.job_index == jobs_num -1 and self.stgae_index < stages_num-1:
        #     self.jobs_sort = np.argsort(self.job_completion_time[self.stage_index])
        #     self.stage_index += 1
        #     self.job_index = 0
        #
        # if self.job_index < jobs_num -1:
        #     self.job_index += 1

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
min_epsilon = 0.2

env = Envior(job_completion_time,machine_completion_time,schedule,job_sort_osl)

# 自定义的！！重要！！！
n_state = 15
stage_sum = 4      # !!!!!!!!!!!!!!阶段数
n_action = 3


target_network = Dueling_DQN(n_state, n_action)
network = Dueling_DQN(n_state, n_action)
target_network.load_state_dict(network.state_dict())
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
r = 0
c = 0
for epoch in count():
    state = env.reset()
    episode_reward = 0
    c += 1
    while True:
        # env.render()

        p = random.random()
        # 动作选择
        epsilon = max(epsilon - c*0.01, min_epsilon)
        if p < epsilon:
            action = random.randint(0, n_action - 1)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
            action = network.select_action(state_tensor)

        # 根据（状态，动作）得到step序列
        next_state, reward, done = env.step(action)
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

            break
        state = next_state
    r += episode_reward
    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 100 == 0:
        print(f"第{epoch / 100}个100epoch的reward为{r / 100}", epsilon)
        r = 0
    # if epoch % 10 == 0:
    #     #     torch.save(network.state_dict(), 'modelnetwark{}.pt'.format("dueling"))