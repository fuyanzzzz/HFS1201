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
job_process_time = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=2, max_rows=10,
                              usecols=(1,3), unpack=True, dtype=int))
ect_delay_wight = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=14, max_rows=10,
                             usecols=(2, 3), unpack=True, dtype=int))
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = torch.tensor((np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=25, max_rows=10,
                              dtype=int)))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

setup_time = {}
setup_time[0] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=37, max_rows=10,
                           dtype=int))
setup_time[1] = torch.tensor(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=48, max_rows=10,
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
    def __init__(self,job_completion_time,machine_completion_time,schedule,job_process_time,job_sort):
        self.job_completion_time = job_completion_time
        self.machine_completion_time = machine_completion_time
        self.schedule = schedule

        self.job_process_time = job_process_time
        self.reward = 0
        self.job_index = jobs_num-1
        self.stage_index = stages_num-1
        self.job_sort_first_stage = job_sort
        self.job_value = [0]*jobs_num

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
        self.job_sort_first_stage = []

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
                            self.job_completion_time[stage][job] = self.job_process_time[stage][job]
                        else:  # 如果是工件不是第一个工件，则直接用上一个工件的完工时间 +　切换时间 + 工件的加工时间
                            self.job_completion_time[stage][job] = self.job_completion_time[stage][pre_job] + setup_time[stage][pre_job][job] + self.job_process_time[stage][job]
                            pre_job = job
                    else:   # 第二到n个阶段
                        # 如果该工件是机器上的第一个工件，工件的完工时间 = 该工件在上一个阶段的完工时间 + 工件的加工时间
                        if schedule[(stage,machine)].index(job) == 0:
                            pre_job = job
                            self.job_completion_time[stage][job] = self.job_completion_time[stage-1][job] + self.job_process_time[stage][job]
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


    def job_assignment(self):
        '''
        传入的参数：6个
        1. 第一阶段的工件加工顺序
        2. 开始调度阶段，结束调度阶段
        3. 所有阶段上的工件加工顺序

        传出的参数：4个
        1. 三个更新的变量
        2. 当前的目标值【未进行空闲插入程序】
        '''
        terminate = True
        stage = 0
        job_sort = self.job_sort_first_stage
        while terminate:
            for job in job_sort:
                job_on_machine = []
                for machine in range(machine_num_on_stage[stage]):
                    if stage == 0:
                        pro_job = self.machine_completion_time[(stage,machine)][0]
                        # 如果是第一个阶段的第一个工件
                        if pro_job == -1:
                            # ect = 该机器上最后一个工件的完工时间 +　切换时间　　【第一阶段的第一个工件：０＋０】
                            ect_value = self.machine_completion_time[(stage,machine)][1] + 0

                        else:   # 第一阶段，非第一个工件　　ect = 该机器上最后一个工件的完工时间 +　切换时间
                            ect_value = self.machine_completion_time[(stage,machine)][1]

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
                            job_on_pro_machine = self.job_completion_time[stage-1][job]
                            ect_value = max(job_on_pro_machine,self.machine_completion_time[(stage,machine)][1])  # 机器最早可开始加工时间 = 上个工件的完工时间 + 切换时间
                    job_on_machine.append(ect_value)

                chosen_machine = job_on_machine.index(min(job_on_machine))
                self.job_completion_time[stage][job] = max(0,ect_value) + self.job_process_time[stage][job]
                self.schedule[(stage,chosen_machine)].append(job)

                self.machine_completion_time[(stage,chosen_machine)][0] = job
                self.machine_completion_time[(stage,chosen_machine)][1] = max(0,ect_value) + self.job_process_time[stage][job]

            if stage == 1:
                terminate = False
            else:
                job_sort = np.argsort(self.job_completion_time[stage])
                stage += 1

        # 计算目标值
        obj = 0
        for job in range(jobs_num):
            if self.job_completion_time[stage][job] > ddl_windows[job]:
                obj += (self.job_completion_time[stage][job] - ddl_windows[job]) * ddl_weight[job]
            elif self.job_completion_time[stage][job] < ect_windows[job]:
                obj += (ect_windows[job] - self.job_completion_time[stage][job]) * ect_weight[job]
            else:
                obj += 0

        return obj



    def idle_time_insertion(self,obj):
        #　外部循环，遍历最后一个阶段上的所有机器
        all_job_block = []
        stage = stages_num - 1
        for machine in range(machine_num_on_stage[-1]):
            # 内部循环，从机器上最后一个工件开始往前遍历
            job_block = []      # 声明一个空的工件块
            all_job_block = []
            delay_job = []          # 延误的工件
            early_job = []          # 早到的工件
            on_time_job = []        # 按时生产的工件

            job_list_machine = self.schedule[int((stages_num-1)),machine].copy()        # 确定该机器上的所有工件
            job_num_machine = len(job_list_machine) # 判断该机器上有几个工件       #   确定该机器上的工件数量
            while job_num_machine > 0:
                job = job_list_machine[job_num_machine-1]           # 从最后一个工件开始往后推迟
                if job_num_machine == len(job_list_machine):        # 如果是倒数第一个工件
                    later_job = None
                else:
                    later_job = job_list_machine[job_num_machine]

                # 判断是否该工件和后面的工件块合并在一起，导致重新运算

                # 判断这个工件和下一个工件有没有并在一块
                # 如果工件有并在有一块，那么直接插入这个工件块中
                # if job_makespan[job] == (job_makespan[later_job] - setup_time[stages_num][job][later_job] - job_process_time[stages_num-1][later_job]):
                # 如果是最后一个工件,或者第一次查看的时候，这个工件就是紧挨着下一个工件的
                if (job == job_list_machine[-1] or (self.job_completion_time[stage][job] == (self.job_completion_time[stage][later_job] - self.job_process_time[stages_num-1][later_job]))) and job not in job_block:
                    job_block.insert(0,job)           # 构建工件块

                elif job not in job_block:   # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
                    all_job_block.insert(0,job_block)
                    job_block = []      # 再声明一个新的工件块
                    job_block.insert(0,job)     # 重新插入新的工件块中


                job_before_idle = job_block[-1]
                if len(all_job_block) != 0:        # 如果当前工件块右侧存在工件
                    job_after_idle = all_job_block[0][0]
                    job_completion_time = self.job_completion_time[stage][job_before_idle]  # 当前工件块最后一个工件的完工时间
                    later_block_start_time = self.job_completion_time[stage][job_after_idle] - self.job_process_time[stages_num - 1][job_after_idle]
                    idle_2 = later_block_start_time - job_completion_time
                else:
                    idle_2 = 999999         # 如果右边没有工件了，赋值无穷大

                # 根据当前工件块生成三个子集
                early_job.clear()
                delay_job.clear()
                on_time_job.clear()
                for job in job_block:
                    if self.job_completion_time[stage][job] < due_date_windows[job][0]:
                        early_job.append(job)
                    elif self.job_completion_time[stage][job] >= due_date_windows[job][1]:
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
                        early.append(ect_windows[job] - self.job_completion_time[stage][job])               # !!!!!!!!这个变量很重要，要实时更新

                    # 计算超过准时的最小延误的空闲时间
                    for job in on_time_job:
                        # delay.append(job_makespan[job] - ddl_windows[job])                # !!!!!!!!这个变量很重要，要实时更新
                        delay.append(ddl_windows[job] - self.job_completion_time[stage][job])
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
                        self.job_completion_time[stage][job] += insert_idle_time
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
        for job in range(jobs_num):
            if self.job_completion_time[stage][job] > ddl_windows[job]:
                self.job_value[job] = (self.job_completion_time[stage][job] - ddl_windows[job]) * ddl_weight[job]
            elif self.job_completion_time[stage][job] < ect_windows[job]:
                self.job_value[job] = (self.job_completion_time[stage][job] - ect_windows[job]) * ect_weight[job]
            else:
                self.job_value[job] = 0


        return obj,all_job_block


    def job_operation(self,operation):
        # 获取第一阶段的工件顺序
        avail_loaction = []
        if operation[0] == 'delay':
            job = self.job_value.index(max(self.job_value))
            for i_job in self.job_sort_first_stage:
                if self.job_completion_time[stage][i_job] < self.job_completion_time[stage][job]:
                    avail_loaction.append((i_job,False))

        elif operation[0] == 'early':
            job = self.job_value.index(min(self.job_value))
            for i_job in self.job_sort_first_stage:
                if self.job_completion_time[stage][i_job] > self.job_completion_time[stage][job]:
                    if self.job_sort_first_stage.index(job) == jobs_num - 1:
                        avail_loaction.append((i_job, False))
                        avail_loaction.append((i_job, True))
                    else:
                        avail_loaction.append((i_job,False))
        else:
            job = random.choice(range(jobs_num))
            for i_job in self.job_sort_first_stage:
                if job == i_job:
                    continue
                if job == self.job_sort_first_stage[-1]:
                    avail_loaction.append((job,False))
                    avail_loaction.append((job,True))
                else:
                    avail_loaction.append((job,False))


        return job,operation,avail_loaction
        # 以第一阶段的工件为单位往前插：

    def do_operation_job(self,job,avail_loaction,operation):
        if avail_loaction:
            oper = random.choice(avail_loaction)
            # 移除工件
            self.job_sort_first_stage.remove(job)
            # 获取要插入的位置
            if oper[1]:
                self.job_sort_first_stage.append(job)
            else:
                index = self.job_sort_first_stage.index(oper[0])
                self.job_sort_first_stage.insert(index, job)

            if operation[1]:        # 进行swap操作
                self.job_sort_first_stage.remove(oper[0])
                index = self.job_sort_first_stage.index(job)
                self.job_sort_first_stage.insert(index, job)





    def final_stage_operation(self, operation):

        avail_loaction = []
        job = None
        if operation == 'early':
            job = self.job_value.index(min(self.job_value))
            for machine in range(machine_num_on_stage[stages_num-1]):
                for i_job in self.schedule[(stage, machine)]:
                    if self.job_completion_time[stage][i_job] > self.job_completion_time[stage][job]:
                        avail_loaction.append(((machine, i_job),False))

                    elif len(self.schedule[(stage,machine)])-1 == i_job:
                        avail_loaction.append(((machine, i_job),False))
                        avail_loaction.append(((machine, i_job),True))


        elif operation != 'delay':      # 就是随机操作
            # 随机选择一个机器，随机选择机器的一个位置
            job = random.choice(range(jobs_num))
            for machine in range(machine_num_on_stage[stages_num - 1]):
                for i_job in self.schedule[(stage, machine)]:
                    if job == i_job:
                        continue
                    if len(self.schedule[(stage, machine)]) - 1 == i_job:
                        avail_loaction.append(((machine, i_job), False))
                        avail_loaction.append(((machine, i_job), True))
                    else:
                        avail_loaction.append(((machine, i_job), False))

        return job,avail_loaction,operation


    def do_operation_final_stage(self,job,avail_loaction,operation):

        if avail_loaction:
            if operation == 'early':
                for oper in avail_loaction:
                    # 移除工件
                    for machine in range(machine_num_on_stage[stages_num-1]):
                        if job in self.schedule[(stages_num-1,machine)]:
                            self.schedule[(stages_num-1,machine)].remove(job)

                    for machine in range(machine_num_on_stage[stages_num - 1]):
                        if avail_loaction[0] in self.schedule[(stages_num-1,machine)]:
                            index = self.schedule.index(avail_loaction[0])
                            if avail_loaction[1]:
                                self.schedule[(stages_num-1,machine)].append(avail_loaction[0])
                            else:
                                self.schedule[(stages_num-1,machine)].insert(index,avail_loaction[0])


    def two_steps_operation(self, operation):

        avail_loaction = []
        if operation == 'delay':        # 最后一阶段的delay没法调
            job = self.job_value.index(max(self.job_value))
            for stage in range(stages_num):
                for machine in range(machine_num_on_stage[stage]):
                    for i_job in self.schedule[(stage, machine)]:
                        if self.job_completion_time[stage][i_job] < self.job_completion_time[stage][job]:
                            avail_loaction.append((self.schedule[(stage, machine)],False))
                        else:
                            break
            for

        elif operation == 'early':
            job = self.job_value.index(min(self.job_value))
            for machine in range(machine_num_on_stage[stages_num-1]):
                for i_job in self.schedule[(stage, machine)]:
                    if self.job_completion_time[stage][i_job] > self.job_completion_time[stage][job]:
                        avail_loaction.append(((machine, i_job),False))

                    elif len(self.schedule[(stage,machine)])-1 == i_job:
                        avail_loaction.append(((machine, i_job),False))
                        avail_loaction.append(((machine, i_job),True))
                        break
        else:
            # 随机选择一个机器，随机选择机器的一个位置
            job = self.job_value.index(min(self.job_value))
            for machine in range(machine_num_on_stage[stages_num - 1]):
                for i_job in self.schedule[(stage, machine)]:
                    if len(self.schedule[(stage, machine)]) - 1 == i_job:
                        avail_loaction.append(((machine, i_job), False))
                        avail_loaction.append(((machine, i_job), True))
                    elif len(self.schedule[(stage, machine)]) - 1 == i_job:
                        avail_loaction.append(((machine, i_job), False))

        return avail_loaction,operation




job_sort = np.argsort(ddl_windows)
env = Envior(job_completion_time,machine_completion_time,schedule,job_process_time,job_sort)
# 确认工件的分配顺序+

# 工件分配【生成初始解】
obj = env.job_assignment()
# 进行搜索
obj,all_job_block = env.idle_time_insertion(obj)
print(1)
# 将工件的完工顺序
