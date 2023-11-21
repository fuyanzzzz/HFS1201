'''
-------------------------------------------------
File Name: ss_rl_0522.py
Author: LRS
Create Time: 2023/5/22 09:02
-------------------------------------------------
'''
import os
import numpy as np
import random
from diagram import job_diagram

path = r'C:\paper_code_0501\HFS1201\useful0424\data'
filename = '00_Instance_10_2_2_0,2_0,2_10_Rep0.txt'
job_process_time = np.loadtxt("data\{0}".format(filename), skiprows=2, max_rows=10,
                              usecols=(1, 3), unpack=True, dtype=int)
ect_delay_wight = np.loadtxt("data\{0}".format(filename), skiprows=14, max_rows=10,
                             usecols=(2, 3), unpack=True, dtype=int)
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = (np.loadtxt("data\{0}".format(filename), skiprows=25, max_rows=10,
                               dtype=int))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

jobs_num = int(np.loadtxt("data\{0}".format(filename), skiprows=1, max_rows=1, usecols=(0),
                          dtype=int))
jobs = list(range(jobs_num))

stages_num = int(np.loadtxt("data\{0}".format(filename), skiprows=1, max_rows=1, usecols=(2),
                            dtype=int))
total_mahcine_num = np.loadtxt("data\{0}".format(filename), skiprows=1, max_rows=1,
                               usecols=(1), dtype=int)
machine_num_on_stage = []
for job in range(stages_num):
    machine_num_on_stage.append(int(total_mahcine_num / stages_num))


# 优质解：
# 第一阶段 = 【EDD，OSL，EDD_weight，ECT，ECT_weight】
# 第二阶段 = 【EDD，OSL_2，EDD_weight】
# 多样解：
# 第一阶段 = 【随机生成】
# 第二阶段 = 【EDD，OSL_2，EDD_weight】


class HFS():
    def __init__(self,machine_num_on_stage,job_process_time,ect_weight,ddl_weight,ddl_windows,ect_windows,jobs_num):
        self.jobs_sort_method_first = ['EDD', 'OSL', 'EDD_weight', 'ECT', 'ECT_weight']
        self.jobs_sort_method_second = ['EDD', 'OSL_2', 'EDD_weight', 'stage1_completion_time']
        self.all_job_block = []
        self.job_info = {}
        self.machine_num_on_stage = machine_num_on_stage
        self.job_process_time = job_process_time
        self.ect_weight = ect_weight
        self.ddl_weight = ddl_weight
        self.ddl_windows = ddl_windows
        self.ect_windows = ect_windows
        self.job_list = list(range(jobs_num))

    def gen_jobs_sort(self, gen_method, seed=None):
        # 阶段一
        if gen_method == 'EDD':  # 根据工件的ddl进行排序
            job_sort = np.argsort(np.array(ddl_windows))
        elif gen_method == 'OSL':  # 考虑工件的ddl-工件在所有阶段的完工时间得到的数，再进行一个排列
            process_time_all_stages = []
            for i in range(jobs_num):
                process_time_all_stages.append(sum([j[i] for j in job_process_time]))
            b = np.array(ddl_windows) - np.array(process_time_all_stages)
            job_sort = np.argsort(b)  # 根据ddl或者工件的调度顺序
        elif gen_method == 'EDD_weight':
            b = np.array(ddl_windows) * np.array(ddl_weight)
            job_sort = np.argsort(b)
        elif gen_method == 'ECT':
            job_sort = np.argsort(np.array(ect_windows))

        elif gen_method == 'ECT_weight':
            b = np.array(ect_windows) * np.array(ect_weight)
            job_sort = np.argsort(b)
        elif gen_method == 'OSL_2':
            b = np.array(ddl_windows) - np.array(job_process_time[-1])
            job_sort = np.argsort(b)
        elif gen_method == 'stage1_completion_time':
            b = np.array([self.job_execute_time[(0, job)] for job in range(jobs_num)])
            job_sort = np.argsort(b)
        elif gen_method == 'random':
            random.seed(seed)  # 设置随机种子为42，可以是任意整数
            # 生成随机排序的数组
            job_sort = list(range(0, 10))  # 生成1到10的有序数组
            random.shuffle(job_sort)  # 随机打乱数组顺序
        else:
            job_sort = []

        return job_sort

    def intal_variable(self):
        self.schedule = {}
        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):
                self.schedule[(stage, machine)] = []

        self.job_execute_time = {}
        for stage in range(stages_num):
            for job in range(jobs_num):
                self.job_execute_time[(stage, job)] = 0

    def recal_reset_variable(self):
        self.update_job_execute_time = {}
        for stage in range(stages_num):
            for job in range(jobs_num):
                self.update_job_execute_time[(stage, job)] = 0

    def get_mahine(self, stage):
        machine_avail_time = []
        for machine in range(machine_num_on_stage[stage]):
            if len(self.schedule[(stage, machine)]) == 0:
                machine_avail_time.append(0)
            else:
                pro_job = self.schedule[(stage, machine)][-1]
                machine_avail_time.append(self.job_execute_time[(stage, pro_job)])

        min_index = machine_avail_time.index(min(machine_avail_time))

        return min_index

    def job_assignment(self, gen_method_1, gen_method_2, seed_num=None):

        # 初始化清空变量
        self.intal_variable()

        # 从指定阶段开始，根据上一个阶段排列出的 job_sort 将工件调配到该阶段的各台机器上
        job_sort = self.gen_jobs_sort(gen_method_1)
        for stage in range(stages_num):
            for job in job_sort:
                pro_job = None
                # 获取平行机上的完工时间：
                machine = self.get_mahine(stage)
                if stage == 0:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)] = job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)] = self.job_execute_time[(stage, pro_job)] + \
                                                              job_process_time[stage][job]
                # 如果非第一阶段
                else:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)] = self.job_execute_time[(stage - 1, job)] + \
                                                              job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)] = max(self.job_execute_time[(stage, pro_job)],
                                                                  self.job_execute_time[(stage - 1, job)]) + \
                                                              job_process_time[stage][job]

                self.schedule[(stage, machine)].append(job)

            if stage == 0:
                job_sort = self.gen_jobs_sort(gen_method_2, seed=seed_num)
        self.job_execute_time,self.obj = self.cal(self.job_execute_time)
        print(1, self.obj)
        self.schedule,self.job_execute_time,self.obj = self.idle_time_insertion(self.schedule,self.job_execute_time,self.obj)
        self.get_job_info()
        print(2, self.obj)

    # def re_cal(self):
    #     intal_variable()
    #     for stage in range(stages_num):
    #         for machine in range(machine_num_on_stage[stage]):
    #             for job in self.schedule[(stage,machine)]:
    #                 if stage == 0:
    #                     if len(self.schedule[(stage, machine)]) == 0:
    #                         self.job_execute_time[(stage, job)] = job_process_time[stage][job]
    #                     else:
    #                         pro_job = self.schedule[(stage, machine)][-1]
    #                         self.job_execute_time[(stage, job)] = self.job_execute_time[(stage, pro_job)] + \
    #                                                               job_process_time[stage][job]
    #                 # 如果非第一阶段
    #                 else:
    #                     if len(self.schedule[(stage, machine)]) == 0:
    #                         self.job_execute_time[(stage, job)] = self.job_execute_time[(stage - 1, job)] + \
    #                                                               job_process_time[stage][job]
    #                     else:
    #                         pro_job = self.schedule[(stage, machine)][-1]
    #                         self.job_execute_time[(stage, job)] = max(self.job_execute_time[(stage, pro_job)],
    #                                                                   self.job_execute_time[(stage - 1, job)]) + \
    #                                                               job_process_time[stage][job]

    def cal(self,job_execute_time):
        ect_value = 0
        ddl_value = 0
        for job in range(jobs_num):
            job_makespan = job_execute_time[(stages_num - 1, job)]
            if job_makespan < ect_windows[job]:  # 早前权重值
                ect_value += (ect_windows[job] - job_makespan) * ect_weight[job]
            elif job_makespan > ddl_windows[job]:  # 延误权重值
                ddl_value += (job_makespan - ddl_windows[job]) * ddl_weight[job]

        obj = ect_value + ddl_value

        return job_execute_time,obj

    def idle_time_insertion(self,schedule,job_execute_time,obj):
        # 　外部循环，遍历最后一个阶段上的所有机器
        # obj,job_completion_time = cal(schedule)
        # job_makespan = job_completion_time[stage]
        self.schedule_job_block = {}
        self.all_job_block = []
        for machine in range(machine_num_on_stage[-1]):
            # 内部循环，从机器上最后一个工件开始往前遍历
            job_block = []  # 声明一个空的工件块
            self.all_job_block = []
            delay_job = []  # 最好用字典存储
            early_job = []
            on_time_job = []

            job_list_machine = schedule[(stages_num - 1), machine].copy()
            job_num_machine = len(job_list_machine)  # 判断该机器上有几个工件
            while job_num_machine > 0:
                job = job_list_machine[job_num_machine - 1]
                if job_num_machine == len(job_list_machine):  # 如果是倒数第一个工件
                    later_job = None
                else:
                    later_job = job_list_machine[job_num_machine]

                # 判断是否该工件和后面的工件块合并在一起，导致重新运算

                # 判断这个工件和下一个工件有没有并在一块
                # 如果工件有并在有一块，那么直接插入这个工件块中
                # 如果是最后一个工件,或者第一次查看的时候，这个工件就是紧挨着下一个工件的
                if (job == job_list_machine[-1] or (job_execute_time[(stages_num - 1, job)] ==
                                                    (job_execute_time[(stages_num - 1, later_job)] -
                                                     job_process_time[stages_num - 1][
                                                         later_job]))) and job not in job_block:
                    job_block.insert(0, job)  # 构建工件块

                elif job not in job_block:  # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
                    self.all_job_block.insert(0, job_block)
                    job_block = []  # 再声明一个新的工件块
                    job_block.insert(0, job)  # 重新插入新的工件块中

                job_before_idle = job_block[-1]
                if len(self.all_job_block) != 0:  # 如果当前工件块右侧存在工件
                    job_after_idle = self.all_job_block[0][0]
                    later_block_start_time = job_execute_time[(stages_num - 1, job_after_idle)] - \
                                             job_process_time[stages_num - 1][job_after_idle]
                    job_before_idle_end_time = job_execute_time[(stages_num - 1, job_before_idle)]
                    idle_2 = later_block_start_time - job_before_idle_end_time
                else:
                    idle_2 = np.inf  # 如果右边没有工件了，赋值无穷大

                # 根据当前工件块生成三个子集
                early_job.clear()
                delay_job.clear()
                on_time_job.clear()
                for job in job_block:
                    if job_execute_time[(stages_num - 1, job)] < ect_windows[job]:
                        early_job.append(job)
                    elif job_execute_time[(stages_num - 1, job)] >= ddl_windows[job]:
                        delay_job.append(job)
                    else:
                        on_time_job.append(job)

                early_job_weight = sum([ect_weight[job] for job in early_job])
                delay_job_weight = sum([ddl_weight[job] for job in delay_job])

                if early_job_weight > delay_job_weight:
                    early = []  # 计算距离准时早到的空闲时间
                    delay = []  # 计算超过准时的延误的空闲时间

                    # 计算距离准时的最小早到的空闲时间
                    for job in early_job:
                        early.append(
                            ect_windows[job] - job_execute_time[(stages_num - 1, job)])  # !!!!!!!!这个变量很重要，要实时更新

                    # 计算超过准时的最小延误的空闲时间
                    for job in on_time_job:
                        delay.append(ddl_windows[job] - job_execute_time[(stages_num - 1, job)])
                    if len(early) == 0 and len(delay) != 0:
                        idle_1 = min(delay)
                    elif len(delay) == 0 and len(early) != 0:
                        idle_1 = min(early)
                    else:
                        idle_1 = min(min(early), min(delay))
                    insert_idle_time = min(idle_1, idle_2)  # 确定需要插入的工件块

                    for job in job_block:
                        job_execute_time[(stages_num - 1, job)] += insert_idle_time
                    improvement_obj = (early_job_weight - delay_job_weight) * insert_idle_time  # 获得改进的目标值
                    obj -= improvement_obj  # 重新计算目标值

                    # 判断插入工件块之后，是否会和后面的工件块进行合并
                    if insert_idle_time == idle_2:
                        job_block.extend(self.all_job_block[0])
                        self.all_job_block.remove(self.all_job_block[0])

                    # 更新工件块内的工件完工时间
                else:
                    job_num_machine -= 1  # 而且对于工件块有合并的选项

            self.all_job_block.insert(0, job_block)
            self.schedule_job_block[machine] = self.all_job_block

            # 得到工件的目标值
            for job in job_block:
                if job_execute_time[(stages_num - 1, job)] < ect_windows[job]:
                    early_job.append(job)
                elif job_execute_time[(stages_num - 1, job)] >= ddl_windows[job]:
                    delay_job.append(job)
                else:
                    on_time_job.append(job)

            # 创建一个字典：
            '''
            工件：目标值，偏差距离，延误、早到或准时的flag，以及距离第一阶段的完工时间的差距
            '''
        # self.get_job_info()
        return schedule,job_execute_time,obj

    def get_job_info(self):
        early_job = []
        delay_job = []
        on_time_job = []
        for job in range(jobs_num):
            if self.job_execute_time[(stages_num - 1, job)] < ect_windows[job]:
                early_job.append(job)
                job_flag = -1
                deviate_distance = ect_windows[job] - self.job_execute_time[(stages_num - 1, job)]
                job_obj = deviate_distance * ect_weight[job]

            elif self.job_execute_time[(stages_num - 1, job)] >= ddl_windows[job]:
                delay_job.append(job)
                job_flag = 1
                deviate_distance = self.job_execute_time[(stages_num - 1, job)] - ddl_windows[job]
                job_obj = deviate_distance * ddl_weight[job]

            else:
                on_time_job.append(job)
                deviate_distance = 0
                job_flag = 0
                job_obj = 0
            distance_from_stage1 = (
                    self.job_execute_time[(stages_num - 1, job)] - job_process_time[stages_num - 1][job] -
                    self.job_execute_time[(0, job)])

            self.job_info[job] = (job_obj, deviate_distance, distance_from_stage1, job_flag)

    def initial_solu(self):
        # 生成优质解
        self.inital_refset = {}
        self.inital_refset['opt_solu'] = []
        self.inital_refset['multi_solu'] = []
        for gen_method_1 in self.jobs_sort_method_first:
            for gen_method_2 in self.jobs_sort_method_second:
                self.job_assignment(gen_method_1, gen_method_2)
                # 存储schedule，obj
                self.inital_refset['opt_solu'].append(
                    (self.schedule, self.obj, self.job_execute_time, self.schedule_job_block))
        self.inital_refset['opt_solu'] = sorted(self.inital_refset['opt_solu'], key=lambda x: x[1])
        self.inital_refset['opt_solu'] = self.inital_refset['opt_solu'][:10]
        gen_method_1 = 'random'
        for i in range(4):
            for gen_method_2 in self.jobs_sort_method_second:
                self.job_assignment(gen_method_1, gen_method_2, seed_num=i)
                # 存储schedule，obj
                self.inital_refset['multi_solu'].append(
                    (self.schedule, self.obj, self.job_execute_time, self.schedule_job_block))
        self.inital_refset['multi_solu'] = sorted(self.inital_refset['multi_solu'], key=lambda x: x[1])
        self.inital_refset['multi_solu'] = self.inital_refset['multi_solu'][:10]
        print(1)

    def insert_opera(self, stage, loca_machine, loca_job, selected_job_index, oper_machine, oper_job,
                            oper_job_index):
        self.update_schedule[(stage, loca_machine)].remove(loca_job)
        # oper_job_index = self.schedule[(stage,oper_machine)].index(oper_job)
        oper_job_index = min(oper_job_index,len(self.update_schedule[(stage, oper_machine)])-1)
        if oper_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, oper_machine)].append(loca_job)
            else:
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)
        else:
            self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)


    def swap_opera(self, stage, loca_machine, selected_job, selected_job_index, oper_machine, oper_job, oper_job_index):
        # 执行swap的操作
        self.update_schedule[(stage, loca_machine)].remove(selected_job)
        # oper_job_index = self.update_schedule[(stage, oper_machine)].index(oper_job)
        oper_job_index = min(oper_job_index,len(self.update_schedule[(stage, oper_machine)])-1)
        if oper_job_index == len(self.update_schedule[(stage, oper_machine)])-1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, oper_machine)].append(selected_job)
            else:
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)
        else:
            self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)

        self.update_schedule[(stage, oper_machine)].remove(oper_job)
        selected_job_index = min(selected_job_index,len(self.update_schedule[(stage, loca_machine)])-1)
        if selected_job_index == len(self.update_schedule[(stage, oper_machine)])-1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, loca_machine)].append(oper_job)
            else:
                self.update_schedule[(stage, loca_machine)].insert(selected_job_index, oper_job)
        else:
            self.update_schedule[(stage, loca_machine)].insert(selected_job_index, oper_job)


    def chosen_job2_oper(self, selected_job, stage, method):

        # 根据一定的规则选择需要insert/swap到的位置
        loca_machine = None
        selected_job_index = None
        for machine in range(machine_num_on_stage[stage]):
            if selected_job in self.update_schedule[(stage, machine)]:
                loca_machine = machine
                selected_job_index = self.update_schedule[stage, machine].index(selected_job)
                # print(1,loca_machine)
        # if not loca_machine:
        #     print(1)

        # 随机选择一个机器
        job_flag = self.job_info[selected_job][-1]
        if stage == 0:
            job_flag = 1
        oper_machine = random.choice(list(range(machine_num_on_stage[stage])))
        start_index = 0
        end_index = len(self.update_schedule[stage, oper_machine]) - 1
        if method == 'effe':
            for i_index, job in enumerate(self.update_schedule[stage, oper_machine]):
                if job_flag == -1 and ((self.job_execute_time[(stage, job)] - job_process_time[stage][job]) > \
                        (self.job_execute_time[(stage, selected_job)] - job_process_time[stage][selected_job])):
                    start_index = i_index
                    end_index = len(self.update_schedule[stage, oper_machine]) - 1
                    break
                elif job_flag == 1 and self.job_execute_time[(stage, job)] > self.job_execute_time[
                    (stage, selected_job)]:
                    start_index = 0
                    end_index = i_index
                    break
            if job_flag == -1:
                if end_index:
                    oper_job_index = random.choice(list(range(start_index, end_index + 1)))
                else:
                    oper_job_index = len(self.update_schedule[stage, oper_machine])
            elif job_flag == 1:
                if end_index:
                    oper_job_index = random.choice(list(range(start_index, end_index + 1)))
                else:
                    oper_job_index = 0
            else:
                oper_job_index = random.choice(list(range(len(self.update_schedule[(stage, oper_machine)]))))

        else:
            start_index = 0
            end_index = len(self.update_schedule[stage, oper_machine]) - 1
            oper_job_index = random.choice(list(range(start_index, end_index + 1)))
        oper_job = self.update_schedule[(stage, oper_machine)][oper_job_index]

        return loca_machine, selected_job_index, oper_machine, oper_job, oper_job_index

    def chosen_job(self, method, stage):
        # 选择需要执行操作的工件
        job_list = list(range(jobs_num))
        if method == 'effe':
            if stage == 0:  # 使用第二阶段的开始加工时间 - 第一阶段的完工时间的松紧程序进行排序
                values = [self.job_info[job][2] for job in range(jobs_num)]
                values = [val if val != 0 else 1e-9 for val in values]
                total_value = sum(values)
                probabilities = [self.job_info[job][2] / total_value for job in range(jobs_num)]        # TODO这个要取导读
                selected_job = np.random.choice(job_list, p=probabilities)

            else:  # 使用工件的目标值进行排序

                values = [self.job_info[job][0] for job in range(jobs_num)]
                values = [val if val != 0 else 1e-9 for val in values]
                # reciprocal_values = [1 / val for val in values]
                probabilities = np.array(values) / np.sum(values)     # 归一化概率
                selected_job = np.random.choice(job_list, p=probabilities)
            loca_machine, selected_job_index, oper_machine, oper_job, oper_job_index = self.chosen_job2_oper(
                selected_job, stage, method)

        else:
            selected_job = np.random.choice(job_list)
            loca_machine, selected_job_index, oper_machine, oper_job, oper_job_index = self.chosen_job2_oper(
                selected_job, stage, method)

        return loca_machine, selected_job, selected_job_index, oper_machine, oper_job, oper_job_index

    def search_opea(self, opea_name,schedule,job_execute_time,obj):
        # 根据输入的领域搜索的动作，以及当前解的schedule，工件完工时间等
        self.schedule = schedule
        self.job_execute_time = job_execute_time
        self.obj = obj
        self.update_schedule = schedule
        self.update_job_execute_time = job_execute_time
        self.update_obj = obj
        self.get_job_info()
        stage = int(opea_name[-1])
        oper_method = opea_name[4]
        search_method = opea_name[:4]  # 这个得断点看一下前四个是不是‘effe’
        loca_machine, selected_job, selected_job_index, oper_machine, oper_job, oper_job_index = self.chosen_job(
            search_method, stage)
        if oper_method == 'i':
            self.insert_opera(stage,loca_machine, selected_job, selected_job_index, oper_machine, oper_job,
                            oper_job_index)
        else:
            self.swap_opera(stage, loca_machine, selected_job, selected_job_index, oper_machine, oper_job,
                            oper_job_index)
        self.re_cal()

    def re_cal(self):
        # 执行了领域搜素之后，重新计算得到目标值
        self.recal_reset_variable()

        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):
                pro_job = None
                for i_index ,job in enumerate(self.update_schedule[stage, machine]):
                    if stage == 0:
                        if i_index == 0:
                            self.update_job_execute_time[(stage, job)] = job_process_time[stage][job]
                        else:
                            pro_job = self.update_schedule[(stage, machine)][i_index-1]
                            self.update_job_execute_time[(stage, job)] = self.update_job_execute_time[(stage, pro_job)] + \
                                                                  job_process_time[stage][job]
                    else:
                        if i_index == 0:
                            self.update_job_execute_time[(stage, job)] = self.update_job_execute_time[(stage - 1, job)] + \
                                                                  job_process_time[stage][job]
                        else:
                            pro_job = self.update_schedule[(stage, machine)][i_index-1]
                            self.update_job_execute_time[(stage, job)] = max(self.update_job_execute_time[(stage, pro_job)],
                                                                      self.update_job_execute_time[(stage - 1, job)]) + \
                                                                  job_process_time[stage][job]
        self.update_job_execute_time,self.update_obj = self.cal(self.update_job_execute_time)
        print(0,'self.obj:{0},self.update_obj:{1}'.format(self.obj, self.update_obj))
        self.update_schedule,self.update_job_execute_time, self.update_obj = self.idle_time_insertion(self.update_schedule,self.update_job_execute_time,self.update_obj)
        print(1,'self.obj:{0},self.update_obj:{1}'.format(self.obj, self.update_obj))
        if self.update_obj < self.obj:
            print('self.obj:{0},self.update_obj:{1}'.format(self.obj,self.update_obj))
            self.schedule = self.update_schedule
            self.obj = self.update_obj
            self.job_execute_time = self.update_job_execute_time
        self.get_job_info()


    def action(self):
        '''
        设置8个动作，每个动作代表不同的搜索算子
        :return:
        '''
        action_space = {}
        action_space[0] = (0,'effe','insert')
        action_space[1] = (0,'effe','swap')
        action_space[2] = (1,'effe','insert')
        action_space[3] = (1,'effe','insert')
        action_space[4] = (0,'random','insert')
        action_space[5] = (0,'random','swap')
        action_space[6] = (1,'random','insert')
        action_space[7] = (1,'random','insert')

hfs = HFS()
hfs.initial_solu()
print(hfs.inital_refset)

# schedule = hfs.inital_refset['opt_solu'][0][0]
# job_execute_time = hfs.inital_refset['opt_solu'][0][2]
# epoch = 0
# for key in hfs.inital_refset.keys():
#     for item in hfs.inital_refset[key]:
#         schedule = item[0]
#         job_execute_time = item[2]
#         diagram = job_diagram(schedule,job_process_time,job_execute_time,stages_num,jobs_num,ddl_windows,ect_windows,ect_weight,ddl_weight,epoch)
#         diagram.pre()
#         epoch += 1


schedule = hfs.inital_refset['opt_solu'][0][0]
job_execute_time = hfs.inital_refset['opt_solu'][0][2]
epoch = 0
for key in hfs.inital_refset.keys():
    for item in hfs.inital_refset[key]:
        schedule = item[0]
        job_execute_time = item[2]
        obj = item[1]
        action_space = ['effeinsert0','effeinsert1','randinsert0','randinsert1','effeswap0','effeswap1','randswap0','randswap1']
        for opea_name in action_space:
            hfs.search_opea(opea_name,schedule,job_execute_time,obj)
            hfs.inital_refset[key] = (hfs.schedule,hfs.obj,hfs.job_execute_time)
            schedule = hfs.schedule
            job_execute_time = hfs.job_execute_time
            obj = hfs.obj
            print('success')
        break
