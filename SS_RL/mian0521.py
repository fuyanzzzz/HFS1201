'''
-------------------------------------------------
File Name: main.py
Author: LRS
Create Time: 2023/5/21 16:42
-------------------------------------------------
'''
import os
import numpy as np

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
    def __init__(self):
        pass

    def gen_jobs_sort(self,gen_method,seed = None):
        # 阶段一
        if gen_method == 'EDD':     # 根据工件的ddl进行排序
            job_sort = np.argsort(np.array(ddl_windows))
        elif gen_method == 'OSL':     # 考虑工件的ddl-工件在所有阶段的完工时间得到的数，再进行一个排列
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
        elif gen_method == 'random':
            random.seed(seed_num)  # 设置随机种子为42，可以是任意整数
            # 生成随机排序的数组
            arr = list(range(1, 11))  # 生成1到10的有序数组
            job_sort = random.shuffle(arr)  # 随机打乱数组顺序
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
                self.job_execute_time[(stage, job)] = [0, 0]

    def get_mahine(self,stage):
        machine_avail_time = []
        for machine in range(machine_num_on_stage[stage]):
            if len(self.schedule[(stage,machine)]) == 0:
                machine_avail_time.append(0)
            else:
                pro_job = self.schedule[(stage,machine)][-1]
                machine_avail_time.append(self.job_execute_time[(stage,pro_job)][1])

        min_index = machine_avail_time.index(min(machine_avail_time))

        return min_index


    def job_assignment(self,gen_method_1,gen_method_2):

        # 初始化清空变量
        self.intal_variable()

        # 从指定阶段开始，根据上一个阶段排列出的 job_sort 将工件调配到该阶段的各台机器上
        for stage in range(stages_num):
            job_sort = self.gen_jobs_sort(gen_method_1)
            for job in job_sort:
                pro_job = None
                # 获取平行机上的完工时间：
                machine = self.get_mahine(stage)
                if stage == 0:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)][1] = job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)][1] = self.job_execute_time[(stage, pro_job)][1] + job_process_time[stage][job]
                # 如果非第一阶段
                else:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)][1]  = self.job_execute_time[(stage-1, job)][1] + job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)][1]  = max(self.job_execute_time[(stage, pro_job)][1],
                                                                  self.job_execute_time[(stage - 1, job)][1]) + \
                                                              job_process_time[stage][job]

                self.schedule[(stage,machine)].append(job)
                self.job_execute_time[(stage,job)][0] = self.job_execute_time[(stage,job)][1] - job_process_time[stage][job]

            if stage == 0:
                job_sort = self.gen_jobs_sort(gen_method_2)
        self.cal()



    def re_cal(self):
        intal_variable()
        for stage in range(stages_num):
            for machine in range(machine_num_on_stage[stage]):
                for job in self.schedule[(stage,machine)]:
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

                    self.job_execute_time[(stage, job)][0] = self.job_execute_time[(stage, job)][1] - \
                                                             job_process_time[stage][job]\

    def cal(self):
        ect_value = 0
        ddl_value = 0
        for job in range(jobs_num):
            job_makespan = self.job_execute_time[(stages_num-1,job)][1]
            if job_makespan < ect_windows[job]:  # 早前权重值
                ect_value += max(ect_windows[job] - job_makespan, 0) * ect_weight[job]
            elif job_makespan > ddl_windows[job]:  # 延误权重值
                ddl_value += max(job_makespan - ddl_windows[job], 0) * ddl_weight[job]

        self.obj = ect_value + ddl_value

    def idle_time_insertion(schedule):
        #　外部循环，遍历最后一个阶段上的所有机器
        obj,job_completion_time = cal(schedule)
        job_makespan = job_completion_time[stage]
        all_job_block = []
        for machine in range(machine_num_on_stage[-1]):
            # 内部循环，从机器上最后一个工件开始往前遍历
            job_block = []      # 声明一个空的工件块
            all_job_block = []
            delay_job = []
            early_job = []
            on_time_job = []

            job_list_machine = schedule[(stages_num-1),machine].copy()
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
                    idle_2 = np.inf         # 如果右边没有工件了，赋值无穷大

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


    def initial_solu(self):
        # 生成优质解
        stage_1 = ['EDD','OSL','EDD_weight','ECT','ECT_weight']
        stage_2 = ['EDD','OSL_2','EDD_weight']
        inital_refset = {}
        inital_refset['opt_solu'] = []
        inital_refset['multi_solu'] = []
        for gen_method_1 in stage_1:
            for gen_method_2 in stage_2:
                self.job_assignment(gen_method_1,gen_method_2)
                # 存储schedule，obj
                inital_refset['opt_solu'].append((self.schedule,self.obj))
        sorted_lists = sorted(inital_refset['opt_solu'], key=lambda x: x[-1])
        gen_method_1 == 'random'
        for i in range(3):
            for gen_method_2 in stage_2:
                self.job_assignment(gen_method_1,gen_method_2)
                # 存储schedule，obj
                inital_refset['multi_solu'].append((self.schedule,self.obj))
        print(1)





hfs = HFS()
hfs.initial_solu()


