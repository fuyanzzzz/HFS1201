'''
-------------------------------------------------
File Name: inital_solution.py
Author: LRS
Create Time: 2023/6/4 20:43
-------------------------------------------------
'''
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
from config import *
from public import AllConfig
from Schedule import Schedule_Instance


class HFS():
    def __init__(self,file_name,jingying_num):
        '''
        基础数据变量：加工时间，阶段数量，工件数量，每阶段的机器数量，工件列表，延误/早到窗口，延误/早到权重，
        初始参考集变量：阶段一解生成方式，阶段二解生成方式
        其他变量
        '''
        # 基础数据变量
        self.file_name = file_name
        self.config = AllConfig.get_config(file_name)
        self.job_list = list(range(self.config.jobs_num))


        # 初始参考集变量
        self.jobs_sort_method_first = ['EDD', 'OSL', 'EDD_weight', 'ECT', 'ECT_weight']
        self.jobs_sort_method_second = ['EDD', 'OSL_2', 'EDD_weight', 'stage1_completion_time']

        # 其他变量
        self.all_job_block = []
        self.job_info = {}
        self.inital_refset = []
        self.population_refset = []
        self.intal_variable()
        self.schedule_job_block = {}
        self.population = self.config.jobs_num * 2
        self.jingying_num = jingying_num


    def gen_jobs_sort(self, gen_method, seed=None):
        # 工件排列方式
        if gen_method == 'EDD':  # 根据工件的ddl进行排序
            job_sort = np.argsort(np.array(self.config.ddl_windows))
        elif gen_method == 'OSL':  # 考虑工件的ddl-工件在所有阶段的完工时间得到的数，再进行一个排列
            process_time_all_stages = []
            for i in range(self.config.jobs_num):
                process_time_all_stages.append(sum([j[i] for j in self.config.job_process_time]))
            b = np.array(self.config.ddl_windows) - np.array(process_time_all_stages)
            job_sort = np.argsort(b)  # 根据ddl或者工件的调度顺序
        elif gen_method == 'EDD_weight':
            b = np.array(self.config.ddl_windows) * np.array(self.config.ddl_weight)
            job_sort = np.argsort(b)
        elif gen_method == 'ECT':
            job_sort = np.argsort(np.array(self.config.ect_windows))

        elif gen_method == 'ECT_weight':
            b = np.array(self.config.ect_windows) * np.array(self.config.ect_weight)
            job_sort = np.argsort(b)
        elif gen_method == 'OSL_2':
            stage1_completion_time = np.array([self.job_execute_time[(0, job)] for job in range(self.config.jobs_num)])
            b = np.array(self.config.ddl_windows) - np.array(self.config.job_process_time[-1]) - stage1_completion_time
            job_sort = np.argsort(b)
        elif gen_method == 'stage1_completion_time':
            b = np.array([self.job_execute_time[(0, job)] for job in range(self.config.jobs_num)])
            job_sort = np.argsort(b)
        elif gen_method == 'random':
            # random.seed(seed)  # 设置随机种子为42，可以是任意整数
            # 生成随机排序的数组
            job_sort = self.job_list
            random.shuffle(job_sort)  # 随机打乱数组顺序
        else:
            job_sort = []

        return job_sort

    def intal_variable(self):
        # 变量初始化
        self.schedule = {}
        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[stage]):
                self.schedule[(stage, machine)] = []

        self.job_execute_time = {}
        for stage in range(self.config.stages_num):
            for job in range(self.config.jobs_num):
                self.job_execute_time[(stage, job)] = 0

        self.obj = 0

        # TODO 目标值是否需要初始化

    def get_mahine(self, stage):
        # 当前排列工件需要分配到的机器
        machine_avail_time = []
        for machine in range(self.config.machine_num_on_stage[stage]):
            if len(self.schedule[(stage, machine)]) == 0:
                machine_avail_time.append(0)
            else:
                pro_job = self.schedule[(stage, machine)][-1]
                machine_avail_time.append(self.job_execute_time[(stage, pro_job)])

        min_index = machine_avail_time.index(min(machine_avail_time))

        return min_index

    def job_assignment(self, gen_method_1, gen_method_2, seed_num=None):
        # 工件分配

        # 初始化清空变量
        self.intal_variable()

        # 从指定阶段开始，根据上一个阶段排列出的 job_sort 将工件调配到该阶段的各台机器上
        job_sort = self.gen_jobs_sort(gen_method_1)
        for stage in range(self.config.stages_num):
            for job in job_sort:
                pro_job = None
                # 获取平行机上的完工时间：
                machine = self.get_mahine(stage)
                if stage == 0:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)] = self.config.job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)] = self.job_execute_time[(stage, pro_job)] + \
                                                              self.config.job_process_time[stage][job]
                # 如果非第一阶段
                else:
                    if len(self.schedule[(stage, machine)]) == 0:
                        self.job_execute_time[(stage, job)] = self.job_execute_time[(stage - 1, job)] + \
                                                              self.config.job_process_time[stage][job]
                    else:
                        pro_job = self.schedule[(stage, machine)][-1]
                        self.job_execute_time[(stage, job)] = max(self.job_execute_time[(stage, pro_job)],
                                                                  self.job_execute_time[(stage - 1, job)]) + \
                                                              self.config.job_process_time[stage][job]

                self.schedule[(stage, machine)].append(job)

            if stage == 0:
                job_sort = self.gen_jobs_sort(gen_method_2, seed=seed_num)
        # 以上只做工件分配，更新变量self.schedule, self.job_execute

        # 更新目标值
        self.schedule_ins = Schedule_Instance(self.schedule, self.job_execute_time,self.file_name)
        self.obj = self.schedule_ins.cal(self.job_execute_time)
        # 进行空闲插入程序后的目标值

        # from SS_RL.diagram import job_diagram
        # import matplotlib.pyplot as plt
        # dia = job_diagram(self.schedule, self.job_execute_time, self.file_name, 1)
        # dia.pre()
        # plt.savefig('./img1203/pic-{}.png'.format(int(self.obj)))
        if self.obj == 38337:
            print(1)
        self.schedule,self.job_execute_time,self.obj = self.schedule_ins.idle_time_insertion(self.schedule,self.job_execute_time,self.obj)
        # self.get_job_info(self.job_execute_time)
        self.schedule_ins.get_job_info(self.job_execute_time)

    # def cal(self,job_execute_time):
    #     ect_value = 0
    #     ddl_value = 0
    #     for job in range(self.config.jobs_num):
    #         job_makespan = job_execute_time[(self.config.stages_num - 1, job)]
    #         if job_makespan < self.config.ect_windows[job]:  # 早前权重值
    #             ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
    #         elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
    #             ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
    #
    #     obj = ect_value + ddl_value
    #
    #     return obj

    # def idle_time_insertion(self,schedule,job_execute_time,obj):
    #     # 　外部循环，遍历最后一个阶段上的所有机器
    #
    #
    #     for machine in range(self.config.machine_num_on_stage[-1]):
    #         self.schedule_job_block[machine] = []
    #         # 内部循环，从机器上最后一个工件开始往前遍历
    #         job_block = []  # 声明一个空的工件块
    #         self.all_job_block = []
    #         delay_job = []  # 最好用字典存储
    #         early_job = []
    #         on_time_job = []
    #
    #         job_list_machine = schedule[(self.config.stages_num - 1), machine].copy()
    #         job_num_machine = len(job_list_machine)  # 判断该机器上有几个工件
    #         while job_num_machine > 0:
    #             job = job_list_machine[job_num_machine - 1]
    #             if job_num_machine == len(job_list_machine):  # 如果是倒数第一个工件
    #                 later_job = None
    #             else:
    #                 later_job = job_list_machine[job_num_machine]
    #
    #             # 判断是否该工件和后面的工件块合并在一起，导致重新运算
    #
    #             # 判断这个工件和下一个工件有没有并在一块
    #             # 如果工件有并在有一块，那么直接插入这个工件块中
    #             # 如果是最后一个工件,或者第一次查看的时候，这个工件就是紧挨着下一个工件的
    #             if (job == job_list_machine[-1] or (job_execute_time[(self.config.stages_num - 1, job)] ==
    #                                                 (job_execute_time[(self.config.stages_num - 1, later_job)] -
    #                                                  self.config.job_process_time[self.config.stages_num - 1][
    #                                                      later_job]))) and job not in job_block:
    #                 job_block.insert(0, job)  # 构建工件块
    #
    #             elif job not in job_block:  # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
    #                 self.all_job_block.insert(0, job_block)
    #                 job_block = []  # 再声明一个新的工件块
    #                 job_block.insert(0, job)  # 重新插入新的工件块中
    #
    #             job_before_idle = job_block[-1]
    #             if len(self.all_job_block) != 0:  # 如果当前工件块右侧存在工件
    #                 job_after_idle = self.all_job_block[0][0]
    #                 later_block_start_time = job_execute_time[(self.config.stages_num - 1, job_after_idle)] - \
    #                                          self.config.job_process_time[self.config.stages_num - 1][job_after_idle]
    #                 job_before_idle_end_time = job_execute_time[(self.config.stages_num - 1, job_before_idle)]
    #                 idle_2 = later_block_start_time - job_before_idle_end_time
    #             else:
    #                 idle_2 = np.inf  # 如果右边没有工件了，赋值无穷大
    #
    #             # 根据当前工件块生成三个子集
    #             early_job.clear()
    #             delay_job.clear()
    #             on_time_job.clear()
    #             for job in job_block:
    #                 if job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
    #                     early_job.append(job)
    #
    #                 elif job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
    #                     delay_job.append(job)
    #
    #                 else:
    #                     on_time_job.append(job)
    #
    #
    #             early_job_weight = sum([self.config.ect_weight[job] for job in early_job])
    #             delay_job_weight = sum([self.config.ddl_weight[job] for job in delay_job])
    #
    #             if early_job_weight > delay_job_weight:
    #                 early = []  # 计算距离准时早到的空闲时间
    #                 delay = []  # 计算超过准时的延误的空闲时间
    #
    #                 # 计算距离准时的最小早到的空闲时间
    #                 for job in early_job:
    #                     early.append(
    #                         self.config.ect_windows[job] - job_execute_time[(self.config.stages_num - 1, job)])  # !!!!!!!!这个变量很重要，要实时更新
    #
    #                 # 计算超过准时的最小延误的空闲时间
    #                 for job in on_time_job:
    #                     delay.append(self.config.ddl_windows[job] - job_execute_time[(self.config.stages_num - 1, job)])
    #                 if len(early) == 0 and len(delay) != 0:
    #                     idle_1 = min(delay)
    #                 elif len(delay) == 0 and len(early) != 0:
    #                     idle_1 = min(early)
    #                 else:
    #                     idle_1 = min(min(early), min(delay))
    #                 insert_idle_time = min(idle_1, idle_2)  # 确定需要插入的工件块
    #
    #                 for job in job_block:
    #                     job_execute_time[(self.config.stages_num - 1, job)] += insert_idle_time
    #                 improvement_obj = (early_job_weight - delay_job_weight) * insert_idle_time  # 获得改进的目标值
    #                 obj -= improvement_obj  # 重新计算目标值
    #
    #                 # 判断插入工件块之后，是否会和后面的工件块进行合并
    #                 if insert_idle_time == idle_2:
    #                     job_block.extend(self.all_job_block[0])
    #                     self.all_job_block.remove(self.all_job_block[0])
    #
    #                 # 更新工件块内的工件完工时间
    #             else:
    #                 job_num_machine -= 1  # 而且对于工件块有合并的选项
    #
    #         self.all_job_block.insert(0, (job_block,early_job_weight,delay_job_weight-early_job_weight
    #                         ,delay_job_weight,early_job_weight - delay_job_weight))
    #         self.schedule_job_block[machine].append(self.all_job_block)
    #
    #         # 得到工件的目标值
    #         for job in job_block:
    #             if job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
    #                 early_job.append(job)
    #             elif job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
    #                 delay_job.append(job)
    #             else:
    #                 on_time_job.append(job)
    #
    #         # 创建一个字典：
    #         '''
    #         工件：目标值，偏差距离，延误、早到或准时的flag，以及距离第一阶段的完工时间的差距
    #         '''
    #     # self.get_job_info()
    #     return schedule,job_execute_time,obj

    # def get_job_info(self,job_execute_time):
    #     job_info = {}
    #     early_job = []
    #     delay_job = []
    #     on_time_job = []
    #     for job in range(self.config.jobs_num):
    #         if job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
    #             early_job.append(job)
    #             job_flag = -1
    #             deviate_distance = self.config.ect_windows[job] - job_execute_time[(self.config.stages_num - 1, job)]
    #             job_obj = deviate_distance * self.config.ect_weight[job]
    #
    #         elif job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
    #             delay_job.append(job)
    #             job_flag = 1
    #             deviate_distance = job_execute_time[(self.config.stages_num - 1, job)] - self.config.ddl_windows[job]
    #             job_obj = deviate_distance * self.config.ddl_weight[job]
    #
    #         else:
    #             on_time_job.append(job)
    #             deviate_distance = 0
    #             job_flag = 0
    #             job_obj = 0
    #         distance_from_stage1 = (
    #                 job_execute_time[(self.config.stages_num - 1, job)] - self.config.job_process_time[self.config.stages_num - 1][job] -
    #                 job_execute_time[(0, job)])
    #
    #         job_info[job] = (job_obj, deviate_distance, distance_from_stage1, job_flag)

        # return job_info

    def initial_solu(self):
        '''
        优质解：
        第一阶段 = 【EDD，OSL，EDD_weight，ECT，ECT_weight】
        第二阶段 = 【EDD，OSL_2，EDD_weight】
        多样解：
        第一阶段 = 【随机生成】
        第二阶段 = 【EDD，OSL_2，EDD_weight】
        '''
        # 按照工件规模去生成初始种群P，如20n
        # 优质解 占比是1/10 P
        for gen_method_1 in self.jobs_sort_method_first:
            for gen_method_2 in self.jobs_sort_method_second:
                self.job_assignment(gen_method_1, gen_method_2)
                # 存储schedule，obj
                self.population_refset.append((
                    self.schedule, self.obj, self.job_execute_time, self.schedule_job_block))
        self.population_refset = sorted(self.population_refset, key=lambda x: x[1])
        self.population_refset = self.population_refset[:self.jingying_num]
        # self.population_refset['opt_solu'] = self.population_refset['opt_solu'][:2]

        # 随机解
        gen_method_1 = 'random'
        need_rabdom_num = 20 * self.config.jobs_num - len(self.population_refset)
        # 这个记得断一下点看下
        need_break = False
        while True:
            for gen_method_2 in self.jobs_sort_method_second:
                self.job_assignment(gen_method_1, gen_method_2)
                # 存储schedule，obj
                self.population_refset.append(
                    (self.schedule, self.obj, self.job_execute_time, self.schedule_job_block))
                if len(self.population_refset) == 20 * self.config.jobs_num:
                    need_break = True
                    break
            if need_break:
                break
        self.population_refset = sorted(self.population_refset, key=lambda x: x[1])
        # self.population_refset = self.population_refset[:self.population]
        self.bulid_reference(self.population_refset)
        if len(self.population_refset) < 20 * self.config.jobs_num:
            need_break = False
            while True:
                for gen_method_2 in self.jobs_sort_method_second:
                    self.job_assignment(gen_method_1, gen_method_2)
                    # 存储schedule，obj
                    self.population_refset.append(
                        (self.schedule, self.obj, self.job_execute_time, self.schedule_job_block))
                    if len(self.population_refset) == 20 * self.config.jobs_num:
                        need_break = True
                        break
                if need_break:
                    break
            self.population_refset = sorted(self.population_refset, key=lambda x: x[1])

        # self.population_refset['multi_solu'] = self.population_refset['multi_solu'][:]

    def bulid_reference(self,population_refset):
        # 选择整个初始解种群P中的1/10，再乘1/3的精英解
        self.population_refset = population_refset
        import copy
        elite_num = int(len(self.population_refset) * 1/10 * 1/3)
        self.inital_refset = copy.deepcopy(self.population_refset[:elite_num])
        self.population_refset = self.population_refset[elite_num:]     # 这个需要验证一下是否会被替代掉
        # 多样解的构建
        from scipy.stats import kendalltau


        distance_list = []
        need_remove_item = []
        need_break = False
        while True:
            aver_tau_list = []
            for item in self.population_refset:

                schedule = item[0]
                schedule_list_0 = []
                schedule_list_1 = []

                for i_machine in range(self.config.machine_num_on_stage[0]):
                    schedule_list_0 += schedule[(0,i_machine)]
                    schedule_list_1 += schedule[(1,i_machine)]

                for i_i_item in self.inital_refset:
                    i_i_item_schedule = i_i_item[0]
                    i_i_item_schedule_list_0 = []
                    i_i_item_schedule_list_1 = []

                    for i_i_machine in range(self.config.machine_num_on_stage[0]):
                        i_i_item_schedule_list_0 += i_i_item_schedule[(0, i_i_machine)]
                        i_i_item_schedule_list_1 += i_i_item_schedule[(1, i_i_machine)]
                    # 计算Kendall tau距离
                    tau_0, p_value_0 = kendalltau(schedule_list_0, i_i_item_schedule_list_0)
                    tau_1, p_value_1 = kendalltau(schedule_list_1, i_i_item_schedule_list_1)
                    tau = (tau_0+tau_1)/2
                    distance_list.append(tau)
                aver_tau = sum(distance_list) / len(distance_list)
                aver_tau_list.append(aver_tau)
            i_index = aver_tau_list.index(min(aver_tau_list))
            self.inital_refset.append(self.population_refset[i_index])
            self.population_refset.remove(self.inital_refset[-1])
            if len(self.inital_refset) ==  2 * self.config.jobs_num:
                break
        self.inital_refset = copy.deepcopy(self.inital_refset)
        print(1)



