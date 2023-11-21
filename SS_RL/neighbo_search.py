'''
-------------------------------------------------
File Name: neighbo_search.py
Author: LRS
Create Time: 2023/6/4 20:46
-------------------------------------------------
'''
import copy
import os
import numpy as np
import random

from SS_RL import diagram
from SS_RL.public import AllConfig
from diagram import job_diagram
# from inital_solution  import HFS as ini
import inital_solution as ini
from config import *

class Neighbo_Search():
    def __init__(self,schedule, job_execute_time, obj,file_name):
        # self.opea_name = opea_name
        self.schedule = copy.deepcopy(schedule)
        self.obj = obj
        self.job_execute_time = copy.deepcopy(job_execute_time)

        self.update_schedule = copy.deepcopy(schedule)
        self.config = AllConfig.get_config(file_name)
        self.hfs = ini.HFS(file_name)
        self.recal_reset_variable()
        # job_diagram(self.update_schedule,job_process_time,job_execute_time,stages_num,jobs_num,ddl_windows,ect_windows,ect_weight,ddl_weight,epoch)

    def recal_reset_variable(self):
        self.update_job_execute_time = {}
        for stage in range(self.config.stages_num):
            for job in range(self.config.jobs_num):
                self.update_job_execute_time[(stage, job)] = 0

        self.update_obj = 0

    def insert_opera(self, stage, loca_machine, loca_job, oper_machine, oper_job):

        self.update_schedule[(stage, loca_machine)].remove(loca_job)
        oper_job_index = self.schedule[(stage, oper_machine)].index(oper_job)
        oper_job_index = min(oper_job_index, len(self.update_schedule[(stage, oper_machine)]) - 1)  # 其实这个就是一个防错了！
        if oper_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, oper_machine)].append(loca_job)
            else:
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)
        else:
            self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)

    def swap_opera(self, stage, loca_machine, selected_job, oper_machine, oper_job):

        later_selected_job_index = self.update_schedule[(stage, loca_machine)].index(selected_job) + 1
        later_selected_job_index = min(later_selected_job_index, len(self.update_schedule[(stage, loca_machine)]) - 1)
        later_selected_job = self.update_schedule[(stage, loca_machine)][later_selected_job_index]

        self.update_schedule[(stage, loca_machine)].remove(selected_job)

        oper_job_index = self.update_schedule[(stage, oper_machine)].index(oper_job)
        oper_job_index = min(oper_job_index, len(self.update_schedule[(stage, oper_machine)]) - 1)  # 这个就是一个防错
        if oper_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, oper_machine)].append(selected_job)
            else:
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)
        else:
            self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)

        self.update_schedule[(stage, oper_machine)].remove(oper_job)
        if oper_job == later_selected_job:
            later_selected_job_index = len(self.update_schedule[(stage, loca_machine)])-1
        else:
            if later_selected_job == selected_job:
                later_selected_job_index = len(self.update_schedule[(stage, loca_machine)]) -1
            else:
                later_selected_job_index = self.update_schedule[(stage, loca_machine)].index(later_selected_job)
        if later_selected_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
            i = np.random.choice([0, 1])
            if i == 0:
                self.update_schedule[(stage, loca_machine)].append(oper_job)
            else:
                self.update_schedule[(stage, loca_machine)].insert(later_selected_job_index, oper_job)
        else:
            self.update_schedule[(stage, loca_machine)].insert(later_selected_job_index, oper_job)

    def chosen_job2_oper(self, selected_job, stage, method):

        # 确定被选中的工件所在的机器
        loca_machine = None
        for machine in range(self.config.machine_num_on_stage[stage]):
            if selected_job in self.update_schedule[(stage, machine)]:
                loca_machine = machine

        # 确定被选中的工件属性：
        # 1. 延误工件往前insert / swap
        # 2。早到工件往后insert / swap
        # 3. 准时工件随机insert / swap
        # 4. 若是第一阶段的工件，全部视为延误工件，往前insert / swap
        job_flag = self.job_info[selected_job][-1]
        if stage == 0:
            job_flag = 1

        # 对于另一个要选择的工件
        # 1。先确定好哪台机器【随机选择，但需要考虑该机器上是否存在工件】
        # 2。再确定是该机器上的哪个工件
        while True:
            oper_machine = random.choice(list(range(self.config.machine_num_on_stage[stage])))
            if self.update_schedule[stage, oper_machine]:
                break

        #
        start_index = 0
        end_index = len(self.update_schedule[stage, oper_machine]) - 1
        if method == 'effe':
            for i_index, job in enumerate(self.update_schedule[stage, oper_machine]):
                if job_flag == -1 and ((self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) > \
                                       (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][
                                           selected_job])):
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
                    oper_job_index = len(self.update_schedule[stage, oper_machine]) -1
            elif job_flag == 1:
                if end_index:
                    oper_job_index = random.choice(list(range(start_index, end_index + 1)))
                else:
                    oper_job_index = 0
            else:
                oper_job_index = random.choice(list(range(len(self.update_schedule[(stage, oper_machine)]))))

        else:
            start_index = 0
            end_index = len(self.update_schedule[stage, oper_machine])
            # 如果一个机器上已经空了怎么办，要防止这种情况
            while True:
                if end_index != 0:
                    oper_job_index = random.choice(list(range(start_index, end_index)))
                    break
                else:
                    oper_machine = random.choice(list(range(self.config.machine_num_on_stage[stage])))
                    end_index = len(self.update_schedule[stage, oper_machine])



        oper_job = self.update_schedule[(stage, oper_machine)][oper_job_index]

        return loca_machine, oper_machine, oper_job

    def chosen_job(self, method, stage):
        # 分别在有效搜索 / 随机搜索  的条件下进行选择工件
        if method == 'effe':
            if stage == 0:  # 使用第二阶段的开始加工时间 - 第一阶段的完工时间的松紧程序进行排序
                values = [self.job_info[job][2] for job in range(self.config.jobs_num)]
                values = [val if val != 0 else 1e-9 for val in values]
                values = [1 / item for item in values]      # 需要实现数值越小的元素，被选中的概率越大
                total_value = sum(values)
                probabilities = [values[job] / total_value for job in range(self.config.jobs_num)]
                selected_job = np.random.choice(self.hfs.job_list, p=probabilities)

            else:  # 使用工件的目标值进行排序

                values = [self.job_info[job][0] for job in range(self.config.jobs_num)]
                values = [val if val != 0 else 1e-9 for val in values]
                probabilities = np.array(values) / np.sum(values)  # 归一化概率
                selected_job = np.random.choice(self.hfs.job_list, p=probabilities)

        else:
            selected_job = np.random.choice(self.hfs.job_list)

        # 选定工件之后选择另一个需要操作的工件
        loca_machine, oper_machine, oper_job = self.chosen_job2_oper(selected_job, stage, method)

        return loca_machine, selected_job, oper_machine, oper_job

    def search_opea(self,opea_name):
        # self.re_cal()
        # print(self.update_job_execute_time)
        # print(self.update_schedule)
        #
        # oj = self.hfs.cal(self.update_job_execute_time)
        # print('重新计算目标值：{0}'.format(oj))
        # self.diagram = diagram.job_diagram(self.update_schedule,self.update_job_execute_time,'1')
        # self.diagram.pre()

        # 确定领域搜索的相关信息
        stage = int(opea_name[-1])      # 确定操作的阶段
        oper_method = opea_name[4]      # 确定是insert/swap
        search_method = opea_name[:4]   # 确定是有效搜索/随机搜索

        # 确定每个工件的信息特征，用于后续选择需要操作的工件
        self.job_info = self.hfs.get_job_info(self.job_execute_time)

        # 根据上面的工件信息特征，去确定进行领域搜索涉及的工件信息
        loca_machine, selected_job, oper_machine, oper_job = self.chosen_job(search_method, stage)
        if oper_method == 'i':
            if selected_job != oper_job:
                self.insert_opera(stage, loca_machine, selected_job, oper_machine, oper_job)
        else:
            if selected_job != oper_job:
                self.swap_opera(stage, loca_machine, selected_job, oper_machine, oper_job)

        # 更新工件完工时间
        self.re_cal(self.update_schedule)

        # 更新目标值
        self.update_obj = self.hfs.cal(self.update_job_execute_time)

        # 空闲插入邻域，更新工件完工时间、目标值
        self.update_schedule, self.update_job_execute_time, self.update_obj = self.hfs.idle_time_insertion(
            self.update_schedule, self.update_job_execute_time, self.update_obj)

        return self.update_schedule,self.update_obj,self.update_job_execute_time



    def re_cal(self,need_cal_schedule):
        # 执行了领域搜素之后，重新计算得到目标值
        self.recal_reset_variable()

        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[stage]):
                pro_job = None
                for i_index, job in enumerate(need_cal_schedule[stage, machine]):
                    if stage == 0:
                        if i_index == 0:
                            self.update_job_execute_time[(stage, job)] = self.config.job_process_time[stage][job]
                        else:
                            pro_job = need_cal_schedule[(stage, machine)][i_index - 1]
                            self.update_job_execute_time[(stage, job)] = self.update_job_execute_time[
                                                                             (stage, pro_job)] + \
                                                                         self.config.job_process_time[stage][job]
                    else:
                        if i_index == 0:
                            self.update_job_execute_time[(stage, job)] = self.update_job_execute_time[
                                                                             (stage - 1, job)] + \
                                                                         self.config.job_process_time[stage][job]
                        else:
                            pro_job = need_cal_schedule[(stage, machine)][i_index - 1]
                            self.update_job_execute_time[(stage, job)] = max(
                                self.update_job_execute_time[(stage, pro_job)],
                                self.update_job_execute_time[(stage - 1, job)]) + \
                                                                         self.config.job_process_time[stage][job]

