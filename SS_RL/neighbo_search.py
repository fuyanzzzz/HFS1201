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
from Schedule import Schedule_Instance

class Neighbo_Search():
    def __init__(self,schedule, job_execute_time, obj,file_name,jingying_num):
        # self.opea_name = opea_name
        self.schedule = copy.deepcopy(schedule)
        self.obj = obj
        self.job_execute_time = copy.deepcopy(job_execute_time)
        self.file_name = file_name

        self.update_schedule = copy.deepcopy(schedule)
        self.config = AllConfig.get_config(file_name)
        self.hfs = ini.HFS(file_name,jingying_num)
        self.recal_reset_variable()
        self.schedule_ins = Schedule_Instance(schedule, job_execute_time,file_name)
        # job_diagram(self.update_schedule,job_process_time,job_execute_time,stages_num,jobs_num,ddl_windows,ect_windows,ect_weight,ddl_weight,epoch)

    def recal_reset_variable(self):
        self.update_job_execute_time = {}
        for stage in range(self.config.stages_num):
            for job in range(self.config.jobs_num):
                self.update_job_execute_time[(stage, job)] = 0

        self.update_obj = self.obj

    def insert_opera(self, stage, loca_machine, loca_job, oper_machine, oper_job,search_method_1,insert_last):
        # 获取插入工件的索引
        loca_job_index = self.update_schedule[(stage, loca_machine)].index(loca_job)
        # 以下暂时不清楚作用是什么
        if loca_job_index+1 < len(self.update_schedule[(stage, loca_machine)]):
            loca_job_later_ = self.update_schedule[(stage, loca_machine)][loca_job_index+1]
        else:
            loca_job_later_ = loca_job
        loca_job_later_index = self.update_schedule[(stage, loca_machine)].index(loca_job_later_)
        self.update_schedule[(stage, loca_machine)].remove(loca_job)
        # 获取被插入工件的索引
        oper_job_index = self.update_schedule[(stage, oper_machine)].index(oper_job)

        # 确保插入的工件不超过最后一个工件的位置
        oper_job_index = min(oper_job_index, len(self.update_schedule[(stage, oper_machine)]) - 1)  # 其实这个就是一个防错了！
        if search_method_1 == 'dire':
            stage_0_loca_job_machine = None
            for machine in range(self.config.machine_num_on_stage[0]):
                for job in self.update_schedule[(0, machine)]:
                    if job == loca_job:
                        stage_0_loca_job_machine = machine
                        break
            if oper_job == self.update_schedule[(1,oper_machine)][0]:
                # 若是早到工件，插入的位置是该机器上的第一个位置
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)
                # 第一阶段的工件也在insert到最后【应该是第一阶段所有工件中完工时间最早的机器上】        # 还需要完善
                # 获得第一阶段上所有机器上最后一个工件的完工时间，取具有最小完工时间的机器
                # chosen_machine_0 = None
                # chosen_job_0 = None
                # min_compelte_time_0 = np.inf
                # for i_machine in range(self.config.machine_num_on_stage[0]):
                #     i_last_job = self.update_schedule[(0, i_machine)]
                #     if self.job_execute_time[(stage,i_last_job)] < min_compelte_time_0:
                #         min_compelte_time_0 = self.job_execute_time[(stage,i_last_job)]
                #         chosen_job_0 = i_last_job
                #         chosen_machine_0 = i_machine

                self.update_schedule[(0, stage_0_loca_job_machine)].remove(loca_job)
                self.update_schedule[(0, stage_0_loca_job_machine)].insert(0, loca_job)


            else:
                self.update_schedule[(1, oper_machine)].append(loca_job)
                # 第一阶段的工件也在insert到最前面
                chosen_machine_0 = None
                chosen_job_0 = None
                min_compelte_time_0 = np.inf
                for i_machine in range(self.config.machine_num_on_stage[0]):
                    if self.update_schedule[(0, i_machine)]:
                        i_last_job = self.update_schedule[(0, i_machine)][-1]
                        if self.job_execute_time[(0,i_last_job)] < min_compelte_time_0:
                            min_compelte_time_0 = self.job_execute_time[(stage,i_last_job)]
                            chosen_job_0 = i_last_job
                            chosen_machine_0 = i_machine
                self.update_schedule[(0, stage_0_loca_job_machine)].remove(loca_job)
                self.update_schedule[(0, chosen_machine_0)].append(loca_job)

            # 如果因此此次的操作导致有新的工件卡住，则在第一阶段的所有机器中选择一个松弛度最高的机器插入
            self.re_cal(self.update_schedule)
            update_schedule = copy.deepcopy(self.update_schedule)
            update_job_execute_time = copy.deepcopy(self.update_job_execute_time)
            obj_item = self.schedule_ins.cal(self.update_job_execute_time)
            _,_,after_obj = self.schedule_ins.idle_time_insertion(update_schedule, self.update_job_execute_time, obj_item)

            if stage == 1 and loca_job_index < len(self.update_schedule[(0, loca_machine)]):

                for job in self.update_schedule[(0, loca_machine)][loca_job_later_index:]:
                    # 如果有工件卡住，才会进入到这个if条件中，且一旦进入一次，就会跳出
                    max_slackness_job = None
                    max_slackness_job_machine = None
                    max_slackness = 0
                    if self.update_job_execute_time[(stage,job)] - self.config.job_process_time[stage][job] == \
                    self.update_job_execute_time[(0,job)]:
                        # 判断这个卡住的工件在第一阶段的哪台机器
                        for machine in range(self.config.machine_num_on_stage[0]):
                            for i_job in self.update_schedule[(0, machine)]:
                                if i_job == job:
                                    job_machine = machine
                        for machine in range(self.config.machine_num_on_stage[0]):
                            for i_job in self.update_schedule[(0, machine)]:

                                if (self.job_execute_time[(0, i_job)] - self.config.job_process_time[0][i_job]) < \
                                        (self.job_execute_time[(0, job)] - self.config.job_process_time[0][
                                            job]) \
                                        and i_job != self.update_schedule[(0, machine)][-1]:
                                    continue
                                elif (self.job_execute_time[(0, i_job)] - self.config.job_process_time[0][i_job]) < \
                                        (self.job_execute_time[(0, job)] - self.config.job_process_time[0][
                                            job]) \
                                        and i_job == self.update_schedule[(0, machine)][-1]:
                                    slackness_i_job = self.update_job_execute_time[(1,i_job)] - self.config.job_process_time[1][i_job] - self.update_job_execute_time[(0,i_job)]
                                    if slackness_i_job > max_slackness:
                                        max_slackness = slackness_i_job
                                        max_slackness_job = i_job
                                        max_slackness_job_machine = machine
                                else:
                                    # oper_job_list[machine] = self.update_schedule[(stage, machine)][:index]

                                    pre_index = self.update_schedule[(0, machine)].index(i_job) - 1
                                    if pre_index >= 0:
                                        pre_job = self.update_schedule[(0, machine)][pre_index]
                                        slackness_i_job = self.update_job_execute_time[(1, pre_job)] - \
                                                          self.config.job_process_time[1][pre_job] - \
                                                          self.update_job_execute_time[(0, pre_job)]
                                        if slackness_i_job > max_slackness:
                                            max_slackness = slackness_i_job
                                            max_slackness_job = pre_job
                                            max_slackness_job_machine = machine
                        if max_slackness_job:
                            self.swap_opera(0, max_slackness_job_machine, max_slackness_job, job_machine, job)
                            self.re_cal(self.update_schedule)
                            obj_item_2 = self.schedule_ins.cal(self.update_job_execute_time)
                            _, _, after_obj_2 = self.schedule_ins.idle_time_insertion(self.update_schedule,
                                                                                    self.update_job_execute_time, obj_item_2)
                            if after_obj < after_obj_2:
                                self.update_schedule = update_schedule
                                self.update_job_execute_time = update_job_execute_time
                        break


        else:
            # 若需要操作的工件是该机器上的最后一个工件
            if oper_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
                # i = np.random.choice([0, 1])
                if insert_last:
                    self.update_schedule[(stage, oper_machine)].append(loca_job)
                else:
                    self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)
            else:
                self.update_schedule[(stage, oper_machine)].insert(oper_job_index, loca_job)

            # 执行完插入操作之后，判断一下这两个工件是否会被卡，如果被卡住，交换他们在第一阶段的位置
            # 判断现在两个工件第二阶段的开始加工时间，谁在先：
            self.re_cal(self.update_schedule)
            update_schedule = copy.deepcopy(self.update_schedule)
            # loca_job表示需要操作的工件，oper_job表示的是需要被插入或者交换的工件
            if (self.update_job_execute_time[(1,loca_job)] - self.config.job_process_time[1][loca_job]) <\
            (self.update_job_execute_time[(1, oper_job)] - self.config.job_process_time[1][oper_job]):
                per_job = loca_job
                later_job = oper_job
            else:
                per_job = oper_job
                later_job = loca_job

            if (self.update_job_execute_time[(0,per_job)] - self.config.job_process_time[0][per_job]) >\
            (self.update_job_execute_time[(0,later_job)] - self.config.job_process_time[0][later_job]):
                # 找到这两个工件所在的机器
                per_job_machine = None
                later_job_machine = None
                for machine in range(self.config.machine_num_on_stage[0]):
                    for job in self.update_schedule[(0, machine)]:
                        if job == per_job:
                            per_job_machine = machine
                        if job == later_job:
                            later_job_machine = machine

                self.swap_opera(0, per_job_machine, per_job, later_job_machine, later_job)
                self.re_cal(self.update_schedule)


    def swap_opera(self, stage, loca_machine, selected_job, oper_machine, oper_job):

        index_to_swap1 = self.update_schedule[(stage, loca_machine)].index(selected_job)
        index_to_swap2 = self.update_schedule[(stage, oper_machine)].index(oper_job)

        self.update_schedule[(stage, loca_machine)][index_to_swap1], self.update_schedule[(stage, oper_machine)][
            index_to_swap2] = \
            self.update_schedule[(stage, oper_machine)][index_to_swap2], \
            self.update_schedule[(stage, loca_machine)][index_to_swap1]

        self.re_cal(self.update_schedule)
        if (self.job_execute_time[(1, selected_job)] - self.config.job_process_time[1][selected_job]) < \
                (self.job_execute_time[(1, oper_job)] - self.config.job_process_time[1][oper_job]):
            per_job = selected_job
            later_job = oper_job
        else:
            per_job = oper_job
            later_job = selected_job

        if (self.job_execute_time[(0, per_job)] - self.config.job_process_time[1][per_job]) > \
                (self.job_execute_time[(0, later_job)] - self.config.job_process_time[1][later_job]):
            # 找到这两个工件所在的机器
            per_job_machine = None
            later_job_machine = None
            for machine in range(self.config.machine_num_on_stage[0]):
                for job in self.update_schedule[(0, machine)]:
                    if job == per_job:
                        per_job_machine = machine
                    if job == later_job:
                        later_job_machine = machine

            index_to_swap1 = self.update_schedule[(0, per_job_machine)].index(per_job)
            index_to_swap2 = self.update_schedule[(0, later_job_machine)].index(later_job)

            self.update_schedule[(0, per_job_machine)][index_to_swap1], self.update_schedule[(0, later_job_machine)][
                index_to_swap2] = \
                self.update_schedule[(0, later_job_machine)][index_to_swap2], \
                self.update_schedule[(0, per_job_machine)][index_to_swap1]

            # index_to_swap1 = 1
            # index_to_swap2 = 3
            #
            # # 使用索引直接交换两个元素的值
            # my_list[index_to_swap1], my_list[index_to_swap2] = my_list[index_to_swap2], my_list[index_to_swap1]



        # later_selected_job_index = self.update_schedule[(stage, loca_machine)].index(selected_job) + 1
        # later_selected_job_index = min(later_selected_job_index, len(self.update_schedule[(stage, loca_machine)]) - 1)
        # later_selected_job = self.update_schedule[(stage, loca_machine)][later_selected_job_index]
        #
        # self.update_schedule[(stage, loca_machine)].remove(selected_job)
        #
        # oper_job_index = self.update_schedule[(stage, oper_machine)].index(oper_job)
        # oper_job_index = min(oper_job_index, len(self.update_schedule[(stage, oper_machine)]) - 1)  # 这个就是一个防错
        # if oper_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
        #     # i = np.random.choice([0, 1])
        #     # if i == 0:
        #     #     self.update_schedule[(stage, oper_machine)].append(selected_job)
        #     # else:
        #     self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)
        # else:
        #     self.update_schedule[(stage, oper_machine)].insert(oper_job_index, selected_job)
        #
        # self.update_schedule[(stage, oper_machine)].remove(oper_job)
        # if oper_job == later_selected_job:
        #     later_selected_job_index = len(self.update_schedule[(stage, loca_machine)])-1
        # else:
        #     if later_selected_job == selected_job:
        #         later_selected_job_index = len(self.update_schedule[(stage, loca_machine)]) -1
        #     else:
        #         later_selected_job_index = self.update_schedule[(stage, loca_machine)].index(later_selected_job)
        # if later_selected_job_index == len(self.update_schedule[(stage, oper_machine)]) - 1:
        #     i = np.random.choice([0, 1])
        #     if i == 0:
        #         self.update_schedule[(stage, loca_machine)].append(oper_job)
        #     else:
        #         self.update_schedule[(stage, loca_machine)].insert(later_selected_job_index, oper_job)
        # else:
        #     self.update_schedule[(stage, loca_machine)].insert(later_selected_job_index, oper_job)

    def chosen_job2_oper(self, selected_job, stage, search_method_1,config_same_machine,oper_method):

        # 确定被选中的工件所在的机器
        # print('1self.update_schedule:{0}'.format(self.update_schedule))
        loca_machine = None
        oper_job_list = {}
        for machine in range(self.config.machine_num_on_stage[stage]):
            if selected_job in self.update_schedule[(stage, machine)]:
                loca_machine = machine
        min_job_execute_time = np.inf
        oper_machine = None
        if search_method_1 == 'dire':
            if config_same_machine:
                oper_job_list[loca_machine] = [self.update_schedule[(stage, loca_machine)][-1]]
            else:
                oper_machine_list = list(range(self.config.machine_num_on_stage[0]))
                for machine in oper_machine_list:
                    if machine == loca_machine:
                        continue
                    # if len(self.update_schedule[(stage, machine)])>0 and min_job_execute_time > self.job_execute_time[(1,self.update_schedule[(stage, machine)][-1])]:
                    #     min_job_execute_time = self.job_execute_time[(1,self.update_schedule[(stage, machine)][-1])]
                    #     oper_machine = machine
                    if len(self.update_schedule[(stage, machine)]) > 0:
                        oper_job_list[machine] = [self.update_schedule[(stage, machine)][-1]]

        else:
            # 早到工件往后insert/swap,延误工件往前insert/swap6
            job_info = self.schedule_ins.get_job_info(self.job_execute_time)
            job_flag = job_info[selected_job][-1]
            if stage == 0:
                job_flag = 1    # 如果是阶段0的工件，直接将属性赋值为1，往前插入
            # 确定工件是否在同一个机器上进行insert/swap

            if config_same_machine:
                oper_machine = loca_machine
                oper_job_list[oper_machine] = []
                # if self.update_schedule[(stage, loca_machine)][-1] == selected_job:
                #     oper_job = None
                # else:
                selected_job_index = self.update_schedule[(stage, loca_machine)].index(selected_job)
                if job_flag == 1:
                    # oper_job_list[oper_machine] = self.update_schedule[(stage, loca_machine)][:selected_job_index]
                    #
                    if selected_job_index-1 >= 0:       # 往前插入一个位置
                        oper_job_list[oper_machine] = [self.update_schedule[(stage, loca_machine)][selected_job_index-1]]
                elif job_flag == -1:
                    # if selected_job_index+1 < len(self.update_schedule[(stage, loca_machine)]):
                    if selected_job_index+2 < len(self.update_schedule[(stage, loca_machine)]):
                        oper_job_list[oper_machine]= [self.update_schedule[(stage, loca_machine)][selected_job_index+2]]        # 其实这样不是插入到原位置嘛？？？？？？？？？？？
                else:
                    # oper_job_list[oper_machine] = self.update_schedule[(stage, loca_machine)][:selected_job_index] + self.update_schedule[(stage, loca_machine)][selected_job_index+1:]
                    # 如果是准时工件，尝试往前以及往后插入一个位置
                    if selected_job_index-1 >= 0:
                        oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][selected_job_index-1])
                    # if selected_job_index + 1 < len(self.update_schedule[(stage, loca_machine)]):
                    #     oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][selected_job_index+1])
                    if selected_job_index + 2 < len(self.update_schedule[(stage, loca_machine)]):
                        oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][selected_job_index+2])

                # 如果oper_job_list为空，就不执行！！！

                    # 这里如果oper_job_list为空咋办？？？
                # if len(oper_job_list[oper_machine]) == 0:
                #     if selected_job == self.update_schedule[(stage, loca_machine)][-1]:
                #         if len(self.update_schedule[(stage, loca_machine)])>1:
                #             oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][-2])
                #         else:
                #             oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][-1])
                #     elif selected_job == self.update_schedule[(stage, loca_machine)][0]:
                #         if len(self.update_schedule[(stage, loca_machine)])>1:
                #             oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][1])
                #         else:
                #             oper_job_list[oper_machine].append(self.update_schedule[(stage, loca_machine)][0])


            else:       # 如果是不同的机器上
                oper_machine_list = list(range(self.config.machine_num_on_stage[0]))
                for machine in oper_machine_list:
                    if machine == loca_machine:
                        continue
                    oper_job_list[machine] = []
                    for index, job in enumerate(self.update_schedule[(stage, machine)]):
                        if job_flag == -1:      # 既然是早到工件，去到其他机器上，开始加工时间需要>当前机器上的开始加工时间
                            if (self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) < \
                                    (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][selected_job]):
                                continue
                            else:
                                # oper_job_list[machine] = self.update_schedule[(stage, machine)][index+1:]
                                # if index< len(self.update_schedule[(stage, machine)]):
                                oper_job_list[machine] = [self.update_schedule[(stage, machine)][index]]
                                # if len(oper_job_list[machine]) == 0:
                                #     oper_job_list[machine].append(self.update_schedule[(stage, machine)][-1])
                                # break
                        elif job_flag == 1:
                            if (self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) < \
                                    (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][selected_job]) \
                                    and job!=self.update_schedule[(stage, machine)][-1]:
                                continue
                            elif (self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) < \
                                    (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][selected_job]) \
                                    and job==self.update_schedule[(stage, machine)][-1]:
                                oper_job_list[machine] = [self.update_schedule[(stage, machine)][index]]
                            else:
                                # oper_job_list[machine] = self.update_schedule[(stage, machine)][:index]
                                oper_job_list[machine] = [self.update_schedule[(stage, machine)][index-1]]      # 这里的index-1就应该是对的
                                # if len(oper_job_list[machine]) == 0:
                                #     oper_job_list[machine].append(self.update_schedule[(stage, machine)][0])
                                # break
                        else:
                            if (self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) < \
                                    (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][selected_job]):
                                continue
                            elif (self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) < \
                                    (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][
                                        selected_job]) \
                                    and job == self.update_schedule[(stage, machine)][-1]:
                                oper_job_list[machine].append(self.update_schedule[(stage, machine)][index])

                            else:
                                oper_job_list[machine].append(self.update_schedule[(stage, machine)][index])
                                if index - 1 >0:
                                    oper_job_list[machine].append(self.update_schedule[(stage, machine)][index-1])




                    # if self.job_execute_time[(stage, job)] < self.job_execute_time[(stage, selected_job)]:
                    #     continue
                    # else:
                    #     if job_flag == 1:
                    #         oper_job_list[machine] = self.update_schedule[(stage, loca_machine)][:index]
                    #     else:
                    #         oper_job_list[machine] = self.update_schedule[(stage, loca_machine)][index+1:]
                    #     break
            # oper_machine_list_is_none = True
            # for index, machine in enumerate(oper_job_list.keys()):
            #     if len(oper_job_list[machine]) != 0:
            #         oper_machine_list_is_none = False
            # if oper_machine_list_is_none:
            #     for machine in oper_machine_list:
            #         if machine == loca_machine:
            #             continue
            #         if selected_job == self.update_schedule[(stage, loca_machine)][-1]:
            #             oper_job_list[machine].append(self.update_schedule[(stage, machine)][-1])
            #         if selected_job == self.update_schedule[(stage, loca_machine)][0]:
            #             oper_job_list[machine].append(self.update_schedule[(stage, machine)][0])

        # # 确定被选中的工件属性：
        # # 1. 延误工件往前insert / swap
        # # 2。早到工件往后insert / swap
        # # 3. 准时工件随机insert / swap
        # # 4. 若是第一阶段的工件，全部视为延误工件，往前insert / swap
        # job_flag = self.job_info[selected_job][-1]
        # if stage == 0:
        #     job_flag = 1
        #
        # # 对于另一个要选择的工件
        # # 1。先确定好哪台机器【随机选择，但需要考虑该机器上是否存在工件】
        # # 2。再确定是该机器上的哪个工件
        # while True:
        #     oper_machine = random.choice(list(range(self.config.machine_num_on_stage[stage])))
        #     if self.update_schedule[stage, oper_machine]:
        #         break
        #
        # #
        # start_index = 0
        # end_index = len(self.update_schedule[stage, oper_machine]) - 1
        # if method == 'effe':
        #     for i_index, job in enumerate(self.update_schedule[stage, oper_machine]):
        #         if job_flag == -1 and ((self.job_execute_time[(stage, job)] - self.config.job_process_time[stage][job]) > \
        #                                (self.job_execute_time[(stage, selected_job)] - self.config.job_process_time[stage][
        #                                    selected_job])):
        #             start_index = i_index
        #             end_index = len(self.update_schedule[stage, oper_machine]) - 1
        #             break
        #         elif job_flag == 1 and self.job_execute_time[(stage, job)] > self.job_execute_time[
        #             (stage, selected_job)]:
        #             start_index = 0
        #             end_index = i_index
        #             break
        #     if job_flag == -1:
        #         if end_index:
        #             oper_job_index = random.choice(list(range(start_index, end_index + 1)))
        #         else:
        #             oper_job_index = len(self.update_schedule[stage, oper_machine]) -1
        #     elif job_flag == 1:
        #         if end_index:
        #             oper_job_index = random.choice(list(range(start_index, end_index + 1)))
        #         else:
        #             oper_job_index = 0
        #     else:
        #         oper_job_index = random.choice(list(range(len(self.update_schedule[(stage, oper_machine)]))))
        #
        # else:
        #     start_index = 0
        #     end_index = len(self.update_schedule[stage, oper_machine])
        #     # 如果一个机器上已经空了怎么办，要防止这种情况
        #     while True:
        #         if end_index != 0:
        #             oper_job_index = random.choice(list(range(start_index, end_index)))
        #             break
        #         else:
        #             oper_machine = random.choice(list(range(self.config.machine_num_on_stage[stage])))
        #             end_index = len(self.update_schedule[stage, oper_machine])
        #
        #
        #
        # oper_job = self.update_schedule[(stage, oper_machine)][oper_job_index]

        return loca_machine, oper_job_list

    def chosen_job(self,search_method_1, search_method_2,config_same_machine,stage,oper_method):
        # # 分别在有效搜索 / 随机搜索  的条件下进行选择工件
        self.schedule_ins.idle_time_insertion(self.update_schedule, self.job_execute_time, self.obj)
        job_info = self.schedule_ins.get_job_info(self.job_execute_time)

        selected_job = None
        if search_method_1 == 'effe':
            if stage == 0: # 判断是否造成第二阶段卡住，优先处理前面的卡住
                min_stuck_job = None        # 初始化最先被卡住的工件为None
                min_stuck_time = np.inf         # 初始化最先卡住的工件的最早开始时间为无穷大
                for i_machine in range(self.config.machine_num_on_stage[0]):
                    for i_job in self.update_schedule[(stage,i_machine)]:
                        # 这个时候的self.update_job_execute_time的值还全部是0
                        if i_job != self.update_schedule[(stage,i_machine)][0] and self.job_execute_time[(1,i_job)] - \
                                self.config.job_process_time[1][i_job] == self.job_execute_time[(stage,i_job)]:
                            # 若该工件不是第一个工件，说明这个工件卡住了，选择最前面的卡住工件
                            if self.update_schedule[(stage,i_machine)][0] and self.job_execute_time[(stage,i_job)] < min_stuck_time:
                                min_stuck_time = self.job_execute_time[(stage,i_job)]
                                min_stuck_job = i_job
                if min_stuck_job or min_stuck_job == 0:
                    selected_job = min_stuck_job
                # else:
                #     selected_job = 0        # 防错，如果不存在则选择工件0，这个防错不要，如果没有则不执行

            # 若是第二阶段
            else:
                if search_method_2 == 'DRM':
                    values = [job_info[job][0] if job_info[job][-1] == 1 else 0 for job in range(self.config.jobs_num)]
                    values = [val if val != 0 else 1e-9 for val in values]
                    probabilities = np.array(values) / np.sum(values)  # 归一化概率
                    selected_job = np.random.choice(self.hfs.job_list, p=probabilities)
                elif search_method_2 == 'ERM':
                    values = [job_info[job][0] if job_info[job][-1] == -1 else 0 for job in range(self.config.jobs_num)]
                    values = [val if val != 0 else 1e-9 for val in values]
                    probabilities = np.array(values) / np.sum(values)  # 归一化概率
                    selected_job = np.random.choice(self.hfs.job_list, p=probabilities)
                else:
                    if search_method_2[0] == 'D':
                        values = [job_info[job][0] if job_info[job][-1] == 1 else 0 for job in
                                  range(self.config.jobs_num)]
                    else:
                        values = [job_info[job][0] if job_info[job][-1] == -1 else 0 for job in
                                  range(self.config.jobs_num)]
                    selected_job = values.index(max(values))
        elif search_method_1 == 'dire':
            all_jobs = []
            all_jobs_info = []
            # 在第二阶段的除最后一个工件的所有工件中：
            if search_method_2 == 'dweight':
                for machine in range(self.config.machine_num_on_stage[0]):
                    all_jobs += self.update_schedule[(1,machine)][:len(self.update_schedule[(1,machine)])-1]
                for job in all_jobs:
                    # 这里all_jobs_info里面的信息[工件、两个阶段的加工时间之和、延误权重]
                    all_jobs_info.append((job,self.config.job_process_time[0][job] + self.config.job_process_time[1][job],self.config.ddl_weight[job]))
                all_jobs_info = sorted(all_jobs_info, key=lambda x: x[1],reverse=True)       # 按照第一阶段+第二阶段的加工时间，降序
                all_jobs_info = sorted(all_jobs_info, key=lambda x: x[-1])      # 按照延误权重升序
                selected_job = all_jobs_info[0][0]

            else:
                for machine in range(self.config.machine_num_on_stage[0]):
                    all_jobs += self.update_schedule[(1,machine)][1:]
                for job in all_jobs:
                    all_jobs_info.append((job,self.config.job_process_time[0][job] + self.config.job_process_time[1][job],self.config.ect_weight[job]))
                all_jobs_info = sorted(all_jobs_info, key=lambda x: x[1])       # 按照第一阶段+第二阶段的加工时间，升序
                all_jobs_info = sorted(all_jobs_info, key=lambda x: x[-1])      # 按照延误权重升序
                selected_job = all_jobs_info[0][0]

        else:
            selected_job = np.random.choice(self.hfs.job_list)
        if selected_job is None:
            loca_machine, oper_job_list = None,None
        else:
            loca_machine, oper_job_list = self.chosen_job2_oper(selected_job, stage, search_method_1, config_same_machine,
                                                                oper_method)

        # 选定工件之后选择另一个需要操作的工件
        # loca_machine, oper_machine, oper_job = self.chosen_job2_oper(selected_job, stage, search_method_2)

        '''
            选择并确定工件
        '''

        # # 这里画个图
        # from SS_RL.diagram import job_diagram
        # import matplotlib.pyplot as plt
        # dia = job_diagram(self.update_schedule,self.job_execute_time,self.file_name,1)
        # dia.pre()
        # plt.savefig('./img1203/pic-{}.png'.format(int(1)))

        # self.schedule_ins.idle_time_insertion(self.update_schedule, self.job_execute_time, self.obj)
        # selected_job = None
        # if search_method_1 == 'rand':
        #     selected_job = np.random.choice(self.hfs.job_list)
        # else:
        #     if search_method_2 == 'stuck':
        #         stuck_job = {}
        #         # 找到所有卡住的工件，根据每单位往前移动的贡献，来权衡
        #         for i in self.schedule_ins.schedule_job_block.keys():
        #             for j in self.schedule_ins.schedule_job_block[i]:
        #                 if j == self.schedule_ins.schedule_job_block[0]:
        #                     continue
        #                 if len(j[0][0]):
        #                     stuck_job[j[0][0][0]] = j[0][2]
        #         sorted_dict = dict(sorted(stuck_job.items(), key=lambda item: item[1], reverse=True))
        #         selected_job = list(sorted_dict.keys())[0]
        #
        #     else:
        #         # 找到可以单纯可以改善最多的方向，
        #         max_delay_weight = 0
        #         max_early_weight = 0
        #         add_delay_weight = np.inf
        #         add_early_weight = np.inf
        #         early_exceute_block = []
        #         delay_exceute_block = []
        #         find_early_job = True
        #         need_excute_block = []
        #         selected_job = None
        #         if search_method_2[0] == 'I':
        #             for i in self.schedule_ins.schedule_job_block.keys():
        #                 for j in self.schedule_ins.schedule_job_block[i]:
        #                     if j[0][1] > max_early_weight:
        #                         max_early_weight = j[0][1]
        #                         early_exceute_block = j[0][0]
        #                     if j[0][3] > max_delay_weight:
        #                         max_delay_weight = j[0][3]
        #                         delay_exceute_block = j[0][0]
        #             if max_early_weight > max_delay_weight:
        #                 need_excute_block = early_exceute_block
        #             else:
        #                 need_excute_block = delay_exceute_block
        #                 find_early_job = False
        #         elif search_method_2[0] == 'A':
        #             for i in self.schedule_ins.schedule_job_block.keys():
        #                 for j in self.schedule_ins.schedule_job_block[i]:
        #                     if j[0][2] < add_early_weight:
        #                         add_early_weight = j[0][2]
        #                         early_exceute_block = j[0][0]
        #                     if j[0][4] < add_delay_weight:
        #                         add_delay_weight = j[0][4]
        #                         delay_exceute_block = j[0][0]
        #             if add_early_weight > add_delay_weight or add_early_weight <= 0 or add_early_weight <= 0:
        #                 need_excute_block = delay_exceute_block
        #                 find_early_job = False
        #             else:
        #                 need_excute_block = early_exceute_block
        #         # 在选中的工件块中，选择最好的权重最大的工件块
        #         if find_early_job:
        #             early_job = []
        #             for job in need_excute_block:
        #                 if self.job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
        #                     early_job.append([job,self.config.ect_weight[job],self.config.job_process_time[stage][job]])
        #
        #             if len(early_job):
        #                 if search_method_2[1] == 'F':
        #                     selected_job = early_job[-1][0]
        #
        #                 elif search_method_2[1] == 'W':
        #                     if len(early_job) == 1:
        #                         selected_job = early_job[0][0]
        #                     else:
        #                         selected_job = sorted(early_job, key=lambda x: x[1], reverse=True)[0][0]
        #
        #                 elif search_method_2[1] == 'P':
        #                     if len(early_job) == 1:
        #                         selected_job = early_job[0][0]
        #                     else:
        #                         selected_job = sorted(early_job, key=lambda x: x[2])[0][0]
        #
        #         else:
        #             delay_job = []
        #             for job in need_excute_block:
        #                 if self.job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
        #                     delay_job.append([job,self.config.ddl_weight[job],self.config.job_process_time[stage][job]])
        #             if len(delay_job):
        #                 if search_method_2[1] == 'F':
        #                     selected_job = delay_job[0][0]
        #
        #                 elif search_method_2[1] == 'W':
        #                     if len(delay_job) == 1:
        #                         selected_job = delay_job[0][0]
        #                     else:
        #                         selected_job = sorted(delay_job, key=lambda x: x[1], reverse=True)[0][0]
        #
        #                 elif search_method_2[1] == 'P':
        #                     if len(delay_job) == 1:
        #                         selected_job = delay_job[0][0]
        #                     else:
        #                         selected_job = sorted(delay_job, key=lambda x: x[2])[0][0]
        #
        #
        # if selected_job is None:
        #     self.schedule_ins.idle_time_insertion(self.schedule, self.job_execute_time, self.obj)
        #     selected_job = 0

        # loca_machine, oper_job_list = self.chosen_job2_oper(selected_job, stage, search_method_1,config_same_machine,oper_method)

        return loca_machine, selected_job, oper_job_list

    def sort_(self,search_method_1, job_block_rule,A_or_D, stage,sort_rule):
        # 找到所有的延误片断：
        self.schedule_ins.idle_time_insertion(self.update_schedule, self.job_execute_time, self.obj)
        job_info = self.schedule_ins.get_job_info(self.job_execute_time)
        all_early_job_list = []
        all_delay_job_list = []
        all_stuck_job_list = []
        need_excute_job = []
        for machine in range(self.config.machine_num_on_stage[0]):
            early_job_list = []
            delay_job_list = []
            early_penalty_value = 0
            delay_penalty_value = 0
            stuck_job_list = []

            for job in self.update_schedule[(1,machine)]:
                job_falg = job_info[job][-1]
                if job_falg == -1:
                    early_job_list.append(job)
                    early_penalty_value += self.config.ect_weight[job]
                    if len(delay_job_list) > 1:
                        all_delay_job_list.append((delay_job_list,machine,delay_penalty_value))
                    delay_job_list.clear()
                    delay_penalty_value = 0
                elif job_falg == 1:
                    delay_job_list.append(job)
                    delay_penalty_value += self.config.ddl_weight[job]
                    if len(early_job_list)>1:
                        all_early_job_list.append((early_job_list,machine,early_penalty_value))
                    early_job_list.clear()
                    early_penalty_value = 0
                else:
                    if len(early_job_list) > 1:
                        all_early_job_list.append((early_job_list,machine,early_penalty_value))
                    early_job_list.clear()
                    early_penalty_value = 0
                    if len(delay_job_list) > 1:
                        all_delay_job_list.append((delay_job_list,machine,delay_penalty_value))
                    delay_job_list.clear()
                    delay_penalty_value = 0
                # 如果工件的在第二阶段的开始加工时间 == 在第一阶段的结束加工时间，则将该工件添加到列表，
                if job_block_rule == 'stuck':
                    pre_job_index = self.update_schedule[(1, machine)].index(job) - 1
                    pre_job = self.update_schedule[(1, machine)][pre_job_index]
                    if len(stuck_job_list) == 0 and self.job_execute_time[(0, job)] == (self.job_execute_time[(1, job)] - self.config.job_process_time[stage][job]):
                        stuck_job_list.append(job)
                    elif len(stuck_job_list) != 0 and pre_job_index >= 0 and  self.job_execute_time[(1, pre_job)] \
                        == (self.job_execute_time[(1, job)] - self.config.job_process_time[stage][job]):
                        stuck_job_list.append(job)
                        if job == self.update_schedule[(1,machine)][-1]:
                            ect_value = 0
                            ddl_value = 0
                            for job in stuck_job_list:
                                job_makespan = self.job_execute_time[(self.config.stages_num - 1, job)]
                                if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                                    ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
                                elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                                    ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]
                            block_obj = ect_value + ddl_value
                            list_new = stuck_job_list.copy()
                            all_stuck_job_list.append((list_new, machine, block_obj))
                    else:
                        if len(stuck_job_list) > 1:
                            ect_value = 0
                            ddl_value = 0
                            for job in stuck_job_list:
                                job_makespan = self.job_execute_time[(self.config.stages_num - 1, job)]
                                if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                                    ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
                                elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                                    ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]

                            block_obj = ect_value + ddl_value
                            list_new = stuck_job_list.copy()
                            all_stuck_job_list.append((list_new,machine,block_obj))

                        stuck_job_list.clear()



        # action是针对，早到还是延误：
        # 如果是针对早到：
        if job_block_rule == 'early':
            if len(all_early_job_list) != 0:

                all_early_job_list = sorted(all_early_job_list, key=lambda x: x[-1],reverse=True)   # 降序
                # 判断是哪一种类型的排序【加工时间，交付期窗口，权重】
                need_excute_job = all_early_job_list[0][0]
                machine = all_early_job_list[0][1]
        elif job_block_rule == 'delay':
            if len(all_early_job_list) != 0:
                all_early_job_list = sorted(all_early_job_list, key=lambda x: x[-1],reverse=True)  # 降序
                # 判断是哪一种类型的排序【加工时间，交付期窗口，权重】
                need_excute_job = all_early_job_list[0][0]
                machine = all_early_job_list[0][1]
        elif job_block_rule == 'stuck':
            # 还有一种是按照卡住的工件块
            if len(all_stuck_job_list) != 0:
                all_stuck_job_list = sorted(all_stuck_job_list, key=lambda x: x[-1], reverse=True)  # 降序
                need_excute_job = all_stuck_job_list[0][0]
                machine = all_stuck_job_list[0][1]


        if need_excute_job:
            first_job_index = self.update_schedule[(1, machine)].index(need_excute_job[0])
            last_job_index = self.update_schedule[(1, machine)].index(need_excute_job[-1])
            sort_list = []
            if sort_rule == 'P':
                for job in need_excute_job:
                    sort_list.append((job,self.config.job_process_time[stage][job]))
            elif sort_rule == 'D':
                if job_block_rule == 'delay':
                    for job in need_excute_job:
                        sort_list.append((job,self.config.ddl_windows[job]))
                elif job_block_rule == 'early':
                    for job in need_excute_job:
                        sort_list.append((job,self.config.ect_windows[job]))
            elif sort_rule == 'W':
                if job_block_rule == 'delay':
                    for job in need_excute_job:
                        sort_list.append((job,self.config.ddl_weight[job]))
                elif job_block_rule == 'early':
                    for job in need_excute_job:
                        sort_list.append((job,self.config.ect_weight[job]))
            elif sort_rule == 'S0':
                # 根据第一阶段的结束加工时间去排：
                for job in need_excute_job:
                    sort_list.append((job,self.job_execute_time[(0,job)]))


            des_sort = False
            if A_or_D == 'D':
                des_sort = True

            sort_list = sorted(sort_list, key=lambda x: x[-1],reverse = des_sort)    # 升序还是降序
            new_job_list = [item[0] for item in sort_list]
            self.update_schedule[(1, machine)][first_job_index:last_job_index+1] = new_job_list
            self.re_cal(self.update_schedule)

            # 更新目标值
            self.update_obj = self.schedule_ins.cal(self.update_job_execute_time)

            # 空闲插入邻域，更新工件完工时间、目标值
            self.update_schedule, self.update_job_execute_time, self.update_obj = self.schedule_ins.idle_time_insertion(
                self.update_schedule, self.update_job_execute_time, self.update_obj)


        return self.update_schedule, self.update_obj, self.update_job_execute_time



    def search_opea(self,oper_method,obj,stage, loca_machine, selected_job, oper_machine, oper_job,search_method_1,insert_last):
        # self.re_cal()
        # print(self.update_job_execute_time)
        # print(self.update_schedule)
        #
        # oj = self.hfs.cal(self.update_job_execute_time)
        # print('重新计算目标值：{0}'.format(oj))
        # self.diagram = diagram.job_diagram(self.update_schedule,self.update_job_execute_time,'1')
        # self.diagram.pre()



        if selected_job != oper_job:
            '''
            insert操作
            '''
            if oper_method == 'insert':
                self.insert_opera(stage, loca_machine, selected_job, oper_machine, oper_job,search_method_1,insert_last)
                # 更新工件完工时间
                self.re_cal(self.update_schedule)

                # 更新目标值
                self.update_obj = self.schedule_ins.cal(self.update_job_execute_time)

                # 空闲插入邻域，更新工件完工时间、目标值
                self.update_schedule, self.update_job_execute_time, self.update_obj = self.schedule_ins.idle_time_insertion(
                    self.update_schedule, self.update_job_execute_time, self.update_obj)




            '''
            swap操作
            '''
            if oper_method == 'swap':
                self.swap_opera(stage, loca_machine, selected_job, oper_machine, oper_job)

                # 更新工件完工时间
                self.re_cal(self.update_schedule)

                # 更新目标值
                self.update_obj = self.schedule_ins.cal(self.update_job_execute_time)

                # 空闲插入邻域，更新工件完工时间、目标值
                self.update_schedule, self.update_job_execute_time, self.update_obj = self.schedule_ins.idle_time_insertion(
                    self.update_schedule, self.update_job_execute_time, self.update_obj)


        else:
            self.update_obj = obj

        if stage == 1:
            # 这里调用那个函数：
            self.sort_stage0()
            self.re_cal(self.update_schedule)

        # 更新目标值
        self.update_obj = self.schedule_ins.cal(self.update_job_execute_time)

        # 空闲插入邻域，更新工件完工时间、目标值
        self.update_schedule, self.update_job_execute_time, self.update_obj = self.schedule_ins.idle_time_insertion(
            self.update_schedule, self.update_job_execute_time, self.update_obj)


        # # if oper_method == 'i':
        # #     if selected_job != oper_job:
        # #         self.insert_opera(stage, loca_machine, selected_job, oper_machine, oper_job)
        # # else:
        # #     if selected_job != oper_job:
        # #         self.swap_opera(stage, loca_machine, selected_job, oper_machine, oper_job)
        #
        # # 更新工件完工时间
        # self.re_cal(self.update_schedule)
        #
        # # 更新目标值
        # self.update_obj = self.hfs.cal(self.update_job_execute_time)
        #
        # # 空闲插入邻域，更新工件完工时间、目标值
        # self.update_schedule, self.update_job_execute_time, self.update_obj = self.hfs.idle_time_insertion(
        #     self.update_schedule, self.update_job_execute_time, self.update_obj)

        return self.update_schedule,self.update_obj,self.update_job_execute_time


    def sort_stage0(self):
        # 获取第二阶段的工件的开始时间的排序：
        all_job_info = []
        for machine in range(self.config.machine_num_on_stage[0]):
            for job in self.update_schedule[(1,machine)]:
                all_job_info.append((job,self.update_job_execute_time[(1,job)] - self.config.job_process_time[1][job]))
        # 将工件按照在第二阶段的顺序进行升序排序
        all_job_info = sorted(all_job_info, key=lambda x: x[-1], reverse=False)     # 这个还要判断一下是升序还是降序
        job_sort = [item[0] for item in all_job_info]

        for machine in range(self.config.machine_num_on_stage[0]):
            self.update_schedule[(0,machine)] = []

        for job in job_sort:
            machine_end_time = np.inf
            chosen_machine = None
            chosen_machine_pre_job = None
            for machine in range(self.config.machine_num_on_stage[0]):
                if self.update_schedule[(0,machine)]:
                    pre_job = self.update_schedule[(0,machine)][-1]
                    if self.update_job_execute_time[(0,pre_job)] < machine_end_time:
                        machine_end_time = self.update_job_execute_time[(0,pre_job)]
                        chosen_machine = machine
                        chosen_machine_pre_job = pre_job
                else:
                    chosen_machine = machine
                    chosen_machine_pre_job = None
                    break
            self.update_schedule[(0,chosen_machine)].append(job)
            if chosen_machine_pre_job is None:
                self.update_job_execute_time[(0,job)] = self.config.job_process_time[0][job]
            else:
                self.update_job_execute_time[(0, job)] = self.config.job_process_time[0][job] + self.update_job_execute_time[(0, chosen_machine_pre_job)]





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

