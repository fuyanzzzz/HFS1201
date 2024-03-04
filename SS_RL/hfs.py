'''
-------------------------------------------------
File Name: hfs.py
Author: LRS
Create Time: 2023/2/10 15:45
-------------------------------------------------
'''
import time
import os
import torch
import numpy as np
import random
import copy
from SS_RL.public import AllConfig



data_folder = "data"  # 数据文件夹的路径
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]



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


    job_makespan = job_completion_time[stages_num - 1]
    ect_value = 0
    ddl_value = 0
    for i in range(len(job_makespan)):
        if job_makespan[i] < ect_windows[i]:  # 早前权重值
            ect_value += max(ect_windows[i] - job_makespan[i], 0) * ect_weight[i]
        elif job_makespan[i] > ddl_windows[i]:  # 延误权重值
            ddl_value += max(job_makespan[i] - ddl_windows[i], 0) * ddl_weight[i]

    return ect_value + ddl_value,job_completion_time


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


def job_assignment(job_sort,start_stage = 0,end_stage = 2,job_sort_on_stages = None):
    '''
    传入的参数：6个
    1. 第一阶段的工件加工顺序
    2. 开始调度阶段，结束调度阶段
    3. 所有阶段上的工件加工顺序

    传出的参数：4个
    1. 三个更新的变量
    2. 当前的目标值【未进行空闲插入程序】
    '''

    # 初始化清空变量
    schedule,machine_completion_time,job_completion_time = intal_variable()

    # 从指定阶段开始，根据上一个阶段排列出的 job_sort 将工件调配到该阶段的各台机器上
    for stage in range(start_stage,end_stage):
        # 如果已知每个阶段的工件加工顺序，则直接从字典中读取
        if job_sort_on_stages:
            job_sort = job_sort_on_stages[stage]
        # 否则就按照工件的完工时间排序
        elif stage != 0 and start_stage != stage:
            job_sort = np.argsort(job_completion_time[stage - 1])
        # 剩下一种情况是第一个阶段，则直接使用传入的job_sort
        for job in job_sort:
            obj = []                        # 目标值：可删
            on_machines = []                # 用于记录阶段g的机器索引
            ect_rule = []                   # ect规则，用于确定工件在阶段g的哪台机器上进行加工
            for machine in range(machine_num_on_stage[stage]):     # 从第一个阶段开始
                # machine = machines + sum(machine_num_on_stage[:stage])

                # 如果是第一个阶段的
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

                ect_rule.append(ect_value)
                on_machines.append(machine)


            min_ect_rule = min(ect_rule)
            index = ect_rule.index(min_ect_rule)
            machine = on_machines[index]
            job_completion_time[stage][job] = max(0,min_ect_rule) + job_process_time[stage][job]
            schedule[(stage,machine)].append(job)
            # schedule[machine, list(schedule[machine]).index(-1)] = job
            machine_completion_time[(stage,machine)][0] = job
            machine_completion_time[(stage,machine)][1] = max(0,min_ect_rule) + job_process_time[stage][job]

    job_makespan = job_completion_time[stages_num - 1]
    ect_value = 0
    ddl_value = 0
    for i in range(len(job_makespan)):
        if job_makespan[i] < ect_windows[i]:  # 早前权重值
            ect_value += max(ect_windows[i] - job_makespan[i], 0) * ect_weight[i]
        elif job_makespan[i] > ddl_windows[i]:  # 延误权重值
            ddl_value += max(job_makespan[i] - ddl_windows[i], 0) * ddl_weight[i]

    current_obj = ect_value + ddl_value

    return schedule, job_completion_time, machine_completion_time,current_obj


def get_last_stage_job(schedule):
    last_stage_job = []
    for key in schedule:
        if key[0] == (stages_num - 1):
            last_stage_job.append(schedule[key])

    return last_stage_job


def inintal_solution():
    # 1.EDD：根据工件的ddl进行排序
    job_sort_edd = np.argsort(ddl_windows)
    schedule, job_completion_time, machine_completion_time,current_obj = job_assignment(job_sort_edd)

    improve_obj,all_job_block = idle_time_insertion(schedule)
    print(schedule,current_obj,improve_obj,job_sort_edd)
    job_sort_inital = job_sort_edd

    # 2.LSL 工件的ddl - 工件最后一个阶段的加工时间
    a = np.array(ddl_windows) - np.array(job_process_time[-1])
    job_sort_lsl = np.argsort(a)      # 根据ddl或者工件的调度顺序
    schedule_lsl, job_completion_time, machine_completion_time,obj_lsl = job_assignment(job_sort_lsl)
    improve_lsl_obj,all_job_block = idle_time_insertion(schedule_lsl)
    print(schedule_lsl,obj_lsl,improve_lsl_obj,job_sort_lsl)
    if improve_lsl_obj < improve_obj:
        improve_obj = improve_lsl_obj
        schedule = schedule_lsl
        job_sort_inital = job_sort_lsl

    # 3.OSL 考虑工件的ddl-工件在所有阶段的完工时间得到的数，再进行一个排列
    process_time_all_stages = []
    for i in range(jobs_num):
        process_time_all_stages.append(sum([j[i] for j in job_process_time]))
    b = np.array(ddl_windows) - np.array(process_time_all_stages)
    job_sort_osl = np.argsort(b)  # 根据ddl或者工件的调度顺序
    schedule_osl, job_completion_time, machine_completion_time,obj_osl = job_assignment(job_sort_osl)
    improve_osl_obj, all_job_block = idle_time_insertion(schedule_osl)
    print(schedule_osl,obj_osl,improve_osl_obj,job_sort_osl)
    obj = cal(schedule_osl)
    print(obj)
    if improve_osl_obj < improve_obj:
        improve_obj = improve_osl_obj
        schedule = schedule_osl
        job_sort_inital = job_sort_osl

    return schedule,improve_obj,job_sort_inital


# 重构，将移除的工件插入到剩余工件中
def destruction_reconstruction(job_sort_inital,d_list):
    remove_jobs_num = np.random.choice(d_list)  # 得到需要移除的工件个数
    # 通过抽取d个工件，分为移除序列remove_jobs和剩余序列surplus_jobs
    job_sort_inital = list(job_sort_inital)
    remove_jobs = random.sample(list(job_sort_inital), remove_jobs_num)  # 得到移除的工件序列
    for job in remove_jobs:
        job_sort_inital.remove(job)
    # surplus_jobs = list(set(job_sort_inital) - set(remove_jobs))
    best_schedule = None
    best_obj = None
    for r_job in remove_jobs:
        best_insert_index = None
        pos = 0
        k = 1
        best_obj = np.inf
        while pos <= len(job_sort_inital):
            if pos == len(job_sort_inital):
                insert_index = pos+1
            else:
                insert_index = pos
            job_sort_inital.insert(insert_index, r_job)
            cur_schedule, job_completion_time, machine_completion_time,cur_obj = job_assignment(job_sort_inital)
            cur_obj, all_job_block = idle_time_insertion(cur_schedule)      # 每进行一次重构，也需要进行空闲插入程序
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_schedule = cur_schedule
                best_insert_index = insert_index
                k = 1
            else:
                k += 1
            job_sort_inital.remove(r_job)
            pos += k
        job_sort_inital.insert(best_insert_index, r_job)

    return job_sort_inital,best_schedule,best_obj


def local_search(jobs_sort,best_obj,recon_schedule):
    # best_schedule,job_completion_time, machine_completion_time, best_obj = job_assignment(jobs_sort)        # ！！！！有问题
    best_schedule = recon_schedule
    improvement = True
    while improvement == True:
        improvement = False
        for job in jobs_sort:
            ori_index = jobs_sort.index(job)
            jobs_sort.remove(job)
            random_insert = np.random.choice(range(len(jobs_sort)+1))
            jobs_sort.insert(random_insert,job)
            cur_schedule,job_completion_time, machine_completion_time, cur_obj = job_assignment(jobs_sort)
            cur_obj, all_job_block = idle_time_insertion(cur_schedule)      # 每进行一次重构，也需要进行空闲插入程序
            if cur_obj < best_obj:
                improvement = True
                best_obj = cur_obj
                best_schedule = cur_schedule
                break
            jobs_sort.remove(job)
            jobs_sort.insert(ori_index, job)
    return best_schedule,best_obj,jobs_sort


def acceptance_criterion(jobs_sort,local_obj,ts):
    final_obj = local_obj

    # 这个只是第一代的时候先赋值无穷大
    if len(history_obj) == 0:
        best_obj = np.inf
    else:
        best_obj = min(history_obj)
        index = history_obj.index(best_obj)

        best_jobs_sort = history_schedule[index]        # 这里要存的
    # 接受标准

    if final_obj < best_obj:
        #　清空历史序列和历史目标值
        history_schedule.clear()
        history_obj.clear()

        # 把当前最优的序列和目标值存入列表中
        history_schedule.append(jobs_sort)
        history_obj.append(final_obj)

        ts_list = [2,3,4,5,6,7]
        ts = np.random.choice(ts_list)
        cur_seqence = jobs_sort
        cur_obj = final_obj
    else:
        # 若新序列没有更优，不清空列表，直接将得到的新序列添加到列表中
        history_schedule.append(jobs_sort)
        history_obj.append(final_obj)
        #　计算history_obj 列表中的平均值
        avg_history_obj = sum(history_obj)/len(history_obj)
        if final_obj < avg_history_obj:
            ts = min(ts+1,7)        # ts在前面的哪里有先定义嘛,理解一下这个操作
        else:
            ts = max(ts-1,2)
        if ts > len(history_schedule):
            cur_seqence = random.choice(history_schedule)
            index = history_schedule.index(cur_seqence)
            cur_obj = history_obj[index]
        else:
            choice_list = random.sample(history_schedule,ts)
            index = history_schedule.index(choice_list[0])
            cur_seqence = choice_list[0]
            cur_obj = history_obj[index]
            for seqence in choice_list:
                index = history_schedule.index(choice_list[0])
                obj = history_obj[index]
                if obj < cur_obj:
                    cur_obj = obj
                    cur_seqence = seqence
    return cur_seqence,ts,cur_obj,history_obj,history_schedule


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
                if job_makespan[job] < ect_windows[job]:
                    early_job.append(job)
                elif job_makespan[job] >= ddl_windows[job]:
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


def find_limited_neighbors(job_sort_on_stages,stage):
    '''
    传入的参数：5个
    1. 待调度的工件顺序
    2. 三个待更新的变量
    3. 需要探索邻域的阶段

    传出的参数：1个
    1. limited_neighbors
    '''
    # 传入的参数： 需调度的工件顺序， 未排完的schedule， 未排完的各阶段工件完工时间，  开始阶段
    # 返回的变量： 已排完的schedule， 目标值， 工件在最后一个阶段的完工时间

    schedule, job_completion_time, machine_completion_time, current_obj = job_assignment(job_sort_on_stages[0], end_stage=stage,job_sort_on_stages=job_sort_on_stages)

    limited_neighbors = []
    job_sort_on_stages[stage] = np.argsort(job_completion_time[stage-1])
    job_sort = list(job_sort_on_stages[stage])
    for job in job_sort:    # 根据上一个阶段的工件序列进行调度
        obj = []  # 目标值：可删
        on_machines = []  # 用于记录阶段g的机器索引
        ect_rule = []  # ect规则，用于确定工件在阶段g的哪台机器上进行加工
        machine_availability_time = []

        for machine in range(machine_num_on_stage[stage]):  # 在指定阶段上搜索邻域，从第二阶段开始
            # 获取下一个工件
            index = job_sort.index(job)
            if index <len(job_sort)-1:
                swap_job = job_sort[index+1]

            # 判断该阶段上所有机器的完工时间，找到最早可用的机器，并获得其最早可用时间

            # 获取该阶段上的最早可用机器
            machine_availability_time.append(machine_completion_time[(stage, machine)][-1])


            # 执行正常的调度，前面只是收集邻域
            # 需要确定工件i在哪台机器上进行加工，工件i在该阶段的所有机器上都试一下，得到在机器上的ect_value
            pro_job = machine_completion_time[(stage, machine)][0]      # 更新变量
            # 如果非第一阶段，但是该机器上的第一个工件
            if pro_job == -1:
                # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b = 0】
                job_on_pro_machine = job_completion_time[stage - 1][job]
                ect_value = max(machine_completion_time[(stage, machine)][1], job_on_pro_machine) + 0

            else:  # 非第一阶段，且非该机器上的第一个工件
                # ect = max（该机器上最后一个工件的完工时间a，该工件在上一个阶段上的完工时间b） + 切换时间c  【这里a,b ！＝ 0】
                job_on_pro_machine = job_completion_time[stage - 1][job]
                ect_value = max(job_on_pro_machine, machine_completion_time[(stage, machine)][1]) + \
                            setup_time[stage][pro_job][
                                job]  # 机器最早可开始加工时间 = 上个工件的完工时间 + 切换时间

            # 根据在每机器上的最早完工时间，选择工件的加工机器
            ect_rule.append(ect_value)
            on_machines.append(machine)

        # ⭐⭐这里的判定是最早可用机器，没有将切换时间包含在内，目前认为这里不需要考虑切换，扩宽解的空间
        earliest_availability_time = min(machine_availability_time)
        # 判断下一个待调度的工件在上一个阶段的完工时间 是否小于 该阶段机器的最早可用机器时间
        while job_completion_time[stage - 1][swap_job] < earliest_availability_time:
            neighbor = copy.copy(job_sort)  # 深拷贝
            # 交换算一个邻域
            job_index = neighbor.index(job)
            swap_job_index = neighbor.index(swap_job)
            neighbor[job_index], neighbor[swap_job_index] = neighbor[swap_job_index], neighbor[
                swap_job_index]  # 这个语法需要验证一下

            limited_neighbors.append(neighbor)  # 将该列表加入邻域
            swap_job_index += 1
            if swap_job_index >= len(job_sort):
                break
            else:
                swap_job = job_sort[swap_job_index]

        min_ect_rule = min(ect_rule)
        index = ect_rule.index(min_ect_rule)
        machine = on_machines[index]

        # 一共需要更新三个变量，共4行代码
        job_completion_time[stage][job] = max(0, min_ect_rule) + job_process_time[stage][job]       # 更新变量
        schedule[(stage, machine)].append(job)          # 更新变量
        # 更新变量
        machine_completion_time[(stage, machine)][0] = job
        machine_completion_time[(stage, machine)][1] = max(0, min_ect_rule) + job_process_time[stage][job]

    return limited_neighbors



def limited_local_search(job_sort,best_obj,best_schedule):

    # 这里浅浅地记录一下每个阶段的工件开始调度顺序
    job_sort_on_stages = {}
    job_sort_on_stages[0] = job_sort

    # 进入第二阶段的local search，需要记录之前的序列，
    # 把邻域和第二阶段的序列进行比较，如果更优，就替换掉第几个阶段的工件排序
    # 还要继续思考这个代码逻辑应该怎么改

    # 从第二阶段开始，找到每个阶段的有限邻域
    for stage in range(1,stages_num):
        # 获得该阶段上的有限邻域
        limited_neighbors = find_limited_neighbors(job_sort_on_stages,stage)
        # 如果该阶段上有邻域，则执行
        if len(limited_neighbors) > 0:
            # 遍历执行该阶段上的所有邻域
            chosen_sequence = None
            for sequence in limited_neighbors:
                job_sort_on_stages[stage] = sequence
                # 进行工件调度,注意这里的new_schedule是全部阶段的schedule
                new_schedule, job_completion_time, machine_completion_time, current_obj = job_assignment(job_sort_on_stages)              # 如果job_assignment这个函数需要从指定阶段开始，需要传入上一个阶段的工件完工时间
                # 进行空闲插入邻域搜索
                new_obj,all_job_block = idle_time_insertion(new_schedule)

                if new_obj < best_obj:      # 这里得到的东西都是已经到了最后一个阶段的东西，
                    best_schedule = new_schedule
                    job_sort = np.argsort(job_completion_time[stage])   # 确定下一个阶段的工件调度顺序
                    chosen_sequence = np.copy.copy(sequence)
            job_sort_on_stages[stage] = chosen_sequence

        else:
            best_schedule, job_completion_time, machine_completion_time, current_obj = job_assignment(job_sort_on_stages,end_stage=stage+1)
            job_sort = np.argsort(job_completion_time[stage])
            job_sort_on_stages[stage] = job_sort
    cur_obj, all_job_block = idle_time_insertion(best_schedule)  # 每进行一次重构，也需要进行空闲插入程序

    return best_schedule,cur_obj



if __name__ == '__main__':
    # 初始解

    for index, file_name in enumerate(txt_files):
        config = AllConfig.get_config(file_name)
        job_process_time = config.job_process_time
        ect_weight = config.ect_weight
        ddl_weight = config.ddl_weight
        ddl_windows = config.ddl_windows
        ect_windows = config.ect_windows
        jobs_num = config.jobs_num
        jobs = list(range(jobs_num))
        stages_num = config.stages_num
        total_mahcine_num = config.total_mahcine_num

        setup_time = {}

        setup_time[0] = np.zeros((10, 10))
        setup_time[1] = np.zeros((10, 10))

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

        # 声明一个变量，用于存储每个阶段的工件调度顺序【用于第二阶段的local search】
        job_sort_on_satges = {}

        '''
        以上是变量初始化和声明
        '''

        t1 = time.time()
        history_schedule = []
        history_obj = []

        # 生成初始解
        inital_schedule,intal_obj,job_sort_inital= inintal_solution()
        # 解构，划分出两个移除工件序列，剩余工件序列
        d_list = [1,2,3,4]   #  初始解d_list 列表
        i = 0
        ts = None
        total_time = 0
        while total_time <= 0.6:
            print('解构重构')
            # 解构重构: 这里只对第一阶段上的工件排序进行调整
            jobs_sort,recon_schedule,recon_obj = destruction_reconstruction(job_sort_inital,d_list)
            # local search：这里也是只对第一阶段的工件排序进行调整
            print('local search')
            local_schedule,local_obj,jobs_sort = local_search(jobs_sort,recon_obj,recon_schedule)
            # 第二阶段的local search：这里会对其他阶段的工件排序造成影响
            print('第二阶段local search')
            schedule,obj = limited_local_search(jobs_sort,local_obj,local_schedule)     # 目标值输出
            print('目标值：{0}'.format(local_obj))
            # 接受标准
            print('接受标准')
            job_sort_inital,ts,cur_obj,history_obj,history_schedule = acceptance_criterion(jobs_sort,local_obj,ts)
            i += 1
            t2 = time.time()
            total_time = t2 - t1
            print(total_time)
            print(i)

        best_obj = min(history_obj)
        index = history_obj.index(best_obj)
        best_schedule = history_schedule[index]
        print(best_schedule)
        schedule, job_completion_time, machine_completion_time, current_obj = job_assignment(best_schedule)
        improve_obj, all_job_block = idle_time_insertion(schedule)
        print('best_obj:{0}'.format(best_obj))



        from SS_RL.diagram import job_diagram
        import matplotlib.pyplot as plt

        dia = job_diagram(schedule, job_execute_time, file_name, len(txt_files) * iter + index)
        dia.pre()
        plt.savefig('./img1203/pic-{}.png'.format(len(txt_files) * iter + index))













#
# #　读取数据
# i = 0
# job_process_time=np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=2,max_rows=10,usecols = (1,3),unpack=True,dtype=int)
# ect_delay_wight = np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=14,max_rows=10,usecols = (2,3),unpack=True,dtype=int)
# ect_weight = ect_delay_wight[0]
# ddl_weight = ect_delay_wight[1]
# due_date_windows = np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=25,max_rows=10,dtype=int)
# ddl_windows = [i[1] for i in due_date_windows]
# ect_windows = [i[0] for i in due_date_windows]
#
# setup_time = {}
# setup_time[0] = np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=37,max_rows=10,dtype=int)
# setup_time[1] = np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=48,max_rows=10,dtype=int)
#
# jobs_num = int(np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=1,max_rows=1,usecols = (0),dtype=int))
# jobs = list(range(jobs_num))
#
# stages_num =  np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=1,max_rows=1,usecols = (2),dtype=int)
# total_mahcine_num = np.loadtxt("{0}_Instance_10_2_2_0,2_0,2_10_Rep{0}.txt".format(i,i),skiprows=1,max_rows=1,usecols = (1),dtype=int)
# machine_num_on_stage = []
# for job in range(stages_num):
#     machine_num_on_stage.append(int(total_mahcine_num / stages_num))
#
# #　声明schedule的字典变量
# schedule = {}
# for stage in range(stages_num):
#     for machine in range(machine_num_on_stage[stage]):
#         schedule[(stage,machine)] = []
#
#
# # 声明一个空的每个工件的完工时间
# machine_completion_time = {}
# for stage in range(stages_num):
#     for machine in range(machine_num_on_stage[stage]):
#         machine_completion_time[(stage, machine)] = [-1, -1]
#
# # 声明一个阶段的完工时间的字典
# job_completion_time = {}
# for stage in range(stages_num):
#     job_completion_time[stage] = np.zeros(jobs_num,dtype=int)
# # job_completion_time = np.zeros((stages_num, jobs_num), dtype=int)
#
# # 声明一个变量，用于存储每个阶段的工件调度顺序【用于第二阶段的local search】
# job_sort_on_satges = {}








#
# jobs_sort = [5,7,2,4,9,8,1,0,3,6]
# job_index = 5
# stage_index = 1
#
# # def gene_state(job_sort, job_index, stage_index):
# #     '''
# #     输入的参数：
# #     1.工件的排列
# #     2.当前需要调度的工件的索引
# #     3.
# #     '''
#
#
#
# class env():
#     def __init__(self):
#
#     def get_state(self,stage_index,job_index,job_completion_time,machine_completion_time):
#         n_state = torch.zeros(13)
#         remanin_process_time = torch.zeros(10)
#         renmain_setup_time = torch.zeros(10)
#         for index, job in enumerate(jobs_sort[job_index:]):
#             for stage in range(stage_index, stages_num):
#                 remanin_process_time[index] += job_process_time[stage][job]
#                 renmain_setup_time[index] += sum(setup_time[stage][job])
#         renmain_setup_time = renmain_setup_time / jobs_num / len(stages_num - stage_index)
#
#         a = ddl_windows - remanin_process_time - renmain_setup_time
#         b = ect_windows - remanin_process_time - renmain_setup_time
#
#         for index, job in enumerate(jobs_sort[job_index:]):
#             if a < 0:
#                 n_state[job] = a * ddl_weight[job]
#             elif b > 0:
#                 n_state[job] = b * ect_weight[job]
#             else:
#                 n_state[job] = 0
#
#         for index, machine in enumerate(machine_num_on_stage[stage_index]):
#             n_state[10 + index] = job_completion_time[stage_index - 1][jobs_sort[job_index]] - \
#                                   machine_completion_time[(stage_index, machine)][1]
#
#         n_state[12] = job_index / (jobs_num - 1)
#         n_state[13] = stage_index / (stages_num - 1)
#         self.state = n_state
#
#
#     def reset(self):
#         self.get_state(stage_index = 0,job_index = 0)
#
#     def step(self,action):
#         # 执行动作，更新三个变量
#         schedule,job_completion_time,machine_completion_time = job_assign(schedule,job_completion_time,machine_completion_time,action)
#         self.get_state(self,stage_index,job_index,job_completion_time,machine_completion_time)
#
#
#         return next_state,reward,done
#
#
# def step(state,action):
#
#     n_state = torch.zeros(13)
#     remanin_process_time = torch.zeros(10)
#     renmain_setup_time = torch.zeros(10)
#     for index,job in enumerate(jobs_sort[job_index:]):
#         for stage in range(stage_index,stages_num):
#             remanin_process_time[index] += job_process_time[stage][job]
#             renmain_setup_time[index] += sum(setup_time[stage][job])
#     renmain_setup_time = renmain_setup_time / jobs_num / len(stages_num - stage_index)
#
#     a = ddl_windows - remanin_process_time - setup_time
#     b = ect_windows - remanin_process_time - setup_time
#
#     for index,job in enumerate(jobs_sort[job_index:]):
#         if a < 0:
#             n_state[job] = a * ddl_weight[job]
#         elif b > 0:
#             n_state[job] = b * ect_weight[job]
#         else:
#             n_state[job] = 0
#
#     for index,machine in enumerate(machine_num_on_stage[stage_index]):
#         n_state[10+index] = job_completion_time[stage_index-1][jobs_sort[job_index]] - machine_completion_time[(stage_index,machine)][1]
#
#     n_state[12] = job_index / (jobs_num-1)
#     n_state[13] = stage_index / (stages_num-1)







