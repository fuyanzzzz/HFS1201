import random
import numpy as np

due_date_windows = [[62,64],
[67,68],
[70,72],
[74,76],
[79,81],
[88,90],
[93,95],
[97,99]]

job_list_machine = [0, 1, 2, 3, 4, 5, 6, 7]

ddl = [i[1] for i in due_date_windows]
ect = [i[0] for i in due_date_windows]

ect_weight= [1,2,1,3,1,2,1,4]
ddl_weight = [2,1,2,1,2,1,2,3]

job_makespan = [61,65,69,75,82,86,92,97]
machine_num_on_stage = [1,1]
obj = 13

job_process_time = [5,4,4,6,7,2,6,4]

def idle_time_insertion(job_sort):
    #　外部循环，遍历最后一个阶段上的所有机器
    # schedule, obj, job_makespan = job_assignment(job_sort)
    obj = 13
    all_job_block = []
    for machine in range(machine_num_on_stage[-1]):
        # 内部循环，从机器上最后一个工件开始往前遍历
        job_block = []      # 声明一个空的工件块
        all_job_block = []
        delay_job = []
        early_job = []
        on_time_job = []

        # job_list_machine = schedule[(stages_num-1),machine].copy()
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
            if (job == job_list_machine[-1] or (job_makespan[job] == (job_makespan[later_job] - job_process_time[later_job]))) and job not in job_block:
                job_block.insert(0,job)           # 构建工件块

            elif job not in job_block:   # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
                all_job_block.insert(0,job_block)

                job_block = []      # 再声明一个新的工件块

                job_block.insert(0,job)     # 重新插入新的工件块中


            job_before_idle = job_block[-1]
            if len(all_job_block) != 0:        # 如果当前工件块右侧存在工件
                job_after_idle = all_job_block[0][0]
                job_completion_time = job_makespan[job_before_idle]  # 当前工件块最后一个工件的完工时间
                # pre_block_start_time = job_makespan[job_after_idle] - setup_time[stages_num - 1][job_before_idle][job_after_idle] - job_process_time[stages_num - 1][job_after_idle]
                later_block_start_time = job_makespan[job_after_idle] - job_process_time[job_after_idle]
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
                    early.append(ect[job] - job_makespan[job])                            # !!!!!!!!这个变量很重要，要实时更新

                # 计算超过准时的最小延误的空闲时间
                for job in delay_job:
                    delay.append(job_makespan[job] - ddl[job])                              # !!!!!!!!这个变量很重要，要实时更新
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

obj,all_job_block = idle_time_insertion(job_list_machine)
print(obj,all_job_block)