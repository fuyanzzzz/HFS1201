import numpy as np

from SS_RL.diagram import job_diagram
from SS_RL.public import AllConfig

class ScheduleCal():
    def __init__(self,schedule,file_name):
        self.config = AllConfig.get_config(file_name)
        self.schedule = schedule
        self.job_execute_time = {}
        self.gen_job_execute_time()
        self.obj = 0
        self.recal()
        self.cal()
        diag = job_diagram(self.schedule, self.job_execute_time, file_name, 3)
        diag.pre()
        self.idle_time_insertion()

    def gen_job_execute_time(self):
        for stage in range(self.config.stages_num):
            for job in range(self.config.jobs_num):
                self.job_execute_time[(stage, job)] = 0

    def cal(self):
        ect_value = 0
        ddl_value = 0
        for job in range(self.config.jobs_num):
            job_makespan = self.job_execute_time[(self.config.stages_num - 1, job)]
            if job_makespan < self.config.ect_windows[job]:  # 早前权重值
                ect_value += (self.config.ect_windows[job] - job_makespan) * self.config.ect_weight[job]
            elif job_makespan > self.config.ddl_windows[job]:  # 延误权重值
                ddl_value += (job_makespan - self.config.ddl_windows[job]) * self.config.ddl_weight[job]

        self.obj = ect_value + ddl_value


    def recal(self):
        # 这里更新的是self.job_execute_time
        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[stage]):
                pro_job = None
                for i_index, job in enumerate(self.schedule[stage, machine]):
                    if stage == 0:
                        if i_index == 0:
                            self.job_execute_time[(stage, job)] = self.config.job_process_time[stage][job]
                        else:
                            pro_job = self.schedule[(stage, machine)][i_index - 1]
                            self.job_execute_time[(stage, job)] = self.job_execute_time[
                                                                             (stage, pro_job)] + \
                                                                         self.config.job_process_time[stage][job]
                    else:
                        if i_index == 0:
                            self.job_execute_time[(stage, job)] = self.job_execute_time[
                                                                             (stage - 1, job)] + \
                                                                         self.config.job_process_time[stage][job]
                        else:
                            pro_job = self.schedule[(stage, machine)][i_index - 1]
                            self.job_execute_time[(stage, job)] = max(
                                self.job_execute_time[(stage, pro_job)],
                                self.job_execute_time[(stage - 1, job)]) + \
                                                                         self.config.job_process_time[stage][job]

    def idle_time_insertion(self):
        # 外部循环，遍历最后一个阶段上的所有机器
        self.schedule_job_block = {}
        self.all_job_block = []
        for machine in range(self.config.machine_num_on_stage[-1]):
            # 内部循环，从机器上最后一个工件开始往前遍历
            job_block = []  # 声明一个空的工件块
            self.all_job_block = []
            delay_job = []  # 最好用字典存储
            early_job = []
            on_time_job = []

            job_list_machine = self.schedule[(self.config.stages_num - 1), machine].copy()
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
                if (job == job_list_machine[-1] or (self.job_execute_time[(self.config.stages_num - 1, job)] ==
                                                    (self.job_execute_time[(self.config.stages_num - 1, later_job)] -
                                                     self.config.job_process_time[self.config.stages_num - 1][
                                                         later_job]))) and job not in job_block:
                    job_block.insert(0, job)  # 构建工件块

                elif job not in job_block:  # 如果这个工件没有和下一个工件并在一块，则将原来的工件块插入到全部工件块中
                    self.all_job_block.insert(0, job_block)
                    job_block = []  # 再声明一个新的工件块
                    job_block.insert(0, job)  # 重新插入新的工件块中

                job_before_idle = job_block[-1]
                if len(self.all_job_block) != 0:  # 如果当前工件块右侧存在工件
                    job_after_idle = self.all_job_block[0][0]
                    later_block_start_time = self.job_execute_time[(self.config.stages_num - 1, job_after_idle)] - \
                                             self.config.job_process_time[self.config.stages_num - 1][job_after_idle]
                    job_before_idle_end_time = self.job_execute_time[(self.config.stages_num - 1, job_before_idle)]
                    idle_2 = later_block_start_time - job_before_idle_end_time
                else:
                    idle_2 = np.inf  # 如果右边没有工件了，赋值无穷大

                # 根据当前工件块生成三个子集
                early_job.clear()
                delay_job.clear()
                on_time_job.clear()
                for job in job_block:
                    if self.job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
                        early_job.append(job)
                    elif self.job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
                        delay_job.append(job)
                    else:
                        on_time_job.append(job)

                early_job_weight = sum([self.config.ect_weight[job] for job in early_job])
                delay_job_weight = sum([self.config.ddl_weight[job] for job in delay_job])

                if early_job_weight > delay_job_weight:
                    early = []  # 计算距离准时早到的空闲时间
                    delay = []  # 计算超过准时的延误的空闲时间

                    # 计算距离准时的最小早到的空闲时间
                    for job in early_job:
                        early.append(
                            self.config.ect_windows[job] - self.job_execute_time[(self.config.stages_num - 1, job)])  # !!!!!!!!这个变量很重要，要实时更新

                    # 计算超过准时的最小延误的空闲时间
                    for job in on_time_job:
                        delay.append(self.config.ddl_windows[job] - self.job_execute_time[(self.config.stages_num - 1, job)])
                    if len(early) == 0 and len(delay) != 0:
                        idle_1 = min(delay)
                    elif len(delay) == 0 and len(early) != 0:
                        idle_1 = min(early)
                    else:
                        idle_1 = min(min(early), min(delay))
                    insert_idle_time = min(idle_1, idle_2)  # 确定需要插入的工件块

                    for job in job_block:
                        self.job_execute_time[(self.config.stages_num - 1, job)] += insert_idle_time
                    improvement_obj = (early_job_weight - delay_job_weight) * insert_idle_time  # 获得改进的目标值
                    self.obj -= improvement_obj  # 重新计算目标值

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
                if self.job_execute_time[(self.config.stages_num - 1, job)] < self.config.ect_windows[job]:
                    early_job.append(job)
                elif self.job_execute_time[(self.config.stages_num - 1, job)] >= self.config.ddl_windows[job]:
                    delay_job.append(job)
                else:
                    on_time_job.append(job)

            # 创建一个字典：
            '''
            工件：目标值，偏差距离，延误、早到或准时的flag，以及距离第一阶段的完工时间的差距
            '''
        # self.get_job_info()