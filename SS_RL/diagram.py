import matplotlib.pyplot as plt
import random

from SS_RL.public import AllConfig
from config import *

plt.rcParams['font.sans-serif'] = 'SimHei'

class job_diagram():
    def __init__(self,schedule,job_execute_time,file_name,epoch):
        self.schedule = schedule

        self.config = AllConfig.get_config(file_name)
        self.gen_job_completion_time(job_execute_time)

        self.epoch = epoch
        self.early_job = []
        self.delay_job = []
        self.on_time_job = []
        self.colors = []

    def gen_job_completion_time(self,job_execute_time):
        self.job_completion_time = {}
        for stage in range(self.config.stages_num):
            for job in range(self.config.jobs_num):
                self.job_completion_time[(stage,job)] = [job_execute_time[(stage,job)] - self.config.job_process_time[stage][job], job_execute_time[(stage,job)]]

    def judge_early_delay_jobs(self):
        for job in range(self.config.jobs_num):
            if self.job_completion_time[(self.config.stages_num-1,job)][1] < self.config.ect_windows[job]:
                self.early_job.append(job)
            elif self.job_completion_time[(self.config.stages_num-1,job)][1] > self.config.ddl_windows[job]:
                self.delay_job.append(job)
            else:
                self.on_time_job.append(job)

    def generate_light_colors(self):
        for _ in range(self.config.jobs_num):
            r = random.randint(160, 255)  # 生成随机的红色分量
            g = random.randint(160, 255)  # 生成随机的绿色分量
            b = random.randint(160, 255)  # 生成随机的蓝色分量
            color = '#{:02X}{:02X}{:02X}'.format(r, g, b)  # 将RGB分量转换为16进制颜色码
            self.colors.append(color)


    # def gen_job_diagram(self,job_labels,job_colors):
    #     # 绘制工序图
    #     # 绘制第一阶段第一台机器
    #     for stage in range(self.stages_num):
    #         for machine in range(2):
    #             for job in self.schedule[(stage, machine)]:
    #                 if job in self.early_job:
    #                     edge_color = 'green'
    #                 elif job in self.delay_job:
    #                     edge_color = 'red'
    #                 else:
    #                     edge_color = 'black'
    #                 ax.barh(y=0,
    #                         width=self.job_completion_time[(stage, job)][1] - self.job_completion_time[(stage, job)][0],
    #                         left=self.job_completion_time[(stage, job)][0], height=0.6,
    #                         align='center', color=job_colors[job], edgecolor=edge_color)
    #                 va_location = 'center'
    #                 if stage == 1 and job == 7:
    #                     va_location = 'top'
    #                 if stage == 0:
    #                     ax.text(
    #                         (self.job_completion_time[(stage, job)][0] + self.job_completion_time[(stage, job)][1]) / 2,
    #                         0,
    #                         f'{job_labels[self.schedule[(0, 0)][i]]}\n({self.job_completion_time[(stage, job)][0]}-{self.job_completion_time[(stage, job)][1]},)',
    #                         color='black', va=va_location, ha='center')
    #                 else:
    #                     ax.text(
    #                         (self.job_completion_time[(stage, job)][0] + self.job_completion_time[(stage, job)][1]) / 2,
    #                         3,
    #                         f'{job_labels[job]}\n({self.job_completion_time[(stage, job)][0]}-{self.job_completion_time[(stage, job)][1]})\n'
    #                         f'({ect_windows[job]}-{ddl_windows[job]})\n'
    #                         f'({ect_weight[job]}-{ddl_weight[job]})\n'
    #                         f'({move_unit})\n',
    #                         color='black', va=va_location, ha='center')

    def pre(self):
        fig, ax = plt.subplots(figsize=(20, 10))

        # 设置Y轴刻度
        # y_ticks = ['Stage 1, Machine 1', 'Stage 1, Machine 2', 'Stage 2, Machine 1', 'Stage 2, Machine 2']
        y_ticks = []
        for i_stage in range(self.config.stages_num):
            for j_machine in range(self.config.machine_num_on_stage[0]):
                y_ticks.append('{0},{1}'.format(i_stage,j_machine))

        ax.set_yticks(range(len(y_ticks)))
        ax.set_yticklabels(y_ticks)
        ax.set_ylim(-1, len(y_ticks))

        self.generate_light_colors()

        # job_labels = ['Job 0', 'Job 1', 'Job 2', 'Job 3', 'Job 4', 'Job 5', 'Job 6', 'Job 7', 'Job 8', 'Job 9']
        job_labels = ['Job {0}'.format(i) for i in range(self.config.jobs_num)]

        job_colors = {0: '#F8C9CF', 1: '#B1B9E9',2: '#EEA6DB', 3: '#DBBEFC', 4: '#E0CAAF',5: '#AFFDB3', 6: '#EEFFC7', 7: '#C9FAFF',
         8: '#CDC5DD', 9: '#EBDEA6',10: '#F8C9CF', 11: '#B1B9E9',12: '#EEA6DB', 13: '#DBBEFC', 14: '#E0CAAF',15: '#AFFDB3', 16: '#EEFFC7', 17: '#C9FAFF',
         18: '#CDC5DD', 19: '#EBDEA6'}


        # 设置X轴范围
        max_time = max([self.job_completion_time[(1,job)][1] for job in range(self.config.jobs_num)])
        ax.set_xlim(0, max_time + 20)
        # self.gen_job_diagram(job_labels,job_colors)
        # 设置图例
        # ax.legend(['工件 0', '工件 1', '工件 2', '工件 3', '工件 4', '工件 5', '工件 6', '工件 7', '工件 8', '工件 9'],
        #           loc='upper right', bbox_to_anchor=(1.25, 1))
        self.judge_early_delay_jobs()
        for stage in range(self.config.stages_num):
            for machine in range(self.config.machine_num_on_stage[0]):
                for job in self.schedule[(stage, machine)]:
                    move_unit = None
                    if job in self.early_job:
                        edge_color = 'green'
                        move_unit = int(self.config.ect_windows[job] - self.job_completion_time[(stage, job)][1])
                    elif job in self.delay_job:
                        edge_color = 'red'
                        move_unit = int(self.job_completion_time[(stage, job)][1] - self.config.ddl_windows[job])
                    else:
                        edge_color = 'black'
                        move_unit = int(self.config.ddl_windows[job] - self.job_completion_time[(stage, job)][1])
                    ax.barh(y=stage*self.config.machine_num_on_stage[0]+machine,
                            width=self.job_completion_time[(stage, job)][1] - self.job_completion_time[(stage, job)][0],
                            left=self.job_completion_time[(stage, job)][0], height=0.6,
                            align='center', color=job_colors[job], edgecolor=edge_color)
                    va_location = 'center'
                    if stage == 1 and job == 7:
                        va_location = 'top'
                    if stage == 0:
                        ax.text(
                            (self.job_completion_time[(stage, job)][0] + self.job_completion_time[(stage, job)][1]) / 2,
                            stage*self.config.machine_num_on_stage[0]+machine,
                            f'{job_labels[job]}\n({self.job_completion_time[(stage, job)][0]}-{self.job_completion_time[(stage, job)][1]}){self.config.job_process_time[stage][job]}',
                            color='black', va=va_location, ha='center')
                    else:
                        ax.text(
                            (self.job_completion_time[(stage, job)][0] + self.job_completion_time[(stage, job)][1]) / 2,stage*self.config.machine_num_on_stage[0]+machine,
                            f'{job_labels[job]}\n({self.job_completion_time[(stage, job)][0]}-{self.job_completion_time[(stage, job)][1]}){self.config.job_process_time[stage][job]}\n'
                            f'{self.config.ect_weight[job]}({self.config.ect_windows[job]}-{self.config.ddl_windows[job]}){self.config.ddl_weight[job]}\n'
                            # f'({ect_weight[job]}-{ddl_weight[job]})\n'
                            f'({move_unit})\n',
                            color='black', va=va_location, ha='center')
        # 添加标题和标签
        ax.set_title('Two-Stage Hybrid Flowshop Scheduling')
        ax.set_xlabel('Completion Time')
        ax.set_ylabel('Machine')
        ax.invert_yaxis()
        # 显示图形

        # # 绘制图像
        # block_infor = {0: [[3, 5, 8, 1, 6]], 1: [[9], [4, 7, 0, 2]]}
        # # 添加备注信息
        # ax.text(0.5, -10, "[[3, 5, 8, 1, 6]]", ha='center', va='center', fontsize=12)
        # ax.text(0.5, -10, "[[9], [4, 7, 0, 2]]", ha='center', va='center', fontsize=12)
        plt.savefig('./img0629/pic-{}.png'.format(int(self.epoch)))
        plt.show()

