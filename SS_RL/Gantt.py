'''
-------------------------------------------------
File Name: Gantt.py
Author: LRS
Create Time: 2023/5/27 15:55
-------------------------------------------------
'''
import matplotlib.pyplot as plt
import pandas as pd
import random



# 创建数据
# for i in range(jobs_num):
jobs_num = 10
data = {'Task': ['job{0}'.format(i) for i in range(jobs_num)],
        'Start': [0, 2, 4],
        'Finish': [job_execute_time[job] for job in range(jobs_num)],
        # 'Deadline': [2, 5, 4]
        }  # 添加截止日期

df = pd.DataFrame(data)

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(8, 4))

# 定义任务色块和截止日期的颜色
color_palette = ['steelblue', 'darkorange', 'forestgreen']

# 绘制甘特图和截止日期
for i, task in df.iterrows():
    # task_color = random.choice(color_palette)  # 为每个任务随机选择一个颜色
    ax.barh(task['Task'], task['Finish'] - task['Start'], left=task['Start'], height=0.5, color=color_palette[i])
    ax.axvline(x=task['Deadline'], color=color_palette[i], linestyle='--')

# 设置标题和标签
ax.set_title('Gantt Chart')
ax.set_xlabel('Time')
ax.set_ylabel('Task')

# 设置刻度
ax.xaxis.grid(True)
ax.set_xlim(0, 7)
ax.set_xticks(range(8))
ax.set_yticks(range(len(df)))
ax.set_yticklabels(df['Task'])

# 显示图形
plt.show()
