'''
-------------------------------------------------
File Name: groubi_hfs.py
Author: LRS
Create Time: 2023/3/18 21:39
-------------------------------------------------
'''

import gurobipy
import numpy as np


i = 0
job_process_time = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=2, max_rows=10,
                              usecols=(1, 3), unpack=True, dtype=int)
ect_delay_wight = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=14, max_rows=10,
                             usecols=(2, 3), unpack=True, dtype=int)
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]
due_date_windows = (np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=25, max_rows=10,
                              dtype=int))
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

setup_time = {}
setup_time[0] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=37, max_rows=10,
                           dtype=int)
setup_time[1] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=48, max_rows=10,
                           dtype=int)

job_num = int(np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(0),
                          dtype=int))
jobs = list(range(job_num))

stage_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1, usecols=(2),
                        dtype=int)
total_mahcine_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt", skiprows=1, max_rows=1,
                               usecols=(1), dtype=int)
machine_num = 2
machine_num_on_stage = []
for job in range(stage_num):
    machine_num_on_stage.append(int(total_mahcine_num / stage_num))



# 创建模型
MODEL = gurobipy.Model()

# 创建变量
    # 创建多个变量 stage_num, machine_num, job_num, job_num，用 x[1,2,3,4]索引
x = MODEL.addVars(stage_num, machine_num, job_num, job_num, vtype=gurobipy.GRB.BINARY, name="")
    # 阶段l工件i的完工时间  ！！是否需要非负
c = MODEL.addVars(stage_num, job_num, vtype=gurobipy.GRB.INTEGER ,name="")
    # 工件的早到时间
e = MODEL.addVars(job_num, vtype=gurobipy.GRB.INTEGER ,name="")
    # 工件的延误时间
t = MODEL.addVars(job_num, vtype=gurobipy.GRB.INTEGER ,name="")

# 更新变量环境
MODEL.update()

# 创建目标函数
expression = sum(e[j] * ddl_windows[j] + t[j] * ect_windows[j] for j in range(job_num))
MODEL.setObjective(expression, gurobipy.GRB.MINIMIZE)

# 创建约束条件
F = np.inf      # 极大值
for l in range(stage_num):
    for i in range(job_num):
        for j in range(job_num):
            # 约束条件
            MODEL.addConstr(c[l,i] + x.sum(l,"*",i ,j) *(setup_time[l,i,j] + job_process_time[l,j] + F * x.sum(l,"*",i ,j))<= c[l,j])

for l in range(stage_num):
    for k in range(machine_num_on_stage[l]):
        for i in range(job_num):
            for j in range(job_num):
                #      ⭐⭐⭐⭐⭐⭐⭐约束条件5
                MODEL.addConstr(c[l,j] >= x[l,k,i,j] * (setup_time[l,i,j] + job_process_time[l,j]))
                if l == 0:
                    continue
                #      ⭐⭐⭐⭐⭐⭐⭐约束条件3
                MODEL.addConstr(c[l-1,j] + x[l,k,i,j]*(setup_time[l,i,j] + job_process_time[l,j]) <= c[l,j])

for l in range(stage_num):
    for m in range(machine_num):
        for i in range(job_num):
            for j in range(job_num):
                if l == 0:
                    # 如果是第一阶段，完工时间 = 该机器上的上一个工件的完工时间 + 切换时间 + 工件加工时间
                    if x[l, m, i, j] == 1:
                        MODEL.addConstr(c[l, j] >= c[l, i] + x[l, m, i, j] * (job_process_time[l, i] + setup_time[l, i, j]))
                else:
                # 如果非第一阶段，完工时间 = max(该机器上的上一个工件的完工时间，上一个阶段该工件的完工时间）+ 切换时间 + 工件加工时间
                    if x[l, m, i, j] == 1:
                        MODEL.addConstr(c[l, j] >= max(c[l, i], c[l - 1, j]) + x[l, m, i, j] * (job_process_time[l, i] + setup_time[l, i, j]))

for j in range(job_num):
    MODEL.addConstr(e[j] >= max(ddl_windows[j] - c[stage_num, j]))  # ⭐⭐⭐⭐⭐⭐⭐约束条件6
    MODEL.addConstr(t[j] >= max(c[stage_num, j] - ect_windows[j]))  # ⭐⭐⭐⭐⭐⭐⭐约束条件7




# 执行最优化
MODEL.optimize()

for l in range(stage_num):
    for i in range(job_num):
        print('c[{0},{1}]:{2}'.format(l,i,c[l,i]))







