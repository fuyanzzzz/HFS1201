'''
-------------------------------------------------
File Name: test.py
Author: LRS
Create Time: 2023/2/25 00:41
-------------------------------------------------
'''
import numpy as np

job_process_time=np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=2,max_rows=10,usecols = (1,3),unpack=True,dtype=int)

ect_delay_wight = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=14,max_rows=10,usecols = (2,3),unpack=True,dtype=int)
ect_weight = ect_delay_wight[0]
ddl_weight = ect_delay_wight[1]


due_date_windows = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=25,max_rows=10,dtype=int)
ddl_windows = [i[1] for i in due_date_windows]
ect_windows = [i[0] for i in due_date_windows]

set_up_time = {}
set_up_time[0] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=37,max_rows=10,dtype=int)
set_up_time[1] = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=48,max_rows=10,dtype=int)

jobs_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=1,max_rows=1,usecols = (0),dtype=int)
jobs = list(range(jobs_num))


stages_num =  np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=1,max_rows=1,usecols = (2),dtype=int)
total_mahcine_num = np.loadtxt("0_Instance_10_2_2_0,2_0,2_10_Rep0.txt",skiprows=1,max_rows=1,usecols = (1),dtype=int)
machine_num_on_stage = []
for job in range(stages_num):
    machine_num_on_stage.append(int(total_mahcine_num / stages_num))


#　声明schedule的字典变量
schedule = {}
for stage in range(stages_num):
    for machine in range(machine_num_on_stage[stage]):
        schedule[(stage,machine)] = []