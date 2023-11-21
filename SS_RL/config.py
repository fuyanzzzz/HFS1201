'''
-------------------------------------------------
File Name: config.py
Author: LRS
Create Time: 2023/6/13 21:28
-------------------------------------------------
'''
import numpy as np

class DataInfo():
    def __init__(self, filename):
        self.filename = filename
        self.job_process_time = None
        self.jobs_num = None
        self.stages_num = None
        self.total_mahcine_num = None
        self.ect_weight = None
        self.ddl_weight = None
        self.ddl_windows = None
        self.ect_windows = None
        self.machine_num_on_stage = []

        self.init_params()


    def init_params(self):
        # np.loadtxt("data\{0}".format(filename_12), skiprows=14, max_rows=10,
        #            usecols=(2, 3), unpack=True, dtype=int)

        # self.job_process_time = np.loadtxt("data\{0}".format(self.filename), skiprows=2, max_rows=10,
        #                               usecols=(1, 3), unpack=True, dtype=int)
        # ect_delay_wight = np.loadtxt("data\{0}".format(self.filename), skiprows=14, max_rows=10,
        #                              usecols=(2, 3), unpack=True, dtype=int)
        # due_date_windows = (np.loadtxt("data\{0}".format(self.filename), skiprows=25, max_rows=10,
        #                                dtype=int))
        # self.jobs_num = int(np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1, usecols=(0),
        #                           dtype=int))   # 工件数量
        # self.stages_num = int(np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1, usecols=(2),
        #                             dtype=int))
        # self.total_mahcine_num = np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1,
        #                                usecols=(1), dtype=int)




        self.jobs_num = int(np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1, usecols=(0),
                                  dtype=int))   # 工件数量
        self.stages_num = int(np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1, usecols=(2),
                                    dtype=int))
        self.total_mahcine_num = np.loadtxt("data\{0}".format(self.filename), skiprows=1, max_rows=1,
                                       usecols=(1), dtype=int)

        self.job_process_time = np.loadtxt("data\{0}".format(self.filename), skiprows=2, max_rows=self.jobs_num,
                                      usecols=(1, 3), unpack=True, dtype=int)
        ect_delay_wight = np.loadtxt("data\{0}".format(self.filename), skiprows=4+self.jobs_num, max_rows=self.jobs_num,
                                     usecols=(2, 3), unpack=True, dtype=int)
        due_date_windows = (np.loadtxt("data\{0}".format(self.filename), skiprows=5+self.jobs_num*2, max_rows=self.jobs_num,
                                       dtype=int))

        # 将数据转化为相应变量
        self.ect_weight = ect_delay_wight[0]
        self.ddl_weight = ect_delay_wight[1]
        self.ddl_windows = [i[1] for i in due_date_windows]
        self.ect_windows = [i[0] for i in due_date_windows]
        self.jobs = list(range(self.jobs_num))
        for job in range(self.stages_num):
            self.machine_num_on_stage.append(int(self.total_mahcine_num / self.stages_num))

        parts = self.filename.split('_')
        result = int(parts[0])


        # 获取最优解：
        remainder = result % 540

        skiprows_i = (result // 540) *540*5 + remainder*4 + max(0,remainder-180)*1 + max(0,remainder-360)*1
        if result == int(np.loadtxt("HFSDDW_Small_Best_Sequence.txt", skiprows=skiprows_i, max_rows=1, usecols=(1),dtype=int)):   # 工件数量
            self.ture_opt = int(np.loadtxt("HFSDDW_Small_Best_Sequence.txt", skiprows=skiprows_i, max_rows=1, usecols=(7),dtype=int))
        else:
            self.true_opt = None
