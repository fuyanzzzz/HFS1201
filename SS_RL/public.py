'''
-------------------------------------------------
File Name: public.py
Author: LRS
Create Time: 2023/6/27 12:33
-------------------------------------------------
'''
# from config import DataInfo
# class AllConfig:
#     config_1 = DataInfo(filename="")
#     config_2 = DataInfo(filename="")
#     config_3 = DataInfo(filename="")
#     config_4 = DataInfo(filename="")
#
#     @classmethod
#     def get_config(cls, key):
#         if key == "key":
#             return cls.config_1
#
# gen_data = AllConfig.get_config("key")


import os
from config import DataInfo

data_folder = "data"  # 数据文件夹的路径
# 获取data文件夹下所有的文件名
txt_files = [f for f in os.listdir(data_folder) if f.endswith(".txt")]
class AllConfig:
    config = {}
    for file_name in txt_files:
        print(file_name)
        config[file_name] = DataInfo(filename=file_name)

    @classmethod
    def get_config(cls, key):
        return cls.config[key]

# for file_name in txt_files:
#     gen_data = AllConfig.get_config(file_name)
# print(1)

