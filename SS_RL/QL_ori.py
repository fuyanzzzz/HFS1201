'''
-------------------------------------------------
File Name: QL_ori.py
Author: LRS
Create Time: 2023/5/25 10:15
-------------------------------------------------
'''
# -*- coding: utf-8 -*-
# https://github.com/sichkar-valentyn/Reinforcement_Learning_in_Python/blob/master/RL_Q-Learning_E1/run_agent.py
import numpy as np

# 初始化团队的Q值函数
num_teams = 2
num_actions = 4
q_values = np.zeros((num_teams, num_actions))

# 定义团队的动作选择函数
def select_action(team_id):
    # 根据Q值函数选择动作
    action = np.argmax(q_values[team_id])
    return action

# 定义奖励函数
def get_reward(team_id, action):
    # 返回执行动作后的奖励
    # 根据具体任务定义奖励逻辑
    reward = 0
    if team_id == 0 and action == 0:
        reward = 1
    elif team_id == 1 and action == 3:
        reward = 1
    return reward

# 定义Q值更新函数
def update_q_values(team_id, action, reward):
    # 更新Q值函数
    learning_rate = 0.1
    discount_factor = 0.9
    q_values[team_id, action] += learning_rate * (reward + discount_factor * np.max(q_values[team_id]) - q_values[team_id, action])

# 训练团队的Q值函数
num_episodes = 1000
for episode in range(num_episodes):
    # 每个团队执行动作并更新Q值函数
    for team_id in range(num_teams):
        action = select_action(team_id)
        reward = get_reward(team_id, action)
        update_q_values(team_id, action, reward)

# 输出训练后的Q值函数
print("Trained Q-values:")
print(q_values)
