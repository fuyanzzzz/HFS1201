import random
from itertools import count
from tensorboardX import SummaryWriter
import gym
from collections import deque
import numpy as np
from torch.nn import functional as F
import torch
import torch.nn as nn
class Dueling_DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Dueling_DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.f1 = nn.Linear(state_dim, 512)
        self.f2 = nn.Linear(512, 256)

        self.val_hidden = nn.Linear(256, 128)
        self.adv_hidden = nn.Linear(256, 128)

        self.val = nn.Linear(128, 1)

        self.adv = nn.Linear(128, action_dim)

    def forward(self, x):

        x = self.f1(x)
        x = F.relu(x)
        x = self.f2(x)
        x = F.relu(x)

        val_hidden = self.val_hidden(x)
        val_hidden = F.relu(val_hidden)

        adv_hidden = self.adv_hidden(x)
        adv_hidden = F.relu(adv_hidden)

        val = self.val(val_hidden)

        adv = self.adv(adv_hidden)

        # 用平均值代替最大值，能够获得更好的稳定性
        adv_ave = torch.mean(adv, dim=1, keepdim=True)

        x = adv + val - adv_ave

        return x

    def select_action(self, state):
        with torch.no_grad():
            # print(state)
            Q = self.forward(state)
            action_index = torch.argmax(Q, dim=1)
        return action_index.item()

class Memory(object):
    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > self.size():
            batch_size = self.size()
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

GAMMA = 0.99
BATH = 256          # 批量训练256
EXPLORE = 2000000
REPLAY_MEMORY = 50000   # 经验池容量5W
BEGIN_LEARN_SIZE = 1024
memory = Memory(REPLAY_MEMORY)
UPDATA_TAGESTEP = 200       # 目标网络的更新频次
learn_step = 0
epsilon = 0.2
writer = SummaryWriter('logs/dueling_DQN2')
FINAL_EPSILON = 0.00001


env = gym.make('Berzerk-ram-v0')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n
target_network = Dueling_DQN(n_state, n_action)
network = Dueling_DQN(n_state, n_action)
target_network.load_state_dict(network.state_dict())
optimizer = torch.optim.Adam(network.parameters(), lr=0.0001)
r = 0
c = 0
for epoch in count():
    state = env.reset()
    episode_reward = 0
    c += 1
    while True:
        # env.render()
        state = state / 255

        p = random.random()
        # 动作选择
        if p < epsilon:
            action = random.randint(0, n_action - 1)
        else:
            state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
            action = network.select_action(state_tensor)

        # 根据（状态，动作）得到step序列
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward

        # 将得到序列添加到经验池
        memory.add((state, next_state, action, reward, done))

        # 只有当经验池中的样本数量大于批量训练的数量，才会执行训练
        if memory.size() > BEGIN_LEARN_SIZE:
            learn_step += 1

            # 每隔一段时间，更新目标网络
            if learn_step % UPDATA_TAGESTEP:
                target_network.load_state_dict(network.state_dict())

            # 训练的时候，从经验池中进行采样
            batch = memory.sample(BATH, False)
            batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

            batch_state = torch.as_tensor(batch_state, dtype=torch.float)
            batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float)
            batch_action = torch.as_tensor(batch_action, dtype=torch.long).unsqueeze(0)
            batch_reward = torch.as_tensor(batch_reward, dtype=torch.float).unsqueeze(0)
            batch_done = torch.as_tensor(batch_done, dtype=torch.long).unsqueeze(0)


            with torch.no_grad():
                target_Q_next = target_network(batch_next_state)        # 从目标网络中

                # 将next_state输入预测网络，并选择最优的状态动作价值函数
                Q_next = network(batch_next_state)
                Q_max_action = torch.argmax(Q_next, dim=1, keepdim=True)

                # 再将这个输入目标网络中
                y = batch_reward + target_Q_next.gather(1, Q_max_action)


            loss = F.mse_loss(network(batch_state).gather(1, batch_action), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss', loss.item(), global_step=learn_step)

            # if epsilon > FINAL_EPSILON: ## 减小探索
            #     epsilon -= (0.1 - FINAL_EPSILON) / EXPLORE
        if done:
            break
        state = next_state
    r += episode_reward
    writer.add_scalar('episode reward', episode_reward, global_step=epoch)
    if epoch % 100 == 0:
        print(f"第{epoch / 100}个100epoch的reward为{r / 100}", epsilon)
        r = 0
    if epoch % 10 == 0:
        torch.save(network.state_dict(), 'model/netwark{}.pt'.format("dueling"))