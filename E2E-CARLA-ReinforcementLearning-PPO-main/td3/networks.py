import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- 1. 经验回放池 (优化了图像存储内存) ---
class ReplayBuffer:
    def __init__(self, state_shape, action_dim, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        # 为了节省内存，图像以 uint8 格式存储
        self.state = np.zeros((max_size, *state_shape), dtype=np.uint8)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.uint8)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        # 采样时转换为 float32 并归一化到 [0, 1]
        state = torch.FloatTensor(self.state[ind]).to(self.device) / 255.0
        next_state = torch.FloatTensor(self.next_state[ind]).to(self.device) / 255.0

        # 调整维度从 (B, H, W, C) 到 (B, C, H, W)
        state = state.permute(0, 3, 1, 2)
        next_state = next_state.permute(0, 3, 1, 2)

        return (
            state,
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            next_state,
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


# --- 2. CNN 特征提取器 ---
class CNNBase(nn.Module):
    def __init__(self, channels=1):
        super(CNNBase, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 11 * 11, 256)  # 假设输入为 128x128

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        return F.relu(self.fc(x))


# --- 3. Actor 网络 ---
class Actor(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Actor, self).__init__()
        self.cnn = CNNBase(channels=1)
        self.l1 = nn.Linear(256, 256)
        self.l2 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        feat = self.cnn(state)
        a = F.relu(self.l1(feat))
        return self.max_action * torch.tanh(self.l2(a))


# --- 4. Twin Critic 网络 ---
class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.cnn1 = CNNBase(channels=1)
        self.l1 = nn.Linear(256 + action_dim, 256)
        self.l2 = nn.Linear(256, 1)

        # Q2 architecture
        self.cnn2 = CNNBase(channels=1)
        self.l3 = nn.Linear(256 + action_dim, 256)
        self.l4 = nn.Linear(256, 1)

    def forward(self, state, action):
        feat1 = self.cnn1(state)
        q1 = F.relu(self.l1(torch.cat([feat1, action], 1)))
        q1 = self.l2(q1)

        feat2 = self.cnn2(state)
        q2 = F.relu(self.l3(torch.cat([feat2, action], 1)))
        q2 = self.l4(q2)
        return q1, q2

    def Q1(self, state, action):
        feat1 = self.cnn1(state)
        q1 = F.relu(self.l1(torch.cat([feat1, action], 1)))
        return self.l2(q1)