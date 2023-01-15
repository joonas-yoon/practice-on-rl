import os
import random
from collections import namedtuple, deque
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple(
    'Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class LambdaLayer(nn.Module):
    def __init__(self, _lambda):
        super(LambdaLayer, self).__init__()
        self._lambda = _lambda

    def forward(self, x):
        return self._lambda(x)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Parameters:
    BATCH_SIZE = 256   # is the number of transitions sampled from the replay buffer
    GAMMA = 0.999      # is the discount factor as mentioned in the previous section
    EPS_START = 0.95   # is the starting value of epsilon
    EPS_END = 0.05     # is the final value of epsilon
    # controls the rate of exponential decay of epsilon, higher means a slower deca
    EPS_DECAY = 0.95
    TAU = 0.005        # is the update rate of the target network
    LR = 1e-4          # is the learning rate of the AdamW optimizer


class Agent:
    _current_step = 0

    def __init__(
        self,
        policy_net: nn.Module,
        target_net: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
    ) -> None:
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.criterion = loss_fn
        self.optimizer = optimizer

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def get_epsilon(self) -> float:
        start, end, cur = Parameters.EPS_START, Parameters.EPS_END, self._current_step
        eps = (start - end) * ((Parameters.EPS_DECAY + 1e-12) ** cur) + end
        return eps

    def update_target_net(self, soft=True) -> None:
        """
        Soft update of the target network's weights
        θ′ ← τ θ + (1 −τ )θ′
        """
        target_net_state = self.target_net.state_dict()
        if soft:
            policy_net_state = self.policy_net.state_dict()
            for key in policy_net_state:
                target_net_state[key] = policy_net_state[key] * Parameters.TAU \
                    + target_net_state[key] * (1 - Parameters.TAU)
        self.target_net.load_state_dict(target_net_state)

    def optimize(self, memory: ReplayMemory) -> None:
        if len(memory) < Parameters.BATCH_SIZE:
            return

        BATCH_SIZE = Parameters.BATCH_SIZE
        device = self.device

        transitions = memory.sample(BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). 이것은 batch-array의 Transitions을 Transition의 batch-arrays로
        # 전환합니다.
        batch = Transition(*zip(*transitions))

        # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
        # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
        # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
        state_action_values = self.policy_net(
            state_batch).gather(1, action_batch)

        # 모든 다음 상태를 위한 V(s_{t+1}) 계산
        # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
        # max(1)[0]으로 최고의 보상을 선택하십시오.
        # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(
                non_final_next_states).max(1)[0]
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * Parameters.GAMMA)\
            + reward_batch

        # 손실 계산
        loss = self.criterion(state_action_values,
                              expected_state_action_values.unsqueeze(1))

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def episode_done(self):
        # Increase one step
        self._current_step += 1

    def get_action(self, state: Any) -> Any:
        with torch.no_grad():
            return self.policy_net(state).max(1)[1].view(1, 1)

    def load_weights(self, path: str) -> None:
        if os.path.exists(path):
            print('Load model weights from', path)
            self.policy_net.load_state_dict(torch.load(path))
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            print('Failed to Load model weights from', path)

    def save_weights(self, path: str) -> None:
        torch.save(self.policy_net.state_dict(), path)
