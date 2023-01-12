# %%
import os
import gym
import math
import json
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm as progressbar

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print('ipython kernel enabled', is_ipython)

GYM_VERSION = gym.__version__[:4]

# %%
env = gym.make('CartPole-v1', render_mode="rgb_array").unwrapped

# %%
plt.ion()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# %%
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# %%
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# %%
SCREEN_SIZE_TRAIN = 40

# %%
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

# %%
def resize(sz: int):
  return T.Compose([
    T.ToPILImage(),
    T.Resize(sz, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor()
  ])

# %%
def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# %%
def get_screen(fullscreen=False, size=SCREEN_SIZE_TRAIN):
    # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
    # 이것을 Torch order (CHW)로 변환한다.
    screen = env.render().transpose((2, 0, 1))

    if not fullscreen:
        # 카트는 아래쪽에 있으므로 화면의 상단과 하단을 제거하십시오.
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # 카트를 중심으로 정사각형 이미지가 되도록 가장자리를 제거하십시오.
        screen = screen[:, :, slice_range]

    # float 으로 변환하고,  rescale 하고, torch tensor 로 변환하십시오.
    # (이것은 복사를 필요로하지 않습니다)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
    return resize(size)(screen).unsqueeze(0)

# %%
def display_screen(title='Screen', *args, **kwargs):
    env.reset()
    plt.figure()
    plt.imshow(get_screen(*args, **kwargs).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title(title)
    plt.show()

# %%
display_screen(title='Game screen in global', fullscreen=True, size=240) if is_ipython else None

# %%
display_screen(title='Example extracted screen') if is_ipython else None

# %%
env.reset()

# %%
BATCH_SIZE = 256   # is the number of transitions sampled from the replay buffer
GAMMA = 0.999      # is the discount factor as mentioned in the previous section
EPS_START = 0.95   # is the starting value of epsilon
EPS_END = 0.05     # is the final value of epsilon
EPS_DECAY = 0.925  # controls the rate of exponential decay of epsilon, higher means a slower deca
TAU = 0.005        # is the update rate of the target network
LR = 1e-4          # is the learning rate of the AdamW optimizer

# %%
def get_epsilon(step: int, decay: float):
    eps = (EPS_START - EPS_END) * ((decay + 1e-12) ** step) + EPS_END
    return eps

# %%
plt.figure(figsize=(6, 3))
plt.ylim((0.0, 1.0))
plt.title('epsilon graph per decay')
for decay, style in zip(
  [0, 0.95, 0.99, 0.995, 0.999, 1.0],
  ['-', '--', ':', '--', '-.', '-']
):
    step_range = range(1000)
    sns.lineplot(x=step_range, y=[get_epsilon(step, decay) for step in step_range], label=f'decay={decay}', linestyle=style)
plt.show(False) if is_ipython else plt.close()

# %%
# gym 행동 공간에서 행동의 숫자를 얻습니다.
n_actions = env.action_space.n

# Get the number of state observations
if GYM_VERSION == '0.26':
    state, _ = env.reset()
elif GYM_VERSION == '0.25':
    state, _ = env.reset(return_info=True)
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# %%
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

# %%
def load_model_weights(path):
    if os.path.exists(path):
        print('Load model weights from', path)
        policy_net.load_state_dict(torch.load(path))
        target_net.load_state_dict(policy_net.state_dict())
    else:
        print('Failed to Load model weights from', path)

def save_model_weights(path):
    torch.save(policy_net.state_dict(), path)

# %%
def select_action(state, eps):
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

# %%
def load_log(path):
    try:
        with open(path, 'r') as f:
            loaded = json.load(fp=f)
            if not 'screens' in loaded['best']:
                loaded['best']['screens'] = []
            print('Log loaded from', path)
            return loaded
    except Exception as e:
        print('Create new log', e)
        return dict(
            episode_durations=[],
            episode_eps=[],
            best={
                'step': 0,
                'screens': [],
                'duration': 0,
            }
        )

# %%
logs = load_log('logs/log.json')

episode_durations = logs['episode_durations']
episode_eps = logs['episode_eps']
best_score = logs['best']

# %%
def exclude_keys(dictionary, keys):
    return {
        key: value for key, value in dictionary.items()
        if key not in keys
    }

def save_log(path):
    with open(path, 'w') as f:
        obj = dict(
            episode_durations=episode_durations,
            episode_eps=episode_eps,
            best=exclude_keys(best_score, ['screens']),
        )
        json.dump(obj, fp=f)
        print(f'\nsave log to {path}')

# %%
def plot_progress(show_result=False):
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

    _, ax1 = plt.subplots(constrained_layout=True, sharex=True)

    if show_result:
        plt.title('Result')
    else:
        plt.title('Training...')

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration')
    ax2 = ax1.twinx()
    ax2.set_ylabel('eps')

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    df = pd.DataFrame({'duration': durations_t.numpy()}).reset_index()
    df['size'] = (df['duration'] / 10) ** 2
    sns.scatterplot(data=df, x="index", y="duration", hue="size", size="size", sizes=(4, 64), legend='', ax=ax1)

    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        sns.lineplot(means.numpy(), linestyle='--', ax=ax1, color='purple', linewidth=1.5)

    sns.lineplot(episode_eps, ax=ax2, linestyle='-.', color='#ABB7B7', linewidth=.75, alpha=0.5)

    if is_ipython:
        plt.show()
    else:
        plt.savefig('plot/train.png', dpi=300)
        plt.close()

# %%
import contextlib
from PIL import Image, ImageDraw, ImageFont

def tensor_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()

def np_to_pil(np_img):
    return Image.fromarray(np.uint8(np_img * 255.), mode="RGB")

def save_screenshots(screens):
    duration = len(screens)

    fp_out = f"screenshots/cartpolev1_{duration:03}.gif"

    with contextlib.ExitStack() as stack:
        imgs = []
        font = ImageFont.truetype("verdana.ttf", 16)

        for i, tensor in enumerate(screens):
            resize_img = resize(240)(tensor.squeeze(0))
            pil_img = np_to_pil(tensor_to_np(resize_img.cpu()))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f'duration: {i}', fill=(200, 0, 0), font=font)
            if i + 1 == len(screens):
                pil_img.info.update({'duration': 1000})
            imgs.append(pil_img)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        imgs[0].save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=1000/30, loop=0)

# %%
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
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
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1)[0]으로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # 손실 계산
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# %%
def get_progress_desc():
    steps = len(episode_durations)
    eps = get_epsilon(steps, EPS_DECAY)
    strs = [
        f"steps={steps}",
        f"du={np.mean(episode_durations[-1:]):.0f}",
        f"best_du={best_score['duration']}",
        f"best_ep={best_score['step']}",
        f"du_mean30={np.mean(episode_durations[-30:]):.3f}",
        f"random_action={eps*100.:.2f}%",
    ]
    return " | ".join(strs)

# %%
load_model_weights('model.pth')

# %%
if torch.cuda.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

# %%
progress = progressbar(total=num_episodes, leave=True)

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    if GYM_VERSION == '0.26':
        state, _ = env.reset()
    elif GYM_VERSION == '0.25':
        state, _ = env.reset(return_info=True)

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    eps_in_episode = []
    screenshots = []
    episode_index = len(episode_durations)

    for duration in count(start=1):
        # 행동 선택과 수행
        eps_threshold = get_epsilon(episode_index, EPS_DECAY)
        action = select_action(state, eps_threshold)

        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 로깅
        eps_in_episode.append(eps_threshold)
        screenshots.append(get_screen(fullscreen=True, size=160))
        progress.desc = get_progress_desc()

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        # 종료되면 로깅
        if done:
            episode_durations.append(duration)
            episode_eps.append(np.mean(eps_in_episode))
            if (i_episode % 10) == 0:
                plot_progress()
                save_log('logs/log.json')
            if best_score['duration'] < duration:
                best_score['duration'] = duration
                best_score['step'] = episode_index
                best_score['screens'] = screenshots
                save_screenshots(screenshots)
                save_model_weights('model.pth')
            del screenshots
            break

    progress.update(1)

print('Complete')
env.render()
env.close()
progress.close()

# %%
plot_progress(show_result=True)
plt.ioff()
plt.show()

# %%
import matplotlib.animation as animation

def display_animation():
    if not is_ipython: return
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    plt.axis('off')
    imgs = [[
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy(), animated=True),
        plt.text(0.0, 1.0, f'duration: {d}', transform=ax.transAxes),
    ] for d, img in zip(range(best_score['duration']), best_score['screens'])]

    ani = animation.ArtistAnimation(fig, imgs, interval=33, repeat_delay=1000, blit=True)
    return display.HTML(ani.to_jshtml())

display_animation()

# %%
save_screenshots(best_score['screens'])
save_model_weights('model.pth')
