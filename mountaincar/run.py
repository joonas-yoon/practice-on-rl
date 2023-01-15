# %%
import gym
import json
import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from tqdm import tqdm as progressbar

from agent import Agent, ReplayMemory, DQN

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
print('ipython kernel enabled', is_ipython)

GYM_VERSION = gym.__version__[:4]



# %%
env = gym.make('MountainCar-v0', render_mode="rgb_array", max_episode_steps=3000)
env.reset()


# %%
plt.ion()



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)


# %%
def resize(sz: int):
  return T.Compose([
    T.ToPILImage(),
    T.Resize(sz, interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor()
  ])



# %%
def get_screen(size=120):
    # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
    # 이것을 Torch order (CHW)로 변환한다.
    screen = env.render().transpose((2, 0, 1))

    # float 으로 변환하고,  rescale 하고, torch tensor 로 변환하십시오.
    # (이것은 복사를 필요로하지 않습니다)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
    return resize(size)(screen).unsqueeze(0)



# %%
def display_screen(title='Screen', *args, **kwargs):
    plt.figure()
    plt.axis('off')
    plt.imshow(get_screen(*args, **kwargs).cpu().squeeze(0).permute(1, 2, 0).numpy(), interpolation='none')
    plt.title(title)
    plt.show()



# %%
display_screen(title='Game screen in global', size=240) if is_ipython else None



# %%
env.reset()


# %%
# Get the number of state observations
state, _ = env.reset()
n_actions = env.action_space.n
n_observations = len(state)

policy_net = DQN(n_observations, n_actions=n_actions).to(device)
target_net = DQN(n_observations, n_actions=n_actions).to(device)

# %%
memory = ReplayMemory(10000)

# %%
agent = Agent(
    policy_net=policy_net,
    target_net=target_net,
    optimizer=optim.AdamW(policy_net.parameters(), lr=1e-4, amsgrad=True),
    loss_fn=nn.MSELoss(),
)

# %%
def select_action(state):
    eps = agent.get_epsilon()
    sample = random.random()
    if sample > eps:
        return agent.get_action(state)
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
                'actions': [],
                'duration': 1e8,
            }
        )



# %%
logs = load_log('logs/log.json')

episode_durations: list = logs['episode_durations']
episode_eps: list = logs['episode_eps']
best_score: dict = logs['best']



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
        # print(f'\nsave log to {path}')



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
    ax1.invert_yaxis()
    ax2 = ax1.twinx()
    ax2.set_ylabel('random(%)')

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    df = pd.DataFrame({'duration': durations_t.numpy()}).reset_index()
    df['size'] = -(df['duration'] ** .5)
    sns.scatterplot(data=df, x="index", y="duration", hue="size", size="size", sizes=(4, 128), legend='', ax=ax1)

    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        sns.lineplot(means.numpy(), linestyle='--', ax=ax1, color='purple', linewidth=1.5)

    sns.lineplot(episode_eps, ax=ax2, linestyle='-.', color='#ABB7B7', linewidth=.75, alpha=0.5)

    if is_ipython:
        plt.show()
    else:
        plt.savefig('plot/train.png', dpi=100)
        plt.close()



# %%
import contextlib
from PIL import Image, ImageDraw, ImageFont

def tensor_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()

def np_to_pil(np_img):
    return Image.fromarray(np.uint8(np_img * 255.), mode="RGB")

def action_to_str(action):
    return ['LEFT', '', 'RIGHT'][action]

def save_screenshots(screens: list, actions: list):
    duration = max(len(screens), len(actions))
    if duration < 1: return

    fp_out = f"screenshots/{duration:03}.gif"

    with contextlib.ExitStack() as stack:
        imgs = []
        font = ImageFont.truetype("verdana.ttf", 16)

        for i, tensor in enumerate(screens):
            resize_img = resize(240)(tensor.squeeze(0))
            pil_img = np_to_pil(tensor_to_np(resize_img.cpu()))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f'step: {i}', fill=(200, 0, 0), font=font)
            action = action_to_str(actions[i])
            draw.text((10, 30), f'action: {action}', fill=(0, 0, 200), font=font)
            if i + 1 == len(screens):
                pil_img.info.update({'duration': 1000})
            imgs.append(pil_img)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        imgs[0].save(fp=fp_out, format='GIF', append_images=imgs,
                save_all=True, duration=1000/30, loop=0)


# %%
def get_progress_desc():
    steps = len(episode_durations)
    eps = agent.get_epsilon()
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
agent.load_weights('model.pth')



# %%
num_episodes = 50


# %%
progress = progressbar(total=num_episodes, leave=True)

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    eps_in_episode = []
    screenshots = []
    actions_history = []
    episode_index = len(episode_durations)
    episode_durations.append(0)

    for duration in count(start=1):
        episode_durations[-1] = duration

        # 행동 선택과 수행
        action = select_action(state)
        actions_history.append(action.item())

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
        eps_in_episode.append(agent.get_epsilon())
        if duration < 1000:
            screenshots.append(get_screen(size=240))
        progress.set_description(get_progress_desc() + f" | reward: {reward.item()}")

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        agent.optimize(memory)

        agent.update_target_net()

        # 종료되면 로깅
        if done:
            episode_eps.append(np.mean(eps_in_episode))

            if best_score['duration'] > duration:
                best_score['duration'] = duration
                best_score['step'] = episode_index
                best_score['screens'] = screenshots
                best_score['actions'] = actions_history
                save_screenshots(screenshots, actions_history)
                agent.save_weights('model.pth')

            if (i_episode % 10) == 0:
                save_log('logs/log.json')
                plot_progress()

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
save_screenshots(best_score['screens'], best_score['actions'])
agent.save_weights('model.pth')






