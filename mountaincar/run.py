# %%
import random
from itertools import count

import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
from agent import DQN, Agent, ReplayMemory
from logger import Logger
from renderer import (Renderer, ScreenshotRecorder, np_to_pil, resize,
                      tensor_to_np)
from tqdm import tqdm as progressbar

# matplotlib 설정
is_interactive = 'inline' in matplotlib.get_backend()
if is_interactive:
    from IPython import display
print('Interactive python kernel enabled:', is_interactive)

GYM_VERSION = gym.__version__[:4]

PLOT_UPDATE = 10


# %%
env = gym.make('MountainCar-v0', render_mode="rgb_array", max_episode_steps=3000)
env.reset()


# %%
plt.ion()



# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device:', device)



# %%
renderer = Renderer(env)

if is_interactive:
    renderer.display(title='Game screen in global', size=240)



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
def log_format(logs: dict) -> str:
    if len(logs) < 1: return ""

    episodes = logs['episodes']
    scores = list(filter(np.isfinite, logs['scores']))

    has = len(scores) > 0
    if len(episodes) < 1: return ""

    best_score = np.min(scores) if has else np.nan
    best_episode = np.argmin(scores) if has else -1
    score = scores[-1] if has else np.nan
    score_mean = np.mean(scores[-30:]) if has else np.nan
    rand = logs['eps'][-1]

    strs = [
        f"episode={len(episodes)}",
        f"score={score}",
        f"score_mean30={score_mean:.2f}",
        f"best_score={best_score:.0f} at {best_episode}",
        f"random={100.*rand:.2f}%",
    ]
    return " | ".join(strs)


logger = Logger(path='logs/log.json', formatter=log_format)
logger.load()

# %%
def plot_progress(show_result=False):
    if is_interactive:
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
    ax1.set_ylabel('Score')
    ax1.invert_yaxis()
    ax2 = ax1.twinx()
    ax2.set_ylabel('random(%)')

    scores = torch.tensor(logger['scores'], dtype=torch.float)
    df = pd.DataFrame({'score': scores.numpy()}).reset_index()
    df['size'] = -(df['score'] ** .5)
    sns.scatterplot(data=df, x="index", y="score", hue="size", size="size", sizes=(4, 128), legend='', ax=ax1)

    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    if len(scores) >= 100:
        means = scores.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        sns.lineplot(means.numpy(), linestyle='--', ax=ax1, color='purple', linewidth=1.5)

    sns.lineplot(logger['eps'], ax=ax2, linestyle='-.', color='#ABB7B7', linewidth=.75, alpha=0.5)

    if is_interactive:
        plt.show()
    else:
        plt.savefig('plot/train.png', dpi=100)
        plt.close()



# %%
import contextlib

from PIL import ImageDraw, ImageFont


def action_to_str(action):
    return ['LEFT', '', 'RIGHT'][action]

def save_screenshots(screens: list, actions: list):
    duration = min(len(screens), len(actions))
    if duration < 1: return

    fp_out = f"screenshots/{len(actions):04}.gif"

    imgs = []
    font = ImageFont.truetype("verdana.ttf", 16)

    with contextlib.ExitStack():
        for i, tensor in enumerate(screens):
            resize_img = resize(240)(tensor.squeeze(0))
            pil_img = np_to_pil(tensor_to_np(resize_img.cpu()))
            draw = ImageDraw.Draw(pil_img)
            draw.text((10, 10), f'step: {i}', fill=(200, 0, 0), font=font)
            action = action_to_str(actions[i])
            draw.text((10, 30), f'action: {action}', fill=(0, 0, 200), font=font)
            imgs.append(pil_img)

        Renderer.save_gif(images=imgs, output_path=fp_out, duration=1000/30, loop=0)

# %%
agent.load_weights('model.pth')

prev_episode_runs = len(logger)
agent._current_step = prev_episode_runs


# %%
num_episodes = 20


# %%
progress = progressbar(total=num_episodes, leave=True)
screenshots = ScreenshotRecorder(max_frames=1000)

best = dict(score=1e8, images=[], actions=[])

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    logger.add({
        'scores': np.inf,
        'episodes': i_episode + prev_episode_runs,
        'actions': [],
        'eps': agent.get_epsilon(),
    })
    screenshots.flush()

    actions_history = []

    for duration in count(start=1):
        logger.update({
            'scores': duration,
        })

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
        screenshots.add_frame(renderer, size=240)
        progress.set_description(f"{str(logger)} | reward: {reward.item()}")

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        agent.optimize(memory)

        agent.update_target_net()

        # 종료되면 로깅
        if done:
            if logger.best('scores') >= duration:
                best = dict(score=duration, images=screenshots.images, actions=actions_history)
                save_screenshots(best['images'], best['actions'])
                agent.save_weights('model.pth')

            logger.update({
                'scores': duration,
                'actions': actions_history,
            })
            logger.save()

            if i_episode % max(1, PLOT_UPDATE) == 0:
                plot_progress()

            agent.episode_done()
            break

    progress.update(1)

print('\nComplete\n')
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
    if not is_interactive: return
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(111)
    plt.axis('off')
    imgs = [[
        plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0).numpy(), animated=True),
        plt.text(0.0, 1.0, f'duration: {d}', transform=ax.transAxes),
    ] for d, img in zip(range(best['score']), best['images'])]

    ani = animation.ArtistAnimation(fig, imgs, interval=33, repeat_delay=1000, blit=True)
    return display.HTML(ani.to_jshtml())

display_animation()



# %%
save_screenshots(best['images'], best['actions'])
agent.save_weights('model.pth')






