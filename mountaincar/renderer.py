import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision.transforms as T

from typing import Any, List
from PIL import Image


def tensor_to_np(t):
    return t.squeeze(0).permute(1, 2, 0).numpy()


def np_to_pil(np_img):
    return Image.fromarray(np.uint8(np_img * 255.), mode="RGB")


def resize(sz: int):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(sz, interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor()
    ])


class Renderer:
    env: gym.Env

    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_screen(self, size: int = 120):
        # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
        # 이것을 Torch order (CHW)로 변환한다.
        screen = self.env.render().transpose((2, 0, 1))

        # float 으로 변환하고,  rescale 하고, torch tensor 로 변환하십시오.
        # (이것은 복사를 필요로하지 않습니다)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
        return resize(size)(screen).unsqueeze(0)

    def display(self, title='Screen Title', *args, **kwargs):
        plt.figure()
        plt.axis('off')
        plt.imshow(self.get_screen(*args, **kwargs).cpu().squeeze(0)
                       .permute(1, 2, 0).numpy(),
                   interpolation='none')
        plt.title(title)
        plt.show()

    @staticmethod
    def save_gif(images: List[Any], output_path: str, **kwargs):
        """Save PIL images into gif file.

        Args:
            images (List[Any]): List of PIL Images
            output_path (str): output path where to save gif
        """
        if len(images) < 1:
            return

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        images[0].save(fp=output_path, format='GIF',
                       append_images=images, save_all=True, **kwargs)
