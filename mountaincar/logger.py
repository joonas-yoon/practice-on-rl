import json
import numpy as np

from typing import Any, Callable


def exclude_keys(dictionary, keys):
    return {
        key: value for key, value in dictionary.items()
        if key not in keys
    }


class Logger:
    logs: dict = {
        '_length': 0,
    }

    def __init__(self, path: str, formatter: Callable = str) -> None:
        self.path = path
        self.formatter = formatter

    def add(self, data: dict) -> None:
        for key, value in data.items():
            if not key in self.logs:
                self.logs[key] = []
            self.logs[key].append(value)
        self.logs['_length'] += 1

    def update(self, data: dict) -> None:
        for key, value in data.items():
            if not key in self.logs:
                self.logs[key] = []
            self.logs[key][-1] = value

    def load(self) -> None:
        try:
            with open(self.path, 'r') as fp:
                self.logs = json.load(fp=fp)
        except Exception as e:
            print('Create new log:\n', e)
            self.logs = {}

    def save(self, excludes: list = []) -> None:
        with open(self.path, 'w') as fp:
            excluded_logs = exclude_keys(self.logs, excludes)
            json.dump(excluded_logs, fp=fp)

    def best(self, key: str, method: Callable = np.min) -> Any:
        scores = list(filter(np.isfinite, self.logs[key]))
        return method(scores)

    def __getitem__(self, __name: str) -> list:
        return self.logs[__name]

    def __str__(self) -> str:
        return self.formatter(self.logs)

    def __len__(self) -> int:
        return self.logs['_length']
