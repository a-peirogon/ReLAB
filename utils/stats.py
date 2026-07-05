from __future__ import annotations
import csv
import os
from collections import deque
from typing import Any, Optional

class EpisodeStats:

    def __init__(self):
        self.total_reward: float = 0.0
        self.score: int = 0
        self.steps: int = 0

    def update(self, reward: float, info: dict[str, Any]) -> None:
        self.total_reward += reward
        self.score = info.get('score', self.score)
        self.steps = info.get('steps', self.steps + 1)

class MovingAverage:

    def __init__(self, window: int=100):
        self.window = window
        self._values: deque[float] = deque(maxlen=window)

    def push(self, value: float) -> None:
        self._values.append(value)

    @property
    def mean(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def __len__(self) -> int:
        return len(self._values)

class CSVLogger:

    def __init__(self, path: str, fields: list[str]):
        self.path = path
        self.fields = fields
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self._write_header = not os.path.exists(path)

    def log(self, **kwargs: Any) -> None:
        row = {field: kwargs.get(field, '') for field in self.fields}
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(row)
