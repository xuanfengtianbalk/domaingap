from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import numpy as np


class BaseKeypoint(nn.Module, ABC):

    def __init__(self, in_channels, out_channels, network, cfg):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.network=network(cfg['heads'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


