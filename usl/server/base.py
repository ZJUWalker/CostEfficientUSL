from dataclasses import dataclass
from enum import Enum

import threading

MAX_DIM = 8  # 你预设的最大维数
MSG_TYPE_ACTIVATION = 1
MSG_TYPE_GRADIENT = 2
MSG_TYPE_ATTENTION_MASK = 3
MSG_TYPE_POSITION_EMBEDDINGS = 4


class PipelineMode(Enum):
    GPIPE = "gpipe"
    PIPE_DREAM_STRICT = "pipedream_strict"  # use pipedream
    PIPE_DREAM_WC = "pipedream_wc"  # use pipedream's warm-up and cool-down phases,no 1f1b
    PIPE_DREAM_WC_EAGER = "pipedream_wc_eager"  # use pipedream's warm-up and cool-down phases, no 1f1b,but eager
    NAIVE = "naive"


def convert_pipeline_mode(pmode: str) -> str:

    try:
        pmode = pmode.lower()
        if 'pds' in pmode:
            return PipelineMode.PIPE_DREAM_STRICT
        elif 'pdwc' in pmode and 'e' not in pmode:
            return PipelineMode.PIPE_DREAM_WC
        elif 'pdwce' in pmode:
            return PipelineMode.PIPE_DREAM_WC_EAGER
        return PipelineMode(value=pmode)
    except KeyError:
        return PipelineMode.NAIVE


# -------------------------------
# Args
# -------------------------------
@dataclass
class ServerArgs:
    port: int = 8000
    step: int = 5
    use_lora: bool = False
    model: str = "meta-llama/llama3.2-1b"
    server_device: str = "cuda:0"
    split_point: int = 2
    dataset: str = "gsm8k"
    learning_rate: float = 5e-4
    pipeline_mode: PipelineMode = PipelineMode.GPIPE  # "strict" or "loose"
    # NOTE: original had a typo 'rete_limit_mbps'. Kept for backward-compat, but also expose the correct name.
    rate_limit_mbps: float = 10
    offload_activation: bool = False
    offload_activation_mb_num: int = 0
    micro_batch_size: int = 1
    batch_size: int = 4
    prof: bool = False

    def effective_rate_limit(self) -> float:
        # Prefer the correctly spelled one if provided
        return self.rate_limit_mbps if self.rate_limit_mbps is not None else self.rate_limit_mbps


class AtomicInt:
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> int:
        """线程安全地读取整数"""
        with self._lock:
            return self._value

    def set(self, value: int):
        """线程安全地写入整数"""
        with self._lock:
            self._value = value

    def add(self, delta: int) -> int:
        """线程安全地加减整数，返回新值"""
        with self._lock:
            self._value += delta
            return self._value

    def increment(self) -> int:
        """原子 +1"""
        return self.add(1)

    def decrement(self) -> int:
        """原子 -1"""
        return self.add(-1)

    def swap(self, value: int) -> int:
        """原子交换：返回旧值，写入新值"""
        with self._lock:
            old = self._value
            self._value = value
            return old
