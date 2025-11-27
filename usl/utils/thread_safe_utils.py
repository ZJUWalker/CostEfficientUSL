import threading


class AtomicInt:
    def __init__(self, value: int = 0):
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> int:
        """线程安全地读取整数"""
        with self._lock:
            return self._value

    def set(self, value: int = 0):
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


class AtomicBool:
    def __init__(self, value: bool = False):
        self._value = value
        self._lock = threading.Lock()

    def get(self) -> bool:
        """线程安全地读取布尔值"""
        with self._lock:
            return self._value

    def set(self, value: bool = False):
        """线程安全地写入布尔值"""
        with self._lock:
            self._value = value

    def toggle(self) -> bool:
        """原子切换布尔值"""
        with self._lock:
            self._value = not self._value
            return self._value
