import socket
import pickle
import time
from typing import Any, Optional

from usl.utils.log_utils import timeit


class SocketCommunicator:
    """
    单客户端-服务端 Socket通信类，支持限速
    - self.sock: 监听 socket
    - self.conn: 通信 socket
    """

    def __init__(
        self,
        host="localhost",
        port=8888,
        is_server=False,
        buffer_size=1024 * 4,
        rate_limit_mbps=0,
        **kwargs,
    ):
        self.host = host
        self.port = port
        self.is_server = is_server
        self.sock: Optional[socket.socket] = None  # 监听 socket
        self.conn: Optional[socket.socket] = None  # 通信 socket
        self.buffer_size = buffer_size
        self.rate_limit_mbps = rate_limit_mbps
        self.max_retry = kwargs.get("max_retry", 10)
        self.timeout = kwargs.get("timeout", 600)
        self.addr = None

        if self.is_server:
            self._init_server()
        else:
            self._init_client()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _init_server(self):
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # ✅ 启用端口复用
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except AttributeError:
                # Windows 或旧内核不支持
                pass

            self.sock.settimeout(self.timeout)
            self.sock.bind((self.host, self.port))
            self.sock.listen(1)
            print(f"[服务端] 正在监听 {self.host}:{self.port} ...")

        except socket.error as e:
            print(f"[服务端] 绑定失败: {e}")
            raise

    def _init_client(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        except AttributeError:
            pass

        self.sock.settimeout(self.timeout)
        retry_count = 0
        while retry_count < self.max_retry:
            try:
                self.sock.connect((self.host, self.port))
                print(f"[客户端] 已连接服务端 {self.host}:{self.port}")
                self.conn = self.sock  # 客户端直接用 sock 通信
                break
            except socket.error as e:
                retry_count += 1
                print(f"[客户端] 连接失败: {e}, 重试 {retry_count}/{self.max_retry}")
                time.sleep(5)
        if not self.conn:
            raise Exception("连接服务端失败")

    def accept_client(self):
        self.conn, self.addr = self.sock.accept()
        print(f"[服务端] 已连接来自 {self.addr}")

    def _sendall_with_rate(self, sock: socket.socket, data: bytes, chunk_bytes: int, rate_mbps: float):
        """分片发送 + 限速"""
        if not rate_mbps or rate_mbps <= 0:
            sock.sendall(data)
            return

        bytes_per_sec = rate_mbps * 1024 * 1024 / 8.0  # Mbps → B/s
        start = time.time()
        sent = 0

        for i in range(0, len(data), chunk_bytes):
            part = data[i : i + chunk_bytes]
            sock.sendall(part)
            sent += len(part)
            expected_elapsed = sent / bytes_per_sec
            now = time.time()
            if expected_elapsed > (now - start):
                time.sleep(expected_elapsed - (now - start))

    # @timeit(info='send info')
    def send(self, obj: Any):
        """发送对象，带长度前缀 + 限速"""
        if not self.conn:
            raise Exception("未建立连接，无法发送")
        try:
            data = pickle.dumps(obj)
            length = len(data)
            self.conn.sendall(length.to_bytes(4, "big"))
            self._sendall_with_rate(self.conn, data, self.buffer_size, self.rate_limit_mbps)
        except socket.error:
            print(f"发送失败或结束训练")
            raise

    def receive(self):
        """接收对象，基于长度前缀"""
        if not self.conn:
            raise Exception("未建立连接，无法接收")

        try:
            length_bytes = self.conn.recv(4)
            if not length_bytes:
                return None
            length = int.from_bytes(length_bytes, "big")

            data = bytearray()
            while len(data) < length:
                packet = self.conn.recv(min(self.buffer_size, length - len(data)))
                if not packet:
                    return None
                data.extend(packet)

            return pickle.loads(data)
        except Exception:
            print(f"接受失败或结束训练")
            raise

    def close(self):
        """关闭连接"""
        try:
            if self.conn:
                try:
                    self.conn.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.conn.close()
                self.conn = None
            if self.sock and self.sock is not self.conn:
                try:
                    self.sock.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.sock.close()
                self.sock = None
        except Exception as e:
            print(f"关闭 socket 出错: {e}")
