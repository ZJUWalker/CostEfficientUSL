from usl.socket.socket_comm import SocketCommunicator
import torch

print("客户端启动")
send_communicator = SocketCommunicator(
    host="localhost", is_server=False, port=8000, buffer_size=1024 * 4, rate_limit_mbps=230
)  # different port for each node  # 4KB
recv_communicator = SocketCommunicator(
    host="localhost", is_server=False, port=8001, buffer_size=1024 * 4, rate_limit_mbps=230
)  # different port for each node  # 4KB
acti = torch.randn(1000, 1000)
send_communicator.send(acti)
print("发送完成,等待接收")
acti_recv = recv_communicator.receive()
print(acti_recv.shape)

send_communicator.close()
recv_communicator.close()
