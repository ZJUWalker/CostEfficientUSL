from usl.socket.socket_comm import SocketCommunicator
import torch

print("客户端启动")
send_communicator = SocketCommunicator(
    host="localhost", is_server=False, port=9000, buffer_size=1024 * 4, rate_limit_mbps=0
)  # different port for each node  # 4KB
recv_communicator = SocketCommunicator(
    host="localhost", is_server=False, port=9001, buffer_size=1024 * 4, rate_limit_mbps=230
)  # different port for each node  # 4KB
for i in range(1, 5):
    acti = torch.randn(i, 512, i * 100)
    print("发送数据", i, acti.shape)
    send_communicator.send(acti)
    # print("发送完成,等待接收")
    # for i in range(4):
    acti_recv: torch.Tensor = recv_communicator.receive()
    print('acti', i, acti_recv.shape)
    # send_communicator.send(acti)
    grad_send = torch.randn(i, 512, i * 100)
    recv_communicator.send(grad_send)
    grad_recv: torch.Tensor = send_communicator.receive()
    print('grad', i, grad_recv.shape)


send_communicator.close()
recv_communicator.close()
