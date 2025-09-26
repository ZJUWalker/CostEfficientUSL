import os
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass
import logging
from queue import Queue, Empty
import socket
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData, save_gantt_chart_data, plot_gantt_per_batch
from usl.utils.tensor_utils import pad_inputs
from usl.offload import ActivationOffload, ModelParamOffload, OptimizerStateOffload


@dataclass
class ClientArgs:
    port: int = 8000
    model: str = "meta-llama/llama3.2-1b"
    batch_size: int = 4
    max_seq_len: int = 256
    step: int = 10
    dataset: str = "gsm8k"
    epoch: int = 1
    split_point: int = 2
    learning_rate: float = 5e-4
    rate_mbps: float = 10  # rate in Mbps
    micro_batch_size: int = 1
    async_io: bool = False
    offload_activation: bool = False
    offload_model_state: bool = False
    sort_batch: bool = False


class Client:
    def __init__(
        self,
        client_args: ClientArgs,
        head_model: nn.Module,
        tail_model: nn.Module,
        tokenizer: AutoTokenizer,
        client_device: str,
        train_logger: logging.Logger,
        dataset_train: Dataset,
        dataset_test: Dataset,
        num_workers: int = 4,
        normalize_loss: bool = True,  # NEW: 按 accum_steps 归一化 loss
    ):
        # ---- Model & optimizer
        self.client_device = client_device
        self.client_args = client_args
        self.head_model = head_model.to(self.client_device)
        if self.client_args.offload_model_state:
            self.tail_model = tail_model.to("cpu")  # 初始化状态：把tail_model放在cpu中
        else:
            self.tail_model = tail_model.to(self.client_device)  # 初始化状态：把tail_model放在cuda中
        self.tokenizer = tokenizer
        self.logger = train_logger
        self.train_loader = dataset_train
        self.test_loader = dataset_test
        self.local_ep = client_args.epoch  # ✅ 点操作符访问
        self.lr = client_args.learning_rate  # ✅ 点操作符访问
        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=self.lr)
        self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=self.lr)

        # ---- Communicator
        self.communicator = SocketCommunicator(
            is_server=False,
            port=client_args.port,
            buffer_size=1024 * 1024,  # 1MB
            rate_limit_mbps=client_args.rate_mbps,
        )

        # ---- Metrics
        self.compute_time = 0
        self.max_mem_allocated_mb = 0
        self.normalize_loss = normalize_loss
        self.losses = []
        self.profile_data: List[GanttChartData] = []
        # ---- Executors & coordination
        self.main_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=2)
        self.compute_future: Optional[Future] = None
        self.server_future: Optional[Future] = None
        self.stop_event = threading.Event()
        torch.cuda.set_stream(torch.cuda.Stream(self.client_device))  # set cuda compute stream

        # ----Parameter Efficient Offloading
        if self.client_args.offload_activation:
            self.head_activation_collector = ActivationOffload(
                base_model=self.head_model,
                offload_threshold=1024,
                device=self.client_device,
            )
        if self.client_args.offload_model_state:
            # Head model的模型参数和优化器状态卸载器
            self.head_model_param_collector = ModelParamOffload(
                base_model=self.head_model,
                offload_threshold=1024,
                device=self.client_device,
            )
            self.head_model_optimizer_collector = OptimizerStateOffload(
                optimizer=self.optimizer_head,
                offload_threshold=1024,
                device=self.client_device,
            )
            # Tail model的模型参数和优化器状态卸载器
            self.tail_model_param_collector = ModelParamOffload(
                base_model=self.tail_model,
                offload_threshold=1024,
                device=self.client_device,
            )
            self.tail_model_optimizer_collector = OptimizerStateOffload(
                optimizer=self.optimizer_tail,
                offload_threshold=1024,
                device=self.client_device,
            )
            self.tail_model_optimizer_collector.offload()  # 先把tail的优化器状态也放在cpu
        self.pin_on_gpu_tensors_idx = []  # used to pin tensors on GPU that don't need to be offloaded

        # ---- Queues (compute pipeline)
        self.activation_to_server_queue: Queue[Payload] = Queue()  # used for serve fwd
        self.activation_from_server_queue: Queue[Payload] = Queue()  # used for tail fwd
        self.gradient_to_server_queue: Queue[Payload] = Queue()  # used for server bwd
        self.gradient_from_server_queue: Queue[Payload] = Queue()  # used for head bwd

        # -----Other args used to save tensors
        self.head_fwd_activation_dict: Dict[int, torch.Tensor] = {}
        self.pos_embedding_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.atten_mask_dict: Dict[int, torch.Tensor] = {}
        self.labels_dict: Dict[int, torch.Tensor] = {}

    @torch.no_grad()
    # @timeit()
    def _to_cpu_payload(
        self,
        output: Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]],
        *,
        token: str,
        group_id: str,
        mb_idx: int,
        mb_total: int,
        is_activation: bool,  # activation or gradient
    ) -> Dict:
        # save essential tensors to dict for tail and bwd phase to use
        attn_gpu = output[1] if (len(output) > 1 and output[1] is not None) else None
        if attn_gpu is not None:
            self.pin_on_gpu_tensors_idx.append(attn_gpu.data_ptr())  # pos_embedding和attn_mask在tail fwd的时候需要，所以不能卸载
        self.atten_mask_dict[mb_idx] = attn_gpu
        pos_gpu = output[2] if (len(output) > 2 and output[2] is not None) else None
        self.pos_embedding_dict[mb_idx] = pos_gpu
        if pos_gpu is not None:
            self.pin_on_gpu_tensors_idx.extend([t.data_ptr() for t in pos_gpu])
        # send essential tensors to server
        act_cpu = output[0].clone().detach().cpu()
        attn_cpu = attn_gpu.detach().cpu() if attn_gpu is not None else None
        pos_cpu = tuple([t.cpu() for t in pos_gpu]) if pos_gpu is not None else None
        payload = Payload(
            tensor=act_cpu,
            is_activation=is_activation,
            is_training=True,
            # —— 元信息 ——（server 将用 token 作为上下文 key）
            token=token,
            group_id=group_id,
            mb_idx=mb_idx,
            mb_total=mb_total,
            attention_mask=attn_cpu,
            position_embeddings=pos_cpu,
        )
        return payload

    def _payload_nbytes(self, payload: Dict[str, Any]) -> int:
        """计算 payload 中所有 tensor 的占用字节数（单位: Byte）"""
        total = 0

        def tensor_nbytes(t: torch.Tensor) -> int:
            return t.numel() * t.element_size()

        for key, val in payload.items():
            if isinstance(val, torch.Tensor):
                total += tensor_nbytes(val)
            elif isinstance(val, (list, tuple)):
                for v in val:
                    if isinstance(v, torch.Tensor):
                        total += tensor_nbytes(v)
        return total

    def _split_micro(self, tensor: torch.Tensor, micro_bs: int) -> List[torch.Tensor]:
        # 假设 dim=0 为 batch 维
        chunks = []
        total = tensor.size(0)
        for start in range(0, total, micro_bs):
            end = min(start + micro_bs, total)
            chunks.append(tensor[start:end])
        return chunks

    def _split_micro_with_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        micro_bs: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        pad_token_id = self.tokenizer.pad_token_id
        n, seq_len = input_ids.size()
        sequences = []

        # 逐行处理，跳过空序列
        for i in range(n):
            valid_len = attention_mask[i].sum().item()
            if valid_len == 0:
                continue  # 跳过空序列

            ids_row = input_ids[i][:valid_len]
            mask_row = attention_mask[i][:valid_len]
            sequences.append((ids_row, mask_row))

        if not sequences:  # 所有序列都为空
            return []

        # 按长度排序和分块
        sequences.sort(key=lambda x: x[0].size(0), reverse=True)

        chunks = []
        for start in range(0, len(sequences), micro_bs):
            batch = sequences[start : start + micro_bs]
            max_len = max(seq[0].size(0) for seq in batch)

            # 更高效的批处理方式
            ids_batch = torch.full((len(batch), max_len), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)
            mask_batch = torch.zeros(len(batch), max_len, dtype=attention_mask.dtype, device=attention_mask.device)

            for i, (ids, mask) in enumerate(batch):
                ids_batch[i, : ids.size(0)] = ids
                mask_batch[i, : mask.size(0)] = mask

            chunks.append((ids_batch, mask_batch))

        return chunks

    def _resolve_micro_bs(self, total_bs: int) -> Tuple[int, int]:
        # 兼容没有 micro_batch_size 的 ClientArgs
        micro_bs = getattr(self.client_args, "micro_batch_size", None) or self.client_args.batch_size
        if micro_bs <= 0:
            micro_bs = self.client_args.batch_size
        if micro_bs > total_bs:
            micro_bs = total_bs
        grad_accum_steps = (total_bs + micro_bs - 1) // micro_bs
        return micro_bs, grad_accum_steps

    def _head_fwd_micro(
        self,
        group_id: str,
        mb_idx: int,
        grad_accum_steps: int,
        x: torch.Tensor,
        m: torch.Tensor,
        y: torch.Tensor,
    ):
        self.profile_data[mb_idx].head_fwd_timestamp[0] = time.time()
        head_outs = self.head_model.forward(x, m)
        torch.cuda.synchronize()
        head_outs[0].requires_grad_(True)
        self.head_fwd_activation_dict[mb_idx] = head_outs[0]
        token = uuid.uuid4().hex
        payload = self._to_cpu_payload(
            head_outs,
            token=token,
            group_id=group_id,
            mb_idx=mb_idx,
            mb_total=grad_accum_steps,
            is_activation=True,
        )
        self.profile_data[mb_idx].head_fwd_timestamp[1] = time.time()
        return payload

    def _tail_fwd_bwd_micro(self, server_forward_output: Payload) -> Tuple[Payload, torch.Tensor]:
        # print(f"[Client] micro {head_fwd.mb_idx} tial recv,future elapsed:[{head_fwd.recv_future.start}, {head_fwd.recv_future.end}]")
        # token = server_forward_output.get("token")
        # group_id = server_forward_output.get("group_id")
        # mb_idx = int(server_forward_output.get("mb_idx"))
        # mb_total = int(server_forward_output.get("mb_total"))
        # 解析 server 传来的 payload
        token = server_forward_output.token
        group_id = server_forward_output.group_id
        mb_idx = server_forward_output.mb_idx
        mb_total = server_forward_output.mb_total

        self.profile_data[mb_idx].tail_fwd_timestamp[0] = time.time()
        activation_cpu: torch.Tensor = server_forward_output.tensor
        activation_to_tail = activation_cpu.to(self.client_device).requires_grad_(True)
        # tail forward
        output: CausalLMOutputWithPast = self.tail_model.forward(
            hidden_states=activation_to_tail,
            attention_mask=self.atten_mask_dict[mb_idx],
            position_embeddings=self.pos_embedding_dict[mb_idx],
            labels=self.labels_dict[mb_idx],
        )
        torch.cuda.synchronize()
        self.profile_data[mb_idx].tail_fwd_timestamp[1] = time.time()
        # 可选：按 accum 步数归一化，保证与“单大 batch 一次性训练”的梯度数值一致
        loss = output.loss / mb_total if self.normalize_loss else output.loss
        self.losses.append(loss.item())
        self.profile_data[mb_idx].tail_bwd_timestamp[0] = time.time()
        loss.backward()
        torch.cuda.synchronize()
        grad_payload = self._to_cpu_payload(
            output=(activation_to_tail.grad,),
            token=token,
            group_id=group_id,
            mb_idx=mb_idx,
            mb_total=mb_total,
            is_activation=False,
        )
        self.profile_data[mb_idx].tail_bwd_timestamp[1] = time.time()

        return grad_payload, loss

    def _head_bwd_micro(self, server_bwd_output: Payload):
        assert server_bwd_output.is_activation == False, "should be gradient,but activation recieved"
        mb_idx = server_bwd_output.mb_idx
        grad_cpu: torch.Tensor = server_bwd_output.tensor
        # load grad and activation
        self.profile_data[mb_idx].head_bwd_timestamp[0] = time.time()
        grad_to_head = grad_cpu.to(self.client_device)
        head_activation = self.head_fwd_activation_dict[mb_idx]
        # real head model backward
        head_activation.backward(grad_to_head)
        torch.cuda.synchronize()
        self.profile_data[mb_idx].head_bwd_timestamp[1] = time.time()
        # remove not needed tensors to save memory
        del (
            self.head_fwd_activation_dict[mb_idx],
            self.pos_embedding_dict[mb_idx],
            self.atten_mask_dict[mb_idx],
            self.labels_dict[mb_idx],
        )
        pass

    def _check_communication(self):
        # 异步 send
        if not self.communicator.conn:
            # Wait for run() to set a connection; this path is rarely hit.
            while not self.stop_event.is_set() and not self.communicator.conn:
                time.sleep(0.05)
        return self.communicator.conn

    # TODO:处理client端的send操作
    def _head_client_send(self):
        self._check_communication()
        while not self.stop_event.is_set():
            try:
                payload: Optional[Payload | Dict] = self.activation_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    start_send = time.time()
                    self.communicator.send(payload)
                    end_time = time.time()
                    if isinstance(payload, dict) and 'stop' in payload:
                        print('send stop flag')
                        continue
                    else:
                        mb_idx = payload.mb_idx
                        self.profile_data[mb_idx].head_fwd_send_timestamp[0] = start_send
                        self.profile_data[mb_idx].head_fwd_send_timestamp[1] = end_time
                else:
                    continue
            except Empty:
                pass
            try:
                payload = self.gradient_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    # print(f'send gradient payload')
                    start_send = time.time()
                    self.communicator.send(payload)
                    end_time = time.time()
                    mb_idx = payload.mb_idx
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[0] = start_send
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[1] = end_time
                else:
                    continue
            except Empty:
                pass
        print("client send thread exit")
        pass

    # TODO: 处理recv操作
    def _handle_server_send(self):
        self._check_communication()
        self.communicator.conn.settimeout(60.0)
        while not self.stop_event.is_set():
            try:
                data: Optional[Dict | Payload] = self.communicator.receive()
            except socket.timeout:
                print("socket timeout")
                continue
            except Exception as e:
                break
            if data is None:
                break
            if isinstance(data, dict) and 'profile' in data:
                print(f'get profile data')
                self._save_profile_res(data['profile'])
                self.stop_event.set()
                break
            if data.is_activation:
                self.activation_from_server_queue.put(data)
            else:
                self.gradient_from_server_queue.put(data)
        print("server send thread exit")
        # else:
        #     self.logger.warning(f"Unknown data received: {data}")

    def _do_pipeline(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        offload_activation = self.client_args.offload_activation
        offload_model_state = self.client_args.offload_model_state
        # micro batch head fwd and send
        for mb_idx in range(grad_accum_steps):
            #  tail fwd and send
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
            pass
        # offload all activations created by head model fwd
        if offload_activation:
            self.head_activation_collector.offload(except_tensor_idx_list=self.pin_on_gpu_tensors_idx)
        # print(f'Global batch id -> {global_batch_idx} finished head forward')
        
        # TODO head模型参数和优化器状态卸载至CPU
        # TODO 从CPU加载tail模型参数和优化器状态到GPU
        if offload_model_state:
            self.head_model_param_collector.offload()
            self.head_model_optimizer_collector.offload()
            self.tail_model_param_collector.reload()
            self.tail_model_optimizer_collector.reload()
        
        batch_loss = 0
        no_tail_fwd_bwd_mb_list = [False] * grad_accum_steps
        while True:
            if not all(no_tail_fwd_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_activation_payload = self.activation_from_server_queue.get(timeout=0.001)
                    if server_activation_payload is not None:
                        mb_idx = server_activation_payload.mb_idx
                        no_tail_fwd_bwd_mb_list[mb_idx] = True
                        grad_payload, loss = self._tail_fwd_bwd_micro(server_activation_payload)
                        batch_loss += loss.item()
                        # print(f'Global batch id -> {global_batch_idx} finished micro {mb_idx} tail forward and backward')
                        self.gradient_to_server_queue.put(grad_payload)
                except Empty:
                    continue
            else:
                break
            pass
        # print(f'Global batch id -> {global_batch_idx} finished tail forward and backward')
        if offload_activation:
            self.head_activation_collector.reload()
        
        # TODO tail模型参数和优化器状态卸载至CPU
        # TODO 从CPU加载head模型参数和优化器状态到GPU
        if offload_model_state:
            self.tail_model_param_collector.offload()
            self.tail_model_optimizer_collector.offload()
            self.head_model_param_collector.reload()
            self.head_model_optimizer_collector.reload()
        
        # micro batch head bwd and recv
        no_head_bwd_mb_list = [False] * grad_accum_steps
        while True:
            if not all(no_head_bwd_mb_list) and not self.stop_event.is_set():
                try:
                    server_grad_payload = self.gradient_from_server_queue.get(timeout=0.001)
                    if server_grad_payload is not None:
                        mb_idx = server_grad_payload.mb_idx
                        no_head_bwd_mb_list[mb_idx] = True
                        self._head_bwd_micro(server_grad_payload)
                        # print(f'Global batch id -> {global_batch_idx} finished micro {mb_idx} head backward')
                except Empty:
                    continue
            else:
                # print(f'exit global batch id -> {global_batch_idx}')
                break
            pass
        if offload_activation:
            self.head_activation_collector.clear()
        # TODO head模型优化器状态卸载至CPU
        if offload_model_state:
            self.head_model_optimizer_collector.offload()
        
        self.max_mem_allocated_mb = max(self.max_mem_allocated_mb, torch.cuda.max_memory_allocated(self.client_device) / 1024**2)
        torch.cuda.reset_peak_memory_stats(self.client_device)
        return batch_loss

    def _do_sequential(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx):
        # micro batch head fwd and send
        batch_loss = 0
        for mb_idx in range(grad_accum_steps):
            payload = self._head_fwd_micro(group_id, mb_idx, grad_accum_steps, micro_inputs[mb_idx], micro_masks[mb_idx], micro_labels[mb_idx])
            self.labels_dict[mb_idx] = micro_labels[mb_idx]
            self.activation_to_server_queue.put(payload)
            # wait for server activation
            server_activation_payload = self.activation_from_server_queue.get(block=True)
            payload, loss = self._tail_fwd_bwd_micro(server_activation_payload)
            batch_loss += loss.item()
            self.gradient_to_server_queue.put(payload)
            # wait for server gradient
            server_grad_payload = self.gradient_from_server_queue.get(block=True)
            self._head_bwd_micro(server_grad_payload)
        return batch_loss

    def _save_profile_res(self, server_profile_data: List[GanttChartData]):
        batch_train_time_ms = 0
        local_compute_time_ms = 0
        server_compute_time_ms = 0
        head_fwd_time_avg = 0
        head_bwd_time_avg = 0
        server_fwd_time_avg = 0
        server_bwd_time_avg = 0
        tail_fwd_time_avg = 0
        tail_bwd_time_avg = 0
        assert len(self.profile_data) == len(server_profile_data), 'error in profile data length between client and server'
        for client_item, server_item in zip(self.profile_data, server_profile_data):
            client_item.server_fwd_timestamp = server_item.server_fwd_timestamp
            client_item.server_bwd_timestamp = server_item.server_bwd_timestamp
            client_item.server_bwd_send_timestamp = server_item.server_bwd_send_timestamp
            client_item.server_fwd_send_timestamp = server_item.server_fwd_send_timestamp
            client_item.train_time_duration_ms = round((client_item.head_bwd_timestamp[1] - client_item.head_fwd_timestamp[0]) * 1000, 2)
            local_compute_time_ms += (
                client_item.head_fwd_timestamp[1]
                - client_item.head_fwd_timestamp[0]
                + client_item.tail_fwd_timestamp[1]
                - client_item.tail_fwd_timestamp[0]
                + client_item.tail_bwd_timestamp[1]
                - client_item.tail_bwd_timestamp[0]
                + client_item.head_bwd_timestamp[1]
                - client_item.head_bwd_timestamp[0]
            ) * 1000
            server_compute_time_ms += (
                server_item.server_fwd_timestamp[1]
                - server_item.server_fwd_timestamp[0]
                + server_item.server_bwd_timestamp[1]
                - server_item.server_bwd_timestamp[0]
            ) * 1000
            # 计算 每个sub model的训练时间
            head_fwd_time_avg += (client_item.head_fwd_timestamp[1] - client_item.head_fwd_timestamp[0]) * 1000
            head_bwd_time_avg += (client_item.head_bwd_timestamp[1] - client_item.head_bwd_timestamp[0]) * 1000
            server_fwd_time_avg += (server_item.server_fwd_timestamp[1] - server_item.server_fwd_timestamp[0]) * 1000
            server_bwd_time_avg += (server_item.server_bwd_timestamp[1] - server_item.server_bwd_timestamp[0]) * 1000
            tail_fwd_time_avg += (client_item.tail_fwd_timestamp[1] - client_item.tail_fwd_timestamp[0]) * 1000
            tail_bwd_time_avg += (client_item.tail_bwd_timestamp[1] - client_item.tail_bwd_timestamp[0]) * 1000
        # 计算平均时间
        grad_accum_steps = len(self.profile_data)
        head_fwd_time_avg /= grad_accum_steps
        head_bwd_time_avg /= grad_accum_steps
        server_fwd_time_avg /= grad_accum_steps
        server_bwd_time_avg /= grad_accum_steps
        tail_fwd_time_avg /= grad_accum_steps
        tail_bwd_time_avg /= grad_accum_steps
        # 计算平均通信时间

        # 计算batch训练时间
        batch_train_time_ms = round((self.profile_data[-1].head_bwd_timestamp[1] - self.profile_data[0].head_fwd_timestamp[0]) * 1000, 2)
        # 计算本地计算时间

        # plot_gantt_per_batch(self.profile_data, fp=f"log/img/gantt_batch.png")
        save_dir = f'log/profile/{self.client_args.model}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        data_dict = {
            'max_mem_allocated_MB': self.max_mem_allocated_mb,
            'batch_train_time_ms': batch_train_time_ms,
            'head_fwd_time_avg_ms': round(head_fwd_time_avg, 2),
            'head_bwd_time_avg_ms': round(head_bwd_time_avg, 2),
            'server_fwd_time_avg_ms': round(server_fwd_time_avg, 2),
            'server_bwd_time_avg_ms': round(server_bwd_time_avg, 2),
            'tail_fwd_time_avg_ms': round(tail_fwd_time_avg, 2),
            'tail_bwd_time_avg_ms': round(tail_bwd_time_avg, 2),
            'client_compute_time_ms': round(local_compute_time_ms, 2),
            'server_compute_time_ms': round(server_compute_time_ms, 2),
            'client_idle_rate': round((1 - local_compute_time_ms / batch_train_time_ms) * 100, 2),
            'server_idle_rate': round((1 - server_compute_time_ms / batch_train_time_ms) * 100, 2),
            'mini_batch_data': [asdict(item) for item in self.profile_data],
        }
        save_gantt_chart_data(
            data_dict,
            os.path.join(
                save_dir,
                f'sp_{self.client_args.split_point}_b_{self.client_args.batch_size}_mb_{self.client_args.micro_batch_size}_s_'
                f'{self.client_args.max_seq_len}_off_{self.client_args.offload_activation}_mbps_{self.client_args.rate_mbps}_sort_{self.client_args.sort_batch}_offms_{self.client_args.offload_model_state}.json',
            ),
        )
        png_save_dir = f'log/img/{self.client_args.model}'
        if not os.path.exists(png_save_dir):
            os.makedirs(png_save_dir)
        plot_gantt_per_batch(
            self.profile_data,
            fp=os.path.join(
                png_save_dir,
                f'sp_{self.client_args.split_point}_b_{self.client_args.batch_size}_mb_{self.client_args.micro_batch_size}_s_'
                f'{self.client_args.max_seq_len}_off_{self.client_args.offload_activation}_mbps_{self.client_args.rate_mbps}_sort_{self.client_args.sort_batch}_offms_{self.client_args.offload_model_state}.png',
            ),
        )
        self.stop_event.set()

    def train_large_batch_overlapped_accum(self, batch: Dict, global_batch_idx: int):
        """
        对一个大 batch:
        1) 切分成多个 micro-batches
        2) head forward + 异步 send/recv（通信-计算重叠）
        3) 每个 micro 执行 tail forward + backward（只做反传，不 step）
        4) 全部 micro 完成后，统一 optimizer.step()，再 zero_grad()
        """
        self.head_model.train()
        self.tail_model.train()

        # 解析与 pad（先 pad 再切 micro，保持序列对齐）
        input_ids = batch["input_ids"].to(self.client_device)
        attention_mask = batch["attention_mask"].to(self.client_device)
        input_ids, attention_mask = pad_inputs(input_ids, attention_mask, self.client_args.max_seq_len)
        labels_full = input_ids  # 自回归

        total_bs = input_ids.size(0)
        micro_bs, grad_accum_steps = self._resolve_micro_bs(total_bs)
        self.logger.info(f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps}")
        print(f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps}")
        self.profile_data = [GanttChartData(mini_batch_idx=i) for i in range(grad_accum_steps)]
        # —— 每个“大 batch”一次 step —— 梯度先清零
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)

        if self.client_args.sort_batch:
            micro_batches = self._split_micro_with_mask(
                input_ids,
                attention_mask,
                micro_bs,
            )
            # 拆分成两个列表
            micro_inputs = [ids for ids, _ in micro_batches]
            micro_masks = [mask for _, mask in micro_batches]
            # 自回归任务时，labels = inputs
            micro_labels = micro_inputs
        else:
            # 切分微批
            micro_inputs = self._split_micro(input_ids, micro_bs)
            micro_masks = self._split_micro(attention_mask, micro_bs)
            micro_labels = self._split_micro(labels_full, micro_bs)

        # 全局的 group_id（server 用它来做整批 step）
        group_id = uuid.uuid4().hex
        self.stop_event.clear()
        # --------------------------------------------------------main loop--------------------------------------------------------
        if self.client_args.async_io:
            batch_loss = self._do_pipeline(grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx)
        else:
            batch_loss = self._do_sequential(grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx)
        # --------------------------------------------------------main loop end--------------------------------------------------------
        # do step
        self.optimizer_head.step()
        self.optimizer_tail.step()
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)
        self.logger.info(
            f'[Client] big batch {global_batch_idx}: loss={batch_loss/grad_accum_steps:.4f},max mem alloc: {torch.cuda.max_memory_allocated(device=self.client_device)/1024**2:.2f} MB',
        )
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        # print(self.profile_data)
        if global_batch_idx == 5:
            print(f'client finished training and need reduce profile data')
            self.activation_to_server_queue.put({'stop': True})
        # print(f'global batch id -> {global_batch_idx} finished training')
        pass

    def train_epoch(self, profile: bool = False):
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        start_time = time.time()
        send_future = self.main_executor.submit(self._head_client_send)
        recv_future = self.main_executor.submit(self._handle_server_send)
        for epoch in range(self.local_ep):
            self.logger.info(f"[Client] start (overlap+accum) epoch {epoch+1}, len: {len(self.train_loader)}")
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=1, warmup=2, active=2, repeat=0),  # 前 1 step 不采集  # 预热 1 step  # 采集 2 step
                on_trace_ready=(
                    torch.profiler.tensorboard_trace_handler("./log/trace", worker_name="client") if profile else None
                ),  # 保存到 TensorBoard
                # on_trace_ready=None,
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_flops=True,
            ) as prof:
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    self.train_large_batch_overlapped_accum(batch, batch_idx)
                    if profile:
                        prof.step()
                    if batch_idx == self.client_args.step:
                        break
        # self.stop_event.set()
        # wait for send/recv to finish
        send_future.result()
        recv_future.result()
        self.communicator.close()
        self.main_executor.shutdown(wait=True)
        end_time = time.time()
        self.logger.info(f"[Client Finished] epoch time: {end_time - start_time:.2f} s")
