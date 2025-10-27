from abc import abstractmethod
from contextlib import nullcontext
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
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from usl.offload import ModelParamOffload, OptimizerStateOffload, CpuOffloadHookWithOffloadHandler, AsyncDoubleBufferGroupOffloadHandler
from usl.server.single_server import PipelineMode
from usl.socket import SocketCommunicator, Payload
from usl.utils.usl_gantt_plot import GanttChartData, save_gantt_chart_data, plot_gantt_per_batch, plot_gantt_grouped
from usl.utils.tensor_utils import pad_inputs
from transformers import PreTrainedModel


@dataclass
class ClientArgs:
    host: str = "10.82.1.244"  # "10.82.1.244" "localhost"
    port: int = 8000
    model: str = "meta-llama/llama3.2-1b"
    batch_size: int = 4
    max_seq_len: int = 256
    step: int = 5
    dataset: str = "gsm8k"
    epoch: int = 1
    split_point: int = 2
    learning_rate: float = 5e-4
    use_lora: bool = False
    rate_mbps: float = 10  # rate in Mbps
    micro_batch_size: int = 1
    offload_activation: bool = False
    offload_model_state: bool = False
    offload_activation_mb_num: int = 1
    offload_model_state_sp_num: int = 0
    sort_batch: str = "no"  # no,asc,desc
    pipeline_mode: PipelineMode = PipelineMode.GPIPE
    save_dir: str = 'log/profile'
    max_client_mem_mb: int = 12288  # 12GB

    def build_filename(self, prefix: str = "", ext: str = "json") -> str:
        """
        构建文件名：
        - 普通参数总是显示（带 key_ 前缀）
        - Bool 类型为 True 只显示 key，不加值
        - Bool 类型为 False 不显示
        """
        parts = [
            f"sp_{self.split_point}",
            f"b_{self.batch_size}",
            f"mb_{self.micro_batch_size}",
            f"s_{self.max_seq_len}",
            f"mbps_{self.rate_mbps}",
            f"{self.pipeline_mode.value}",
        ]

        # 动态处理布尔字段

        if self.use_lora:
            parts.append("lora")
        if self.offload_activation:
            parts.append(f"coa_{self.offload_activation_mb_num}")
        if self.offload_model_state:
            parts.append(f"cos_{self.offload_model_state_sp_num}")
        # parts.append('{}')
        if self.sort_batch != "no":
            parts.append(f"sort_{self.sort_batch}")

        base = "_".join(parts)
        # name = f"{prefix}{base}{suffix}.{ext}"
        name = os.path.join(prefix, f"{base}{'{}'}.{ext}")  # 加个占位符，方便后续扩展
        return name


def _check_mem_usage(info: str = "", device: str = None) -> float:
    mem_allocated_mb = torch.cuda.memory_allocated(device) / 1024**2
    print(f"[{info}] Memory allocated on {device if device else 'current device'}: {mem_allocated_mb:.2f} MB")


class Client:
    def __init__(
        self,
        client_args: ClientArgs,
        head_model: PreTrainedModel,
        tail_model: PreTrainedModel,
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
        self.head_model_param_mem_alloc = torch.cuda.memory_allocated(self.client_device) / 1024**2
        if self.client_args.offload_model_state:
            self.tail_model = tail_model.to("cpu")  # 初始化状态：把tail_model放在cpu中
        else:
            self.tail_model = tail_model.to(self.client_device)  # 初始化状态：把tail_model放在cuda中
        if self.client_args.split_point == 0 and self.client_args.use_lora:
            # if use lora but no block, all the param don't need to be updated
            for n, p in self.head_model.named_parameters():
                p.requires_grad = False
            for n, p in self.tail_model.named_parameters():
                p.requires_grad = False
        # ---- move embedding layer to cuda
        # self.head_model.get_input_embeddings().to(self.client_device)
        # ---- Tokenizer
        self.tokenizer = tokenizer
        self.logger = train_logger
        self.train_loader = dataset_train
        self.test_loader = dataset_test
        self.local_ep = client_args.epoch  # ✅ 点操作符访问
        self.lr = client_args.learning_rate  # ✅ 点操作符访问
        self.optimizer_head = torch.optim.Adam(self.head_model.parameters(), lr=self.lr)
        self.optimizer_tail = torch.optim.Adam(self.tail_model.parameters(), lr=self.lr)

        # ---- Metrics
        self.curr_step_idx = 0
        self.compute_time = 0
        self.client_max_mem_alloc_mb = 0
        self.head_model_offload_timestamp = [0, 0]
        self.head_model_reload_timestamp = [0, 0]
        self.tail_model_offload_timestamp = [0, 0]
        self.tail_model_reload_timestamp = [0, 0]
        self.head_optimizer_offload_timestamp = [0, 0]
        self.head_optimizer_reload_timestamp = [0, 0]
        self.tail_optimizer_offload_timestamp = [0, 0]
        self.tail_optimizer_reload_timestamp = [0, 0]
        self.head_fwd_time = 0
        self.head_fwd_send_time = 0
        self.head_bwd_time = 0
        self.tail_fwd_time = 0
        self.tail_bwd_send_time = 0
        self.tail_bwd_time = 0
        self.sent_payload_bytes = 0
        self.normalize_loss = normalize_loss
        self.losses = []
        self.profile_data: List[GanttChartData] = []
        # ---- Executors & coordination
        self.main_executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=2)
        self.compute_future: Optional[Future] = None
        self.server_future: Optional[Future] = None
        self.stop_event = threading.Event()
        # ------- CUDA Stream ---------
        torch.cuda.set_stream(torch.cuda.Stream(self.client_device))  # set cuda compute stream
        self.load_stream = torch.cuda.Stream(self.client_device)  # set cuda load stream
        self.offload_stream = torch.cuda.Stream(self.client_device)  # set cuda offload stream

        # ----Parameter Efficient Offloading
        if self.client_args.offload_model_state:
            # do not offload embedding layer,because it will be used in both head and tail models (shared with lm_head)
            embed_layer = self.head_model.get_input_embeddings()
            except_tensor_idx_list = [id(p) for p in embed_layer.parameters()]
            self.head_m_mgr = ModelParamOffload(
                self.head_model,
                offload_layer_num=self.client_args.offload_model_state_sp_num,
                device=self.client_device,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
                except_tensor_idx_list=except_tensor_idx_list,
            )
            self.tail_m_mgr = ModelParamOffload(
                self.tail_model,
                offload_layer_num=self.client_args.offload_model_state_sp_num,
                device=self.client_device,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
                except_tensor_idx_list=except_tensor_idx_list,
            )
            self.head_os_mgr = OptimizerStateOffload(
                self.optimizer_head,
                offload_until_param_id=self.head_m_mgr.offload_until_param_id,
                device=self.client_device,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
            )
            self.tail_os_mgr = OptimizerStateOffload(
                self.optimizer_tail,
                offload_until_param_id=self.tail_m_mgr.offload_until_param_id,
                device=self.client_device,
                load_stream=self.load_stream,
                offload_stream=self.offload_stream,
            )
        self.num_minibatch = (self.client_args.batch_size + self.client_args.micro_batch_size - 1) // self.client_args.micro_batch_size
        self.offload_activation_mb_num = self.client_args.offload_activation_mb_num
        if self.offload_activation_mb_num > 0:
            self.activation_offload_handler = AsyncDoubleBufferGroupOffloadHandler(
                num_minibatch=self.offload_activation_mb_num, load_stream=self.load_stream, offload_stream=self.offload_stream
            )
            self.activation_offload_ctx = CpuOffloadHookWithOffloadHandler(
                self.activation_offload_handler
            )  # this is a context manager , not a handler
        self.pin_on_gpu_tensors_idx = []  # used to pin tensors on GPU that don't need to be offloaded

        # ---- Communicator
        self.communicator = SocketCommunicator(
            host=self.client_args.host,
            is_server=False,
            port=client_args.port,
            buffer_size=1024 * 4,  # 4KB
            rate_limit_mbps=client_args.rate_mbps,
        )

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

    @property
    def offload_model_state(self) -> bool:
        return self.client_args.offload_model_state

    @property
    def offload_activation(self) -> bool:
        return self.client_args.offload_activation

    def _check_mem_usage(self, info: str = "") -> None:
        return _check_mem_usage(info, self.client_device)

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
        phase: str = "FWD",  # 'FWD' or 'BWD'
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
            phase=phase,
            # —— 元信息 ——（server 将用 token 作为上下文 key）
            token=token,
            group_id=group_id,
            mb_idx=mb_idx,
            mb_total=mb_total,
            attention_mask=attn_cpu,
            position_embeddings=pos_cpu,
        )
        return payload

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
        reverse = self.client_args.sort_batch == "desc"
        sequences.sort(key=lambda x: x[0].size(0), reverse=reverse)

        chunks = []
        self.client_args.max_seq_len = 0
        for start in range(0, len(sequences), micro_bs):
            batch = sequences[start : start + micro_bs]
            max_len = max(seq[0].size(0) for seq in batch)
            self.client_args.max_seq_len = max(self.client_args.max_seq_len, max_len)
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
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].head_fwd_timestamp[0] = time.perf_counter()
        # if self.offload_activation:
        # if mb_idx < self.client_args.offload_activation_mb_num:
        #     with self.activation_offload_ctx:
        #         head_outs = self.head_model.forward(x, m)
        #         # self.activation_offload_handler.on_minibatch_commit_forward()
        # else:
        #     head_outs = self.head_model.forward(x, m)
        with self.activation_offload_ctx if mb_idx < self.client_args.offload_activation_mb_num else nullcontext():
            head_outs = self.head_model.forward(x, m)
            torch.cuda.current_stream().synchronize()
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
                phase="FWD" if mb_idx < grad_accum_steps - 1 else "BWD",
            )
            self.profile_data[mb_idx].head_fwd_timestamp[1] = time.perf_counter()
            if self.curr_step_idx > 0:
                self.head_fwd_time += self.profile_data[mb_idx].head_fwd_timestamp[1] - self.profile_data[mb_idx].head_fwd_timestamp[0]
            if mb_idx < self.client_args.offload_activation_mb_num:
                self.activation_offload_handler.on_minibatch_commit_forward()
        return payload

    def _tail_fwd_micro(self, server_forward_output: Payload) -> Tuple[torch.Tensor, torch.Tensor]:
        # 解析 server 传来的 payload
        mb_idx = server_forward_output.mb_idx
        mb_total = server_forward_output.mb_total
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].tail_fwd_timestamp[0] = time.perf_counter()
        activation_cpu: torch.Tensor = server_forward_output.tensor
        activation_to_tail = activation_cpu.to(self.client_device, non_blocking=True).requires_grad_(True)
        # tail forward
        output: CausalLMOutputWithPast = self.tail_model.forward(
            hidden_states=activation_to_tail,
            attention_mask=self.atten_mask_dict[mb_idx],
            position_embeddings=self.pos_embedding_dict[mb_idx],
            labels=self.labels_dict[mb_idx],
        )
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].tail_fwd_timestamp[1] = time.perf_counter()
        if self.curr_step_idx > 0:
            self.tail_fwd_time += self.profile_data[mb_idx].tail_fwd_timestamp[1] - self.profile_data[mb_idx].tail_fwd_timestamp[0]
        # 可选：按 accum 步数归一化，保证与“单大 batch 一次性训练”的梯度数值一致
        loss = output.loss / mb_total if self.normalize_loss else output.loss
        self.losses.append(loss.item())
        return activation_to_tail, loss

    def _tail_bwd_micro(
        self,
        loss: torch.Tensor,
        activation_to_tail: torch.Tensor,
        token: str,
        group_id: str,
        mb_idx: int,
        mb_total: int,
    ) -> Tuple[Payload, torch.Tensor]:
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].tail_bwd_timestamp[0] = time.perf_counter()
        loss.backward()
        torch.cuda.current_stream().synchronize()
        grad_payload = self._to_cpu_payload(
            output=(activation_to_tail.grad,),
            token=token,
            group_id=group_id,
            mb_idx=mb_idx,
            mb_total=mb_total,
            is_activation=False,
            phase="BWD" if mb_idx < mb_total - 1 else "FWD",
        )
        self.profile_data[mb_idx].tail_bwd_timestamp[1] = time.perf_counter()
        if self.curr_step_idx > 0:
            self.tail_bwd_time += self.profile_data[mb_idx].tail_bwd_timestamp[1] - self.profile_data[mb_idx].tail_bwd_timestamp[0]
        return grad_payload

    def _head_bwd_micro(self, server_bwd_output: Payload):
        assert server_bwd_output.is_activation == False, "should be gradient,but activation recieved"
        mb_idx = server_bwd_output.mb_idx
        grad_cpu: torch.Tensor = server_bwd_output.tensor
        # load grad and activation
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].head_bwd_timestamp[0] = time.perf_counter()
        grad_to_head = grad_cpu.to(self.client_device, non_blocking=True)
        head_activation = self.head_fwd_activation_dict[mb_idx]
        # real head model backward
        if mb_idx < self.client_args.offload_activation_mb_num:
            self.activation_offload_handler.on_minibatch_commit_backward()
        head_activation.backward(grad_to_head)
        torch.cuda.current_stream().synchronize()
        self.profile_data[mb_idx].head_bwd_timestamp[1] = time.perf_counter()
        if self.curr_step_idx > 0:
            self.head_bwd_time += self.profile_data[mb_idx].head_bwd_timestamp[1] - self.profile_data[mb_idx].head_bwd_timestamp[0]
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
    @torch.no_grad()
    def _head_client_send(self):
        self._check_communication()
        while not self.stop_event.is_set():
            try:
                payload: Optional[Payload | Dict] = self.activation_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    start_send = time.perf_counter()
                    self.communicator.send(payload)
                    end_time = time.perf_counter()
                    if isinstance(payload, dict) and "stop" in payload:
                        print("send stop flag")
                        continue
                    else:
                        mb_idx = payload.mb_idx
                        self.profile_data[mb_idx].head_fwd_send_timestamp[0] = start_send
                        self.profile_data[mb_idx].head_fwd_send_timestamp[1] = end_time
                        if self.curr_step_idx > 0:
                            self.head_fwd_send_time += end_time - start_send
                else:
                    continue
            except Empty:
                pass
            try:
                if self.client_args.pipeline_mode == PipelineMode.PIPE_DREAM_WC:
                    if not self.activation_to_server_queue.empty():
                        time.sleep(0.001)  # 发送给服务器的等待队列中有数据，避免频繁发送
                        continue
                payload = self.gradient_to_server_queue.get(timeout=0.001)
                if payload is not None:  # 可能是 None（队列空）
                    # print(f'send gradient payload')
                    self.sent_payload_bytes += payload.payload_nbytes()
                    start_send = time.perf_counter()
                    self.communicator.send(payload)
                    end_time = time.perf_counter()
                    mb_idx = payload.mb_idx
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[0] = start_send
                    self.profile_data[mb_idx].tail_bwd_send_timestamp[1] = end_time
                    if self.curr_step_idx > 0:
                        self.tail_bwd_send_time += end_time - start_send
                else:
                    continue
            except Empty:
                pass
        print("client send thread exit")
        pass

    # TODO: 处理recv操作
    @torch.no_grad()
    def _handle_server_send(self):
        self._check_communication()
        self.communicator.conn.settimeout(60.0)
        while not self.stop_event.is_set():
            try:
                data: Optional[Dict | Payload] = self.communicator.receive()
            except Exception as e:
                break
            if data is None:
                break
            if isinstance(data, dict) and "profile" in data:
                print(f"get profile data")
                try:
                    if self.client_max_mem_alloc_mb is not None and self.client_max_mem_alloc_mb > self.client_args.max_client_mem_mb:
                        self.communicator.close()
                        self.main_executor.shutdown(wait=False)
                        sys.exit(1)
                    self._save_profile_res(data)
                except Exception as e:
                    print(f"error when save profile data: {e}")
                finally:
                    self.stop_event.set()
                    break
            data.tensor = data.tensor.pin_memory()
            if data.is_activation:
                self.activation_from_server_queue.put(data)
            else:
                self.gradient_from_server_queue.put(data)
        print("server send thread exit")

    @torch.no_grad()
    def _save_profile_res(self, server_profile_res: Dict[str, Any]):
        batch_train_time_ms = 0
        local_compute_time_ms = 0
        server_compute_time_ms = 0
        server_profile_gantt_data: List[GanttChartData] = server_profile_res.get('profile', [])
        server_fwd_time = server_profile_res.get('server_fwd_time', 0)
        server_fwd_send_time = server_profile_res.get('server_fwd_send_time', 0)
        server_bwd_time = server_profile_res.get('server_bwd_time', 0)
        server_bwd_send_time = server_profile_res.get('server_bwd_send_time', 0)
        server_offload_time_durations = server_profile_res.get('server_offload_time_durations', [])
        server_reload_time_durations = server_profile_res.get('server_reload_time_durations', [])
        server_policy_str = server_profile_res.get('file_suffix', '')
        if server_policy_str:
            server_policy_str = f"_{server_policy_str}"
        client_send_time_ms = 0
        server_send_time_ms = 0
        delay_time_ms_in_send_and_compute = 0
        assert len(self.profile_data) == len(server_profile_gantt_data), "error in profile data length between client and server"
        for client_item, server_item in zip(self.profile_data, server_profile_gantt_data):
            client_item.server_fwd_timestamp = server_item.server_fwd_timestamp
            client_item.server_bwd_timestamp = server_item.server_bwd_timestamp
            client_item.server_fwd_send_timestamp = server_item.server_fwd_send_timestamp
            client_item.server_bwd_send_timestamp = server_item.server_bwd_send_timestamp
            client_item.train_time_duration_ms = round((client_item.head_bwd_timestamp[1] - client_item.head_fwd_timestamp[0]) * 1000, 2)
            # 计算通信时间
            client_send_time_ms += (
                client_item.head_fwd_send_timestamp[1]
                - client_item.head_fwd_send_timestamp[0]
                + client_item.tail_bwd_send_timestamp[1]
                - client_item.tail_bwd_send_timestamp[0]
            ) * 1000
            server_send_time_ms += (
                client_item.server_fwd_send_timestamp[1]
                - client_item.server_fwd_send_timestamp[0]
                + client_item.server_bwd_send_timestamp[1]
                - client_item.server_bwd_send_timestamp[0]
            ) * 1000
            # 计算训练时间
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
            client_item.head_m_offload_ts = self.head_model_offload_timestamp
            client_item.head_m_reload_ts = self.head_model_reload_timestamp
            client_item.tail_m_offload_ts = self.tail_model_offload_timestamp
            client_item.tail_m_reload_ts = self.tail_model_reload_timestamp
            head_m_offload_time_ms = self.head_model_offload_timestamp[1] - self.head_model_offload_timestamp[0]
            head_m_reload_time_ms = self.head_model_reload_timestamp[1] - self.head_model_reload_timestamp[0]
            tail_m_offload_time_ms = self.tail_model_offload_timestamp[1] - self.tail_model_offload_timestamp[0]
            tail_m_reload_time_ms = self.tail_model_reload_timestamp[1] - self.tail_model_reload_timestamp[0]
            client_item.head_optimizer_offload_ts = [var + head_m_offload_time_ms for var in self.head_optimizer_offload_timestamp]
            client_item.head_optimizer_reload_ts = [var + head_m_reload_time_ms for var in self.head_optimizer_reload_timestamp]
            client_item.tail_optimizer_offload_ts = [var + tail_m_offload_time_ms for var in self.tail_optimizer_offload_timestamp]
            client_item.tail_optimizer_reload_ts = [var + tail_m_reload_time_ms for var in self.tail_optimizer_reload_timestamp]

        # 计算通信和计算之间的延迟时间
        for i in range(len(self.profile_data)):
            delay_time_ms_in_send_and_compute += (client_item.server_fwd_timestamp[0] - client_item.head_fwd_send_timestamp[1]) * 1000
            delay_time_ms_in_send_and_compute += (client_item.tail_fwd_timestamp[0] - client_item.server_fwd_send_timestamp[1]) * 1000
            delay_time_ms_in_send_and_compute += (client_item.server_bwd_timestamp[0] - client_item.tail_bwd_send_timestamp[1]) * 1000
            delay_time_ms_in_send_and_compute += (client_item.head_bwd_timestamp[0] - client_item.server_bwd_send_timestamp[1]) * 1000
        # 计算平均时间
        grad_accum_steps = len(self.profile_data)
        head_fwd_time_avg = self.head_fwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        head_fwd_send_time_avg = self.head_fwd_send_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        head_bwd_time_avg = self.head_bwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        server_fwd_time_avg = server_fwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        server_fwd_send_time_avg = server_fwd_send_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        server_bwd_time_avg = server_bwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        server_bwd_send_time_avg = server_bwd_send_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        tail_fwd_time_avg = self.tail_fwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        tail_bwd_time_avg = self.tail_bwd_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        tail_bwd_send_time_avg = self.tail_bwd_send_time * 1000 / grad_accum_steps / (self.client_args.step - 1)
        delay_time_avg_ms = delay_time_ms_in_send_and_compute / grad_accum_steps / 4
        # 计算平均通信时间

        # 计算batch训练时间
        batch_train_time_ms = round((self.profile_data[-1].head_bwd_timestamp[1] - self.profile_data[0].head_fwd_timestamp[0]) * 1000, 2)
        # 计算本地计算时间

        # plot_gantt_per_batch(self.profile_data, fp=f"log/img/gantt_batch.png")
        layer_num = self.client_args.split_point if self.client_args.split_point > 0 else 1
        data_dict = {
            "mbps": self.client_args.rate_mbps,
            "split_point": self.client_args.split_point,
            "batch_size": self.client_args.batch_size,
            "micro_batch_size": self.client_args.micro_batch_size,
            "max_seq_len": self.client_args.max_seq_len,
            "offload_model_state": self.client_args.offload_model_state,
            "offload_activation": self.client_args.offload_activation,
            "head_model_size": self.head_model_param_mem_alloc,
            "offload_model_state_sp_num": self.client_args.offload_model_state_sp_num,
            "offload_activation_mb_num": self.client_args.offload_activation_mb_num,
            "client_max_mem_alloc_mb": round(self.client_max_mem_alloc_mb, 4),
            "server_max_mem_alloc_mb": server_profile_res.get("max_mem_alloc", 0),
            "batch_train_time_ms": batch_train_time_ms,
            "GPU_rent_cost": round(batch_train_time_ms * server_profile_res.get("max_mem_alloc", 0) / 1e6, 6),
            "head_fwd_time_avg_ms": round(head_fwd_time_avg, 2),
            "head_fwd_send_time_avg_ms": round(head_fwd_send_time_avg, 2),
            "head_bwd_time_avg_ms": round(head_bwd_time_avg, 2),
            "server_fwd_time_avg_ms": round(server_fwd_time_avg, 2),
            "server_fwd_send_time_avg_ms": round(server_fwd_send_time_avg, 2),
            "server_bwd_time_avg_ms": round(server_bwd_time_avg, 2),
            "server_bwd_send_time_avg_ms": round(server_bwd_send_time_avg, 2),
            "tail_fwd_time_avg_ms": round(tail_fwd_time_avg, 2),
            "tail_bwd_send_time_avg_ms": round(tail_bwd_send_time_avg, 2),
            "tail_bwd_time_avg_ms": round(tail_bwd_time_avg, 2),
            "client_compute_time_ms": round(local_compute_time_ms, 2),
            "server_compute_time_ms": round(server_compute_time_ms, 2),
            "delay_time_avg_ms": round(delay_time_avg_ms, 2),
            "head_m_offload_time_ms": round((self.head_model_offload_timestamp[1] - self.head_model_offload_timestamp[0]) * 1000, 2),
            "head_m_reload_time_ms": round((self.head_model_reload_timestamp[1] - self.head_model_reload_timestamp[0]) * 1000, 2),
            "tail_m_offload_time_ms": round((self.tail_model_offload_timestamp[1] - self.tail_model_offload_timestamp[0]) * 1000, 2),
            "tail_m_reload_time_ms": round((self.tail_model_reload_timestamp[1] - self.tail_model_reload_timestamp[0]) * 1000, 2),
            "head_os_offload_time_ms": round((self.head_optimizer_offload_timestamp[1] - self.head_optimizer_offload_timestamp[0]) * 1000, 2),
            "head_os_reload_time_ms": round((self.head_optimizer_reload_timestamp[1] - self.head_optimizer_reload_timestamp[0]) * 1000, 2),
            "tail_os_offload_time_ms": round((self.tail_optimizer_offload_timestamp[1] - self.tail_optimizer_offload_timestamp[0]) * 1000, 2),
            "tail_os_reload_time_ms": round((self.tail_optimizer_reload_timestamp[1] - self.tail_optimizer_reload_timestamp[0]) * 1000, 2),
            "activation_offload_time_ms": self.activation_offload_handler.offload_time_durations if self.offload_activation else 0,
            "activation_reload_time_ms": self.activation_offload_handler.reload_time_durations if self.offload_activation else 0,
            "server_activation_offload_time_ms": server_offload_time_durations,
            "server_activation_reload_time_ms": server_reload_time_durations,
            "client_send_rate": round(client_send_time_ms / batch_train_time_ms * 100, 2),
            "server_send_rate": round(server_send_time_ms / batch_train_time_ms * 100, 2),
            "client_idle_rate": round((1 - local_compute_time_ms / batch_train_time_ms) * 100, 2),
            "server_idle_rate": round((1 - server_compute_time_ms / batch_train_time_ms) * 100, 2),
            "total_bytes_sent": self.sent_payload_bytes,
            "bytes_sent_per_ms": round(self.sent_payload_bytes / client_send_time_ms, 0),
            "mini_batch_data": [asdict(item) for item in self.profile_data],
        }
        print(data_dict["client_max_mem_alloc_mb"], data_dict["server_max_mem_alloc_mb"], data_dict["batch_train_time_ms"])
        # dt_save_dir = f"{self.client_args.save_dir}/{self.client_args.model}"
        dt_save_dir = os.path.join(self.client_args.save_dir, self.client_args.model)
        if not os.path.exists(dt_save_dir):
            os.makedirs(dt_save_dir)
        save_gantt_chart_data(data_dict, fp=self.client_args.build_filename(prefix=dt_save_dir, ext="json").format(server_policy_str))
        # png_save_dir = f"log/img/{self.client_args.model}"
        # if not os.path.exists(png_save_dir):
        #     os.makedirs(png_save_dir)
        # plot_gantt_per_batch(self.profile_data, fp=self.client_args.build_filename(prefix=png_save_dir, ext="png"))
        png_save_dir_c = f"log/img/grouped/{self.client_args.model}"
        if not os.path.exists(png_save_dir_c):
            os.makedirs(png_save_dir_c)
        plot_gantt_grouped(self.profile_data, fp=self.client_args.build_filename(prefix=png_save_dir_c, ext="png").format(server_policy_str))
        self.stop_event.set()

    @abstractmethod
    def _train_minibatches(self, grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx) -> float:
        raise NotImplementedError('Subclass must implement abstract method')

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
        # if not self.client_args.sort_batch:
        input_ids, attention_mask = pad_inputs(input_ids, attention_mask, self.client_args.max_seq_len)
        labels_full = input_ids  # 自回归

        total_bs = input_ids.size(0)
        micro_bs, grad_accum_steps = self._resolve_micro_bs(total_bs)
        self.logger.info(f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps}")
        self.profile_data = [GanttChartData(mini_batch_idx=i) for i in range(grad_accum_steps)]
        # —— 每个“大 batch”一次 step —— 梯度先清零
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)

        if self.client_args.sort_batch != "no":
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
        batch_loss = self._train_minibatches(grad_accum_steps, micro_inputs, micro_masks, micro_labels, group_id, global_batch_idx)
        # --------------------------------------------------------main loop end--------------------------------------------------------
        self.logger.info(
            f"[Client] big batch {global_batch_idx}: loss={batch_loss/grad_accum_steps:.4f},max mem alloc: {torch.cuda.max_memory_allocated(device=self.client_device)/1024**2:.2f} MB",
        )
        print(f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps},loss = {batch_loss/grad_accum_steps:.4f}")
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        # print(self.profile_data)
        if global_batch_idx == self.client_args.step:
            print(f"client finished training and need reduce profile data")
            self.activation_to_server_queue.put({"stop": True})
        # print(f'global batch id -> {global_batch_idx} finished training')
        pass

    def train_epoch(self, profile: bool = False):
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        start_time = time.perf_counter()
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
                    self.curr_step_idx += 1
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
        end_time = time.perf_counter()
        self.logger.info(f"[Client Finished] epoch time: {end_time - start_time:.2f} s")
