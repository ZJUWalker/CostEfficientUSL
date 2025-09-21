from concurrent.futures import ThreadPoolExecutor, Future
from collections import deque
from dataclasses import dataclass, field
import logging
import time
import uuid
from typing import Any, Dict, Optional, Tuple, List, Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from usl.client.client import Client, ClientArgs
from usl.socket import SocketCommunicator
from usl.utils.dataset.exp import AverageMeter
from usl.utils.log_utils import timeit
from usl.utils.usl_gantt_plot import GanttChartData, plot_gantt_per_batch
from usl.utils.tensor_utils import pad_inputs


def timed_submit(executor: ThreadPoolExecutor, fn: Callable, *args, **kwargs):
    start = time.time()
    fut = executor.submit(fn, *args, **kwargs)
    fut.start = start

    def _done(f: Future):

        f.end = time.time()
        f.elapsed = round((f.end - start), 4)  # ms

    fut.add_done_callback(_done)
    return fut


@dataclass
class IOJob:
    batch_idx: int
    # 训练需要保留的上下文
    labels: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    pos_embeds: Optional[Tuple[torch.Tensor, ...]]
    head_output_activation: torch.Tensor  # 用于反传的 activation（GPU 上）
    # I/O 线程 future
    send_future: Future
    recv_future: Future
    # —— 元信息（用于对齐 server 端的上下文）——
    token: str
    group_id: str
    micro_idx: int
    micro_total: int


class AsyncClient(Client):
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
        super().__init__(
            client_args, head_model, tail_model, tokenizer, client_device, train_logger, dataset_train, dataset_test
        )
        self.io_pool = ThreadPoolExecutor(max_workers=num_workers)
        self.normalize_loss = normalize_loss
        self.mini_batch_time_gantt: List[GanttChartData] = []  # used for profiling

    @torch.no_grad()
    # @timeit()
    def _to_cpu_payload(
        self,
        head_outs: Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor]]],
        *,
        token: str,
        group_id: str,
        micro_idx: int,
        micro_total: int,
    ) -> Dict:
        act_cpu = head_outs[0].detach().cpu()
        attn_cpu = head_outs[1].detach().cpu() if head_outs[1] is not None else None
        pos_cpu = (
            tuple([t.float().cpu() for t in head_outs[2]])
            if (len(head_outs) > 2 and head_outs[2] is not None)
            else None
        )
        payload = {
            "activation": act_cpu,
            "is_training": True,
            # —— 元信息 ——（server 将用 token 作为上下文 key）
            "token": token,
            "group_id": group_id,
            "micro_idx": micro_idx,
            "micro_total": micro_total,
        }
        if attn_cpu is not None:
            payload["attention_mask"] = attn_cpu
        if pos_cpu is not None:
            payload["position_embeddings"] = list(pos_cpu)
        # print(f"[Client] payload size: {self._payload_nbytes(payload)/(1024*1024):.2f} MB")
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

    def _resolve_micro_bs(self, total_bs: int) -> Tuple[int, int]:
        # 兼容没有 micro_batch_size 的 ClientArgs
        micro_bs = getattr(self.client_args, "micro_batch_size", None) or self.client_args.batch_size
        if micro_bs <= 0:
            micro_bs = self.client_args.batch_size
        if micro_bs > total_bs:
            micro_bs = total_bs
        grad_accum_steps = (total_bs + micro_bs - 1) // micro_bs
        return micro_bs, grad_accum_steps

    def train_large_batch_overlapped_accum(self, batch: Dict, client_conn: SocketCommunicator, global_batch_idx: int):
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
        self.train_logger.info(
            f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps}"
        )
        print(f"[Client] big batch {global_batch_idx}: micro_bs={micro_bs}, accum_steps={grad_accum_steps}")
        self.mini_batch_time_gantt = [GanttChartData(mini_batch_idx=i) for i in range(grad_accum_steps)]
        # —— 每个“大 batch”一次 step —— 梯度先清零
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)

        # 切分微批
        micro_inputs = self._split_micro(input_ids, micro_bs)
        micro_masks = self._split_micro(attention_mask, micro_bs)
        micro_labels = self._split_micro(labels_full, micro_bs)

        # 全局的 group_id（server 用它来做整批 step）
        group_id = uuid.uuid4().hex

        # 队列维护未完成 micro 的 I/O 与反传上下文
        fwd_pending: deque[IOJob] = deque()  # 等待 server 回传的 forward 结果, 用于 tail fwd & bwd
        bwd_pending: deque[IOJob] = deque()  # 等待 server 回传的 backward grad 结果,用于head bwd
        start_time = time.time()
        # ---- 主循环：流水线调度 ----
        # 主循环：0..N-1 micro head FWD + 异步 I/O
        for mi in range(grad_accum_steps):

            # 当前 micro：先做 head forward + 异步 I/O（与前一个 micro 的等待重叠）
            job = self._head_fwd_micro(
                client_conn, group_id, mi, grad_accum_steps, micro_inputs[mi], micro_masks[mi], micro_labels[mi]
            )
            fwd_pending.append(job)
            # 等待所有的 0..N-1 micro tail BWD 完成
            # print(f'size of fwd_pending: {len(fwd_pending)}')
        while fwd_pending:
            head_bwd_job = self._tail_fwd_and_bwd_micro(fwd_pending.popleft(), client_conn, grad_accum_steps)
            bwd_pending.append(head_bwd_job)

            # 等待所有的 head bwd 完成
            # print(f'size of bwd_pending: {len(bwd_pending)}')
        while bwd_pending:
            head_bwd_job = bwd_pending.popleft()
            self._head_bwd_micro(head_bwd_job)
        # head bwd
        # print(self.mini_batch_time_gantt)
        plot_gantt_per_batch(self.mini_batch_time_gantt, fp=f"log/img/gantt_batch_{global_batch_idx}.png")
        # —— 到这里，所有 micro 的反传都已经把梯度累积在参数上 ——
        # 统一做一次参数更新（与 server 的 step 相呼应）
        self.optimizer_head.step()
        self.optimizer_tail.step()
        self.optimizer_head.zero_grad(set_to_none=True)
        self.optimizer_tail.zero_grad(set_to_none=True)

        # 记录
        self.train_logger.info(
            f"[Client][overlap+accum] big batch {global_batch_idx}, accum_steps={grad_accum_steps}, avg_loss={self.avg_loss.avg:.4f},batch_time={time.time()-start_time:.4f} s"
        )

    # @timeit(info='fwd')
    def _head_fwd_micro(
        self,
        client_conn: SocketCommunicator,
        group_id: str,
        micro_idx: int,
        grad_accum_steps: int,
        x: torch.Tensor,
        m: torch.Tensor,
        y: torch.Tensor,
    ):
        self.mini_batch_time_gantt[micro_idx].head_fwd_timestamp[0] = time.time()
        head_outs = self.head_model.forward(x, m)
        torch.cuda.synchronize()
        head_outs[0].requires_grad_(True)
        pos_embeds = head_outs[2] if (len(head_outs) > 2) else None

        token = uuid.uuid4().hex
        payload = self._to_cpu_payload(
            head_outs,
            token=token,
            group_id=group_id,
            micro_idx=micro_idx,
            micro_total=grad_accum_steps,
        )
        self.mini_batch_time_gantt[micro_idx].head_fwd_timestamp[1] = time.time()
        # send_fut = self.io_pool.submit(client_conn.send, payload)
        # recv_fut = self.io_pool.submit(client_conn.receive)
        send_fut = timed_submit(self.io_pool, client_conn.send, payload)
        recv_fut = timed_submit(self.io_pool, client_conn.receive)
        # print(f"[Client] micro {micro_idx}")
        head_fwd_job = IOJob(
            batch_idx=0,
            labels=y,
            attention_mask=head_outs[1],
            pos_embeds=pos_embeds,
            head_output_activation=head_outs[0],
            send_future=send_fut,
            recv_future=recv_fut,
            token=token,
            group_id=group_id,
            micro_idx=micro_idx,
            micro_total=grad_accum_steps,
        )

        # print(f"[Client] micro {micro_idx} fwd comm time: {time.time()-s:.4f} s")
        return head_fwd_job

    # @timeit(info='tail fwd & bwd')
    def _tail_fwd_and_bwd_micro(self, head_fwd: IOJob, client_conn: SocketCommunicator, grad_accum_steps: int):
        # 确保发送完成
        head_fwd.send_future.result()
        self.mini_batch_time_gantt[head_fwd.micro_idx].head_fwd_send_timestamp = [
            head_fwd.send_future.start,
            head_fwd.send_future.end,
        ]
        # print(f"[Client] micro {head_fwd.micro_idx} head sent,future elapsed:[{head_fwd.send_future.start}, {head_fwd.send_future.end}]")

        # 等待 server forward 返回（优先使用 server 回传的 token 等元信息，容错）
        server_forward_output: Dict = head_fwd.recv_future.result()
        self.mini_batch_time_gantt[head_fwd.micro_idx].tail_fwd_recv_timestamp = [
            head_fwd.recv_future.start,
            head_fwd.recv_future.end,
        ]
        # print(f"[Client] micro {head_fwd.micro_idx} tial recv,future elapsed:[{head_fwd.recv_future.start}, {head_fwd.recv_future.end}]")
        token = server_forward_output.get("token", head_fwd.token)
        group_id = server_forward_output.get("group_id", head_fwd.group_id)
        micro_idx = int(server_forward_output.get("micro_idx", head_fwd.micro_idx))
        micro_total = int(server_forward_output.get("micro_total", head_fwd.micro_total))

        activation_cpu = server_forward_output["server_activation"]
        activation_to_tail = torch.tensor(
            activation_cpu,
            device=self.client_device,
            dtype=head_fwd.head_output_activation.dtype,
            requires_grad=True,
        )
        self.mini_batch_time_gantt[head_fwd.micro_idx].tail_fwd_timestamp[0] = time.time()
        # tail forward
        output: CausalLMOutputWithPast = self.tail_model.forward(
            hidden_states=activation_to_tail,
            attention_mask=head_fwd.attention_mask,
            position_embeddings=head_fwd.pos_embeds,
            labels=head_fwd.labels,
        )
        torch.cuda.synchronize()
        self.mini_batch_time_gantt[head_fwd.micro_idx].tail_fwd_timestamp[1] = time.time()
        # 可选：按 accum 步数归一化，保证与“单大 batch 一次性训练”的梯度数值一致
        loss = output.loss / grad_accum_steps if self.normalize_loss else output.loss
        self.avg_loss.update(loss.item() * (grad_accum_steps if self.normalize_loss else 1.0))

        # backward（只反传，不做 step）——梯度累积到参数上
        self.mini_batch_time_gantt[head_fwd.micro_idx].tail_bwd_timestamp[0] = time.time()
        loss.backward()
        torch.cuda.synchronize()
        # 取出 tail 输入处梯度 → 发回 server 做 head 对应部分的反传
        grads_to_server = activation_to_tail.grad.detach().cpu()
        tail_grads_to_server = {
            "gradient": grads_to_server,
            # 元信息回传，server 用于定位上下文并统计 group 完成度
            "token": token,
            "group_id": group_id,
            "micro_idx": micro_idx,
            "micro_total": micro_total,
        }
        self.mini_batch_time_gantt[head_fwd.micro_idx].tail_bwd_timestamp[1] = time.time()
        send_bwd_fut = timed_submit(self.io_pool, client_conn.send, tail_grads_to_server)
        recv_bwd_fut = timed_submit(self.io_pool, client_conn.receive)
        # send_bwd_fut = self.io_pool.submit(client_conn.send, tail_grads_to_server)
        # recv_bwd_fut = self.io_pool.submit(client_conn.receive)
        return IOJob(
            batch_idx=micro_idx,
            labels=head_fwd.labels,
            attention_mask=head_fwd.attention_mask,
            pos_embeds=head_fwd.pos_embeds,
            head_output_activation=head_fwd.head_output_activation,
            send_future=send_bwd_fut,
            recv_future=recv_bwd_fut,
            token=token,
            group_id=group_id,
            micro_idx=micro_idx,
            micro_total=micro_total,
        )

    # @timeit(info='head bwd')
    def _head_bwd_micro(self, head_bwd_job: IOJob):
        head_bwd_job.send_future.result()
        self.mini_batch_time_gantt[head_bwd_job.micro_idx].tail_bwd_send_timestamp = [
            head_bwd_job.send_future.start,
            head_bwd_job.send_future.end,
        ]
        server_backward_output = head_bwd_job.recv_future.result()
        self.mini_batch_time_gantt[head_bwd_job.micro_idx].head_bwd_recv_timestamp = [
            head_bwd_job.recv_future.start,
            head_bwd_job.recv_future.end,
        ]
        # 同样优先使用 server 回传的元信息（便于日志/校验）
        # token_bwd = server_backward_output.get("token", head_bwd_job.token)
        self.mini_batch_time_gantt[head_bwd_job.micro_idx].head_bwd_timestamp[0] = time.time()
        grads_from_server = torch.tensor(
            server_backward_output["server_gradient"],
            device=self.client_device,
            dtype=head_bwd_job.head_output_activation.dtype,
        )

        # 把 server 的梯度接回到 head 输出上（这一步也会把梯度累积到 head 参数上）
        head_bwd_job.head_output_activation.backward(grads_from_server)
        torch.cuda.synchronize()
        self.mini_batch_time_gantt[head_bwd_job.micro_idx].head_bwd_timestamp[1] = time.time()
        # 不 step，不 zero_grad —— 由外层统一处理
        torch.cuda.empty_cache()

    def train_epoch(self):
        torch.cuda.reset_peak_memory_stats(device=self.client_device)
        self.avg_loss.reset()
        with SocketCommunicator(
            host="localhost",
            port=self.client_args.port,
            is_server=False,
            rate_limit_mbps=self.client_args.rate_mbps,
        ) as client_conn:
            start_time = time.time()
            for epoch in range(self.local_ep):
                self.train_logger.info(f"[Client] start (overlap+accum) epoch {epoch+1}, len: {len(self.train_loader)}")
                for batch_idx, batch in enumerate(self.train_loader, 1):
                    self.train_large_batch_overlapped_accum(batch, client_conn, batch_idx)
                    if batch_idx == self.client_args.step:
                        break
            end_time = time.time()
            self.train_logger.info(f"[Client Finished] epoch time: {end_time - start_time:.2f} s")
