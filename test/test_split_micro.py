# import pytest
# import torch

# from usl.client.client import Client  # 假设 client.py 里定义了 Client 类，并包含 _split_micro_with_mask

# class DummyTokenizer:
#     def __init__(self, pad_token_id=0):
#         self.pad_token_id = pad_token_id


# @pytest.fixture
# def my_client():
#     # 你可能需要根据实际 Client.__init__() 的参数改造这里
#     client = Client.__new__(Client)  
#     client.tokenizer = DummyTokenizer(pad_token_id=0)
#     return client


# def test_skip_empty_sequences(my_client):
#     input_ids = torch.tensor([[0, 0, 0], [101, 200, 0]])
#     attention_mask = torch.tensor([[0, 0, 0], [1, 1, 0]])
#     chunks = my_client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
#     assert len(chunks) == 1
#     ids, mask = chunks[0]
#     assert ids.shape == (1, 2)
#     assert torch.equal(ids[0], torch.tensor([101, 200]))


# def test_all_empty_sequences(my_client):
#     input_ids = torch.zeros((3, 4), dtype=torch.long)
#     attention_mask = torch.zeros((3, 4), dtype=torch.long)
#     chunks = my_client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
#     assert chunks == []


# def test_batch_padding_and_order(my_client):
#     input_ids = torch.tensor([
#         [101, 200, 201, 0, 0],   # len=3
#         [101, 202,   0, 0, 0],   # len=2
#         [101, 203, 204, 205, 0], # len=4
#     ])
#     attention_mask = torch.tensor([
#         [1, 1, 1, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 1, 0],
#     ])

#     chunks = my_client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
#     assert len(chunks) == 2

#     ids1, mask1 = chunks[0]
#     assert ids1.shape == (2, 4)
#     assert mask1.shape == (2, 4)
#     assert mask1[0].sum() == 4
#     assert mask1[1].sum() == 3

#     ids2, mask2 = chunks[1]
#     assert ids2.shape == (1, 2)
#     assert mask2.shape == (1, 2)
#     assert mask2[0].sum() == 2


# def test_micro_bs_greater_than_n(my_client):
#     input_ids = torch.tensor([
#         [101, 102, 0],
#         [103, 104, 105],
#     ])
#     attention_mask = torch.tensor([
#         [1, 1, 0],
#         [1, 1, 1],
#     ])
#     chunks = my_client._split_micro_with_mask(input_ids, attention_mask, micro_bs=5)
#     assert len(chunks) == 1
#     ids, mask = chunks[0]
#     assert ids.shape[0] == 2  # batch size = 2


# def test_micro_bs_equal_one(my_client):
#     input_ids = torch.tensor([
#         [101, 102, 0],
#         [103, 104, 105],
#     ])
#     attention_mask = torch.tensor([
#         [1, 1, 0],
#         [1, 1, 1],
#     ])
#     chunks = my_client._split_micro_with_mask(input_ids, attention_mask, micro_bs=1)
#     assert len(chunks) == 2
#     for ids, mask in chunks:
#         assert ids.shape[0] == 1
#         assert (mask.sum().item() == (ids != 0).sum().item())

import torch
from usl.client.client import Client  # 假设 client.py 里定义了 Client 类，并包含 _split_micro_with_mask


class DummyTokenizer:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id


def make_client():
    client = Client.__new__(Client)  # 跳过 __init__
    client.tokenizer = DummyTokenizer(pad_token_id=0)
    return client


def test_skip_empty_sequences(client):
    print("\n[TEST] skip_empty_sequences")
    print("期望: 只保留第2行 `[101,200]`，shape=(1,2)")
    input_ids = torch.tensor([[0, 0, 0], [101, 200, 0]])
    attention_mask = torch.tensor([[0, 0, 0], [1, 1, 0]])
    chunks = client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
    for ids, mask in chunks:
        print("实际 ids:", ids)
        print("实际 mask:", mask)


def test_all_empty_sequences(client):
    print("\n[TEST] all_empty_sequences")
    print("期望: 输出 []（因为全是 padding）")
    input_ids = torch.zeros((3, 4), dtype=torch.long)
    attention_mask = torch.zeros((3, 4), dtype=torch.long)
    chunks = client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
    print("实际 chunks:", chunks)


def test_batch_padding_and_order(client):
    print("\n[TEST] batch_padding_and_order")
    print("期望: 2 个 chunk")
    print(" - 第1个 chunk (batch=2, seq_len=4)，mask 行和分别为4, 3")
    print(" - 第2个 chunk (batch=1, seq_len=2)，mask 行和为2")
    input_ids = torch.tensor([
        [101, 200, 201, 0, 0],   # len=3
        [101, 202,   0, 0, 0],   # len=2
        [101, 203, 204, 205, 0], # len=4
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0],
    ])

    chunks = client._split_micro_with_mask(input_ids, attention_mask, micro_bs=2)
    for i, (ids, mask) in enumerate(chunks):
        print(f"实际 chunk {i}:")
        print("ids:", ids)
        print("mask:", mask)
        print("mask.sum(dim=1):", mask.sum(dim=1))


def test_micro_bs_greater_than_n(client):
    print("\n[TEST] micro_bs_greater_than_n")
    print("期望: 1 个 chunk，batch=2")
    input_ids = torch.tensor([
        [101, 102, 0],
        [103, 104, 105],
    ])
    attention_mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ])
    chunks = client._split_micro_with_mask(input_ids, attention_mask, micro_bs=5)
    for ids, mask in chunks:
        print("实际 ids:", ids)
        print("实际 mask:", mask)


def test_micro_bs_equal_one(client):
    print("\n[TEST] micro_bs_equal_one")
    print("期望: 拆成 2 个 chunk，每个 batch=1；mask.sum == 非零 token 数")
    input_ids = torch.tensor([
        [101, 102, 0],
        [103, 104, 105],
    ])
    attention_mask = torch.tensor([
        [1, 1, 0],
        [1, 1, 1],
    ])
    chunks = client._split_micro_with_mask(input_ids, attention_mask, micro_bs=1)
    for i, (ids, mask) in enumerate(chunks):
        print(f"实际 chunk {i}:")
        print("ids:", ids)
        print("mask:", mask)
        print("mask.sum:", mask.sum().item(), " | 非零 token 数:", (ids != 0).sum().item())


if __name__ == "__main__":
    client = make_client()
    test_skip_empty_sequences(client)
    test_all_empty_sequences(client)
    test_batch_padding_and_order(client)
    test_micro_bs_greater_than_n(client)
    test_micro_bs_equal_one(client)
