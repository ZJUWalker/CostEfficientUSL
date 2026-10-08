"""Small, explicit client data pipeline with an offline synthetic mode."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from usl_ray.config import ExperimentConfig


@dataclass(frozen=True)
class Batch:
    """One padded global batch."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class SyntheticDataset(Dataset[dict[str, torch.Tensor]]):
    """Deterministic token sequences for offline correctness and profiling."""

    def __init__(self, sample_count: int, sequence_length: int, vocab_size: int, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(1, vocab_size, (sample_count, sequence_length), generator=generator)
        self.attention_mask = torch.ones_like(self.input_ids)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[index], "attention_mask": self.attention_mask[index]}


def _collate(items: list[dict[str, torch.Tensor]]) -> Batch:
    input_ids = torch.stack([item["input_ids"] for item in items])
    attention_mask = torch.stack([item["attention_mask"] for item in items])
    labels = input_ids.clone().masked_fill(attention_mask == 0, -100)
    return Batch(input_ids, attention_mask, labels)


def _load_text_dataset(config: ExperimentConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    from datasets import DatasetDict, load_dataset, load_from_disk

    if config.client is None:
        raise ValueError("client section is required")
    data = config.client.data
    loaded = load_from_disk(data.path) if data.path else load_dataset(data.dataset)
    dataset = loaded["train"] if isinstance(loaded, DatasetDict) else loaded
    limit = min(len(dataset), int(len(dataset) * data.shrink_frac))
    dataset = dataset.select(range(limit))

    def tokenize(example: dict[str, object]) -> dict[str, list[int]]:
        if data.dataset.lower() == "dialogsum":
            text = f"### Dialogue: {example['dialogue']}\n### Summary: {example['summary']}"
        elif data.dataset.lower() == "gsm8k":
            text = f"### Question: {example['question']}\n### Answer: {example['answer']}"
        else:
            text = str(example.get("text", example.get("input", "")))
        encoded = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=config.batch.max_seq_len,
        )
        return {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"]}

    return dataset.map(tokenize, remove_columns=dataset.column_names).with_format("torch")


def create_dataloader(
    config: ExperimentConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> DataLoader:
    """Create complete global batches; incomplete batches never advance a step."""
    if config.client is None:
        raise ValueError("client section is required")
    data = config.client.data
    if data.dataset == "synthetic":
        dataset: Dataset = SyntheticDataset(
            data.sample_count,
            config.batch.max_seq_len,
            config.model.vocab_size,
            config.run.seed,
        )
    else:
        if tokenizer is None:
            raise ValueError("A tokenizer is required for text datasets")
        dataset = _load_text_dataset(config, tokenizer)
    return DataLoader(
        dataset,
        batch_size=config.batch.batch_size,
        shuffle=data.shuffle,
        drop_last=True,
        collate_fn=_collate,
    )


def exactly_steps(loader: DataLoader, steps: int) -> Iterator[Batch]:
    """Yield exactly the requested batches or fail before peers diverge."""
    if len(loader) < steps:
        raise ValueError(f"Dataset provides {len(loader)} complete batches, but train.steps={steps}")
    iterator = iter(loader)
    for _ in range(steps):
        batch = next(iterator)
        if not isinstance(batch, Batch):
            raise TypeError(f"Expected Batch, got {type(batch)}")
        yield batch
