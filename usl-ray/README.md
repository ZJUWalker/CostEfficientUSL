# USL-Ray

`usl-ray` is the independent Ray-based implementation of U-shaped split
learning used by this repository. The legacy `usl/` package remains unchanged;
the new package never imports it.

## Architecture

- Client/server boundary tensors use Ray's object store: the sender calls
  `ray.put`, passes the top-level `ObjectRef` to `peer.push.remote`, and the
  receiver's raylet pulls it automatically.
- Server GPUs use torch 2.9 `PipelineStage` with `ScheduleGPipe` or
  `Schedule1F1B`; stage-to-stage tensors use NCCL.
- Attention masks, position IDs, and RoPE tensors never cross the boundary.
  `SequenceSpec` carries only scalar metadata and every partition reconstructs
  these tensors from its local model config.
- Client schedules are bounded `for` loops plus futures. There are no polling
  threads, `while True`, or application-level network sleeps.
- Measured and simulated runs share `Span`, `StepTimeline`, `StepMetrics`,
  `LinkModel`, and the same layer partition function.

## Install

The versions match the development machine (Python 3.11, torch 2.9.1,
transformers 5.0.0, Ray 2.58.0):

```bash
cd /opt/tiger/CostEfficientUSL/usl-ray
pip install -e '.[dev]'
```

`pyproject.toml` is the source of truth. The repository-level
`requirements.txt` mirrors its runtime pins. The legacy package is not
guaranteed to work with these versions.

Network shaping additionally requires:

```bash
tc -V
sudo -n true
```

## Configuration

Every experiment directory contains:

- `common.yaml`: model, split, batch, global steps, pipeline, expected network
  conditions, and profile settings;
- `server.yaml`: inherits common settings and adds server world size/backend;
- `client.yaml`: inherits common settings and adds data/offload settings.

The client and server compare every shared field during attach. A mismatch
fails with a field-level diff before training starts.

Command-line values are typed overrides:

```bash
usl-ray server \
  --config experiments/configs/qwen3-1.7b/server.yaml \
  --set batch.batch_size=16
```

Unknown fields and invalid combinations fail during loading. Important
constraints include:

- `batch_size` must be divisible by `micro_batch_size`;
- a symmetric split must leave at least one server layer;
- a multi-stage server uses fixed sequence shapes and cannot sort variable
  sequence lengths;
- left padding requires `position_kind: from_valid_lens`;
- NF4 communication requires hidden size divisible by 64.

Resolved configurations are immutable and are saved beside each profile.

## Single-machine experiment

This functional topology uses one raylet. Ray transfers remain in shared
memory, so traffic control does not affect them:

```bash
TOPO=same-node experiments/run_single.sh \
  experiments/configs/smoke-qwen3-0.6b
```

The script follows the original experiment lifecycle:

1. start Ray;
2. start the server in the background;
3. start the client in the foreground;
4. wait for the server;
5. stop Ray on exit or failure.

Role-specific overrides use `--server-set` and `--client-set`:

```bash
experiments/run_single.sh experiments/configs/smoke-qwen3-0.6b \
  --set train.steps=2 \
  --server-set server.activation_offload_mb=2 \
  --client-set client.activation_offload_mb=2 \
  --client-set client.model_state_offload_layers=1
```

## Two-machine WAN experiment

Ray's object transfer must cross two raylets for `tc` to apply.

On the server machine:

```bash
ray start --head \
  --node-ip-address=<SERVER_IP> \
  --port=26379 \
  --num-gpus=4 \
  --resources='{"usl_server": 4}' \
  --disable-usage-stats

sudo experiments/tc/shape.sh apply <BOND_IFACE> <CLIENT_IP> 230 10
export RAY_ADDRESS=<SERVER_IP>:26379
experiments/run_server.sh experiments/configs/qwen3-1.7b/server.yaml
```

On the client machine:

```bash
ray start \
  --address=<SERVER_IP>:26379 \
  --node-ip-address=<CLIENT_IP> \
  --num-gpus=1 \
  --resources='{"usl_client": 1}' \
  --disable-usage-stats

sudo experiments/tc/shape.sh apply <BOND_IFACE> <SERVER_IP> 230 10
export RAY_ADDRESS=<SERVER_IP>:26379
experiments/run_client.sh experiments/configs/qwen3-1.7b/client.yaml
```

`shape.sh` uses HTB plus netem and filters by peer IP, so local NCCL and
unrelated traffic are not limited. Run `shape.sh clear <iface>` afterward.
When physical NICs are bond slaves, apply shaping to the bond device.

`experiments/bench_link.py` transfers fresh 1/4/16/64 MiB Ray objects in both
directions and fits \(T(n)=a+n/b\):

```bash
python experiments/bench_link.py --expected-mbps 230
```

Bandwidth uses tc's SI unit: 1 Mbit/s is 1,000,000 bit/s.

### Network namespace topology

`experiments/tc/netns.sh` and `TOPO=netns` create two local raylets connected
through a veth pair. This requires `CAP_SYS_ADMIN`. The development container
does not provide that capability (`mount --make-shared /run/netns` is denied),
so this topology could not be executed here. Use it on a host with the
capability, or use the supported two-machine topology above.

## Pipeline modes

| New name | Legacy name | Client order | Server |
|---|---|---|---|
| `sequential` | `naive` | one complete U path; one microbatch | GPipe |
| `gpipe` | `gpipe` | fill-drain on head and tail | GPipe |
| `splitmind` | `pipedream_wc` | all head forward, tail F/B, all head backward | GPipe |
| `splitmind_eager` | `pipedream_wc_eager` | consume activation/gradient futures as ready | GPipe |
| `1f1b` | `pipedream_strict` | bounded warmup then one-forward/one-backward | 1F1B |

Torch 2.9.1 has a bug in stock pipeline-neighbor initialization: it sends an
int64 scalar into a float32 receive buffer. `USLPipelineStage` overrides only
that initializer so gloo and NCCL use matching float32 metadata; scheduling
remains stock torch.

## Model support and extension

Qwen3, Llama, and Gemma2 use one `DecoderPartition`. Architecture differences
live in `models/arch.py`. To add a model:

1. add one `ArchSpec`;
2. add a small random config case to
   `tests/unit/test_partition_equivalence.py`;
3. verify connected and partitioned logits match.

Gemma2 keeps original layer indexes and uses transformers' actual sliding
attention masks. GPT-2 is intentionally unsupported.

## Profile and plots

When `profile.enabled` is true, the client writes:

```text
runs/<run.name>/<run_tag>/
├── config.yaml
└── profile.json
```

`profile.json` contains resolved config, versions, client/server memory,
operation and Ray-transfer spans, and metrics after excluding warmup steps.
CUDA operations are synchronized around measured spans to retain the legacy
measurement meaning.

Render the same timeline used by simulation:

```bash
usl-ray plot runs/.../profile.json
```

## Calibration, simulation, and search

Seven measured configurations fit a cost model:

```bash
MAX_SPLIT_POINT=14 \
experiments/calibrate.sh experiments/configs/qwen3-1.7b
```

Or call the fitter directly:

```bash
usl-ray calibrate <seven profile.json paths> \
  --base-split-point 2 \
  --base-batch-size 8 \
  --max-split-point 14 \
  --lora \
  --output runs/cost_model.json
```

Simulate and compare:

```bash
usl-ray simulate \
  --config experiments/configs/qwen3-1.7b/client.yaml \
  --cost-model runs/cost_model.json \
  --output runs/simulation.json

usl-ray compare \
  --profile runs/.../profile.json \
  --cost-model runs/cost_model.json
```

Pareto search:

```bash
usl-ray search \
  --config experiments/configs/qwen3-1.7b/client.yaml \
  --cost-model runs/cost_model.json \
  --max-batch-size 64 \
  --max-client-memory-mib 49152 \
  --output runs/pareto.json
```

`legacy_compat=True` in the Python simulator reproduces the old SplitMind
arithmetic, including its fixed 10 ms communication overhead and B39's
client-offload condition. Normal simulation fixes B39 and uses SI link units.

## Verification record

Executed on 8×NVIDIA A800-SXM4-80GB:

- 41 unit tests and 2 dedicated CUDA offload tests passed;
- 8 Ray/gloo integration tests passed;
- all five client schedules matched the connected model's one-step loss and
  every parameter update (`atol=1e-5`);
- Qwen3-0.6B random initialization ran three steps with two NCCL server GPUs
  and one client GPU;
- NCCL logged `P2P/CUMEM/read`;
- Qwen3-0.6B also ran two profiled steps with client/server activation
  offload and one client model layer swapped;
- profile JSON and Gantt plot were produced;
- activation and model-state offload passed dedicated gradient/update tests;
- every GPU returned to 0 MiB after the experiment;
- the deterministic simulator matches the legacy fixture at `1e-6`;
- single-machine netns shaping is blocked by the container capability, as
  documented above.

## Behavior differences from legacy

- pad labels are `-100`; `loss_on: completion` can restrict supervision;
- network bandwidth is controlled by tc rather than socket sleeps;
- server and client share one learning rate;
- every measured step after warmup contributes to profile averages;
- Gemma2 sliding attention is enabled correctly;
- simulation jitter defaults to zero;
- epoch estimates use the configured dataset size rather than 12,460;
- GPT-2 support was removed.

## Troubleshooting

- **Client cannot find server**: check `ray status`, `RAY_ADDRESS`, namespace
  `usl-ray`, and that both YAML files inherit the same common file.
- **Placement group waits forever**: the server node needs at least
  `server.world_size` units each of `GPU` and `usl_server`; the client node
  needs `GPU: 1` and `usl_client: 1`.
- **NCCL hangs**: confirm fixed microbatch shapes, inspect `NCCL_DEBUG=INFO`,
  and verify every stage is on one server node.
- **tc has no effect**: client/server likely share one raylet. Use two
  machines or the privileged netns topology.
- **Timeout**: exceptions include `(step, microbatch, kind)`; inspect the first
  missing key and the peer's preceding compute span.
