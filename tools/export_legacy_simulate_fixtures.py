"""Export deterministic outputs from the legacy simulator.

This one-time tool intentionally imports ``usl``. Runtime code under ``usl-ray``
must never depend on the legacy package.
"""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch


def _export() -> None:
    """Write the fixture after the legacy package is importable."""
    from usl.simulate.base import MainVariable, MemoryConstant, TimeConstant
    from usl.simulate.simulate import simulate

    main_variable = MainVariable(
        total_sample_count=10_000,
        batch_size=8,
        split_point=3,
        client_offload_mb_num=2,
        server_offload_mb_num=2,
        client_offload_model_state_sp_num=1,
        lora=True,
    )
    memory = MemoryConstant(
        max_split_point=14,
        micro_batch_size=1,
        max_seq_len=128,
        baseline_split_point=2,
        baseline_minibatch_num=8,
        base_client_mem_alloc=3000.0,
        base_server_mem_alloc=12_000.0,
        mem_increment_per_sp_client=200.0,
        mem_increment_per_sp_server=350.0,
        mem_increment_per_sp_mb_client=12.0,
        mem_increment_per_sp_mb_server=10.0,
        base_model_state_mem_alloc_except_blocks=100.0,
        model_mem_increment_per_sp_client=120.0,
    )
    timing = TimeConstant(
        rate_mbps=230.0,
        base_head_fwd_time_per_mb=10.0,
        base_head_bwd_time_per_mb=20.0,
        base_server_fwd_time_per_mb=60.0,
        base_server_bwd_time_per_mb=100.0,
        base_tail_fwd_time_per_mb=12.0,
        base_tail_bwd_time_per_mb=22.0,
        head_activation_send_time=150.0,
        server_activation_send_time=140.0,
        tail_gradient_send_time=130.0,
        server_gradient_send_time=120.0,
    )
    with patch.object(random, "randint", return_value=0):
        result = simulate(main_variable, timing, memory)
    destination = Path(__file__).parents[1] / "usl-ray" / "tests" / "fixtures" / "legacy_simulate" / "splitmind.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "main_variable": asdict(main_variable),
                "memory": asdict(memory),
                "timing": asdict(timing),
                "objective": asdict(result.objective),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    """Run the legacy simulator without modifying its source tree.

    The legacy simulator contains a Python 3.12-only nested f-string. The new
    project targets the machine's Python 3.11, so the exporter patches a
    temporary package copy solely to capture a compatibility fixture.
    """
    if "--worker" in sys.argv:
        _export()
        return
    repository = Path(__file__).parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        package_copy = Path(temporary) / "usl"
        shutil.copytree(repository / "usl", package_copy)
        simulator = package_copy / "simulate" / "simulate.py"
        source = simulator.read_text(encoding="utf-8")
        lines = source.splitlines()
        for index, line in enumerate(lines):
            if "f'log/simulate_results_" in line:
                lines[index] = (
                    '        "log/simulate_results_" + model_name.split("/")[-1]'
                    ' + f"_bs_{max_batch_size}_sp_{max_split_point}"'
                    ' + (f"_mps_{mps_gpu}" if mps_gpu < 100 else "")'
                    ' + ("_lora" if lora else "") + ".csv",'
                )
                break
        source = "\n".join(lines) + "\n"
        simulator.write_text(source, encoding="utf-8")
        environment = {"PYTHONPATH": temporary}
        subprocess.run([sys.executable, __file__, "--worker"], check=True, cwd=repository, env=environment)


if __name__ == "__main__":
    main()
