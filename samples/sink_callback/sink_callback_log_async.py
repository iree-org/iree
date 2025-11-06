# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import threading
from typing import Callable, List, Dict
import numpy as np
import iree.runtime as rt
import iree.compiler as compiler


class MeanTracker:
    """Stores tensor means during execution, and compares with a peer -- all within a callback function."""

    def __init__(self, name: str, my_stats: List[Dict], peer_stats: List[Dict]):
        self.name = name
        self.my_stats = my_stats
        self.peer_stats = peer_stats
        # Indices in [0, self.index) have already been compared to peer stats.
        self.index = 0
        # Differences with peer will be logged here.
        self.messages: List[dict] = []

    def callback(self, key: str, buffer_views: List[rt.HalBufferView]):
        for tensor_index, bv in enumerate(buffer_views):
            arr = bv.map().asarray(
                bv.shape, rt.HalElementType.map_to_dtype(bv.element_type)
            )
            self.my_stats.append(
                {"mean": float(arr.mean()), "key": key, "index": tensor_index}
            )

        new_index = min(len(self.my_stats), len(self.peer_stats))
        for i in range(self.index, new_index):
            self._compare(i)
        self.index = new_index

    def _compare(self, i: int):
        diff = abs(self.my_stats[i]["mean"] - self.peer_stats[i]["mean"])
        if diff > 1e-6:
            self.messages.append(
                {
                    "name": self.name,
                    "key": self.my_stats[i]["key"],
                    "index": self.my_stats[i]["index"],
                    "my_mean": self.my_stats[i]["mean"],
                    "peer_mean": self.peer_stats[i]["mean"],
                    "difference": diff,
                }
            )


def run_model(
    model_vmfb: bytes,
    driver_vmfb: bytes,
    callback: Callable[[str, List[rt.HalBufferView]], None],
    config: rt.Config,
    executor: Callable[[rt.VmModule], None],
    weights: rt.ParameterProvider,
):
    model_module = rt.VmModule.copy_buffer(config.vm_instance, model_vmfb)
    driver_module = rt.VmModule.copy_buffer(config.vm_instance, driver_vmfb)
    io_module = rt.create_io_parameters_module(config.vm_instance, weights)

    hal_module = rt.create_hal_module(
        config.vm_instance, config.device, debug_sink=rt.HalModuleDebugSink(callback)
    )

    modules = rt.load_vm_modules(
        hal_module, io_module, model_module, driver_module, config=config
    )
    executor(modules[-1])


def main():
    config = rt.Config("local-sync")
    base_dir = os.path.dirname(__file__)

    # --- Compile both models (Model B differs only by a constant) ---
    vmfb_a = compiler.compile_file(
        os.path.join(base_dir, "model_a.mlir"),
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-flow-trace-dispatch-tensors",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )

    vmfb_b = compiler.compile_file(
        os.path.join(base_dir, "model_b.mlir"),
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-flow-trace-dispatch-tensors",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )

    # --- Compile driver ---
    driver_path = os.path.join(base_dir, "driver_for_async.mlir")
    vmfb_driver = compiler.compile_file(
        driver_path,
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-flow-trace-dispatch-tensors",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )

    # --- Provide weights ---
    weights = np.arange(18 * 18, dtype=np.float32).reshape(18, 18) / (18 * 18)
    params = rt.ParameterIndex()
    params.add_buffer("my_weights", weights)
    provider = params.create_provider(scope="my_scope")

    # --- Execution function for both models ---
    def executor(module: rt.VmModule):
        module.main(np.ones((4, 18), np.float32))

    stats_a, stats_b = [], []
    tracker_a = MeanTracker("A", stats_a, stats_b)
    tracker_b = MeanTracker("B", stats_b, stats_a)

    threads = [
        threading.Thread(
            target=run_model,
            args=(vmfb_a, vmfb_driver, tracker_a.callback, config, executor, provider),
        ),
        threading.Thread(
            target=run_model,
            args=(vmfb_b, vmfb_driver, tracker_b.callback, config, executor, provider),
        ),
    ]

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(tracker_a.messages) < 2
    assert len(tracker_b.messages) < 2
    assert len(tracker_a.messages) + len(tracker_b.messages) > 0
    messages = tracker_a.messages + tracker_b.messages
    for message in messages:
        # Output 0 of the dispatch containing the generic is the one that differs:
        assert "generic" in message["key"]
        assert "outputs" in message["key"]
        assert message["index"] == 0
        assert np.abs(message["difference"] - np.abs(np.pi - np.e)) < 1e-3


if __name__ == "__main__":
    main()
