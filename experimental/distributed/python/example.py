# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.experimental.runtime.distributed import run_ranks
import iree.compiler
import tempfile
import numpy as np
import os

"""
Example of distributed execution across 2 devices of a small model
with just an all-reduce operation.
all_reduce([1, 2, 3, 4], [5, 6, 7, 8]) -> [6, 8, 10, 12].

Dependecies at:
runtime/bindings/python/iree/runtime/distributed/setup.sh
"""
mlir = """
    func.func @all_reduce_sum(%input : tensor<4xf32>) -> tensor<4xf32> {
    %out = "stablehlo.all_reduce"(%input) ({
        ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
        %sum = stablehlo.add %arg0, %arg1 : tensor<f32>
        stablehlo.return %sum : tensor<f32>
        }) {channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
            replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
            use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
    return %out : tensor<4xf32>
    }
"""

inputs = [
    [np.array([1, 2, 3, 4], dtype=np.float32)],
    [np.array([5, 6, 7, 8], dtype=np.float32)],
]

for rank in range(len(inputs)):
    print(f"Rank {rank} argument = {inputs[rank]}")

with tempfile.TemporaryDirectory() as tmp_dir:
    module_filepath = os.path.join(tmp_dir, "module.vmfb")
    iree.compiler.tools.compile_str(
        input_str=mlir,
        output_file=module_filepath,
        target_backends=["cuda"],
        input_type="stablehlo",
    )

    num_ranks = len(inputs)
    # Ranks on the 0th axis.
    outputs = run_ranks(
        num_ranks=num_ranks,
        function="all_reduce_sum",
        driver="cuda",
        module_filepath=module_filepath,
        inputs=inputs,
    )
    for rank in range(num_ranks):
        print(f"Rank {rank} result = {outputs[rank]}")
