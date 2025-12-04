# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import iree.runtime as rt
import iree.compiler as compiler
import numpy as np
import os


def main():
    # Create a vmfb for the llvm-cpu backend.
    #
    # GPU debug hint: use iree-compile with all the flags to reproduce the
    # numerical error, plus `--iree-flow-trace-dispatch-tensors` for
    # trace points. Then use rt.VmModule.copy_buffer(config.vm_instance, .)
    # to load the vmfb.
    vmfb_contents = compiler.compile_file(
        os.path.join(os.path.dirname(__file__), "model.mlir"),
        target_backends=["llvm-cpu"],
        extra_args=[
            "--iree-flow-trace-dispatch-tensors",
            "--iree-llvmcpu-target-cpu=host",
        ],
    )

    # GPU debug hint: Use 'hip' if running on an AMD GPU.
    config = rt.Config("local-sync")

    # The callback records the means of all tensors.
    callback_results = []

    def callback(key: str, buffer_views: list[rt.HalBufferView]):
        for i, bv in enumerate(buffer_views):
            arr = bv.map().asarray(
                bv.shape, rt.HalElementType.map_to_dtype(bv.element_type)
            )
            callback_results.append([key, i, float(arr.sum())])

    hal_module = rt.create_hal_module(
        config.vm_instance,
        config.device,
        debug_sink=rt.HalModuleDebugSink(callback),
    )

    module = rt.VmModule.copy_buffer(config.vm_instance, vmfb_contents)
    vm_modules = rt.load_vm_modules(hal_module, module, config=config)

    # Perform softmax(matmul(A, B)):
    A = 0.25 * np.ones((4, 16), dtype=np.float32)
    B = np.ones((16, 16), dtype=np.float32)
    vm_modules[-1].main(A, B)

    # We assume that softmax(matmul(A, B)) compiled to 2 dispatches.
    assert (
        len(callback_results) == 5
    ), "2 dispatches. mm:2->1, sm:1->1. Expected 5 tensors."

    # A
    dispatch_name = callback_results[0][0]
    tensor_index = callback_results[0][1]
    tensor_sum = callback_results[0][2]
    assert dispatch_name.startswith("main_dispatch_0")
    assert dispatch_name.endswith("inputs")
    assert tensor_index == 0
    assert tensor_sum == 16.0

    # B
    assert callback_results[1][0].startswith("main_dispatch_0")
    assert callback_results[1][0].endswith("inputs")
    assert callback_results[1][1] == 1
    assert callback_results[1][2] == 256.0

    # matmul(A, B)
    assert callback_results[2][0].startswith("main_dispatch_0")
    assert callback_results[2][0].endswith("outputs")
    assert callback_results[2][1] == 0
    assert callback_results[2][2] == 256.0

    # softmax inputs: matmul(A, B)
    assert callback_results[3][0].startswith("main_dispatch_1")
    assert callback_results[3][0].endswith("inputs")
    assert callback_results[3][1] == 0
    assert (
        callback_results[3][2] == callback_results[2][2]
    ), "Softmax input sum should match matmul output sum."

    # softmax(matmul(A, B))
    assert callback_results[4][0].startswith("main_dispatch_1")
    assert callback_results[4][0].endswith("outputs")
    assert callback_results[4][1] == 0
    assert callback_results[4][2] == 4.0


if __name__ == "__main__":
    main()
