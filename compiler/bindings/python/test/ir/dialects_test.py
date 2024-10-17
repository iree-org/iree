# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir

# Make sure that our dialects import.
from iree.compiler.dialects import flow, hal, stream, vm, util, iree_gpu
from iree.compiler.dialects._iree_gpu_enum_gen import (
    _ireegpu_reorderworkgroupsstrategyattr as ReorderWorkgroupsStrategyAttr,
)


@lambda _: _()
def gpu_pipeline_options_attr():
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            reorder_attr = ReorderWorkgroupsStrategyAttr(
                iree_gpu.ReorderWorkgroupsStrategy.Swizzle, ctx
            )
            gpu_attr = iree_gpu.GPUPipelineOptionsAttr.get(
                True,
                False,
                reorder_attr,
            )
            assert (
                str(gpu_attr.reorder_work_groups_strategy)
                == "#iree_gpu<reorder_work_groups_strategy Swizzle>"
            )
            assert (
                str(gpu_attr)
                == "#iree_gpu.pipeline_options<prefetch_shared_memory = true, no_reduce_shared_memory_bank_conflicts = false, reorder_workgroups_strategy =  Swizzle>"
            )
            assert type(gpu_attr) is iree_gpu.GPUPipelineOptionsAttr
            assert gpu_attr.prefetch_shared_memory
            assert not gpu_attr.no_reduce_shared_memory_bank_conflicts

            gpu_attr = iree_gpu.GPUPipelineOptionsAttr.get(
                False,
                True,
                ReorderWorkgroupsStrategyAttr(
                    iree_gpu.ReorderWorkgroupsStrategy.Transpose, ctx
                ),
            )
            assert not gpu_attr.prefetch_shared_memory
            assert gpu_attr.no_reduce_shared_memory_bank_conflicts
