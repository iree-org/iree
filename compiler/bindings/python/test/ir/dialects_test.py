# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir

# Make sure that our dialects import.
from iree.compiler.dialects import flow, hal, stream, vm, util, iree_gpu


@lambda _: _()
def gpu_pipeline_options_attr():
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            reorder_attr = iree_gpu.ReorderWorkgroupsStrategyAttr.get(
                iree_gpu.ReorderWorkgroupsStrategy.Transpose, ctx
            )
            gpu_attr = iree_gpu.PipelineOptionsAttr.get(
                True,
                False,
                False,
                reorder_attr,
            )
            assert type(gpu_attr) is iree_gpu.PipelineOptionsAttr
            assert gpu_attr.prefetch_shared_memory
            assert not gpu_attr.no_reduce_shared_memory_bank_conflicts
            assert not gpu_attr.use_igemm_convolution

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(
                False,
                True,
                True,
                iree_gpu.ReorderWorkgroupsStrategyAttr.get(
                    iree_gpu.ReorderWorkgroupsStrategy.Transpose, ctx
                ),
            )
            assert not gpu_attr.prefetch_shared_memory
            assert gpu_attr.no_reduce_shared_memory_bank_conflicts
            assert gpu_attr.use_igemm_convolution

            gpu_attr = iree_gpu.PipelineOptionsAttr.get()
            assert (
                gpu_attr.prefetch_shared_memory is None
                and gpu_attr.no_reduce_shared_memory_bank_conflicts is None
                and gpu_attr.use_igemm_convolution is None
                and gpu_attr.reorder_workgroups_strategy is None
            )

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(True)
            assert gpu_attr.prefetch_shared_memory
            assert (
                gpu_attr.no_reduce_shared_memory_bank_conflicts is None
                and gpu_attr.use_igemm_convolution is None
                and gpu_attr.reorder_workgroups_strategy is None
            )

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(True, False)
            assert (
                gpu_attr.use_igemm_convolution is None
                and gpu_attr.reorder_workgroups_strategy is None
            )

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(True, False, False)
            assert gpu_attr.reorder_workgroups_strategy is None

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(
                no_reduce_shared_memory_bank_conflicts=False
            )
            assert (
                gpu_attr.no_reduce_shared_memory_bank_conflicts is not None
                and not gpu_attr.no_reduce_shared_memory_bank_conflicts
            )
            assert gpu_attr.prefetch_shared_memory is None
            assert gpu_attr.use_igemm_convolution is None
            assert gpu_attr.reorder_workgroups_strategy is None

            gpu_attr = iree_gpu.PipelineOptionsAttr.get(
                reorder_workgroups_strategy=reorder_attr
            )
            assert gpu_attr.reorder_workgroups_strategy is not None
            assert (
                gpu_attr.reorder_workgroups_strategy.value
                # unfortunately not `is`
                == iree_gpu.ReorderWorkgroupsStrategy.Transpose
            )
