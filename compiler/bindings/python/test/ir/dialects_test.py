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
            assert reorder_attr.value == iree_gpu.ReorderWorkgroupsStrategy.Transpose

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


@lambda _: _()
def mma_intrinsic_attr():
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            mma_intrinsic_attr = iree_gpu.MMAIntrinsicAttr.get(
                iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16, ctx
            )
            assert mma_intrinsic_attr is not None
            assert (
                str(mma_intrinsic_attr)
                == "#iree_gpu<mma_intrinsic MFMA_F32_32x32x8_F16>"
            )

            raw_value = mma_intrinsic_attr.raw_value
            assert raw_value == iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
            value = mma_intrinsic_attr.value
            assert str(value) == "MFMA_F32_32x32x8_F16"
            assert int(value) == raw_value

            mma_attr = iree_gpu.MMAAttr.get(raw_value, ctx)
            assert mma_attr is not None

            f16 = ir.F16Type.get()
            f32 = ir.F32Type.get()
            a_type, b_type, c_type = mma_attr.abc_element_types
            assert a_type == f16
            assert b_type == f16
            assert c_type == f32

            vec_4xf16 = ir.VectorType.get((4,), f16)
            a_vec_type, b_vec_type, _c_vec_type = mma_attr.abc_vector_types
            assert a_vec_type == vec_4xf16
            assert b_vec_type == vec_4xf16

            M, N, K = mma_attr.mnk_shape
            assert M == 32
            assert N == 32
            assert K == 8

            assert mma_intrinsic_attr.mma == mma_attr


@lambda _: _()
def lowering_config_attr():
    with ir.Context() as ctx, ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            attributes = ir.DictAttr.get({"reduction": ir.ArrayAttr.get([])}, ctx)
            lowering_config = iree_gpu.LoweringConfigAttr.get(attributes, ctx)
            assert lowering_config is not None

            assert lowering_config.attributes == attributes
