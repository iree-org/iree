# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir

# Make sure that our dialects import.
from iree.compiler.dialects import flow, hal, stream, vm, util, iree_codegen, iree_gpu


def run(fn):
    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            print("\nTEST:", fn.__name__)
            fn()
    return fn


# ======================================================================
# IREE Codegen Dialect
# ======================================================================


@run
def codegen_dispatch_lowering_pass_pipeline():
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
    )
    assert pipeline_attr is not None
    assert (
        pipeline_attr.value
        == iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
    )
    assert pipeline_attr.raw_value == int(
        iree_codegen.DispatchLoweringPassPipeline.LLVMGPUTileAndFuse
    )
    assert "LLVMGPUTileAndFuse" in str(pipeline_attr)


# ======================================================================
# IREE GPU Dialect
# ======================================================================


@run
def gpu_pipeline_options_attr():
    reorder_attr = iree_gpu.ReorderWorkgroupsStrategyAttr.get(
        iree_gpu.ReorderWorkgroupsStrategy.Transpose
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
            iree_gpu.ReorderWorkgroupsStrategy.Transpose
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


@run
def mma_intrinsic_attr():
    mma_intrinsic_attr = iree_gpu.MMAIntrinsicAttr.get(
        iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    )
    assert mma_intrinsic_attr is not None
    assert str(mma_intrinsic_attr) == "#iree_gpu<mma_intrinsic MFMA_F32_32x32x8_F16>"

    raw_value = mma_intrinsic_attr.raw_value
    assert raw_value == iree_gpu.MMAIntrinsic.MFMA_F32_32x32x8_F16
    value = mma_intrinsic_attr.value
    assert str(value) == "MFMA_F32_32x32x8_F16"
    assert int(value) == raw_value

    mma_attr = iree_gpu.MMAAttr.get(raw_value)
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


@run
def lowering_config_attr():
    attributes = ir.DictAttr.get({"reduction": ir.ArrayAttr.get([])})
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    assert lowering_config is not None

    assert lowering_config.attributes == attributes
