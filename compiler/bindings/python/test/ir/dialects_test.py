# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir


# Substitute `replace=True` so that colliding registration don't error.
# TODO(makslevental): remove after https://github.com/llvm/llvm-project/pull/117918 is resolved.
def register_attribute_builder(kind, replace=True):
    def decorator_builder(func):
        ir.AttrBuilder.insert(kind, func, replace=replace)
        return func

    return decorator_builder


ir.register_attribute_builder = register_attribute_builder

# Test upstream dialects import
from iree.compiler.dialects import (
    affine,
    amdgpu,
    arith,
    builtin,
    cf,
    complex,
    func,
    gpu,
    # TODO: importing linalg pulls yaml dependency, disable for now
    # linalg,
    llvm,
    math,
    memref,
    pdl,
    rocdl,
    scf,
    shape,
    tensor,
    tosa,
    transform,
    vector,
)

# Smoke test for vector transforms
from iree.compiler.dialects.transform import vector as vt

# Make sure that our dialects import.
from iree.compiler.dialects import flow, hal, stream, vm, util, iree_codegen, iree_gpu


def get_index_attr(val: int) -> ir.IntegerAttr:
    return ir.IntegerAttr.get(ir.IndexType.get(), val)


def get_index_array_attr(vals: list[int]) -> ir.ArrayAttr:
    return ir.ArrayAttr.get([get_index_attr(val) for val in vals])


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


@run
def codegen_translation_info_minimal():
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.None_
    )
    translation_info = iree_codegen.TranslationInfoAttr.get(pipeline_attr)
    assert translation_info is not None
    assert str(translation_info) == "#iree_codegen.translation_info<pipeline = None>"
    assert translation_info.pass_pipeline == pipeline_attr
    assert translation_info.codegen_spec is None
    assert translation_info.workgroup_size == []
    assert translation_info.subgroup_size == 0
    assert translation_info.configuration is None


@run
def codegen_translation_info_with_sizes():
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.Custom
    )
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, None, [64, 4, 1], 32
    )
    assert translation_info is not None
    assert translation_info.pass_pipeline == pipeline_attr
    assert translation_info.codegen_spec is None
    assert translation_info.workgroup_size == [64, 4, 1]
    assert translation_info.subgroup_size == 32
    assert translation_info.configuration is None


@run
def codegen_translation_info_full():
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.TransformDialectCodegen
    )
    foo_symbol = ir.SymbolRefAttr.get(["foo"])
    configuration = ir.DictAttr.get({"A": ir.IntegerAttr.get(ir.IndexType.get(), 42)})
    translation_info = iree_codegen.TranslationInfoAttr.get(
        pipeline_attr, foo_symbol, [128], 32, configuration
    )
    assert translation_info is not None
    assert translation_info.pass_pipeline == pipeline_attr
    assert translation_info.codegen_spec == foo_symbol
    assert translation_info.workgroup_size == [128]
    assert translation_info.subgroup_size == 32
    assert translation_info.configuration == configuration


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
    attributes = ir.DictAttr.get(
        {
            "reduction": get_index_array_attr([]),
        }
    )
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    assert lowering_config is not None

    assert lowering_config.attributes == attributes
    assert lowering_config.workgroup_tile_sizes == []
    assert lowering_config.reduction_tile_sizes == []
    assert lowering_config.subgroup_count_mn == (None, None)
    assert lowering_config.mma_kind == None

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    attributes = ir.DictAttr.get(
        {
            "reduction": get_index_array_attr([1]),
            "workgroup": get_index_array_attr([2, 3]),
            "subgroup_m_count": get_index_attr(1),
            "subgroup_n_count": get_index_attr(2),
            "mma_kind": mma_attr,
        }
    )
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    assert lowering_config.workgroup_tile_sizes == [2, 3]
    assert lowering_config.reduction_tile_sizes == [1]
    assert lowering_config.subgroup_count_mn == (1, 2)
    assert lowering_config.mma_kind == mma_attr
    assert (
        str(lowering_config.mma_kind) == "#iree_gpu.mma_layout<MFMA_F32_16x16x16_F16>"
    )


@run
def compilation_info():
    attributes = ir.DictAttr.get({"reduction": get_index_array_attr([])})
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    pipeline_attr = iree_codegen.DispatchLoweringPassPipelineAttr.get(
        iree_codegen.DispatchLoweringPassPipeline.None_
    )
    translation_info = iree_codegen.TranslationInfoAttr.get(pipeline_attr)

    compilation_info = iree_codegen.CompilationInfoAttr.get(
        lowering_config, translation_info
    )
    assert compilation_info is not None
    assert compilation_info.lowering_config == lowering_config
    assert compilation_info.translation_info == translation_info
