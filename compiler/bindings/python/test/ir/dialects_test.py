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
    nvvm,
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
from iree.compiler.dialects.transform import loop

# Make sure that our dialects import.
from iree.compiler.dialects import (
    flow,
    hal,
    stream,
    vm,
    util,
    iree_codegen,
    iree_gpu,
    preprocessing_transform,
)


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

    # Fragment: 0 = lrhs, 1 = rhs, 2 = acc.
    fragment = 0
    mma_single_subgroup_layout = iree_gpu.get_single_subgroup_layout(
        attr=mma_intrinsic_attr, fragment=fragment
    )
    assert isinstance(mma_single_subgroup_layout, iree_gpu.GPUMMASingleSubgroupLayout)
    assert mma_single_subgroup_layout.outer == [1, 1]
    assert mma_single_subgroup_layout.thread == [32, 2]
    assert mma_single_subgroup_layout.tstrides == [1, 32]
    assert mma_single_subgroup_layout.element == [1, 4]

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
    vec_16xf32 = ir.VectorType.get((16,), f32)
    a_vec_type, b_vec_type, c_vec_type = mma_attr.abc_vector_types
    assert a_vec_type == vec_4xf16
    assert b_vec_type == vec_4xf16
    assert c_vec_type == vec_16xf32

    M, N, K = mma_attr.mnk_shape
    assert M == 32
    assert N == 32
    assert K == 8

    assert mma_intrinsic_attr.mma == mma_attr

    virtual_mma_intrinsics = mma_attr.get_virtual_intrinsics()
    assert isinstance(virtual_mma_intrinsics[0], iree_gpu.VirtualMMAIntrinsic)
    assert (
        virtual_mma_intrinsics[0] == iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_32x32x16_F16
    )

    mma_attr = iree_gpu.MMAAttr.get(iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32)
    virtual_mma_intrinsics = mma_attr.get_virtual_intrinsics()
    assert virtual_mma_intrinsics == []


@run
def virtual_mma_intrinsic_attr():
    virtual_mma_intrinsic_attr = iree_gpu.VirtualMMAIntrinsicAttr.get(
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16
    )
    assert virtual_mma_intrinsic_attr is not None
    assert (
        str(virtual_mma_intrinsic_attr)
        == "#iree_gpu<virtual_mma_intrinsic VMFMA_F32_16x16x32_F16>"
    )

    # Fragment: 0 = lhs, 1 = rhs, 2 = acc.
    fragment = 0
    virtual_mma_single_subgroup_layout = iree_gpu.get_single_subgroup_layout(
        virtual_mma_intrinsic_attr, fragment
    )
    assert isinstance(
        virtual_mma_single_subgroup_layout, iree_gpu.GPUMMASingleSubgroupLayout
    )
    assert virtual_mma_single_subgroup_layout.outer == [1, 1]
    assert virtual_mma_single_subgroup_layout.thread == [16, 4]
    assert virtual_mma_single_subgroup_layout.tstrides == [
        1,
        16,
    ]
    assert virtual_mma_single_subgroup_layout.element == [1, 8]

    raw_value = virtual_mma_intrinsic_attr.raw_value
    assert raw_value == iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16
    value = virtual_mma_intrinsic_attr.value
    assert str(value) == "VMFMA_F32_16x16x32_F16"
    assert int(value) == raw_value

    virtual_mma_attr = iree_gpu.VirtualMMAAttr.get(raw_value)
    assert virtual_mma_attr is not None

    f16 = ir.F16Type.get()
    f32 = ir.F32Type.get()
    a_type, b_type, c_type = virtual_mma_attr.abc_element_types
    assert a_type == f16
    assert b_type == f16
    assert c_type == f32

    vec_4xf32 = ir.VectorType.get((4,), f32)
    vec_8xf16 = ir.VectorType.get((8,), f16)
    a_vec_type, b_vec_type, c_vec_type = virtual_mma_attr.abc_vector_types
    assert a_vec_type == vec_8xf16
    assert b_vec_type == vec_8xf16
    assert c_vec_type == vec_4xf32

    M, N, K = virtual_mma_attr.mnk_shape
    assert M == 16
    assert N == 16
    assert K == 32

    assert virtual_mma_intrinsic_attr.mma == virtual_mma_attr


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
    assert lowering_config.mma_kind == None
    assert lowering_config.subgroup_basis == ([], [])

    mma_intrinsic = iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16
    mma_attr = iree_gpu.MMAAttr.get(mma_intrinsic)
    attributes = ir.DictAttr.get(
        {
            "reduction": get_index_array_attr([1]),
            "workgroup": get_index_array_attr([2, 3]),
            "mma_kind": mma_attr,
            "subgroup_basis": ir.ArrayAttr.get(
                [get_index_array_attr([1, 1, 4]), get_index_array_attr([0, 1, 2])]
            ),
        }
    )
    lowering_config = iree_gpu.LoweringConfigAttr.get(attributes)
    assert lowering_config.workgroup_tile_sizes == [2, 3]
    assert lowering_config.reduction_tile_sizes == [1]
    assert lowering_config.mma_kind == mma_attr
    assert lowering_config.subgroup_basis == ([1, 1, 4], [0, 1, 2])
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


@run
def gpu_target_info_attribute_parsing():
    mlir_string = """
    hal.executable private @main_dispatch_0 {
        hal.executable.variant public @rocm_hsaco_fb
            target(<"rocm", "rocm-hsaco-fb",
                {
                abi = "hip",
                iree_codegen.target_info = #iree_gpu.target<
                    arch = "gfx942",
                    features = "",
                    wgp = <
                    compute = fp64,
                    storage = b64,
                    subgroup = none,
                    dot = none,
                    mma = [<MFMA_F32_16x16x4_F32>, <MFMA_F32_16x16x16_F16>],
                    subgroup_size_choices = [32, 64],
                    max_workgroup_sizes = [256, 512, 1024],
                    max_thread_count_per_workgroup = 1024,
                    max_workgroup_memory_bytes = 65536,
                    max_workgroup_counts = [256, 512, 1024]
                    >
                >
                }>
            ) {
        }
    }
    """

    module = ir.Module.parse(mlir_string)
    variant_op_list = iree_codegen.get_executable_variant_ops(module)
    assert len(variant_op_list) == 1, "Expect one executable variant op"
    variant_op = variant_op_list[0]
    executable_variant_op = variant_op.opview
    target = executable_variant_op.target
    gpu_target_info = iree_gpu.TargetInfo.get_gpu_target_info(target)

    arch = gpu_target_info.arch
    assert arch == "gfx942", f"Expected arch 'gfx942', got '{arch}'"

    subgroup_size_choices = gpu_target_info.subgroup_size_choices
    assert subgroup_size_choices == [
        32,
        64,
    ], f"Expected subgroup_size_choice [32, 64], got {subgroup_size_choices}"

    max_thread_count = gpu_target_info.max_thread_count_per_workgroup
    assert (
        max_thread_count == 1024
    ), f"Expected max_thread_count_per_workgroup 1024, got {max_thread_count}"

    max_memory_bytes = gpu_target_info.max_workgroup_memory_bytes
    assert (
        max_memory_bytes == 65536
    ), f"Expected max_workgroup_memory_bytes 65536, got {max_memory_bytes}"

    max_workgroup_sizes = gpu_target_info.max_workgroup_sizes
    assert max_workgroup_sizes == [
        256,
        512,
        1024,
    ], f"Expected max_workgroup_sizes [256, 512, 1024], got {max_workgroup_sizes}"

    mma_intrinsics = gpu_target_info.mma_intrinsics
    assert mma_intrinsics == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
    ], f"Expected mma_intrinsics [MFMA_F32_16x16x4_F32, MFMA_F32_16x16x16_F16, VMFMA_F32_16x16x32_F16], got {mma_intrinsics}"


@run
def gpu_target_info_constructor():
    context = ir.Context()

    target_info = iree_gpu.TargetInfo(
        context=context,
        arch="gfx942",
        subgroup_size_choices=[32, 64],
        max_workgroup_sizes=[256, 512, 1024],
        max_thread_count_per_workgroup=1024,
        max_workgroup_memory_bytes=65536,
        mma_intrinsics=[
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
            iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
            iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
        ],
    )

    assert (
        target_info.arch == "gfx942"
    ), f"Expected arch 'gfx942', got '{target_info.arch}'"
    assert target_info.subgroup_size_choices == [
        32,
        64,
    ], f"Expected subgroup_size_choices [32, 64], got {target_info.subgroup_size_choices}"
    assert target_info.max_workgroup_sizes == [
        256,
        512,
        1024,
    ], f"Expected max_workgroup_sizes [256, 512, 1024], got {target_info.max_workgroup_sizes}"
    assert (
        target_info.max_thread_count_per_workgroup == 1024
    ), f"Expected max_thread_count_per_workgroup 1024, got {target_info.max_thread_count_per_workgroup}"
    assert (
        target_info.max_workgroup_memory_bytes == 65536
    ), f"Expected max_workgroup_memory_bytes 65536, got {target_info.max_workgroup_memory_bytes}"
    mma_intrinsics = target_info.mma_intrinsics
    assert mma_intrinsics == [
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x4_F32,
        iree_gpu.MMAIntrinsic.MFMA_F32_16x16x16_F16,
        iree_gpu.VirtualMMAIntrinsic.VMFMA_F32_16x16x32_F16,
    ], f"Expected mma_intrinsics [MFMA_F32_16x16x4_F32, MFMA_F32_16x16x16_F16, VMFMA_F32_16x16x32_F16], got {mma_intrinsics}"

    assert isinstance(mma_intrinsics[0], iree_gpu.MMAIntrinsic)
    assert isinstance(mma_intrinsics[1], iree_gpu.MMAIntrinsic)
    assert isinstance(mma_intrinsics[2], iree_gpu.VirtualMMAIntrinsic)


@run
def gpu_target_info_constructor_error_cases():
    context = ir.Context()

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch=123,  # should be string.
            subgroup_size_choices=[32, 64],
            max_workgroup_sizes=[256, 512, 1024],
            max_thread_count_per_workgroup=1024,
            max_workgroup_memory_bytes=65536,
            mma_intrinsics=[],
        )
        assert False, "Expected TypeError for wrong arch type"
    except TypeError:
        pass

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch="gfx942",
            subgroup_size_choices=[64.0],  # should be list of int.
            max_workgroup_sizes=[256, 512, 1024],
            max_thread_count_per_workgroup=1024,
            max_workgroup_memory_bytes=65536,
            mma_intrinsics=[],
        )
        assert False, "Expected TypeError for wrong subgroup_size_choices type"
    except TypeError:
        pass

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch="gfx942",
            subgroup_size_choices=[32, 64],
            max_workgroup_sizes=[256.0, 512, 1024],  # should be list of int.
            max_thread_count_per_workgroup=1024,
            max_workgroup_memory_bytes=65536,
            mma_intrinsics=[],
        )
        assert False, "Expected TypeError for wrong max_workgroup_sizes type"
    except TypeError:
        pass

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch="gfx942",
            subgroup_size_choices=[32, 64],
            max_workgroup_sizes=[256, 512, 1024],
            max_thread_count_per_workgroup=1024.0,  # should be int.
            max_workgroup_memory_bytes=65536,
            mma_intrinsics=[],
        )
        assert False, "Expected TypeError for wrong max_thread_count_per_workgroup type"
    except TypeError:
        pass

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch="gfx942",
            subgroup_size_choices=[32, 64],
            max_workgroup_sizes=[256, 512, 1024],
            max_thread_count_per_workgroup=1024,
            max_workgroup_memory_bytes=65536.0,  # should be int.
            mma_intrinsics=[],
        )
        assert False, "Expected TypeError for wrong max_workgroup_memory_bytes type"
    except TypeError:
        pass

    try:
        iree_gpu.TargetInfo(
            context=context,
            arch="gfx942",
            subgroup_size_choices=[32, 64],
            max_workgroup_sizes=[256, 512, 1024],
            max_thread_count_per_workgroup=1024,
            max_workgroup_memory_bytes=65536,
            mma_intrinsics=[123],  # should be MMA intrinsic objects.
        )
        assert False, "Expected TypeError for wrong MMA intrinsic object type"
    except TypeError:
        pass


# ======================================================================
# Preprocessing Transform Extensions
# ======================================================================


@run
def preprocessing_transform_match_contraction_in_named_sequence():
    module_op = ir.Module.create()
    module_op.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    map_lhs = ir.AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(2)],
    )
    map_rhs = ir.AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[ir.AffineExpr.get_dim(2), ir.AffineExpr.get_dim(1)],
    )
    map_output = ir.AffineMap.get(
        dim_count=3,
        symbol_count=0,
        exprs=[ir.AffineExpr.get_dim(0), ir.AffineExpr.get_dim(1)],
    )

    with ir.InsertionPoint(module_op.body):
        named_seq = transform.NamedSequenceOp(
            "match_matmul", [transform.AnyOpType.get()], [transform.AnyOpType.get()]
        )
        with ir.InsertionPoint(named_seq.body):
            batch, m, n, k = preprocessing_transform.MatchContractionOp(
                operand_handle=named_seq.bodyTarget,
                lhs_type=ir.F32Type.get(),
                rhs_type=ir.F32Type.get(),
                output_type=ir.F32Type.get(),
                indexing_maps=[map_lhs, map_rhs, map_output],
            )
            transform.YieldOp([named_seq.bodyTarget])

    module_str = str(module_op)
    assert "affine_map<(d0, d1, d2) -> (d0, d2)>" in module_str
    assert "affine_map<(d0, d1, d2) -> (d2, d1)>" in module_str
    assert "affine_map<(d0, d1, d2) -> (d0, d1)>" in module_str
    assert "transform.with_named_sequence" in module_str
    assert "transform.named_sequence @match_matmul" in module_str
    assert "transform.iree.match.contraction %arg0" in module_str
    assert "lhs_type = f32, rhs_type = f32, output_type = f32" in module_str
    assert "indexing_maps = [#map, #map1, #map2]" in module_str


@run
def preprocessing_transform_match_convolution_in_named_sequence():
    module_op = ir.Module.create()
    module_op.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()

    map_input = ir.AffineMap.get(
        dim_count=7,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1) + ir.AffineExpr.get_dim(4),
            ir.AffineExpr.get_dim(2) + ir.AffineExpr.get_dim(5),
            ir.AffineExpr.get_dim(6),
        ],
    )
    map_filter = ir.AffineMap.get(
        dim_count=7,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(4),
            ir.AffineExpr.get_dim(5),
            ir.AffineExpr.get_dim(6),
            ir.AffineExpr.get_dim(3),
        ],
    )
    map_output = ir.AffineMap.get(
        dim_count=7,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1),
            ir.AffineExpr.get_dim(2),
            ir.AffineExpr.get_dim(3),
        ],
    )

    with ir.InsertionPoint(module_op.body):
        named_seq = transform.NamedSequenceOp(
            "match_conv", [transform.AnyOpType.get()], [transform.AnyOpType.get()]
        )
        with ir.InsertionPoint(named_seq.body):
            (
                batch,
                out_img,
                out_ch,
                filt,
                in_ch,
                depth,
                strides,
                dilations,
            ) = preprocessing_transform.MatchConvolutionOp(
                operand_handle=named_seq.bodyTarget,
                lhs_type=ir.F32Type.get(),
                rhs_type=ir.F32Type.get(),
                output_type=ir.F32Type.get(),
                indexing_maps=[map_input, map_filter, map_output],
            )
            transform.YieldOp([named_seq.bodyTarget])

    module_str = str(module_op)
    assert (
        "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1 + d4, d2 + d5, d6)>"
        in module_str
    )
    assert "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>" in module_str
    assert "affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>" in module_str
    assert "transform.with_named_sequence" in module_str
    assert "transform.named_sequence @match_conv" in module_str
    assert "transform.iree.match.convolution %arg0" in module_str
    assert "lhs_type = f32, rhs_type = f32, output_type = f32" in module_str
    assert "indexing_maps = [#map, #map1, #map2]" in module_str


@run
def preprocessing_transform_match_attention_in_named_sequence():
    module_op = ir.Module.create()
    module_op.operation.attributes["transform.with_named_sequence"] = ir.UnitAttr.get()
    map_query = ir.AffineMap.get(
        dim_count=6,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1),
            ir.AffineExpr.get_dim(2),
            ir.AffineExpr.get_dim(4),
        ],
    )
    map_key = ir.AffineMap.get(
        dim_count=6,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1),
            ir.AffineExpr.get_dim(5),
            ir.AffineExpr.get_dim(4),
        ],
    )
    map_value = ir.AffineMap.get(
        dim_count=6,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1),
            ir.AffineExpr.get_dim(3),
            ir.AffineExpr.get_dim(5),
        ],
    )
    map_scale = ir.AffineMap.get(
        dim_count=6,
        symbol_count=0,
        exprs=[],
    )
    map_output = ir.AffineMap.get(
        dim_count=6,
        symbol_count=0,
        exprs=[
            ir.AffineExpr.get_dim(0),
            ir.AffineExpr.get_dim(1),
            ir.AffineExpr.get_dim(2),
            ir.AffineExpr.get_dim(3),
        ],
    )

    with ir.InsertionPoint(module_op.body):
        named_seq = transform.NamedSequenceOp(
            "match_attention", [transform.AnyOpType.get()], [transform.AnyOpType.get()]
        )
        with ir.InsertionPoint(named_seq.body):
            batch, m, n, k1, k2 = preprocessing_transform.MatchAttentionOp(
                operand_handle=named_seq.bodyTarget,
                query_type=ir.F16Type.get(),
                key_type=ir.F16Type.get(),
                value_type=ir.F16Type.get(),
                output_type=ir.F16Type.get(),
                indexing_maps=[map_query, map_key, map_value, map_scale, map_output],
            )
            transform.YieldOp([named_seq.bodyTarget])

    module_str = str(module_op)
    assert "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d4)>" in module_str
    assert "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d5, d4)>" in module_str
    assert "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d5)>" in module_str
    assert "affine_map<(d0, d1, d2, d3, d4, d5) -> ()>" in module_str
    assert "affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3)>" in module_str
    assert "transform.with_named_sequence" in module_str
    assert "transform.named_sequence @match_attention" in module_str
    assert "transform.iree.match.attention %arg0" in module_str
    assert (
        "query_type = f16, key_type = f16, value_type = f16, output_type = f16"
        in module_str
    )
    assert "indexing_maps = [#map, #map1, #map2, #map3, #map4]" in module_str
