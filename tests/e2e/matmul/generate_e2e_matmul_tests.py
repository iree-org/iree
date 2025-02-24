#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""iree_generated_e2e_matmul_test generator for e2e matmul tests.
"""

from typing import Optional
import argparse
import os
import re
import enum
import dataclasses
import typing
import itertools


# Data type of matrix entries. The string values must match MLIR data types.
# This is a superset of the values accepted for the --lhs_rhs_types= flag,
# as this also includes accumulator-specific types like i32.
@enum.unique
class MatrixElemTypeId(enum.Enum):
    NONE = ""
    I8 = "i8"
    I32 = "i32"
    F64 = "f64"
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    F8E5M2 = "f8E5M2"
    F8E4M3 = "f8E4M3"
    F8E5M2FNUZ = "f8E5M2FNUZ"
    F8E4M3FNUZ = "f8E4M3FNUZ"


# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
    DEFAULT = "default"
    SMALL = "small"
    LARGE = "large"
    EASY_LARGE_STATIC = "easy_large_static"


# Enumerates of the collections of compilation info that we can generate tests
# for. The values are the accepted values for the --compilation_info= flag.
@enum.unique
class CompilationInfoId(enum.Enum):
    NONE = ""
    LLVMGPUMatmulTensorCore = "LLVMGPUMatmulTensorCore"
    LLVMGPUMatmulTensorCoreMmaSync = "LLVMGPUMatmulTensorCoreMmaSync"
    LLVMGPUVectorDistributeMFMA = "LLVMGPUVectorDistributeMFMA"
    LLVMGPUVectorDistributeWMMAR3 = "LLVMGPUVectorDistributeWMMAR3"
    SPIRVCooperativeMatrixVectorize = "SPIRVCooperativeMatrixVectorize"
    SPIRVVectorizeMali = "SPIRVVectorizeMali"
    SPIRVVectorizeNVIDIA = "SPIRVVectorizeNVIDIA"


# Enumerates ways to construct MLIR tensor types.
@enum.unique
class Dynamicity(enum.Enum):
    DYNAMIC = "dynamic"  # Use '?' everywhere. Example: tensor<?x?xf32>.
    STATIC = "static"  # Use fixed values everywhere. Example: tensor<4x6xf32>.
    MIXED = "mixed"  # Randomly mix '?' and values. Example: tensor<?x4xf32>.


# Enumerates ways to initialize matrix buffer contents.
@enum.unique
class MatrixGenerator(enum.Enum):
    ZERO = "zero"  # Fill with zeros
    RANDOM = "random"  # Fill with (deterministic) pseudorandom values.


# Describes the shape of a matrix multiplication in the usual convention:
# the LHS is {m}x{k}, the RHS is {k}x{n}, the accumulator/result is {m}x{n}.
# The extra `accumulate` boolean tells whether the matmul is accumulating into
# an existing accumulator (C += A * B) or just overwriting the result
# (C = A * B).
@dataclasses.dataclass
class TestShape:
    m: int
    k: int
    n: int
    accumulate: bool


# Describes a workgroup and tiling schedule to target a specific MMA intrinsic.
@dataclasses.dataclass
class MMASchedule:
    intrinsic: str
    m_count: int  # Number of subgroups per workgroup along M
    n_count: int  # Number of subgroups per workgroup along N
    m_tile_count: int
    n_tile_count: int
    k_tile_count: int

    def __str__(self):
        return (
            "mma_schedule = #iree_gpu.mma_schedule<"
            + f"intrinsic = #iree_gpu.mma_layout<{self.intrinsic}>, "
            + f"subgroup_m_count = {self.m_count}, "
            + f"subgroup_n_count = {self.n_count}>"
        )


# Describes how to construct compilation info for the testcase.
@dataclasses.dataclass
class CompilationInfo:
    # Compilation info
    workgroup_size: typing.List[int]
    subgroup_size: Optional[int]
    # Translation info
    dispatch_lowering_pass_pipeline: str

    # Prints the workgroup size
    def workgroup_size_str(self):
        return "workgroup_size = [" + ", ".join(map(str, self.workgroup_size)) + "]"

    def get_compilation_info_attr(self) -> str:
        ...


@dataclasses.dataclass
class IREEGPUCompilationInfo(CompilationInfo):
    # Lowering Config
    workgroup_tile: list[int]
    reduction_tile: list[int]
    # Translation Info
    mma_schedule: Optional[MMASchedule]

    def get_compilation_info_attr(self) -> str:
        requested_pipeline = self.dispatch_lowering_pass_pipeline
        compiler_pipeline = requested_pipeline

        subgroup_size_str = ""
        if self.subgroup_size is not None:
            subgroup_size_str = f"subgroup_size = {self.subgroup_size}"

        return (
            "#iree_codegen.compilation_info<\n"
            f"  lowering_config = #iree_gpu.lowering_config<{{"
            f"  mma_kind = #iree_gpu.mma_layout<{self.mma_schedule.intrinsic}>, "
            f"  subgroup_m_count = {self.mma_schedule.m_count}, "
            f"  subgroup_n_count = {self.mma_schedule.n_count}, "
            f"  workgroup = {self.workgroup_tile}, "
            f"  reduction = {self.reduction_tile} }}>,\n"
            f"  translation_info = #iree_codegen.translation_info<pipeline = {compiler_pipeline} {self.workgroup_size_str()}\n"
            f"  {subgroup_size_str}>>\n"
        )


@dataclasses.dataclass
class LegacyCompilationInfo(CompilationInfo):
    # Lowering Config
    tile_sizes: typing.List[typing.List[int]]
    # Translation Info
    software_pipeline_depth: int

    def get_compilation_info_attr(self) -> str:
        requested_pipeline = self.dispatch_lowering_pass_pipeline
        compiler_pipeline = requested_pipeline
        if requested_pipeline == "SPIRVVectorizeMali":
            compiler_pipeline = "SPIRVBaseVectorize"
        elif requested_pipeline == "SPIRVCooperativeMatrixVectorize":
            compiler_pipeline = "SPIRVCooperativeMatrixVectorize"
        elif requested_pipeline == "SPIRVVectorizeNVIDIA":
            # TODO: change to test SPIRVMatmulPromoteVectorize too
            compiler_pipeline = "SPIRVBaseVectorize"

        subgroup_size_str = ""
        if self.subgroup_size is not None:
            subgroup_size_str = f"subgroup_size = {self.subgroup_size}"

        return (
            "#iree_codegen.compilation_info<\n"
            f"  lowering_config = #iree_codegen.lowering_config<tile_sizes = {self.tile_sizes}>,\n"
            f"  translation_info = #iree_codegen.translation_info<pipeline = {compiler_pipeline} {self.workgroup_size_str()}\n"
            f"  {subgroup_size_str},\n"
            f"  {{ pipeline_depth = {self.software_pipeline_depth}, store_stage = 1}}>>"
        )


# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id.
def get_test_shapes(shapes_id: ShapesId):
    # Notes:
    # 1. Be conservative in adding more shapes, as that can increase both the
    #    build and execution latency of tests. The build latency is nearly the
    #    same for all shapes, while execution latency grows cubicly i.e.
    #    linearly with m*k*n.
    # 2. Some shapes are commented out: they used to be tested but have been
    #    disabled to improve the trade-off between test coverage and build
    #    latency.
    if shapes_id == ShapesId.DEFAULT:
        return get_test_shapes(ShapesId.SMALL) + get_test_shapes(ShapesId.LARGE)
    if shapes_id == ShapesId.SMALL:
        return [
            # square matrices. Start by the simplest case of 1x1x1.
            TestShape(m=1, k=1, n=1, accumulate=True),
            TestShape(m=1, k=1, n=1, accumulate=False),
            # Test some small powers of two, that exercise in particular the
            # adjustment of data-tiling tile sizes to narrow cases.
            TestShape(m=2, k=2, n=2, accumulate=True),
            TestShape(m=4, k=4, n=4, accumulate=True),
            TestShape(m=8, k=8, n=8, accumulate=True),
            # test 9x9x9 because as many kernel M0/K0/N0 dims are equal to 8,
            # this will often be the smallest value that exercises something above
            # the kernel's size.
            TestShape(m=9, k=9, n=9, accumulate=True),
            # rectangular matrices.
            # >= 2x differences between M/N/K dims may exercise tiling corner cases
            # not exercised by nearly-square matrices.
            TestShape(m=6, k=13, n=3, accumulate=True),
            TestShape(m=15, k=37, n=7, accumulate=False),
            TestShape(m=81, k=19, n=41, accumulate=True),
            # shapes involving vectors (i.e. most rectangular cases)
            # This is particularly relevant because we have dedicated kernels for
            # the matrix*vector / vector*matrix case.
            TestShape(m=1, k=10, n=10, accumulate=True),  # vector*matrix
            TestShape(m=1, k=10, n=10, accumulate=False),  # vector*matrix
            TestShape(m=10, k=1, n=10, accumulate=True),  # outer-product
            TestShape(m=10, k=10, n=1, accumulate=True),  # matrix*vector
            TestShape(m=10, k=10, n=1, accumulate=False),  # matrix*vector
        ]
    if shapes_id == ShapesId.LARGE:
        return [
            # unaligned cases.
            TestShape(m=457, k=330, n=512, accumulate=False),
            TestShape(m=438, k=331, n=513, accumulate=False),
            TestShape(m=540, k=332, n=516, accumulate=False),
            TestShape(m=1000, k=4, n=512, accumulate=False),
            TestShape(m=4, k=1000, n=512, accumulate=False),
            TestShape(m=512, k=1000, n=4, accumulate=False),
            TestShape(m=513, k=128, n=55, accumulate=False),
            TestShape(m=7, k=160, n=31, accumulate=False),
            TestShape(m=512, k=330, n=33, accumulate=False),
            # shapes involving vectors (i.e. most rectangular cases)
            TestShape(m=1, k=1000, n=1000, accumulate=True),  # large vector*matrix
            TestShape(m=1000, k=1000, n=1, accumulate=True),  # large matrix*vector
            TestShape(m=1000, k=1000, n=1, accumulate=False),  # large matrix*vector
            # Be conservative in adding larger shapes. They can result in
            # high latency tests. If you have to, consider splitting them
            # out in a way that constrains the latency impact, e.g. by
            # running on fewer backends/drivers or with fewer generators
            # (see get_test_generators).
        ]
    if shapes_id == ShapesId.EASY_LARGE_STATIC:
        return [
            TestShape(m=512, k=128, n=512, accumulate=True),
            TestShape(m=512, k=128, n=512, accumulate=False),
        ]

    raise ValueError(shapes_id)


# Returns the list of Dynamicity's to use for the collection of shapes
# identified by shapes_id.
def get_dynamicities(shapes_id: ShapesId):
    if shapes_id == ShapesId.EASY_LARGE_STATIC:
        return [
            Dynamicity.STATIC,
        ]
    else:
        return [
            Dynamicity.DYNAMIC,
            Dynamicity.STATIC,
        ]
    raise ValueError(shapes_id)


@dataclasses.dataclass
class TileWorkgroupSizePair:
    tile_size: typing.List[typing.List[int]]
    workgroup_size: typing.List[int]


# Constructs a TileWorkgroupSizePair for SPIR-V targets enforcing the
# constraints between the workgroup_size and tile size
def get_spirv_tile_workgroup_size_pair(
    workgroup_size, t_tile_k, t_tile_m=4, t_tile_n=4
):
    x, y, z = workgroup_size
    wg_tile_m = y * t_tile_m
    wg_tile_n = x * t_tile_n
    return TileWorkgroupSizePair(
        [[wg_tile_m, wg_tile_n], [t_tile_m, t_tile_n], [0, 0, t_tile_k]], workgroup_size
    )


# Returns all the TileWorkgroupSizePairs for a given SPIRV Target
def get_all_spirv_tile_workgroup_size_pairs(t_tile_k):
    tile_workgroup_size_pairs = [
        get_spirv_tile_workgroup_size_pair([32, 8, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([16, 8, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([64, 2, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([8, 8, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([32, 1, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([16, 2, 1], t_tile_k),
        get_spirv_tile_workgroup_size_pair([32, 1, 1], t_tile_k),
    ]
    return tile_workgroup_size_pairs


def get_rocm_test_compilation_infos(
    compilation_info_id: CompilationInfoId, lhs_rhs_type: MatrixElemTypeId
):
    intrinsic = ""
    if compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeMFMA:
        intrinsic = "MFMA"
    elif compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeWMMAR3:
        intrinsic = "WMMAR3"
    else:
        raise ValueError("Unknown pipeline for rocm")

    schedules = []
    if intrinsic == "MFMA":
        schedules = [
            MMASchedule("MFMA_F32_16x16x4_F32", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_16x16x4_F32", 1, 1, 1, 1, 2),
            MMASchedule("MFMA_F32_16x16x4_F32", 1, 1, 1, 2, 1),
            MMASchedule("MFMA_F32_16x16x4_F32", 1, 1, 2, 1, 1),
            MMASchedule("MFMA_F32_16x16x4_F32", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_F32_16x16x16_F16", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_16x16x16_F16", 1, 1, 1, 1, 2),
            MMASchedule("MFMA_F32_16x16x16_F16", 1, 1, 1, 2, 1),
            MMASchedule("MFMA_F32_16x16x16_F16", 1, 1, 2, 1, 1),
            MMASchedule("MFMA_F32_16x16x16_F16", 2, 2, 1, 1, 1),
            MMASchedule("MFMA_F32_16x16x16_F16", 2, 4, 2, 1, 2),
            MMASchedule("MFMA_F32_16x16x16_F16", 4, 2, 4, 2, 2),
            MMASchedule("MFMA_F32_32x32x8_F16", 1, 1, 1, 2, 2),
            MMASchedule("MFMA_F32_32x32x8_F16", 2, 2, 1, 1, 1),
            MMASchedule("MFMA_F32_32x32x8_F16", 1, 4, 2, 1, 2),
            MMASchedule("MFMA_F32_32x32x8_F16", 4, 2, 1, 2, 4),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FNUZ", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FNUZ", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FNUZ", 4, 1, 4, 1, 1),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FNUZ", 4, 2, 4, 2, 1),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FNUZ", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FNUZ", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FNUZ", 4, 1, 1, 2, 2),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FNUZ", 4, 2, 2, 2, 2),
            MMASchedule("MFMA_I32_16x16x32_I8", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_I32_16x16x32_I8", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_I32_16x16x32_I8", 4, 1, 4, 1, 1),
            MMASchedule("MFMA_I32_16x16x32_I8", 4, 2, 4, 2, 1),
            MMASchedule("MFMA_I32_32x32x16_I8", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_I32_32x32x16_I8", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_I32_32x32x16_I8", 4, 1, 1, 2, 2),
            MMASchedule("MFMA_I32_32x32x16_I8", 4, 2, 2, 2, 2),
            MMASchedule("VMFMA_F32_16x16x32_F16", 1, 1, 1, 1, 1),
            MMASchedule("VMFMA_F32_16x16x32_F16", 4, 2, 1, 2, 4),
            MMASchedule("VMFMA_F32_32x32x16_F16", 1, 1, 1, 1, 1),
            MMASchedule("VMFMA_F32_32x32x16_F16", 4, 2, 1, 2, 4),
            MMASchedule("VMFMA_F32_16x16x32_F8E4M3FNUZ", 1, 1, 1, 1, 1),
            MMASchedule("VMFMA_F32_16x16x32_F8E4M3FNUZ", 4, 1, 4, 1, 1),
        ]
    elif intrinsic == "WMMAR3":
        schedules = [
            MMASchedule("WMMAR3_F32_16x16x16_F16", 1, 1, 1, 1, 1),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 1, 1, 1, 1, 2),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 1, 1, 1, 2, 1),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 1, 1, 2, 1, 1),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 2, 2, 1, 1, 1),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 2, 4, 2, 1, 2),
            MMASchedule("WMMAR3_F32_16x16x16_F16", 4, 2, 4, 2, 2),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 1, 1, 1, 1, 1),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 1, 1, 1, 1, 2),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 1, 1, 1, 2, 1),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 1, 1, 2, 1, 1),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 2, 2, 1, 1, 1),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 2, 4, 2, 1, 2),
            MMASchedule("WMMAR3_I32_16x16x16_I8", 4, 2, 4, 2, 2),
        ]
    else:
        raise NotImplementedError("unhandled intrinsic case")

    subgroup_size = 64 if intrinsic == "MFMA" else 32

    infos = []
    for schedule in schedules:
        # Skip schedules with an intrinsic which element type does not
        # match the requested one.
        # Extracts the input type from strings. The naming convention is
        # [output_type]_MxNxK_[input_type].
        input_type = schedule.intrinsic.split("_")[-1]
        if lhs_rhs_type.value.upper() != input_type:
            continue

        if schedule.intrinsic == "MFMA_F32_16x16x4_F32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 4
        elif schedule.intrinsic == "MFMA_F32_16x16x16_F16":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "MFMA_F32_32x32x8_F16":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 32
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 32
            wg_tile_k = schedule.k_tile_count * 8
        elif (
            schedule.intrinsic == "VMFMA_F32_16x16x32_F16"
            or schedule.intrinsic == "MFMA_I32_16x16x32_I8"
            or schedule.intrinsic == "MFMA_F32_16x16x32_F8E4M3FNUZ"
            or schedule.intrinsic == "VMFMA_F32_16x16x32_F8E4M3FNUZ"
        ):
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 32
        elif (
            schedule.intrinsic == "VMFMA_F32_32x32x16_F16"
            or schedule.intrinsic == "MFMA_F32_32x32x16_F8E4M3FNUZ"
            or schedule.intrinsic == "MFMA_I32_32x32x16_I8"
        ):
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 32
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 32
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "WMMAR3_F32_16x16x16_F16":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "WMMAR3_I32_16x16x16_I8":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        else:
            raise NotImplementedError("unhandled intrinsic case")

        workgroup_tile = [wg_tile_m, wg_tile_n, 0]
        reduction_tile = [0, 0, wg_tile_k]
        workgroup_size = [schedule.n_count * subgroup_size, schedule.m_count, 1]
        infos.append(
            IREEGPUCompilationInfo(
                workgroup_tile=workgroup_tile,
                reduction_tile=reduction_tile,
                dispatch_lowering_pass_pipeline="LLVMGPUVectorDistribute",
                workgroup_size=workgroup_size,
                mma_schedule=schedule,
                subgroup_size=subgroup_size,
            )
        )
    return infos


# Returns the list of CompilationInfo's to use for the CompilationInfoId.
def get_test_compilation_infos(
    compilation_info_id: CompilationInfoId, lhs_rhs_type: MatrixElemTypeId
) -> typing.List[typing.Optional[CompilationInfo]]:
    if compilation_info_id == CompilationInfoId.NONE:
        return [None]

    if compilation_info_id in [
        CompilationInfoId.LLVMGPUVectorDistributeMFMA,
        CompilationInfoId.LLVMGPUVectorDistributeWMMAR3,
    ]:
        return get_rocm_test_compilation_infos(compilation_info_id, lhs_rhs_type)

    software_pipeline_depth = 0
    tile_workgroup_size_pairs = []
    if compilation_info_id == CompilationInfoId.SPIRVCooperativeMatrixVectorize:
        tile_workgroup_size_pairs = [
            TileWorkgroupSizePair(
                [[64, 128], [32, 64], [0, 0, 32], [16, 16, 16]], [64, 2, 1]
            )
        ]
    elif compilation_info_id == CompilationInfoId.SPIRVVectorizeNVIDIA:
        tile_workgroup_size_pairs = get_all_spirv_tile_workgroup_size_pairs(32)
    elif compilation_info_id == CompilationInfoId.SPIRVVectorizeMali:
        tile_workgroup_size_pairs = get_all_spirv_tile_workgroup_size_pairs(4)
    elif (
        compilation_info_id == CompilationInfoId.LLVMGPUMatmulTensorCore
        or compilation_info_id == CompilationInfoId.LLVMGPUMatmulTensorCoreMmaSync
    ):
        tile_workgroup_size_pairs = []
        ## WarpShape = 2x2
        tile_workgroup_size_pairs.append(
            TileWorkgroupSizePair([[32, 32, 16]], [64, 2, 1])
        )
        tile_workgroup_size_pairs.append(
            TileWorkgroupSizePair([[64, 64, 64]], [64, 2, 1])
        )

        ## WarpShape = 4x1
        tile_workgroup_size_pairs.append(
            TileWorkgroupSizePair([[32, 32, 32]], [64, 1, 1])
        )

        ## WarpShape = 2x2 with large tiles using larger Shared Memory capacity.
        if lhs_rhs_type == MatrixElemTypeId.F16:
            tile_workgroup_size_pairs.append(
                TileWorkgroupSizePair([[128, 128, 64]], [64, 2, 1])
            )
        elif lhs_rhs_type == MatrixElemTypeId.F32:
            tile_workgroup_size_pairs.append(
                TileWorkgroupSizePair([[128, 128, 16]], [64, 2, 1])
            )
        software_pipeline_depth = 3

    compilation_infos = []
    for tile_workgroup_size_pair in tile_workgroup_size_pairs:
        compilation_infos.append(
            LegacyCompilationInfo(
                tile_sizes=tile_workgroup_size_pair.tile_size,
                dispatch_lowering_pass_pipeline=compilation_info_id.value,
                workgroup_size=tile_workgroup_size_pair.workgroup_size,
                subgroup_size=None,
                software_pipeline_depth=software_pipeline_depth,
            )
        )
    return compilation_infos


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with pseudorandom_generator_seed to limit the ways
# in which shuffling testcases changes which random values are generated.
local_pseudorandom_state = 1


# A shape dimension value, i.e. a size value that could appear in a MLIR type
# such as 'tensor<?x4xf32>'. None means a dynamic size, similar to '?' in MLIR.
@dataclasses.dataclass
class DimSize:
    value: typing.Optional[int]


# Generates a compile-time MLIR size value, i.e. either a fixed positive integer
# or None (which maps to MLIR '?') depending on dynamicity.
def shape_dim(x: int, dynamicity: Dynamicity):
    if dynamicity == Dynamicity.DYNAMIC:
        return DimSize(None)
    elif dynamicity == Dynamicity.STATIC:
        return DimSize(x)
    else:
        raise ValueError(dynamicity)


# Stringification used for generating MLIR types, e.g. tensor<?x?xf32>.
def int_or_question_mark(s: DimSize):
    return s.value or "?"


# Stringification used for generating alphanumeric identifiers, e.g.
# func.func @somefunction_DYNxDYNxf32, where we can't use "?" characters.
def int_or_DYN(s: DimSize):
    return s.value or "DYN"


# Describes the fully resolved shape dimensions of all 3 input matrices,
# LHS, RHS, and Accumulator, in a testcase.
# Each value is a string, which may either represent a positive integer such as "123",
# or a "?" string, meaning a dynamic dimension as in MLIR.
# These string values are used to generate MLIR function names and tensor shapes.
@dataclasses.dataclass
class TestInputMatricesShapes:
    lhs_rows: DimSize
    lhs_cols: DimSize
    rhs_rows: DimSize
    rhs_cols: DimSize
    acc_rows: DimSize
    acc_cols: DimSize


# Helper for generate_function. Generates TestInputMatricesShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicity to
# the set of shapes to be used in a test function's input tensors.
def generate_shapes(shape: TestShape, transpose_rhs: bool, dynamicity: Dynamicity):
    lhs_rows = shape_dim(shape.m, dynamicity)
    lhs_cols = shape_dim(shape.k, dynamicity)
    acc_rows = shape_dim(shape.m, dynamicity)
    acc_cols = shape_dim(shape.n, dynamicity)
    if transpose_rhs:
        rhs_rows = shape_dim(shape.n, dynamicity)
        rhs_cols = shape_dim(shape.k, dynamicity)
    else:
        rhs_rows = shape_dim(shape.k, dynamicity)
        rhs_cols = shape_dim(shape.n, dynamicity)
    shapes = TestInputMatricesShapes(
        lhs_rows=lhs_rows,
        lhs_cols=lhs_cols,
        rhs_rows=rhs_rows,
        rhs_cols=rhs_cols,
        acc_rows=acc_rows,
        acc_cols=acc_cols,
    )
    return shapes


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes: TestInputMatricesShapes,
    accumulate: bool,
    compilation_info: typing.Optional[CompilationInfo] = None,
):
    input_t = lhs_rhs_type.value
    acc_t = acc_type.value
    lhs_r = int_or_DYN(shapes.lhs_rows)
    lhs_c = int_or_DYN(shapes.lhs_cols)
    rhs_r = int_or_DYN(shapes.rhs_rows)
    rhs_c = int_or_DYN(shapes.rhs_cols)
    acc_r = int_or_DYN(shapes.acc_rows)
    acc_c = int_or_DYN(shapes.acc_cols)

    info = ""
    if compilation_info:
        info = f"_for_{compilation_info.dispatch_lowering_pass_pipeline}"

    matmul_kind = "matmul_accumulate" if accumulate else "matmul"
    return (
        f"{matmul_kind}_{lhs_r}x{lhs_c}x{input_t}_times_"
        + f"{rhs_r}x{rhs_c}x{input_t}_into_{acc_r}x{acc_c}x{acc_t}{info}"
    )


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
    name: str
    signature: str
    import_declaration: str
    definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul variants
# and will just call linalg.matmul variants with them, returning its result.
def generate_function(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: TestShape,
    transpose_rhs: bool,
    dynamicity: Dynamicity,
    compilation_info: Optional[CompilationInfo] = None,
):
    shapes = generate_shapes(shape, transpose_rhs, dynamicity)
    func_name = generate_function_name(
        lhs_rhs_type, acc_type, shapes, shape.accumulate, compilation_info
    )
    lhs_r = int_or_question_mark(shapes.lhs_rows)
    lhs_c = int_or_question_mark(shapes.lhs_cols)
    rhs_r = int_or_question_mark(shapes.rhs_rows)
    rhs_c = int_or_question_mark(shapes.rhs_cols)
    acc_r = int_or_question_mark(shapes.acc_rows)
    acc_c = int_or_question_mark(shapes.acc_cols)

    lhs_tensor_type = f"tensor<{lhs_r}x{lhs_c}x{lhs_rhs_type.value}>"
    rhs_tensor_type = f"tensor<{rhs_r}x{rhs_c}x{lhs_rhs_type.value}>"
    acc_tensor_type = f"tensor<{acc_r}x{acc_c}x{acc_type.value}>"

    if transpose_rhs:
        op_name = "linalg.matmul_transpose_b"
    else:
        op_name = "linalg.matmul"

    # Compilation info is optional; prints empty string by default.
    func_definition = ""
    compilation_info_attr = ""
    if compilation_info:
        compilation_info_string = f"#compilation{generate_function.compilation_index} = {compilation_info.get_compilation_info_attr()}"
        compilation_info_attr = (
            f"{{compilation_info = #compilation{generate_function.compilation_index}}} "
        )
        func_definition = func_definition + compilation_info_string
        generate_function.compilation_index += 1
    compute = f"  %result = {op_name} {compilation_info_attr}ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
    if shape.accumulate:
        signature = f"({lhs_tensor_type}, {rhs_tensor_type}, {acc_tensor_type}) -> {acc_tensor_type}"
        import_declaration = f"func.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view"
        func_definition = func_definition + (
            f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
            f"{compute}\n"
            f"  return %result: {acc_tensor_type}\n"
            f"}}\n"
        )
    else:
        literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
        if acc_r == "?":
            signature = f"({lhs_tensor_type}, {rhs_tensor_type}) -> {acc_tensor_type}"
            import_declaration = f"func.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view"
            func_definition = func_definition + (
                f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
                f"  %c0 = arith.constant 0 : index\n"
                f"  %c1 = arith.constant 1 : index\n"
                f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
                f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
                f"  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : {acc_tensor_type}\n"
                f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
                f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
                f"{compute}"
                f"  return %result: {acc_tensor_type}\n"
                f"}}\n"
            )
        else:
            signature = f"({lhs_tensor_type}, {rhs_tensor_type}) -> {acc_tensor_type}"
            import_declaration = f"func.func private @module.{func_name}(%lhs: !hal.buffer_view, %rhs: !hal.buffer_view) -> !hal.buffer_view"
            func_definition = func_definition + (
                f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
                f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"
                f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
                f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
                f"{compute}"
                f"  return %result: {acc_tensor_type}\n"
                f"}}\n"
            )
    return MLIRFunction(
        name=func_name,
        signature=signature,
        import_declaration=import_declaration,
        definition=func_definition,
    )


# Counter for producing unique compilation info attrs
generate_function.compilation_index = 0


# Represents a call to a generated test function.
@dataclasses.dataclass
class TestCall:
    function: MLIRFunction
    op: str


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with local_pseudorandom_state to limit the ways
# in which shuffling testcases changes which random values are generated.
pseudorandom_generator_seed = 1


def contents_generator_tag(generator: MatrixGenerator):
    if generator == MatrixGenerator.ZERO:
        return ""
    elif generator == MatrixGenerator.RANDOM:
        global pseudorandom_generator_seed
        pseudorandom_generator_seed = pseudorandom_generator_seed + 1
        return f"!tag:iree:fully_specified_pseudorandom {pseudorandom_generator_seed}"
    else:
        raise ValueError(generator)


# Generate a matrix function argument of the given size as `%name`.
def generate_random_matrix(
    name: str,
    matrix_shape: list,
    element_type: MatrixElemTypeId,
):
    global pseudorandom_generator_seed
    pseudorandom_generator_seed = pseudorandom_generator_seed + 1
    return (
        f"  %{name}_dim0 = arith.constant {matrix_shape[0]} : i64\n"
        f"  %{name}_dim1 = arith.constant {matrix_shape[1]} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {pseudorandom_generator_seed} : i32\n"
        f"  %{name} = call @matmul_test.generate_random_matrix(%device, %{name}_dim0, %{name}_dim1, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )


call_id = 0


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_call(
    function: MLIRFunction,
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: TestShape,
    transpose_rhs: bool,
):
    global call_id
    func_name = f"{function.name}_{shape.m}_{shape.k}_{shape.n}"
    if shape.accumulate:
        func_name = f"{func_name}_acc"
    func_name = f"{func_name}_{call_id}"
    call_id = call_id + 1

    description = f"Matmul shape (MxKxN): {shape.m}x{shape.k}x{shape.n}"
    op = (
        f"func.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    lhs_shape = [shape.m, shape.k]
    if transpose_rhs:
        rhs_shape = [shape.n, shape.k]
        transpose_rhs = 1
    else:
        rhs_shape = [shape.k, shape.n]
        transpose_rhs = 0

    op = op + generate_random_matrix("lhs", lhs_shape, lhs_rhs_type)
    op = op + generate_random_matrix("rhs", rhs_shape, lhs_rhs_type)
    if shape.accumulate:
        op = op + generate_random_matrix("acc", [shape.m, shape.n], acc_type)
        # TODO(#16168): there's a bug with in-place input->output aliasing and
        # we work around it here by passing in a unique copy.
        global pseudorandom_generator_seed
        pseudorandom_generator_seed = pseudorandom_generator_seed - 1
        op = op + generate_random_matrix("acc_copy", [shape.m, shape.n], acc_type)
        op = op + (
            f"  %result = call @module.{function.name}(%lhs, %rhs, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )
    else:
        op = op + (
            f"  %acc = util.null : !hal.buffer_view\n"
            f"  %result = call @module.{function.name}(%lhs, %rhs) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )

    op = op + (
        f"  %m = arith.constant {shape.m} : i64\n"
        f"  %k = arith.constant {shape.k} : i64\n"
        f"  %n = arith.constant {shape.n} : i64\n"
        f"  %transpose_rhs = arith.constant {transpose_rhs} : i32\n"
        f"  call @matmul_test.check_matmul_results(%device, %m, %k, %n, %transpose_rhs, %lhs, %rhs, %acc, %result) : (!hal.device, i64, i64, i64, i32, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
    )

    op = op + "  return\n"
    op = op + "}\n"

    return TestCall(function=function, op=op)


# Generates all output files' contents as strings.
def generate(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes_id: ShapesId,
    transpose_rhs: bool,
    compilation_info_id: CompilationInfoId,
):
    functions = {}
    calls = []

    for compilation_info in get_test_compilation_infos(
        compilation_info_id, lhs_rhs_type
    ):
        for shape in get_test_shapes(shapes_id):
            for dynamicity in get_dynamicities(shapes_id):
                function = generate_function(
                    lhs_rhs_type,
                    acc_type,
                    shape,
                    transpose_rhs,
                    dynamicity,
                    compilation_info,
                )
                # Different testcases may differ only by runtime parameters but
                # share the same code. For example, dynamic-shapes testcases
                # share the same code involing tensor<?x?xf32> even though the runtime
                # value in the trace are different. That's why we append conditionally
                # to calls, but unconditionally to function_definitions.
                if function.name not in functions:
                    functions[function.name] = function
                calls.append(
                    generate_call(
                        function, lhs_rhs_type, acc_type, shape, transpose_rhs
                    )
                )

    return (functions, calls)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generator of e2e matmul tests")
    parser.add_argument(
        "--output_matmul_mlir",
        type=str,
        help="Path of output .mlir file containing the generated matmuls",
        required=True,
    )
    parser.add_argument(
        "--output_calls_mlir",
        type=str,
        help="Path of output .mlir file containing the calls",
        required=True,
    )
    parser.add_argument(
        "--lhs_rhs_type",
        type=str,
        choices=[
            "i32",
            "i8",
            "f64",
            "f32",
            "f16",
            "bf16",
            "f8E5M2",
            "f8E4M3",
            "f8E5M2FNUZ",
            "f8E4M3FNUZ",
        ],
        help="Numeric type of input LHS and RHS matrices",
        required=True,
    )
    parser.add_argument(
        "--acc_type",
        type=str,
        choices=["i32", "f64", "f32", "f16", "bf16"],
        help="Numeric type of the accumulator and result matrices",
        required=True,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        choices=[s.value for s in ShapesId],
        help="Collection of matrix shapes to test",
        default="default",
        required=False,
    )
    parser.add_argument(
        "--transpose_rhs",
        action="store_true",
        help="Whether to transpose RHS",
        default=False,
        required=False,
    )
    parser.add_argument(
        "--compilation_info",
        type=str,
        choices=[i.value for i in CompilationInfoId],
        help="Collection of compilation info setups to test",
        default="",
        required=False,
    )
    parser.add_argument(
        "--requirements",
        type=str,
        help="Target requirements for this module. Comma-separated. As in -iree-llvmcpu-target-cpu-features. If the target device does not meet all of the requirements, the test will be skipped.",
        required=False,
    )
    return parser.parse_args()


def write_code_file(functions, filename):
    with open(filename, "w") as file:
        for function in functions.values():
            file.write(function.definition + "\n")


def write_calls_file(functions, calls, filename, requirements):
    # Module-level reflection information used to control the test tool.
    reflection = ""
    if requirements:
        reflection = (
            "iree.reflection = {"
            'target_features = "'
            + ",".join([req.lstrip("+") for req in requirements.split(",")])
            + '"'
            "}"
        )
    module_definition = (
        f"builtin.module @calls attributes {{\n" f"  {reflection}\n" f"}} {{\n\n"
    )

    # Declare the custom module that generates arguments.
    module_definition = module_definition + (
        "func.func private @matmul_test.generate_random_matrix(%device: !hal.device, %dim0: i64, %dim1: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view\n"
        "func.func private @matmul_test.check_matmul_results(%device: !hal.device, %m: i64, %k: i64, %n: i64, %transpose_rhs: i32, %lhs: !hal.buffer_view, %rhs: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)\n"
        "\n"
    )

    # Declare the functions that will be called.
    for function in functions.values():
        module_definition = module_definition + function.import_declaration + "\n"
    module_definition = module_definition + "\n"

    # Emit the test cases for each call.
    for call in calls:
        module_definition = module_definition + call.op + "\n"

    module_definition = module_definition + "\n}\n"

    with open(filename, "w") as file:
        file.write(module_definition)


def main(args):
    lhs_rhs_type = MatrixElemTypeId(args.lhs_rhs_type)
    acc_type = MatrixElemTypeId(args.acc_type)
    shapes_id = ShapesId(args.shapes)
    compilation_info_id = CompilationInfoId(args.compilation_info)

    (functions, calls) = generate(
        lhs_rhs_type, acc_type, shapes_id, args.transpose_rhs, compilation_info_id
    )

    write_code_file(functions, args.output_matmul_mlir)
    write_calls_file(
        functions,
        calls,
        args.output_calls_mlir,
        args.requirements,
    )


if __name__ == "__main__":
    main(parse_arguments())
