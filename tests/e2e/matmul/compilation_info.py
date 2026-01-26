#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""compilation_info support in e2e matmul tests"""

from typing import Optional
import enum
import dataclasses
import typing

from tests.e2e.matmul.common import *


# Describes a workgroup and tiling schedule to target a specific MMA intrinsic.
@dataclasses.dataclass
class MMASchedule:
    intrinsic: str
    m_count: int  # Number of subgroups per workgroup along M
    n_count: int  # Number of subgroups per workgroup along N
    m_tile_count: int
    n_tile_count: int
    k_tile_count: int

    def get_subgroup_basis(self) -> str:
        return f"[[{self.m_count}, {self.n_count}, 1], [0, 1, 2]]"


# Enumerates of the collections of compilation info that we can generate tests
# for. The values are the accepted values for the --compilation_info= flag.
@enum.unique
class CompilationInfoId(enum.Enum):
    NONE = ""
    LLVMGPUMatmulTensorCore = "LLVMGPUMatmulTensorCore"
    LLVMGPUMatmulTensorCoreMmaSync = "LLVMGPUMatmulTensorCoreMmaSync"
    LLVMGPUVectorDistributeMFMA = "LLVMGPUVectorDistributeMFMA"
    LLVMGPUVectorDistributeWMMAR3 = "LLVMGPUVectorDistributeWMMAR3"
    LLVMGPUVectorDistributeWMMAR4 = "LLVMGPUVectorDistributeWMMAR4"
    LLVMGPUVectorDistributeWMMA1250 = "LLVMGPUVectorDistributeWMMA1250"
    SPIRVCooperativeMatrixVectorize = "SPIRVCooperativeMatrixVectorize"
    SPIRVVectorizeMali = "SPIRVVectorizeMali"
    SPIRVVectorizeNVIDIA = "SPIRVVectorizeNVIDIA"


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
            f"  subgroup_basis = {self.mma_schedule.get_subgroup_basis()}, "
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


def generate_compilation_info_string_and_attr(compilation_info: CompilationInfo):
    if not compilation_info:
        return ("", "")

    index = generate_compilation_info_string_and_attr.compilation_index
    generate_compilation_info_string_and_attr.compilation_index += 1
    return (
        f"#compilation{index} = {compilation_info.get_compilation_info_attr()}",
        f"{{compilation_info = #compilation{index}}} ",
    )


generate_compilation_info_string_and_attr.compilation_index = 0


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
    elif compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeWMMAR4:
        intrinsic = "WMMAR4"
    elif compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeWMMA1250:
        intrinsic = "WMMA1250"
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
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FN", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FN", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FN", 4, 1, 4, 1, 1),
            MMASchedule("MFMA_F32_16x16x32_F8E4M3FN", 4, 2, 4, 2, 1),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FN", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FN", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FN", 4, 1, 1, 2, 2),
            MMASchedule("MFMA_F32_32x32x16_F8E4M3FN", 4, 2, 2, 2, 2),
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
    elif intrinsic == "WMMAR4":
        schedules = [
            MMASchedule("WMMAR4_F32_16x16x16_F16", 1, 1, 1, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 1, 1, 1, 1, 2),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 1, 1, 1, 2, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 1, 1, 2, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 2, 2, 1, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 2, 4, 2, 1, 2),
            MMASchedule("WMMAR4_F32_16x16x16_F16", 4, 2, 4, 2, 2),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 1, 1, 1, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 1, 1, 1, 1, 2),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 1, 1, 1, 2, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 1, 1, 2, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 2, 2, 1, 1, 1),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 2, 4, 2, 1, 2),
            MMASchedule("WMMAR4_F32_16x16x16_F8E4M3FN", 4, 2, 4, 2, 2),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 1, 1, 1, 1, 1),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 1, 1, 1, 1, 2),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 1, 1, 1, 2, 1),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 1, 1, 2, 1, 1),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 2, 2, 1, 1, 1),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 2, 4, 2, 1, 2),
            MMASchedule("WMMAR4_I32_16x16x16_I8", 4, 2, 4, 2, 2),
        ]
    elif intrinsic == "WMMA1250":
        # gfx1250 WMMA intrinsics: 16x16 tiles with various K sizes.
        # F16: K=32, F8E4M3FN: K=64 or K=128, I8: K=64
        schedules = [
            MMASchedule("WMMA_F32_16x16x32_F16", 1, 1, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x32_F16", 1, 1, 1, 1, 2),
            MMASchedule("WMMA_F32_16x16x32_F16", 1, 1, 1, 2, 1),
            MMASchedule("WMMA_F32_16x16x32_F16", 1, 1, 2, 1, 1),
            MMASchedule("WMMA_F32_16x16x32_F16", 2, 2, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x32_F16", 2, 4, 2, 1, 2),
            MMASchedule("WMMA_F32_16x16x32_F16", 4, 2, 4, 2, 2),
            # K=64 F8E4M3FN
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 1, 1, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 1, 1, 1, 1, 2),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 1, 1, 1, 2, 1),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 1, 1, 2, 1, 1),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 2, 2, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 2, 4, 2, 1, 2),
            MMASchedule("WMMA_F32_16x16x64_F8E4M3FN", 4, 2, 4, 2, 2),
            # K=128 F8E4M3FN
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 1, 1, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 1, 1, 1, 1, 2),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 1, 1, 1, 2, 1),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 1, 1, 2, 1, 1),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 2, 2, 1, 1, 1),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 2, 4, 2, 1, 2),
            MMASchedule("WMMA_F32_16x16x128_F8E4M3FN", 4, 2, 4, 2, 2),
            # I8
            MMASchedule("WMMA_I32_16x16x64_I8", 1, 1, 1, 1, 1),
            MMASchedule("WMMA_I32_16x16x64_I8", 1, 1, 1, 1, 2),
            MMASchedule("WMMA_I32_16x16x64_I8", 1, 1, 1, 2, 1),
            MMASchedule("WMMA_I32_16x16x64_I8", 1, 1, 2, 1, 1),
            MMASchedule("WMMA_I32_16x16x64_I8", 2, 2, 1, 1, 1),
            MMASchedule("WMMA_I32_16x16x64_I8", 2, 4, 2, 1, 2),
            MMASchedule("WMMA_I32_16x16x64_I8", 4, 2, 4, 2, 2),
        ]
    else:
        raise NotImplementedError("unhandled intrinsic case")

    subgroup_size = (
        64 if intrinsic == "MFMA" else 32
    )  # WMMAR3, WMMAR4, WMMA1250 all use 32.

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
            or schedule.intrinsic == "MFMA_F32_16x16x32_F8E4M3FN"
            or schedule.intrinsic == "VMFMA_F32_16x16x32_F8E4M3FN"
        ):
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 32
        elif (
            schedule.intrinsic == "VMFMA_F32_32x32x16_F16"
            or schedule.intrinsic == "MFMA_F32_32x32x16_F8E4M3FNUZ"
            or schedule.intrinsic == "MFMA_F32_32x32x16_F8E4M3FN"
            or schedule.intrinsic == "MFMA_I32_32x32x16_I8"
        ):
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 32
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 32
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic in (
            "WMMAR3_F32_16x16x16_F16",
            "WMMAR3_I32_16x16x16_I8",
            "WMMAR4_F32_16x16x16_F16",
            "WMMAR4_F32_16x16x16_F8E4M3FN",
            "WMMAR4_I32_16x16x16_I8",
        ):
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "WMMA_F32_16x16x32_F16":
            # gfx1250: M=16, N=16, K=32
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 32
        elif schedule.intrinsic in (
            "WMMA_F32_16x16x64_F8E4M3FN",
            "WMMA_I32_16x16x64_I8",
        ):
            # gfx1250: M=16, N=16, K=64
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 64
        elif schedule.intrinsic == "WMMA_F32_16x16x128_F8E4M3FN":
            # gfx1250: M=16, N=16, K=128
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 128
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
        CompilationInfoId.LLVMGPUVectorDistributeWMMAR4,
        CompilationInfoId.LLVMGPUVectorDistributeWMMA1250,
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
