#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator for e2e conv2d tests.
"""

from typing import Optional
import argparse
import enum
import dataclasses
import typing
import math
import itertools
import re


# Data type of kernel entries. The string values must match MLIR data types.
@enum.unique
class KernelElemTypeId(enum.Enum):
    NONE = ""
    I8 = "i8"
    F32 = "f32"
    F16 = "f16"


# Data type of input entries. The string values must match MLIR data types.
@enum.unique
class InputElemTypeId(enum.Enum):
    NONE = ""
    I8 = "i8"
    F32 = "f32"
    F16 = "f16"


# Data type of input entries. The string values must match MLIR data types.
@enum.unique
class AccElemTypeId(enum.Enum):
    NONE = ""
    I32 = "i32"
    F32 = "f32"
    F16 = "f16"


# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    GPU_LARGE = "gpu_large"


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


# Enumerates of the collections of compilation info that we can generate tests
# for. The values are the accepted values for the --compilation_info= flag.
@enum.unique
class CompilationInfoId(enum.Enum):
    NONE = ""
    LLVMGPUVectorDistributeMFMA = "LLVMGPUVectorDistributeMFMA"
    LLVMGPUVectorDistributeWMMA = "LLVMGPUVectorDistributeWMMA"
    LLVMGPUVectorizeCDNA = "LLVMGPUVectorizeCDNA"
    LLVMGPUVectorizeRDNA = "LLVMGPUVectorizeRDNA"


# Describes how to construct compilation info for the testcase.
@dataclasses.dataclass
class CompilationInfo:
    # Lowering Config
    tile_sizes: typing.List[typing.List[int]]
    # Translation Info
    dispatch_lowering_pass_pipeline: str
    software_pipeline_depth: int
    mma_schedule: typing.Optional[MMASchedule]
    # Compilation info
    workgroup_size: typing.List[int]
    subgroup_size: Optional[int] = None

    # Prints the workgroup size
    def workgroup_size_str(self):
        return "workgroup_size = [" + ", ".join(map(str, self.workgroup_size)) + "]"


# Enumerates ways to construct MLIR tensor types.
# TODO: Enable dynamic dimensions once the tests start passing.
@enum.unique
class Dynamicity(enum.Enum):
    DYNAMIC = "dynamic"  # Use '?' everywhere. Example: tensor<?x?xf32>.
    STATIC = "static"  # Use fixed values everywhere. Example: tensor<4x6xf32>.
    MIXED = "mixed"  # Randomly mix '?' and values. Example: tensor<?x4xf32>.


# Enumerates ways to initialize input buffer contents.
@enum.unique
class InputGenerator(enum.Enum):
    ZERO = "zero"  # Fill with zeros
    RANDOM = "random"  # Fill with (deterministic) pseudorandom values.


# Enumerates ways to initialize kernel buffer contents.
@enum.unique
class KernelGenerator(enum.Enum):
    ZERO = "zero"  # Fill with zeros
    RANDOM = "random"  # Fill with (deterministic) pseudorandom values.


# TODO: Add more input layouts as needed. The layout determines the dim of input and kernel.
@enum.unique
class InputLayout(enum.Enum):
    NCHW = "nchw"
    NHWC = "nhwc"


# TODO: Add more kernel layouts as needed.
@enum.unique
class KernelLayout(enum.Enum):
    FCHW = "fchw"
    HWCF = "hwcf"


# Describes the shape of a tensor conv2d in the usual convention:
# the input is {n}x{c}x{h}x{w}, the kernel is {f}x{c}x{kh}x{kw}, the accumulator/result is
# {n}x{f}x{oh}x{ow}.
# The extra `accumulate` boolean tells whether the conv2d is accumulating into
# an existing accumulator (C += A * B) or just overwriting the result
# (C = A * B).
@dataclasses.dataclass
class TestShape:
    n: int
    c: int
    h: int
    w: int
    kh: int
    kw: int
    f: int
    accumulate: bool


# Attributes for the linalg.conv2d operation.
@dataclasses.dataclass
class ConvAttrs:
    STRIDE: typing.Tuple[int, int] = (1, 1)
    DILATION: typing.Tuple[int, int] = (1, 1)


# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id.
def get_test_shapes(shapes_id: ShapesId):
    # Notes:
    # 1. Be conservative in adding more shapes, as that can increase both the
    #    build and execution latency of tests. The build latency is nearly the
    #    same for all shapes, while execution latency grows linearly with
    #    n*f*ow*oh*kh*kw.

    if shapes_id == ShapesId.SMALL:
        return [
            TestShape(n=1, c=1, h=1, w=1, kh=1, kw=1, f=1, accumulate=True),
            TestShape(n=1, c=1, h=16, w=16, kh=2, kw=2, f=1, accumulate=True),
            TestShape(n=2, c=2, h=32, w=32, kh=3, kw=3, f=2, accumulate=True),
        ]
    if shapes_id == ShapesId.MEDIUM:
        return [
            TestShape(n=2, h=32, w=32, c=32, kh=3, kw=3, f=64, accumulate=True),
        ]
    if shapes_id == ShapesId.LARGE:
        return [
            TestShape(n=2, c=4, h=128, w=128, kh=3, kw=3, f=8, accumulate=True),
            TestShape(n=2, c=3, h=128, w=128, kh=3, kw=3, f=12, accumulate=True),
        ]
    if shapes_id == ShapesId.GPU_LARGE:
        return [TestShape(n=1, c=64, h=130, w=130, kh=3, kw=3, f=64, accumulate=True)]

    raise ValueError(shapes_id)


# Returns the list of Dynamicity's to use for the collection of shapes
# identified by shapes_id.
def get_dynamicities(shapes_id: ShapesId):
    if shapes_id == ShapesId.LARGE or shapes_id == ShapesId.GPU_LARGE:
        return [
            Dynamicity.STATIC,
        ]
    # TODO: Enable dynamic dimensions once the tests start passing.
    else:
        return [
            Dynamicity.STATIC,
        ]
    raise ValueError(shapes_id)


@dataclasses.dataclass
class TileWorkgroupSizePair:
    tile_size: typing.List[typing.List[int]]
    workgroup_size: typing.List[int]


def get_rocm_test_compilation_infos(
    compilation_info_id: CompilationInfoId,
    lhs_rhs_type: InputElemTypeId,
    acc_type: AccElemTypeId,
):
    intrinsic = ""
    if compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeMFMA:
        intrinsic = "MFMA"
    elif compilation_info_id == CompilationInfoId.LLVMGPUVectorDistributeWMMA:
        intrinsic = "WMMA"
    else:
        raise ValueError("Unknown pipeline for rocm")

    schedules = []
    if intrinsic == "MFMA":
        schedules = [
            MMASchedule("MFMA_F16_16x16x16_F32", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_F16_16x16x16_F32", 1, 1, 1, 1, 2),
            MMASchedule("MFMA_F16_16x16x16_F32", 1, 1, 1, 2, 1),
            MMASchedule("MFMA_F16_16x16x16_F32", 1, 1, 2, 1, 1),
            MMASchedule("MFMA_F16_16x16x16_F32", 2, 2, 1, 1, 1),
            MMASchedule("MFMA_F16_16x16x16_F32", 2, 4, 2, 1, 2),
            MMASchedule("MFMA_F16_32x32x8_F32", 1, 1, 1, 2, 2),
            MMASchedule("MFMA_F16_32x32x8_F32", 2, 2, 1, 1, 1),
            MMASchedule("MFMA_I8_16x16x32_I32", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_I8_16x16x32_I32", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_I8_16x16x32_I32", 4, 1, 4, 1, 1),
            MMASchedule("MFMA_I8_16x16x32_I32", 4, 2, 4, 2, 1),
            MMASchedule("MFMA_I8_32x32x16_I32", 1, 1, 1, 1, 1),
            MMASchedule("MFMA_I8_32x32x16_I32", 2, 2, 1, 1, 2),
            MMASchedule("MFMA_I8_32x32x16_I32", 4, 1, 1, 2, 2),
        ]
    elif intrinsic == "WMMA":
        schedules = [
            MMASchedule("WMMA_F16_16x16x16_F32", 1, 1, 1, 1, 1),
            MMASchedule("WMMA_F16_16x16x16_F32", 1, 1, 1, 1, 2),
            MMASchedule("WMMA_F16_16x16x16_F32", 1, 1, 1, 2, 1),
            MMASchedule("WMMA_F16_16x16x16_F32", 1, 1, 2, 1, 1),
            MMASchedule("WMMA_F16_16x16x16_F32", 2, 2, 1, 1, 1),
            MMASchedule("WMMA_F16_16x16x16_F32", 2, 4, 2, 1, 2),
            MMASchedule("WMMA_F16_16x16x16_F32", 4, 2, 4, 2, 2),
        ]
    else:
        raise NotImplementedError("unhandled intrinsic case")

    subgroup_size = 64 if intrinsic == "MFMA" else 32

    infos = []
    for schedule in schedules:
        # Skip schedules with an intrinsic which element type does not
        # match the requested one.

        # Extract the input and acc type from intrinsic strings
        extract_input_type = lambda s: re.search(r"(?:MFMA|WMMA)_([^_]+)_", s).group(1)
        extract_output_type = lambda s: (
            match := re.search(r"_(F\d+|I\d+)$", s)
        ) and match.group(1)

        if lhs_rhs_type.value.upper() != extract_input_type(
            schedule.intrinsic
        ) or acc_type.value.upper() != extract_output_type(schedule.intrinsic):
            continue

        if schedule.intrinsic == "MFMA_F16_16x16x16_F32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "MFMA_F16_32x32x8_F32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 32
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 32
            wg_tile_k = schedule.k_tile_count * 8
        elif schedule.intrinsic == "MFMA_I8_16x16x32_I32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 32
        elif schedule.intrinsic == "MFMA_I8_32x32x16_I32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 32
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 32
            wg_tile_k = schedule.k_tile_count * 16
        elif schedule.intrinsic == "WMMA_F16_16x16x16_F32":
            wg_tile_m = schedule.m_count * schedule.m_tile_count * 16
            wg_tile_n = schedule.n_count * schedule.n_tile_count * 16
            wg_tile_k = schedule.k_tile_count * 16
        else:
            raise NotImplementedError("unhandled intrinsic case")

        workgroup_tile = [[1, 1, wg_tile_m, wg_tile_n, 1, 1, wg_tile_k]]
        workgroup_size = [schedule.n_count * subgroup_size, schedule.m_count, 1]

        infos.append(
            CompilationInfo(
                tile_sizes=workgroup_tile,
                dispatch_lowering_pass_pipeline="LLVMGPUVectorDistribute",
                workgroup_size=workgroup_size,
                software_pipeline_depth=0,
                mma_schedule=schedule,
                subgroup_size=subgroup_size,
            )
        )
    return infos


# Returns the list of CompilationInfo's to use for the CompilationInfoId.
def get_test_compilation_infos(
    compilation_info_id: CompilationInfoId,
    lhs_rhs_type: InputElemTypeId,
    acc_type: AccElemTypeId,
) -> typing.List[typing.Optional[CompilationInfo]]:
    if compilation_info_id == CompilationInfoId.NONE:
        return [None]

    if compilation_info_id in [
        CompilationInfoId.LLVMGPUVectorDistributeMFMA,
        CompilationInfoId.LLVMGPUVectorDistributeWMMA,
    ]:
        return get_rocm_test_compilation_infos(
            compilation_info_id, lhs_rhs_type, acc_type
        )

    subgroup_size = -1
    if compilation_info_id == CompilationInfoId.LLVMGPUVectorizeCDNA:
        subgroup_size = 64
    elif compilation_info_id == CompilationInfoId.LLVMGPUVectorizeRDNA:
        subgroup_size = 32

    if subgroup_size == 64 or subgroup_size == 32:
        workgroup_tile = [[1, 1, 8, 128, 1, 1, 4]]
        workgroup_size = [64, 1, 1]
        infos = []
        infos.append(
            CompilationInfo(
                tile_sizes=workgroup_tile,
                dispatch_lowering_pass_pipeline="LLVMGPUVectorize",
                workgroup_size=workgroup_size,
                software_pipeline_depth=0,
                mma_schedule=None,
                subgroup_size=subgroup_size,
            )
        )
        return infos

    raise ValueError("Unknown pipeline for rocm")


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


# Determines the shape of input and kernel tensors.
@dataclasses.dataclass
class TestInputTensorShapes:
    n: DimSize
    c: DimSize
    h: DimSize
    w: DimSize
    kh: DimSize
    kw: DimSize
    f: DimSize


# Helper for generate_function. Generates TestInputTensorShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicity to
# the set of shapes to be used in a test function's input tensors.
def generate_shapes(shape: TestShape, dynamicity: Dynamicity):
    n = shape_dim(shape.n, dynamicity)
    c = shape_dim(shape.c, dynamicity)
    h = shape_dim(shape.h, dynamicity)
    w = shape_dim(shape.w, dynamicity)
    kh = shape_dim(shape.kh, dynamicity)
    kw = shape_dim(shape.kw, dynamicity)
    f = shape_dim(shape.f, dynamicity)
    shapes = TestInputTensorShapes(
        n=n,
        c=c,
        h=h,
        w=w,
        kh=kh,
        kw=kw,
        f=f,
    )
    return shapes


# Helper to calculate the output shape based on the input shape, kernel shape,
# dilation and stride.
def calc_out_shape(i_shape: int, k_shape: int, dilation_val: int, stride_val: int):
    x = (k_shape - 1) * (dilation_val - 1)
    x = i_shape - k_shape - x
    return math.floor(x / stride_val) + 1


# Helper to return input, kernel and output shapes based on the layout and Conv2dParams.
def get_tensor_shape(
    shapes: TestShape,
    kernel_layout: KernelLayout,
    input_layout: InputLayout,
    conv_attr: ConvAttrs,
):
    n = shapes.n
    c = shapes.c
    h = shapes.h
    w = shapes.w
    kh = shapes.kh
    kw = shapes.kw
    f = shapes.f

    # Extract input dimensions
    input_height, input_width = h, w

    # Extract kernel dimensions
    kernel_height, kernel_width = kh, kw

    # Get the dilation and stride
    dilation = conv_attr.DILATION
    stride = conv_attr.STRIDE

    # Calculate output height.
    oh = calc_out_shape(input_height, kernel_height, dilation[0], stride[0])
    # Calculate output width.
    ow = calc_out_shape(input_width, kernel_width, dilation[1], stride[1])

    input_tensor_shape, kernel_tensor_shape, output_tensor_shape = [], [], []

    if input_layout == InputLayout.NCHW:
        input_tensor_shape = [n, c, h, w]
        output_tensor_shape = [n, f, oh, ow]
    elif input_layout == InputLayout.NHWC:
        input_tensor_shape = [n, h, w, c]
        output_tensor_shape = [n, oh, ow, f]
    else:
        raise ValueError(input_layout)

    if kernel_layout == KernelLayout.FCHW:
        kernel_tensor_shape = [f, c, kh, kw]
    elif kernel_layout == KernelLayout.HWCF:
        kernel_tensor_shape = [kh, kw, c, f]
    else:
        raise ValueError(kernel_layout)

    return input_tensor_shape, kernel_tensor_shape, output_tensor_shape


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(
    input_type: InputElemTypeId,
    kernel_type: KernelElemTypeId,
    output_type: AccElemTypeId,
    shapes: TestInputTensorShapes,
    accumulate: bool,
    compilation_info: typing.Optional[CompilationInfo] = None,
):
    input_t = input_type.value
    kernel_t = kernel_type.value
    acc_t = output_type.value
    n = int_or_DYN(shapes.n)
    c = int_or_DYN(shapes.c)
    h = int_or_DYN(shapes.h)
    w = int_or_DYN(shapes.w)
    kh = int_or_DYN(shapes.kh)
    kw = int_or_DYN(shapes.kw)
    f = int_or_DYN(shapes.f)

    info = ""
    if compilation_info:
        tile_sizes = list(itertools.chain(*compilation_info.tile_sizes))
        tile_workgroup_key = (
            "_".join([str(a) for a in tile_sizes])
            + "_"
            + "_".join([str(a) for a in compilation_info.workgroup_size])
        )
        info = f"_for_{compilation_info.dispatch_lowering_pass_pipeline}_{tile_workgroup_key}"

    conv2d_kind = "conv2d_accumulate" if accumulate else "conv2d"
    return (
        f"{conv2d_kind}_{n}_{c}_{h}_{w}_times_"
        + f"{kh}_{kw}_{f}_dtype_{input_t}_{kernel_t}_{acc_t}{info}"
    )


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
    name: str
    signature: str
    import_declaration: str
    definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.conv2d variants
# and will just call linalg.conv2d variants with them, returning its result.
def generate_function(
    input_type: InputElemTypeId,
    input_layout: InputLayout,
    kernel_type: KernelElemTypeId,
    kernel_layout: KernelLayout,
    acc_type: AccElemTypeId,
    conv2d_attr: ConvAttrs,
    shape: TestShape,
    dynamicity: Dynamicity,
    compilation_info: typing.Optional[CompilationInfo] = None,
):
    shapes = generate_shapes(shape, dynamicity)
    func_name = generate_function_name(
        input_type, kernel_type, acc_type, shapes, shape.accumulate, compilation_info
    )

    input_shape, kernel_shape, output_shape = get_tensor_shape(
        shape, kernel_layout, input_layout, conv2d_attr
    )
    input_tensor_type = f"tensor<{input_shape[0]}x{input_shape[1]}x{input_shape[2]}x{input_shape[3]}x{input_type.value}>"
    kernel_tensor_type = f"tensor<{kernel_shape[0]}x{kernel_shape[1]}x{kernel_shape[2]}x{kernel_shape[3]}x{kernel_type.value}>"

    acc_tensor_type = f"tensor<{output_shape[0]}x{output_shape[1]}x{output_shape[2]}x{output_shape[3]}x{acc_type.value}>"

    op_name = None
    if input_layout == InputLayout.NCHW:
        if kernel_layout == KernelLayout.FCHW:
            op_name = "linalg.conv_2d_nchw_fchw"
        if kernel_layout == KernelLayout.HWCF:
            op_name = "linalg.conv_2d_nchw_hwcf"
    elif input_layout == InputLayout.NHWC:
        if kernel_layout == KernelLayout.HWCF:
            op_name = "linalg.conv_2d_nhwc_hwcf"

    if op_name is None:
        raise ValueError("Invalid combination of input_layout and kernel_layout")

    conv_attr = f"{{dilations = dense<{list(conv2d_attr.DILATION)}> : tensor<2xi64>, strides = dense<{list(conv2d_attr.STRIDE)}> : tensor<2xi64>}}"

    # Compilation info is optional; prints empty string by default.
    func_definition = ""
    # compilation_info_attr = ""
    if compilation_info:
        requested_pipeline = compilation_info.dispatch_lowering_pass_pipeline
        compiler_pipeline = requested_pipeline

        mma_schedule = ""
        if compilation_info.mma_schedule is not None:
            mma_schedule = ", {}".format(compilation_info.mma_schedule)
        subgroup_size_str = ""
        if compilation_info.subgroup_size is not None:
            subgroup_size_str = f"subgroup_size = {compilation_info.subgroup_size}"

        compilation_info_string = (
            f"#compilation{generate_function.compilation_index} = "
            "#iree_codegen.compilation_info<\n"
            f"  lowering_config = #iree_codegen.lowering_config<tile_sizes = {compilation_info.tile_sizes}>,\n"
            f"  translation_info = <{compiler_pipeline} {compilation_info.workgroup_size_str()}\n"
            f"  {subgroup_size_str},\n"
            f"  {{ pipeline_depth = {compilation_info.software_pipeline_depth}, "
            f"  store_stage = 1{mma_schedule} }}>>\n"
        )

        func_definition = func_definition + compilation_info_string
        conv_attr = f"{{compilation_info = #compilation{generate_function.compilation_index}, dilations = dense<{list(conv2d_attr.DILATION)}> : tensor<2xi64>, strides = dense<{list(conv2d_attr.STRIDE)}> : tensor<2xi64>}}"
        generate_function.compilation_index += 1

    signature = f"({input_tensor_type}, {kernel_tensor_type}, {acc_tensor_type}) -> {acc_tensor_type}"
    import_declaration = f"func.func private @module.{func_name}(%input: !hal.buffer_view, %kernel: !hal.buffer_view, %acc: !hal.buffer_view) -> !hal.buffer_view"
    func_definition = func_definition + (
        f"func.func @{func_name}(%lhs: {input_tensor_type}, %rhs: {kernel_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
        f"  %result = {op_name} {conv_attr} ins(%lhs, %rhs: {input_tensor_type}, {kernel_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
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


# Enumerates ways to initialize tensor buffer contents.
@enum.unique
class TensorGenerator(enum.Enum):
    ZERO = "zero"  # Fill with zeros
    RANDOM = "random"  # Fill with (deterministic) pseudorandom values.


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with local_pseudorandom_state to limit the ways
# in which shuffling testcases changes which random values are generated.
pseudorandom_generator_seed = 1


def contents_generator_tag(generator: TensorGenerator):
    if generator == TensorGenerator.ZERO:
        return ""
    elif generator == TensorGenerator.RANDOM:
        global pseudorandom_generator_seed
        pseudorandom_generator_seed = pseudorandom_generator_seed + 1
        return f"!tag:iree:fully_specified_pseudorandom {pseudorandom_generator_seed}"
    else:
        raise ValueError(generator)


# Generate a 4d tensor function argument of the given size as `%name`.
def generate_random_4d_tensor(
    name: str,
    tensor_shape: list,
    element_type: typing.Union[InputElemTypeId, KernelElemTypeId],
):
    global pseudorandom_generator_seed
    pseudorandom_generator_seed = pseudorandom_generator_seed + 1
    return (
        f"  %{name}_dim0 = arith.constant {tensor_shape[0]} : i64\n"
        f"  %{name}_dim1 = arith.constant {tensor_shape[1]} : i64\n"
        f"  %{name}_dim2 = arith.constant {tensor_shape[2]} : i64\n"
        f"  %{name}_dim3 = arith.constant {tensor_shape[3]} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {pseudorandom_generator_seed} : i32\n"
        f"  %{name} = call @conv2d_test.generate_random_tensor(%device, %{name}_dim0, %{name}_dim1, %{name}_dim2, %{name}_dim3, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )


call_id = 0


def generate_call(
    function: MLIRFunction,
    input_type: InputElemTypeId,
    input_layout: InputLayout,
    kernel_type: KernelElemTypeId,
    kernel_layout: KernelLayout,
    conv2d_attr: ConvAttrs,
    acc_type: AccElemTypeId,
    shape: TestShape,
):
    global call_id
    func_name = f"{function.name}_{shape.n}_{shape.c}_{shape.h}_{shape.w}_{shape.f}_{shape.kh}_{shape.kw}"
    if shape.accumulate:
        func_name = f"{func_name}_acc"
    func_name = f"{func_name}_{call_id}"
    call_id = call_id + 1

    # layout of output tensor for checking correctness
    layout = -1
    description = f"Conv2d shape (NxCxHxWxFxKHxKW): {shape.n}x{shape.c}x{shape.h}x{shape.w}x{shape.f}x{shape.kh}x{shape.kw}"
    if input_layout == InputLayout.NCHW:
        if kernel_layout == KernelLayout.FCHW or kernel_layout == KernelLayout.HWCF:
            layout = 0  # for output tensor NxFxOHxOW
        else:
            raise ValueError(kernel_layout)
    elif input_layout == InputLayout.NHWC:
        if kernel_layout == KernelLayout.HWCF:
            layout = 1  # for output tensor NxOHxOWxF
        else:
            raise ValueError(kernel_layout)
    else:
        raise ValueError(InputLayout)

    op = (
        f"func.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    inp_shape, kernel_shape, out_shape = get_tensor_shape(
        shape,
        kernel_layout,
        input_layout,
        conv2d_attr,
    )

    op = op + generate_random_4d_tensor("input", inp_shape, input_type)
    op = op + generate_random_4d_tensor("kernel", kernel_shape, kernel_type)
    if shape.accumulate:
        op = op + generate_random_4d_tensor("acc", out_shape, acc_type)
        # TODO(#16168): there's a bug with in-place input->output aliasing and
        # we work around it here by passing in a unique copy.
        global pseudorandom_generator_seed
        pseudorandom_generator_seed = pseudorandom_generator_seed - 1
        op = op + generate_random_4d_tensor("acc_copy", out_shape, acc_type)
        op = op + (
            f"  %result = call @module.{function.name}(%input, %kernel, %acc_copy) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )
    else:
        op = op + (
            f"  %acc = util.null : !hal.buffer_view\n"
            f"  %result = call @module.{function.name}(%input, %kernel) : (!hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
        )

    op = op + (
        f"  %n = arith.constant {shape.n} : i64\n"
        f"  %c = arith.constant {shape.c} : i64\n"
        f"  %h = arith.constant {shape.h} : i64\n"
        f"  %w = arith.constant {shape.w} : i64\n"
        f"  %f = arith.constant {shape.f} : i64\n"
        f"  %kh = arith.constant {shape.kh} : i64\n"
        f"  %kw = arith.constant {shape.kw} : i64\n"
        f"  %layout = arith.constant {layout} : i64\n"
        f"  %sh = arith.constant {conv2d_attr.STRIDE[0]} : i64\n"
        f"  %sw = arith.constant {conv2d_attr.STRIDE[1]} : i64\n"
        f"  %dh = arith.constant {conv2d_attr.DILATION[0]} : i64\n"
        f"  %dw = arith.constant {conv2d_attr.DILATION[1]} : i64\n"
        f"  call @conv2d_test.check_conv2d_results(%device, %n, %c, %h, %w, %f, %kh, %kw, %layout, %sh, %sw, %dh, %dw, %input, %kernel, %acc, %result) : (!hal.device, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
    )

    op = op + "  return\n"
    op = op + "}\n"

    return TestCall(function=function, op=op)


# Generates all output files' contents as strings.
def generate(
    input_elem_type: InputElemTypeId,
    input_layout: InputLayout,
    kernel_elem_type: KernelElemTypeId,
    kernel_layout: KernelLayout,
    conv2d_attr: ConvAttrs,
    acc_elem_type: AccElemTypeId,
    shapes_id: ShapesId,
    compilation_info_id: CompilationInfoId,
):
    functions = {}
    calls = []

    for compilation_info in get_test_compilation_infos(
        compilation_info_id, input_elem_type, acc_elem_type
    ):
        for shape in get_test_shapes(shapes_id):
            for dynamicity in get_dynamicities(shapes_id):
                function = generate_function(
                    input_elem_type,
                    input_layout,
                    kernel_elem_type,
                    kernel_layout,
                    acc_elem_type,
                    conv2d_attr,
                    shape,
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
                        function,
                        input_elem_type,
                        input_layout,
                        kernel_elem_type,
                        kernel_layout,
                        conv2d_attr,
                        acc_elem_type,
                        shape,
                    )
                )

    return (functions, calls)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generator of e2e conv2d tests")
    parser.add_argument(
        "--output_conv2d_mlir",
        type=str,
        help="Path of output .mlir file containing the generated conv2d functions",
        required=True,
    )
    parser.add_argument(
        "--output_calls_mlir",
        type=str,
        help="Path of output .mlir file containing the calls",
        required=True,
    )
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["i8", "f32", "f16"],
        help="Numeric type of input tensors",
        required=True,
    )
    parser.add_argument(
        "--input_layout",
        type=str,
        default="nchw",
        choices=["nchw", "nhwc"],
        help="Layout of the input tensor. Currently, only nchw is supported.",
        required=False,
    )
    parser.add_argument(
        "--kernel_type",
        type=str,
        choices=["i8", "f32", "f16"],
        help="Numeric type of input tensors",
        required=True,
    )
    parser.add_argument(
        "--kernel_layout",
        type=str,
        default="fchw",
        choices=["fchw", "hwcf"],
        help="Layout of kernel tensor. Currently, only fchw is supported.",
        required=False,
    )
    parser.add_argument(
        "--acc_type",
        type=str,
        choices=["i32", "f32", "f16"],
        help="Numeric type of input tensors",
        default="",
        required=False,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        choices=[s.value for s in ShapesId],
        help="Collection of tensor shapes to test",
        required=True,
    )
    parser.add_argument(
        "--dilation",
        type=str,
        default="1,1",
        help="The dilation factor for the convolution operation. Comma-separated. As in 1,1",
        required=False,
    )
    parser.add_argument(
        "--stride",
        type=str,
        default="1,1",
        help="The stride factor for the convolution operation. Comma-separated. As in 1,1",
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
        "func.func private @conv2d_test.generate_random_tensor(%device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64, %dim3: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view\n"
        "func.func private @conv2d_test.check_conv2d_results(%device: !hal.device, %n: i64, %c: i64, %h: i64, %w: i64, %f:i64, %kh:i64, %kw:i64, %layout:i64, %sh:i64, %sw:i64, %dh:i64, %dw:i64, %input: !hal.buffer_view, %kernel: !hal.buffer_view, %acc: !hal.buffer_view, %actual_result: !hal.buffer_view)\n"
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
    input_type = InputElemTypeId(args.input_type)
    input_layout = InputLayout(args.input_layout)
    kernel_type = KernelElemTypeId(args.kernel_type)
    kernel_layout = KernelLayout(args.kernel_layout)

    acc_type = AccElemTypeId(args.acc_type)

    shapes_id = ShapesId(args.shapes)
    compilation_info_id = CompilationInfoId(args.compilation_info)

    conv2d_attr = ConvAttrs(
        tuple(map(int, args.stride.split(","))),
        tuple(map(int, args.dilation.split(","))),
    )

    (functions, calls) = generate(
        input_type,
        input_layout,
        kernel_type,
        kernel_layout,
        conv2d_attr,
        acc_type,
        shapes_id,
        compilation_info_id,
    )

    write_code_file(functions, args.output_conv2d_mlir)
    write_calls_file(
        functions,
        calls,
        args.output_calls_mlir,
        args.requirements,
    )


if __name__ == "__main__":
    main(parse_arguments())
