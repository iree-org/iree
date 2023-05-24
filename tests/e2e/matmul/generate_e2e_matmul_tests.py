#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""iree_generated_trace_runner_test generator for e2e matmul tests.
"""

import argparse
import os
import yaml
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
  I8 = "i8"
  I32 = "i32"
  F32 = "f32"
  F16 = "f16"


# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
  SMALL = "small"
  LARGE = "large"
  GPU_LARGE = "gpu_large"


# Enumerates of the collections of compilation info that we can generate tests
# for. The values are the accepted values for the --compilation_info= flag.
@enum.unique
class CompilationInfoId(enum.Enum):
  NONE = ""
  LLVMGPUMatmulSimt = "LLVMGPUMatmulSimt"
  LLVMGPUMatmulTensorCore = "LLVMGPUMatmulTensorCore"
  LLVMGPUMatmulTensorCoreMmaSync = "LLVMGPUMatmulTensorCoreMmaSync"
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


# Describes how to construct compilation info for the testcase.
@dataclasses.dataclass
class CompilationInfo:
  # Lowering Config
  tile_sizes: typing.List[typing.List[int]]
  # Translation Info
  dispatch_lowering_pass_pipeline: str
  workload_per_wg: typing.List[int]
  software_pipeline_depth: int
  # Compilation info
  workgroup_size: typing.List[int]

  # Prints the workgroup size as 'index' types
  def workgroup_size_str(self):
    return "[" + ", ".join([f"{size} : index" for size in self.workgroup_size
                           ]) + "]"


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
  if shapes_id == ShapesId.SMALL:
    return [
        # square matrices. Start by the simplest case of 1x1x1.
        TestShape(m=1, k=1, n=1, accumulate=True),
        TestShape(m=1, k=1, n=1, accumulate=False),
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
        # some random large sizes
        TestShape(m=123, k=456, n=789, accumulate=True),
        TestShape(m=654, k=321, n=234, accumulate=False),
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
  if shapes_id == ShapesId.GPU_LARGE:
    return [
        TestShape(m=256, k=128, n=512, accumulate=True),
        TestShape(m=256, k=128, n=512, accumulate=False),
    ]

  raise ValueError(shapes_id)


# Returns the list of Dynamicity's to use for the collection of shapes
# identified by shapes_id.
def get_dynamicities(shapes_id: ShapesId):
  if shapes_id == ShapesId.GPU_LARGE:
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


# Constructs a TileWorkgroupSizePair for SPIRV Targets enforcing the constraints between
# the workgroup_size and tile size
def get_spirv_tile_workgroup_size_pair(workgroup_size,
                                       t_tile_k,
                                       t_tile_m=4,
                                       t_tile_n=4):
  x, y, z = workgroup_size
  wg_tile_m = y * t_tile_m
  wg_tile_n = x * t_tile_n
  return TileWorkgroupSizePair(
      [[wg_tile_m, wg_tile_n], [t_tile_m, t_tile_n], [0, 0, t_tile_k]],
      workgroup_size)


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


# Returns the list of CompilationInfo's to use for the CompilationInfoId.
def get_test_compilation_infos(
    compilation_info_id: CompilationInfoId, lhs_rhs_type: MatrixElemTypeId
) -> typing.List[typing.Optional[CompilationInfo]]:
  if compilation_info_id == CompilationInfoId.NONE:
    return [None]
  if compilation_info_id == CompilationInfoId.LLVMGPUMatmulSimt:
    tile_workgroup_size_pairs = [
        TileWorkgroupSizePair([[32, 128, 32]], [32, 8, 1]),
        TileWorkgroupSizePair([[128, 64, 8]], [16, 8, 1]),
        TileWorkgroupSizePair([[16, 256, 32]], [64, 2, 1]),
        TileWorkgroupSizePair([[8, 32, 32]], [8, 8, 1]),
        TileWorkgroupSizePair([[8, 128, 4]], [32, 1, 1]),
        TileWorkgroupSizePair([[16, 64, 4]], [16, 2, 1]),
        TileWorkgroupSizePair([[1, 128, 8]], [32, 1, 1]),
    ]
  elif compilation_info_id == CompilationInfoId.SPIRVVectorizeNVIDIA:
    tile_workgroup_size_pairs = get_all_spirv_tile_workgroup_size_pairs(32)
  elif compilation_info_id == CompilationInfoId.SPIRVVectorizeMali:
    tile_workgroup_size_pairs = get_all_spirv_tile_workgroup_size_pairs(4)
  elif compilation_info_id == CompilationInfoId.LLVMGPUMatmulTensorCore or compilation_info_id == CompilationInfoId.LLVMGPUMatmulTensorCoreMmaSync:
    tile_workgroup_size_pairs = []
    ## WarpShape = 2x2
    tile_workgroup_size_pairs.append(
        TileWorkgroupSizePair([[32, 32, 16]], [64, 2, 1]))
    tile_workgroup_size_pairs.append(
        TileWorkgroupSizePair([[64, 64, 64]], [64, 2, 1]))

    ## WarpShape = 4x1
    tile_workgroup_size_pairs.append(
        TileWorkgroupSizePair([[32, 32, 32]], [64, 1, 1]))

    ## WarpShape = 2x2 with large tiles using larger Shared Memory capacity.
    if lhs_rhs_type == MatrixElemTypeId.F16:
      tile_workgroup_size_pairs.append(
          TileWorkgroupSizePair([[128, 128, 64]], [64, 2, 1]))
    elif lhs_rhs_type == MatrixElemTypeId.F32:
      tile_workgroup_size_pairs.append(
          TileWorkgroupSizePair([[128, 128, 16]], [64, 2, 1]))

  compilation_infos = []
  for tile_workgroup_size_pair in tile_workgroup_size_pairs:
    compilation_infos.append(
        CompilationInfo(
            tile_sizes=tile_workgroup_size_pair.tile_size,
            dispatch_lowering_pass_pipeline=compilation_info_id.value,
            workload_per_wg=[
                a for a in reversed(tile_workgroup_size_pair.tile_size[0:2])
            ],
            workgroup_size=tile_workgroup_size_pair.workgroup_size,
            software_pipeline_depth=3))
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
def generate_shapes(shape: TestShape, dynamicity: Dynamicity):
  shapes = TestInputMatricesShapes(
      lhs_rows=shape_dim(shape.m, dynamicity),
      lhs_cols=shape_dim(shape.k, dynamicity),
      rhs_rows=shape_dim(shape.k, dynamicity),
      rhs_cols=shape_dim(shape.n, dynamicity),
      acc_rows=shape_dim(shape.m, dynamicity),
      acc_cols=shape_dim(shape.n, dynamicity),
  )
  return shapes


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shapes: TestInputMatricesShapes,
    accumulate: bool,
    compilation_info: typing.Optional[CompilationInfo] = None):
  input_t = lhs_rhs_type.value
  acc_t = acc_type.value
  lhs_m = int_or_DYN(shapes.lhs_rows)
  lhs_k = int_or_DYN(shapes.lhs_cols)
  rhs_k = int_or_DYN(shapes.rhs_rows)
  rhs_n = int_or_DYN(shapes.rhs_cols)
  acc_m = int_or_DYN(shapes.acc_rows)
  acc_n = int_or_DYN(shapes.acc_cols)

  info = ""
  if compilation_info:
    tile_sizes = list(itertools.chain(*compilation_info.tile_sizes))
    tile_workgroup_key = "_".join([
        str(a) for a in tile_sizes
    ]) + "_" + "_".join([str(a) for a in compilation_info.workgroup_size])
    info = f"_for_{compilation_info.dispatch_lowering_pass_pipeline}_{tile_workgroup_key}"

  matmul_kind = "matmul_accumulate" if accumulate else "matmul"
  return f"{matmul_kind}_{lhs_m}x{lhs_k}x{input_t}_times_{rhs_k}x{rhs_n}x{input_t}_into_{acc_m}x{acc_n}x{acc_t}{info}"


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
  name: str
  definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul and
# will just call linalg.matmul with them, returning its result.
def generate_function(
    lhs_rhs_type: MatrixElemTypeId,
    acc_type: MatrixElemTypeId,
    shape: TestShape,
    dynamicity: Dynamicity,
    compilation_info: typing.Optional[CompilationInfo] = None):
  shapes = generate_shapes(shape, dynamicity)
  func_name = generate_function_name(lhs_rhs_type, acc_type, shapes,
                                     shape.accumulate, compilation_info)
  lhs_m = int_or_question_mark(shapes.lhs_rows)
  lhs_k = int_or_question_mark(shapes.lhs_cols)
  rhs_k = int_or_question_mark(shapes.rhs_rows)
  rhs_n = int_or_question_mark(shapes.rhs_cols)
  acc_m = int_or_question_mark(shapes.acc_rows)
  acc_n = int_or_question_mark(shapes.acc_cols)
  lhs_tensor_type = f"tensor<{lhs_m}x{lhs_k}x{lhs_rhs_type.value}>"
  rhs_tensor_type = f"tensor<{rhs_k}x{rhs_n}x{lhs_rhs_type.value}>"
  acc_tensor_type = f"tensor<{acc_m}x{acc_n}x{acc_type.value}>"

  # Compilation info is optional; prints empty string by default.
  func_definition = ""
  compilation_info_attr = ""
  if compilation_info:
    if "SPIRV" in compilation_info.dispatch_lowering_pass_pipeline == "SPIRVVectorizeMali":
      dispatch_lowering_pass_pipeline = "SPIRVBaseVectorize"
    elif compilation_info.dispatch_lowering_pass_pipeline == "SPIRVVectorizeNVIDIA":
      # TODO: change to test SPIRVMatmulPromoteVectorize too
      dispatch_lowering_pass_pipeline = "SPIRVBaseVectorize"
    else:
      dispatch_lowering_pass_pipeline = compilation_info.dispatch_lowering_pass_pipeline
    compilation_info_string = (
        f"#compilation{generate_function.compilation_index} = #iree_codegen.compilation_info<\n"
        f"  lowering_config = <tile_sizes = {compilation_info.tile_sizes}>,\n"
        f"  translation_info = <{dispatch_lowering_pass_pipeline}\n"
        f"  pipeline_depth = {compilation_info.software_pipeline_depth}>,\n"
        f"  workgroup_size = {compilation_info.workgroup_size_str()}>\n")
    compilation_info_attr = f"{{compilation_info = #compilation{generate_function.compilation_index}}} "
    func_definition = func_definition + compilation_info_string
    generate_function.compilation_index += 1

  if shape.accumulate:
    func_definition = func_definition + (
        f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
        f"  %result = linalg.matmul {compilation_info_attr}ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
        f"  return %result: {acc_tensor_type}\n"
        f"}}\n")
  else:
    literal_zero_for_acc_type = "0.0" if "f" in acc_type.value else "0"
    acc_dyn_sizes = []
    if acc_m == "?":
      func_definition = func_definition + (
          f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
          f"  %c0 = arith.constant 0 : index\n"
          f"  %c1 = arith.constant 1 : index\n"
          f"  %acc_dim0 = tensor.dim %lhs, %c0 : {lhs_tensor_type}\n"
          f"  %acc_dim1 = tensor.dim %rhs, %c1 : {rhs_tensor_type}\n"
          f"  %init_acc = tensor.empty(%acc_dim0, %acc_dim1) : {acc_tensor_type}\n"
          f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
          f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
          f"  %result = linalg.matmul {compilation_info_attr}ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
          f"  return %result: {acc_tensor_type}\n"
          f"}}\n")
    else:
      func_definition = func_definition + (
          f"func.func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}) -> {acc_tensor_type} {{\n"
          f"  %init_acc = tensor.empty() : {acc_tensor_type}\n"
          f"  %c0_acc_type = arith.constant {literal_zero_for_acc_type}: {acc_type.value}\n"
          f"  %acc = linalg.fill ins(%c0_acc_type : {acc_type.value}) outs(%init_acc : {acc_tensor_type}) -> {acc_tensor_type}\n"
          f"  %result = linalg.matmul {compilation_info_attr}ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
          f"  return %result: {acc_tensor_type}\n"
          f"}}\n")
  return MLIRFunction(
      name=func_name,
      definition=func_definition,
  )


# Counter for producing unique compilation info attrs
generate_function.compilation_index = 0

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


# Generate a matrix function argument in the output trace, as a dictionary
# to be passed to yaml.dump.
def generate_trace_matrix_arg(matrix_shape: list,
                              element_type: MatrixElemTypeId,
                              generator: MatrixGenerator):
  result = {
      "type": "hal.buffer_view",
      "shape": matrix_shape,
      "element_type": element_type.value,
  }
  generator_tag = contents_generator_tag(generator)
  if generator_tag:
    result["contents_generator"] = generator_tag
  return result


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_trace(func_name: str, lhs_rhs_type: MatrixElemTypeId,
                   acc_type: MatrixElemTypeId, shape: TestShape):
  args = [
      generate_trace_matrix_arg([shape.m, shape.k], lhs_rhs_type,
                                MatrixGenerator.RANDOM),
      generate_trace_matrix_arg([shape.k, shape.n], lhs_rhs_type,
                                MatrixGenerator.RANDOM),
  ]
  if shape.accumulate:
    args.append(
        generate_trace_matrix_arg([shape.m, shape.n], acc_type,
                                  MatrixGenerator.RANDOM))

  result = generate_trace_matrix_arg([shape.m, shape.n], acc_type,
                                     MatrixGenerator.ZERO)
  return {
      "type": "call",
      "function": "module." + func_name,
      "args": args,
      "results": [result],
  }


# Generates all output files' contents as strings.
def generate(lhs_rhs_type: MatrixElemTypeId, acc_type: MatrixElemTypeId,
             shapes_id: ShapesId, compilation_info_id: CompilationInfoId):
  function_definitions = {}
  traces = []

  for compilation_info in get_test_compilation_infos(compilation_info_id,
                                                     lhs_rhs_type):
    for shape in get_test_shapes(shapes_id):
      for dynamicity in get_dynamicities(shapes_id):
        function = generate_function(lhs_rhs_type, acc_type, shape, dynamicity,
                                     compilation_info)
        # Different testcases may differ only by runtime parameters but
        # share the same code. For example, dynamic-shapes testcases
        # share the same code involing tensor<?x?xf32> even though the runtime
        # value in the trace are different. That's why we append conditionally
        # to traces, but unconditionally to function_definitions.
        if function.name not in function_definitions:
          function_definitions[function.name] = function.definition
        traces.append(
            generate_trace(function.name, lhs_rhs_type, acc_type, shape))

  return (function_definitions, traces)


def parse_arguments():
  parser = argparse.ArgumentParser(description="Generator of e2e matmul tests")
  parser.add_argument("--output_code",
                      type=str,
                      help="Path of output .mlir file",
                      required=True)
  parser.add_argument("--output_trace",
                      type=str,
                      help="Path of output .yaml trace file",
                      required=True)
  parser.add_argument("--lhs_rhs_type",
                      type=str,
                      choices=["i8", "f32", "f16"],
                      help="Numeric type of input matrices",
                      required=True)
  parser.add_argument("--shapes",
                      type=str,
                      choices=[s.value for s in ShapesId],
                      help="Collection of matrix shapes to test",
                      required=True)
  parser.add_argument("--compilation_info",
                      type=str,
                      choices=[i.value for i in CompilationInfoId],
                      help="Collection of compilation info setups to test",
                      default="",
                      required=False)

  parser.add_argument(
      "--module_path",
      type=str,
      help=
      "Module path (typically .vmfb) to be referenced in the output trace. Should match the output path of the iree-compile command generating the module.",
      required=True)
  parser.add_argument(
      "--requirements",
      type=str,
      help=
      "Target requirements for this module. Comma-separated. As in -iree-llvmcpu-target-cpu-features. If the target device does not meet all of the requirements, the test will be skipped.",
      required=False)
  return parser.parse_args()


def write_code_file(function_definitions, filename):
  with open(filename, "w") as file:
    for funcname in function_definitions:
      file.write(function_definitions[funcname] + "\n")


def write_trace_file(traces, filename, module_path, requirements):
  yaml_documents = [
      {
          "type": "context_load",
      },
      {
          "type": "module_load",
          "module": {
              "name": "hal",
              "type": "builtin",
          }
      },
      {
          "type": "module_load",
          "module": {
              "name": "module",
              "type": "bytecode",
              "path": os.path.relpath(module_path, os.path.dirname(filename))
          }
      },
  ]
  if requirements:
    yaml_documents.append({
        "type": "requirements",
        "target_features": [req.lstrip("+") for req in requirements.split(",")],
    })

  for trace in traces:
    yaml_documents.append(trace)

  dumped_yaml = yaml.dump_all(yaml_documents)

  # TODO: This regex substitution is a hack as I couldn't figure how to have
  # PyYAML dump our custom contents_generator into the desired format, e.g.
  #   contents_generator: !tag:iree:fully_specified_pseudorandom 368
  # Someone with better knowledge of YAML is welcome to fix this, possibly by
  # changing that format if that's appropriate! So long as the e2e_matmul tests
  # pass.
  processed_yaml = re.sub(r"'(![^']*)'", "\\1", dumped_yaml)

  with open(filename, "w") as file:
    file.write(processed_yaml)


# For now, the accumulator type can always be inferred from the input LHS/RHS
# type, so we do that. That is temporary: eventually there will be cases
# where the same input types are used with different accumulator types, e.g.
# f16 inputs with both f16 and f32 accumulator.
def infer_acc_type(lhs_rhs_type: MatrixElemTypeId):
  if lhs_rhs_type == MatrixElemTypeId.I8:
    return MatrixElemTypeId.I32
  else:
    return lhs_rhs_type


def main(args):
  lhs_rhs_type = MatrixElemTypeId(args.lhs_rhs_type)
  acc_type = infer_acc_type(lhs_rhs_type)
  shapes_id = ShapesId(args.shapes)
  compilation_info_id = CompilationInfoId(args.compilation_info)
  (function_definitions, traces) = generate(lhs_rhs_type, acc_type, shapes_id,
                                            compilation_info_id)

  write_code_file(function_definitions, args.output_code)
  write_trace_file(traces, args.output_trace, args.module_path,
                   args.requirements)


if __name__ == "__main__":
  main(parse_arguments())
