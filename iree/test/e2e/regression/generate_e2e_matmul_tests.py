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


# Data type of matrix entries. The string values must match MLIR data types.
# This is a superset of the values accepted for the --lhs_rhs_types= flag,
# as this also includes accumulator-specific types like i32.
@enum.unique
class MatrixElemTypeId(enum.Enum):
  I8 = "i8"
  I32 = "i32"
  F32 = "f32"


# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
  SMALL = "small"
  LARGE = "large"


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
  IDENTITY = "identity"  # Make an identity matrix (generalized to any shape).
  RANDOM = "random"  # Fill with (deterministic) pseudorandom values.


# Describes the shape of a matrix multiplication in the usual convention:
# the LHS is {m}x{k}, the RHS is {k}x{n}, the accumulator/result is {m}x{n}.
@dataclasses.dataclass
class TestShape:
  m: int
  k: int
  n: int


# Describes how to construct MLIR tensor types and how to initialize buffer
# contents for a test case (for an already given TestShape, and already given
# matrix element data types).
@dataclasses.dataclass
class TestGenerator:
  lhs: MatrixGenerator
  rhs: MatrixGenerator
  acc: MatrixGenerator
  dynamicity: Dynamicity


# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id.
def get_test_shapes(shapes_id: ShapesId):
  # Notes:
  # 1. Be conservative in adding more shapes, as that can include both the
  #    build and execution latency of tests. The build latency is nearly the
  #    same for all shapes, while execution latency grows cubicly i.e.
  #    linearly with m*k*n.
  # 2. Some shapes are commented out: they used to be tested but have been
  #    disabled to improve the trade-off between test coverage and build
  #    latency.
  if shapes_id == ShapesId.SMALL:
    return [  # Small sizes, square matrices
        # was range(1, 40) before trimming. The choice of 18 is so that we
        # exercise a case just above 16, as 16 will be a common kernel width.
        TestShape(m=x, k=x, n=x) for x in range(1, 18)
    ] + [
        # Small sizes, slightly rectangular matrices
        TestShape(m=2, k=3, n=4),
        #TestShape(m=8, k=7, n=6),
        #TestShape(m=15, k=16, n=17),
        TestShape(m=14, k=19, n=23),
        #TestShape(m=31, k=33, n=32),
        TestShape(m=25, k=41, n=35),
        # Small sizes, involving vectors (i.e. most rectangular cases)
        TestShape(m=10, k=1, n=1),
        TestShape(m=1, k=10, n=1),
        TestShape(m=1, k=1, n=10),
        #TestShape(m=1, k=10, n=10),
        #TestShape(m=10, k=1, n=10),
        #TestShape(m=10, k=10, n=1),
        # Small sizes, involving other very small dimensions just above 1
        TestShape(m=13, k=14, n=2),
        TestShape(m=3, k=17, n=12),
        TestShape(m=21, k=4, n=18),
        # Medium sizes, square matrices
        #TestShape(m=100, k=100, n=100),
        # Medium sizes, slightly rectangular matrices
        TestShape(m=101, k=102, n=103),
        # Medium sizes, involving vectors (i.e. most rectangular cases)
        TestShape(m=10000, k=1, n=1),
        TestShape(m=1, k=10000, n=1),
        TestShape(m=1, k=1, n=10000),
        #TestShape(m=1, k=1000, n=1000),
        #TestShape(m=1000, k=1, n=1000),
        #TestShape(m=1000, k=1000, n=1),
        # Medium sizes, involving other very small dimensions just above 1
        TestShape(m=1300, k=1300, n=2),
        #TestShape(m=1300, k=1300, n=3),
        #TestShape(m=1300, k=1300, n=4),
    ]
  if shapes_id == ShapesId.LARGE:
    return [
        # Large sizes, powers of two
        TestShape(m=256, k=256, n=512),
        #TestShape(m=512, k=512, n=128),
        #TestShape(m=1024, k=512, n=512),
        #TestShape(m=512, k=1024, n=512),
        # Large sizes, powers of two minus one
        TestShape(m=127, k=63, n=511),
        # Large sizes, powers of two plus one
        TestShape(m=129, k=65, n=513),
        # Large sizes, misc.
        #TestShape(m=200, k=300, n=400),
        TestShape(m=123, k=456, n=789),
        #TestShape(m=500, k=500, n=50),
        # Be conservative in adding larger shapes. They can result in
        # high latency tests. If you have to, consider splitting them
        # out in a way that constrains the latency impact, e.g. by
        # running on fewer backends/drivers or with fewer generators
        # (see get_test_generators).
    ]
  raise ValueError(shapes_id)


# Returns the list of TestGenerator's to use for the collection of shapes
# identified by shapes_id.
def get_test_generators(shapes_id: ShapesId):
  if shapes_id == ShapesId.SMALL:
    return [
        # Generators using simple matrices for ease of numerical debugging.
        # They don't add significant test coverage (all bugs are hit by
        # tests using random matrices anyway). They are only here to make
        # the bulk of our debugging easier.
        TestGenerator(lhs=MatrixGenerator.IDENTITY,
                      rhs=MatrixGenerator.IDENTITY,
                      acc=MatrixGenerator.ZERO,
                      dynamicity=Dynamicity.DYNAMIC),
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.IDENTITY,
                      acc=MatrixGenerator.ZERO,
                      dynamicity=Dynamicity.DYNAMIC),
        TestGenerator(lhs=MatrixGenerator.IDENTITY,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.ZERO,
                      dynamicity=Dynamicity.DYNAMIC),
        TestGenerator(lhs=MatrixGenerator.IDENTITY,
                      rhs=MatrixGenerator.IDENTITY,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.DYNAMIC),
        # Generators using general random matrices
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.DYNAMIC),
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.STATIC),
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.MIXED),
    ]
  if shapes_id == ShapesId.LARGE:
    return [
        # Fewer generators are used for large shapes, to limit the
        # latency impact. Most bugs are going to be caught on small
        # shapes anyway.
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.DYNAMIC),
        TestGenerator(lhs=MatrixGenerator.RANDOM,
                      rhs=MatrixGenerator.RANDOM,
                      acc=MatrixGenerator.RANDOM,
                      dynamicity=Dynamicity.STATIC),
    ]
  raise ValueError(shapes_id)


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with pseudorandom_generator_seed to limit the ways
# in which shuffling testcases changes which random values are generated.
local_pseudorandom_state = 1


# A static size value, i.e. a size value that could appear in a MLIR type
# such as 'tensor<?x4xf32>'. None means a dynamic size, similar to '?' in MLIR.
@dataclasses.dataclass
class DimSize:
  value: typing.Optional[int]


# Generates a compile-time MLIR size value, i.e. either a fixed positive integer
# or None (which maps to MLIR '?') depending on dynamicity.
def static_size(x: int, dynamicity: Dynamicity):
  if dynamicity == Dynamicity.DYNAMIC:
    return DimSize(None)
  elif dynamicity == Dynamicity.STATIC:
    return DimSize(x)
  elif dynamicity == Dynamicity.MIXED:
    global local_pseudorandom_state
    # Same as C++ std::minstd_rand.
    # Using a local pseudorandom generator implementation ensures that it's
    # completely reproducible, across runs and across machines.
    local_pseudorandom_state = (local_pseudorandom_state * 48271) % 2147483647
    return DimSize(x if local_pseudorandom_state > 1073741824 else None)
  else:
    raise ValueError(dynamicity)


# Stringification used for generating MLIR types, e.g. tensor<?x?xf32>.
def int_or_question_mark(s: DimSize):
  return s.value or "?"


# Stringification used for generating alphanumeric identifiers, e.g.
# func @somefunction_DYNxDYNxf32, where we can't use "?" characters.
def int_or_DYN(s: DimSize):
  return s.value or "DYN"


# Describes the fully resolved static dimensions of all 3 input matrices,
# LHS, RHS, and Accumulator, in a testcase.
# Each value is a string, which may either represent a positive integer such as "123",
# or a "?" string, meaning a dynamic dimension as in MLIR.
# These string values are used to generate MLIR function names and tensor shapes.
@dataclasses.dataclass
class TestInputMatricesStaticShapes:
  lhs_rows: DimSize
  lhs_cols: DimSize
  rhs_rows: DimSize
  rhs_cols: DimSize
  acc_rows: DimSize
  acc_cols: DimSize


# Helper for generate_function. Generates TestInputMatricesStaticShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicity to
# the set of static shapes to be used in a test function's input tensors.
def generate_static_shapes(shape: TestShape, dynamicity: Dynamicity):
  return TestInputMatricesStaticShapes(
      lhs_rows=static_size(shape.m, dynamicity),
      lhs_cols=static_size(shape.k, dynamicity),
      rhs_rows=static_size(shape.k, dynamicity),
      rhs_cols=static_size(shape.n, dynamicity),
      acc_rows=static_size(shape.m, dynamicity),
      acc_cols=static_size(shape.n, dynamicity),
  )


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(lhs_rhs_type: MatrixElemTypeId,
                           acc_type: MatrixElemTypeId,
                           static_shapes: TestInputMatricesStaticShapes):
  input_t = lhs_rhs_type.value
  acc_t = acc_type.value
  lhs_m = int_or_DYN(static_shapes.lhs_rows)
  lhs_k = int_or_DYN(static_shapes.lhs_cols)
  rhs_k = int_or_DYN(static_shapes.rhs_rows)
  rhs_n = int_or_DYN(static_shapes.rhs_cols)
  acc_m = int_or_DYN(static_shapes.acc_rows)
  acc_n = int_or_DYN(static_shapes.acc_cols)
  return f"matmul_{lhs_m}x{lhs_k}x{input_t}_times_{rhs_k}x{rhs_n}x{input_t}_into_{acc_m}x{acc_n}x{acc_t}"


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
  name: str
  definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul and
# will just call linalg.matmul with them, returning its result.
def generate_function(lhs_rhs_type: MatrixElemTypeId,
                      acc_type: MatrixElemTypeId, shape: TestShape,
                      dynamicity: Dynamicity):
  static_shapes = generate_static_shapes(shape, dynamicity)
  func_name = generate_function_name(lhs_rhs_type, acc_type, static_shapes)
  lhs_m = int_or_question_mark(static_shapes.lhs_rows)
  lhs_k = int_or_question_mark(static_shapes.lhs_cols)
  rhs_k = int_or_question_mark(static_shapes.rhs_rows)
  rhs_n = int_or_question_mark(static_shapes.rhs_cols)
  acc_m = int_or_question_mark(static_shapes.acc_rows)
  acc_n = int_or_question_mark(static_shapes.acc_cols)
  lhs_tensor_type = f"tensor<{lhs_m}x{lhs_k}x{lhs_rhs_type.value}>"
  rhs_tensor_type = f"tensor<{rhs_k}x{rhs_n}x{lhs_rhs_type.value}>"
  acc_tensor_type = f"tensor<{acc_m}x{acc_n}x{acc_type.value}>"
  func_definition = (
      f"func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
      f"  %result = linalg.matmul ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
      f"  return %result: {acc_tensor_type}\n"
      f"}}\n")
  return MLIRFunction(
      name=func_name,
      definition=func_definition,
  )


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with local_pseudorandom_state to limit the ways
# in which shuffling testcases changes which random values are generated.
pseudorandom_generator_seed = 1


# Generates a contents_generator tag to use in the output trace.
def contents_generator_tag(generator: MatrixGenerator):
  if generator == MatrixGenerator.ZERO:
    return ""
  elif generator == MatrixGenerator.IDENTITY:
    return "!tag:iree:identity_matrix"
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
                   acc_type: MatrixElemTypeId, shape: TestShape,
                   gen: TestGenerator):
  lhs_arg = generate_trace_matrix_arg([shape.m, shape.k], lhs_rhs_type, gen.lhs)
  rhs_arg = generate_trace_matrix_arg([shape.k, shape.n], lhs_rhs_type, gen.rhs)
  acc_arg = generate_trace_matrix_arg([shape.m, shape.n], acc_type, gen.acc)
  result_arg = generate_trace_matrix_arg([shape.m, shape.n], acc_type,
                                         MatrixGenerator.ZERO)
  return {
      "type": "call",
      "function": "module." + func_name,
      "args": [
          lhs_arg,
          rhs_arg,
          acc_arg,
      ],
      "results": [result_arg,],
  }


# Generates all output files' contents as strings.
def generate(lhs_rhs_type: MatrixElemTypeId, acc_type: MatrixElemTypeId,
             shapes_id: ShapesId):
  function_definitions = {}
  traces = []
  for shape in get_test_shapes(shapes_id):
    for gen in get_test_generators(shapes_id):
      function = generate_function(lhs_rhs_type, acc_type, shape,
                                   gen.dynamicity)
      # Different testcases may differ only by runtime parameters but
      # share the same code. For example, dynamic-shapes testcases
      # share the same code involing tensor<?x?xf32> even though the runtime
      # value in the trace are different. That's why we call
      # generate_function conditionally, and generate_trace unconditionally.
      if function.name not in function_definitions:
        function_definitions[function.name] = function.definition
      traces.append(
          generate_trace(function.name, lhs_rhs_type, acc_type, shape, gen))
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
                      choices=["i8", "f32"],
                      help="Numeric type of input matrices",
                      required=True)
  parser.add_argument("--shapes",
                      type=str,
                      choices=[s.value for s in ShapesId],
                      help="Collection of matrix shapes to test",
                      required=True)
  parser.add_argument(
      "--module_path",
      type=str,
      help=
      "Module path (typically .vmfb) to be referenced in the output trace. Should match the output path of the iree-translate command generating the module.",
      required=True)

  return parser.parse_args()


def write_code_file(function_definitions, filename):
  with open(filename, "w") as file:
    for funcname in function_definitions:
      file.write(function_definitions[funcname] + "\n")


def write_trace_file(traces, filename, module_path):
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

  for trace in traces:
    yaml_documents.append(trace)

  dumped_yaml = yaml.dump_all(yaml_documents, sort_keys=False)

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
  (function_definitions, traces) = generate(lhs_rhs_type, acc_type, shapes_id)
  write_code_file(function_definitions, args.output_code)
  write_trace_file(traces, args.output_trace, args.module_path)


if __name__ == "__main__":
  main(parse_arguments())
