#!/usr/bin/env python3
# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""iree_generated_check_test generator for end-to-end matrix multiplication.
"""

import argparse
import random
import os
import yaml
import re


# Returns lists of shapes as (M, K, N) tuples.
# For example (M, K, 1) is a matrix*vector product, and (M, 1, N) is an outer
# product.
def get_test_shapes():
  return {
      "small": [  # Small sizes, square matrices
          (x, x, x) for x in range(1, 40)
      ] + [
          # Small sizes, slightly rectangular matrices
          (2, 3, 4),
          (8, 7, 6),
          (15, 16, 17),
          (14, 19, 23),
          (31, 33, 32),
          (25, 41, 35),
          # Small sizes, involving vectors (i.e. most rectangular cases)
          (10, 1, 1),
          (1, 10, 1),
          (1, 1, 10),
          (1, 10, 10),
          (10, 1, 10),
          (10, 10, 1),
          # Small sizes, involving other very small dimensions just above 1
          (13, 14, 2),
          (3, 17, 12),
          (21, 4, 18),
          # Medium sizes, square matrices
          (100, 100, 100),
          # Medium sizes, slightly rectangular matrices
          (101, 102, 103),
          # Medium sizes, involving vectors (i.e. most rectangular cases)
          (10000, 1, 1),
          (1, 10000, 1),
          (1, 1, 10000),
          (1, 1000, 1000),
          (1000, 1, 1000),
          (1000, 1000, 1),
          # Medium sizes, involving other very small dimensions just above 1
          (1300, 1300, 2),
          (1300, 1300, 3),
          (1300, 1300, 4),
      ],
      "large": [
          # Large sizes, powers of two
          (256, 256, 512),
          (512, 512, 128),
          (1024, 512, 512),
          (512, 1024, 512),
          # Large sizes, powers of two minus one
          (127, 63, 511),
          # Large sizes, powers of two plus one
          (129, 65, 513),
          # Large sizes, misc.
          (200, 300, 400),
          (123, 456, 789),
          (500, 500, 50),
          # Be conservative in adding larger shapes. They can result in
          # high latency tests. If you have to, consider splitting them
          # out in a way that constrains the latency impact, e.g. by
          # running on fewer backends/drivers or with fewer generators
          # (see get_test_generators).
      ]
  }


# Returns lists of 'generators', which are tuples of the form
# (lhs_generator, rhs_generator, acc_generator, dynamicity)
# The first 3 entries specify how to generate test input data.
# The dynamicity entry chooses between static, dynamic or mixed shapes.
#
# TODO (Issue #7431): turn into enum and dataclass.
def get_test_generators():
  return {
      "small": [
          # Generators using simple matrices for ease of numerical debugging.
          # They don't add significant test coverage (all bugs are hit by
          # tests using random matrices anyway). They are only here to make
          # the bulk of our debugging easier.
          ("identity", "identity", "zero", "dynamic"),
          ("random", "identity", "zero", "dynamic"),
          ("identity", "random", "zero", "dynamic"),
          ("identity", "identity", "random", "dynamic"),
          # Generators using general random matrices
          ("random", "random", "random", "dynamic"),
          ("random", "random", "random", "static"),
          # TODO: enable 'mixed' testcases. For now they cause iree-opt
          # errors.
          #("random", "random", "random", "mixed"),
      ],
      "large": [
          # Fewer generators are used for large shapes, to limit the
          # latency impact. Most bugs are going to be caught on small
          # shapes anyway.
          ("random", "random", "random", "dynamic"),
          ("random", "random", "random", "static"),
      ]
  }


# Generates a name for a test function in the generated MLIR code.
def function_name(lhs_rhs_type, accum_type, shape, gen):
  return f"{lhs_rhs_type}_{gen[3]}_{gen[0]}_{shape[0]}x{shape[1]}_times_{gen[1]}_{shape[1]}x{shape[2]}_plus_{gen[2]}_{accum_type}"


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with pseudorandom_generator_seed to limit the ways
# in which shuffling testcases changes which random values are generated.
local_pseudorandom_state = 1


# Generates a compile-time MLIR size value, i.e. either a fixed positive integer
# or a '?' depending on dynamicity.
def static_size(x, dynamicity):
  if dynamicity == "dynamic":
    return "?"
  elif dynamicity == "static":
    return x
  elif dynamicity == "mixed":
    global local_pseudorandom_state
    # Same as C++ std::minstd_rand.
    # Using a local pseudorandom generator implementation ensures that it's
    # completely reproducible, across runs and across machines.
    local_pseudorandom_state = (local_pseudorandom_state * 48271) % 2147483647
    return x if local_pseudorandom_state > 1073741824 else "?"
  else:
    raise ValueError(dynamicity)


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.matmul and
# will just call linalg.matmul with them, returning its result.
def generate_function(func_name, lhs_rhs_type, accum_type, shape, gen):
  (m, k, n) = shape
  lhs_m = static_size(m, gen[3])
  lhs_k = static_size(k, gen[3])
  rhs_k = static_size(k, gen[3])
  rhs_n = static_size(n, gen[3])
  acc_m = static_size(m, gen[3])
  acc_n = static_size(n, gen[3])
  lhs_tensor_type = f"tensor<{lhs_m}x{lhs_k}x{lhs_rhs_type}>"
  rhs_tensor_type = f"tensor<{rhs_k}x{rhs_n}x{lhs_rhs_type}>"
  acc_tensor_type = f"tensor<{acc_m}x{acc_n}x{accum_type}>"
  return (
      f"func @{func_name}(%lhs: {lhs_tensor_type}, %rhs: {rhs_tensor_type}, %acc: {acc_tensor_type}) -> {acc_tensor_type} {{\n"
      f"  %result = linalg.matmul ins(%lhs, %rhs: {lhs_tensor_type}, {rhs_tensor_type}) outs(%acc: {acc_tensor_type}) -> {acc_tensor_type}\n"
      f"  return %result: {acc_tensor_type}\n"
      f"}}\n")


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with local_pseudorandom_state to limit the ways
# in which shuffling testcases changes which random values are generated.
pseudorandom_generator_seed = 1


# Generates a contents_generator tag to use in the output trace.
def contents_generator_tag(generator):
  if generator == "zero":
    return ""
  elif generator == "identity":
    return "!tag:iree:identity_matrix"
  elif generator == "random":
    global pseudorandom_generator_seed
    pseudorandom_generator_seed = pseudorandom_generator_seed + 1
    return f"!tag:iree:fully_specified_pseudorandom {pseudorandom_generator_seed}"
  else:
    raise ValueError(generator)


# Generate a matrix function argument in the output trace, as a dictionary
# to be passed to yaml.dump.
def generate_trace_matrix_arg(matrix_shape, element_type, generator):
  result = {
      "type": "hal.buffer_view",
      "shape": matrix_shape,
      "element_type": element_type,
  }
  generator_tag = contents_generator_tag(generator)
  if generator_tag:
    result["contents_generator"] = generator_tag
  return result


# Generates the output trace for a testcase i.e. a single test function call,
# as a dictionary to be passed to yaml.dump.
def generate_trace(func_name, lhs_rhs_type, acc_type, shape, gen):
  (m, k, n) = shape
  lhs_arg = generate_trace_matrix_arg([m, k], lhs_rhs_type, gen[0])
  rhs_arg = generate_trace_matrix_arg([k, n], lhs_rhs_type, gen[1])
  acc_arg = generate_trace_matrix_arg([m, n], acc_type, gen[2])
  result_arg = generate_trace_matrix_arg([m, n], acc_type, "zero")
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
def generate(args):
  functions = {}
  traces = []
  lhs_rhs_type = args.lhs_rhs_type
  accum_type = 'i32' if lhs_rhs_type == 'i8' else lhs_rhs_type
  for shape in get_test_shapes()[args.shapes]:
    for gen in get_test_generators()[args.shapes]:
      func_name = function_name(lhs_rhs_type, accum_type, shape, gen)
      # Different testcases may differ only by runtime parameters but
      # share the same code. For example, dynamic-shapes testcases
      # share the same code involing tensor<?x?xf32> even though the runtime
      # value in the trace are different. That's why we call
      # generate_function conditionally, and generate_trace unconditionally.
      if func_name not in functions:
        functions[func_name] = generate_function(func_name, lhs_rhs_type,
                                                 accum_type, shape, gen)
      traces.append(
          generate_trace(func_name, lhs_rhs_type, accum_type, shape, gen))
  return (functions, traces)


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
                      choices=["small", "large"],
                      help="Collection of matrix shapes to test",
                      required=True)
  parser.add_argument(
      "--module_path",
      type=str,
      help=
      "Module path (typically .vmfb) to be referenced in the output trace. Should match the output path of the iree-translate command generating the module.",
      required=True)

  return parser.parse_args()


def write_code_file(functions, filename):
  with open(filename, "w") as file:
    for funcname in functions:
      file.write(functions[funcname] + "\n")


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


def main(args):
  (functions, traces) = generate(args)
  write_code_file(functions, args.output_code)
  write_trace_file(traces, args.output_trace, args.module_path)


if __name__ == "__main__":
  main(parse_arguments())
