#!/usr/bin/env python3
# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Generator for e2e fa2 tests.
"""

import argparse
import enum
import dataclasses
import typing
import math


# Data type of kernel entries. The string values must match MLIR data types.
@enum.unique
class QueryElemTypeId(enum.Enum):
    NONE = ""
    F32 = "f32"
    F16 = "f16"


# Data type of input entries. The string values must match MLIR data types.
@enum.unique
class KeyElemTypeId(enum.Enum):
    NONE = ""
    F32 = "f32"
    F16 = "f16"


# Data type of input entries. The string values must match MLIR data types.
@enum.unique
class ValueElemTypeId(enum.Enum):
    NONE = ""
    F32 = "f32"
    F16 = "f16"

# Data type of input entries. The string values must match MLIR data types.
@enum.unique
class ResultElemTypeId(enum.Enum):
    NONE = ""
    F32 = "f32"
    F16 = "f16"

# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


# Enumerates ways to construct MLIR tensor types.
@enum.unique
class Dynamicity(enum.Enum):
    STATIC = "static"  # Use fixed values everywhere. Example: tensor<4x6xf32>.


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

@dataclasses.dataclass
class TestShapeAndScale:
    batch: int
    numHeads: int
    seqLen: int
    headDim: int
    scale: float

# Returns the list of TestShape's to use for the collection of shapes
# identified by shapes_id.
def get_test_shapes(shapes_id: ShapesId):

    if shapes_id == ShapesId.SMALL:
        return [
            TestShapeAndScale(batch=4, numHeads=4, seqLen=1024, headDim=64, scale=1.0),
        ]
    if shapes_id == ShapesId.MEDIUM:
        return [
            TestShapeAndScale(batch=8, numHeads=4, seqLen=2048, headDim=128, scale=1.0),
        ]
    if shapes_id == ShapesId.LARGE:
        return [
            TestShapeAndScale(batch=16, numHeads=4, seqLen=4096, headDim=128, scale=1.0),
            TestShapeAndScale(batch=16, numHeads=4, seqLen=16384, headDim=64, scale=1.0),
        ]

    raise ValueError(shapes_id)


# Intentionally fixed seed! We want full reproducibility here, both across runs
# and across machines.
# Intentionally not shared with pseudorandom_generator_seed to limit the ways
# in which shuffling testcases changes which random values are generated.
local_pseudorandom_state = 1

# Determines the shape of input and kernel tensors.
@dataclasses.dataclass
class TestInputTensorShapes:
    batch: int
    numHeads: int
    seqLen: int
    headDim: int
    scale: float
# Helper for generate_function. Generates TestInputTensorShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicity to
# the set of shapes to be used in a test function's input tensors.
def generate_shapes_and_scale(shape: TestShapeAndScale):
    batch = shape.batch
    numHeads = shape.numHeads
    seqLen = shape.seqLen 
    headDim = shape.headDim
    scale = shape.scale

    shapes_scale = TestInputTensorShapes(
        batch=batch,
        numHeads=numHeads,
        seqLen=seqLen,
        headDim=headDim,
        scale=scale,
    )
    return shapes_scale


# Helper to return input, kernel and output shapes based on the layout and FA2Params.
def get_tensor_shapes(
    shapes_scale: TestShapeAndScale,
):
    batch = shapes_scale.batch
    numHeads = shapes_scale.numHeads
    seqLen = shapes_scale.seqLen
    headDim = shapes_scale.headDim
    scale = shapes_scale.scale
    # Extract input dimensions
    height, width = seqLen, headDim

    query_tensor_shape, key_tensor_shape, value_tensor_shape, result_tensor_shape = [], [], [], []

    query_tensor_shape = [batch, numHeads, seqLen, headDim]
    key_tensor_shape = [batch, numHeads, seqLen, headDim]
    value_tensor_shape = [batch, numHeads, seqLen, headDim]
    result_tensor_shape = [batch, numHeads, seqLen, headDim]

    return query_tensor_shape, key_tensor_shape, value_tensor_shape, result_tensor_shape


# Helper for generate_function.
# Generates a name for a test function in the generated MLIR code.
def generate_function_name(
    query_type: QueryElemTypeId,
    key_type: KeyElemTypeId,
    value_type: ValueElemTypeId,
    result_type: ResultElemTypeId,
    shapes_scale: TestInputTensorShapes,
):
    query_t = query_type.value
    key_t = key_type.value
    value_t = value_type.value
    result_t = result_type.value

    batch = shapes_scale.batch
    numHeads = shapes_scale.numHeads
    seqLen = shapes_scale.seqLen
    headDim = shapes_scale.headDim
    
    fa2 = "fa2"
    return (
        f"{fa2}_{batch}_{numHeads}_{seqLen}_{headDim}"
        + f"_dtype_{query_t}_{key_t}_{value_t}_{result_t}"
    )


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
    name: str
    signature: str
    import_declaration: str
    definition: str


# Generates a test function in the generated MLIR code.
# The generated function will take the same arguments as linalg.fa2 variants
# and will just call linalg.fa2 variants with them, returning its result.
def generate_function(
    query_type: QueryElemTypeId,
    key_type: KeyElemTypeId,
    value_type: ValueElemTypeId,
    result_type: ResultElemTypeId,
    shape_scale: TestShapeAndScale,
):
    shapes_scale = generate_shapes_and_scale(shape_scale)
    func_name = generate_function_name(
        query_type,
        key_type,
        value_type,
        result_type,
        shapes_scale,
    )

    query_shape, key_shape, value_shape, result_shape = get_tensor_shapes(
        shapes_scale
    )
    query_tensor_type = f"tensor<{query_shape[0]}x{query_shape[1]}x{query_shape[2]}x{query_shape[3]}x{query_type.value}>"
    key_tensor_type = f"tensor<{key_shape[0]}x{key_shape[1]}x{key_shape[2]}x{key_shape[3]}x{key_type.value}>"
    value_tensor_type = f"tensor<{value_shape[0]}x{value_shape[1]}x{value_shape[2]}x{value_shape[3]}x{value_type.value}>"
    result_tensor_type = f"tensor<{result_shape[0]}x{result_shape[1]}x{result_shape[2]}x{result_shape[3]}x{result_type.value}>"

    op_name = "iree_linalg_ext.attention"


    # Compilation info is optional; prints empty string by default.
    func_definition = ""

    signature = f"({query_tensor_type}, {key_tensor_type}, {value_tensor_type}, {result_tensor_type}) -> {result_tensor_type}"
    import_declaration = f"func.func private @module.{func_name}(%query: !hal.buffer_view, %key: !hal.buffer_view, %value: !hal.buffer_view, , %result: !hal.buffer_view) -> !hal.buffer_view"
    func_definition = func_definition + (
        f"func.func @{func_name}(%query: {query_tensor_type}, %key: {key_tensor_type}, %value: {value_tensor_type}) -> {result_tensor_type} {{\n"
        f"  %result = tensor.empty(): {result_tensor_type}\n"
        f"  %scale = arith.constant {shapes_scale.scale} : f16 \n"
        f"  %result = {op_name} ins(%query, %key, %value, %scale: {query_tensor_type}, {key_tensor_type}, {value_tensor_type}, f16) outs(%result: {result_tensor_type}) -> {result_tensor_type}\n"
        f"  return %result: {result_tensor_type}\n"
        f"}}\n"
    )

    return MLIRFunction(
        name=func_name,
        signature=signature,
        import_declaration=import_declaration,
        definition=func_definition,
    )


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
    element_type: typing.Union[QueryElemTypeId, ResultElemTypeId],
):
    global pseudorandom_generator_seed
    pseudorandom_generator_seed = pseudorandom_generator_seed + 1
    return (
        f"  %{'name'}_dim0 = arith.constant {tensor_shape[0]} : i64\n"
        f"  %{name}_dim1 = arith.constant {tensor_shape[1]} : i64\n"
        f"  %{name}_dim2 = arith.constant {tensor_shape[2]} : i64\n"
        f"  %{name}_dim3 = arith.constant {tensor_shape[3]} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {pseudorandom_generator_seed} : i32\n"
        f"  %{name} = call @fa2_test.generate_random_tensor(%device, %{name}_dim0, %{name}_dim1, %{name}_dim2, %{name}_dim3, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )
call_id = 0
def generate_call(
    function: MLIRFunction,
    query_type: QueryElemTypeId,
    key_type: KeyElemTypeId,
    value_type: ValueElemTypeId,
    result_type: ResultElemTypeId,
    shapes_scale: TestShapeAndScale,
):
    global call_id
    func_name = f"{function.name}_{shapes_scale.batch}_{shapes_scale.numHeads}_{shapes_scale.seqLen}_{shapes_scale.headDim}_{shapes_scale.scale}"
    func_name = f"{func_name}_{call_id}"
    call_id = call_id + 1

    description = f"FA2 shape (BATCHxNUMHEADSxSEQxHEAD): {shapes_scale.batch}x{shapes_scale.numHeads}x{shapes_scale.seqLen}x{shapes_scale.headDim}"
    op = (
        f"func.func @{func_name}() attributes {{\n"
        f'  iree.reflection = {{description = "{description}"}}\n'
        "} {\n"
        "  %device_index = arith.constant 0 : index\n"
        "  %device = hal.devices.get %device_index : !hal.device\n"
    )

    query_shape, key_shape, value_shape, result_shape = get_tensor_shapes(
        shapes_scale,
    )

    op = op + generate_random_4d_tensor("query", query_shape, query_type)
    op = op + generate_random_4d_tensor("key", key_shape, key_type)
    op = op + generate_random_4d_tensor("value", value_shape, value_type)
    op = op + generate_random_4d_tensor("result", result_shape, result_type)

    global pseudorandom_generator_seed
    pseudorandom_generator_seed = pseudorandom_generator_seed - 1
    op = op + generate_random_4d_tensor("result_copy", result_shape, result_type)
    op = op + (
        f"  %result = call @module.{function.name}(%query, %key, %value, %result) : (!hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> !hal.buffer_view\n"
    )

    op = op + (
        f"  %batch = arith.constant {shapes_scale.batch} : i64\n"
        f"  %numHeads = arith.constant {shapes_scale.numHeads} : i64\n"
        f"  %seqLen = arith.constant {shapes_scale.seqLen} : i64\n"
        f"  %headDim = arith.constant {shapes_scale.headDim} : i64\n"
        f"  call @fa2_test.check_fa2_results(%device, %batch, %numHeads, %seqLen, %headDim, %query, %key, %value, %result) : (!hal.device, i64, i64, i64, i64, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view, !hal.buffer_view) -> ()\n"
    )

    op = op + "  return\n"
    op = op + "}\n"

    return TestCall(function=function, op=op)


# Generates all output files' contents as strings.
def generate(
    query_type: QueryElemTypeId,
    key_type: KeyElemTypeId,
    value_type: ValueElemTypeId,
    result_type: ResultElemTypeId,
    shapes_id: ShapesId,
):
    functions = {}
    calls = []

    for shape in get_test_shapes(shapes_id):
            function = generate_function(
                query_type,
                key_type,
                value_type ,
                result_type,
                shape,
            )
            if function.name not in functions:
                functions[function.name] = function
            calls.append(
                generate_call(
                    function,
                    query_type,
                    key_type,
                    value_type,
                    result_type,
                    shape,
                )
            )

    return (functions, calls)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generator of e2e fa2 tests")
    parser.add_argument(
        "--output_fa2_mlir",
        type=str,
        help="Path of output .mlir file containing the generated fa2 functions",
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
        choices=["f32", "f16"],
        help="Numeric type of input tensors ",
        required=True,
    )
    parser.add_argument(
        "--result_type",
        type=str,
        choices=["f32", "f16"],
        help="Numeric type of result tensor",
        required=True,
    )
    parser.add_argument(
        "--shapes",
        type=str,
        choices=[s.value for s in ShapesId],
        help="Collection of tensor shapes to test",
        required=True,
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
        "func.func private @fa2_test.generate_random_tensor(%device: !hal.device, %dim0: i64, %dim1: i64, %dim2: i64, %dim3: i64, %element_type: i32, %seed: i32) -> !hal.buffer_view\n"
        "func.func private @fa2_test.check_fa2_results(%device: !hal.device, %batch: i64, %numHeads: i64, %seqLen: i64, %headDim: i64, %query: !hal.buffer_view, %key: !hal.buffer_view, %value: !hal.buffer_view, %result: !hal.buffer_view)\n"
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
    query_type = QueryElemTypeId(args.input_type)
    key_type = KeyElemTypeId(args.input_type)
    value_type = ValueElemTypeId(args.input_type)
    result_type = ResultElemTypeId(args.result_type)
    shapes_id = ShapesId(args.shapes)

    (functions, calls) = generate(
      query_type,
      key_type,
      value_type,
      result_type,
      shapes_id,
    )

    write_code_file(functions, args.output_fa2_mlir)
    write_calls_file(
        functions,
        calls,
        args.output_calls_mlir,
        args.requirements,
    )


if __name__ == "__main__":
    main(parse_arguments())
