#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""common utilities and data structures for e2e matmul tests"""

import enum
import dataclasses
import re
import typing


# Data type of matrix entries. The string values must match MLIR data types.
# This is a superset of the values accepted for the --lhs_rhs_types= flag,
# as this also includes accumulator-specific types like i32.
@enum.unique
class MatrixElemTypeId(enum.Enum):
    NONE = ""
    UI8 = "ui8"
    I8 = "i8"
    I32 = "i32"
    F64 = "f64"
    F32 = "f32"
    F16 = "f16"
    BF16 = "bf16"
    F8E5M2 = "f8E5M2"
    F8E4M3FN = "f8E4M3FN"
    F8E5M2FNUZ = "f8E5M2FNUZ"
    F8E4M3FNUZ = "f8E4M3FNUZ"
    F8E8M0FNU = "f8E8M0FNU"
    F6E3M2FN = "f6E3M2FN"
    F6E2M3FN = "f6E2M3FN"
    F4E2M1FN = "f4E2M1FN"


def get_size_in_bits(type_id: MatrixElemTypeId):
    return int(re.search(r"\d+", str(type_id)).group())


# Enumerates of the collections of shapes that we can generate tests for.
# The values are the accepted values for the --shapes= flag.
@enum.unique
class ShapesId(enum.Enum):
    DEFAULT = "default"
    SMALL = "small"
    LARGE = "large"
    EASY_LARGE_STATIC = "easy_large_static"
    CUSTOM_MNK = "custom_mnk"  # Used for custom shapes specified by --mnk= flag.

    @classmethod
    def set_custom_mnk(cls, shapes_id, mnk_string):
        """Parse and set custom MNK values from command line argument."""
        if shapes_id != cls.CUSTOM_MNK:
            if mnk_string:
                raise ValueError("--mnk can only be used with --shapes=custom_mnk")
            return

        # shapes_id is CUSTOM_MNK
        if not mnk_string:
            raise ValueError("--mnk must be specified when using --shapes=custom_mnk")
        try:
            mnk_parts = mnk_string.split(",")
            if len(mnk_parts) != 3:
                raise ValueError("--mnk must have exactly 3 values: m,n,k")
            cls.custom_mnk_values = tuple(int(x) for x in mnk_parts)
        except ValueError as e:
            raise ValueError(f"Invalid --mnk format: {e}")

    @classmethod
    def set_dynamicity(cls, shapes_id, dynamicity_string):
        """Parse and set MNK dynamicity from command line argument."""
        if shapes_id != cls.CUSTOM_MNK:
            if dynamicity_string:
                raise ValueError(
                    "--mnk_dynamicities can only be used with --shapes=custom_mnk"
                )
            return

        if dynamicity_string:
            try:
                dynamicity_parts = dynamicity_string.split(",")
                if len(dynamicity_parts) != 3:
                    raise ValueError(
                        "--mnk_dynamicities must have exactly 3 values, each being either 'dynamic' or 'static'"
                    )
                allowed_values = {"dynamic", "static"}
                for x in dynamicity_parts:
                    if x.lower() not in allowed_values:
                        raise ValueError(f"Invalid dynamicity value: {x}")

                cls.custom_dynamicity = tuple(
                    Dynamicity.DYNAMIC if x.lower() == "dynamic" else Dynamicity.STATIC
                    for x in dynamicity_parts
                )
            except ValueError as e:
                raise ValueError(f"Invalid --mnk_dynamicities format: {e}")


# Class attribute to store custom MNK values
ShapesId.custom_mnk_values = None
# Class attribute to store custom dynamicities for MNK
ShapesId.custom_dynamicity = None


# Returns the list of Dynamicity's to use for the collection of shapes
# identified by shapes_id.
def get_dynamicities(shapes_id: ShapesId):
    if shapes_id == ShapesId.EASY_LARGE_STATIC:
        return [(Dynamicity.STATIC, Dynamicity.STATIC, Dynamicity.STATIC)]
    elif shapes_id.custom_dynamicity:
        return [shapes_id.custom_dynamicity]
    else:
        return [
            (Dynamicity.DYNAMIC, Dynamicity.DYNAMIC, Dynamicity.DYNAMIC),
            (Dynamicity.STATIC, Dynamicity.STATIC, Dynamicity.STATIC),
        ]
    raise ValueError(shapes_id)


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
# util.func @somefunction_DYNxDYNxf32, where we can't use "?" characters.
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


# Represents a generated test function.
@dataclasses.dataclass
class MLIRFunction:
    name: str
    signature: str
    import_declaration: str
    definition: str


# Represents a call to a generated test function.
@dataclasses.dataclass
class TestCall:
    function: MLIRFunction
    op: str


# Helper for generate_function. Generates TestInputMatricesShapes, i.e.
# converts from the runtime shape dimensions in TestShape and given dynamicities(m,n,k) to
# the set of shapes to be used in a test function's input tensors.
def generate_shapes(
    shape: TestShape,
    transpose_rhs: bool,
    dynamicities: tuple[Dynamicity, Dynamicity, Dynamicity],
):
    dynamicity_m, dynamicity_n, dynamicity_k = dynamicities

    lhs_rows = shape_dim(shape.m, dynamicity_m)
    lhs_cols = shape_dim(shape.k, dynamicity_k)
    acc_rows = shape_dim(shape.m, dynamicity_m)
    acc_cols = shape_dim(shape.n, dynamicity_n)
    if transpose_rhs:
        rhs_rows = shape_dim(shape.n, dynamicity_n)
        rhs_cols = shape_dim(shape.k, dynamicity_k)
    else:
        rhs_rows = shape_dim(shape.k, dynamicity_k)
        rhs_cols = shape_dim(shape.n, dynamicity_n)
    shapes = TestInputMatricesShapes(
        lhs_rows=lhs_rows,
        lhs_cols=lhs_cols,
        rhs_rows=rhs_rows,
        rhs_cols=rhs_cols,
        acc_rows=acc_rows,
        acc_cols=acc_cols,
    )
    return shapes


random_matrix_seed = 0


# Generate a matrix function argument of the given size as `%name`.
def generate_random_matrix(
    name: str, matrix_shape: list, element_type: MatrixElemTypeId, increment_seed=True
):
    global random_matrix_seed
    if increment_seed:
        random_matrix_seed += 1
    return (
        f"  %{name}_dim0 = arith.constant {matrix_shape[0]} : i64\n"
        f"  %{name}_dim1 = arith.constant {matrix_shape[1]} : i64\n"
        f"  %{name}_element_type = hal.element_type<{element_type.value}> : i32\n"
        f"  %{name}_seed = arith.constant {random_matrix_seed} : i32\n"
        f"  %{name} = util.call @matmul_test.generate_random_matrix(%device, %{name}_dim0, %{name}_dim1, %{name}_element_type, %{name}_seed) : (!hal.device, i64, i64, i32, i32) -> !hal.buffer_view\n"
    )
