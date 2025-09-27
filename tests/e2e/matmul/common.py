#!/usr/bin/env python3
# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""common utilities and data structures for e2e matmul tests"""

import enum
import dataclasses
import typing


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
    F8E4M3FN = "f8E4M3FN"
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


# Class attribute to store custom MNK values
ShapesId.custom_mnk_values = None


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
