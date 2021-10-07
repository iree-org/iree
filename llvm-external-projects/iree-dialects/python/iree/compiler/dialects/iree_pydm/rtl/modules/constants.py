# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Constants that are inlined into RTL modules."""

# TypeCode constants. These are mirrored from the BuiltinTypeCode enum on the
# C++ side (and must match or chaos will ensue).
TYPE_NONE = 1
TYPE_TUPLE = 2
TYPE_LIST = 3
TYPE_STR = 4
TYPE_BYTES = 5
TYPE_EXCEPTION_RESULT = 6
TYPE_TYPE = 7
TYPE_OBJECT = 0x100

# Numeric type exist in a range between 0x20 and 0x7f (inclusive).
TYPE_NUMERIC_START = 0x20
TYPE_NUMERIC_END = 0x7f

# To test if numeric, shift right by this amount and compare to
# TYPE_NUMERIC_SHITED_VALUE
TYPE_NUMERIC_SHIFT = 6
TYPE_NUMERIC_SHIFTED_VALUE = 1

# The lower 6 bits represent bit-packed numeric category and sub-type codes:
#   C C C C S S
TYPE_NUMERIC_MASK = 0x3f

# Mask of just the category bits.
TYPE_NUMERIC_CATEGORY_MASK = 0x3c
TYPE_NUMERIC_CATEGORY_SHIFT = 2
TYPE_NUMERIC_CATEGORY_BOOL = 0x0
TYPE_NUMERIC_CATEGORY_WEAK_INTEGER = 0x1
TYPE_NUMERIC_CATEGORY_UNSIGNED = 0x2
TYPE_NUMERIC_CATEGORY_SIGNED = 0x3
TYPE_NUMERIC_CATEGORY_APSIGNED = 0x4
TYPE_NUMERIC_CATEGORY_WEAK_REAL = 0x5
TYPE_NUMERIC_CATEGORY_REAL = 0x6
TYPE_NUMERIC_CATEGORY_WEAK_COMPLEX = 0x7
TYPE_NUMERIC_CATEGORY_COMPLEX = 0x8

# Mask of the sub-type bits.
TYPE_NUMERIC_SUBTYPE_MASK = 0x3
# Integer subtypes (applies to UNSIGNED and SIGNED categories).
TYPE_NUMERIC_SUBTYPE_INTEGER8 = 0x0
TYPE_NUMERIC_SUBTYPE_INTEGER16 = 0x1
TYPE_NUMERIC_SUBTYPE_INTEGER32 = 0x2
TYPE_NUMERIC_SUBTYPE_INTEGER64 = 0x3
# Real subtypes.
TYPE_NUMERIC_SUBTYPE_FP16 = 0x0
TYPE_NUMERIC_SUBTYPE_BF16 = 0x1
TYPE_NUMERIC_SUBTYPE_FP32 = 0x2
TYPE_NUMERIC_SUBTYPE_FP64 = 0x3
# Complex subtypes.
TYPE_NUMERIC_SUBTYPE_COMPLEX64 = 0x2
TYPE_NUMERIC_SUBTYPE_COMPLEX128 = 0x3
