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
TYPE_BOOL = 8
TYPE_INTEGER = 9
TYPE_REAL = 0xa
TYPE_COMPLEX = 0xb
TYPE_OBJECT = 0x100
