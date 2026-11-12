# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Python dialect registration; operations currently use generic OpViews."""

from ._ods_common import _cext


@_cext.register_dialect
class _Dialect(_cext.ir.Dialect):
    DIALECT_NAMESPACE = "torch_c"
