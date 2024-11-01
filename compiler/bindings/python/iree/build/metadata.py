# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Common `BuildMeta` subclasses for built-in actions.

These are maintained here purely as an aid to avoiding circular dependencies.
Typically, in out of tree actions, they would just be inlined into the implementation
file.
"""

from .executor import BuildMeta


class CompileSourceMeta(BuildMeta):
    """CompileSourceMeta tracks source level properties that can influence compilation.

    This meta can be set on any dependency that ultimately is used as a source to a
    `compile` action.
    """

    # Slots in this case simply will catch attempts to set undefined attributes.
    __slots__ = [
        "input_type",
    ]
    KEY = "iree.compile.source"

    def __init__(self):
        super().__init__()

        # The value to the --iree-input-type= flag for this source file.
        self.input_type: str = "auto"

    def __repr__(self):
        return f"CompileSourceMeta(input_type={self.input_type})"
