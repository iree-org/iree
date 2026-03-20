# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""IREE compiler Python bindings."""

# Re-export some legacy APIs from the tools package to this top-level.
# TODO: Deprecate and remove these names once clients are migrated.
from .tools import *

from iree.compiler import ir


# Resolves collisions for attribute builders defined in multiple TD files.
# TODO: Remove this once IREE integrates
# https://github.com/llvm/llvm-project/pull/187191.
def register_attribute_builder(kind, replace=True):
    def decorator_builder(func):
        ir.AttrBuilder.insert(kind, func, replace=replace)
        return func

    return decorator_builder


ir.register_attribute_builder = register_attribute_builder
