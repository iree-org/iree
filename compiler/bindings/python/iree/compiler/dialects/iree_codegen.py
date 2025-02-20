# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._iree_codegen_ops_gen import *
from ._iree_codegen_enum_gen import *
from .._mlir_libs._ireeCompilerDialects.iree_codegen import *


@register_attribute_builder("builtin.DispatchLoweringPassPipelineEnum")
def _dispatchloweringpasspipelineenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.WorkgroupIdEnum")
def _workgroupidenum(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))
