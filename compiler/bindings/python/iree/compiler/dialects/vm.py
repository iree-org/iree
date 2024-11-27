# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._vm_ops_gen import *
from ._vm_enum_gen import *


@register_attribute_builder("builtin.VM_CoreOpcodeAttr")
def _vm_coreopcodeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(8, context=context), int(x))


@register_attribute_builder("builtin.VM_ExtF32OpcodeAttr")
def _vm_extf32opcodeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(8, context=context), int(x))


@register_attribute_builder("builtin.VM_ExtF64OpcodeAttr")
def _vm_extf64opcodeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(8, context=context), int(x))
