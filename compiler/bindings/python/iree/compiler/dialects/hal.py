# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._hal_ops_gen import *
from ._hal_enum_gen import *


@register_attribute_builder("builtin.HAL_AccessScopeBitfieldAttr")
def _hal_accessscopebitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_BufferUsageBitfieldAttr")
def _hal_bufferusagebitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CallingConventionAttr")
def _hal_callingconventionattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CollectiveElementTypeAttr")
def _hal_collectiveelementtypeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CollectiveKindAttr")
def _hal_collectivekindattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CollectiveReductionOpAttr")
def _hal_collectivereductionopattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CommandBufferModeBitfieldAttr")
def _hal_commandbuffermodebitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_CommandCategoryBitfieldAttr")
def _hal_commandcategorybitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_DescriptorFlagsAttr")
def _hal_descriptorflagsattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_DispatchFlagsAttr")
def _hal_dispatchflagsattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(64, context=context), int(x))


@register_attribute_builder("builtin.HAL_ExecutionBarrierFlagBitfieldAttr")
def _hal_executionbarrierflagbitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_ExecutionStageBitfieldAttr")
def _hal_executionstagebitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_FenceFlagBitfieldAttr")
def _hal_fenceflagbitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_MemoryAccessBitfieldAttr")
def _hal_memoryaccessbitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_MemoryModelAttr")
def _hal_memorymodelattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_MemoryTypeBitfieldAttr")
def _hal_memorytypebitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.HAL_PipelineLayoutFlagsAttr")
def _hal_pipelinelayoutflagsattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.anonymous_538")
def _anonymous_538(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
