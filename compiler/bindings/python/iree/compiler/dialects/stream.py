# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._stream_ops_gen import *
from ._stream_enum_gen import *


@register_attribute_builder("builtin.Stream_CollectiveElementTypeAttr")
def _stream_collectiveelementtypeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_CollectiveKindAttr")
def _stream_collectivekindattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_CollectiveReductionOpAttr")
def _stream_collectivereductionopattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_FavorAttr")
def _stream_favorattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_LifetimeAttr")
def _stream_lifetimeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_MemoryModelAttr")
def _stream_memorymodelattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.Stream_ResourceAccessBitfieldAttr")
def _stream_resourceaccessbitfieldattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
