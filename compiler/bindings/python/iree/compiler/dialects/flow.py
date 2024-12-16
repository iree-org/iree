# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._flow_ops_gen import *
from ._flow_enum_gen import *


@register_attribute_builder("builtin.FLOW_CollectiveElementTypeAttr")
def _flow_collectiveelementtypeattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.FLOW_CollectiveReductionOpAttr")
def _flow_collectivereductionopattr(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
