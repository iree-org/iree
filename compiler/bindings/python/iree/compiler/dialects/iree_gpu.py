# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ..ir import IntegerAttr, IntegerType, register_attribute_builder
from ._iree_gpu_ops_gen import *
from ._iree_gpu_enum_gen import *
from .._mlir_libs._ireeCompilerDialects.iree_gpu import *


@register_attribute_builder("builtin.IREEGPU_ComputeBitwidths")
def _ireegpu_computebitwidths(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_DotProductOps")
def _ireegpu_dotproductops(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_MMAFragment")
def _ireegpu_mmafragment(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_MMAIntrinsic")
def _ireegpu_mmaintrinsic(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_MMAScope")
def _ireegpu_mmascope(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_ReorderWorkgroupsStrategy")
def _ireegpu_reorderworkgroupsstrategy(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_StorageBitwidths")
def _ireegpu_storagebitwidths(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_SubgroupOps")
def _ireegpu_subgroupops(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_TilingLevel")
def _ireegpu_tilinglevel(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))


@register_attribute_builder("builtin.IREEGPU_VirtualMMAIntrinsic")
def _ireegpu_virtualmmaintrinsic(x, context):
    return IntegerAttr.get(IntegerType.get_signless(32, context=context), int(x))
