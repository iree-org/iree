"""Module init for the python bindings."""

# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pylint: disable=g-multiple-import
# pylint: disable=g-bad-import-order
# pylint: disable=wildcard-import

from . import _binding

# Pull some of the native symbols into the public API.
# Hal imports
from ._binding import (
    BufferCompatibility,
    BufferUsage,
    HalAllocator,
    HalBuffer,
    HalBufferView,
    HalDevice,
    HalDriver,
    HalElementType,
    MemoryAccess,
    MemoryType,
    Shape,
)

# Vm imports
from ._binding import (
    create_hal_module,
    Linkage,
    VmVariantList,
    VmFunction,
    VmInstance,
    VmContext,
    VmModule,
)

from .array_interop import *
from .system_api import *
from .function import *
from .tracing import *

from . import flags
