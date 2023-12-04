"""IREE runtime Python bindings."""

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
# Io imports
from ._binding import (
    FileHandle,
    ParameterIndex,
    ParameterProvider,
    create_io_parameters_module,
)

# Hal imports
from ._binding import (
    BufferCompatibility,
    BufferUsage,
    HalAllocator,
    HalBuffer,
    HalBufferView,
    HalCommandBuffer,
    HalDevice,
    HalDriver,
    HalElementType,
    HalFence,
    HalSemaphore,
    MappedMemory,
    MemoryAccess,
    MemoryType,
    PyModuleInterface,
    Shape,
    create_hal_module,
)

# Vm imports
from ._binding import (
    Linkage,
    VmBuffer,
    VmVariantList,
    VmFunction,
    VmInstance,
    VmContext,
    VmModule,
)

from .array_interop import *
from .benchmark import *
from .system_api import *
from .system_setup import (
    get_device,
    get_first_device,
    get_driver,
    query_available_drivers,
)
from .function import *
from .tracing import *

from . import flags
