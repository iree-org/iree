# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Type hints."""

from typing import Callable, List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import HalBufferView

TraceKey = str
HalModuleBufferViewTraceCallback = Callable[[TraceKey, List["HalBufferView"]], None]
"""Tracing function for buffers to pass to the runtime.
This allows custom behavior when executing an IREE module with tensor tracing
instructions. MLIR e.g.

```
flow.tensor.trace "MyTensors" [
    %tensor1 : tensor<1xf32>,
    %tensor2 : tensor<2xf32>
]
```
"""
