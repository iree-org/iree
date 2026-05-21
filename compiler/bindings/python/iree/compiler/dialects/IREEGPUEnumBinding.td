// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Enum-only binding that excludes IREEGPUOps.td to avoid pulling in
// StructuredOpsUtils.td and its duplicate IteratorType builder registration.

#ifndef PYTHON_BINDINGS_IREEGPU_ENUMS
#define PYTHON_BINDINGS_IREEGPU_ENUMS

include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.td"

#endif // PYTHON_BINDINGS_IREEGPU_ENUMS
