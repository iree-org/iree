// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/CodegenPipelineOptions.h"

namespace mlir::iree_compiler {

// Out-of-line definition to anchor the vtable in this TU.
// See
// https://llvm.org/docs/CodingStandards.html#provide-a-virtual-method-anchor-for-classes-in-headers.
CodegenPipelineOptions::~CodegenPipelineOptions() = default;

} // namespace mlir::iree_compiler
