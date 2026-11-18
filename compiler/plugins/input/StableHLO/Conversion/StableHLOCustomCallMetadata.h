// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOCUSTOMCALLMETADATA_H_
#define IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOCUSTOMCALLMETADATA_H_

#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

// Returns true if |op| is a typed-FFI custom call with an empty or absent
// backend_config and no called computations, layouts, result tilings or
// discardable attributes. Side effects and aliases are left to the caller.
bool hasEmptyCustomCallMetadata(mlir::stablehlo::CustomCallOp op);

} // namespace mlir::iree_compiler::stablehlo

#endif // IREE_COMPILER_PLUGINS_INPUT_STABLEHLO_CONVERSION_STABLEHLOCUSTOMCALLMETADATA_H_
