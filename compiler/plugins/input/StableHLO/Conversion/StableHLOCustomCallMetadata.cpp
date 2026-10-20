// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOCustomCallMetadata.h"

namespace mlir::iree_compiler::stablehlo {

bool hasEmptyCustomCallMetadata(mlir::stablehlo::CustomCallOp op) {
  return op.getApiVersion() ==
             mlir::stablehlo::CustomCallApiVersion::API_VERSION_TYPED_FFI &&
         op.hasEmptyBackendConfig() && op.getCalledComputations().empty() &&
         (!op.getOperandLayouts() || op.getOperandLayouts()->empty()) &&
         (!op.getResultLayouts() || op.getResultLayouts()->empty()) &&
         (!op.getResultTilings() || op.getResultTilings()->empty()) &&
         op->getDiscardableAttrs().empty();
}

} // namespace mlir::iree_compiler::stablehlo
