// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUSCOPEEXTERNALMODELS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUSCOPEEXTERNALMODELS_H_

#include "mlir/IR/DialectRegistry.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Registers external model implementations for PCF::ScopeAttrInterface on
/// GPU scope attributes (subgroup_scope and lane_scope).
void registerGPUScopeExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_EXTERNALINTERFACES_GPUSCOPEEXTERNALMODELS_H_
