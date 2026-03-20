// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_MAP_EXTERNALINTERFACES_VECTORLAYOUTINTERFACEIMPL_H_
#define IREE_COMPILER_CODEGEN_DIALECT_MAP_EXTERNALINTERFACES_VECTORLAYOUTINTERFACEIMPL_H_

#include "mlir/IR/DialectRegistry.h"

namespace mlir::iree_compiler::IREE::Map {

void registerVectorLayoutInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler::IREE::Map

#endif // IREE_COMPILER_CODEGEN_DIALECT_MAP_EXTERNALINTERFACES_VECTORLAYOUTINTERFACEIMPL_H_
