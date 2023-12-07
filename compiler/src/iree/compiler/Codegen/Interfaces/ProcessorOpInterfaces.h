// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_PROCESSOROPINTERFACES_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_PROCESSOROPINTERFACES_H_

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

/// Include the generated interface declarations.
#include "iree/compiler/Codegen/Interfaces/ProcessorOpInterfaces.h.inc" // IWYU pragma: export

namespace mlir::iree_compiler {

/// Registers external models implemented for the `TiledOpInterface`.
void registerProcessorOpInterfaceExternalModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_INTERFACES_PROCESSOROPINTERFACES_H_
