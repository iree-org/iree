// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE_H_
#define IREE_COMPILER_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE_H_

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h.inc"  // IWYU pragma: export
// clang-format on

namespace mlir::iree_compiler {

/// Register external models for PartitionableLoopsInterface.
void registerPartitionableLoopsInterfaceModels(DialectRegistry &registry);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_INTERFACES_PARTITIONABLE_LOOPS_INTERFACE_H_
