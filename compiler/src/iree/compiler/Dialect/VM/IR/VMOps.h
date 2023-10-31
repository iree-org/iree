// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_VM_IR_VMOPS_H_
#define IREE_COMPILER_DIALECT_VM_IR_VMOPS_H_

#include <cstdint>
#include <numeric>

#include "iree/compiler/Dialect/Util/IR/UtilTraits.h"
#include "iree/compiler/Dialect/VM/IR/VMDialect.h"
#include "iree/compiler/Dialect/VM/IR/VMTraits.h"
#include "iree/compiler/Dialect/VM/IR/VMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace VM {

/// Generic method for verifying VM fail ops.
LogicalResult verifyFailOp(Operation *op, Value statusVal);

} // namespace VM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/VM/IR/VMOps.h.inc" // IWYU pragma: export

#endif // IREE_COMPILER_DIALECT_VM_IR_VMOPS_H_
