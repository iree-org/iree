// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_INPUT_OPS_H
#define IREE_DIALECTS_DIALECT_INPUT_OPS_H

#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

//===----------------------------------------------------------------------===//
// IREE::Input::TiedOpInterface
//===----------------------------------------------------------------------===//

namespace mlir::iree_compiler::IREE::Input {

// Forward declare
class TiedOpInterface;

namespace detail {

std::optional<unsigned> getTiedResultOperandIndex(Operation *op,
                                                  unsigned resultIndex);
void setTiedResultOperandIndex(Operation *op, unsigned resultIndex,
                               std::optional<unsigned> operandIndex);
SmallVector<int64_t> getTiedResultOperandIndices(Operation *op);
bool isOperandTied(Operation *tiedOp, unsigned operandIndex);
SmallVector<Value> getOperandTiedResults(Operation *op, unsigned operandIndex);
LogicalResult verifyTiedOp(TiedOpInterface tiedOp);

} // namespace detail
} // namespace mlir::iree_compiler::IREE::Input

//===----------------------------------------------------------------------===//

// Include generated interfaces code (this comment blocks clang-format from
// clobbering order).
#include "iree-dialects/Dialect/Input/InputOpInterfaces.h.inc"

#define GET_OP_CLASSES
#include "iree-dialects/Dialect/Input/InputOps.h.inc"

#endif // IREE_DIALECTS_DIALECT_INPUT_OPS_H
