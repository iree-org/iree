// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_
#define IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {

// Registers dialects needed to query or construct const-expr information.
void registerConstExprDependentDialects(DialectRegistry &registry);

// Whether an op can be considered a pure expression, producing a constant if
// provided constants and having no side effects beyond that.
//
// In order to enable testing, some unregistered ops are also recognized:
//   - iree_unregistered.non_leaf_const_expr : Will be treated as const-expr.
//   - iree_unregistered.const_expr : Will be treated as const-expr
//   - iree_unregistered.var_expr : Will be treated as not const-expr
// Any other unregistered ops are treated as not const-expr.
bool isEligibleConstExprOp(Operation *op);

// Whether a const-expr op is eligible to be hoistable. This enforces
// policies for excluding certain, otherwise eligible, const-expr ops from
// being hoisted to a global.
//
// In order to enable testing, some unregistered ops are also recognized:
//   - iree_unregistered.non_leaf_const_expr : Will return false.
bool isHoistableConstExprLeaf(const ConstExprAnalysis::ConstValueInfo *info);

// Whether an operand which consumes a const-expr is eligible for hoisting.
// This is used to exclude certain operands that we never want in globals.
bool isHoistableConstExprConsumingOperand(OpOperand *operand);

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_
