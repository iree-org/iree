// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_
#define IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_

#include "iree/compiler/Dialect/Util/Analysis/Constant/ConstExpr.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "mlir/IR/Operation.h"

namespace mlir::iree_compiler::IREE::Util {

// Registers dialects needed to query or construct const-expr information.
void registerConstExprDependentDialects(DialectRegistry &registry);

// Information about a possible const-expr op.
struct ConstExprOpInfo {
  // Whether the op is eligible to be considered const-expr, assuming that
  // all of its producers are eligible.
  bool isEligible = false;

  // Producer values that must be const-expr for this op to be considered
  // const-expr. This minimally includes operands, and for region-based ops
  // may include implicit captures.
  llvm::SmallPtrSet<Value, 8> producers;

  // Gets information for an op.
  // Whether an op can be considered a pure expression, producing a constant if
  // provided constants and having no side effects beyond that.
  //
  // In order to enable testing, some unregistered ops are also recognized:
  //   - iree_unregistered.non_leaf_const_expr : Will be treated as const-expr.
  //   - iree_unregistered.const_expr : Will be treated as const-expr
  //   - iree_unregistered.var_expr : Will be treated as not const-expr
  // Any other unregistered ops are treated as not const-expr.
  static ConstExprOpInfo getForOp(Operation *op);
};

// Whether the type is considered legal for a constexpr root. For example,
// this would be called with the i32 type below:
//   %cst = arith.constant 4 : i32
bool isLegalConstExprRootType(Type t);

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

} // namespace mlir::iree_compiler::IREE::Util

#endif // IREE_COMPILER_DIALECT_IREE_UTIL_ANALYSIS_CONSTANT_OP_ORACLE_H_
