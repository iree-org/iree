// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/StableHLO/Conversion/StableHLOStatusCheck.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir::iree_compiler::stablehlo {

void emitStatusCheck(OpBuilder &builder, Location loc, Value condition,
                     StringRef message) {
  Value ok = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value failed = arith::ConstantIntOp::create(
      builder, loc,
      static_cast<int32_t>(IREE::Util::StatusCode::FailedPrecondition), 32);
  Value status = arith::SelectOp::create(builder, loc, condition, ok, failed);
  IREE::Util::StatusCheckOkOp::create(builder, loc, status, message);
}

} // namespace mlir::iree_compiler::stablehlo
