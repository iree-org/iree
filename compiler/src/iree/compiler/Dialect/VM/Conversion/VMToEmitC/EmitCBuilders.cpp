// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VM/Conversion/VMToEmitC/EmitCBuilders.h"

#include "llvm/ADT/ArrayRef.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace iree_compiler {
namespace emitc_builders {

Value structPtrMember(OpBuilder builder, Location location, Type type,
                      StringRef memberName, Value operand) {
  auto ctx = builder.getContext();
  return builder
      .create<emitc::CallOp>(
          /*location=*/location,
          /*type=*/type,
          /*callee=*/StringAttr::get(ctx, "EMITC_STRUCT_PTR_MEMBER"),
          /*args=*/
          ArrayAttr::get(ctx, {builder.getIndexAttr(0),
                               emitc::OpaqueAttr::get(ctx, memberName)}),
          /*templateArgs=*/ArrayAttr{},
          /*operands=*/ArrayRef<Value>{operand})
      .getResult(0);
}

}  // namespace emitc_builders
}  // namespace iree_compiler
}  // namespace mlir
