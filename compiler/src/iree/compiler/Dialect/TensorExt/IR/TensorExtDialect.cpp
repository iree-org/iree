// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::IREE::TensorExt {

void IREETensorExtDialect::initialize() {
  addTypes<DispatchTensorType>();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.cpp.inc"
      >();

  getContext()->getOrLoadDialect<tensor::TensorDialect>();
}

Operation *IREETensorExtDialect::materializeConstant(OpBuilder &builder,
                                                     Attribute value, Type type,
                                                     Location loc) {
  if (arith::ConstantOp::isBuildableWith(value, type)) {
    return builder.create<arith::ConstantOp>(loc, type, cast<TypedAttr>(value));
  }
  return nullptr;
}

} // namespace mlir::iree_compiler::IREE::TensorExt

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"
