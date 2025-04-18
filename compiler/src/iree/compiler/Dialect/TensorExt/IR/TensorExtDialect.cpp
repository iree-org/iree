// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
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

} // namespace mlir::iree_compiler::IREE::TensorExt

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.cpp.inc"
