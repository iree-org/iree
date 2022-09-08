//===------------------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorInterfaces.h"

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::TMTensor;

LogicalResult
mlir::torch::TMTensor::detail::verifyTMTensorOpInterface(Operation *op) {
  TMTensorOp mtTensorOp = cast<TMTensorOp>(op);
  if (op->getNumResults()) {
    if (!mtTensorOp.hasTensorSemantics()) {
      return mtTensorOp.emitOpError(
          "expected inputs and outputs to be RankedTensorType or scalar");
    }

    if (op->getNumResults() != mtTensorOp.getOutputs().size()) {
      return mtTensorOp.emitOpError(
          "expected number of outputs to be same as the number of results");
    }
    for (auto en : llvm::enumerate(op->getResultTypes())) {
      Type outputType = mtTensorOp.getOutputs()[en.index()].getType();
      if (en.value() != outputType) {
        return mtTensorOp.emitOpError("expected type of `outs` operand #")
               << en.index() << " " << outputType
               << " to be same as result type " << en.value();
      }
    }
  } else {
    if (!mtTensorOp.hasBufferSemantics()) {
      return mtTensorOp.emitOpError(
          "expected inputs and outputs to be MemRefType or scalar");
    }
  }
  return success();
}

#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOpInterfaces.cpp.inc" // IWYU pragma: export
