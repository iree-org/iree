// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

OpOperandVector::operator SmallVector<Value>() {
  SmallVector<Value> result;
  result.reserve(this->size());
  llvm::transform(*this, std::back_inserter(result),
                  [](OpOperand *opOperand) { return opOperand->get(); });
  return result;
}

LogicalResult
IREE::LinalgExt::detail::verifyLinalgExtOpInterface(Operation *op) {
  LinalgExtOp linalgExtOp = cast<LinalgExtOp>(op);
  if (op->getNumResults()) {
    if (!linalgExtOp.hasTensorSemantics()) {
      return linalgExtOp.emitOpError(
          "expected inputs and outputs to be RankedTensorType or scalar");
    }

    if (op->getNumResults() != linalgExtOp.outputs().size()) {
      return linalgExtOp.emitOpError(
          "expected number of outputs to be same as the number of results");
    }
    for (auto en : llvm::enumerate(op->getResultTypes())) {
      Type outputType = linalgExtOp.outputs()[en.index()].getType();
      if (en.value() != outputType) {
        return linalgExtOp.emitOpError("expected type of `outs` operand #")
               << en.index() << " " << outputType
               << " to be same as result type " << en.value();
      }
    }
  } else {
    if (!linalgExtOp.hasBufferSemantics()) {
      return linalgExtOp.emitOpError(
          "expected inputs and outputs to be MemRefType or scalar");
    }
  }
  return success();
}

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOpInterfaces.cpp.inc" // IWYU pragma: export

template <typename Ty, typename DimOpTy>
static void getDimValues(OpBuilder &b, Location loc, Value v, Ty t,
                         SmallVector<Value> &dimVals) {
  for (auto dim : llvm::enumerate(t.getShape())) {
    if (ShapedType::isDynamic(dim.value())) {
      dimVals.push_back(b.create<DimOpTy>(loc, v, dim.index()));
    } else {
      dimVals.push_back(b.create<arith::ConstantIndexOp>(loc, dim.value()));
    }
  }
}

LogicalResult LinalgExtOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Operation *op = getOperation();
  for (auto output : outputs()) {
    SmallVector<Value> dims;
    Type outputType = output.getType();
    if (auto rankedTensorType = outputType.dyn_cast<RankedTensorType>()) {
      getDimValues<RankedTensorType, tensor::DimOp>(b, op->getLoc(), output,
                                                    rankedTensorType, dims);
    } else if (auto memrefType = outputType.dyn_cast<MemRefType>()) {
      getDimValues<MemRefType, memref::DimOp>(b, op->getLoc(), output,
                                              memrefType, dims);
    } else if (!outputType.isIntOrIndexOrFloat()) {
      return op->emitOpError(
          "invalid type for output operand, expected tensor, "
          "memref or scalar type");
    }
    reifiedReturnShapes.emplace_back(std::move(dims));
  }
  return success();
}