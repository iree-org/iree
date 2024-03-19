// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

LogicalResult
IREE::LinalgExt::detail::verifyLinalgExtOpInterface(Operation *op) {
  // TODO(ravishankarm): Make `LinalgExt` Interface inherit from
  // `DestinationStyleOpInterface`
  LinalgExtOp linalgExtOp = cast<LinalgExtOp>(op);
  if (!isa<DestinationStyleOpInterface>(op)) {
    return linalgExtOp.emitOpError(
        "expected operation that implements LinalgExtInterface to also "
        "implement DestinationStyleOpInterface");
  }
  return success();
}

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.cpp.inc" // IWYU pragma: export

template <typename Ty, typename DimOpTy>
static void getDimValues(OpBuilder &b, Location loc, Value v, Ty t,
                         SmallVector<OpFoldResult> &dimVals) {
  for (auto [idx, val] : llvm::enumerate(t.getShape())) {
    if (ShapedType::isDynamic(val)) {
      dimVals.push_back(b.createOrFold<DimOpTy>(loc, v, idx));
    } else {
      dimVals.push_back(b.getIndexAttr(val));
    }
  }
}

LogicalResult LinalgExtOp::reifyResultShapes(
    OpBuilder &b, ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  Operation *op = getOperation();
  for (auto output : cast<DestinationStyleOpInterface>(op).getDpsInits()) {
    SmallVector<OpFoldResult> dims;
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
