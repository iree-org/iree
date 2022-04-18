// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "iree-tiled-op-interface"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;
using namespace IREE::LinalgExt;

#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.cpp.inc"

/// Converts an `OpFoldResult` to a `Value` by building a constant op if
/// if the `OpFoldResult` is an `IntegerAttr`.
static Value getValue(OpBuilder &builder, Location loc,
                      OpFoldResult valueOrAttr) {
  if (auto attr = valueOrAttr.dyn_cast<Attribute>()) {
    return builder.create<arith::ConstantIndexOp>(
        loc, attr.cast<IntegerAttr>().getInt());
  }
  return valueOrAttr.get<Value>();
}

//===----------------------------------------------------------------------===//
// Interface implementations for external operations.
//===----------------------------------------------------------------------===//

namespace {

/// Forwards the implementation of `TiledOpInterface` to upstream
/// `TilingInterface`. Note that this forwarding is only valid when the
/// iteration space is same as the data space of the result(s). This is due to
/// the difference in the tiling algorithm being developed around
/// `TilingInterface` and that used with `TiledOpInterface`. The difference
/// comes down to the former only needing the tiled operation, and not the value
/// of the whole tensor.
template <typename OpTy>
struct ForwardToTilingInterface
    : public TiledOpInterface::ExternalModel<ForwardToTilingInterface<OpTy>,
                                             OpTy> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    return cast<OpTy>(op).getDestinationOperands(b);
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    return cast<OpTy>(op).getLoopIteratorTypes();
  }
  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    return cast<OpTy>(op).getIterationDomain(b);
  }
  Operation *getTiledImplementation(Operation *op, OpBuilder &b,
                                    ValueRange dest,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> sizes,
                                    SmallVectorImpl<Value> &results) const {
    SmallVector<Operation *> tiledOps = cast<OpTy>(op).getTiledImplementation(
        b, dest, offsets, sizes, /*tileDestOperands=*/true);
    if (tiledOps.empty()) {
      op->emitOpError("failed to tile operation");
      return nullptr;
    }
    assert(tiledOps.size() == 1 && "expected single tiled op");
    Operation *tiledOp = tiledOps.front();
    if (tiledOp->getNumResults() != dest.size()) {
      op->emitOpError(
          "mismatch in the number of results of the tiled operation and the "
          "number of results expected");
      return nullptr;
    }
    Location loc = op->getLoc();
    auto oneAttr = b.getI64IntegerAttr(1);
    SmallVector<OpFoldResult> strides(offsets.size(), oneAttr);
    for (auto result : llvm::enumerate(tiledOp->getResults())) {
      // Assume that the shape of the result is same as the loop bounds of the
      // op. This implies the result can be inserted into the `dest` at
      // `offsets` and `sizes`. This would be illegal if that is not the
      // case. This is a point of difference between the `TiledOpInterface` in
      // IREE and `TilingInterface` in MLIR, since the latter sees fusion and
      // tiling as the same things. So it returns just the tiled op, and not the
      // result of the full tensor as the current tiling algorithm expects.
      auto tiledInsertOp = b.create<tensor::InsertSliceOp>(
          loc, result.value(), dest[result.index()], offsets, sizes, strides);
      results.push_back(tiledInsertOp);
    }
    return tiledOp;
  }
};

} // namespace

void IREE::LinalgExt::registerTiledOpInterfaceExternalModels(
    DialectRegistry &registry) {
  LLVM_DEBUG(
      { llvm::dbgs() << "Adding external models of tiled op interface\n"; });

  // TODO(ravishankarm): Needs custom PadTiledOpInterface or equiv.
  // registry.addOpInterface<tensor::PadOp,
  //                         ForwardToTilingInterface<tensor::PadOp>>();
}
