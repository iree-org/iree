// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"

#define DEBUG_TYPE "iree-codegen-gpu-tile-reduction"

namespace mlir {
namespace iree_compiler {

namespace {

static LogicalResult tileReduction(linalg::LinalgOp op) {
  SmallVector<unsigned> dims;
  op.getReductionDims(dims);
  SmallVector<int64_t> tileSize = getTileSizes(op, 1);
  if (tileSize.empty())
    return success();
  // Make sure reduction dimensions are the innermost ones.
  for (int i = 0; i < dims.size(); ++i) {
    if (dims[dims.size() - 1 - i] != op.getNumLoops() - 1 - i)
      return success();
  }
  IRRewriter rewriter(op.getContext());
  SmallVector<OpFoldResult> sizes;
  for (int64_t size : tileSize) {
    sizes.push_back(rewriter.getIndexAttr(size));
  }
  rewriter.setInsertionPoint(op);
  FailureOr<scf::SCFReductionTilingResult> results = scf::tileReductionUsingScf(
      rewriter, cast<PartialReductionOpInterface>(op.getOperation()), sizes);
  if (failed(results))
    return failure();
  return success();
}

static LogicalResult tileFusedOps(linalg::LinalgOp op) {
  IRRewriter rewriter(op.getContext());
  rewriter.setInsertionPoint(op);
  SmallVector<int64_t> tileSizes = getTileSizes(op, 1);
  if (tileSizes.empty())
    return success();
  linalg::LinalgTilingOptions tileOption;
  tileOption.setTileSizes(tileSizes);
  FailureOr<linalg::TiledLinalgOp> tiledOps =
      linalg::tileLinalgOp(rewriter, op, tileOption);
  if (failed(tiledOps))
    return failure();
  rewriter.replaceOp(op, tiledOps->tensorResults);
  return success();
}

struct GPUTileReductionPass
    : public GPUTileReductionBase<GPUTileReductionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    SmallVector<linalg::LinalgOp> linalgOps;
    funcOp.walk([&](linalg::LinalgOp op) { linalgOps.push_back(op); });
    for (linalg::LinalgOp op : linalgOps) {
      if (op.getNumReductionLoops() > 0) {
        if (failed(tileReduction(op))) {
          return signalPassFailure();
        }
      } else if (isa<linalg::GenericOp>(op)) {
        if (failed(tileFusedOps(op))) {
          return signalPassFailure();
        }
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createGPUTileReductionPass() {
  return std::make_unique<GPUTileReductionPass>();
}

} // namespace iree_compiler
} // namespace mlir
