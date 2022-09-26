// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== WorkgroupSpecializationPass.cpp ------------------------------------===//
//
// This pass specializes the workgroup distribution loops with the tile sizes.
//
// For example, it converts
//
// scf.for ...
//   %boundedTileSizeY = affine.min ...
//   scf.for ...
//     %boundedTileSizeX = affine.min ...
//     the_op with bounded sizes (The tensor is of dynamic shape.)
//
// into
//
// %cmp0 = arith.cmpi %worksizeY, %tilesizeY
// %cmp1 = arith.cmpi %worksizeX, %tilesizeX
// %cond = arith.and %cmp0, %cmp1
// scf.if %cond
//   scf.for
//     scf.for
//       operation with the static shape with the main tile sizes
// else
//   original nested loops with dynamic shaped op
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-codegen-workgroup-specialization"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> clEnableWorkgroupSpecialization(
    "iree-codegen-enable-workgroup-specialization",
    llvm::cl::desc("Enable workgroup specialization."),
    llvm::cl::init(false));

static bool isBoundedTileSizeOp(Operation *op, int64_t tileSize) {
  if (!isa<AffineMinOp>(op)) return false;

  AffineMinOp minOp = cast<AffineMinOp>(op);
  AffineMap map = minOp.getAffineMap();

  // The map is supposed to be the form of <(d0) -> (expr, TILE_SIZE)>
  if (map.getNumResults() != 2) return false;

  // Check either of the expr is the tile size. Check the RHS first since
  // a constant appears in RHS after normalization.
  if (map.getResult(1).isa<AffineConstantExpr>() &&
      map.getResult(1).cast<AffineConstantExpr>().getValue() == tileSize)
    return true;

  if (map.getResult(0).isa<AffineConstantExpr>() &&
      map.getResult(0).cast<AffineConstantExpr>().getValue() == tileSize)
    return true;

  return false;
}

static SmallVector<AffineMinOp> getBoundedTileSizeOps(scf::ForOp forOp,
                                                      int64_t tileSize) {
  SmallVector<AffineMinOp> minOps;

  for (Operation *user : forOp.getInductionVar().getUsers()) {
    if (isBoundedTileSizeOp(user, tileSize)) {
      minOps.push_back(cast<AffineMinOp>(user));
    }
  }

  return minOps;
}

// Specialize the tiled distribution loops with the main tile sizes.
//
// Input: A vector of LoopTilingAndDistributionInfo
//
// Transformed output
//   cond = (boundedTileSizeY != TileX) && (boundedTileSizeX != TileY) && ...
//   scf.if cond
//     distribution loops with static shapes with the tile size
//   else
//     distribution loops with dynamic shapes with the tile size
//
// Steps:
//   1. bail out if the loop bounds are already a multiple of the tile size.
//   2. create a condition for scf.if
//   3. clone the nested loops in the else block.
//   4. update the bounded size ops of the original loop nest with the constant
//      tile size.
//   5. clone the updated loop nest in the then block.
static void specializeDistributionLoops(
    SmallVector<LoopTilingAndDistributionInfo> &infoVec) {
  // get the scf for loop nests and their tile size.
  SmallVector<scf::ForOp> distLoops;
  SmallVector<int64_t> tileSizes;
  SmallVector<AffineMinOp> minSizeOps;

  // Distribution info vector has the innermost loop at index 0. Let's reverse
  // it to make the outermost loop positioned at index 0, and collect forOps,
  // tile sizes, and the bounded tile size ops.
  for (auto &info : reverse(infoVec)) {
    auto forOp = cast<scf::ForOp>(info.loop);
    distLoops.push_back(forOp);
    if (!info.tileSize) {
      return;
    }
    int64_t tileSize = *info.tileSize;
    tileSizes.push_back(tileSize);
    SmallVector<AffineMinOp> boundedTileSizesOps =
        getBoundedTileSizeOps(forOp, tileSize);
    // Supposed to see only one affine min for the bounded size.
    if (boundedTileSizesOps.size() != 1) {
      return;
    }
    minSizeOps.push_back(boundedTileSizesOps[0]);
  }

  // Check the eligilbility. Unsupported cases are:
  //   1. Dynamic cases
  //   2. UB < Tile size
  //   3. UB == a multiple of the tile size
  bool isAlreadyMultiple = true;
  for (unsigned i = 0, e = distLoops.size(); i != e; ++i) {
    scf::ForOp loop = distLoops[i];
    auto ubOp = loop.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    if (!ubOp) {
      // TODO: handle dynamic cases by adding more code to check the
      // conditions.
      return;
    }
    int64_t ub = ubOp.value();
    int64_t tileSize = tileSizes[i];
    if (ub % tileSize != 0) {
      isAlreadyMultiple = false;
    }
    if (ub < tileSize) {
      return;
    }
  }

  if (isAlreadyMultiple) return;

  scf::ForOp outermostLoop = distLoops[0];
  auto loc = outermostLoop.getLoc();
  OpBuilder builder(outermostLoop->getContext());
  PatternRewriter::InsertionGuard guard(builder);
  builder.setInsertionPoint(outermostLoop);

  // create a condition for scf.if
  Value cond;
  SmallVector<Value> constantOps;  // ConstantIndexOps for tile sizes
  for (unsigned i = 0, e = distLoops.size(); i != e; ++i) {
    int64_t tileSize = tileSizes[i];
    if (tileSize == 0 || tileSize == 1) {
      constantOps.push_back(Value());
      continue;
    }

    // clone the minSize op in the loop and place it before scf.if
    AffineMinOp minOp = minSizeOps[i];
    scf::ForOp distLoop = distLoops[i];
    // clone the lower bound and put before the nested loops.
    BlockAndValueMapping mapperForLB;
    Operation *lb =
        builder.clone(*distLoop.getLowerBound().getDefiningOp(), mapperForLB);

    // Clone the affine min op for the dynamic size in the current loop and
    // place it before the nested loops. The induction variable is replaced by
    // the cloned lower-bound above.
    BlockAndValueMapping mapperForIV;
    Value iv = distLoop.getInductionVar();
    mapperForIV.map(iv, lb->getResult(0));
    Value size =
        builder.clone(*minOp.getOperation(), mapperForIV)->getResult(0);

    // Generate a compare op that checks the dynamic size is equal to the
    // constant main tile size.
    Value constant = builder.create<arith::ConstantIndexOp>(loc, tileSize);
    constantOps.push_back(constant);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              size, constant);
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, cmp) : cmp;
  }

  // generate scf.if %cond then { scf.for ... } else { scf.for ... }
  auto ifOp = builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/true);
  ifOp.getElseBodyBuilder().clone(*outermostLoop.getOperation());

  // Specialize the size operations (affine min) in each for loop.
  for (int i = 0, e = distLoops.size(); i != e; ++i) {
    AffineMinOp minOp = minSizeOps[i];
    if (!minOp) {
      assert(tileSizes[i] == 0 || tileSizes[i] == 1);
      continue;
    }
    LLVM_DEBUG({
      llvm::errs() << "Replacing ";
      minOp.dump();
      llvm::errs() << "with ";
      constantOps[i].dump();
    });
    minOp.replaceAllUsesWith(constantOps[i]);
  }
  outermostLoop->moveBefore(&ifOp.getThenRegion().front().front());
  return;
}

namespace {
struct WorkgroupSpecializationPass
    : public WorkgroupSpecializationBase<WorkgroupSpecializationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    if (!clEnableWorkgroupSpecialization) return;

    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp innerModule = variantOp.getInnerModule();
    llvm::StringMap<IREE::HAL::ExecutableExportOp> entryPoints =
        getAllEntryPoints(innerModule);

    for (auto funcOp : innerModule.getOps<func::FuncOp>()) {
      SmallVector<LoopTilingAndDistributionInfo> infoVec =
          getTiledAndDistributedLoopInfo(funcOp);

      if (infoVec.empty()) return;

      specializeDistributionLoops(infoVec);
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createWorkgroupSpecializationPass() {
  return std::make_unique<WorkgroupSpecializationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
