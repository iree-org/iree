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
//   %tileSizeY = affine.min ...
//   scf.for ...
//     %tileSizeX = affine.min ...
//     the_op with bounded tile sizes (The tensor is of dynamic shape.)
//
// into
//
// scf.for
//   %tileSizeY = affine.min ...
//   scf.for
//     %tileSizeX = affine.min ...
//     %cmp0 = arith.cmpi %worksizeY, %tilesizeY
//     %cmp1 = arith.cmpi %worksizeX, %tilesizeX
//     %cond = arith.and %cmp0, %cmp1
//     scf.if %cond
//       operation with the static shape with the main tile sizes
//     else
//       original nested loops with dynamic shaped op
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#define DEBUG_TYPE "iree-codegen-workgroup-specialization"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> clEnableWorkgroupSpecialization(
    "iree-codegen-enable-workgroup-specialization",
    llvm::cl::desc("Enable workgroup specialization."), llvm::cl::init(true));

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
      // check if all operands of the minOp are defined outside. This
      // guarantees that the bounded tile size op does not depend on
      // any value in the current loop body.
      bool definedOutisde =
          llvm::all_of(user->getOpOperands(), [&](OpOperand &opOperand) {
            Operation *op = opOperand.get().getDefiningOp();
            if (op) {
              return op->getBlock() != forOp.getBody();
            } else {
              // It is a block argument, so it is already outside of body.
              return true;
            }
          });

      if (!definedOutisde) continue;

      AffineMinOp minOp = cast<AffineMinOp>(user);
      minOps.push_back(cast<AffineMinOp>(minOp));
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
//   3. clone the body of the innermost loop in the else block.
//   4. clone the body of the else block into the then block and update the
//      bounded size ops with the constant tile size during the cloning.
static void specializeDistributionLoops(
    SmallVector<LoopTilingAndDistributionInfo> &infoVec) {
  // get the scf for loop nests and their tile size.
  SmallVector<scf::ForOp> distLoops;
  SmallVector<int64_t> tileSizes;
  SmallVector<AffineMinOp> minSizeOps;

  // Distribution info vector has the innermost loop at index 0.
  for (auto &info : infoVec) {
    auto forOp = cast<scf::ForOp>(info.loop);
    distLoops.push_back(forOp);
    if (!info.tileSize) {
      return;
    }
    int64_t tileSize = *info.tileSize;
    tileSizes.push_back(tileSize);
    SmallVector<AffineMinOp> boundedTileSizesOps =
        getBoundedTileSizeOps(forOp, tileSize);
    if (boundedTileSizesOps.empty()) {
      // When there is no bounded tile size op, it means that the tensor
      // size is already a multiple of tile size.
      minSizeOps.push_back(AffineMinOp());
    } else if (boundedTileSizesOps.size() != 1) {
      // Supposed to see only one op for the bounded tile size if any.
      return;
    } else {
      minSizeOps.push_back(boundedTileSizesOps[0]);
    }
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

  scf::ForOp innermostLoop = distLoops[0];
  auto loc = innermostLoop.getLoc();
  Block *block = innermostLoop.getBody();

  OpBuilder builder(innermostLoop->getContext());
  OpBuilder::InsertionGuard guard(builder);

  AffineMinOp minOp0 = minSizeOps[0];
  if (minOp0) {
    // Make sure the affine.min in the innermost loop is the first
    // instruction in the body.
    auto minOp = llvm::dyn_cast_or_null<AffineMinOp>(block->front());
    if (!minOp || minOp != minOp0) {
      minOp0->moveBefore(&block->front());
    }
    builder.setInsertionPointAfter(&block->front());
  } else {
    builder.setInsertionPoint(&block->front());
  }

  // create a condition for scf.if
  Value cond;
  SmallVector<Value> constantOps;  // ConstantIndexOps for tile sizes
  for (unsigned i = 0, e = distLoops.size(); i != e; ++i) {
    AffineMinOp minOp = minSizeOps[i];
    if (!minOp) {
      // add a dummy value to indicate that this loop does not have an
      // op for the bounded tile size.
      constantOps.push_back(Value());
      continue;
    }

    // Generate a compare op that checks the dynamic size is equal to the
    // constant main tile size.
    Value constant = builder.create<arith::ConstantIndexOp>(loc, tileSizes[i]);
    constantOps.push_back(constant);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              minOp, constant);
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, cmp) : cmp;
  }

  // generate scf.if %cond
  auto ifOp = builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/true);

  // Transfer the original body to the scf.else body.
  auto origBodyBegin = ++Block::iterator(ifOp);
  auto origBodyEnd = --block->end();  // yield

  Block *elseBlock = ifOp.elseBlock();
  elseBlock->getOperations().splice(elseBlock->begin(), block->getOperations(),
                                    origBodyBegin, origBodyEnd);
  // Clone the else block into the then block. minOps are replaced during the
  // cloning.
  auto b = ifOp.getThenBodyBuilder();
  IRMapping bvm;
  for (unsigned i = 0, e = minSizeOps.size(); i != e; ++i) {
    if (minSizeOps[i]) {
      bvm.map(minSizeOps[i], constantOps[i]);
    }
  }
  for (auto &blockOp : elseBlock->without_terminator()) {
    b.clone(blockOp, bvm);
  }
  return;
}

namespace {
struct WorkgroupSpecializationPass
    : public WorkgroupSpecializationBase<WorkgroupSpecializationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    if (!clEnableWorkgroupSpecialization) return;

    func::FuncOp funcOp = getOperation();
    SmallVector<LoopTilingAndDistributionInfo> infoVec =
        getTiledAndDistributedLoopInfo(funcOp);

    if (infoVec.empty()) return;

    specializeDistributionLoops(infoVec);
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass() {
  return std::make_unique<WorkgroupSpecializationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
