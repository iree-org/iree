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
//   %tileSizeY = affine.min ...
//   %tileSizeX = affine.min ...
//   the_op with bounded tile sizes (The tensor is of dynamic shape.)
//
// into
//
//   %tileSizeY = affine.min ...
//   %tileSizeX = affine.min ...
//   %cmp0 = arith.cmpi %worksizeY, %tilesizeY
//   %cmp1 = arith.cmpi %worksizeX, %tilesizeX
//   %cond = arith.and %cmp0, %cmp1
//   scf.if %cond
//     operation with the static shape with the main tile sizes
//   else
//     original nested loops with dynamic shaped op
//
//===---------------------------------------------------------------------===//
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define DEBUG_TYPE "iree-codegen-workgroup-specialization"

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> clEnableWorkgroupSpecialization(
    "iree-codegen-enable-workgroup-specialization",
    llvm::cl::desc("Enable workgroup specialization."), llvm::cl::init(true));

static std::optional<int64_t>
getConstantLowerBound(affine::AffineMinOp affineMinOp) {
  for (AffineExpr expr : affineMinOp.getMap().getResults()) {
    if (auto cst = dyn_cast<AffineConstantExpr>(expr)) {
      return cst.getValue();
    }
  }
  return std::nullopt;
}

// Specialize the distributed function with the main tile sizes.
//
// Transformed output
//   cond = (boundedTileSizeY != TileX) && (boundedTileSizeX != TileY) && ...
//   scf.if cond
//     distribution loops with static shapes with the tile size
//   else
//     distribution loops with dynamic shapes with the tile size
//
// Steps:
// 1. Walk the code and collect affine.min that only depend on workgroup.id
// and have one constant result.
// 2. Move those at the top of the function
// 3. Create a condition that ANDs all the affineMin == constant
// 4. Splice the rest of the block and clone into a specialized if/else
static void specializeFunction(func::FuncOp funcOp) {
  SmallVector<affine::AffineMinOp> minSizeOps;
  SmallVector<Operation *> ids;
  funcOp.walk([&minSizeOps, &ids](Operation *op) {
    if (auto affineMin = dyn_cast<affine::AffineMinOp>(op)) {
      for (Value operand : affineMin->getOperands()) {
        if (!operand.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>()) {
          return WalkResult::advance();
        }
        ids.push_back(operand.getDefiningOp());
      }
      if (!getConstantLowerBound(affineMin)) {
        return WalkResult::advance();
      }
      minSizeOps.push_back(affineMin);
    }
    return WalkResult::advance();
  });
  if (minSizeOps.empty()) {
    return;
  }

  auto loc = funcOp.getLoc();
  Block *block = &(*funcOp.getRegion().getBlocks().begin());

  OpBuilder builder(funcOp->getContext());
  OpBuilder::InsertionGuard guard(builder);
  // Move ops at the top of the function. This is always correct as those only
  // depends on workgroup ids.
  for (affine::AffineMinOp affineMin : llvm::reverse(minSizeOps)) {
    affineMin->moveBefore(&block->front());
  }
  for (Operation *id : llvm::reverse(ids)) {
    id->moveBefore(&block->front());
  }
  builder.setInsertionPointAfter(minSizeOps.back());
  // create a condition for scf.if
  Value cond;
  SmallVector<Value> constantOps; // ConstantIndexOps for tile sizes
  for (unsigned i = 0, e = minSizeOps.size(); i != e; ++i) {
    affine::AffineMinOp minOp = minSizeOps[i];
    int64_t lowerBound = *getConstantLowerBound(minOp);
    // Generate a compare op that checks the dynamic size is equal to the
    // constant main tile size.
    Value constant = builder.create<arith::ConstantIndexOp>(loc, lowerBound);
    constantOps.push_back(constant);
    Value cmp = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              minOp, constant);
    cond = cond ? builder.create<arith::AndIOp>(loc, cond, cmp) : cmp;
  }

  // generate scf.if %cond
  auto ifOp = builder.create<scf::IfOp>(loc, cond, /*withElseRegion=*/true);

  // Transfer the original body to the scf.else body.
  auto origBodyBegin = ++Block::iterator(ifOp);
  auto origBodyEnd = --block->end(); // yield

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
    registry.insert<affine::AffineDialect, linalg::LinalgDialect,
                    scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    if (!clEnableWorkgroupSpecialization)
      return;

    func::FuncOp funcOp = getOperation();
    specializeFunction(funcOp);
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createWorkgroupSpecializationPass() {
  return std::make_unique<WorkgroupSpecializationPass>();
}

} // namespace iree_compiler
} // namespace mlir
