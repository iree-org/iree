// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-reorder-workgroups"

namespace mlir::iree_compiler {

/// Changes the traversal order from left -> right then top -> bottom
/// to top -> down then left -> right.
static std::pair<Value, Value>
reorderIds(Location loc, OpBuilder b, Value workgroupIdX, Value workgroupIdY,
           ArrayRef<int64_t> workgroupCount, unsigned /*swizzleTile*/) {
  Value gridSizeX = b.create<arith::ConstantIndexOp>(loc, workgroupCount[0]);
  Value gridSizeY = b.create<arith::ConstantIndexOp>(loc, workgroupCount[1]);
  Value linearized = b.create<arith::MulIOp>(loc, workgroupIdY, gridSizeX);
  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
  Value newX = b.create<arith::DivUIOp>(loc, linearized, gridSizeY);
  Value newY = b.create<arith::RemUIOp>(loc, linearized, gridSizeY);
  return {newX, newY};
}

LogicalResult swizzleWorkgroupsInFunc(mlir::FunctionOpInterface funcOp,
                                      unsigned swizzleLogTile,
                                      ArrayRef<int64_t> workgroupCount) {
  assert(workgroupCount.size() == 3 && "Expected a 3D grid");
  if (swizzleLogTile == 0)
    return success();

  unsigned swizzleTile = 1u << swizzleLogTile;
  IREE::HAL::InterfaceWorkgroupIDOp oldXId;
  IREE::HAL::InterfaceWorkgroupIDOp oldYId;
  unsigned numXIdOps = 0;
  unsigned numYIdOps = 0;
  funcOp.walk([&](IREE::HAL::InterfaceWorkgroupIDOp idOp) {
    unsigned index = idOp.getDimension().getZExtValue();
    if (index == 0) {
      oldXId = idOp;
      ++numXIdOps;
    } else if (index == 1) {
      oldYId = idOp;
      ++numYIdOps;
    }
  });

  if (numXIdOps != 1 || numYIdOps != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Could not find X or Y\n");
    return failure();
  }

  OpBuilder builder(funcOp);
  builder.setInsertionPointToStart(&funcOp.front());
  // We create two new workgroup ID ops at the very top of the function and use
  // that to RAUW the old ones. This way we don't have to worry about the
  // picking the exact insertion points that do not violate dominance between
  // their defs and users.
  Value workgroupIdX =
      builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(funcOp.getLoc(), 0);
  Value workgroupIdY =
      builder.create<IREE::HAL::InterfaceWorkgroupIDOp>(funcOp.getLoc(), 1);
  auto [newX, newY] = reorderIds(funcOp.getLoc(), builder, workgroupIdX,
                                 workgroupIdY, workgroupCount, swizzleTile);
  oldXId.replaceAllUsesWith(newX);
  oldYId.replaceAllUsesWith(newY);
  oldXId->erase();
  oldYId->erase();
  return success();
}

namespace {
struct ReorderWorkgroupsPass final
    : ReorderWorkgroupsBase<ReorderWorkgroupsPass> {
  ReorderWorkgroupsPass(
      unsigned swizzleLogTile,
      std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn)
      : swizzleLogTile(swizzleLogTile), filterFn(std::move(filterFn)) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    swizzleLogTile = logTile;
    return success();
  }
  void runOnOperation() override {
    if (swizzleLogTile == 0)
      return;

    FunctionOpInterface funcOp = getOperation();
    if (filterFn && failed(filterFn(funcOp)))
      return;

    SmallVector<int64_t> workgroupCount = getStaticNumWorkgroups(funcOp);
    if (workgroupCount.size() != 3) {
      LLVM_DEBUG(llvm::dbgs() << "Reorder Workgroups: failed to find static "
                                 "workgroup counts. Bailing out.");
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- Before reorder workgroups with workgroup counts: [";
      llvm::interleaveComma(workgroupCount, llvm::dbgs());
      llvm::dbgs() << "] ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (failed(
            swizzleWorkgroupsInFunc(funcOp, swizzleLogTile, workgroupCount))) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to reorder workgroups\n");
      return;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After reorder workgroups ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }

private:
  unsigned swizzleLogTile;
  std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createReorderWorkgroups(
    unsigned swizzleLogTile,
    std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn) {
  return std::make_unique<ReorderWorkgroupsPass>(swizzleLogTile, filterFn);
}

} // namespace mlir::iree_compiler
