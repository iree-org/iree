// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>
#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#define DEBUG_TYPE "iree-codegen-reorder-workgroups"

namespace mlir::iree_compiler {

/// Implements the following swizzling logic:
/// void getTiledId2(unsigned x, unsigned y, unsigned* tiledx,
///                  unsigned* tiledy) {
///  unsigned t_tiledx = (x + (y % tile) * grid_size_x) / tile;
///  unsigned t_tiledy = (y / tile) * tile +
///      (x + (y % tile) * grid_size_x) % tile;
///  bool c = grid_size_y % tile != 0 &&
///      ((y / tile) * tile + tile) > grid_size_y;
///  *tiledx = c ? x : t_tiledx;
///  *tiledy = c ? y : t_tiledy;
/// }
static std::pair<Value, Value>
makeSwizzledIds(Location loc, OpBuilder b, Value workgroupIdX,
                Value workgroupIdY, Value workgroupCountX,
                Value workgroupCountY, unsigned swizzleTile) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value tile = b.create<arith::ConstantIndexOp>(loc, swizzleTile);
  Value yModTile = b.create<arith::RemUIOp>(loc, workgroupIdY, tile);
  Value yDivTile = b.create<arith::DivUIOp>(loc, workgroupIdY, tile);
  Value swizzleParam = b.create<arith::MulIOp>(loc, yModTile, workgroupCountX);
  Value swizzleParam2 =
      b.create<arith::AddIOp>(loc, workgroupIdX, swizzleParam);
  Value swizzleParam3 = b.create<arith::RemUIOp>(loc, swizzleParam2, tile);
  Value swizzleParam4 = b.create<arith::MulIOp>(loc, yDivTile, tile);
  Value unboundedSwizzledIdX =
      b.create<arith::DivUIOp>(loc, swizzleParam2, tile);
  Value unboundedSwizzledIdY =
      b.create<arith::AddIOp>(loc, swizzleParam3, swizzleParam4);
  Value gyModTile = b.create<arith::RemUIOp>(loc, workgroupCountY, tile);
  Value gyAddTile = b.create<arith::AddIOp>(loc, swizzleParam4, tile);
  Value condition1 =
      b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, gyModTile, zero);
  Value condition2 = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ugt,
                                             gyAddTile, workgroupCountY);
  Value condition3 = b.create<arith::AndIOp>(loc, condition1, condition2);
  Value swizzledIdX = b.create<arith::SelectOp>(loc, condition3, workgroupIdX,
                                                unboundedSwizzledIdX);
  Value swizzledIdY = b.create<arith::SelectOp>(loc, condition3, workgroupIdY,
                                                unboundedSwizzledIdY);
  return {swizzledIdX, swizzledIdY};
}

/// Transpose IDs, i.e., changes the traversal order from left -> right then
/// top -> bottom to top -> bottom then left -> right.
static std::pair<Value, Value> makeTransposedIds(Location loc, OpBuilder b,
                                                 Value workgroupIdX,
                                                 Value workgroupIdY,
                                                 Value workgroupCountX,
                                                 Value workgroupCountY) {
  Value linearized =
      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);
  Value newX = b.create<arith::DivUIOp>(loc, linearized, workgroupCountY);
  Value newY = b.create<arith::RemUIOp>(loc, linearized, workgroupCountY);
  return {newX, newY};
}

/// Returns the workgroup counts along the X and Y dimensions. These will be
/// constants when static in the corresponding `hal.executable.export` op.
static std::pair<Value, Value>
getWorkgroupCountsXY(OpBuilder &builder, FunctionOpInterface funcOp) {
  Location loc = funcOp.getLoc();
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  bool isStaticWgCount = llvm::none_of(workgroupCounts, ShapedType::isDynamic);
  // Check if we can rely on a static grid.
  if (isStaticWgCount && workgroupCounts.size() >= 2) {
    LLVM_DEBUG(llvm::dbgs()
               << "Using static workgroup counts: X = " << workgroupCounts[0]
               << ", Y = " << workgroupCounts[1] << "\n");
    Value workgroupCountX =
        builder.create<arith::ConstantIndexOp>(loc, workgroupCounts[0]);
    Value workgroupCountY =
        builder.create<arith::ConstantIndexOp>(loc, workgroupCounts[1]);
    return {workgroupCountX, workgroupCountY};
  }

  LLVM_DEBUG(llvm::dbgs() << "Using dynamic workgroup counts\n");
  Value dynamicCountX =
      builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, 0);
  Value dynamicCountY =
      builder.create<IREE::HAL::InterfaceWorkgroupCountOp>(loc, 1);
  return {dynamicCountX, dynamicCountY};
}

static LogicalResult reorderWorkgroupsInFunc(FunctionOpInterface funcOp,
                                             ReorderWorkgrupsStrategy strategy,
                                             unsigned swizzleLogTile) {
  assert(strategy != ReorderWorkgrupsStrategy::None &&
         "Expected a concrete strategy");

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
  auto [workgroupCntX, workgroupCntY] = getWorkgroupCountsXY(builder, funcOp);
  Value newWorkgroupIdX;
  Value newWorkgroupIdY;
  if (strategy == ReorderWorkgrupsStrategy::Swizzle) {
    std::tie(newWorkgroupIdX, newWorkgroupIdY) =
        makeSwizzledIds(funcOp.getLoc(), builder, workgroupIdX, workgroupIdY,
                        workgroupCntX, workgroupCntY, swizzleTile);
  } else {
    assert(strategy == ReorderWorkgrupsStrategy::Transpose &&
           "Unhandled strategy");
    std::tie(newWorkgroupIdX, newWorkgroupIdY) =
        makeTransposedIds(funcOp.getLoc(), builder, workgroupIdX, workgroupIdY,
                          workgroupCntX, workgroupCntY);
  }

  oldXId.replaceAllUsesWith(newWorkgroupIdX);
  oldYId.replaceAllUsesWith(newWorkgroupIdY);
  oldXId->erase();
  oldYId->erase();
  return success();
}

LogicalResult swizzleWorkgroupsInFunc(FunctionOpInterface funcOp,
                                      unsigned swizzleLogTile) {
  // We have the same early exit in the pass but we need to duplicate it here
  // for the transform op.
  if (swizzleLogTile == 0)
    return success();

  return reorderWorkgroupsInFunc(funcOp, ReorderWorkgrupsStrategy::Swizzle,
                                 swizzleLogTile);
}

namespace {
struct ReorderWorkgroupsPass final
    : ReorderWorkgroupsBase<ReorderWorkgroupsPass> {
  ReorderWorkgroupsPass(
      ReorderWorkgrupsStrategy strategy, unsigned logSwizzleTile,
      std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn)
      : reorderingStrategy(strategy), logSwizzleTile(logSwizzleTile),
        filterFn(std::move(filterFn)) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }
  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler))) {
      return failure();
    }
    logSwizzleTile = logTile;
    auto selectedStrategy =
        llvm::StringSwitch<FailureOr<ReorderWorkgrupsStrategy>>(strategy)
            .Case("", ReorderWorkgrupsStrategy::None)
            .Case("swizzle", ReorderWorkgrupsStrategy::Swizzle)
            .Case("transpose", ReorderWorkgrupsStrategy::Transpose)
            .Default(failure());
    if (failed(selectedStrategy))
      return failure();

    reorderingStrategy = *selectedStrategy;
    return success();
  }

  void runOnOperation() override {
    if (reorderingStrategy == ReorderWorkgrupsStrategy::None)
      return;

    if (reorderingStrategy == ReorderWorkgrupsStrategy::Swizzle &&
        logSwizzleTile == 0)
      return;

    FunctionOpInterface funcOp = getOperation();
    if (filterFn && failed(filterFn(funcOp)))
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "--- Before reorder workgroups with workgroup counts ---";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (failed(reorderWorkgroupsInFunc(funcOp, reorderingStrategy, logTile))) {
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
  ReorderWorkgrupsStrategy reorderingStrategy = ReorderWorkgrupsStrategy::None;
  unsigned logSwizzleTile = 0;
  std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createReorderWorkgroups(
    ReorderWorkgrupsStrategy strategy, unsigned swizzleLogTile,
    std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn) {
  return std::make_unique<ReorderWorkgroupsPass>(strategy, swizzleLogTile,
                                                 filterFn);
}

} // namespace mlir::iree_compiler
