// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cassert>

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

#define GEN_PASS_DEF_REORDERWORKGROUPSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
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

// Reoredering to make workgroup ids move slowly between chiplet groups.

// Example:
// Currently, the GPU launches workgroups in a round-robin fashion across
// each XCD partition on the GPU.
// Assume we have 16 workgroups and XCDPartitionsOnGPU is 4.
// The default GPU schedule will launch workgroups {0, 1, 2, 3, ..., 15} in
// the following order:
// Partition 0: {0, 4, 8, 12}
// Partition 1: {1, 5, 9, 13}
// Partition 2: {2, 6, 10, 14}
// Partition 3: {3, 7, 11, 15}

// After reordering, the workgroup IDs are {0, 4, 8, 12, 1, ..., 15},
// resulting in the launch order:
// Partition 0: {0, 1, 2, 3}
// Partition 1: {4, 5, 6, 7}
// Partition 2: {8, 9, 10, 11}
// Partition 3: {12, 13, 14, 15}

// Returns permuted workgroup id (linearized ID).
// In the above example:
// linearedId 0's permuted Id is still 0.
// linearedId 1's permiuted Id is 4.
static Value chipletAwareWorkgroupReordering(Location loc, OpBuilder b,
                                             Value linearizedId,
                                             Value workgroupCountX,
                                             Value workgroupCountY,
                                             int64_t XCDParitionsOnGPU) {
  Value numChipletsVal =
      b.createOrFold<arith::ConstantIndexOp>(loc, XCDParitionsOnGPU);
  Value workgroupCount =
      b.create<arith::MulIOp>(loc, workgroupCountX, workgroupCountY);
  Value workgroupCountPerChiplet =
      b.create<arith::DivUIOp>(loc, workgroupCount, numChipletsVal);
  Value chipletId = b.create<arith::RemUIOp>(loc, linearizedId, numChipletsVal);
  Value wgIdWithinChiplet =
      b.create<arith::DivUIOp>(loc, linearizedId, numChipletsVal);
  Value reorderedId = b.create<arith::AddIOp>(
      loc, wgIdWithinChiplet,
      b.create<arith::MulIOp>(loc, chipletId, workgroupCountPerChiplet));

  // Handle the remainder part.
  Value constOne = b.createOrFold<arith::ConstantIndexOp>(loc, 1);
  Value lastWorkgroupId =
      b.create<arith::SubIOp>(loc, workgroupCount, constOne);
  Value modulatedLastWorkgroupId = b.create<arith::SubIOp>(
      loc, lastWorkgroupId,
      b.create<arith::RemUIOp>(loc, workgroupCount, numChipletsVal));
  Value isGreaterThanFinalWorkgroupId = b.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ugt, linearizedId, modulatedLastWorkgroupId);
  Value finalId = b.create<arith::SelectOp>(loc, isGreaterThanFinalWorkgroupId,
                                            linearizedId, reorderedId);

  return finalId;
}

// Chiplet-aware workgroup reordering strategy: reordering + super-grouping.
// Step 1: Reorder the workgroup grid to move slowly between
// chiplet groups (Function: chipletAwareWorkgroupReordering).
// Step 2: Implement 'super-grouping' of workgroups before switching to the next
// column.
static std::pair<Value, Value>
makeChipletGroupedIds(Location loc, OpBuilder b, Value workgroupIdX,
                      Value workgroupIdY, Value workgroupCountX,
                      Value workgroupCountY, unsigned chipletGroupTile,
                      unsigned numXCDs) {
  // Create one dimension ID for workgroup.
  Value linearized =
      b.create<arith::MulIOp>(loc, workgroupIdY, workgroupCountX);
  linearized = b.create<arith::AddIOp>(loc, linearized, workgroupIdX);

  assert(numXCDs > 1);
  // Map chiplets to perform a spatially local tile operation.
  // Reorder the linearized ID such that every consecutive group of chiplets
  // is the slowest-changing dimension in the grid.
  // Emphirically found that two chiplets as a group has better locality
  // throughout.
  linearized = chipletAwareWorkgroupReordering(
      loc, b, linearized, workgroupCountX, workgroupCountY, numXCDs / 2);

  // Detailed explanation about the idea behind the below implementation:
  // the L2 Cache Optimizations subsection in
  // https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#
  unsigned rowGroupSize = chipletGroupTile;
  Value rowGroupSizeVal =
      b.createOrFold<arith::ConstantIndexOp>(loc, rowGroupSize);

  // Emphirically, found rowGroupSize=16 for MI300X achieves good performance
  // group every 16 workgroups along Y dimension.

  // Number of workgroups in the group.
  Value numWorkGroupsPerRowBlock =
      b.create<arith::MulIOp>(loc, rowGroupSizeVal, workgroupCountX);

  Value groupId =
      b.create<arith::DivUIOp>(loc, linearized, numWorkGroupsPerRowBlock);
  Value firstRowID = b.create<arith::MulIOp>(loc, groupId, rowGroupSizeVal);

  Value currentRowGroupSize = b.create<arith::MinUIOp>(
      loc, b.create<arith::SubIOp>(loc, workgroupCountY, firstRowID),
      rowGroupSizeVal);

  Value newY = b.create<arith::AddIOp>(
      loc, firstRowID,
      b.create<arith::RemUIOp>(loc, linearized, currentRowGroupSize));

  Value newX = b.create<arith::DivUIOp>(
      loc, b.create<arith::RemUIOp>(loc, linearized, numWorkGroupsPerRowBlock),
      currentRowGroupSize);
  return {newX, newY};
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
                                             ReorderWorkgroupsStrategy strategy,
                                             unsigned logTile,
                                             unsigned numXCDs = 2) {
  assert(strategy != ReorderWorkgroupsStrategy::None &&
         "Expected a concrete strategy");

  unsigned reorderWgTileSize = 1u << logTile;
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
  if (strategy == ReorderWorkgroupsStrategy::Swizzle) {
    std::tie(newWorkgroupIdX, newWorkgroupIdY) =
        makeSwizzledIds(funcOp.getLoc(), builder, workgroupIdX, workgroupIdY,
                        workgroupCntX, workgroupCntY, reorderWgTileSize);
  } else if (strategy == ReorderWorkgroupsStrategy::ChipletGroup) {
    std::tie(newWorkgroupIdX, newWorkgroupIdY) = makeChipletGroupedIds(
        funcOp.getLoc(), builder, workgroupIdX, workgroupIdY, workgroupCntX,
        workgroupCntY, reorderWgTileSize, numXCDs);
  } else {
    assert(strategy == ReorderWorkgroupsStrategy::Transpose &&
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

} // namespace

LogicalResult swizzleWorkgroupsInFunc(FunctionOpInterface funcOp,
                                      unsigned swizzleLogTile) {
  // We have the same early exit in the pass but we need to duplicate it here
  // for the transform op.
  if (swizzleLogTile == 0)
    return success();

  return reorderWorkgroupsInFunc(funcOp, ReorderWorkgroupsStrategy::Swizzle,
                                 swizzleLogTile);
}

namespace {
struct ReorderWorkgroupsPass final
    : impl::ReorderWorkgroupsPassBase<ReorderWorkgroupsPass> {
  ReorderWorkgroupsPass(
      ReorderWorkgroupsStrategy strategy, unsigned logTile,
      std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn)
      : reorderingStrategy(strategy), reorderWgLogTileSize(logTile),
        filterFn(std::move(filterFn)) {}

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler))) {
      return failure();
    }

    auto selectedStrategy =
        llvm::StringSwitch<FailureOr<ReorderWorkgroupsStrategy>>(strategy)
            .Case("", ReorderWorkgroupsStrategy::None)
            .Case("chipletgroup", ReorderWorkgroupsStrategy::ChipletGroup)
            .Case("swizzle", ReorderWorkgroupsStrategy::Swizzle)
            .Case("transpose", ReorderWorkgroupsStrategy::Transpose)
            .Default(failure());
    if (failed(selectedStrategy))
      return failure();

    reorderingStrategy = *selectedStrategy;
    if (reorderingStrategy == ReorderWorkgroupsStrategy::Swizzle &&
        reorderWgLogTileSize == 0)
      reorderWgLogTileSize = logSwizzleTile;
    else if (reorderingStrategy == ReorderWorkgroupsStrategy::ChipletGroup &&
             reorderWgLogTileSize == 0)
      reorderWgLogTileSize = logChipletgroupTile;

    return success();
  }

  void runOnOperation() override {
    if (reorderingStrategy == ReorderWorkgroupsStrategy::None)
      return;

    if (reorderingStrategy == ReorderWorkgroupsStrategy::Swizzle &&
        reorderWgLogTileSize == 0)
      return;

    if (reorderingStrategy == ReorderWorkgroupsStrategy::ChipletGroup &&
        reorderWgLogTileSize == 0)
      return;

    FunctionOpInterface funcOp = getOperation();
    if (filterFn && failed(filterFn(funcOp)))
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "--- Before reorder workgroups with workgroup counts ---";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    uint32_t numXCDs = 1;
    if (IREE::HAL::ExecutableTargetAttr targetAttr =
            IREE::HAL::ExecutableTargetAttr::lookup(funcOp)) {
      if (DictionaryAttr config = targetAttr.getConfiguration()) {
        if (IREE::GPU::TargetAttr attr =
                config.getAs<IREE::GPU::TargetAttr>("iree.gpu.target")) {
          IREE::GPU::TargetChipAttr chipAttr = attr.getChip();
          if (chipAttr)
            numXCDs = chipAttr.getChipletCount();
        }
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "Number of XCDs = " << numXCDs << "\n");
    if (numXCDs == 1 &&
        reorderingStrategy == ReorderWorkgroupsStrategy::ChipletGroup)
      return;

    if (failed(reorderWorkgroupsInFunc(funcOp, reorderingStrategy,
                                       reorderWgLogTileSize, numXCDs))) {
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
  ReorderWorkgroupsStrategy reorderingStrategy =
      ReorderWorkgroupsStrategy::None;
  unsigned reorderWgLogTileSize = 0;
  std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createReorderWorkgroups(
    ReorderWorkgroupsStrategy strategy, unsigned reorderWgLogTile,
    std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn) {
  return std::make_unique<ReorderWorkgroupsPass>(strategy, reorderWgLogTile,
                                                 filterFn);
}

} // namespace mlir::iree_compiler
