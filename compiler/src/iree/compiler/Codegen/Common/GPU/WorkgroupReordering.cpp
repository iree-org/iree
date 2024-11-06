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

static LogicalResult
reorderWorkgroupsInFunc(FunctionOpInterface funcOp,
                        ReorderWorkgroupsStrategy strategy) {
  assert(strategy != ReorderWorkgroupsStrategy::None &&
         "Expected a concrete strategy");

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
  assert(strategy == ReorderWorkgroupsStrategy::Transpose &&
         "Unhandled strategy");
  std::tie(newWorkgroupIdX, newWorkgroupIdY) =
      makeTransposedIds(funcOp.getLoc(), builder, workgroupIdX, workgroupIdY,
                        workgroupCntX, workgroupCntY);
  oldXId.replaceAllUsesWith(newWorkgroupIdX);
  oldYId.replaceAllUsesWith(newWorkgroupIdY);
  oldXId->erase();
  oldYId->erase();
  return success();
}

struct ReorderWorkgroupsPass final
    : impl::ReorderWorkgroupsPassBase<ReorderWorkgroupsPass> {
  ReorderWorkgroupsPass(
      ReorderWorkgroupsStrategy strategy,
      std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn)
      : reorderingStrategy(strategy), filterFn(std::move(filterFn)) {}

  LogicalResult initializeOptions(
      StringRef options,
      function_ref<LogicalResult(const Twine &)> errorHandler) override {
    if (failed(Pass::initializeOptions(options, errorHandler))) {
      return failure();
    }
    auto selectedStrategy =
        llvm::StringSwitch<FailureOr<ReorderWorkgroupsStrategy>>(strategy)
            .Case("", ReorderWorkgroupsStrategy::None)
            .Case("transpose", ReorderWorkgroupsStrategy::Transpose)
            .Default(failure());
    if (failed(selectedStrategy))
      return failure();

    reorderingStrategy = *selectedStrategy;
    return success();
  }

  void runOnOperation() override {
    if (reorderingStrategy == ReorderWorkgroupsStrategy::None)
      return;

    FunctionOpInterface funcOp = getOperation();
    if (filterFn && failed(filterFn(funcOp)))
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "--- Before reorder workgroups with workgroup counts ---";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    if (failed(reorderWorkgroupsInFunc(funcOp, reorderingStrategy))) {
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
  std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createReorderWorkgroups(
    ReorderWorkgroupsStrategy strategy,
    std::function<LogicalResult(mlir::FunctionOpInterface)> filterFn) {
  return std::make_unique<ReorderWorkgroupsPass>(strategy, filterFn);
}

} // namespace mlir::iree_compiler
