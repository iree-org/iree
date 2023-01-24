// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-codegen-hoist-statically-bound-allocations"

namespace mlir {
namespace iree_compiler {

namespace {

struct HoistStaticallyBoundAllocationsPass
    : HoistStaticallyBoundAllocationsBase<HoistStaticallyBoundAllocationsPass> {
  void runOnOperation() override;
};

}  // namespace

static std::optional<Value> hoistStaticallyBoundAllocations(
    OpBuilder &builder, func::FuncOp funcOp, memref::AllocaOp allocaOp) {
  if (allocaOp->getBlock() == &funcOp.getBody().front()) return std::nullopt;

  auto allocaType = allocaOp.getType();
  ValueRange dynamicOperands = allocaOp.getDynamicSizes();
  if (dynamicOperands.empty()) {
    OpBuilder::InsertionGuard g(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().front());
    Value allocation = builder.create<memref::AllocaOp>(
        allocaOp.getLoc(), allocaType, allocaOp.getAlignmentAttr());
    return allocation;
  }

  int index = 0;

  SmallVector<int64_t> staticShape;
  SmallVector<OpFoldResult> subviewSizes;
  staticShape.reserve(allocaType.getRank());
  subviewSizes.reserve(allocaType.getRank());

  for (auto dimSize : allocaType.getShape()) {
    if (!ShapedType::isDynamic(dimSize)) {
      staticShape.push_back(dimSize);
      subviewSizes.push_back(builder.getIndexAttr(dimSize));
      continue;
    }
    Value dynamicSize = dynamicOperands[index++];
    auto ub = linalg::getConstantUpperBoundForIndex(dynamicSize);
    if (failed(ub)) {
      return std::nullopt;
    }
    staticShape.push_back(ub.value());
    subviewSizes.push_back(dynamicSize);
  }
  SmallVector<OpFoldResult> offsets(allocaType.getRank(),
                                    builder.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(allocaType.getRank(),
                                    builder.getIndexAttr(1));

  OpBuilder::InsertionGuard g(builder);
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  auto allocationType =
      MemRefType::get(staticShape, allocaType.getElementType());
  Value allocation = builder.create<memref::AllocaOp>(
      allocaOp.getLoc(), allocationType, allocaOp.getAlignmentAttr());

  builder.setInsertionPoint(allocaOp);
  Value subviewOp = builder.create<memref::SubViewOp>(
      allocaOp.getLoc(), allocation, offsets, subviewSizes, strides);
  return subviewOp;
}

static bool isUseReplacableWithSubview(OpOperand &use) {
  Operation *user = use.getOwner();
  if (isa<linalg::LinalgOp, memref::StoreOp, memref::SubViewOp>(user))
    return true;
  return false;
}

static LogicalResult replaceUseWith(OpOperand &use, Value replacement) {
  Operation *user = use.getOwner();
  auto doReplace = [&](OpOperand &replacedUse) { return &use == &replacedUse; };
  if (use.get().getType().cast<MemRefType>().hasStaticShape()) {
    use.get().replaceUsesWithIf(replacement, doReplace);
    return success();
  }
  if (isa<linalg::LinalgOp, memref::StoreOp, memref::SubViewOp>(user)) {
    use.get().replaceUsesWithIf(replacement, doReplace);
    return success();
  }
  return user->emitOpError("failed to replace operand ")
         << use.getOperandNumber() << " with " << replacement;
}

void HoistStaticallyBoundAllocationsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  SmallVector<memref::AllocaOp> allocaOps;
  funcOp.walk([&](memref::AllocaOp allocaOp) {
    if (allocaOp->getBlock() == &funcOp.getBody().front()) return;
    if (allocaOp.getDynamicSizes().empty()) {
      allocaOps.push_back(allocaOp);
      return;
    }
    if (llvm::all_of(allocaOp->getUses(), [](OpOperand &use) {
          return isUseReplacableWithSubview(use);
        })) {
      allocaOps.push_back(allocaOp);
      return;
    }
  });

  OpBuilder builder(&getContext());
  for (auto allocaOp : allocaOps) {
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Alloca Op : ";
      allocaOp->dump();
      int numUses = std::distance(allocaOp.getResult().use_begin(),
                                  allocaOp.getResult().use_end());
      llvm::dbgs() << " num Uses : " << numUses;
    });
    std::optional<Value> replacement =
        hoistStaticallyBoundAllocations(builder, funcOp, allocaOp);
    if (!replacement) continue;
    DEBUG_WITH_TYPE(DEBUG_TYPE, {
      llvm::dbgs() << "Replacement : ";
      replacement->dump();
    });
    Value replacementVal = replacement.value();
    allocaOp.getResult().replaceAllUsesWith(replacementVal);
    allocaOp->erase();
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
createHoistStaticallyBoundAllocationsPass() {
  return std::make_unique<HoistStaticallyBoundAllocationsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir