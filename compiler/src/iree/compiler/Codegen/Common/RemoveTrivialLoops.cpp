// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-remove-trivial-loops"

namespace mlir {
namespace iree_compiler {

/// Converts a symbolic GPU processor dimension to its numeric one.
static unsigned dimToIndex(gpu::Dimension dim) {
  switch (dim) {
    case gpu::Dimension::x:
      return 0;
    case gpu::Dimension::y:
      return 1;
    case gpu::Dimension::z:
      return 2;
    default:
      assert(false && "invalid dimension");
      return 0;
  }
}

/// If the value is a threadID return the range [0, workgroupSize-1].
/// If the number of workgroup is known also return the range of workgroupId ad
/// workgroupCount.
static std::optional<std::pair<AffineExpr, AffineExpr>> getWorkgroupRange(
    Value processorValue, SmallVectorImpl<Value> & /*dims*/,
    SmallVectorImpl<Value> & /*symbols*/, ArrayRef<int64_t> workgroupCount,
    ArrayRef<int64_t> workgroupSize) {
  if (auto idOp = processorValue.getDefiningOp<gpu::ThreadIdOp>()) {
    unsigned index = dimToIndex(idOp.getDimension());
    OpBuilder b(processorValue.getContext());
    AffineExpr zero = b.getAffineConstantExpr(0);
    AffineExpr ubExpr = b.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue.getDefiningOp<gpu::BlockDimOp>()) {
    OpBuilder builder(processorValue.getContext());
    unsigned index = dimToIndex(dimOp.getDimension());
    AffineExpr bound = builder.getAffineConstantExpr(workgroupSize[index]);
    return std::make_pair(bound, bound);
  }

  if (workgroupCount.empty()) return std::nullopt;

  if (auto idOp =
          processorValue.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = idOp.getDimension().getZExtValue();
    if (!workgroupCount[index]) return std::nullopt;

    AffineExpr zero = builder.getAffineConstantExpr(0);
    AffineExpr ubExpr = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(zero, ubExpr - 1);
  }
  if (auto dimOp = processorValue
                       .getDefiningOp<IREE::HAL::InterfaceWorkgroupCountOp>()) {
    OpBuilder builder(processorValue.getContext());

    // Can't infer the range when workroupCount is unknown.
    unsigned index = dimOp.getDimension().getZExtValue();
    if (!workgroupCount[index]) return std::nullopt;

    AffineExpr bound = builder.getAffineConstantExpr(workgroupCount[index]);
    return std::make_pair(bound, bound);
  }
  return std::nullopt;
}

/// Return true if the given tiled loop is distributed to workgroups.
static bool isWorkgroupLoop(const LoopTilingAndDistributionInfo &info) {
  auto forOp = cast<scf::ForOp>(info.loop);
  Operation *lbOp = forOp.getLowerBound().getDefiningOp();
  if (isa<IREE::HAL::InterfaceWorkgroupIDOp>(lbOp)) return true;
  auto applyOp = dyn_cast<AffineApplyOp>(lbOp);
  return applyOp && llvm::any_of(applyOp.getMapOperands(), [](Value operand) {
           return operand.getDefiningOp<IREE::HAL::InterfaceWorkgroupIDOp>();
         });
}

/// Infer the number of workgroups from exportOp.
static SmallVector<int64_t> getNumWorkgroup(
    func::FuncOp funcOp, IREE::HAL::ExecutableExportOp exportOp) {
  SmallVector<int64_t> result;

  Block *body = exportOp.getWorkgroupCountBody();
  if (!body) return result;

  auto returnOp = cast<IREE::HAL::ReturnOp>(body->getTerminator());
  assert(returnOp.getNumOperands() == 3);

  for (unsigned i = 0; i < 3; ++i) {
    Operation *defOp = returnOp.getOperand(i).getDefiningOp();
    if (auto indexOp = dyn_cast_or_null<arith::ConstantIndexOp>(defOp)) {
      result.push_back(indexOp.value());
    } else {
      return SmallVector<int64_t>();
    }
  }

  return result;
}

static LogicalResult removeOneTripTiledLoops(func::FuncOp funcOp,
                                             ArrayRef<int64_t> workgroupSize,
                                             ArrayRef<int64_t> numWorkgroups) {
  auto getWorkgroupRangeFn = [numWorkgroups, workgroupSize](
                                 Value processorValue,
                                 SmallVectorImpl<Value> &dims,
                                 SmallVectorImpl<Value> &symbols) {
    return getWorkgroupRange(processorValue, dims, symbols, numWorkgroups,
                             workgroupSize);
  };
  RewritePatternSet patterns(funcOp.getContext());
  populateRemoveSingleIterationLoopPattern(patterns, getWorkgroupRangeFn);
  return applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

namespace {

class RemoveSingleIterationLoopPass final
    : public RemoveSingleIterationLoopBase<RemoveSingleIterationLoopPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    FailureOr<IREE::HAL::ExecutableExportOp> exportOp = getEntryPoint(funcOp);
    if (failed(exportOp)) return;

    SmallVector<int64_t> workgroupSize = getWorkgroupSize(*exportOp);
    SmallVector<int64_t> numWorkgroups = getNumWorkgroup(funcOp, *exportOp);

    if (failed(removeOneTripTiledLoops(funcOp, workgroupSize, numWorkgroups))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass() {
  return std::make_unique<RemoveSingleIterationLoopPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
