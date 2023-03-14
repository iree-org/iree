// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fixup-subspan-with-offsets"

namespace mlir {
namespace iree_compiler {

namespace {
class FixupSubspanWithOffsetsPass
    : public FixupSubspanWithOffsetsBase<FixupSubspanWithOffsetsPass> {
 public:
  FixupSubspanWithOffsetsPass() = default;
  FixupSubspanWithOffsetsPass(const FixupSubspanWithOffsetsPass &) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

/// Returns true if the `subspanOp` has a return type of `MemRefType` and has
/// non-zero offsets.
static bool hasNonZeroOffsetAndMemRefType(
    IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  auto byteOffset = subspanOp.getByteOffset();
  if (!byteOffset || matchPattern(byteOffset, m_Zero())) {
    return false;
  }
  return subspanOp.getType().isa<MemRefType>();
}

/// Rewrites a `subspanOp` with `memref` return type and non-zero offsets
/// to a new `hal.interface.binding.subspan` op with the result `memref`
/// type representing the offset, making the type consistent. For example,
///
/// ```mlir
///  hal.interface.binding.subspan set(0) binding(0) offset(%o0)
///      : memref<?x?xf32>{%s0, %s1}
/// ```
///
/// is re-written to
///
/// ```mlir
///  hal.interface.binding.subspan set(0) binding(0)
///      : memref<?x?xf32, strided<[?, 1], offset: ?>{%s, s1}
/// ```
FailureOr<IREE::HAL::InterfaceBindingSubspanOp> rewriteSubspansWithOffset(
    RewriterBase &rewriter, IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  if (!hasNonZeroOffsetAndMemRefType(subspanOp)) return failure();

  LLVM_DEBUG({
    llvm::dbgs() << "Rewriting to zero offset : ";
    subspanOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  auto memRefType = subspanOp.getType().dyn_cast<MemRefType>();
  SmallVector<int64_t> strides;
  int64_t offset;
  if (failed(getStridesAndOffset(memRefType, strides, offset)) || offset != 0) {
    return subspanOp->emitOpError(
        "expected subspan result type to have canonical strides and zero "
        "offset");
  }

  // Compute the offset of the memref type.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(subspanOp);
  Location loc = subspanOp.getLoc();
  OpFoldResult elementOffset = convertByteOffsetToElementOffset(
      rewriter, loc, subspanOp.getByteOffset(), memRefType.getElementType());
  std::optional<int64_t> elementOffsetInt = getConstantIntValue(elementOffset);
  if (!elementOffsetInt) {
    elementOffsetInt = ShapedType::kDynamic;
  }

  // Create a new subspan op with an offset.
  auto layoutAttr = StridedLayoutAttr::get(rewriter.getContext(),
                                           elementOffsetInt.value(), strides);
  auto newMemRefType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      layoutAttr, memRefType.getMemorySpace());

  auto newOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
      loc, newMemRefType, subspanOp.getSetAttr(), subspanOp.getBindingAttr(),
      subspanOp.getDescriptorTypeAttr(), subspanOp.getByteOffset(),
      subspanOp.getDynamicDims(), subspanOp.getAlignmentAttr(),
      subspanOp.getDescriptorFlagsAttr());
  LLVM_DEBUG({
    llvm::dbgs() << "Rewritten to : ";
    newOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });
  return newOp;
}

/// Walks the function are rewrites all subspan pos that have non-zero offsets.
LogicalResult rewriteAllSubspanOpsWithOffsets(func::FuncOp funcOp) {
  // Collect all subspan ops with memref return types and non-zero offsets.
  SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    if (hasNonZeroOffsetAndMemRefType(subspanOp)) {
      subspanOps.push_back(subspanOp);
    }
  });

  IRRewriter rewriter(funcOp.getContext());
  for (auto subspanOp : subspanOps) {
    FailureOr<IREE::HAL::InterfaceBindingSubspanOp> newSubspanOp =
        rewriteSubspansWithOffset(rewriter, subspanOp);
    if (failed(newSubspanOp)) {
      return failure();
    }
    replaceMemrefUsesAndPropagateType(rewriter, subspanOp.getLoc(), subspanOp,
                                      newSubspanOp.value());
  }
  return success();
}

void FixupSubspanWithOffsetsPass::runOnOperation() {
  // For the resolution to work correctly, subspans with offsets need to be
  // handled appropriately.
  if (failed(rewriteAllSubspanOpsWithOffsets(getOperation()))) {
    getOperation()->emitOpError(
        "failed rewrite of subspan with non-zero offsets");
    return signalPassFailure();
  }

  if (!keepDeadSubspanOps) {
    // Remove any dead subspan operations.
    {
      RewritePatternSet patterns(&getContext());
      populateRemoveDeadMemAllocPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
createFixupSubspanWithOffsetsPass() {
  return std::make_unique<FixupSubspanWithOffsetsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
