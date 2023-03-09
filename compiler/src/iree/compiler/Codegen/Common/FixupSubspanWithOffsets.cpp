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

/// Replaces a `use` with the `replacement` for cases where a simple substition
/// might lead to verification errors.
static std::optional<SmallVector<Value>> replaceNonTrivialUse(
    RewriterBase &rewriter, Location loc, OpOperand &use, Value replacement) {
  Operation *user = use.getOwner();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(user);

  LLVM_DEBUG({
    llvm::dbgs() << "\tReplacing in user by creating new user : ";
    user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  if (auto castOp = dyn_cast<memref::CastOp>(user)) {
    auto replacementType = replacement.getType().cast<MemRefType>();
    auto currentResultType = castOp.getResult().getType().cast<MemRefType>();
    auto newResultType = MemRefType::get(
        currentResultType.getShape(), currentResultType.getElementType(),
        replacementType.getLayout(), replacementType.getMemorySpace());
    auto newCastOp =
        rewriter.create<memref::CastOp>(loc, newResultType, replacement);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newCastOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newCastOp->result_begin(),
                              newCastOp->result_end());
  }
  if (auto subviewOp = dyn_cast<memref::SubViewOp>(user)) {
    auto currResultType = subviewOp.getResult().getType().cast<MemRefType>();
    auto newSourceType = replacement.getType().cast<MemRefType>();
    SmallVector<OpFoldResult> offsets = subviewOp.getMixedOffsets();
    SmallVector<OpFoldResult> sizes = subviewOp.getMixedSizes();
    SmallVector<OpFoldResult> strides = subviewOp.getMixedStrides();
    MemRefType newResultType =
        (currResultType.getRank() != newSourceType.getRank()
             ? memref::SubViewOp::inferRankReducedResultType(
                   currResultType.getShape(), newSourceType, offsets, sizes,
                   strides)
                   .cast<MemRefType>()
             : nullptr);
    auto newSubviewOp = rewriter.create<memref::SubViewOp>(
        loc, newResultType, replacement, offsets, sizes, strides);

    LLVM_DEBUG({
      llvm::dbgs() << "\t\tNew user : ";
      newSubviewOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });
    return SmallVector<Value>(newSubviewOp->result_begin(),
                              newSubviewOp->result_end());
  }
  return std::nullopt;
}

/// Replace `origValue` with `replacementValue`. The replacement might
/// require replacing the users of `origValue`. The results of the users
/// themselves have to be replaced with results of the new users.
static void replaceUsesTransitively(RewriterBase &rewriter, Location loc,
                                    Value origValue, Value replacementValue) {
  SmallVector<std::pair<Value, Value>> worklist;
  worklist.push_back({origValue, replacementValue});

  while (!worklist.empty()) {
    auto [original, replacement] = worklist.pop_back_val();

    LLVM_DEBUG({
      llvm::dbgs() << "//===------------------------------------------===//\n";
      llvm::dbgs() << "Replacing : ";
      original.print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
      llvm::dbgs() << "\n";
    });

    llvm::SmallDenseSet<OpOperand *> preservedUses;
    for (OpOperand &use : original.getUses()) {
      Operation *user = use.getOwner();
      // Some uses cannot be replaced.
      if (isa<func::ReturnOp, scf::YieldOp>(user)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\tUnhandled user : ";
          user->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
          llvm::dbgs() << "\n";
        });
        preservedUses.insert(&use);
        continue;
      }

      // Some uses might be replace-able but require creating new versions
      // of the users to pass verification.
      std::optional<SmallVector<Value>> nonTrivialUse =
          replaceNonTrivialUse(rewriter, loc, use, replacement);
      if (nonTrivialUse) {
        // Add the results of the new users created as replacements
        // for the old users. Push this back on the to worklist.
        preservedUses.insert(&use);
        for (auto [v1, v2] :
             llvm::zip_equal(user->getResults(), nonTrivialUse.value())) {
          worklist.push_back({v1, v2});
        }
        continue;
      }
    }

    // Replace all non-preserved uses.
    rewriter.replaceUsesWithIf(original, replacement, [&](OpOperand &use) {
      if (!preservedUses.count(&use)) {
        LLVM_DEBUG({
          llvm::dbgs() << "\t\tReplacing use in :";
          use.getOwner()->print(llvm::dbgs(),
                                OpPrintingFlags().assumeVerified());
          llvm::dbgs() << "\n";
        });
        return true;
      }
      return false;
    });
  }
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
    replaceUsesTransitively(rewriter, subspanOp.getLoc(), subspanOp,
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
