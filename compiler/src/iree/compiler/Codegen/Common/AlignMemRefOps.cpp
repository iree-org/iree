// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric> // for std::gcd
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "Passes.h"
namespace mlir {
#define GEN_PASS_DEF_ALIGNMEMREFOPSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

unsigned getElementTypeSize(DataLayout datalayout, MemRefType memRefType) {
  Type elemTy = memRefType.getElementType();
  auto elemTySize = datalayout.getTypeSize(elemTy);
  auto elemTySizeInBytes = elemTySize.getFixedValue();
  return elemTySizeInBytes;
}

unsigned int gcd(const SmallVectorImpl<unsigned int> &coefficients) {
  assert(!coefficients.empty() && "coefficients must not be empty");
  auto i = coefficients.begin();
  unsigned int GCD = *i;
  i++;
  auto e = coefficients.end();
  while (i != e) {
    GCD = std::gcd(GCD, *i);
    i++;
  }
  return GCD;
}

unsigned int getGreatestPowerOfTwoDivisor(unsigned int value) {
  return value & -value;
}

unsigned int
getStaticAlignmentGuarantee(const SmallVectorImpl<unsigned int> &coefficients) {
  assert(!coefficients.empty() && "coefficients must not be empty");
  return getGreatestPowerOfTwoDivisor(gcd(coefficients));
}

unsigned int
getStaticAlignmentGuarantee(unsigned int sourceAlignment,
                           SmallVectorImpl<unsigned int> &coefficients) {
  coefficients.push_back(sourceAlignment);
  return getStaticAlignmentGuarantee(coefficients);
}

FailureOr<unsigned int> getStaticAlignmentGuarantee(vector::LoadOp op,
                                                    RewriterBase &rewriter) {
  if (op.getAlignment()) {
    return rewriter.notifyMatchFailure(op, "already aligned");
  }

  auto base = op.getBase();
  auto definition = base.getDefiningOp();
  if (!definition) {
    return rewriter.notifyMatchFailure(op, "no known definition for operation");
  }

  // TODO: generalize to all operations
  std::optional<uint64_t> alignment;
  {
    if (auto alloc = dyn_cast<memref::AllocOp>(definition)) {
      alignment = alloc.getAlignment();
    }
  }

  if (!alignment) {
    return rewriter.notifyMatchFailure(op, "nothing to be done");
  }

  auto memRefType = cast<mlir::MemRefType>(base.getType());
  llvm::SmallVector<int64_t> strides;
  int64_t memref_offset;
  if (mlir::failed(memRefType.getStridesAndOffset(strides, memref_offset))) {
    return rewriter.notifyMatchFailure(
        op, "this memref does not have statically known strided layout");
  }

  // TODO: Generalize
  if (memref_offset != 0) {
    return rewriter.notifyMatchFailure(
        op, "this memref has an offset, and we are not handling it yet");
  }

  auto datalayout = mlir::DataLayout::closest(op);
  unsigned elemTySizeInBytes = getElementTypeSize(datalayout, memRefType);

  SmallVector<uint64_t> stridesInBytes;
  for (auto stride : strides) {
    stridesInBytes.push_back(stride * elemTySizeInBytes);
  }

  SmallVector<int> staticIndices;
  auto indices = op.getIndices();
  for (auto index : indices) {
    auto indexDefinition = index.getDefiningOp();
    if (!indexDefinition) {
      staticIndices.push_back(-1);
      continue;
    }

    auto cst = dyn_cast<arith::ConstantOp>(indexDefinition);
    if (!cst) {
      staticIndices.push_back(-1);
      continue;
    }
    auto ithOffset = cast<IntegerAttr>(cst.getValue()).getInt();
    staticIndices.push_back(ithOffset);
  }

  SmallVector<unsigned int> coefficients;
  for (auto [ithStaticIndex, ithStrideInBytes] :
       llvm::reverse(llvm::zip(staticIndices, stridesInBytes))) {
    bool staticallyKnownIndex = ithStaticIndex != -1;
    if (staticallyKnownIndex) {
      auto staticOffset = ithStaticIndex * ithStrideInBytes;
      auto offsetOrAlignment = staticOffset == 0 ? *alignment : staticOffset;
      coefficients.push_back(offsetOrAlignment);
      continue;
    }
    // This is weird since it corresponds to a dimension of size zero.
    auto offsetOrAlignment = ithStrideInBytes == 0 ? *alignment : ithStrideInBytes;
    coefficients.push_back(offsetOrAlignment);
  }

  return getStaticAlignmentGuarantee(*alignment, coefficients);
}

struct AlignVectorLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern<vector::LoadOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(vector::LoadOp op,
                                PatternRewriter &rewriter) const override {

    auto maybeNewAlignment = getStaticAlignmentGuarantee(op, rewriter);
    if (failed(maybeNewAlignment)) {
      return failure();
    }
    auto newAlignment = *maybeNewAlignment;
    rewriter.modifyOpInPlace(op, [&] { op.setAlignment(newAlignment); });
    return success();
  }
};

struct AlignMemRefOpsPass
    : public impl::AlignMemRefOpsPassBase<AlignMemRefOpsPass> {
  using impl::AlignMemRefOpsPassBase<
      AlignMemRefOpsPass>::AlignMemRefOpsPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());

    mlir::iree_compiler::populateAlignMemRefOpsPatterns(patterns);

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    config.enableFolding(false);
    config.enableConstantCSE(false);
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(patterns), config)))
      return signalPassFailure();
  }
};

} // namespace

namespace mlir::iree_compiler {
void populateAlignMemRefOpsPatterns(RewritePatternSet &patterns) {
  patterns.insert<AlignVectorLoad>(patterns.getContext());
}

std::unique_ptr<Pass> createAlignMemRefOpsPass() {
  return std::make_unique<AlignMemRefOpsPass>();
}
} // namespace mlir::iree_compiler
