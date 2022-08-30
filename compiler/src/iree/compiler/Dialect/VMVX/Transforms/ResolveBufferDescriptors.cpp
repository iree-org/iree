// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/VMVX/IR/VMVXDialect.h"
#include "iree/compiler/Dialect/VMVX/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/VMVX/Transforms/Passes.h"
#include "iree/compiler/Utils/IndexSet.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::VMVX {

namespace {

struct FromMemRefSubView : public OpRewritePattern<GetBufferDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetBufferDescriptorOp op,
                                PatternRewriter &rewriter) const override {
    auto subview = op.getSource().getDefiningOp<memref::SubViewOp>();
    if (!subview) return failure();
    auto loc = op.getLoc();
    IndexSet indexSet(loc, rewriter);

    // Get types.
    auto subType = subview.getResult().getType().cast<MemRefType>();
    Value source = subview.getSource();
    auto sourceType = source.getType().cast<MemRefType>();
    if (sourceType.getRank() != subType.getRank()) {
      return rewriter.notifyMatchFailure(op,
                                         "rank reducing subview not supported");
    }
    int rank = sourceType.getRank();

    // Create a descriptor for the source.
    IndexType indexType = rewriter.getIndexType();
    SmallVector<Type> sizeStrideTypes;
    for (int i = 0; i < rank; i++) {
      sizeStrideTypes.push_back(indexType);
    }
    auto sourceDesc = rewriter.create<IREE::VMVX::GetBufferDescriptorOp>(
        loc, op.getBaseBuffer().getType(), indexType, sizeStrideTypes,
        sizeStrideTypes, subview.getSource());

    // For sizes, we just use the new ones, discarding the source.
    SmallVector<Value> newSizes;
    for (int i = 0; i < rank; ++i) {
      if (subview.isDynamicSize(i)) {
        newSizes.push_back(subview.getDynamicSize(i));
      } else {
        newSizes.push_back(indexSet.get(subview.getStaticSize(i)));
      }
      op.getSizes()[i].replaceAllUsesWith(newSizes.back());
    }

    // Apply stride multipliers.
    SmallVector<Value> strides;
    for (int i = 0; i < rank; ++i) {
      Value currentStride;
      if (subview.isDynamicStride(i)) {
        currentStride = subview.getDynamicStride(i);
      } else {
        currentStride = indexSet.get(subview.getStaticStride(i));
      }
      currentStride = rewriter.createOrFold<arith::MulIOp>(
          loc, sourceDesc.getStrides()[i], currentStride);
      strides.push_back(currentStride);
      op.getStrides()[i].replaceAllUsesWith(currentStride);
    }

    // Offsets.
    Value offset = sourceDesc.getOffset();
    for (int i = 0; i < rank; ++i) {
      Value logicalOffset;
      if (subview.isDynamicOffset(i)) {
        logicalOffset = subview.getDynamicOffset(i);
      } else {
        int64_t staticOffset = subview.getStaticOffset(i);
        if (staticOffset == 0) {
          // Since added, just omit (will multiply to 0).
          continue;
        }
        logicalOffset = indexSet.get(staticOffset);
      }
      Value physicalOffset =
          rewriter.createOrFold<arith::MulIOp>(loc, logicalOffset, strides[i]);
      offset =
          rewriter.createOrFold<arith::AddIOp>(loc, offset, physicalOffset);
    }
    op.getOffset().replaceAllUsesWith(offset);

    // Base.
    op.getBaseBuffer().replaceAllUsesWith(sourceDesc.getBaseBuffer());
    rewriter.eraseOp(op);
    return success();
  }
};

struct FromHalInterfaceBindingSubspan
    : public OpRewritePattern<GetBufferDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetBufferDescriptorOp op,
                                PatternRewriter &rewriter) const override {
    auto binding =
        op.getSource().getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!binding) return failure();
    auto loc = op.getLoc();
    IndexSet indexSet(loc, rewriter);

    // Replace the op with values:
    //   base_buffer: The subspan result
    //   offset: byte offset from subspan divided by element type size
    //   sizes: static and dynamic sizes from the subspan
    //   strides: identity strides
    auto memRefType = binding.getResult().getType().cast<MemRefType>();
    int rank = memRefType.getRank();

    // Compute sizes.
    SmallVector<Value> sizes;
    auto dynamicDimIt = binding.getDynamicDims().begin();
    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        sizes.push_back(*dynamicDimIt);
        dynamicDimIt++;
      } else {
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, memRefType.getDimSize(i)));
      }

      // Replace as we go.
      op.getSizes()[i].replaceAllUsesWith(sizes.back());
    }

    // Strides.
    if (rank > 0) {
      SmallVector<Value> strides;
      strides.resize(rank);
      strides[rank - 1] = indexSet.get(1);
      for (int i = rank - 2; i >= 0; --i) {
        strides[i] = rewriter.createOrFold<arith::MulIOp>(loc, strides[i + 1],
                                                          sizes[i + 1]);
      }
      for (int i = 0; i < rank; ++i) {
        op.getStrides()[i].replaceAllUsesWith(strides[i]);
      }
    }

    // Offset.
    auto elementSize =
        rewriter.create<IREE::Util::SizeOfOp>(loc, memRefType.getElementType());
    op.getOffset().replaceAllUsesWith(rewriter.createOrFold<arith::DivUIOp>(
        loc, binding.getByteOffset(), elementSize));

    // Base buffer.
    op.getBaseBuffer().replaceAllUsesWith(
        rewriter
            .create<IREE::VMVX::GetRawInterfaceBindingBufferOp>(
                loc, op.getBaseBuffer().getType(), binding.getSetAttr(),
                binding.getBindingAttr())
            .getResult());

    rewriter.eraseOp(op);
    return success();
  }
};

// Allocations (and anything else the returns a non-offset identity memref)
// are matched by this pattern.
struct FromAllocation : public OpRewritePattern<GetBufferDescriptorOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(GetBufferDescriptorOp op,
                                PatternRewriter &rewriter) const override {
    auto alloca = op.getSource().getDefiningOp<memref::AllocaOp>();
    if (!alloca) return failure();
    auto memRefType = alloca.getResult().getType().cast<MemRefType>();
    if (!memRefType.getLayout().isIdentity()) {
      return rewriter.notifyMatchFailure(op, "not identity allocation");
    }

    auto loc = op.getLoc();
    IndexSet indexSet(loc, rewriter);

    // Replace the op with values:
    //   base_buffer: The subspan result
    //   offset: byte offset from subspan divided by element type size
    //   sizes: static and dynamic sizes from the subspan
    //   strides: identity strides
    int rank = memRefType.getRank();

    // Compute sizes.
    SmallVector<Value> sizes;
    auto dynamicDimIt = alloca.getDynamicSizes().begin();
    for (int i = 0; i < rank; ++i) {
      if (memRefType.isDynamicDim(i)) {
        sizes.push_back(*dynamicDimIt);
        dynamicDimIt++;
      } else {
        sizes.push_back(rewriter.create<arith::ConstantIndexOp>(
            loc, memRefType.getDimSize(i)));
      }

      // Replace as we go.
      op.getSizes()[i].replaceAllUsesWith(sizes.back());
    }

    // Strides (just creates identity strides).
    if (rank > 0) {
      SmallVector<Value> strides;
      strides.resize(rank);
      strides[rank - 1] = indexSet.get(1);
      for (int i = rank - 2; i >= 0; --i) {
        strides[i] = rewriter.createOrFold<arith::MulIOp>(loc, strides[i + 1],
                                                          sizes[i + 1]);
      }
      for (int i = 0; i < rank; ++i) {
        op.getStrides()[i].replaceAllUsesWith(strides[i]);
      }
    }

    // Offset.
    op.getOffset().replaceAllUsesWith(indexSet.get(0));

    // Base buffer.
    op.getBaseBuffer().replaceAllUsesWith(
        rewriter
            .create<UnrealizedConversionCastOp>(
                loc, op.getBaseBuffer().getType(), alloca.getResult())
            .getResult(0));

    rewriter.eraseOp(op);
    return success();
  }
};

class ResolveBufferDescriptorsPass
    : public ResolveBufferDescriptorsBase<ResolveBufferDescriptorsPass> {
 public:
  ResolveBufferDescriptorsPass() = default;
  ResolveBufferDescriptorsPass(const ResolveBufferDescriptorsPass &) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VMVX::VMVXDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<FromAllocation, FromHalInterfaceBindingSubspan,
                    FromMemRefSubView>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }

    // If any get_buffer_descriptor patterns remain, we fail.
    if (!allowUnresolved) {
      SmallVector<Operation *> remaining;
      getOperation()->walk([&](IREE::VMVX::GetBufferDescriptorOp op) {
        remaining.push_back(op);
      });

      if (!remaining.empty()) {
        auto diag = getOperation()->emitError()
                    << "Unable to resolve all strided buffer descriptors:";
        for (auto *op : remaining) {
          diag.attachNote(op->getLoc()) << "remaining live use";
        }
        signalPassFailure();
      }
    }
  }

  Option<bool> allowUnresolved{
      *this, "allow-unresolved",
      llvm::cl::desc("Allow unresolved descriptors (for testing)")};
};

}  // namespace

std::unique_ptr<mlir::OperationPass<>> createResolveBufferDescriptorsPass() {
  return std::make_unique<ResolveBufferDescriptorsPass>();
}

}  // namespace mlir::iree_compiler::IREE::VMVX
