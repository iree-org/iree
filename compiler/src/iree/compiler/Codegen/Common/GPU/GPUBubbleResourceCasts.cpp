// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUBUBBLERESOURCECASTSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUBubbleResourceCastsPass final
    : impl::GPUBubbleResourceCastsPassBase<GPUBubbleResourceCastsPass> {
  void runOnOperation() override;
};
} // namespace

/// Walk the producers of |input| and return the hal.interface.binding_subspan
/// that produces it if the binding is readonly.
static Operation *isTriviallyProducedByReadOnlyViewLike(Value input) {
  Value next = input;
  while (auto slice = next.getDefiningOp<tensor::ExtractSliceOp>()) {
    next = slice.getSource();
  }

  auto tensorLoad = next.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!tensorLoad) {
    return nullptr;
  }

  auto binding = tensorLoad.getSource()
                     .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!binding ||
      cast<IREE::TensorExt::DispatchTensorType>(binding.getType())
              .getAccess() != IREE::TensorExt::TensorAccess::ReadOnly) {
    return nullptr;
  }
  return binding;
}

namespace {
struct BubbleResourceCastPattern
    : public OpRewritePattern<IREE::GPU::BufferResourceCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IREE::GPU::BufferResourceCastOp castOp,
                                PatternRewriter &rewriter) const override {
    // Skip ops with cache swizzle strides because the stride could be
    // rendered invalid by the swap.
    // TODO: Support bubbling swizzled casts in cases where it makes sense.
    // e.g. iree_gpu.buffer_resource_cast(tensor.extract_slice) if the cache
    // swizzle stride does not depend on the extract_slice.
    if (castOp.getCacheSwizzleStride()) {
      return failure();
    }

    auto producer = castOp.getInput().getDefiningOp();
    if (!producer) {
      return failure();
    }

    Location loc = castOp.getLoc();
    bool swapped =
        TypeSwitch<Operation *, bool>(producer)
            .Case<tensor::ExtractSliceOp>([&](tensor::ExtractSliceOp extract) {
              if (!extract->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(extract);
              auto newCast = rewriter.create<IREE::GPU::BufferResourceCastOp>(
                  loc, extract.getSource().getType(), extract.getSource());
              extract.getSourceMutable().assign(newCast);
              return true;
            })
            .Case<tensor::ExpandShapeOp>([&](tensor::ExpandShapeOp expand) {
              if (!expand->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(expand);
              auto newCast = rewriter.create<IREE::GPU::BufferResourceCastOp>(
                  loc, expand.getSrcType(), expand.getSrc());
              expand.getSrcMutable().assign(newCast);
              return true;
            })
            .Case<tensor::CollapseShapeOp>(
                [&](tensor::CollapseShapeOp collapse) {
                  if (!collapse->hasOneUse()) {
                    return false;
                  }

                  rewriter.setInsertionPoint(collapse);
                  auto newCast =
                      rewriter.create<IREE::GPU::BufferResourceCastOp>(
                          loc, collapse.getSrcType(), collapse.getSrc());
                  collapse.getSrcMutable().assign(newCast);
                  return true;
                })
            .Case<tensor::PadOp>([&](tensor::PadOp pad) {
              if (!pad->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(pad);
              auto newCast = rewriter.create<IREE::GPU::BufferResourceCastOp>(
                  loc, pad.getSourceType(), pad.getSource());
              pad.getSourceMutable().assign(newCast);
              return true;
            })
            .Case<linalg::LinalgOp>([&](linalg::LinalgOp linalgOp) {
              // Skip gather-like linalg ops.
              if (linalgOp.hasIndexSemantics() || !linalgOp->hasOneUse()) {
                return false;
              }
              rewriter.setInsertionPoint(linalgOp);
              // Only propagate to input operands.
              for (auto inputOperand : linalgOp.getDpsInputOperands()) {
                auto newCast = rewriter.create<IREE::GPU::BufferResourceCastOp>(
                    loc, inputOperand->get().getType(), inputOperand->get());
                inputOperand->assign(newCast);
              }
              return true;
            })
            .Case<tensor::EmptyOp>([&](tensor::EmptyOp empty) {
              // Drop all casts of empties.
              return true;
            })
            .Default([](Operation *op) {
              // Leave all other casts alone.
              return false;
            });

    // If swapping did not occur, do not drop the cast.
    if (!swapped) {
      return failure();
    }

    rewriter.replaceOp(castOp, castOp.getInput());
    return success();
  }
};
} // namespace

void GPUBubbleResourceCastsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<BubbleResourceCastPattern>(context);
    if (failed(applyPatternsGreedily(op, std::move(patterns)))) {
      return signalPassFailure();
    }
  }

  // GPU Dialect to lookup rocdl. This pass depends on IREEGPUDialect so this
  // always succeeds.
  auto *ireeGpuDialect = context->getLoadedDialect<IREE::GPU::IREEGPUDialect>();

  SmallVector<IREE::GPU::BufferResourceCastOp> castsToDrop;
  op->walk([&](IREE::GPU::BufferResourceCastOp castOp) {
    // Skip ops without cache swizzle values set. There is no need to drop
    // these because we will try to bubble them up.
    if (!castOp.getCacheSwizzleStride()) {
      return;
    }
    if (Operation *binding =
            isTriviallyProducedByReadOnlyViewLike(castOp.getInput())) {
      if (ireeGpuDialect->getUseRocdlBufferInstructionsAttrHelper()
              .isAttrPresent(binding)) {
        // TODO: Support updating buffer structs so we don't have to replace
        // the cast on the binding with this cast.
        ireeGpuDialect->getUseRocdlBufferInstructionsAttrHelper().removeAttr(
            binding);
      }
    } else {
      // If the cast can't trivially map to a view-like producer of a binding,
      // drop it. Common if there are any unfused producers.
      castsToDrop.push_back(castOp);
    }
  });

  // Drop all cache swizzling ops that might be blocking fusion.
  for (IREE::GPU::BufferResourceCastOp cast : castsToDrop) {
    cast->replaceAllUsesWith(ValueRange{cast.getInput()});
  }
}

}; // namespace mlir::iree_compiler
