// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
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
/// that produces it if the binding is readonly. If the binding is already
/// bufferized, then check that there is at most one FatRawBufferCastOp in its
/// use chain, and add it to |rawBufferCasts| if found.
static Operation *isTriviallyProducedByReadOnlyViewLike(
    Value input, SetVector<amdgpu::FatRawBufferCastOp> &rawBufferCasts) {
  Value next = input;
  while (auto slice = next.getDefiningOp<tensor::ExtractSliceOp>()) {
    next = slice.getSource();
  }

  // Case 1: Subspan is not bufferized.
  auto tensorLoad = next.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (tensorLoad) {
    IREE::HAL::InterfaceBindingSubspanOp binding =
        tensorLoad.getSource()
            .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
    if (!binding ||
        cast<IREE::TensorExt::DispatchTensorType>(binding.getType())
                .getAccess() != IREE::TensorExt::TensorAccess::ReadOnly) {
      return nullptr;
    }
    return binding;
  }

  // Case 2: Subspan is bufferized.
  auto loadFromBuffer = next.getDefiningOp<IREE::Codegen::LoadFromBufferOp>();
  if (!loadFromBuffer) {
    return nullptr;
  }
  std::optional<IREE::HAL::InterfaceBindingSubspanOp> binding =
      getSourceSubspanMemref(
          cast<TypedValue<MemRefType>>(loadFromBuffer.getBuffer()));
  if (!binding) {
    return nullptr;
  }
  std::optional<IREE::HAL::DescriptorFlags> descriptorFlags =
      binding->getDescriptorFlags();
  if (!descriptorFlags.has_value() ||
      !bitEnumContainsAll(*descriptorFlags,
                          IREE::HAL::DescriptorFlags::ReadOnly)) {
    return nullptr;
  }
  // Check that the binding has at most one FatRawBufferCastOp in its use
  // chain.
  ForwardSliceOptions options;
  options.filter = [&](Operation *op) {
    // Only follow memref results.
    if (llvm::none_of(op->getOpResults(), [](OpResult result) {
          return isa<MemRefType>(result.getType());
        })) {
      return false;
    }
    if (isa<amdgpu::FatRawBufferCastOp>(op)) {
      rawBufferCasts.insert(cast<amdgpu::FatRawBufferCastOp>(op));
    }
    return true;
  };
  SetVector<Operation *> slice;
  getForwardSlice(binding->getOperation(), &slice, options);
  if (rawBufferCasts.size() > 1) {
    return nullptr;
  }
  return binding.value();
}

namespace {
struct BubbleResourceCastPattern
    : public OpRewritePattern<IREE::GPU::BufferResourceCastOp> {
  using Base::Base;

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
            .Case([&](tensor::ExtractSliceOp extract) {
              if (!extract->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(extract);
              auto newCast = IREE::GPU::BufferResourceCastOp::create(
                  rewriter, loc, extract.getSource().getType(),
                  extract.getSource());
              extract.getSourceMutable().assign(newCast);
              return true;
            })
            .Case([&](tensor::ExpandShapeOp expand) {
              if (!expand->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(expand);
              auto newCast = IREE::GPU::BufferResourceCastOp::create(
                  rewriter, loc, expand.getSrcType(), expand.getSrc());
              expand.getSrcMutable().assign(newCast);
              return true;
            })
            .Case([&](tensor::CollapseShapeOp collapse) {
              if (!collapse->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(collapse);
              auto newCast = IREE::GPU::BufferResourceCastOp::create(
                  rewriter, loc, collapse.getSrcType(), collapse.getSrc());
              collapse.getSrcMutable().assign(newCast);
              return true;
            })
            .Case([&](tensor::PadOp pad) {
              if (!pad->hasOneUse()) {
                return false;
              }

              rewriter.setInsertionPoint(pad);
              auto newCast = IREE::GPU::BufferResourceCastOp::create(
                  rewriter, loc, pad.getSourceType(), pad.getSource());
              pad.getSourceMutable().assign(newCast);
              return true;
            })
            .Case([&](linalg::LinalgOp linalgOp) {
              // Skip gather-like linalg ops.
              if (linalgOp.hasIndexSemantics() || !linalgOp->hasOneUse()) {
                return false;
              }
              rewriter.setInsertionPoint(linalgOp);
              // Only propagate to input operands.
              for (auto inputOperand : linalgOp.getDpsInputOperands()) {
                auto newCast = IREE::GPU::BufferResourceCastOp::create(
                    rewriter, loc, inputOperand->get().getType(),
                    inputOperand->get());
                inputOperand->assign(newCast);
              }
              return true;
            })
            .Case([&](tensor::EmptyOp empty) {
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
  IRRewriter rewriter(op);
  op->walk([&](IREE::GPU::BufferResourceCastOp castOp) {
    // Skip ops without cache swizzle values set. There is no need to drop
    // these because we will try to bubble them up.
    if (!castOp.getCacheSwizzleStride()) {
      return;
    }
    SetVector<amdgpu::FatRawBufferCastOp> rawBufferCasts;
    if (Operation *binding = isTriviallyProducedByReadOnlyViewLike(
            castOp.getInput(), rawBufferCasts)) {
      for (amdgpu::FatRawBufferCastOp rawBufferCast : rawBufferCasts) {
        replaceMemrefUsesAndPropagateType(rewriter, rawBufferCast.getLoc(),
                                          rawBufferCast.getResult(),
                                          rawBufferCast.getSource());
        rewriter.eraseOp(rawBufferCast);
      }
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
