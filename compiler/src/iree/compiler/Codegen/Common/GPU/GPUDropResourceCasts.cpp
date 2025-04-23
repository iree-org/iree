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
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_GPUDROPRESOURCECASTSPASS
#include "iree/compiler/Codegen/Common/GPU/Passes.h.inc"

namespace {
struct GPUDropResourceCastsPass final
    : impl::GPUDropResourceCastsPassBase<GPUDropResourceCastsPass> {
  void runOnOperation() override;
};
} // namespace

/// Walk the producers of |input| and return the hal.interface.binding_subspan
/// that produces it. If any operation other than a slice is in between, return
/// null. If |requiresSingleSource| is set, this must be the sole use of the
/// producing binding.
static Operation *isTriviallyProducedByViewLike(Value input,
                                                bool requiresSingleSource) {
  Value next = input;
  while (auto slice = next.getDefiningOp<tensor::ExtractSliceOp>()) {
    if (requiresSingleSource) {
      if (!slice->hasOneUse()) {
        return nullptr;
      }
    }
    next = slice.getSource();
  }

  auto tensorLoad = next.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!tensorLoad || (requiresSingleSource && !tensorLoad->hasOneUse())) {
    return nullptr;
  }

  // HACK: The binding is the wrong thing to be relying on for noalias info
  // here, but there currently is no alternative. This also is only an
  // approximation as multiple function could simultaneously reference the
  // same binding, but reaching out into other functions is not possible in
  // a function nest.
  auto binding = tensorLoad.getSource()
                     .getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!binding || (requiresSingleSource && !binding->hasOneUse())) {
    return nullptr;
  }
  return binding;
}

void GPUDropResourceCastsPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  // GPU Dialect to lookup rocdl. This pass depends on IREEGPUDialect so this
  // always succeeds.
  auto *ireeGpuDialect = context->getLoadedDialect<IREE::GPU::IREEGPUDialect>();

  SmallVector<IREE::GPU::BufferResourceCastOp> castsToDrop;
  op->walk([&](IREE::GPU::BufferResourceCastOp castOp) {
    // Typed values not convertible to bool?
    bool hasCacheSwizzle = false;
    if (castOp.getCacheSwizzleStride())
      hasCacheSwizzle = true;
    if (Operation *binding =
            isTriviallyProducedByViewLike(castOp.getInput(), hasCacheSwizzle)) {
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

  for (IREE::GPU::BufferResourceCastOp cast : castsToDrop) {
    cast->replaceAllUsesWith(ValueRange{cast.getInput()});
  }
}

}; // namespace mlir::iree_compiler
