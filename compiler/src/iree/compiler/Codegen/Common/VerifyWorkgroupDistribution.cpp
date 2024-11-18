// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VERIFYWORKGROUPDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

/// Pass to verify that all writes to global memory are explicitly mapped to
/// workgroups. This means that in cases where we use loops (scf.forall) to
/// manage distribution to workgroups, we require that all ops with write
/// side effects are contained within a workgroup distributed loop.
struct VerifyWorkgroupDistributionPass final
    : impl::VerifyWorkgroupDistributionPassBase<
          VerifyWorkgroupDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    WalkResult hasForall = funcOp.walk([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        if (forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr>(
                forallOp)) {
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // Without a workgroup level forall, either this is a single workgroup
    // dispatch, in which case no verification is needed, or this is already
    // distributed in which case verification is no longer possible.
    if (!hasForall.wasInterrupted()) {
      return;
    }

    auto globalAddressSpace = IREE::HAL::DescriptorTypeAttr::get(
        &getContext(), IREE::HAL::DescriptorType::StorageBuffer);

    bool failed = false;
    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within workgroup foralls to be
    // skipped.
    funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        // Skip ops contained within forall ops with workgroup mappings.
        if (forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr>(
                forallOp)) {
          return WalkResult::skip();
        }
      }
      if (auto memoryEffectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
        for (Value operand : memoryEffectOp->getOperands()) {
          auto type = dyn_cast<MemRefType>(operand.getType());
          if (!type ||
              !memoryEffectOp.getEffectOnValue<MemoryEffects::Write>(operand)) {
            continue;
          }

          // Writes to non-global memory are fine.
          if (type.getMemorySpace() != globalAddressSpace) {
            continue;
          }

          op->emitOpError(
              "write affecting operations on global resources are restricted "
              "to workgroup distributed contexts.");
          failed = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    // We can't use the WalkResult here because it's an enum
    if (failed) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
