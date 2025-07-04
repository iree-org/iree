// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VERIFYWORKGROUPDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct VerifyWorkgroupDistributionPass final
    : impl::VerifyWorkgroupDistributionPassBase<
          VerifyWorkgroupDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    WalkResult hasForall = funcOp.walk([&](scf::ForallOp forallOp) {
      if (forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr,
                                 IREE::LinalgExt::SplitReductionMappingAttr>(
              forallOp)) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    // Without a workgroup level forall, either this is a single workgroup
    // dispatch, in which case no verification is needed, or this is already
    // distributed in which case verification is no longer possible.
    if (!hasForall.wasInterrupted()) {
      return;
    }

    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within workgroup foralls to be
    // skipped.
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        // Skip ops contained within forall ops with workgroup mappings.
        if (forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr,
                                   IREE::LinalgExt::SplitReductionMappingAttr>(
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

          // Writes to non-global memory are fine. hasGlobalMemoryAddressSpace()
          // treats the unspecified address space as global, but we don't have
          // that assumption yet.
          if (!type.getMemorySpace() || !hasGlobalMemoryAddressSpace(type)) {
            continue;
          }

          op->emitOpError(
              "write affecting operations on global resources are restricted "
              "to workgroup distributed contexts.");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      funcOp.emitOpError("failed on workgroup distribution verification");
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
