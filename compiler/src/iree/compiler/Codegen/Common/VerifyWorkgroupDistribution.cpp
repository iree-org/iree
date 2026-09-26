// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_VERIFYWORKGROUPDISTRIBUTIONPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

static bool isWorkgroupForall(scf::ForallOp forallOp) {
  return forallOpHasMappingType<IREE::Codegen::WorkgroupMappingAttr,
                                IREE::LinalgExt::SplitReductionMappingAttr>(
      forallOp);
}

static bool hasAtMostOneStaticIteration(scf::ForallOp forallOp) {
  for (auto [lb, ub, step] : llvm::zip_equal(forallOp.getMixedLowerBound(),
                                             forallOp.getMixedUpperBound(),
                                             forallOp.getMixedStep())) {
    std::optional<int64_t> lbVal = getConstantIntValue(lb);
    std::optional<int64_t> ubVal = getConstantIntValue(ub);
    std::optional<int64_t> stepVal = getConstantIntValue(step);
    if (!lbVal || !ubVal || !stepVal || *stepVal <= 0) {
      return false;
    }
    if (*ubVal - *lbVal > *stepVal) {
      return false;
    }
  }
  return true;
}

struct VerifyWorkgroupDistributionPass final
    : impl::VerifyWorkgroupDistributionPassBase<
          VerifyWorkgroupDistributionPass> {

  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();

    WalkResult hasMultiWorkgroupForall =
        funcOp.walk([&](scf::ForallOp forallOp) {
          if (isWorkgroupForall(forallOp) &&
              !hasAtMostOneStaticIteration(forallOp)) {
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });

    // Without a multi-workgroup level forall, either this is a single workgroup
    // dispatch, in which case no verification is needed, or this is already
    // distributed in which case verification is no longer possible.
    if (!hasMultiWorkgroupForall.wasInterrupted()) {
      return;
    }

    // Walk in PreOrder so that parent operations are visited before children,
    // thus allowing all operations contained within workgroup foralls to be
    // skipped.
    WalkResult res = funcOp.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (auto forallOp = dyn_cast<scf::ForallOp>(op)) {
        // Skip ops contained within forall ops with workgroup mappings.
        if (isWorkgroupForall(forallOp)) {
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
