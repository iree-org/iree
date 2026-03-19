// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenInterfaces.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "mlir/Dialect/SMT/IR/SMTDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_INSERTSMTCONSTRAINTSPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct InsertSMTConstraintsPass final
    : impl::InsertSMTConstraintsPassBase<InsertSMTConstraintsPass> {
  using Base::Base;
  void runOnOperation() override;
};

} // namespace

void InsertSMTConstraintsPass::runOnOperation() {
  if (!shouldSetTunerAttributes()) {
    return;
  }

  FunctionOpInterface funcOp = getOperation();

  // Get target's available pipelines via TargetPipelineProviderAttrInterface.
  IREE::HAL::ExecutableTargetAttr halTarget =
      IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!halTarget) {
    return;
  }
  auto provider =
      dyn_cast_if_present<IREE::Codegen::TargetPipelineProviderAttrInterface>(
          getConfigTargetInfo(halTarget.getConfiguration()));
  if (!provider) {
    return;
  }

  // Collect pre-annotated root ops and group by set number.
  SmallVector<Operation *> rootOps;
  funcOp.walk([&](Operation *op) {
    if (hasRootOpInfo(op)) {
      rootOps.push_back(op);
    }
  });

  if (rootOps.empty()) {
    return;
  }

  llvm::stable_sort(rootOps, [](Operation *a, Operation *b) {
    return getRootOpInfo(a).getSet() < getRootOpInfo(b).getSet();
  });

  // Pre-filter pipelines that support constraint generation.
  SmallVector<IREE::Codegen::PipelineConstraintAttrInterface> constraintIfaces;
  for (Attribute pipelineAttr : provider.getAvailablePipelines()) {
    if (auto iface = dyn_cast<IREE::Codegen::PipelineConstraintAttrInterface>(
            pipelineAttr)) {
      constraintIfaces.push_back(iface);
    }
  }

  // For each set of root ops, call each pipeline's constraint emitter.
  for (auto range = rootOps.begin(); range != rootOps.end();) {
    int64_t setNumber = getRootOpInfo(*range).getSet();
    auto setEnd = std::find_if(range, rootOps.end(), [&](Operation *op) {
      return getRootOpInfo(op).getSet() != setNumber;
    });
    ArrayRef<Operation *> setOps(&*range, std::distance(range, setEnd));

    for (IREE::Codegen::PipelineConstraintAttrInterface iface :
         constraintIfaces) {
      if (failed(iface.emitConstraintOps(funcOp, setOps))) {
        return signalPassFailure();
      }
    }
    range = setEnd;
  }
}

} // namespace mlir::iree_compiler
