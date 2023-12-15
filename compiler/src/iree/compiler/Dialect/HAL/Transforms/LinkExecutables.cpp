// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_LINKEXECUTABLESPASS
#define GEN_PASS_DEF_LINKTARGETEXECUTABLESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-link-target-executables
//===----------------------------------------------------------------------===//

struct LinkTargetExecutablesPass
    : public IREE::HAL::impl::LinkTargetExecutablesPassBase<
          LinkTargetExecutablesPass> {
  using IREE::HAL::impl::LinkTargetExecutablesPassBase<
      LinkTargetExecutablesPass>::LinkTargetExecutablesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (!targetBackend) {
      moduleOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    OpPassManager passManager(moduleOp.getOperationName());
    targetBackend->buildLinkingPassPipeline(passManager);
    if (failed(runPipeline(passManager, moduleOp))) {
      moduleOp.emitError()
          << "failed to run linking of executable variants for backend "
          << target;
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-link-executables
//===----------------------------------------------------------------------===//

struct LinkExecutablesPass
    : public IREE::HAL::impl::LinkExecutablesPassBase<LinkExecutablesPass> {
  using IREE::HAL::impl::LinkExecutablesPassBase<
      LinkExecutablesPass>::LinkExecutablesPassBase;
  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Add pipelines for each target backend used in the module.
    // These will create/rearrange executables.
    OpPassManager passManager(moduleOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(moduleOp)) {
      passManager.addPass(IREE::HAL::createLinkTargetExecutablesPass(
          {targetRegistry, targetName}));
    }

    // Cleanup any remaining empty executables after each pipeline has run.
    // We do this to aid debugging as then the pipelines can (mostly) be run in
    // any order and not radically change the IR.
    passManager.addPass(mlir::createSymbolDCEPass());

    if (failed(runPipeline(passManager, moduleOp))) {
      moduleOp.emitError() << "failed to link executables";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
