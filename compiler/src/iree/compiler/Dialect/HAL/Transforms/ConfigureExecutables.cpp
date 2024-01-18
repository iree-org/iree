// Copyright 2023 The IREE Authors
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
#include "iree/compiler/Utils/TracingUtils.h"
#include "llvm/ADT/StringSet.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_CONFIGUREEXECUTABLESPASS
#define GEN_PASS_DEF_CONFIGURETARGETEXECUTABLEVARIANTSPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-configure-target-executable-variants
//===----------------------------------------------------------------------===//

class ConfigureTargetExecutableVariantsPass
    : public IREE::HAL::impl::ConfigureTargetExecutableVariantsPassBase<
          ConfigureTargetExecutableVariantsPass> {
  using IREE::HAL::impl::ConfigureTargetExecutableVariantsPassBase<
      ConfigureTargetExecutableVariantsPass>::
      ConfigureTargetExecutableVariantsPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto variantOp = getOperation();
    if (variantOp.getTarget().getBackend().getValue() != target)
      return;

    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (!targetBackend) {
      variantOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    OpPassManager passManager(variantOp.getOperationName());
    targetBackend->buildConfigurationPassPipeline(variantOp, passManager);

    // This pipeline is optional, and the default is no passes, in which case
    // nothing is needed.
    if (passManager.empty()) {
      return;
    }

    if (failed(runPipeline(passManager, variantOp))) {
      variantOp.emitError() << "failed to run configuration of source "
                               "executable to target executable for backend "
                            << variantOp.getTarget();
      return signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-configure-executables
//===----------------------------------------------------------------------===//

struct ConfigureExecutablesPass
    : public IREE::HAL::impl::ConfigureExecutablesPassBase<
          ConfigureExecutablesPass> {
  using IREE::HAL::impl::ConfigureExecutablesPassBase<
      ConfigureExecutablesPass>::ConfigureExecutablesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackends = targetRegistry->getTargetBackends(
        targetRegistry->getRegisteredTargetBackends());
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
          IREE::HAL::createConfigureTargetExecutableVariantsPass(
              {targetRegistry, targetName}));
    }

    IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(INFO, executableOp.getSymName().str());

    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to configure executables";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
