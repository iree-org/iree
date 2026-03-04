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
#include "iree/compiler/Utils/PassUtils.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "llvm/ADT/StringSet.h"
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

public:
  // Constructor that also accepts a shared pipeline cache.
  ConfigureTargetExecutableVariantsPass(
      ConfigureTargetExecutableVariantsPassOptions options,
      std::shared_ptr<PipelineCache> cache)
      : ConfigureTargetExecutableVariantsPassBase(std::move(options)),
        pipelineCache(std::move(cache)) {}

private:
  // Shared across clones of this pass for thread-safe pipeline caching.
  std::shared_ptr<PipelineCache> pipelineCache;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    if (variantOp.getTarget().getBackend().getValue() != target) {
      return;
    }

    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (!targetBackend) {
      variantOp.emitError() << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    // Build or retrieve the cached pass pipeline for this target attribute.
    // When many executables share the same target, this avoids redundantly
    // reconstructing the same pipeline for each one.
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTargetAttr();
    OpPassManager passManager(variantOp.getOperationName());
    if (pipelineCache) {
      passManager = pipelineCache->getOrCreate(
          targetAttr, variantOp.getOperationName(), [&](OpPassManager &pm) {
            targetBackend->buildConfigurationPassPipeline(targetAttr, pm);
          });
    } else {
      // Fallback for standalone pass usage (e.g., iree-opt).
      targetBackend->buildConfigurationPassPipeline(targetAttr, passManager);
    }

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

  // Shared across all clones of this pass for thread-safe pipeline caching.
  // When MLIR clones this pass for parallel execution on different
  // ExecutableOps, the shared_ptr is copied so all clones share the same
  // cache.
  std::shared_ptr<PipelineCache> pipelineCache =
      std::make_shared<PipelineCache>();

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackends = targetRegistry->getTargetBackends(
        targetRegistry->getRegisteredTargetBackends());
    for (auto &targetBackend : targetBackends) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableOp executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addNestedPass<IREE::HAL::ExecutableVariantOp>(
          std::make_unique<ConfigureTargetExecutableVariantsPass>(
              ConfigureTargetExecutableVariantsPassOptions{targetRegistry,
                                                           targetName},
              pipelineCache));
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
