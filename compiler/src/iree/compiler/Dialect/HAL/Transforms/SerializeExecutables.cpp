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
#include "iree/compiler/Utils/TracingUtils.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::IREE::HAL {

#define GEN_PASS_DEF_SERIALIZEEXECUTABLESPASS
#define GEN_PASS_DEF_SERIALIZETARGETEXECUTABLESPASS
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-hal-serialize-target-executables
//===----------------------------------------------------------------------===//

struct SerializeTargetExecutablesPass
    : public IREE::HAL::impl::SerializeTargetExecutablesPassBase<
          SerializeTargetExecutablesPass> {
  using IREE::HAL::impl::SerializeTargetExecutablesPassBase<
      SerializeTargetExecutablesPass>::SerializeTargetExecutablesPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect>();
    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (targetBackend) {
      targetBackend->getDependentDialects(registry);
    }
  }

  void runOnOperation() override {
    auto executableOp = getOperation();
    auto moduleOp = executableOp->getParentOfType<mlir::ModuleOp>();

    auto targetBackend = targetRegistry->getTargetBackend(target);
    if (!targetBackend) {
      executableOp.emitError()
          << "unregistered target backend '" << target << "'";
      return signalPassFailure();
    }

    // Setup serialization options for dumping artifacts (if requested).
    // Target backends are allowed to generate multiple binaries for each
    // serialization.
    TargetBackend::SerializationOptions serializationOptions;
    serializationOptions.dumpBaseName =
        (moduleOp.getName().value_or("module") + "_" + executableOp.getName())
            .str();
    serializationOptions.debugLevel = debugLevel;
    serializationOptions.dumpIntermediatesPath = dumpIntermediatesPath;
    serializationOptions.dumpBinariesPath = dumpBinariesPath;
    if (!dumpIntermediatesPath.empty()) {
      llvm::sys::fs::create_directories(dumpIntermediatesPath);
    }
    if (!dumpBinariesPath.empty()) {
      llvm::sys::fs::create_directories(dumpBinariesPath);
    }

    auto variantOps = llvm::to_vector(
        executableOp.getBlock().getOps<IREE::HAL::ExecutableVariantOp>());
    for (auto variantOp : variantOps) {
      if (variantOp.getTarget().getBackend().getValue() != target)
        continue;
      OpBuilder executableBuilder(variantOp);
      // Ask the target backend to serialize the executable. Note that it
      // may create one or more hal.executable.binary ops in the case of
      // multi-architecture binaries.
      if (failed(targetBackend->serializeExecutable(
              serializationOptions, variantOp, executableBuilder))) {
        variantOp.emitError()
            << "failed to serialize executable for target backend " << target;
        return signalPassFailure();
      }
      variantOp.erase();
    }
  }
};

//===----------------------------------------------------------------------===//
// --iree-hal-serialize-executables
//===----------------------------------------------------------------------===//

struct SerializeExecutablesPass
    : public IREE::HAL::impl::SerializeExecutablesPassBase<
          SerializeExecutablesPass> {
  using IREE::HAL::impl::SerializeExecutablesPassBase<
      SerializeExecutablesPass>::SerializeExecutablesPassBase;
  void runOnOperation() override {
    auto executableOp = getOperation();
    OpPassManager passManager(executableOp.getOperationName());
    for (const auto &targetName : gatherExecutableTargetNames(executableOp)) {
      passManager.addPass(IREE::HAL::createSerializeTargetExecutablesPass(
          {targetRegistry, targetName, debugLevel, dumpIntermediatesPath,
           dumpBinariesPath}));
    }

    IREE_COMPILER_TRACE_MESSAGE_DYNAMIC(INFO, executableOp.getSymName().str());

    if (failed(runPipeline(passManager, executableOp))) {
      executableOp.emitError() << "failed to serialize executables";
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::HAL
