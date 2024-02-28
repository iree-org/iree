// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/LoweringStrategy.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace mlir::iree_compiler::IREE::HAL {

namespace {

class LLVMCPUPatternLoweringStrategy : public IREE::HAL::LoweringStrategy {
  LogicalResult matchAndSetTranslationInfo(
      FunctionOpInterface funcOp) override {
    llvm::dbgs() << "test\n";
    return failure();
  }
};

class PluginRegistration
    : public PluginSession<PluginRegistration, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  void configureHALTargetBackends(
      IREE::HAL::TargetRegistry &registry) override {
    auto backend = registry.getTargetBackend("llvm-cpu");
    backend->addLoweringStrategy(
        std::make_unique<LLVMCPUPatternLoweringStrategy>());
  }
};

}  // namespace

}  // namespace mlir::iree_compiler::IREE::HAL

extern "C" bool
iree_register_compiler_plugin_hal_lowering_strategy_llvmcpu_pattern_lowering_strategy(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar
      ->registerPlugin< ::mlir::iree_compiler::IREE::HAL::PluginRegistration>(
          "hal_lowering_strategy_llvmcpu_pattern_lowering_strategy");
  return true;
}
