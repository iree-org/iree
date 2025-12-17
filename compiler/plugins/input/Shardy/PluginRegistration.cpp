// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/input/Shardy/InputConversion/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/register.h"

namespace mlir::iree_compiler::shardy {

namespace {

struct ShardyOptions {
  void bindOptions(OptionsBinder &binder) {}
};

// Shardy (sdy) dialect support plugin.
// Registers the sdy dialect and provides input conversion passes to strip
// sdy ops for single-device execution.
struct ShardySession
    : public PluginSession<ShardySession, ShardyOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() { registerShardyInputConversionPasses(); }

  void onRegisterDialects(DialectRegistry &registry) override {
    mlir::sdy::registerAllDialects(registry);
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    if (typeMnemonic == "sdy") {
      buildShardyInputConversionPassPipeline(passManager);
      return true;
    }
    return false;
  }

  void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("sdy");
  }

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, StringSet<> &typeMnemonics) override {
    auto *ctx = module.getContext();
    const Dialect *sdyDialect = ctx->getLoadedDialect("sdy");
    if (!sdyDialect)
      return;

    module.walk([&](Operation *op) {
      if (op->getDialect() == sdyDialect) {
        typeMnemonics.insert("sdy");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler::shardy

IREE_DEFINE_COMPILER_OPTION_FLAGS(::mlir::iree_compiler::shardy::ShardyOptions);

extern "C" bool iree_register_compiler_plugin_input_shardy(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::shardy::ShardySession>(
      "input_shardy");
  return true;
}
