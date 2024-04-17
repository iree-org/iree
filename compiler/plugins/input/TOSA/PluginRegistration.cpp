// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "compiler/plugins/input/TOSA/InputConversion/Passes.h"

namespace mlir::iree_compiler {

namespace {

// TOSA (Tensor Operator Set Architecture) support plugin.
//   * https://www.mlplatform.org/tosa
//   * https://mlir.llvm.org/docs/Dialects/TOSA/
//
// The TOSA plugin provides dialects, passes and opt-in options.
// Therefore, it is appropriate for default activation.
struct TOSASession
    : public PluginSession<TOSASession, EmptyPluginOptions,
                           PluginActivationPolicy::DefaultActivated> {
  static void registerPasses() {
    registerTOSAConversionPasses();
    registerTosaToArith();
    registerTosaToLinalg();
    registerTosaToTensor();
  }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<tosa::TosaDialect>();
  }

  bool extendCustomInputConversionPassPipeline(
      OpPassManager &passManager, std::string_view typeMnemonic) override {
    if (typeMnemonic == "tosa") {
      buildTOSAInputConversionPassPipeline(passManager);
      return true;
    }

    return false;
  }

  void populateCustomInputConversionTypes(StringSet<> &typeMnemonics) override {
    typeMnemonics.insert("tosa");
  }

  void populateDetectedCustomInputConversionTypes(
      ModuleOp &module, StringSet<> &typeMnemonics) override {
    auto *ctx = module.getContext();
    const Dialect *tosaDialect = ctx->getLoadedDialect("tosa");

    module.walk([&](Operation *op) {
      Dialect *d = op->getDialect();
      if (d == tosaDialect) {
        typeMnemonics.insert("tosa");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
};

} // namespace

} // namespace mlir::iree_compiler

extern "C" bool iree_register_compiler_plugin_input_tosa(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<::mlir::iree_compiler::TOSASession>("input_tosa");
  return true;
}
