// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// An out-of-tree plugin: its own dialect, its own pass with an option, and a
// pass pipeline extension that puts the pass into every compilation.

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/Pass/PassManager.h"
#include "ootex/IR/OotexDialect.h"
#include "ootex/Transforms/Passes.h"

namespace {

struct OotexOptions {
  std::string tag = "marked";

  void bindOptions(mlir::iree_compiler::OptionsBinder& binder) {
    static llvm::cl::OptionCategory category("Ootex plugin");
    binder.opt<std::string>(
        "ootex-tag", tag,
        llvm::cl::desc("Value the ootex plugin writes onto marked functions"),
        llvm::cl::cat(category));
  }
};

struct OotexSession
    : public mlir::iree_compiler::PluginSession<OotexSession, OotexOptions> {
  static void registerPasses() { ootex::registerOotexPasses(); }

  void onRegisterDialects(mlir::DialectRegistry& registry) override {
    registry.insert<ootex::OotexDialect>();
  }

  void extendPreprocessingPassPipeline(mlir::OpPassManager& pm) override {
    ootex::AnnotateMarkedFunctionsOptions passOptions;
    passOptions.tag = options.tag;
    pm.addPass(ootex::createAnnotateMarkedFunctions(passOptions));
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(OotexOptions);

static bool registerOotexPlugin(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<OotexSession>("ootex");
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(ootex, registerOotexPlugin)
