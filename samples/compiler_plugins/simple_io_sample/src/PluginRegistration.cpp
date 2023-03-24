// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "simple_io_sample/IR/SimpleIODialect.h"
#include "simple_io_sample/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace detail {
namespace {

#define GEN_PASS_REGISTRATION
#include "simple_io_sample/Transforms/Passes.h.inc"

}  // namespace
}  // namespace detail

namespace {

struct MyOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct MySession : public PluginSession<MySession, MyOptions> {
  static void registerPasses() { ::detail::registerPasses(); }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<IREE::SimpleIO::SimpleIODialect>();
  }

  LogicalResult onActivate() override { return success(); }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(IREE::SimpleIO::createLegalizeSimpleIOPass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_simple_io_sample(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<MySession>("simple_io_sample");
  return true;
}
