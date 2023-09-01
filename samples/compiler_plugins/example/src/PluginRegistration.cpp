// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct MyOptions {
  bool flag = false;

  void bindOptions(OptionsBinder &binder) {
    static llvm::cl::OptionCategory category("IREE Example Plugin");
    binder.opt<bool>("iree-example-flag", flag,
                     llvm::cl::desc("Dummy flag for the example plugin"),
                     llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {
  LogicalResult onActivate() override {
    mlir::emitRemark(mlir::UnknownLoc::get(context))
        << "This remark is from the example plugin activation (flag="
        << options.flag << ")";
    return success();
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_example(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<MySession>("example");
  return true;
}
