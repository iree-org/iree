// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/PluginAPI/PluginEntryPoint.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

// The id is a build parameter so one source can be both linked into the
// compiler and dlopen'd by it, which test/dynamic_link.mlir compares. Every
// global name derives from it, because a second registration of one id aborts,
// as does a second llvm::cl option or category of one name.
#ifndef IREE_EXAMPLE_PLUGIN_ID
#define IREE_EXAMPLE_PLUGIN_ID example
#endif

#define IREE_EXAMPLE_CONCAT_IMPL(a, b) a##b
#define IREE_EXAMPLE_CONCAT(a, b) IREE_EXAMPLE_CONCAT_IMPL(a, b)
#define IREE_EXAMPLE_STRINGIFY_IMPL(x) #x
#define IREE_EXAMPLE_STRINGIFY(x) IREE_EXAMPLE_STRINGIFY_IMPL(x)

#define IREE_EXAMPLE_PLUGIN_ID_STR \
  IREE_EXAMPLE_STRINGIFY(IREE_EXAMPLE_PLUGIN_ID)
#define IREE_EXAMPLE_PLUGIN_FLAG "iree-" IREE_EXAMPLE_PLUGIN_ID_STR "-flag"
#define IREE_EXAMPLE_PLUGIN_CATEGORY \
  "IREE Example Plugin (" IREE_EXAMPLE_PLUGIN_ID_STR ")"
#define IREE_EXAMPLE_REGISTER_PLUGIN \
  IREE_EXAMPLE_CONCAT(iree_register_compiler_plugin_, IREE_EXAMPLE_PLUGIN_ID)

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct MyOptions {
  bool flag = false;

  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category(IREE_EXAMPLE_PLUGIN_CATEGORY);
    binder.opt<bool>(IREE_EXAMPLE_PLUGIN_FLAG, flag,
                     llvm::cl::desc("Dummy flag for the example plugin"),
                     llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {
  LogicalResult onActivate() override {
    // Both tests assert on this text, so it must not vary by build.
    mlir::emitRemark(mlir::UnknownLoc::get(context))
        << "This remark is from the example plugin activation (flag="
        << options.flag << ")";
    return success();
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

static bool registerExamplePlugin(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<MySession>(IREE_EXAMPLE_PLUGIN_ID_STR);
  return true;
}

IREE_DEFINE_COMPILER_PLUGIN(IREE_EXAMPLE_PLUGIN_ID, registerExamplePlugin)
