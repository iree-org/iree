// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

// The plugin id is a build parameter so that this one source can be both
// linked into the compiler and dlopen'd by it within the same process, which
// is what test/dynamic_link.mlir compares. Everything the process registers
// globally has to be derived from it: registering one plugin id twice aborts,
// and llvm::cl asserts on both a duplicate option name and a duplicate option
// category name.
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
    // Deliberately identical for every build of this source: the static and
    // dynamic tests assert on the same text.
    mlir::emitRemark(mlir::UnknownLoc::get(context))
        << "This remark is from the example plugin activation (flag="
        << options.flag << ")";
    return success();
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool IREE_EXAMPLE_REGISTER_PLUGIN(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<MySession>(IREE_EXAMPLE_PLUGIN_ID_STR);
  return true;
}
