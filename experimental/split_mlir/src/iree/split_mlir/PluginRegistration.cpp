// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "iree/split_mlir/Passes.h"


using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct SplitMlirOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct SplitMlirSession : public PluginSession<SplitMlirSession, SplitMlirOptions> {
  static void registerPasses() {
    iree::split_mlir::registerPasses();
  }
};
}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(SplitMlirOptions);

extern "C" bool iree_register_compiler_plugin_split_mlir(PluginRegistrar *registrar) {
  registrar->registerPlugin<SplitMlirSession>("split_mlir");
  return true;
}
