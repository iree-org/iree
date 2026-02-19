// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ClangTidy.h"
#include "ClangTidyModuleRegistry.h"
#include "IREETestCheck.h"

namespace clang::tidy::iree {

class IREEModule : public ClangTidyModule {
 public:
  void addCheckFactories(ClangTidyCheckFactories &CheckFactories) override {
    // Register test check.
    CheckFactories.registerCheck<IREETestCheck>("iree-test");

    // Future checks will be registered here:
    // CheckFactories.registerCheck<IREECStatusLeakCheck>("iree-c-status-leak");
    // CheckFactories.registerCheck<IREECUnbalancedTraceZonesCheck>("iree-c-unbalanced-trace-zones");
    // etc.
  }
};

// Register the module with clang-tidy using its global registry.
static ClangTidyModuleRegistry::Add<IREEModule> X("iree-module",
                                                  "Adds IREE-specific checks.");

}  // namespace clang::tidy::iree

// This anchor is used to force the linker to link in the generated object file
// and thus register the module.
volatile int IREEClangTidyModuleAnchorSource = 0;
