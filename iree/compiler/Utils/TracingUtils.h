// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_TRACINGUTILS_H_
#define IREE_COMPILER_UTILS_TRACINGUTILS_H_

#include "iree/base/tracing.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassInstrumentation.h"

namespace mlir {
namespace iree_compiler {

// Instruments passes using IREE's runtime tracing support.
//
// Usage:
//   passManager.addInstrumentation(std::make_unique<PassTracing>());
struct PassTracing : public PassInstrumentation {
  PassTracing() {}
  ~PassTracing() override = default;

#if IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION
  // Note: we could also trace pipelines and analyses.

  void runBeforePass(Pass *pass, Operation *op) override {
    IREE_TRACE_ZONE_BEGIN_EXTERNAL(z0, __FILE__, strlen(__FILE__), __LINE__,
                                   pass->getName().data(),
                                   pass->getName().size(), NULL, 0);
    passTraceZonesStack.push_back(z0);
  }
  void runAfterPass(Pass *pass, Operation *op) override {
    IREE_TRACE_ZONE_END(passTraceZonesStack.back());
    passTraceZonesStack.pop_back();
  }
  void runAfterPassFailed(Pass *pass, Operation *op) override {
    IREE_TRACE_ZONE_END(passTraceZonesStack.back());
    passTraceZonesStack.pop_back();
  }
#endif  // IREE_TRACING_FEATURES & IREE_TRACING_FEATURE_INSTRUMENTATION

  llvm::SmallVector<iree_zone_id_t, 8> passTraceZonesStack;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_UTILS_TRACINGUTILS_H_
