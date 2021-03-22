// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    std::string passName = pass->getName().str();
    IREE_TRACE_ZONE_BEGIN_NAMED_DYNAMIC(z0, passName.data(), passName.size());
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
