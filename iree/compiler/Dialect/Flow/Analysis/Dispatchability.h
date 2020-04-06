// Copyright 2019 Google LLC
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

#ifndef IREE_COMPILER_DIALECT_FLOW_ANALYSIS_DISPATCHABILITY_H_
#define IREE_COMPILER_DIALECT_FLOW_ANALYSIS_DISPATCHABILITY_H_

#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {

// Analyzes functions in a module to determine whether they can be performed as
// part of a dispatch operation. Functions must meet a set of criteria defining
// "dispatchability" such as the lack of side effects.
class Dispatchability {
 public:
  // Annotates the IR with the dispatchability information. This is only
  // required if the dispatchability information is interesting to persist
  // beyond transformation, such as in tests.
  static LogicalResult annotateIR(ModuleOp moduleOp);

  Dispatchability() = default;
  explicit Dispatchability(Operation *op) { recalculate(cast<ModuleOp>(op)); }
  Dispatchability(Dispatchability &&) = default;
  Dispatchability &operator=(Dispatchability &&) = default;
  Dispatchability(const Dispatchability &) = delete;
  Dispatchability &operator=(const Dispatchability &) = delete;

  // Recalculates the dispatchability information for the given module.
  LogicalResult recalculate(ModuleOp moduleOp);

  // Calls |fn| for each dispatchable function.
  void walkDispatchableOps(function_ref<void(FuncOp funcOp)> fn);

  // Returns true if |funcOp| is dispatchable.
  bool isDispatchable(StringRef funcName);
  bool isDispatchable(FuncOp funcOp);
  bool isInvalidated(const AnalysisManager::PreservedAnalyses &pa);

 private:
  // Returns true if the given function is dispatch compatible.
  // Returns None if the dispatchability can't yet be calculated as dependent
  // functions have not been processed.
  Optional<bool> computeDispatchability(FuncOp funcOp);

  DenseMap<StringRef, bool> funcDispatchability_;
  OwningModuleRef funcCloneModuleOp_;
  DenseMap<StringRef, FuncOp> funcClones_;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_FLOW_ANALYSIS_DISPATCHABILITY_H_
