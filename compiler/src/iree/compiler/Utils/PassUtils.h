// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_FUNCTIONUTILS_H_
#define IREE_COMPILER_UTILS_FUNCTIONUTILS_H_

#include <array>

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

/// Constructs a pipeline of passes across multiple nested op types.
///
/// Usage:
///   using FunctionLikeNest = MultiOpNest<IREE::Util::InitializerOp,
///                                        IREE::Util::FuncOp>;
///
///   FunctionLikeNest(passManager)
///     .addPass(createMyPass)
///     .addPredicatedPass(enable, createMyOtherPass);
template <typename... OpTys>
struct MultiOpNest {
public:
  MultiOpNest(OpPassManager &parentPm) : parentPm(parentPm) {
    addNest<0, OpTys...>();
  }

  template <typename F>
  MultiOpNest &addPass(F constructor) {
    addPassInternal(constructor);
    return *this;
  }

  // We have an explicit overload for a concrete function to support
  // passing overload constructors (i.e. createCanonicalizerPass).
  MultiOpNest &addPass(std::unique_ptr<Pass> (*constructor)()) {
    addPassInternal(constructor);
    return *this;
  }

  template <typename F>
  MultiOpNest &addPredicatedPass(bool enable, F constructor) {
    if (enable) {
      addPassInternal(constructor);
    }
    return *this;
  }

private:
  // Initialize a nest.
  template <int index, typename T, typename... Rest>
  void addNest() {
    std::get<index>(nestedPassManagers) = &parentPm.nest<T>();
    addNest<index + 1, Rest...>();
  }
  template <int index>
  void addNest() {}

  // Add a pass to all nests by constructor.
  template <typename F>
  void addPassInternal(F constructor) {
    addPassRecurse<F, 0, OpTys...>(constructor);
  }
  template <typename F, int index, typename T, typename... Rest>
  void addPassRecurse(F constructor) {
    std::get<index>(nestedPassManagers)->addPass(constructor());
    addPassRecurse<F, index + 1, Rest...>(constructor);
  }
  template <typename F, int index>
  void addPassRecurse(F constructor) {}

  OpPassManager &parentPm;
  std::array<OpPassManager *, sizeof...(OpTys)> nestedPassManagers;
};

// If running under a FixedPointIterator pass, annotate that a modification
// has been made which requires another iteration. No-op otherwise.
void signalFixedPointModified(Operation *rootOp);

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_FUNCTIONUTILS_H_
