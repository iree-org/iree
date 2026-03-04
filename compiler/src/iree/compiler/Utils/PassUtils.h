// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PASSUTILS_H_
#define IREE_COMPILER_UTILS_PASSUTILS_H_

#include <array>
#include <memory>
#include <mutex>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

// Thread-safe cache for compiled pass pipelines keyed by target attribute.
// When multiple executable variants share the same target attribute, the pass
// pipeline only needs to be constructed once. The cache is shared across clones
// of the outer pass that MLIR creates for parallel execution on different
// ExecutableOps via a shared_ptr.
//
// getOrCreate() returns a deep copy of the cached pipeline rather than a
// reference because MLIR passes carry mutable state (analysis caches,
// statistics) that is modified during execution. The outer per-ExecutableOp
// passes run in parallel, so two threads processing different executables with
// the same target attribute would race on a shared OpPassManager. The copy cost
// is negligible compared to pipeline execution; the savings come from avoiding
// redundant pipeline construction (registry lookups, dynamic pass creation) for
// every variant.
struct PipelineCache {
  std::mutex mutex;
  llvm::DenseMap<Attribute, std::unique_ptr<OpPassManager>> entries;

  // Returns a deep copy of the cached pipeline for |targetAttr|, building it
  // on first access using |builder|. Thread-safe.
  OpPassManager getOrCreate(Attribute targetAttr, StringRef operationName,
                            llvm::function_ref<void(OpPassManager &)> builder) {
    std::lock_guard<std::mutex> lock(mutex);
    auto &entry = entries[targetAttr];
    if (!entry) {
      entry = std::make_unique<OpPassManager>(operationName);
      builder(*entry);
    }
    return OpPassManager(*entry);
  }
};

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

  // We give the template param a default to support passing overload
  // constructors (i.e. createCanonicalizerPass).
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiOpNest &addPass(F constructor) {
    addPassInternal(constructor);
    return *this;
  }

  template <typename F = std::unique_ptr<Pass> (*)()>
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

#endif // IREE_COMPILER_UTILS_PASSUTILS_H_
