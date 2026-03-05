// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_UTILS_PASSUTILS_H_
#define IREE_COMPILER_UTILS_PASSUTILS_H_

#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include "llvm/ADT/DenseMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler {

// If running under a FixedPointIterator pass, annotate that a modification
// has been made which requires another iteration. No-op otherwise.
void signalFixedPointModified(Operation *rootOp);

//===----------------------------------------------------------------------===//
// PipelineCache
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// OpPipelineAdaptorPass
//===----------------------------------------------------------------------===//

namespace detail {

/// An unregistered pass that adapts a parent pass manager to run sub-pipelines
/// on child operations. The pass walks top-level operations in all regions of
/// the parent operation, evaluates conditions in order for each child, and runs
/// the first matching sub-pipeline. Children that match no condition are
/// skipped.
///
/// When MLIR multithreading is enabled, child operations are dispatched in
/// parallel using the same pattern as OpToOpPassAdaptor: sub-pipelines are
/// cloned per thread, and operations are processed via parallelForEach.
///
/// This pass is used internally by MultiPipelineNest and is not intended for
/// direct use.
class OpPipelineAdaptorPass final
    : public PassWrapper<OpPipelineAdaptorPass, OperationPass<>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OpPipelineAdaptorPass)

  /// Predicate that determines whether a pipeline should run on a given
  /// operation. Conditions must only inspect immutable properties of the
  /// operation (e.g., operation name, type via isa<T>). In async mode,
  /// all conditions are evaluated eagerly before any pipeline runs, so
  /// conditions must not depend on IR state that a pipeline might modify.
  using ConditionFn = std::function<bool(Operation *)>;

  /// A condition+pipeline pair. When the condition matches an operation,
  /// the pipeline is run on it.
  struct Entry {
    ConditionFn condition;
    OpPassManager pipeline;
    /// TypeID of the op type this entry targets (set by nest<T>()). Entries
    /// without a TypeID cannot participate in automatic merging.
    std::optional<TypeID> opTypeID;
    /// Batch index for merge-aware dispatch. Entries from the same original
    /// MultiPipelineNest share a batch index. Within a batch, first-match-wins
    /// semantics apply. Across batches (from merged adaptors), all matching
    /// batches run.
    unsigned batchIndex = 0;

    Entry(ConditionFn condition, OpPassManager pipeline,
          std::optional<TypeID> opTypeID = std::nullopt,
          unsigned batchIndex = 0)
        : condition(std::move(condition)), pipeline(std::move(pipeline)),
          opTypeID(opTypeID), batchIndex(batchIndex) {}
  };

  /// Construct with no entries. Entries are added later via addEntry().
  OpPipelineAdaptorPass() = default;

  /// Copy for thread-safe cloning. OpPassManagers are deep-copied;
  /// ConditionFn objects are value-copied (safe because they are read-only).
  OpPipelineAdaptorPass(const OpPipelineAdaptorPass &other);

  StringRef getArgument() const override { return "iree-op-pipeline-adaptor"; }

  StringRef getDescription() const override {
    return "Adapts a parent pass manager to run sub-pipelines on child "
           "operations based on conditions.";
  }

  /// Add a condition+pipeline entry. Returns the pipeline for the caller
  /// to populate with passes. Must only be called during pipeline
  /// construction, before the pass manager is run.
  OpPassManager &addEntry(ConditionFn condition, StringRef anchorOpName = "",
                          std::optional<TypeID> opTypeID = std::nullopt);

  /// Merge entries from another adaptor into this one. Entries from |other|
  /// are assigned a new batch index so that dispatch preserves
  /// first-match-wins within each original nest scope while running all
  /// matching batches. |other| is left with empty entries after the merge.
  void mergeFrom(OpPipelineAdaptorPass &other);

  /// Returns true if this adaptor has no entries.
  bool empty() const { return entries.empty(); }

  /// Returns true if all entries have TypeID annotations (required for
  /// automatic merging).
  bool allEntriesHaveTypeID() const;

  /// Add a pass to all existing sub-pipelines.
  template <typename F>
  void addPassToAll(F constructor) {
    for (Entry &entry : entries) {
      entry.pipeline.addPass(constructor());
    }
  }

  /// Add a pass only to sub-pipelines targeting the given op type.
  template <typename F>
  void addPassToEntriesWithTypeID(TypeID targetID, F constructor) {
    for (Entry &entry : entries) {
      if (entry.opTypeID == targetID) {
        entry.pipeline.addPass(constructor());
      }
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

private:
  /// Run dispatch synchronously (single-threaded).
  void runOnOperationSync();

  /// Run dispatch in parallel using parallelForEach.
  void runOnOperationAsync();

  /// The condition+pipeline entries. First match wins.
  SmallVector<Entry> entries;

  /// Per-thread copies of entries for parallel execution. Lazily initialized
  /// when multithreading is enabled. Each element is a complete copy of
  /// `entries` for one thread.
  SmallVector<SmallVector<Entry>> asyncExecutors;
};

} // namespace detail

//===----------------------------------------------------------------------===//
// MultiPipelineNest
//===----------------------------------------------------------------------===//

/// Builder for conditional pipeline dispatch over child operations. Each
/// sub-pipeline is guarded by a condition predicate; the first matching
/// condition wins at runtime. When MLIR multithreading is enabled, matched
/// operations are processed in parallel.
///
/// Pass insertion is deferred: the adaptor pass is owned by this builder and
/// only inserted into the parent pass manager on destruction (or via an
/// explicit commitPass() call). On destruction, the builder first attempts to
/// merge with the immediately preceding adaptor pass in the parent PM (if
/// both have TypeID-annotated entries). If merging succeeds, the pass is
/// never inserted and no empty shell remains. If merging fails (or there is
/// no compatible predecessor), the pass is inserted at the back of the PM.
///
/// For most usage patterns (temporaries destroyed at end-of-statement), the
/// deferred insertion produces identical ordering to eager insertion. When
/// a named MultiPipelineNest must coexist with interleaved parent-level
/// passes, call commitPass() at the desired insertion point.
///
/// Usage:
///   // Typical: temporary destroyed at semicolon — pass inserted here.
///   MultiPipelineNest(pm).nest<FuncOp>().addPass(createMyPass);
///
///   // Named variable with explicit commit point:
///   MultiPipelineNest nest(pm);
///   nest.nest<FuncOp>();
///   nest.commitPass();           // insert now, before subsequent passes
///   pm.addPass(createOtherPass); // guaranteed to be after nest's pass
class MultiPipelineNest {
public:
  using ConditionFn = std::function<bool(Operation *)>;

  /// Construct a builder targeting \p parentPm. The adaptor pass is created
  /// but NOT yet inserted into the PM. It will be inserted on destruction
  /// (possibly merged with the predecessor) or when commitPass() is called.
  explicit MultiPipelineNest(OpPassManager &parentPm);

  /// On destruction: if the pass has not been committed, attempt to merge
  /// with the predecessor adaptor. If merge fails, insert the pass.
  ~MultiPipelineNest();

  // Movable but not copyable. The moved-from object is left in a null state.
  MultiPipelineNest(MultiPipelineNest &&other) noexcept;
  MultiPipelineNest &operator=(MultiPipelineNest &&other) noexcept;
  MultiPipelineNest(const MultiPipelineNest &) = delete;
  MultiPipelineNest &operator=(const MultiPipelineNest &) = delete;

  /// Immediately insert the pass into the parent PM at the current position.
  /// After this call, the pass is owned by the PM and no merge will be
  /// attempted on destruction. Use this when interleaving parent-level and
  /// nested passes with a named MultiPipelineNest variable.
  void commitPass();

  /// Add a sub-pipeline that runs when the condition returns true.
  /// Returns the OpPassManager for the caller to populate with passes.
  /// The first matching condition wins at runtime.
  ///
  /// NOTE: The returned reference is invalidated by subsequent nestIf/nest
  /// calls (due to SmallVector reallocation). Use addPassFor<OpT>() for
  /// safe per-type pass addition after all entries are created.
  OpPassManager &nestIf(ConditionFn condition, StringRef anchorOpName = "",
                        std::optional<TypeID> opTypeID = std::nullopt);

  /// Convenience: add a sub-pipeline for a specific op type. The pipeline
  /// is anchored to OpT so that typed passes can be added directly.
  /// Records the TypeID for automatic merging.
  template <typename OpT>
  OpPassManager &nest() {
    return nestIf([](Operation *op) { return isa<OpT>(op); },
                  OpT::getOperationName(), TypeID::get<OpT>());
  }

  /// Variadic convenience: add sub-pipelines for multiple op types at once.
  /// Equivalent to calling nest<T>() for each type individually.
  template <typename T1, typename T2, typename... Rest>
  void nest() {
    nest<T1>();
    nest<T2>();
    (nest<Rest>(), ...);
  }

  /// Add a pass to ALL existing sub-pipelines.
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiPipelineNest &addPass(F constructor) {
    adaptorPass->addPassToAll(constructor);
    return *this;
  }

  /// Add a pass to ALL existing sub-pipelines if the predicate is true.
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiPipelineNest &addPredicatedPass(bool enable, F constructor) {
    if (enable) {
      addPass(constructor);
    }
    return *this;
  }

  /// Add a pass only to sub-pipelines targeting the given op type.
  /// This is safe to call at any point (unlike holding a reference from
  /// nest<T>(), which can be invalidated by subsequent nest calls).
  template <typename OpT, typename F = std::unique_ptr<Pass> (*)()>
  MultiPipelineNest &addPassFor(F constructor) {
    adaptorPass->addPassToEntriesWithTypeID(TypeID::get<OpT>(), constructor);
    return *this;
  }

private:
  /// Try to merge this adaptor's entries into the last pass in the parent PM
  /// (if it is a compatible OpPipelineAdaptorPass). Returns true on success.
  bool tryMergeIntoPredecessor();

  /// Pointer to the parent pass manager.
  OpPassManager *parentPm = nullptr;

  /// Size of the parent PM at construction time. Used to assert that no
  /// passes were added to the parent PM between construction and commit.
  size_t parentPmSizeAtConstruction = 0;

  /// The adaptor pass. Owned here until commitPass() or destructor transfers
  /// it to the parent PM (or discards it after a successful merge).
  std::unique_ptr<detail::OpPipelineAdaptorPass> ownedPass;

  /// Raw pointer for convenient access (always == ownedPass.get() while
  /// owned, or the raw pointer into the PM after commitPass()).
  detail::OpPipelineAdaptorPass *adaptorPass = nullptr;
};

//===----------------------------------------------------------------------===//
// MultiOpNest
//===----------------------------------------------------------------------===//

/// Constructs a pipeline of passes across multiple nested op types.
/// Uses MultiPipelineNest internally, enabling parallel dispatch across
/// all op types when MLIR multithreading is enabled.
///
/// When used as a temporary (the typical pattern), the adaptor pass is
/// inserted at the end of the full expression. Adjacent temporaries are
/// automatically merged into a single dispatch pass.
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
  MultiOpNest(OpPassManager &parentPm) : nest(parentPm) {
    nest.template nest<OpTys...>();
  }

  // We give the template param a default to support passing overload
  // constructors (i.e. createCanonicalizerPass).
  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiOpNest &addPass(F constructor) {
    nest.addPass(constructor);
    return *this;
  }

  template <typename F = std::unique_ptr<Pass> (*)()>
  MultiOpNest &addPredicatedPass(bool enable, F constructor) {
    nest.addPredicatedPass(enable, constructor);
    return *this;
  }

  /// Immediately insert the adaptor pass into the parent PM. Use this when
  /// a named MultiOpNest must be committed before interleaved parent passes.
  void commitPass() { nest.commitPass(); }

private:
  MultiPipelineNest nest;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_UTILS_PASSUTILS_H_
