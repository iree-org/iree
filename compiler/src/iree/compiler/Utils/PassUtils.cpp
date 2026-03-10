// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Utils/PassUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/Threading.h"

#define DEBUG_TYPE "iree-pass-utils"

namespace mlir::iree_compiler {

void signalFixedPointModified(Operation *rootOp) {
  MLIRContext *context = rootOp->getContext();
  if (!rootOp->hasAttr("iree.fixedpoint.iteration")) {
    LLVM_DEBUG(llvm::dbgs() << "Not signaling fixed-point modification: not "
                               "running under fixed point iterator");
    return;
  }

  LLVM_DEBUG(llvm::dbgs() << "Signalling fixed-point iterator modification");
  rootOp->setAttr("iree.fixedpoint.modified", UnitAttr::get(context));
}

//===----------------------------------------------------------------------===//
// OpPipelineAdaptorPass
//===----------------------------------------------------------------------===//

namespace detail {

OpPipelineAdaptorPass::OpPipelineAdaptorPass(const OpPipelineAdaptorPass &other)
    : PassWrapper(other) {
  entries.reserve(other.entries.size());
  for (const Entry &e : other.entries) {
    entries.emplace_back(e.condition, OpPassManager(e.pipeline), e.opTypeID,
                         e.batchIndex);
  }
}

OpPassManager &OpPipelineAdaptorPass::addEntry(ConditionFn condition,
                                               StringRef anchorOpName,
                                               std::optional<TypeID> opTypeID) {
  if (anchorOpName.empty()) {
    entries.emplace_back(std::move(condition), OpPassManager(), opTypeID);
  } else {
    entries.emplace_back(std::move(condition), OpPassManager(anchorOpName),
                         opTypeID);
  }
  return entries.back().pipeline;
}

bool OpPipelineAdaptorPass::allEntriesHaveTypeID() const {
  return llvm::all_of(entries,
                      [](const Entry &e) { return e.opTypeID.has_value(); });
}

void OpPipelineAdaptorPass::mergeFrom(OpPipelineAdaptorPass &other) {
  // Determine the next batch index for incoming entries.
  unsigned newBatch = 0;
  for (const Entry &e : entries) {
    newBatch = std::max(newBatch, e.batchIndex + 1);
  }
  for (Entry &e : other.entries) {
    e.batchIndex = newBatch;
    entries.push_back(std::move(e));
  }
  other.entries.clear();
  // Invalidate async executors since entries changed.
  asyncExecutors.clear();
}

void OpPipelineAdaptorPass::getDependentDialects(
    DialectRegistry &registry) const {
  for (const Entry &e : entries) {
    e.pipeline.getDependentDialects(registry);
  }
}

void OpPipelineAdaptorPass::runOnOperation() {
  if (getContext().isMultithreadingEnabled()) {
    runOnOperationAsync();
  } else {
    runOnOperationSync();
  }
}

void OpPipelineAdaptorPass::runOnOperationSync() {
  Operation *parentOp = getOperation();
  LDBG() << "Running op pipeline adaptor synchronously on '"
         << parentOp->getName() << "'";

  for (Region &region : parentOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      // Within a batch, first match wins. Across batches (from merged
      // adaptors), all matching batches run.
      std::optional<unsigned> lastMatchedBatch;
      for (Entry &e : entries) {
        if (lastMatchedBatch == e.batchIndex) {
          continue;
        }
        if (e.condition(&op)) {
          LDBG() << "  Dispatching '" << op.getName() << "' to pipeline"
                 << " (batch " << e.batchIndex << ")";
          if (failed(runPipeline(e.pipeline, &op))) {
            return signalPassFailure();
          }
          lastMatchedBatch = e.batchIndex;
        }
      }
    }
  }
}

void OpPipelineAdaptorPass::runOnOperationAsync() {
  Operation *parentOp = getOperation();
  MLIRContext *context = &getContext();

  LDBG() << "Running op pipeline adaptor asynchronously on '"
         << parentOp->getName() << "'";

  // --- Phase 1: Collect matched operations (sequential). ---

  // Information for a single matched operation. An op may match multiple
  // entries when batches from merged adaptors are present.
  struct OpDispatchInfo {
    Operation *op = nullptr;
    SmallVector<unsigned, 2> entryIndices;
  };

  SmallVector<OpDispatchInfo> opInfos;
  for (Region &region : parentOp->getRegions()) {
    for (Operation &op : region.getOps()) {
      OpDispatchInfo info;
      info.op = &op;
      // Within a batch, first match wins. Across batches, all matching
      // batches run.
      std::optional<unsigned> lastMatchedBatch;
      for (auto [i, entry] : llvm::enumerate(entries)) {
        if (lastMatchedBatch == entry.batchIndex) {
          continue;
        }
        if (entry.condition(&op)) {
          info.entryIndices.push_back(i);
          lastMatchedBatch = entry.batchIndex;
        }
      }
      if (!info.entryIndices.empty()) {
        opInfos.push_back(std::move(info));
      }
    }
  }

  if (opInfos.empty()) {
    return;
  }

  // Pre-create nested analysis managers so that the dynamic pipeline
  // callback (invoked from runPipeline) only performs lookups -- not
  // insertions -- on the parent AnalysisManager's child map during the
  // parallel section. This is required because DenseMap insertion is not
  // thread-safe.
  AnalysisManager am = getAnalysisManager();
  for (OpDispatchInfo &info : opInfos) {
    (void)am.nest(info.op);
  }

  // --- Phase 2: Ensure per-thread entry clones exist. ---

  unsigned numThreads = context->getThreadPool().getMaxConcurrency();
  // Entries are immutable after pipeline construction, so size comparison
  // suffices for staleness detection.
  if (asyncExecutors.empty() ||
      asyncExecutors.front().size() != entries.size()) {
    LDBG() << "  Creating " << numThreads << " async executors with "
           << entries.size() << " entries each";
    asyncExecutors.clear();
    asyncExecutors.resize(numThreads);
    for (SmallVectorImpl<Entry> &executor : asyncExecutors) {
      executor.reserve(entries.size());
      for (const Entry &e : entries) {
        executor.emplace_back(e.condition, OpPassManager(e.pipeline),
                              e.opTypeID, e.batchIndex);
      }
    }
  }

  // --- Phase 3: Parallel dispatch. ---

  // Track which executors are in use (one per thread). Using unique_ptr<T[]>
  // because std::atomic is neither copyable nor moveable, which is required
  // by std::vector's contract even if we never resize.
  auto activeExecutors =
      std::make_unique<std::atomic<bool>[]>(asyncExecutors.size());
  std::atomic<bool> hasFailure(false);

  // NOTE: Using the dynamic pipeline API (Pass::runPipeline) rather than
  // the internal OpToOpPassAdaptor::runPipeline means instrumentation events
  // will report the parent thread ID rather than the actual worker thread.
  // This is acceptable since OpPipelineAdaptorPass is an internal
  // implementation detail.
  //
  // The number of active executors will never exceed numThreads because
  // parallelForEach dispatches at most that many concurrent tasks.
  parallelForEach(context, opInfos, [&](OpDispatchInfo &info) {
    // Claim an inactive executor via atomic compare-and-swap.
    unsigned executorIdx = 0;
    for (unsigned i = 0, e = asyncExecutors.size(); i < e; ++i) {
      bool expected = false;
      if (activeExecutors[i].compare_exchange_strong(expected, true)) {
        executorIdx = i;
        break;
      }
    }

    // Run all matched pipelines from this executor's clone. Multiple entries
    // can match when batches from merged adaptors are present.
    for (unsigned entryIdx : info.entryIndices) {
      OpPassManager &pm = asyncExecutors[executorIdx][entryIdx].pipeline;
      if (failed(runPipeline(pm, info.op))) {
        hasFailure.store(true);
        break;
      }
    }

    // Release the executor.
    activeExecutors[executorIdx].store(false);
  });

  if (hasFailure) {
    signalPassFailure();
  }
}

} // namespace detail

//===----------------------------------------------------------------------===//
// MultiPipelineNest
//===----------------------------------------------------------------------===//

MultiPipelineNest::MultiPipelineNest(OpPassManager &parentPm)
    : parentPm(&parentPm), parentPmSizeAtConstruction(parentPm.size()),
      ownedPass(std::make_unique<detail::OpPipelineAdaptorPass>()),
      adaptorPass(ownedPass.get()) {}

MultiPipelineNest::~MultiPipelineNest() {
  // Nothing to do if moved-from, already committed, or no entries added.
  if (!adaptorPass || !ownedPass || adaptorPass->empty()) {
    return;
  }
  assert(parentPm->size() == parentPmSizeAtConstruction &&
         "passes were added to the parent PM between MultiPipelineNest "
         "construction and destruction; use commitPass() to insert the "
         "adaptor at the desired position before adding parent passes");
  // Try to merge into the predecessor. On success the entries are moved
  // and ownedPass is destroyed (now empty) when we return.
  if (tryMergeIntoPredecessor()) {
    return;
  }
  // No merge -- insert the pass into the parent PM.
  parentPm->addPass(std::move(ownedPass));
}

MultiPipelineNest::MultiPipelineNest(MultiPipelineNest &&other)
    : parentPm(other.parentPm),
      parentPmSizeAtConstruction(other.parentPmSizeAtConstruction),
      ownedPass(std::move(other.ownedPass)), adaptorPass(other.adaptorPass) {
  other.parentPm = nullptr;
  other.adaptorPass = nullptr;
}

MultiPipelineNest &MultiPipelineNest::operator=(MultiPipelineNest &&other) {
  if (this != &other) {
    // Flush the current pass before replacing.
    if (adaptorPass && ownedPass && !adaptorPass->empty()) {
      if (!tryMergeIntoPredecessor()) {
        assert(parentPm->size() == parentPmSizeAtConstruction &&
               "passes were added to the parent PM between MultiPipelineNest "
               "construction and destruction; use commitPass() to insert the "
               "adaptor at the desired position before adding parent passes");
        parentPm->addPass(std::move(ownedPass));
      }
    }
    parentPm = other.parentPm;
    parentPmSizeAtConstruction = other.parentPmSizeAtConstruction;
    ownedPass = std::move(other.ownedPass);
    adaptorPass = other.adaptorPass;
    other.parentPm = nullptr;
    other.adaptorPass = nullptr;
  }
  return *this;
}

void MultiPipelineNest::commitPass() {
  if (!ownedPass) {
    return; // Already committed.
  }
  // No entries were added -- discard the pass silently.
  if (adaptorPass->empty()) {
    ownedPass.reset();
    return;
  }
  assert(parentPm->size() == parentPmSizeAtConstruction &&
         "passes were added to the parent PM between MultiPipelineNest "
         "construction and commitPass(); call commitPass() before adding "
         "parent passes");
  parentPm->addPass(std::move(ownedPass));
  // adaptorPass remains valid -- it now points into the PM.
}

bool MultiPipelineNest::tryMergeIntoPredecessor() {
  // Can't merge if we have no entries or any lack TypeIDs.
  if (adaptorPass->empty() || !adaptorPass->allEntriesHaveTypeID()) {
    return false;
  }

  // Our pass is NOT in the PM (deferred). Check the last pass in the PM.
  if (parentPm->empty()) {
    return false;
  }
  Pass &lastPass = *std::prev(parentPm->end());
  if (lastPass.getTypeID() != TypeID::get<detail::OpPipelineAdaptorPass>()) {
    return false;
  }
  auto &predAdaptor = static_cast<detail::OpPipelineAdaptorPass &>(lastPass);
  // Predecessor must have entries and all must have TypeIDs.
  if (predAdaptor.empty() || !predAdaptor.allEntriesHaveTypeID()) {
    return false;
  }
  predAdaptor.mergeFrom(*adaptorPass);
  return true;
}

OpPassManager &MultiPipelineNest::nestIf(ConditionFn condition,
                                         StringRef anchorOpName,
                                         std::optional<TypeID> opTypeID) {
  return adaptorPass->addEntry(std::move(condition), anchorOpName, opTypeID);
}

} // namespace mlir::iree_compiler
