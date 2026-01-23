// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Utils/IntegerSet.h"
#include "iree/compiler/Utils/ModuleUtils.h"
#include "iree/compiler/Utils/RegionOpUtils.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/CSE.h"
#include "mlir/Transforms/RegionUtils.h"

#define DEBUG_TYPE "iree-stream-split-parameter-encoder"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_SPLITPARAMETERENCODERPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Collection of IndexSets for managing index memoization across functions.
class IndexSetCollection {
public:
  // Returns an index set for the parent function of |op|.
  IndexSet *get(Operation *op) {
    auto parentOp = op->getParentOfType<mlir::FunctionOpInterface>();
    auto it = funcMap.find(parentOp);
    if (it != funcMap.end()) {
      return it->second.get();
    }
    auto indexSet = std::make_unique<IndexSet>(
        op->getLoc(), OpBuilder::atBlockBegin(&parentOp.front()));
    IndexSet *indexSetPtr = indexSet.get();
    funcMap.insert({parentOp, std::move(indexSet)});
    return indexSetPtr;
  }

private:
  DenseMap<Operation *, std::unique_ptr<IndexSet>> funcMap;
};

// Erases all ops in |leafOps| and all of their potentially newly-dead
// transitive producer dependencies.
//
// This custom DCE is required because MLIR's standard mlir::dce::removeDeadCode
// doesn't handle two cases we need:
// 1. Ops implementing HoistableOpInterface - control flow ops like scf.for/if
//    whose bodies contain only hoistable (pure) operations can be deleted.
// 2. Ops with MemoryEffects::Read but no Write - these are "pure-ish" ops that
//    can't be marked Pure (which would allow CSE) but are still safe to delete.
//
// TODO(benvanik): figure out how to move this to RegionOpUtils - it relies on
// some util op interfaces, though, so is hard to get out there now.
static void pruneDeadOps(ArrayRef<Operation *> leafOps) {
  SmallVector<Operation *> deadOpWorklist{leafOps};

  // Use a DenseSet to track already-processed operations to avoid duplicate
  // processing when operations appear multiple times in the worklist.
  DenseSet<Operation *> processedOps;
  while (!deadOpWorklist.empty()) {
    Operation *op = deadOpWorklist.pop_back_val();

    // Skip if we've already processed this operation.
    if (!processedOps.insert(op).second) {
      continue;
    }

    // Skip if the operation is no longer trivially dead (may have been
    // deleted already or gained new uses).
    // Also elide ops with no uses that have MemoryEffects::Effect::Read but no
    // writes - these match the semantics of canonicalization ElideUnusedOp
    // patterns (ops that are pure-ish but can't be marked Pure due to CSE).
    bool canDelete = mlir::isOpTriviallyDead(op);
    if (!canDelete && op->use_empty()) {
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
      if (memInterface) {
        SmallVector<MemoryEffects::EffectInstance> effects;
        memInterface.getEffects(effects);
        // Safe to delete if it only has Allocate/Read effects (no Write).
        canDelete =
            llvm::none_of(effects, [](const MemoryEffects::EffectInstance &it) {
              return isa<MemoryEffects::Write>(it.getEffect());
            });
      } else if (auto hoistableOp =
                     dyn_cast<IREE::Util::HoistableOpInterface>(op)) {
        // Operations with HoistableOpInterface can be deleted if unused and
        // hoistable (pure). This handles control flow ops like scf.for/scf.if
        // whose bodies contain only hoistable operations.
        canDelete = hoistableOp.isHoistableOp();
      }
    }
    if (!canDelete) {
      continue;
    }

    // Collect defining operations before we delete this op.
    SmallVector<Operation *> producerOps;
    for (Value operand : op->getOperands()) {
      if (Operation *producer = operand.getDefiningOp()) {
        producerOps.push_back(producer);
      }
    }

    // Erase the dead operation.
    op->erase();

    // Check if any of the producers now have no uses and add them to the
    // worklist. The worklist loop will determine if they're safe to delete.
    for (Operation *producer : producerOps) {
      if (producer->use_empty()) {
        deadOpWorklist.push_back(producer);
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// EncodingExpr
//===----------------------------------------------------------------------===//

// Configuration controlling which expressions are hoisted to the encoder
// module. This policy determines hoisting eligibility based on expression type,
// size growth limits, and parameter/constant handling preferences.
struct EncodingPolicy {
  // Pack multiple parameters into larger slabs to reduce overheads.
  // This can dramatically improve startup time, reduces memory fragmentation,
  // and reduces dispatch overheads.
  bool packParameters = true; // false;
  // Include direct parameter loads that have no modifications.
  // When true the output parameter indices will have all required parameters
  // and any original parameters will not be required by the base program at
  // runtime. When false the user must provide the original parameters.
  bool includeUnmodified = true;
  // Any splat under this size will be serialized to the output parameter index
  // as if it were data instead of being embedded as a splat.
  // This increases the file size but allows for better parameter batching and
  // can reduce runtime overhead.
  int64_t serializeSplatSizeThreshold = 1024;

  // Enable hoisting parameter transformation expressions.
  // When true, expressions that transform parameters (parameter →
  // dispatch/encoding) will be extracted into the encoder module for offline
  // evaluation.
  bool hoistParameterExpressions = true;

  // Enable hoisting pure constant expressions with transformations.
  // When true, expressions that transform pure constants (constant →
  // dispatch/encoding) will be extracted into the encoder module for offline
  // evaluation.
  bool hoistConstantExpressions = true;

  // Maximum ratio of output size to input size before rejecting hoisting.
  // This prevents expressions that significantly increase storage from being
  // hoisted. Example: 1.2 allows 20% growth for padding/alignment.
  float maxEncodingGrowthFactor = 1.2f;
};

// An encoding expression represents a subgraph of operations that transforms
// input parameters/constants into output values stored to globals. Each
// expression can have multiple inputs (parameter loads) and multiple outputs
// (global stores). The expression is hoisted to the encoder module where it
// can be evaluated offline, with the results stored as pre-encoded parameters.
struct EncodingExpr {
  // Affinity of consumers of the expression in the original program.
  // All outputs share the same affinity.
  IREE::Stream::AffinityAttr affinityAttr;

  struct Input {
    // Inlined constant resource or parameter load.
    mutable IREE::Stream::AsyncConstantOp constantOp;

    Location getLoc() const { return constantOp.getLoc(); }

    // Returns true if the input is sourced from a parameter.
    bool isParameter() const {
      return isa<IREE::Stream::NamedParameterAttr>(constantOp.getValue());
    }
  };
  SmallVector<Input> inputs;

  struct Output {
    // Size in bytes of the output resource.
    int64_t size = 0;
    // Constant pattern value if this is a splat.
    TypedAttr splatPattern;
    // Sink op storing the produced output into a global.
    mutable IREE::Util::GlobalStoreOpInterface storeOp;
    // Produced value feeding into the store.
    // This may be either be directly consumed by the store or an op earlier in
    // the slice in cases where there are metadata ops we want to skip.
    Value producedValue;

    Location getLoc() const { return storeOp.getLoc(); }

    // Returns true if the output is a constant splat that needs no execution.
    // Only certain data types/widths are supported in the format and if not
    // supported natively we'll need to splat the value into the file. It's
    // rare for there to be splats that end up like this and it's unlikely the
    // user wants a file full of splatted values but at this point in the
    // pipeline we can only assume they asked for it.
    bool isSupportedSplat() const {
      if (!splatPattern || !splatPattern.getType().isIntOrFloat()) {
        return false;
      }
      const unsigned bitWidth = splatPattern.getType().getIntOrFloatBitWidth();
      return bitWidth == 8 || bitWidth == 16 || bitWidth == 32 ||
             bitWidth == 64;
    }
  };
  SmallVector<Output> outputs;

  // All operations (excluding outputs).
  SetVector<Operation *> ops;

  // Returns a fused location from all operations in the expression.
  Location getLoc() const {
    SetVector<Location> locs;
    for (auto *op : ops) {
      locs.insert(op->getLoc());
    }
    for (auto &output : outputs) {
      locs.insert(output.getLoc());
    }
    return FusedLoc::get(ops.front()->getContext(), locs.getArrayRef());
  }

  // Returns the resource config for the expression by checking all outputs.
  // If any outputs have differing configs
  IREE::Stream::ResourceConfigAttr getResourceConfigAttr() const {
    // Expressions should only be formed from outputs that share an affinity
    // so we can look at the first output and assume they all match.
    if (outputs.empty()) {
      return {};
    }
    auto globalStoreOp = outputs.front().storeOp;
    Value storedValue = globalStoreOp.getStoredGlobalValue();
    auto *producingOp = storedValue.getDefiningOp();
    return IREE::Stream::ResourceConfigAttr::lookup(
        producingOp ? producingOp : globalStoreOp);
  }

  // Returns true if the expression has any parameter inputs.
  bool hasParameterInputs() const {
    return llvm::any_of(inputs,
                        [](const Input &input) { return input.isParameter(); });
  }

  // Returns true if the expression has any constant inputs (non-parameter).
  bool hasConstantInputs() const {
    return llvm::any_of(
        inputs, [](const Input &input) { return !input.isParameter(); });
  }

  // Estimates total input size from all inputs in bytes.
  int64_t estimateInputSize() const {
    int64_t total = 0;
    for (const auto &input : inputs) {
      if (input.constantOp) {
        Value sizeValue = input.constantOp.getResultSize();
        APInt size;
        if (matchPattern(sizeValue, m_ConstantInt(&size))) {
          total += size.getZExtValue();
        }
      }
    }
    return total;
  }

  // Estimates total output size from all outputs in bytes.
  int64_t estimateOutputSize() const {
    int64_t total = 0;
    for (const auto &output : outputs) {
      total += output.size;
    }
    return total;
  }
};

struct EncodingExprSet {
  // All expressions terminating in parameter outputs in the order they were
  // originally present in the module (even if split across initializers).
  SmallVector<EncodingExpr> exprs;

  bool empty() const { return exprs.empty(); }
};

// Collects all external timepoint dependencies from the expression. This
// includes await timepoints from TimelineOpInterface ops in the expression that
// reference external values, and timepoints from external resource operands
// extracted via getResultTimepoint or by inserting a barrier.
static Value collectExternalTimepoints(const EncodingExpr &expr,
                                       OpBuilder &builder) {
  SetVector<Value> timepoints;

  // Build a set of ops that contribute RESOURCES (not just timepoints) to the
  // expression. An op is a "resource contributor" if at least one of its
  // non-timepoint results is used by another op in the expression.
  //
  // This distinction is important because the backward slice follows ALL
  // operands including await timepoints. Ops that only contribute timepoints
  // (like a timeline_op whose resource output is unused) should be considered
  // "external" for synchronization purposes - their timepoints need to be
  // awaited by the replacement op.
  DenseSet<Operation *> resourceContributors;
  for (Operation *op : expr.ops) {
    for (Value result : op->getResults()) {
      // Skip timepoint results - we only care about resource contributions.
      if (isa<IREE::Stream::TimepointType>(result.getType())) {
        continue;
      }
      // Check if any user of this non-timepoint result is in the expression.
      for (Operation *user : result.getUsers()) {
        if (expr.ops.contains(user)) {
          resourceContributors.insert(op);
          break;
        }
      }
      if (resourceContributors.contains(op)) {
        break;
      }
    }
  }

  // A timepoint is "internal" only if its defining op contributes resources
  // (not just timepoints) to the expression.
  auto isInternalTimepoint = [&](Value tp) -> bool {
    Operation *defOp = tp.getDefiningOp();
    return defOp && resourceContributors.contains(defOp);
  };

  // Collect external await timepoints from resource-contributing ops only.
  // We only look at resource contributors because:
  // 1. They represent the "core" data flow of the expression
  // 2. Non-resource-contributor ops (like joins, unused timeline ops) are
  //    "synchronization helpers" whose await timepoints are transitively
  //    covered by the resource contributors' awaits
  // This ensures we don't collect both a joined timepoint AND its component
  // timepoints when a join is in the expression but doesn't contribute
  // resources.
  for (Operation *op : resourceContributors) {
    auto timelineOp = dyn_cast<IREE::Stream::TimelineOpInterface>(op);
    if (!timelineOp) {
      continue;
    }
    for (Value awaitTp : timelineOp.getAwaitTimepoints()) {
      if (!isInternalTimepoint(awaitTp)) {
        timepoints.insert(awaitTp);
      }
    }
  }

  // A resource is "internal" only if its defining op contributes resources
  // (not just timepoints) to the expression.
  auto isInternalResource = [&](Value resource) -> bool {
    Operation *defOp = resource.getDefiningOp();
    return defOp && resourceContributors.contains(defOp);
  };

  // Collect timepoints from external resource operands.
  for (Operation *op : expr.ops) {
    for (Value operand : op->getOperands()) {
      if (!isa<IREE::Stream::ResourceType>(operand.getType())) {
        continue;
      }
      if (isInternalResource(operand)) {
        continue;
      }

      // Try to get timepoint from TimelineOpInterface.
      Value timepoint;
      Operation *definingOp = operand.getDefiningOp();
      if (definingOp) {
        if (auto timelineOp =
                dyn_cast<IREE::Stream::TimelineOpInterface>(definingOp)) {
          timepoint = timelineOp.getResultTimepoint();
        }
      }

      // If no timepoint available, insert barrier to extract it.
      if (!timepoint) {
        Value resourceSize = IREE::Util::SizeAwareTypeInterface::queryValueSize(
            operand.getLoc(), operand, builder);
        assert(resourceSize && "stream resource must have queryable size");
        auto affinityAttr = IREE::Stream::AffinityAttr::lookup(definingOp);
        auto barrierOp = IREE::Stream::TimepointBarrierOp::create(
            builder, operand.getLoc(), operand.getType(),
            builder.getType<IREE::Stream::TimepointType>(), operand,
            resourceSize, affinityAttr);
        timepoint = barrierOp.getResultTimepoint();
      }

      if (timepoint) {
        timepoints.insert(timepoint);
      }
    }
  }

  if (timepoints.empty()) {
    return {};
  }
  return IREE::Stream::joinTimepoints(
      expr.getLoc(), SmallVector<Value>(timepoints.begin(), timepoints.end()),
      builder);
}

// Finds all util.global.store-like ops that store constant resources in
// initializers. Stores are returned in program order.
//
// TODO: note that this does not check for stores in functions called by
// initializers and also does not currently check for variables (as they are
// usually uninitialized).
static SmallVector<IREE::Util::GlobalStoreOpInterface>
findAllConstantStoreOps(mlir::ModuleOp moduleOp) {
  SmallVector<IREE::Util::GlobalStoreOpInterface> storeOps;
  for (auto initializerOp :
       moduleOp.getOps<IREE::Util::InitializerOpInterface>()) {
    // Skip initializers that have CFGs. We don't handle conditional
    // initialization of globals today.
    auto &region = initializerOp.getInitializerRegion();
    if (!region.hasOneBlock()) {
      LLVM_DEBUG(DBGS() << "ignoring initializer as it has multiple blocks\n");
      continue;
    }
    // Find all stores. Note that we purposefully skip nested regions today.
    for (auto storeOp :
         region.front().getOps<IREE::Util::GlobalStoreOpInterface>()) {
      Type storedType = storeOp.getStoredGlobalValue().getType();
      if (auto resourceType =
              dyn_cast<IREE::Stream::ResourceType>(storedType)) {
        if (resourceType.getLifetime() == IREE::Stream::Lifetime::Constant) {
          storeOps.push_back(storeOp);
        }
      }
    }
  }
  return storeOps;
}

// Returns true if the operation's memory effects allow it to be hoisted as a
// const-expr operation. We allow Allocate and Free effects (memory management)
// but reject Read/Write effects to external memory.
static bool hasHoistableMemoryEffects(Operation *op) {
  auto effectInterface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectInterface) {
    // No memory effect interface means no effects - hoistable.
    return true;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  effectInterface.getEffects(effects);

  for (const auto &effect : effects) {
    // Allocate effects are fine (creating new memory).
    if (isa<MemoryEffects::Allocate>(effect.getEffect())) {
      continue;
    }
    // Free effects are also fine (releasing memory).
    if (isa<MemoryEffects::Free>(effect.getEffect())) {
      continue;
    }
    // Read or Write effects on non-result values are not const-expr.
    // Operations can write to their own results (that's how they produce
    // them), but reading/writing external memory is not allowed.
    if (isa<MemoryEffects::Read>(effect.getEffect()) ||
        isa<MemoryEffects::Write>(effect.getEffect())) {
      // Check if the effect is on a result of this op (allowed) or
      // on external memory (not allowed).
      if (Value value = llvm::dyn_cast_if_present<Value>(effect.getValue())) {
        // If it's a result of this op, it's fine.
        if (value.getDefiningOp() == op) {
          continue;
        }
      }
      // Read/Write to external memory - not const-expr.
      return false;
    }
  }

  return true;
}

static bool isConstExprOp(Operation *op) {
  // Optimization barriers cannot be folded.
  if (isa<IREE::Util::OptimizationBarrierOp>(op)) {
    return false;
  }

  // By default, ops without results are not const-expr.
  if (op->getNumResults() == 0) {
    return false;
  }

  // If implementing the HoistableOpInterface, just use the decision made by
  // the interface.
  if (auto hoistableOp = dyn_cast<IREE::Util::HoistableOpInterface>(op)) {
    return hoistableOp.isHoistableOp();
  }

  // Forbid if part of a parent that should be treated atomically.
  Operation *parent = op;
  while (auto hoistableParent =
             parent->getParentOfType<IREE::Util::HoistableOpInterface>()) {
    if (hoistableParent.isAtomicallyHoistableOp()) {
      return false;
    }
    parent = hoistableParent;
  }

  // Check memory effects: we allow Allocate effects (creating new memory)
  // but reject Read/Write effects to external memory. This is more permissive
  // than OpOracle.cpp's isMemoryEffectFree check, allowing operations like
  // stream.async.splat that allocate but don't have other side effects.
  return hasHoistableMemoryEffects(op);
}

static IREE::Stream::AffinityAttr
lookupConsumerAffinityAttr(Value storedValue) {
  if (auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(
          storedValue.getDefiningOp())) {
    return affinityOp.getResultAffinityAttr();
  }
  return IREE::Stream::AffinityAttr::lookupOrDefault(
      storedValue.getDefiningOp());
}

// Returns true if the expression producing |storedValue| is an input without
// any modification (such as inlined constants/parameters).
static bool isPassThroughStore(Value storedValue) {
  Operation *op = storedValue.getDefiningOp();
  do {
    if (auto transferOp = dyn_cast<IREE::Stream::AsyncTransferOp>(op)) {
      op = transferOp.getSource().getDefiningOp();
    } else if (auto constantOp = dyn_cast<IREE::Stream::AsyncConstantOp>(op)) {
      return true;
    } else {
      return false;
    }
  } while (op);
  return false;
}

// Returns the result index of |result| in the parent operation.
// The result must be a valid result of op.
static unsigned findResultIndex(Operation *op, Value result) {
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    if (op->getResult(i) == result) {
      return i;
    }
  }
  llvm_unreachable("result not found in operation");
}

// Attempts to evaluate a size value to a constant integer.
// This handles direct constants and analyzes through control flow operations
// where the size is provably constant (e.g., scf.if with matching branch
// sizes).
static std::optional<uint64_t> tryEvaluateConstantSize(Value sizeValue) {
  if (!sizeValue) {
    return std::nullopt;
  }

  // Try direct constant match (existing behavior).
  APInt size;
  if (matchPattern(sizeValue, m_ConstantInt(&size))) {
    return size.getZExtValue();
  }

  // For scf.if, check if both branches yield the same constant size.
  if (auto ifOp = sizeValue.getDefiningOp<scf::IfOp>()) {
    unsigned resultIndex = findResultIndex(ifOp, sizeValue);

    // Get the yielded values from both regions.
    auto thenYield =
        cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
    auto elseYield =
        cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());

    // Recursively evaluate both branch sizes.
    Value thenValue = thenYield.getOperand(resultIndex);
    Value elseValue = elseYield.getOperand(resultIndex);

    // Find sizes for the yielded resource values.
    auto thenSizeValue = IREE::Util::SizeAwareTypeInterface::findSizeValue(
        thenValue, &ifOp.getThenRegion().front(), Block::iterator(thenYield));
    auto elseSizeValue = IREE::Util::SizeAwareTypeInterface::findSizeValue(
        elseValue, &ifOp.getElseRegion().front(), Block::iterator(elseYield));

    auto thenSize = tryEvaluateConstantSize(thenSizeValue);
    auto elseSize = tryEvaluateConstantSize(elseSizeValue);

    // If both branches have the same constant size, return it.
    if (thenSize && elseSize && *thenSize == *elseSize) {
      return *thenSize;
    }

    return std::nullopt;
  }

  // For scf.for, check if the size is loop-invariant.
  if (auto forOp = sizeValue.getDefiningOp<scf::ForOp>()) {
    unsigned resultIndex = findResultIndex(forOp, sizeValue);

    // Check the initial value (iter_arg).
    Value initArg = forOp.getInitArgs()[resultIndex];
    auto initSizeValue = IREE::Util::SizeAwareTypeInterface::findSizeValue(
        initArg, forOp->getBlock(), Block::iterator(forOp));
    auto initSize = tryEvaluateConstantSize(initSizeValue);

    if (!initSize) {
      return std::nullopt;
    }

    // Check the yielded value in the loop body.
    auto yieldOp =
        cast<scf::YieldOp>(forOp.getRegion().front().getTerminator());
    Value yieldedValue = yieldOp.getOperand(resultIndex);

    // Find size for the yielded resource value.
    auto yieldedSizeValue = IREE::Util::SizeAwareTypeInterface::findSizeValue(
        yieldedValue, &forOp.getRegion().front(), Block::iterator(yieldOp));
    auto yieldedSize = tryEvaluateConstantSize(yieldedSizeValue);

    // If the yielded size matches the initial size, it's invariant.
    if (yieldedSize && *yieldedSize == *initSize) {
      return *initSize;
    }

    return std::nullopt;
  }

  // Could not evaluate to a constant.
  return std::nullopt;
}

// Returns a constant pattern for a value derived entirely from a splatted
// value. Returns nullptr if the value is not derived from a splat or has a
// non-constant pattern.
static TypedAttr findConstantSplatPattern(Value storedValue) {
  Operation *op = storedValue.getDefiningOp();
  do {
    if (auto transferOp = dyn_cast<IREE::Stream::AsyncTransferOp>(op)) {
      op = transferOp.getSource().getDefiningOp();
    } else if (auto splatOp = dyn_cast<IREE::Stream::AsyncSplatOp>(op)) {
      TypedAttr pattern;
      if (matchPattern(splatOp.getValue(), m_Constant(&pattern))) {
        return pattern;
      }
      return {};
    } else {
      return {};
    }
  } while (op);
  return {};
}

// Returns the last value produced that is non-metadata (according to us).
// This lets us skip meaningless ops like transfers and clones that change
// lifetime when cloning into the target program. Those ops, though valid, make
// the IR a lot more confusing to follow and prevent some early folding
// opportunities.
static Value findProducedValue(Value value) {
  while (Operation *defOp = value.getDefiningOp()) {
    if (auto transferOp = dyn_cast<IREE::Stream::AsyncTransferOp>(defOp)) {
      // We never care about transfers unless they are transferring to unknown.
      auto resultType =
          cast<IREE::Stream::ResourceType>(transferOp.getResult().getType());
      if (resultType.getLifetime() != IREE::Stream::Lifetime::Unknown) {
        value = transferOp.getSource();
        continue;
      }
    } else if (auto cloneOp = dyn_cast<IREE::Stream::AsyncCloneOp>(defOp)) {
      // Skip past clones to find the actual producing operation.
      // Clones are just type/lifetime conversions, not data producers.
      value = cloneOp.getSource();
      continue;
    }
    break;
  }
  return value;
}

// Returns true if the expression should be hoisted based on policy.
static bool shouldHoistExpression(const EncodingExpr &expr,
                                  const EncodingPolicy &policy) {
  bool hasParams = expr.hasParameterInputs();
  bool hasConstants = expr.hasConstantInputs();

  // Check if this expression type should be hoisted per policy.
  if (hasParams && !policy.hoistParameterExpressions) {
    LLVM_DEBUG(DBGS() << "skipping parameter expression per policy\n");
    return false;
  }
  if (!hasParams && hasConstants && !policy.hoistConstantExpressions) {
    LLVM_DEBUG(DBGS() << "skipping constant expression per policy\n");
    return false;
  }
  if (!hasParams && !hasConstants) {
    // No inputs at all - probably an error case or pure splat.
    LLVM_DEBUG(DBGS() << "skipping expression with no inputs\n");
    return false;
  }

  // Check size growth threshold.
  int64_t inputSize = expr.estimateInputSize();
  int64_t outputSize = expr.estimateOutputSize();
  if (inputSize > 0) {
    float growthFactor = static_cast<float>(outputSize) / inputSize;
    if (growthFactor > policy.maxEncodingGrowthFactor) {
      LLVM_DEBUG(DBGS() << "rejecting expression due to size growth: "
                        << growthFactor << "x (threshold: "
                        << policy.maxEncodingGrowthFactor << "x)\n");
      return false;
    }
  }

  return true;
}

// Analyzes |moduleOp| to find all expressions producing global constants that
// we can turn into parameters, if any.
static EncodingExprSet gatherEncodingExprSet(mlir::ModuleOp moduleOp,
                                             EncodingPolicy policy) {
  auto constantStoreOps = findAllConstantStoreOps(moduleOp);

  EncodingExprSet exprSet;

  std::unique_ptr<AsmState> asmState;
  LLVM_DEBUG(asmState = std::make_unique<AsmState>(
                 moduleOp, OpPrintingFlags().elideLargeElementsAttrs()));

  for (auto storeOp : constantStoreOps) {
    LLVM_DEBUG({
      DBGS() << "evaluating store slice for inclusion: ";
      storeOp->print(llvm::dbgs(), *asmState);
      llvm::dbgs() << "\n";
    });
    Value storedValue = storeOp.getStoredGlobalValue();

    BackwardSliceOptions sliceOptions;
    sliceOptions.inclusive = true;
    bool foundAnyNonConstExprOps = false;
    sliceOptions.filter = [&](Operation *op) {
      if (isConstExprOp(op)) {
        return true;
      }
      foundAnyNonConstExprOps = true;
      return false;
    };
    // Collect all values that need to be included in the slice:
    // - The stored value itself
    // - Values used inside nested regions that are defined outside
    //
    // We compute backward slices for all of them into the same SetVector,
    // which gives us proper topological ordering with deduplication.
    SetVector<Value> rootValues;
    rootValues.insert(storedValue);

    // Do a first pass to find region-containing operations.
    SetVector<Operation *> tempSlice;
    if (failed(mlir::getBackwardSlice(storedValue, &tempSlice, sliceOptions)) ||
        foundAnyNonConstExprOps) {
      LLVM_DEBUG(DBGS() << "failed to calculate backward slice for op or found "
                           "non-const-expr ops, skipping\n");
      continue;
    }

    // Find external dependencies from nested regions using MLIR's standard API.
    // getUsedValuesDefinedAbove returns all values used inside a region but
    // defined outside of it - exactly what we need for region captures.
    for (auto *op : tempSlice) {
      for (Region &region : op->getRegions()) {
        SetVector<Value> capturedValues;
        mlir::getUsedValuesDefinedAbove(region, capturedValues);
        LLVM_DEBUG({
          if (!capturedValues.empty()) {
            DBGS() << "found " << capturedValues.size()
                   << " captured values in region of ";
            op->print(llvm::dbgs());
            llvm::dbgs() << ":\n";
            for (Value captured : capturedValues) {
              llvm::dbgs() << "  ";
              captured.print(llvm::dbgs());
              llvm::dbgs() << "\n";
            }
          }
        });
        for (Value captured : capturedValues) {
          rootValues.insert(captured);
        }
      }
    }

    // Now compute backward slices for all root values.
    // When we have multiple roots (due to captured values), calling
    // getBackwardSlice iteratively can break topological order because new
    // operations get appended. We need to sort after merging.
    bool needsSort = rootValues.size() > 1;
    SetVector<Operation *> slice;
    for (Value rootValue : rootValues) {
      if (failed(mlir::getBackwardSlice(rootValue, &slice, sliceOptions)) ||
          foundAnyNonConstExprOps) {
        LLVM_DEBUG(DBGS() << "failed to calculate backward slice for op or "
                             "found non-const-expr ops, skipping\n");
        break;
      }
    }

    if (foundAnyNonConstExprOps) {
      continue;
    }

    // Sort only when we merged multiple slices (i.e., had captured values).
    // This is a small set (one expression), not the whole program.
    // Use mlir::topologicalSort which correctly handles operations across
    // different blocks and regions, unlike isBeforeInBlock which only works
    // for operations within the same block.
    if (needsSort) {
      slice = mlir::topologicalSort(slice);
    }

    LLVM_DEBUG({
      DBGS() << "slice:\n";
      llvm::interleave(
          slice, llvm::dbgs(),
          [&](Operation *op) {
            llvm::dbgs() << "  ";
            op->print(llvm::dbgs(), *asmState);
          },
          "\n");
      llvm::dbgs() << "\n";
    });

    // Overlay mode optimization: When a slice is just a parameter load with no
    // transformation (detected by isPassThroughStore below), we skip including
    // it as an output in overlay mode since the original parameter is
    // unchanged. This is controlled by policy.includeUnmodified:
    // - Consolidate mode (includeUnmodified=true): includes all parameters
    // - Overlay mode (includeUnmodified=false): skips pass-through parameters
    //
    // Future enhancement: Could add overlap detection to merge expressions that
    // write to overlapping parameter regions, possibly requiring a two-pass
    // approach. For now, non-overlapping expressions work correctly.

    EncodingExpr expr;
    expr.affinityAttr =
        lookupConsumerAffinityAttr(storeOp.getStoredGlobalValue());

    for (auto *op : slice) {
      if (auto constantOp = dyn_cast<IREE::Stream::AsyncConstantOp>(op)) {
        EncodingExpr::Input input;
        input.constantOp = constantOp;
        expr.inputs.push_back(input);
      }
    }

    if (!isPassThroughStore(storeOp.getStoredGlobalValue()) ||
        policy.includeUnmodified) {
      // Check if the produced value prefers cloning (like pure splats).
      // These should be included in the slice for cloning but not serialized
      // as outputs.
      Value producedValue = findProducedValue(storeOp.getStoredGlobalValue());
      auto *producingOp = producedValue.getDefiningOp();
      if (producingOp) {
        if (auto streamableOp =
                dyn_cast<IREE::Stream::StreamableOpInterface>(producingOp)) {
          if (streamableOp.preferCloneToConsumers()) {
            LLVM_DEBUG(DBGS()
                       << "skipping output for op that prefers cloning\n");
            continue;
          }
        }
      }

      Value storedValue = storeOp.getStoredGlobalValue();
      Value sizeValue = IREE::Util::SizeAwareTypeInterface::findSizeValue(
          storedValue, storeOp->getBlock(), Block::iterator(storeOp));

      // If findSizeValue returns null, it might be because the value comes from
      // a control flow operation (like scf.for or scf.if) that doesn't
      // implement SizeAwareOpInterface. Try analyzing the control flow
      // directly.
      std::optional<uint64_t> sizeOpt;
      if (sizeValue) {
        sizeOpt = tryEvaluateConstantSize(sizeValue);
      } else if (auto *defOp = storedValue.getDefiningOp()) {
        // Try analyzing control flow operations directly.
        if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
          // Find which result this is.
          unsigned resultIdx = 0;
          for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
            if (forOp.getResult(i) == storedValue) {
              resultIdx = i;
              break;
            }
          }
          // Get size from init arg.
          Value initArg = forOp.getInitArgs()[resultIdx];
          Value initSizeValue =
              IREE::Util::SizeAwareTypeInterface::findSizeValue(
                  initArg, forOp->getBlock(), Block::iterator(forOp));
          sizeOpt = tryEvaluateConstantSize(initSizeValue);
        } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
          // Find which result this is.
          unsigned resultIdx = 0;
          for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
            if (ifOp.getResult(i) == storedValue) {
              resultIdx = i;
              break;
            }
          }

          // Get sizes from both branches.
          auto thenYield =
              cast<scf::YieldOp>(ifOp.getThenRegion().front().getTerminator());
          auto elseYield =
              cast<scf::YieldOp>(ifOp.getElseRegion().front().getTerminator());
          Value thenValue = thenYield.getOperand(resultIdx);
          Value elseValue = elseYield.getOperand(resultIdx);
          auto thenSizeValue =
              IREE::Util::SizeAwareTypeInterface::findSizeValue(
                  thenValue, &ifOp.getThenRegion().front(),
                  Block::iterator(thenYield));
          auto elseSizeValue =
              IREE::Util::SizeAwareTypeInterface::findSizeValue(
                  elseValue, &ifOp.getElseRegion().front(),
                  Block::iterator(elseYield));
          auto thenSize = tryEvaluateConstantSize(thenSizeValue);
          auto elseSize = tryEvaluateConstantSize(elseSizeValue);

          // Both branches must have the same constant size.
          if (thenSize && elseSize && *thenSize == *elseSize) {
            sizeOpt = *thenSize;
          }
        }
      }

      if (!sizeOpt) {
        LLVM_DEBUG(DBGS() << "failed to find stored resource size, skipping\n");
        continue;
      }
      EncodingExpr::Output output;
      output.size = *sizeOpt;
      output.splatPattern =
          findConstantSplatPattern(storeOp.getStoredGlobalValue());
      output.storeOp = storeOp;
      output.producedValue = producedValue;
      expr.outputs.push_back(output);
    }

    if (expr.outputs.empty()) {
      LLVM_DEBUG(DBGS() << "no outputs produced by policy, skipping\n");
      continue;
    }

    expr.ops = std::move(slice);
    exprSet.exprs.push_back(std::move(expr));
  }

  return exprSet;
}

//===----------------------------------------------------------------------===//
// ParameterIndex and builders
//===----------------------------------------------------------------------===//

// An entry in the parameter index describing a single output parameter.
// Entries can be either SPLAT (constant pattern fill) or DATA (computed bytes).
// A single EncodingExpr may produce multiple entries if it has multiple
// outputs.
struct ParameterEntry {
  // Location of the parameter based on the original consumer op.
  std::optional<Location> loc;
  enum class Type {
    SPLAT = 0,
    DATA = 1,
  };
  // Type of the entry (indicates which value field is valid).
  Type type;
  // Key of the entry within the parameter scope.
  StringAttr key;
  // Optional metadata embedded with the entry.
  SmallVector<uint8_t> metadata;
  // Total byte length of the parameter in memory.
  int64_t length;
  // Type-specific value.
  union {
    struct SplatEntry {
      int64_t pattern;
      int64_t patternLength;
    } splat;
    struct DataEntry {
      int64_t minimumAlignment;
    } data;
  } value;

  static ParameterEntry createSplat(Location loc, StringAttr key,
                                    int64_t length, int64_t pattern,
                                    int64_t patternLength) {
    ParameterEntry entry{loc};
    entry.type = Type::SPLAT;
    entry.key = key;
    entry.length = length;
    entry.value.splat.pattern = pattern;
    entry.value.splat.patternLength = patternLength;
    return entry;
  }

  static ParameterEntry createData(Location loc, StringAttr key, int64_t length,
                                   int64_t minimumAlignment) {
    ParameterEntry entry{loc};
    entry.type = Type::DATA;
    entry.key = key;
    entry.length = length;
    entry.value.data.minimumAlignment = minimumAlignment;
    return entry;
  }

  Location getLoc() const {
    return loc.has_value() ? loc.value() : UnknownLoc::get(key.getContext());
  }
};

// An IRPA parameter index.
struct ParameterIndex {
  // Fused location derived from all included parameter locations.
  Location loc;
  // Scope name the index is referenced with, if any.
  StringAttr scope;
  // All parameter entries in the index.
  SmallVector<ParameterEntry> entries;

  void dump(llvm::raw_ostream &os) const {
    os << "ParameterIndex[" << scope << "]:\n";
    llvm::interleave(
        entries, os,
        [&](const ParameterEntry &entry) {
          os << "  '" << entry.key << "' " << entry.length << " bytes ";
          if (!entry.metadata.empty()) {
            os << "(metadata: " << entry.metadata.size() << "B) ";
          }
          switch (entry.type) {
          case ParameterEntry::Type::SPLAT:
            os << "splat: "
               << APInt(entry.value.splat.patternLength * 8,
                        entry.value.splat.pattern);
            break;
          case ParameterEntry::Type::DATA:
            os << "data: min alignment " << entry.value.data.minimumAlignment
               << "B";
            break;
          }
        },
        "\n");
    os << "\n";
  }
};

struct ParameterBuilder {
  MLIRContext *context;
  StringAttr scope;
  StringAttr key;

  ParameterBuilder() = delete;
  explicit ParameterBuilder(MLIRContext *context, StringAttr scope,
                            StringAttr key)
      : context(context), scope(scope), key(key) {}
  virtual ~ParameterBuilder() = default;
  virtual ParameterEntry finalize() = 0;
};

struct SplatParameterBuilder : public ParameterBuilder {
  Location loc;
  int64_t length = 0;
  Attribute pattern;

  SplatParameterBuilder(StringAttr scope, StringAttr key, Location loc,
                        int64_t length, Attribute pattern)
      : ParameterBuilder(loc.getContext(), scope, key), loc(loc),
        length(length), pattern(pattern) {}

  ParameterEntry finalize() override {
    APInt intValue;
    APFloat floatValue(0.0f);
    if (matchPattern(pattern, m_ConstantFloat(&floatValue))) {
      intValue = floatValue.bitcastToAPInt();
    } else if (matchPattern(pattern, m_ConstantInt(&intValue))) {
    } else {
      assert(false && "ints/floats only; should have been verified");
    }
    return ParameterEntry::createSplat(
        loc, key, length, intValue.getZExtValue(), intValue.getBitWidth() / 8);
  }
};

struct DataParameterBuilder : public ParameterBuilder {
  IREE::Stream::AffinityAttr affinityAttr;
  int64_t maxSize = 0;
  int64_t offsetAlignment = 0;
  int64_t rangeAlignment = 0;
  int64_t currentOffset = 0;
  SmallVector<Location> locs;

  DataParameterBuilder(StringAttr scope, StringAttr key,
                       IREE::Stream::AffinityAttr affinityAttr,
                       IREE::Stream::ResourceConfigAttr resourceConfigAttr)
      : ParameterBuilder(resourceConfigAttr.getContext(), scope, key),
        affinityAttr(affinityAttr),
        maxSize(resourceConfigAttr.getMaxAllocationSize()),
        offsetAlignment(resourceConfigAttr.getMinBufferOffsetAlignment()),
        rangeAlignment(resourceConfigAttr.getMinBufferRangeAlignment()) {}

  // Reserves |length| bytes of storage in the parameter and returns the aligned
  // offset within the parameter if there is sufficient capacity remaining.
  std::optional<int64_t> tryReserve(Location loc, int64_t length) {
    int64_t alignedOffset = IREE::Util::align(currentOffset, offsetAlignment);
    int64_t alignedLength = IREE::Util::align(length, rangeAlignment);
    int64_t newOffset = std::max(currentOffset, alignedOffset + alignedLength);
    if (newOffset > maxSize) {
      // Capacity exceeded.
      return std::nullopt;
    }
    currentOffset = newOffset;
    return alignedOffset;
  }

  ParameterEntry finalize() override {
    return ParameterEntry::createData(
        FusedLoc::get(context, locs), key,
        IREE::Util::align(currentOffset, rangeAlignment), offsetAlignment);
  }
};

// A subrange of an output parameter produced by an encoding expression.
// Note that a single expression may produce multiple output subranges.
struct ParameterSubrange {
  // Parameter index scope.
  StringAttr scope;
  // Parameter key the subrange is referencing.
  StringAttr key;
  // Offset within the parameter where the produced value will be placed.
  // Aligned to the requirements of the parameter.
  int64_t offset = 0;
  // Length of subrange the produced value occupies. Note that if padding is
  // present this may not extend to all of the parameter storage.
  int64_t length = 0;

  ParameterSubrange(StringAttr scope, StringAttr key, int64_t offset,
                    int64_t length)
      : scope(scope), key(key), offset(offset), length(length) {}

  // Creates a named parameter attribute for this subrange with the given total
  // length of the storage parameter.
  IREE::Stream::NamedParameterAttr
  createNamedParameterAttr(int64_t totalLength) const {
    Type i8Type = IntegerType::get(scope.getContext(), 8);
    auto parameterType = RankedTensorType::get({totalLength}, i8Type);
    return IREE::Stream::NamedParameterAttr::get(
        scope.getContext(), parameterType, scope, key, DictionaryAttr{});
  }
};

// Map of expression outputs to a reserved parameter subrange.
using OutputParameterSubrangeMap =
    llvm::MapVector<const EncodingExpr::Output *, ParameterSubrange>;

// Incremental ParameterIndex builder with support for parameter combining.
class ParameterIndexBuilder {
public:
  ParameterIndexBuilder(StringAttr scope, const EncodingPolicy &encodingPolicy)
      : scope(scope), encodingPolicy(encodingPolicy) {}

  FailureOr<OutputParameterSubrangeMap> insertExpr(const EncodingExpr *expr) {
    OutputParameterSubrangeMap outputMap;
    for (const auto &output : expr->outputs) {
      FailureOr<std::optional<ParameterSubrange>> subrangeOr;
      if (output.isSupportedSplat() &&
          output.size > encodingPolicy.serializeSplatSizeThreshold) {
        subrangeOr = insertSplatOutput(expr, &output);
      } else {
        subrangeOr = insertDataOutput(expr, &output);
      }
      if (failed(subrangeOr)) {
        return failure();
      }
      if (subrangeOr->has_value()) {
        outputMap.insert(
            std::make_pair(&output, std::move(subrangeOr->value())));
      }
    }
    return outputMap;
  }

  ParameterIndex finalize() {
    SmallVector<Location> parameterLocs;
    SmallVector<ParameterEntry> parameterEntries;
    for (auto &parameter : parameters) {
      ParameterEntry parameterEntry = parameter->finalize();
      parameterLocs.push_back(parameterEntry.getLoc());
      parameterEntries.push_back(std::move(parameterEntry));
    }
    ParameterIndex index{FusedLoc::get(scope.getContext(), parameterLocs)};
    index.scope = scope;
    index.entries = std::move(parameterEntries);
    return index;
  }

private:
  StringAttr makeParameterName() {
    return StringAttr::get(scope.getContext(),
                           Twine("parameter") + std::to_string(nextId++));
  }

  FailureOr<std::optional<ParameterSubrange>>
  insertSplatOutput(const EncodingExpr *expr,
                    const EncodingExpr::Output *output) {
    auto splatBuilder = std::make_unique<SplatParameterBuilder>(
        scope, makeParameterName(), expr->getLoc(), output->size,
        output->splatPattern);
    auto subrange = ParameterSubrange(splatBuilder->scope, splatBuilder->key, 0,
                                      output->size);
    parameters.push_back(std::move(splatBuilder));
    return {subrange};
  }

  // Inserts a data output into the parameter index, packing into existing
  // parameters when possible.
  //
  // Uses first-fit allocation: iterates through existing parameters in order
  // and places the output in the first one with matching affinity and available
  // space. This is simple and fast for compilation, though not optimal for
  // minimizing fragmentation. A best-fit or sorted-by-size approach could
  // improve memory efficiency if parameter packing becomes a bottleneck.
  FailureOr<std::optional<ParameterSubrange>>
  insertDataOutput(const EncodingExpr *expr,
                   const EncodingExpr::Output *output) {
    if (encodingPolicy.packParameters) {
      for (auto *existingBuilder : dataParameters) {
        if (existingBuilder->affinityAttr == expr->affinityAttr) {
          std::optional<int64_t> offset =
              existingBuilder->tryReserve(expr->getLoc(), output->size);
          if (offset.has_value()) {
            auto subrange =
                ParameterSubrange(existingBuilder->scope, existingBuilder->key,
                                  offset.value(), output->size);
            return {subrange};
          }
        }
      }
    }

    auto newBuilder = std::make_unique<DataParameterBuilder>(
        scope, makeParameterName(), expr->affinityAttr,
        expr->getResourceConfigAttr());
    std::optional<int64_t> offset =
        newBuilder->tryReserve(expr->getLoc(), output->size);
    if (offset.has_value()) {
      auto subrange = ParameterSubrange(newBuilder->scope, newBuilder->key,
                                        offset.value(), output->size);
      dataParameters.push_back(newBuilder.get());
      parameters.push_back(std::move(newBuilder));
      return {subrange};
    }

    LLVM_DEBUG(llvm::dbgs()
               << "  ! failed to reserve " << output->size
               << " bytes for output at " << output->getLoc() << "\n");
    return mlir::emitError(output->getLoc(),
                           "failed to reserve parameter space for output\n");
  }

  StringAttr scope;
  const EncodingPolicy &encodingPolicy;
  SmallVector<std::unique_ptr<ParameterBuilder>> parameters;
  SmallVector<DataParameterBuilder *> dataParameters;
  unsigned nextId = 0;
};

//===----------------------------------------------------------------------===//
// Encoder work scheduling
//===----------------------------------------------------------------------===//

// A target configuration for a set of specialized encodings.
// Contains the parameter indices (what parameters will be produced), the
// execution schedule (steps), and a lookup map from (scope, key) to entries.
// Targets may specialize for multiple devices simultaneously if the
// configuration is for heterogeneous execution and may produce multiple
// parameter indices. Currently only a single "all" target is supported.
struct TargetPlan {
  // Name of the target for the user to specify in tools.
  std::string name;

  // Affinity of the device performing the encoding in the encoder module.
  // When cross-targeting encoders this will differ from the devices in the
  // original program. For consistency it always has a new name.
  IREE::Stream::AffinityAttr affinityAttr;

  // Parameter indices produced by the target.
  std::vector<ParameterIndex> parameterIndices;

  // A map of (scope, key) to the parameter in the specified index.
  DenseMap<std::pair<StringAttr, StringAttr>, ParameterEntry> parameterEntries;

  // A discrete step in the encoding process.
  struct Step {
    std::string description;
    int64_t globalByteOffset = 0;
    int64_t globalByteLength = 0;
    const EncodingExpr *expr = nullptr;
    OutputParameterSubrangeMap outputMap;

    Location getLoc() const { return expr->getLoc(); }
  };

  // An unordered sequence of encoding steps.
  // Steps _generally_ start in order but may end in any order and can be
  // considered more as "chunks of work" than some point on a timeline.
  // Each step may encode more than one parameter.
  SmallVector<Step> steps;

  // Cumulative size of all writes to all parameters in all scopes.
  int64_t globalByteSize = 0;

  // Appends an encoding expression and its output mapping to the schedule.
  void appendExpr(const EncodingExpr *expr,
                  OutputParameterSubrangeMap outputMap) {
    Step step;

    // Today we just name the steps in sequence, but could use the parameter
    // names in the output map.
    step.description = "step" + std::to_string(steps.size());

    // Since order is largely undefined and each step may produce multiple
    // parameters we track a cumulative write offset in a virtual global
    // parameter file and use that. Tools can present % completed or use the
    // virtual subranges to indicate fine-grained progress.
    step.globalByteOffset = globalByteSize;
    step.globalByteLength = std::accumulate(
        expr->outputs.begin(), expr->outputs.end(), int64_t{0},
        [](int64_t sum, const EncodingExpr::Output &output) -> int64_t {
          return sum + output.size;
        });
    LLVM_DEBUG(DBGS() << "defining step `" << step.description << "` (at "
                      << step.globalByteOffset << " for "
                      << step.globalByteLength << ")\n");
    globalByteSize += step.globalByteLength;

    step.expr = expr;
    step.outputMap = std::move(outputMap);
    steps.push_back(std::move(step));
  }

  // Returns the named parameter reference attribute for the given subrange.
  IREE::Stream::NamedParameterAttr
  getNamedParameterAttr(const ParameterSubrange &subrange) const {
    auto parameterEntryIt =
        parameterEntries.find(std::make_pair(subrange.scope, subrange.key));
    assert(parameterEntryIt != parameterEntries.end() &&
           "map must contain all entries");
    const ParameterEntry &parameterEntry = parameterEntryIt->second;
    return subrange.createNamedParameterAttr(parameterEntry.length);
  }
};

//===----------------------------------------------------------------------===//
// Parameter encoder construction
//===----------------------------------------------------------------------===//

// Adds a function to the new encoder module that tries to automatically detect
// the target configuration given the list of HAL devices. The intent is that it
// performs the same device detection logic the main module performs at runtime
// but with a provided list instead of what the HAL module provides: the only
// device(s) we have at the global level are those of the host performing the
// encoding.
//
// Signature, returning a string constant target name:
//   util.func public @__encode_parameter_detect_target(
//       %devices: !util.list<!hal.device>) -> !util.buffer
static void addAutoTargetDetectFunc(Location loc,
                                    ArrayRef<TargetPlan> targetPlans,
                                    OpBuilder &encoderBuilder) {
  std::string funcName = "__encode_parameter_detect_target";
  LLVM_DEBUG(DBGS() << "emitting auto target detection function: " << funcName
                    << "...\n");

  auto bufferType = encoderBuilder.getType<IREE::Util::BufferType>();
  auto deviceType = encoderBuilder.getType<IREE::HAL::DeviceType>();
  auto deviceListType =
      encoderBuilder.getType<IREE::Util::ListType>(deviceType);
  auto funcOp = IREE::Util::FuncOp::create(
      encoderBuilder, loc, funcName,
      encoderBuilder.getFunctionType({deviceListType}, {bufferType}));
  funcOp.setVisibility(SymbolTable::Visibility::Public);
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());
  funcOp->setAttr(
      "iree.reflection",
      funcBuilder.getDictionaryAttr({
          NamedAttribute("iree.encode.function",
                         funcBuilder.getStringAttr("detect_target")),
      }));

  // Always unconditionally choose the first target today.
  assert(!targetPlans.empty());
  Value targetName = IREE::Util::BufferConstantOp::create(
      funcBuilder, loc, targetPlans.front().name);
  IREE::Util::ReturnOp::create(funcBuilder, loc, targetName);
}

// Builds a struct of `[scope name, [entries]]`.
// Supported entry types:
//
// SPLAT (iree_io_parameter_archive_builder_add_splat_entry):
//   [0]: i64 type=0
//   [1]: !util.buffer key (not be NUL terminated)
//   [2]: !util.buffer metadata (optional)
//   [3]: i64 data length (total size of the parameter)
//   [4]: i64 pattern (only up to pattern_bytes_length bytes used)
//   [5]: i64 pattern_byte_length
//
// DATA (iree_io_parameter_archive_builder_add_data_entry):
//   [0]: i64 type=1
//   [1]: !util.buffer key (not be NUL terminated)
//   [2]: !util.buffer metadata (optional)
//   [3]: i64 data length (total size of the parameter)
//   [4]: i64 minimum alignment (or 0 if don't care)
static Value buildParameterIndexStruct(const ParameterIndex &parameterIndex,
                                       IntegerSet<int64_t> &i64Set,
                                       OpBuilder &builder) {
  LLVM_DEBUG({
    DBGS() << "emitting index with scope: `" << parameterIndex.scope << "` ("
           << parameterIndex.entries.size() << " entries)\n";
    parameterIndex.dump(llvm::dbgs());
  });

  auto loc = parameterIndex.loc;
  auto listType = builder.getType<IREE::Util::ListType>();

  Value scopeName =
      IREE::Util::BufferConstantOp::create(builder, loc, parameterIndex.scope);

  SmallVector<Value> entryValues;
  for (auto &entry : parameterIndex.entries) {
    Location entryLoc = entry.getLoc();
    Value typeValue = i64Set.get(static_cast<int64_t>(entry.type));
    Value keyValue =
        IREE::Util::BufferConstantOp::create(builder, entryLoc, entry.key);
    Value metadataValue = IREE::Util::BufferConstantOp::createOrNull(
        builder, entryLoc, entry.metadata);
    SmallVector<Value> structFields = {
        typeValue,
        keyValue,
        metadataValue,
        i64Set.get(entry.length),
    };
    switch (entry.type) {
    case ParameterEntry::Type::SPLAT:
      structFields.push_back(i64Set.get(entry.value.splat.pattern));
      structFields.push_back(i64Set.get(entry.value.splat.patternLength));
      break;
    case ParameterEntry::Type::DATA:
      structFields.push_back(i64Set.get(entry.value.data.minimumAlignment));
      break;
    }
    Value entryValue = IREE::Util::ListConstructOp::create(
        builder, entryLoc, listType, structFields);
    entryValues.push_back(entryValue);
  }

  Value entryList =
      IREE::Util::ListConstructOp::create(builder, loc, listType, entryValues);
  Value indexStruct = IREE::Util::ListConstructOp::create(
      builder, loc, listType, {scopeName, entryList});
  return indexStruct;
};

// Adds a function to the new encoder module that returns the parameter indices
// produced for a given target. A single target may result in more than one
// parameter file in cases where we want to shard parameters.
//
// Signature:
//   util.func public @__encode_parameter_indices_TARGET() -> !util.list<?>
static void addTargetIndexBuilderFunc(Location loc,
                                      const TargetPlan &targetPlan,
                                      OpBuilder &encoderBuilder) {
  auto listType = encoderBuilder.getType<IREE::Util::ListType>();
  std::string funcName = "__encode_parameter_indices_" + targetPlan.name;
  LLVM_DEBUG(DBGS() << "emitting index builder function: " << funcName
                    << "...\n");
  auto funcOp = IREE::Util::FuncOp::create(
      encoderBuilder, loc, funcName,
      encoderBuilder.getFunctionType({}, {listType}));
  funcOp.setVisibility(SymbolTable::Visibility::Public);
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());

  // Reflection information lets the tool list available targets and required
  // scopes without having to call each function.
  // VM bytecode only supports string/integer reflection attributes, so we
  // encode scopes as a comma-separated string.
  // Note: Empty string values crash the flatbuffer serializer, so we only
  // include the scopes attribute if there are non-empty scopes.
  std::string scopesStr;
  for (const auto &parameterIndex : targetPlan.parameterIndices) {
    StringRef scope = parameterIndex.scope.getValue();
    if (scope.empty()) {
      continue;
    }
    if (!scopesStr.empty()) {
      scopesStr += ",";
    }
    scopesStr += scope;
  }
  SmallVector<NamedAttribute> reflectionAttrs;
  reflectionAttrs.push_back(NamedAttribute(
      "iree.encode.function", funcBuilder.getStringAttr("indices")));
  reflectionAttrs.push_back(NamedAttribute(
      "iree.encode.target", funcBuilder.getStringAttr(targetPlan.name)));
  if (!scopesStr.empty()) {
    reflectionAttrs.push_back(NamedAttribute(
        "iree.encode.scopes", funcBuilder.getStringAttr(scopesStr)));
  }
  funcOp->setAttr("iree.reflection",
                  funcBuilder.getDictionaryAttr(reflectionAttrs));

  IntegerSet<int64_t> i64Set(loc, funcBuilder);
  SmallVector<Value> indicesStructs;
  for (const auto &parameterIndex : targetPlan.parameterIndices) {
    indicesStructs.push_back(
        buildParameterIndexStruct(parameterIndex, i64Set, funcBuilder));
  }

  Value indicesList = IREE::Util::ListConstructOp::create(
      funcBuilder, loc, listType, indicesStructs);
  IREE::Util::ReturnOp::create(funcBuilder, loc, {indicesList});
}

// Adds a function to the new encoder module that produces a list of steps
// involved in encoding the parameters for a specific target. Steps do not
// correspond 1:1 with parameters in either the input or output module and may
// complete in any order so we return a list of structs and fences that can be
// used to observe the state and report on progress. If progress capture is
// desired the list needs to be passed back into the encoder function so that it
// can instrument the encoding process with the fences.
//
// A global byte range is attached to each step for presentation purposes only:
// multiple parameter indices may be constructed and an individual step may
// produce values for each. Tools may only use the global byte range to denote
// cumulative bytes written by each step.
//
// Each step entry consists of:
//   [0]: i64 reserved 0
//   [1]: !hal.fence indicating encoding has begun
//   [2]: !hal.fence indicating encoding has ended
//   [3]: !util.buffer descriptive comment (not be NUL terminated)
//   [4]: i64 synthetic global byte offset
//   [5]: i64 synthetic global byte length
//
// Signature:
//   util.func public @__encode_parameter_steps_TARGET() -> !util.list<?>
static void addTargetEncoderStepsFunc(Location loc,
                                      const TargetPlan &targetPlan,
                                      OpBuilder &encoderBuilder) {
  std::string funcName = "__encode_parameter_steps_" + targetPlan.name;
  LLVM_DEBUG(DBGS() << "emitting encoder steps function: " << funcName
                    << "...\n");
  auto listType = encoderBuilder.getType<IREE::Util::ListType>();
  auto funcOp = IREE::Util::FuncOp::create(
      encoderBuilder, loc, funcName,
      encoderBuilder.getFunctionType({}, {listType}));
  funcOp.setVisibility(SymbolTable::Visibility::Public);
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());
  funcOp->setAttr(
      "iree.reflection",
      funcBuilder.getDictionaryAttr({
          NamedAttribute("iree.encode.function",
                         funcBuilder.getStringAttr("steps")),
          NamedAttribute("iree.encode.target",
                         funcBuilder.getStringAttr(targetPlan.name)),
      }));

  Type deviceType = funcBuilder.getType<IREE::HAL::DeviceType>();
  Value deviceValue =
      IREE::Stream::ContextResolveOp::create(funcBuilder, loc, {deviceType},
                                             targetPlan.affinityAttr)
          .getResult(0);

  SmallVector<Value> stepStructs;
  IntegerSet<int64_t> i64Set(loc, funcBuilder);
  for (auto &step : targetPlan.steps) {
    Value beginFence = IREE::HAL::FenceCreateOp::create(
        funcBuilder, loc, deviceValue, IREE::HAL::FenceFlagBitfield::None);
    Value endFence = IREE::HAL::FenceCreateOp::create(
        funcBuilder, loc, deviceValue, IREE::HAL::FenceFlagBitfield::None);
    Value descriptionValue = IREE::Util::BufferConstantOp::create(
        funcBuilder, loc, step.description);
    stepStructs.push_back(IREE::Util::ListConstructOp::create(
        funcBuilder, loc, listType,
        {
            i64Set.get(0),
            beginFence,
            endFence,
            descriptionValue,
            i64Set.get(step.globalByteOffset),
            i64Set.get(step.globalByteLength),
        }));
  }

  Value stepsList = IREE::Util::ListConstructOp::create(funcBuilder, loc,
                                                        listType, stepStructs);
  IREE::Util::ReturnOp::create(funcBuilder, loc, stepsList);
}

using MarkObjectReference =
    std::function<LogicalResult(Operation *userOp, SymbolRefAttr symbolRef)>;

// Adds a function to the new encoder module that encodes parameters for a
// specific target. Encoding will wait for the provided `wait_fence` prior to
// starting any processing and signal the provided `signal_fence` when all
// processing has completed. The steps list is the result of the paired
// `__encode_parameter_steps_TARGET` function and the fences within will be
// signaled as encoding progresses.
//
// Signature (follows standard coarse-fences ABI with fences at end):
//   util.func public @__encode_parameters_TARGET(
//       %steps: !util.list<?>,
//       %wait_fence: !hal.fence,
//       %signal_fence: !hal.fence)
static LogicalResult
addTargetEncoderFunc(Location loc, const TargetPlan &targetPlan,
                     const MarkObjectReference &markObjectReference,
                     OpBuilder &encoderBuilder) {
  std::string funcName = "__encode_parameters_" + targetPlan.name;
  LLVM_DEBUG(DBGS() << "emitting encoder function: " << funcName << "...\n");
  auto fenceType = encoderBuilder.getType<IREE::HAL::FenceType>();
  auto listType = encoderBuilder.getType<IREE::Util::ListType>();
  auto funcOp = IREE::Util::FuncOp::create(
      encoderBuilder, loc, funcName,
      encoderBuilder.getFunctionType({listType, fenceType, fenceType}, {}));
  funcOp.setVisibility(SymbolTable::Visibility::Public);
  OpBuilder funcBuilder = OpBuilder::atBlockBegin(funcOp.addEntryBlock());
  funcOp->setAttr(
      "iree.reflection",
      funcBuilder.getDictionaryAttr({
          NamedAttribute("iree.abi.model",
                         funcBuilder.getStringAttr("coarse-fences")),
          NamedAttribute("iree.encode.function",
                         funcBuilder.getStringAttr("encode")),
          NamedAttribute("iree.encode.target",
                         funcBuilder.getStringAttr(targetPlan.name)),
      }));

  // TODO(benvanik): make steps optional, probably by just calling the steps
  // function internally when not provided so that we can keep all the encoding
  // code branch-free. For now we require it be provided.

  Value waitFence = funcOp.getArgument(1);
  Value signalFence = funcOp.getArgument(2);

  Type timepointType = funcBuilder.getType<IREE::Stream::TimepointType>();
  Value lastTimepoint = IREE::Stream::TimepointImportOp::create(
      funcBuilder, loc, timepointType, waitFence, targetPlan.affinityAttr);

  // Use explicit transient lifetime for all output slab allocations.
  // This storage is allocated at the start of each step and deallocated at the
  // end, making transient the correct lifetime.
  Type resourceType = funcBuilder.getType<IREE::Stream::ResourceType>(
      IREE::Stream::Lifetime::Transient);
  IndexSet indexSet(loc, funcBuilder);
  IntegerSet<int64_t> i64Set(loc, funcBuilder);
  for (const auto &step : targetPlan.steps) {
    Location stepLoc = step.getLoc();

    // Build a map of scope name to the outputs going to it and their parameter
    // references. Note that this mapping is target-specific (as each target may
    // have a different mix of parameters and parameter sizes due to differences
    // in encodings).
    struct OutputReservation {
      const EncodingExpr::Output *output = nullptr;
      const ParameterSubrange *parameterSubrange = nullptr;
      IREE::Stream::NamedParameterAttr parameterAttr;
      size_t slabOffsetOrdinal = 0;
    };
    llvm::MapVector<StringAttr, SmallVector<OutputReservation>> scopeOutputs;
    SmallVector<Value> outputSizes;
    for (const EncodingExpr::Output &output : step.expr->outputs) {
      auto it = step.outputMap.find(&output);
      if (it == step.outputMap.end()) {
        continue; // no serialization required
      }
      const ParameterSubrange &subrange = it->second;
      OutputReservation reservation;
      reservation.output = &output;
      reservation.parameterSubrange = &subrange;
      reservation.parameterAttr = targetPlan.getNamedParameterAttr(subrange);
      reservation.slabOffsetOrdinal = outputSizes.size();
      scopeOutputs[reservation.parameterAttr.getScope()].push_back(reservation);
      outputSizes.push_back(indexSet.get(subrange.length));
    }

    // Allocate transient storage for all the parameter outputs.
    // If we were overlapping we'd want to get this from a ringbuffer.
    // TODO(benvanik): stream.async.ringbuffer-style ops for safely doing bump
    // pointer allocation with timeline-awareness at this level.
    auto reservationPackOp = IREE::Stream::ResourcePackOp::create(
        funcBuilder, stepLoc, /*offset=*/nullptr, outputSizes,
        targetPlan.affinityAttr);
    Value outputSlabSize = reservationPackOp.getTotalLength();
    auto outputSlabAllocaOp = IREE::Stream::ResourceAllocaOp::create(
        funcBuilder, stepLoc, resourceType, timepointType, outputSlabSize,
        /*indeterminate_lifetime=*/nullptr, lastTimepoint,
        targetPlan.affinityAttr);
    Value outputSlab = outputSlabAllocaOp.getResult();

    // Note: Input parameters are NOT included in this slab allocation.
    // Inputs are loaded via stream.async.constant operations (cloned below)
    // which reference external parameter storage and don't require allocation.
    // Only outputs need slab allocation as transient working memory before
    // being scattered to their final parameter locations.
    //
    // Wait for the slab to be ready before we transition back into async IR.
    outputSlab = IREE::Stream::TimepointAwaitOp::create(
                     funcBuilder, stepLoc, {outputSlab}, {outputSlabSize},
                     outputSlabAllocaOp.getResultTimepoint())
                     .getResult(0);

    // Clone the expression IR and fix it up for use in the new module.
    // We have to remove any affinities referencing the devices in the source
    // program and ensure we also bring along any referenced objects
    // (executables, etc).
    //
    // The slice is already in topological order from getBackwardSlice, and
    // all captured values from nested regions have been included via
    // getUsedValuesDefinedAbove, so we can clone directly without sorting.
    //
    // AsyncConstantOp with parameter values are converted to
    // AsyncParameterLoadOp during cloning because the lowering path through
    // ResourceConstantsOp does not preserve await_timepoint.
    // AsyncParameterLoadOp lowers directly to CmdParameterLoadOp which does
    // preserve await_timepoint.
    IRMapping exprMapping;
    for (auto *sourceOp : step.expr->ops) {
      auto *clonedOp = funcBuilder.clone(*sourceOp, exprMapping);
      if (auto affinityOp =
              dyn_cast<IREE::Stream::AffinityOpInterface>(clonedOp)) {
        affinityOp.removeAffinityAttrs();
      }
      // Convert AsyncConstantOp with parameter values to AsyncParameterLoadOp.
      // This ensures await_timepoint is preserved through lowering, since
      // AsyncConstantOp goes through ResourceConstantsOp which drops await.
      if (auto constantOp = dyn_cast<IREE::Stream::AsyncConstantOp>(clonedOp)) {
        if (auto parameterAttr =
                dyn_cast<NamedParameterAttr>(constantOp.getValue())) {
          // Extract parameter scope and key from the attribute.
          StringAttr scopeAttr = parameterAttr.getScope();
          StringAttr keyAttr = parameterAttr.getKey();
          // Create zero offset for full parameter load.
          Value zeroOffset = i64Set.get(0);
          Value resultSize = constantOp.getResultSize();
          // Create AsyncParameterLoadOp with the wait fence as await.
          auto paramLoadOp = IREE::Stream::AsyncParameterLoadOp::create(
              funcBuilder, constantOp.getLoc(),
              constantOp.getResult().getType(),
              funcBuilder.getType<IREE::Stream::TimepointType>(),
              /*await_timepoint=*/lastTimepoint, scopeAttr, keyAttr, zeroOffset,
              resultSize, targetPlan.affinityAttr);
          // Await the result timepoint to get a resolved resource that can be
          // used by streamable ops without explicit synchronization.
          auto awaitOp = IREE::Stream::TimepointAwaitOp::create(
              funcBuilder, constantOp.getLoc(), paramLoadOp.getResult(),
              resultSize, paramLoadOp.getResultTimepoint());
          // Update mapping to use the awaited result.
          exprMapping.map(sourceOp->getResult(0), awaitOp.getResults().front());
          // Erase the cloned AsyncConstantOp.
          constantOp.erase();
          clonedOp = awaitOp;
        } else {
          // Non-parameter constant: just set await_timepoint.
          constantOp.getAwaitTimepointMutable().assign(lastTimepoint);
        }
      }
      auto symbolUses = SymbolTable::getSymbolUses(clonedOp);
      if (symbolUses.has_value()) {
        for (auto &use : symbolUses.value()) {
          if (failed(markObjectReference(clonedOp, use.getSymbolRef()))) {
            return failure();
          }
        }
      }
    }

    // Scatter the outputs into the parameter(s) for each scope.
    for (const auto &[scope, outputReservations] : scopeOutputs) {
      for (auto &reservation : outputReservations) {
        Location outputLoc = reservation.output->getLoc();
        Value outputValue =
            exprMapping.lookup(reservation.output->producedValue);
        Value packedOffset =
            reservationPackOp.getPackedOffsets()[reservation.slabOffsetOrdinal];
        Value packedEnd =
            indexSet.add(packedOffset, reservation.parameterSubrange->length);
        Value outputSize = indexSet.get(reservation.parameterSubrange->length);
        auto updateOp = IREE::Stream::AsyncUpdateOp::create(
            funcBuilder, outputLoc, outputSlab.getType(), outputSlab,
            outputSlabSize, packedOffset, packedEnd, outputValue, outputSize,
            targetPlan.affinityAttr);
        outputSlab = updateOp.getResult();
      }
    }
    auto outputBarrierOp = IREE::Stream::TimepointBarrierOp::create(
        funcBuilder, step.getLoc(), outputSlab, outputSlabSize,
        targetPlan.affinityAttr);
    outputSlab = outputBarrierOp.getResult();

    // Scatter parameters from the transient slab into each target scope.
    SmallVector<Value> scatterTimepoints;
    for (const auto &[scope, outputReservations] : scopeOutputs) {
      SmallVector<Location> outputLocs;
      SmallVector<Value> sourceOffsets;
      SmallVector<Value> sourceEnds;
      SmallVector<Value> sourceLengths;
      SmallVector<Attribute> targetKeys;
      SmallVector<Value> targetOffsets;
      for (auto &reservation : outputReservations) {
        outputLocs.push_back(reservation.output->getLoc());
        Value packedOffset =
            reservationPackOp.getPackedOffsets()[reservation.slabOffsetOrdinal];
        Value packedSize = indexSet.get(reservation.parameterSubrange->length);
        sourceOffsets.push_back(packedOffset);
        sourceLengths.push_back(packedSize);
        targetKeys.push_back(reservation.parameterAttr.getKey());
        targetOffsets.push_back(
            i64Set.get(reservation.parameterSubrange->offset));
      }
      // Compute source ends (offset + length) for async parameter scatter.
      for (auto [offset, length] :
           llvm::zip_equal(sourceOffsets, sourceLengths)) {
        auto end = funcBuilder.createOrFold<mlir::arith::AddIOp>(
            funcBuilder.getFusedLoc(outputLocs), offset, length);
        sourceEnds.push_back(end);
      }
      auto scatterOp = IREE::Stream::AsyncParameterScatterOp::create(
          funcBuilder, funcBuilder.getFusedLoc(outputLocs), outputSlab,
          outputSlabSize, sourceOffsets, sourceEnds, sourceLengths, scope,
          funcBuilder.getArrayAttr(targetKeys), targetOffsets,
          outputBarrierOp.getResultTimepoint(), targetPlan.affinityAttr);
      // AsyncParameterScatterOp returns (resource, timepoint) tuple.
      outputSlab = scatterOp.getResult();
      scatterTimepoints.push_back(scatterOp.getResultTimepoint());
    }
    Value scattersTimepoint = IREE::Stream::TimepointJoinOp::create(
        funcBuilder, stepLoc, scatterTimepoints);

    // Deallocate the output slab (now the scattered resource).
    Value deallocaTimepoint = IREE::Stream::ResourceDeallocaOp::create(
        funcBuilder, stepLoc, outputSlab, outputSlabSize,
        /*prefer_origin=*/false, scattersTimepoint, targetPlan.affinityAttr);

    lastTimepoint = deallocaTimepoint;
  }

  // Chain the final timepoint (which depends on all steps via the loop above)
  // with the external signal fence. This signals completion of all encoding
  // steps. We use a single chain at the end rather than chaining after each
  // step because: (1) the function has only one signal fence parameter, and
  // (2) callers wait on the fence to know when all encoding is complete, not
  // individual steps.
  IREE::Stream::TimepointChainExternalOp::create(funcBuilder, funcOp.getLoc(),
                                                 lastTimepoint, {signalFence},
                                                 targetPlan.affinityAttr);

  IREE::Util::ReturnOp::create(funcBuilder, loc);

  return success();
}

// Replaces all encoded exprs in the original module with loads/gathers from the
// new encoded parameters.
static void replaceEncodedExprs(ArrayRef<TargetPlan> targetPlans) {
  // TODO: support multiple targets by emitting a big switch, a detection
  // function, and then conditionally execute each plan. Each plan should
  // encompass all the required expressions but heterogeneous makes things
  // more complicated in a way I can't yet see. For now we assume all
  // expressions are grouped into a single target and always evaluated (vs.
  // conditionally evaluated per target).
  const TargetPlan &targetPlan = targetPlans.front();

  // Since expressions may share ops we accumulate all the root ops we believe
  // are dead and then burn them down after we're done accessing them.
  SmallVector<Operation *> deadOpWorklist;

  // Note that it's possible for targets to not have all expressions: if we are
  // specializing a heterogeneous module we may produce one encoder module per
  // target each with its own set of placed parameters.
  IndexSetCollection indexSetCollection;
  for (auto &step : targetPlan.steps) {
    // Collect external timepoints once per expression (shared by all outputs).
    Value expressionAwaitTimepoint;
    if (!step.expr->outputs.empty()) {
      OpBuilder timepointBuilder(step.expr->outputs.front().storeOp);
      expressionAwaitTimepoint =
          collectExternalTimepoints(*step.expr, timepointBuilder);
    }

    for (auto &output : step.expr->outputs) {
      const auto *it = step.outputMap.find(&output);
      if (it == step.outputMap.end()) {
        continue; // no serialization required
      }
      auto *indexSet = indexSetCollection.get(output.storeOp);
      OpBuilder builder(output.storeOp);

      // Since each target may have a unique size and packing of their
      // encoded parameters we need to reference the plan-specific parameter.
      const ParameterSubrange &subrange = it->second;
      auto parameterAttr = targetPlan.getNamedParameterAttr(subrange);
      const int64_t storageSize = parameterAttr.getStorageSize();
      Value storageSizeValue = indexSet->get(storageSize);

      // Embed an inline constant referencing the parameter and slice out the
      // subrange (if any).
      Value oldValue = output.storeOp.getStoredGlobalValue();

      Value constantValue = IREE::Stream::AsyncConstantOp::create(
          builder, output.getLoc(), oldValue.getType(),
          expressionAwaitTimepoint, parameterAttr, storageSizeValue,
          step.expr->affinityAttr);
      Value newValue = constantValue;
      if (subrange.offset != 0 || subrange.length != storageSize) {
        // TODO(benvanik): use AsyncSliceOp instead; today ElideAsyncCopiesPass
        // does not do any IPO and inserting slices here forces each parameter
        // to be cloned at execution. Inserting ResourceSubviewOp is only barely
        // safe here because we otherwise don't allow it and know we can run a
        // propagation pass immediately after this pass. It's shady, though, and
        // may block other optimizations.
        //
        // Should be:
        //   newValue = IREE::Stream::AsyncSliceOp::create(
        //       builder, output.getLoc(), constantValue, storageSizeValue,
        //       indexSet->get(subrange.offset),
        //       indexSet->add(subrange.offset, subrange.length),
        //       indexSet->get(subrange.length), step.expr->affinityAttr);
        newValue = IREE::Stream::ResourceSubviewOp::create(
            builder, output.getLoc(), constantValue, storageSizeValue,
            indexSet->get(subrange.offset), indexSet->get(subrange.length));
      }
      output.storeOp.setStoredGlobalValue(newValue);

      // Now that we've replaced a use (but maybe not all uses!) we may be able
      // to kill one or more ops. Since expressions/outputs may share IR we
      // enqueue the deletion check to the end.
      if (auto *producerRootOp = oldValue.getDefiningOp()) {
        // Enqueue ops with no uses for pruning - pruneDeadOps will determine
        // if they're actually safe to delete.
        if (producerRootOp->use_empty()) {
          deadOpWorklist.push_back(producerRootOp);
        }
      }
    }
  }

  // Recursively delete unused operations and their producers.
  pruneDeadOps(std::move(deadOpWorklist));
}

//===----------------------------------------------------------------------===//
// --iree-stream-split-parameter-encoder
//===----------------------------------------------------------------------===//

// Placeholder planning for taking an expression set and producing a
// target-specialized set of parameter indices and an encoding schedule.
//
// TODO: use analysis to identify a set of a target configurations. This may
// be too tricky to do automatically (what would we call the
// configurations?) and require the user to specify the exact names and
// constituent devices. We'd want to take the configuration and prune the
// expression set to those used with involved devices, potentially allow for
// a second specialization round, etc. For now we just have one default
// target and let the tool auto select it.
static FailureOr<TargetPlan> planDefaultTarget(const EncodingExprSet &exprSet,
                                               StringAttr scope,
                                               EncodingPolicy encodingPolicy) {
  LLVM_DEBUG(
      DBGS()
      << "building parameter index and schedule for default target in scope `"
      << scope << "`\n");

  TargetPlan targetPlan;
  targetPlan.name = "all";

  // For now we leave the encoding host target unspecified. This allows the
  // user to compile for any device they want. We could copy the device from
  // the source module if we wanted to do 1:1 encoding:execution.
  targetPlan.affinityAttr = IREE::HAL::DevicePromiseAttr::get(
      scope.getContext(), StringAttr::get(scope.getContext(), "__device_0"),
      -1);

  ParameterIndexBuilder parameterIndexBuilder(scope, encodingPolicy);
  for (const EncodingExpr &expr : exprSet.exprs) {
    auto outputMapOr = parameterIndexBuilder.insertExpr(&expr);
    if (failed(outputMapOr)) {
      return mlir::emitError(expr.getLoc(),
                             "failed to add expression to parameter index");
    }
    targetPlan.appendExpr(&expr, std::move(outputMapOr.value()));
  }
  ParameterIndex parameterIndex = parameterIndexBuilder.finalize();
  for (ParameterEntry &entry : parameterIndex.entries) {
    targetPlan.parameterEntries[{scope, entry.key}] = entry;
  }
  targetPlan.parameterIndices.push_back(std::move(parameterIndex));
  return targetPlan;
}

struct SplitParameterEncoderPass final
    : IREE::Stream::impl::SplitParameterEncoderPassBase<
          SplitParameterEncoderPass> {
  using Base::Base;
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    mlir::ModuleOp moduleOp = getOperation();

    // Scan the program and find candidate expressions.
    EncodingPolicy encodingPolicy;
    encodingPolicy.includeUnmodified =
        mode == IREE::Stream::ParameterEncoderMode::Consolidate;
    encodingPolicy.hoistParameterExpressions = hoistParameterExpressions;
    encodingPolicy.hoistConstantExpressions = hoistConstantExpressions;
    encodingPolicy.maxEncodingGrowthFactor = maxEncodingGrowthFactor;

    EncodingExprSet exprSet = gatherEncodingExprSet(moduleOp, encodingPolicy);

    // Filter expressions by policy (size growth, expression type).
    EncodingExprSet filteredExprSet;
    for (const auto &expr : exprSet.exprs) {
      if (shouldHoistExpression(expr, encodingPolicy)) {
        filteredExprSet.exprs.push_back(expr);
      } else {
        LLVM_DEBUG(DBGS() << "skipping expression based on policy\n");
      }
    }

    if (filteredExprSet.empty()) {
      // No candidates detected (or none the policy approves) so no-op.
      //
      // The user invoking this pass did ask for a new file, though, so we need
      // to at least delete any existing one so the user doesn't get confused
      // (old artifacts from a run where we did write something carried across).
      LLVM_DEBUG(DBGS() << "no candidate expressions detected; skipping pass "
                           "and deleting existing output file\n");
      if (!outputFile.empty()) {
        (void)llvm::sys::fs::remove(outputFile);
      }
      return;
    }

    // Create the new encoder module we'll be populating. Note that we may have
    // multiple targets that contribute functions to the module.
    OwningOpRef<mlir::ModuleOp> encoderModuleOpRef =
        mlir::ModuleOp::create(moduleOp.getLoc(), "encoder");
    mlir::ModuleOp encoderModuleOp = *encoderModuleOpRef;
    encoderModuleOp->setAttr(
        "iree.reflection",
        DictionaryAttr::get(
            context, {
                         NamedAttribute("iree.tool",
                                        StringAttr::get(
                                            context, "iree-encode-parameters")),
                         NamedAttribute("iree.encode.version",
                                        IntegerAttr::get(
                                            IntegerType::get(context, 32), 1)),
                     }));
    OpBuilder encoderBuilder =
        OpBuilder::atBlockBegin(encoderModuleOp.getBody());

    // Today we only support a single target and build the index for that.
    // A few things in here will need to change when we specialize but most of
    // the data structures are set up for it.
    std::string targetOutputScope =
        outputScope.hasValue() ? outputScope.getValue() : "";
    auto defaultTargetOr = planDefaultTarget(
        filteredExprSet, StringAttr::get(context, targetOutputScope),
        encodingPolicy);
    if (failed(defaultTargetOr)) {
      return signalPassFailure();
    }
    LLVM_DEBUG(DBGS() << "note: default target '" << defaultTargetOr->name
                      << "' used in place of target specialization\n");
    SmallVector<TargetPlan> targetPlans;
    targetPlans.push_back(std::move(defaultTargetOr).value());

    // Emit the target detection function used by tools to try to infer the host
    // target (useful for post-deployment encoding).
    addAutoTargetDetectFunc(moduleOp->getLoc(), targetPlans, encoderBuilder);

    // Emit the per-target metadata functions.
    for (const auto &targetPlan : targetPlans) {
      addTargetIndexBuilderFunc(moduleOp->getLoc(), targetPlan, encoderBuilder);
      addTargetEncoderStepsFunc(moduleOp->getLoc(), targetPlan, encoderBuilder);
    }

    // Accumulate object references during cloning so that we can deduplicate
    // and clone them all afterward. This avoids interleaving the objects with
    // the encoder functions - sometimes that is good, but it's easier to read
    // the IR when they aren't.
    SymbolTable sourceSymbolTable(moduleOp);
    SetVector<Operation *> objectsToClone;

    // Capture the last op (if any) so we can insert after it later.
    // This ensures objects go before any encoder functions we're about to add.
    Operation *lastOpBeforeEncoders =
        &*std::prev(encoderModuleOp.getBody()->end(), 1);
    auto markObjectReference = [&](Operation *userOp,
                                   SymbolRefAttr symbolRef) -> LogicalResult {
      auto objectNameAttr = symbolRef.getRootReference();
      auto *objectOp = sourceSymbolTable.lookup(objectNameAttr);
      if (!objectOp) {
        return userOp->emitOpError()
               << "reference to undefined symbol " << symbolRef;
      }
      if (!objectOp->hasTrait<OpTrait::IREE::Util::ObjectLike>()) {
        return userOp->emitOpError()
               << "reference to non-object-like symbol " << symbolRef;
      }
      objectsToClone.insert(objectOp);
      return success();
    };

    // Produce all of the encoder functions and gather the objects we need to
    // clone.
    for (const auto &targetPlan : targetPlans) {
      if (failed(addTargetEncoderFunc(moduleOp->getLoc(), targetPlan,
                                      markObjectReference, encoderBuilder))) {
        return signalPassFailure();
      }
    }

    // Clone all objects referenced by the encoder module.
    // Object-like ops are isolated and safe to copy wholesale.
    // Insert after the last op that existed before we added encoder functions.
    encoderBuilder.setInsertionPointAfter(lastOpBeforeEncoders);
    for (Operation *objectOp : objectsToClone) {
      encoderBuilder.clone(*objectOp);
    }

    // Replace the expressions in the original module with parameter lookups.
    replaceEncodedExprs(targetPlans);

    // CSE to clean up the encoder IR before dumping.
    // This is important for deduplicating operations shared across multiple
    // encoding expressions. When expressions are cloned into the encoder
    // module, shared intermediate operations get duplicated at clone time. CSE
    // removes these duplicates, ensuring efficient encoder module output. The
    // original module likely needs a bit of cleanup but as compilation
    // continues that'll happen.
    {
      IRRewriter rewriter(context);
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, encoderModuleOp);
    }

    if (failed(mlir::verify(encoderModuleOp))) {
      mlir::emitError(encoderModuleOp.getLoc())
          << "failed to verify produced encoder module";
      return signalPassFailure();
    }

    // Write module to the file specified, or stdout if empty.
    if (outputFile.empty()) {
      LLVM_DEBUG(DBGS() << "writing encoder module to stdout...\n");
      OpPrintingFlags flags;
      encoderModuleOp.print(llvm::outs(), flags);
      llvm::outs() << "\n";
    } else {
      LLVM_DEBUG(DBGS() << "writing encoder module to '" << outputFile
                        << "'...\n");
      if (failed(writeModule(encoderModuleOp, outputFile))) {
        LLVM_DEBUG(DBGS() << "MODULE WRITE FAILED\n");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
