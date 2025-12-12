// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Analysis/Explorer.h"
#include "iree/compiler/Dialect/Util/Analysis/GlobalTable.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/DebugLog.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-unify-encoding-for-globals"

#define GEN_PASS_DEF_UNIFYENCODINGFORGLOBALSPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

/// Returns a stably sorted list of dialect interfaces of T for all dialects
/// used within the given module.
template <typename T>
SmallVector<const T *> gatherUsedDialectInterfaces(mlir::ModuleOp moduleOp) {
  SmallPtrSet<const T *, 4> resultSet;
  for (auto dialect : moduleOp.getContext()->getLoadedDialects()) {
    auto *dialectInterface = dialect->getRegisteredInterface<T>();
    if (!dialectInterface)
      continue;
    resultSet.insert(dialectInterface);
  }

  // NOTE: to ensure deterministic output we sort the result so that imports are
  // always added in a consistent order.
  auto results = llvm::to_vector_of<const T *>(resultSet);
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

//===----------------------------------------------------------------------===//
// Analysis.
//===----------------------------------------------------------------------===//

// Information about an encoded global and its relationship to its source.
struct EncodedGlobalInfo {
  // The destination global for the final encoded data.
  IREE::Util::GlobalOpInterface encodedGlobal;
  Attribute encodingAttr;
  IREE::Stream::TensorSizeOfOp sizeofOp;
  IREE::Stream::TensorEncodeOp encodeOp;
};

// Information about a source global and all its encoded versions.
struct SourceGlobalInfo {
  IREE::Util::GlobalOpInterface sourceGlobal;
  SmallVector<EncodedGlobalInfo> encodedVersions;
};

// Returns the global store op that uses the given value, traversing through
// passthrough ops like clone. Returns nullptr if not found or multiple users.
static IREE::Util::GlobalStoreOpInterface findStoreOp(Value value) {
  if (!llvm::hasSingleElement(value.getUsers())) {
    return nullptr;
  }
  Operation *user = *value.getUsers().begin();
  if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(user)) {
    return store;
  }
  if (auto cloneOp = dyn_cast<IREE::Stream::AsyncCloneOp>(user)) {
    return findStoreOp(cloneOp.getResult());
  }
  return nullptr;
}

// Analyzes a module to find immutable globals that have multiple encoded
// versions, and computes unified encodings for them using the layout resolver
// from the dialect interface. Use run() to perform analysis, then query results
// with getSourcesWithMultipleEncodings(), getSourceGlobals(), or
// getUnifiedEncoding().
class GlobalEncodingAnalyzer {
public:
  explicit GlobalEncodingAnalyzer(ModuleOp moduleOp)
      : moduleOp(moduleOp), symbolTable(moduleOp), globalTable(moduleOp) {}

  // Runs the full analysis: sets up resolver, collects encodings, and computes
  // unified encodings.
  LogicalResult run();

  // Returns all source globals that have multiple distinct encodings.
  SmallVector<StringRef> getSourcesWithMultipleEncodings() const {
    SmallVector<StringRef> result;
    for (const auto &[name, info] : sourceGlobals) {
      llvm::SmallDenseSet<Attribute, 4> uniqueEncodings;
      for (const EncodedGlobalInfo &encoded : info.encodedVersions) {
        uniqueEncodings.insert(encoded.encodingAttr);
      }
      if (uniqueEncodings.size() > 1) {
        result.push_back(name);
      }
    }
    return result;
  }

  // Returns the SourceGlobalInfo for the given source global name, or
  // std::nullopt if not found.
  std::optional<SourceGlobalInfo> getSourceGlobals(StringRef name) const {
    if (sourceGlobals.contains(name)) {
      return sourceGlobals.find(name)->second;
    }
    return std::nullopt;
  }

  // Returns the unified encoding for the given source global name, or
  // std::nullopt if not found.
  std::optional<Attribute> getUnifiedEncoding(StringRef name) const {
    auto it = unifiedEncodings.find(name);
    if (it != unifiedEncodings.end()) {
      return it->second;
    }
    return std::nullopt;
  }

private:
  // Sets up the layout resolver from dialect interfaces.
  LogicalResult setupLayoutResolver();

  // Walks all initializers to find encoding patterns and populates
  // sourceGlobals map. Looks for patterns like:
  //   %source = util.global.load @source_global
  //   %encoded = stream.tensor.encode %source ...
  //   util.global.store %encoded, @encoded_global
  // Only considers immutable source and encoded globals.
  LogicalResult collectGlobalEncodings();

  // Computes unified encodings for all source globals with multiple encodings.
  LogicalResult computeUnifiedEncodings();

  // Traces from encode op's source operand back to a source global.
  // Returns nullptr if tracing fails or source is mutable.
  IREE::Util::GlobalOpInterface traceToSourceGlobal(Value value);

  // Returns true if the type has a recognized encoding attribute.
  bool hasRecognizedEncoding(Type type) const;

  ModuleOp moduleOp;
  SymbolTable symbolTable;
  IREE::Util::GlobalTable globalTable;

  // Layout resolver function from dialect interface.
  IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr;

  // Maps source global name to its info. Populated by run(). The global name
  // must match the name of `sourceGlobal` inside SourceGlobalInfo. StringRef is
  // used for easier lookup, which works better with SymbolTable, etc.
  llvm::MapVector<StringRef, SourceGlobalInfo> sourceGlobals;

  // Maps source global name to its unified encoding. Populated by run().
  llvm::DenseMap<StringRef, Attribute> unifiedEncodings;
};

LogicalResult GlobalEncodingAnalyzer::run() {
  LDBG() << "=== GlobalEncodingAnalyzer::run() ===";

  if (failed(setupLayoutResolver())) {
    return failure();
  }

  globalTable.rebuild();
  if (failed(collectGlobalEncodings())) {
    return failure();
  }
  LDBG_OS([&](llvm::raw_ostream &os) {
    os << "Analysis complete:\n";
    os << "  Source globals: " << sourceGlobals.size() << "\n";
    for (const auto &[name, info] : sourceGlobals) {
      os << "    " << name << ": " << info.encodedVersions.size()
         << " encoded versions\n";
    }
  });

  if (failed(computeUnifiedEncodings())) {
    return failure();
  }

  return success();
}

LogicalResult GlobalEncodingAnalyzer::setupLayoutResolver() {
  auto usedDialects = gatherUsedDialectInterfaces<
      IREE::Stream::AffinityAnalysisDialectInterface>(moduleOp);
  if (usedDialects.size() != 1) {
    LDBG() << "Expected only one dialect implementing "
              "AffinityAnalysisDialectInterface";
    return failure();
  }
  resolveLayoutAttr = usedDialects[0]->makeLayoutAttrResolver(moduleOp);
  return success();
}

LogicalResult GlobalEncodingAnalyzer::computeUnifiedEncodings() {
  SmallVector<StringRef> candidates = getSourcesWithMultipleEncodings();
  if (candidates.empty()) {
    LDBG() << "No source globals with multiple encodings found.";
    return success();
  }

  LDBG() << "Found " << candidates.size()
         << " source globals with multiple encodings:";
  for (auto name : candidates) {
    LDBG() << "  - " << name;
  }

  // Build queries for layout resolution.
  SmallVector<IREE::Stream::AffinityAndOpPair> queries;
  for (StringRef sourceName : candidates) {
    std::optional<SourceGlobalInfo> sourceInfo = getSourceGlobals(sourceName);
    for (EncodedGlobalInfo &encodedInfo : sourceInfo->encodedVersions) {
      queries.push_back({encodedInfo.encodeOp.getAffinityAttr(),
                         encodedInfo.encodedGlobal});
    }
  }

  // Resolve layout attributes for all queries.
  llvm::DenseMap<IREE::Stream::AffinityAndOpPair, SetVector<Attribute>>
      cachedLayoutAttrs;
  if (failed(resolveLayoutAttr(queries, cachedLayoutAttrs))) {
    LDBG() << "failed to resolve layouts for a query";
    return failure();
  }

  // Compute unified encoding for each source global.
  MLIRContext *ctx = moduleOp.getContext();
  for (StringRef sourceName : candidates) {
    SetVector<Attribute> layoutResolvers;
    SmallVector<Attribute> encodingAttrVersions;
    std::optional<SourceGlobalInfo> sourceInfo = getSourceGlobals(sourceName);
    for (EncodedGlobalInfo &encodedInfo : sourceInfo->encodedVersions) {
      const SetVector<Attribute> &resolvers =
          cachedLayoutAttrs[IREE::Stream::AffinityAndOpPair(
              encodedInfo.encodeOp.getAffinityAttr(),
              encodedInfo.encodedGlobal)];
      layoutResolvers.insert(resolvers.begin(), resolvers.end());
      encodingAttrVersions.push_back(encodedInfo.encodingAttr);
    }

    // TODO: It is not clear which encoding to pick when there are multiple
    // layout resolvers. For now, just use identity encoding for safety.
    if (layoutResolvers.size() != 1) {
      unifiedEncodings[sourceName] = IREE::Encoding::IdentityAttr::get(ctx);
      continue;
    }

    // Invalid layout resolver, use identity encoding.
    IREE::Encoding::LayoutResolverAttr layoutResolver =
        dyn_cast<IREE::Encoding::LayoutResolverAttr>(layoutResolvers[0]);
    if (!layoutResolver) {
      unifiedEncodings[sourceName] = IREE::Encoding::IdentityAttr::get(ctx);
      continue;
    }

    LDBG() << "Use encoding resolver " << layoutResolver
           << " to unify encodings for source global: " << sourceName;
    unifiedEncodings[sourceName] =
        layoutResolver.getUnifiedEncoding(encodingAttrVersions);
  }

  return success();
}

bool GlobalEncodingAnalyzer::hasRecognizedEncoding(Type type) const {
  auto rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (!rankedTensorType) {
    return false;
  }
  auto encoding = rankedTensorType.getEncoding();
  if (!encoding) {
    return false;
  }
  return isa<IREE::Encoding::SerializableAttr>(encoding);
}

LogicalResult GlobalEncodingAnalyzer::collectGlobalEncodings() {
  LDBG() << "--- collectGlobalEncodings ---";
  // Walk all initializers to find encoding patterns:
  //   %source = util.global.load @source_global
  //   %encoded = stream.tensor.encode %source ...
  //   util.global.store %encoded, @encoded_global
  for (auto initOp : moduleOp.getOps<IREE::Util::InitializerOp>()) {
    initOp.walk([&](IREE::Stream::TensorEncodeOp encodeOp) {
      LDBG() << "  Found TensorEncodeOp: " << encodeOp;
      Type resultType = encodeOp.getResultEncoding();
      if (!hasRecognizedEncoding(resultType)) {
        LDBG() << "    Skipping: no recognized encoding";
        return;
      }

      auto storeOp = findStoreOp(encodeOp.getResult());
      if (!storeOp) {
        LDBG() << "    Skipping: no store found";
        return;
      }
      StringRef encodedGlobalName = storeOp.getGlobalName();
      auto encodedGlobalOp =
          symbolTable.lookup<IREE::Util::GlobalOpInterface>(encodedGlobalName);
      if (!encodedGlobalOp) {
        LDBG() << "    Skipping: encoded global not found";
        return;
      }
      if (encodedGlobalOp.isGlobalMutable()) {
        LDBG() << "    Skipping: mutable encoded global";
        return;
      }
      auto sourceGlobal = traceToSourceGlobal(encodeOp.getSource());
      if (!sourceGlobal) {
        LDBG() << "    Skipping: failed to trace to source global";
        return;
      }

      LDBG() << "    Source: " << sourceGlobal.getGlobalName();
      EncodedGlobalInfo encodedInfo;
      encodedInfo.encodedGlobal = encodedGlobalOp;
      encodedInfo.encodeOp = encodeOp;
      encodedInfo.encodingAttr =
          cast<RankedTensorType>(resultType).getEncoding();
      encodedInfo.sizeofOp = encodeOp.getResultSize()
                                 .getDefiningOp<IREE::Stream::TensorSizeOfOp>();
      if (!encodedInfo.sizeofOp) {
        LDBG() << "    Skipping: no sizeof op found for result size, can't "
                  "update corresponding size computations later";
        return;
      }
      SourceGlobalInfo &sourceInfo =
          sourceGlobals[sourceGlobal.getGlobalName()];
      if (!sourceInfo.sourceGlobal) {
        sourceInfo.sourceGlobal = sourceGlobal;
      }
      sourceInfo.encodedVersions.push_back(encodedInfo);
    });
  }
  return success();
}

IREE::Util::GlobalOpInterface
GlobalEncodingAnalyzer::traceToSourceGlobal(Value value) {
  IREE::Util::GlobalOpInterface foundSourceGlobal;
  bool shouldContinue = true;
  while (shouldContinue) {
    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
      LDBG() << "      Bail: block argument";
      return nullptr;
    }
    shouldContinue = false;
    bool isValid =
        llvm::TypeSwitch<Operation *, bool>(defOp)
            .Case([&](IREE::Util::GlobalLoadOpInterface loadOp) {
              StringRef globalName = loadOp.getGlobalName();
              auto globalOp =
                  symbolTable.lookup<IREE::Util::GlobalOpInterface>(globalName);
              if (!globalOp) {
                LDBG() << "      Bail: global not found: " << globalName;
                return false;
              }
              if (globalOp.isGlobalMutable()) {
                LDBG() << "      Bail: mutable source global";
                return false;
              }
              assert(!foundSourceGlobal &&
                     "should only find one source global");
              foundSourceGlobal = globalOp;
              // Has inline initial value, done tracing.
              if (globalOp.getGlobalInitialValue()) {
                return true;
              }
              auto &global = globalTable.lookup(globalName);
              if (global.storeOps.size() != 1) {
                LDBG() << "      Bail: expected immutable global is only "
                          "initialized once";
                return false;
              }
              // Continue tracing the source of the store op.
              value = global.storeOps[0].getStoredGlobalValue();
              shouldContinue = true;
              return true;
            })
            .Case([&](IREE::Stream::TensorConstantOp) {
              // Constant is a valid leaf source.
              return true;
            })
            .Case([&](IREE::Stream::AsyncCloneOp cloneOp) {
              // Clone is a passthrough op, continue tracing through its source.
              value = cloneOp.getSource();
              shouldContinue = true;
              return true;
            })
            .Default([&](Operation *op) {
              LDBG() << "      Bail: unknown op " << op->getName();
              return false;
            });
    if (!isValid) {
      return nullptr;
    }
  }
  return foundSourceGlobal;
}

// Maps a tensor op to a set of (operand index -> new encoding).
using OperandEncodingUpdates = llvm::DenseMap<unsigned, Attribute>;
using TensorEncodingUpdates =
    llvm::DenseMap<Operation *, OperandEncodingUpdates>;

// Updates encoding attributes for a TensorDispatchOp.
static void updateTensorDispatchOp(TensorDispatchOp dispatchOp,
                                   const OperandEncodingUpdates &operandUpdates,
                                   IRRewriter &rewriter) {
  // Update operand encodings.
  // The operand_encodings attribute has the same length as getMixedOperands().
  // For non-affinity types (e.g., index), the encoding is just the type.
  // For affinity types, the encoding is a RankedTensorType with encoding attr.
  SmallVector<Attribute> newOperandEncodings;
  for (auto [idx, operand, typeAttr] :
       llvm::enumerate(dispatchOp.getMixedOperands(),
                       dispatchOp.getOperandEncodings().getValue())) {
    Type type = cast<TypeAttr>(typeAttr).getValue();
    if (!isa<IREE::Stream::AffinityTypeInterface>(type)) {
      newOperandEncodings.push_back(typeAttr);
      continue;
    }
    if (!operandUpdates.contains(idx)) {
      newOperandEncodings.push_back(typeAttr);
      continue;
    }
    Attribute newEncoding = operandUpdates.lookup(idx);
    auto tensorType = cast<RankedTensorType>(type);
    newOperandEncodings.push_back(
        TypeAttr::get(tensorType.cloneWithEncoding(newEncoding)));
    LDBG() << "  Updated dispatch operand encoding at index " << idx << " to "
           << newEncoding;
  }
  dispatchOp.setOperandEncodingsAttr(
      ArrayAttr::get(dispatchOp.getContext(), newOperandEncodings));

  // Update result encodings for tied operands and track which results need
  // re-encoding for downstream users.
  //
  // NOTE: This is a rare case that exists primarily for correctness as a safe
  // fallback. In practice, tied operands with encodings that need unification
  // are uncommon. The re-encode op inserted here ensures downstream users see
  // the original encoding they expect, even though the dispatch internally
  // uses the unified encoding.
  auto tiedOp = cast<IREE::Util::TiedOpInterface>(dispatchOp.getOperation());

  // Collect old encodings and build new result encodings.
  SmallVector<Type> newResultEncodings;
  SmallVector<std::pair<OpResult, RankedTensorType>> resultsToReencode;
  for (auto [result, typeAttr] :
       llvm::zip_equal(dispatchOp.getResults(),
                       dispatchOp.getResultEncodings().getValue())) {
    Type type = cast<TypeAttr>(typeAttr).getValue();
    if (!isa<IREE::Stream::ResourceType>(result.getType())) {
      newResultEncodings.push_back(type);
      continue;
    }
    OpOperand *tiedOperand = tiedOp.getTiedResultOpOperand(result);
    if (!tiedOperand) {
      newResultEncodings.push_back(type);
      continue;
    }
    if (!operandUpdates.contains(tiedOperand->getOperandNumber())) {
      newResultEncodings.push_back(type);
      continue;
    }
    // Track old encoding for re-encode op insertion.
    auto rankedTensorType = cast<RankedTensorType>(type);
    resultsToReencode.push_back({result, rankedTensorType});
    newResultEncodings.push_back(rankedTensorType.cloneWithEncoding(
        operandUpdates.lookup(tiedOperand->getOperandNumber())));
  }
  dispatchOp.setResultEncodingsAttr(
      rewriter.getTypeArrayAttr(newResultEncodings));

  // Insert re-encode ops after the dispatch for results that were updated.
  // This converts results back to the original encoding for downstream users.
  if (resultsToReencode.empty()) {
    return;
  }

  // Build a map from result index to its encoding dims by iterating through
  // the flattened result_encoding_dims list.
  SmallVector<ValueRange> resultEncodingDimsMap(newResultEncodings.size());
  ValueRange remainingDims = dispatchOp.getResultEncodingDims();
  for (auto [idx, encodingType] : llvm::enumerate(newResultEncodings)) {
    auto shapedType = cast<ShapedType>(encodingType);
    int64_t dynamicDimCount = shapedType.getNumDynamicDims();
    resultEncodingDimsMap[idx] = remainingDims.take_front(dynamicDimCount);
    remainingDims = remainingDims.drop_front(dynamicDimCount);
  }

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(dispatchOp);
  for (auto [result, oldType] : resultsToReencode) {
    unsigned resultIdx = result.getResultNumber();
    Value resultSize = dispatchOp.getResultSize(resultIdx);
    auto newType = cast<RankedTensorType>(newResultEncodings[resultIdx]);
    ValueRange encodingDims = resultEncodingDimsMap[resultIdx];
    Value oldSize = TensorSizeOfOp::create(
        rewriter, dispatchOp.getLoc(), rewriter.getIndexType(),
        TypeAttr::get(oldType), encodingDims, dispatchOp.getAffinityAttr());
    auto reencodeOp =
        TensorEncodeOp::create(rewriter, dispatchOp.getLoc(), result.getType(),
                               result, TypeAttr::get(newType),
                               /*source_encoding_dims=*/encodingDims,
                               resultSize, TypeAttr::get(oldType),
                               /*result_encoding_dims=*/encodingDims, oldSize,
                               dispatchOp.getAffinityAttr());
    rewriter.replaceAllUsesExcept(result, reencodeOp.getResult(), reencodeOp);
    LDBG() << "  Inserted re-encode op for result " << resultIdx << ": "
           << reencodeOp;
  }
}

// Applies all cached encoding updates to tensor ops.
static void applyTensorEncodingUpdates(TensorEncodingUpdates &updates) {
  for (auto &[op, operandUpdates] : updates) {
    IRRewriter rewriter(op->getContext());
    // TODO: Handle other TensorPhaseOp ops (TensorFillOp, etc.) via TypeSwitch.
    if (auto dispatchOp = dyn_cast<TensorDispatchOp>(op)) {
      updateTensorDispatchOp(dispatchOp, operandUpdates, rewriter);
    }
  }
}

// Collects updates for stream tensor ops by walking from global loads.
static void collectUpdatesForStreamTensorOps(Explorer &explorer,
                                             EncodedGlobalInfo &encodedInfo,
                                             Attribute newEncoding,
                                             TensorEncodingUpdates &updates) {
  StringRef globalName = encodedInfo.encodedGlobal.getGlobalName();
  LDBG() << "  Collecting updates for global: " << globalName;

  // Get loads from the explorer's global info.
  const Explorer::GlobalInfo *globalInfo =
      explorer.queryGlobalInfoFrom(globalName, encodedInfo.encodedGlobal);
  assert(globalInfo &&
         "expected global info to be present in explorer for encoded global");
  SmallVector<Value> worklist;
  for (auto loadOp : globalInfo->getLoads()) {
    worklist.push_back(loadOp.getLoadedGlobalValue());
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    explorer.walkTransitiveUses(value, [&](OpOperand &operand) {
      Operation *user = operand.getOwner();
      if (auto cloneOp = dyn_cast<AsyncCloneOp>(user)) {
        LDBG() << "      Following clone: " << cloneOp;
        worklist.push_back(cloneOp.getResult());
        return WalkResult::advance();
      }

      // Only stream tensor ops need to be updated. Skip other operations.
      if (!user->hasTrait<OpTrait::IREE::Stream::TensorPhaseOp>()) {
        return WalkResult::advance();
      }

      // TODO: Handle other tensor phase ops (TensorFillOp, etc.)
      auto dispatchOp = dyn_cast<IREE::Stream::TensorDispatchOp>(user);
      if (!dispatchOp) {
        return WalkResult::advance();
      }

      // The operand number is the index in the full operand list (including
      // workload). We need the index in getMixedOperands() for encoding lookup.
      unsigned mixedOperandIdx =
          operand.getOperandNumber() - dispatchOp.getWorkload().size();
      LDBG() << "      Found TensorDispatchOp operand " << mixedOperandIdx;
      updates[user][mixedOperandIdx] = newEncoding;
      return WalkResult::advance();
    });
  }
}

struct UnifyEncodingForGlobalsPass
    : public impl::UnifyEncodingForGlobalsPassBase<
          UnifyEncodingForGlobalsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    GlobalEncodingAnalyzer analyzer(moduleOp);
    if (failed(analyzer.run())) {
      LDBG() << "Analysis failed, skipping.";
      return;
    }

    SmallVector<StringRef> candidates =
        analyzer.getSourcesWithMultipleEncodings();
    if (candidates.empty()) {
      return;
    }

    // Unify encodings for each source global with multiple encodings, and cache
    // the updates.
    Explorer explorer(moduleOp, TraversalAction::RECURSE);
    explorer.setOpAction<IREE::Stream::ExecutableOp>(TraversalAction::IGNORE);
    explorer.initialize();
    TensorEncodingUpdates tensorEncodingUpdates;
    for (StringRef sourceName : candidates) {
      std::optional<SourceGlobalInfo> sourceInfo =
          analyzer.getSourceGlobals(sourceName);
      // Update each encode op to use the unified encoding.
      Attribute unifiedEncoding = *analyzer.getUnifiedEncoding(sourceName);
      LDBG() << "Unifying encodings for source global: " << sourceName
             << " to " << unifiedEncoding;
      for (EncodedGlobalInfo &encodedInfo : sourceInfo->encodedVersions) {
        auto encodeOp = encodedInfo.encodeOp;
        auto oldResultType =
            cast<RankedTensorType>(encodeOp.getResultEncoding());
        RankedTensorType newResultType =
            oldResultType.cloneWithEncoding(unifiedEncoding);
        encodeOp.setResultEncodingAttr(TypeAttr::get(newResultType));
        LDBG() << "  Updated encode op: " << encodeOp;
        encodedInfo.sizeofOp.setEncodingAttr(TypeAttr::get(newResultType));
        LDBG() << "  Updated sizeof op: " << encodedInfo.sizeofOp;
        collectUpdatesForStreamTensorOps(explorer, encodedInfo, unifiedEncoding,
                                         tensorEncodingUpdates);
      }
    }

    // Apply all tensor encoding updates in one shot.
    applyTensorEncodingUpdates(tensorEncodingUpdates);

    // TODO(#22485): Update executables.
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
