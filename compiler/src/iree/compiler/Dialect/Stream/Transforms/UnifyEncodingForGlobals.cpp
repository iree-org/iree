// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
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

//===----------------------------------------------------------------------===//
// Analysis.
//===----------------------------------------------------------------------===//

// Information about an encoded global and its relationship to its source.
struct EncodedGlobalInfo {
  IREE::Util::GlobalOpInterface encodedGlobal;
  Attribute encodingAttr;
  IREE::Stream::TensorEncodeOp encodeOp;
  IREE::Stream::AffinityAttr affinityAttr;
};

// Information about a source global and all its encoded versions.
struct SourceGlobalInfo {
  IREE::Util::GlobalOpInterface sourceGlobal;
  SmallVector<EncodedGlobalInfo> encodedVersions;
};

// Finds the global store op that uses the given value, traversing through
// passthrough ops like clone.
// TODO(#22485): Handle other passthrough ops (e.g., transfer).
static IREE::Util::GlobalStoreOpInterface findStoreOp(Value value) {
  for (auto user : value.getUsers()) {
    if (auto store = dyn_cast<IREE::Util::GlobalStoreOpInterface>(user)) {
      return store;
    }
    if (auto cloneOp = dyn_cast<IREE::Stream::AsyncCloneOp>(user)) {
      if (auto store = findStoreOp(cloneOp.getResult())) {
        return store;
      }
    }
  }
  return nullptr;
}

// Analyzes a module to find immutable globals that have multiple encoded
// versions. Use run() to perform analysis, then query results with
// getSourcesWithMultipleEncodings() or getSourceGlobals().
class GlobalEncodingAnalyzer {
public:
  explicit GlobalEncodingAnalyzer(ModuleOp moduleOp)
      : moduleOp(moduleOp), symbolTable(moduleOp) {}

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

  const llvm::MapVector<StringRef, SourceGlobalInfo> &getSourceGlobals() const {
    return sourceGlobals;
  }

private:
  // Walks all initializers to find encoding patterns and populates
  // sourceGlobals map. Looks for patterns like:
  //   %source = util.global.load @source_global
  //   %encoded = stream.tensor.encode %source ...
  //   util.global.store %encoded, @encoded_global
  // Only considers immutable source and encoded globals.
  LogicalResult collectGlobalEncodings();

  // Traces from encode op's source operand back to a source global.
  // Returns nullptr if tracing fails or source is mutable.
  IREE::Util::GlobalOpInterface traceToSourceGlobal(Value value);

  // Returns true if the type has a recognized encoding attribute.
  bool hasRecognizedEncoding(Type type) const;

  ModuleOp moduleOp;
  SymbolTable symbolTable;

  // Maps source global name to its info. Populated by run().
  llvm::MapVector<StringRef, SourceGlobalInfo> sourceGlobals;
};

LogicalResult GlobalEncodingAnalyzer::run() {
  LDBG() << "=== GlobalEncodingAnalyzer::run() ===";
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
      if (auto affinityOp =
              dyn_cast<IREE::Stream::AffinityOpInterface>(*encodeOp)) {
        encodedInfo.affinityAttr = affinityOp.getAffinityAttr();
      }
      auto &sourceInfo = sourceGlobals[sourceGlobal.getGlobalName()];
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
  // TODO(#22485): Extend to general subgraph canonicalization.
  // Currently only handles direct global loads + passthrough ops.
  SmallVector<Value> worklist;
  DenseSet<Value> visited;
  IREE::Util::GlobalOpInterface foundSourceGlobal;
  worklist.push_back(value);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second) {
      continue;
    }
    Operation *defOp = current.getDefiningOp();
    if (!defOp) {
      LDBG() << "      Bail: block argument";
      return nullptr;
    }
    bool shouldContinue =
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
              if (foundSourceGlobal && foundSourceGlobal != globalOp) {
                LDBG() << "      Bail: multiple source globals";
                return false;
              }
              foundSourceGlobal = globalOp;
              auto initAttr = globalOp.getGlobalInitialValue();
              if (initAttr) {
                // Has inline initial value, done tracing.
                return true;
              }
              // No inline initial value; trace through stores in initializers.
              IREE::Util::GlobalTable globalTable(moduleOp);
              globalTable.rebuild();
              auto &global = globalTable.lookup(globalName);
              for (auto storeOp : global.storeOps) {
                worklist.push_back(storeOp.getStoredGlobalValue());
              }
              return true;
            })
            .Case([&](IREE::Stream::TensorConstantOp) {
              // Constant is a valid leaf source.
              return true;
            })
            .Default([&](Operation *op) {
              LDBG() << "      Bail: unknown op " << op->getName();
              return false;
            });
    if (!shouldContinue) {
      return nullptr;
    }
  }
  return foundSourceGlobal;
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

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
    auto candidates = analyzer.getSourcesWithMultipleEncodings();
    if (candidates.empty()) {
      LDBG() << "No source globals with multiple encodings found.";
      return;
    }
    LDBG() << "Found " << candidates.size()
           << " source globals with multiple encodings:";
    for (auto name : candidates) {
      LDBG() << "  - " << name;
    }

    // Unify encodings for each source global with multiple encodings.
    for (auto sourceName : candidates) {
      const auto &sourceInfo =
          analyzer.getSourceGlobals().find(sourceName)->second;

      // TODO(#22485): Select unified encoding via resolver. For now, use
      // identity.
      auto unifiedEncoding =
          IREE::Encoding::IdentityAttr::get(moduleOp.getContext());

      // Update each encode op to use the unified encoding.
      for (const auto &encodedInfo : sourceInfo.encodedVersions) {
        auto encodeOp = encodedInfo.encodeOp;
        auto oldResultType =
            cast<RankedTensorType>(encodeOp.getResultEncoding());
        auto newResultType = RankedTensorType::get(
            oldResultType.getShape(), oldResultType.getElementType(),
            unifiedEncoding);
        encodeOp.setResultEncodingAttr(TypeAttr::get(newResultType));
        LDBG() << "  Updated encode op: " << encodeOp;

        // Update stream.tensor.sizeof if it defines the result_size.
        if (auto sizeofOp =
                encodeOp.getResultSize()
                    .getDefiningOp<IREE::Stream::TensorSizeOfOp>()) {
          sizeofOp.setEncodingAttr(TypeAttr::get(newResultType));
          LDBG() << "  Updated sizeof op: " << sizeofOp;
        }

        // TODO(#22485): Update dispatch sites and executables.
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
