// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MATERIALIZETRANSIENTSIZEQUERIESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-materialize-transient-size-queries
//===----------------------------------------------------------------------===//

// NOTE: this is an experimental implementation that is tightly coupled to the
// current behavior of EmplaceTransientsPass. In the future when we have proper
// analysis we'll use that here instead of walking the IR to try to recover our
// pack ops.
//
// For now, we walk the functions with stream.resource.pack ops and if one has
// our magic `stream.experimental.transients` attribute on it we'll know that
// maps to a particular transient resource (currently we assume only one).

// Returns true if an operation is pure (no memory effects, no calls) and safe
// to clone into a size query function.
static bool isPureOp(Operation *op) {
  // Check for function calls.
  if (isa<CallOpInterface>(op)) {
    return false;
  }

  // Check for memory effects.
  // Note: isMemoryEffectFree automatically handles HasRecursiveMemoryEffects
  // and will check nested operations in regions.
  return mlir::isMemoryEffectFree(op);
}

// Returns an iree.reflection dictionary with |namedAttr|.
// If |existingAttr| is non-null all existing attrs will be included in the
// result.
static DictionaryAttr
getReflectionAttrWithNamedAttr(DictionaryAttr existingAttr,
                               NamedAttribute namedAttr) {
  SmallVector<NamedAttribute> reflectionAttrs;
  if (existingAttr) {
    llvm::append_range(reflectionAttrs, existingAttr.getValue());
  }
  reflectionAttrs.push_back(namedAttr);
  return DictionaryAttr::get(namedAttr.getName().getContext(), reflectionAttrs);
}

// Builds a size query function for the given function and pack operations.
static LogicalResult
generateSizeQueryFunction(FunctionOpInterface funcOp, StringRef sizeQueryName,
                          ArrayRef<IREE::Stream::ResourcePackOp> packOps,
                          OpBuilder &moduleBuilder) {
  if (packOps.empty()) {
    return success();
  }

  // Build function type:
  // - For each pack, we'll return its total size.
  // - For simplicity, we return one index per pack (the total size).
  SmallVector<Type> inputTypes(funcOp.getArgumentTypes());
  SmallVector<Type> resultTypes(packOps.size(), moduleBuilder.getIndexType());
  auto sizeQueryFuncType =
      FunctionType::get(funcOp.getContext(), inputTypes, resultTypes);

  // Create the size query function.
  auto sizeQueryFunc = IREE::Util::FuncOp::create(
      moduleBuilder, funcOp.getLoc(), sizeQueryName, sizeQueryFuncType);
  sizeQueryFunc.setPublic();

  // Create entry block and map function arguments.
  Block *entryBlock = sizeQueryFunc.addEntryBlock();
  OpBuilder builder = OpBuilder::atBlockBegin(entryBlock);
  IRMapping mapping;
  for (auto [originalArg, newArg] :
       llvm::zip(funcOp.getArguments(), entryBlock->getArguments())) {
    mapping.map(originalArg, newArg);
  }

  // For each pack, clone the backward slice to compute its total size.
  SmallVector<Value> returnValues;
  for (IREE::Stream::ResourcePackOp packOp : packOps) {
    // Compute backward slice for the total size.
    BackwardSliceOptions options;
    options.omitBlockArguments = false;
    options.inclusive = true;
    options.filter = [&](Operation *op) {
      // Only include pure ops in the slice.
      return isPureOp(op);
    };

    // Get the backward slice from the total length (the only result we need).
    Value totalLengthValue = packOp.getTotalLength();
    llvm::SetVector<Operation *> sliceOps;
    [[maybe_unused]] LogicalResult sliceResult =
        getBackwardSlice(totalLengthValue, &sliceOps, options);

    // Ensure the pack op itself is in the slice.
    sliceOps.insert(packOp);

    // Sort ops in topological order for cloning.
    SetVector<Operation *> sortedOps = mlir::topologicalSort(sliceOps);

    // Clone each operation in the slice.
    for (Operation *op : sortedOps) {
      builder.clone(*op, mapping);
    }

    // Get the mapped total size value.
    Value mappedTotalSize = mapping.lookup(totalLengthValue);
    returnValues.push_back(mappedTotalSize);
  }

  IREE::Util::ReturnOp::create(builder, funcOp.getLoc(), returnValues);

  return success();
}

struct MaterializeTransientSizeQueriesPass
    : public IREE::Stream::impl::MaterializeTransientSizeQueriesPassBase<
          MaterializeTransientSizeQueriesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    OpBuilder moduleBuilder(moduleOp.getContext());

    // Walk all functions and find those with pack ops that have the
    // stream.experimental.transients attribute.
    SmallVector<std::pair<FunctionOpInterface,
                          SmallVector<IREE::Stream::ResourcePackOp>>>
        funcsWithTransients;
    const auto gatherPackOps = [](FunctionOpInterface funcOp) {
      SmallVector<IREE::Stream::ResourcePackOp> packOps;
      funcOp.walk([&](IREE::Stream::ResourcePackOp packOp) {
        if (packOp->hasAttr("stream.experimental.transients")) {
          packOp->removeAttr("stream.experimental.transients");
          packOps.push_back(packOp);
        }
      });
      return packOps;
    };
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      auto packOps = gatherPackOps(funcOp);
      if (!packOps.empty()) {
        funcsWithTransients.push_back(std::make_pair(funcOp, packOps));
      }
    }

    // For each function with transients generate a size query function.
    for (auto &[funcOp, packOps] : funcsWithTransients) {
      moduleBuilder.setInsertionPointAfter(funcOp);

      std::string sizeQueryName = (funcOp.getName() + "_transients_size").str();
      if (failed(generateSizeQueryFunction(funcOp, sizeQueryName, packOps,
                                           moduleBuilder))) {
        return signalPassFailure();
      }

      // Add iree.abi.transients.size reflection entry so users know which size
      // query function they should use for the entry point.
      auto sizeQueryAttr = moduleBuilder.getNamedAttr(
          "iree.abi.transients.size",
          FlatSymbolRefAttr::get(moduleOp.getContext(), sizeQueryName));
      funcOp->setAttr(
          "iree.reflection",
          getReflectionAttrWithNamedAttr(
              funcOp->getAttrOfType<DictionaryAttr>("iree.reflection"),
              sizeQueryAttr));
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
