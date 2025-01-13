// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTraits.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define DEBUG_TYPE "iree-stream-specialize-encodings"

#define GEN_PASS_DEF_SPECIALIZEENCODINGSPASS
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
  SmallVector<const T *> results = {resultSet.begin(), resultSet.end()};
  llvm::sort(
      results, +[](const T *a, const T *b) {
        return a->getDialect()->getNamespace().compare(
                   b->getDialect()->getNamespace()) < 0;
      });
  return results;
}

/// Returns the affinities of the `dispatchOp`'s resource operands. An empty
/// array attribute indicates that the resource operand affinity is not found.
/// Usually, it happens when it fails on affinity analysis.
/// Note that the size of the result might not equal to the number of resource
/// operands. If a resource operand type is not AffinityType, it is skipped.
static SmallVector<Attribute>
getResourceOperandsAffinities(IREE::Stream::AffinityAnalysis &affinityAnalysis,
                              IREE::Stream::AsyncDispatchOp dispatchOp) {
  SmallVector<Attribute> result;
  Builder b(dispatchOp.getContext());
  auto emptyArray = b.getArrayAttr({});
  for (auto operand : dispatchOp.getResourceOperands()) {
    // Skip if the operand type is not AffinityType.
    if (!isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
      continue;
    }
    SmallVector<IREE::Stream::AffinityAttr> affinities;
    if (!affinityAnalysis.tryLookupResourceAffinity(operand, affinities)) {
      result.push_back(emptyArray);
      continue;
    }
    result.push_back(b.getArrayAttr(llvm::to_vector_of<Attribute>(affinities)));
  }
  return result;
}

/// Duplicates stream.executables based on the affinity analysis of
/// stream.async.dispatch ops. Some executables can be launched by different
/// devices. It can produce wrong codegen artifacts when bindings types are
/// encoded (i.e., the tensor type has an encoding attribute). Because they can
/// result in different layouts, especially when multi-device is involved. E.g.,
/// say that device_a and device_b interpret a tensor type with encodings in
/// different layouts, and there is an executable that can be launch with
/// resources from either device_a or device_b. It is confusing what the input
/// layouts for the executable because there are two possibilities. In this
/// case, we have to duplicate the executable with updated encoding, and modify
/// the dispatch to launch proper executable based on device analysis.
static LogicalResult duplicateExecutablesPerAffinityVariant(
    ModuleOp moduleOp, SymbolTable symbolTable, FunctionOpInterface funcOp,
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  MLIRContext *ctx = moduleOp.getContext();
  IRRewriter rewriter(ctx);

  // 1. Gather per-export [execution affinity -> [resource affinities]] map.
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return moduleOp.emitError("failed on running affinity analysis");
  }
  SmallVector<IREE::Stream::AsyncDispatchOp> candidates;
  funcOp.walk(
      [&](IREE::Stream::AsyncDispatchOp op) { candidates.push_back(op); });

  // export -> [affinity -> array per resource of affinities PVS].
  DenseMap<IREE::Stream::ExecutableExportOp,
           SetVector<std::pair<IREE::Stream::AffinityAttr, ArrayAttr>>>
      exportToDispatchSites;

  llvm::MapVector<IREE::Stream::AsyncDispatchOp, SmallVector<Attribute>>
      resourceAffinities;
  for (auto dispatchOp : candidates) {
    SmallVector<IREE::Stream::AffinityAttr> execAffinities;
    if (!affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                     execAffinities)) {
      return dispatchOp.emitError("failed on execution affinity lookup");
    }
    assert(execAffinities.size() == 1 &&
           "We should only have a single execution "
           "affinity when running the pass.");

    SmallVector<Attribute> operandAffinityAttrs =
        getResourceOperandsAffinities(affinityAnalysis, dispatchOp);
    resourceAffinities[dispatchOp] = operandAffinityAttrs;

    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      exportToDispatchSites[exportOp].insert(std::make_pair(
          execAffinities[0], rewriter.getArrayAttr(operandAffinityAttrs)));
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Dump of exportToDispatchSites\n";
    for (auto [exportOp, affinities] : exportToDispatchSites) {
      llvm::dbgs() << "  ExportOp: " << exportOp.getSymName() << "\n";
      for (auto [execAffinity, resourceAffinities] : affinities) {
        llvm::dbgs() << "    executaion affinity: " << execAffinity << "\n";
        llvm::dbgs() << "    resource affinities: " << resourceAffinities
                     << "\n";
      }
    }
  });

  // 2. Duplicate executables for each unqiue resource affinities.

  // Mapping from [execution affinity, resource operands affinities, export] to
  // the executable op.
  using DispatchSiteInfo = std::tuple<IREE::Stream::AffinityAttr, ArrayAttr,
                                      IREE::Stream::ExecutableExportOp>;
  DenseMap<DispatchSiteInfo, IREE::Stream::ExecutableOp>
      dispatchSiteToExecutableOp;
  for (auto [exportOp, execAndResourceAffinities] : exportToDispatchSites) {
    auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
    // No need to duplicate the executable if all the uses have the same
    // affinities.
    // TODO(hanchung): Do not duplicate the executables if bindings are not
    // encoded. I.e., all the tensor types do not have encodings.
    if (execAndResourceAffinities.size() == 1) {
      auto [execAffinity, resourceAffinities] = execAndResourceAffinities[0];
      dispatchSiteToExecutableOp[DispatchSiteInfo(
          execAffinity, resourceAffinities, exportOp)] = executableOp;
      continue;
    }

    int64_t dupId = -1;
    for (auto [execAffinity, resourceAffinities] : execAndResourceAffinities) {
      rewriter.setInsertionPointAfter(executableOp);
      IREE::Stream::ExecutableOp dupOp = executableOp;
      if (dupId != -1) {
        auto symName = std::string(executableOp.getSymName());
        symName += "_dup" + std::to_string(dupId);
        dupOp = rewriter.cloneWithoutRegions(executableOp);
        rewriter.modifyOpInPlace(dupOp, [&] {
          dupOp.setSymName(symName);
          IRMapping mapping;
          executableOp.getRegion().cloneInto(&dupOp.getRegion(), mapping);
        });
      }
      dispatchSiteToExecutableOp[DispatchSiteInfo(
          execAffinity, resourceAffinities, exportOp)] = dupOp;
      dupId++;
    }
  }

  // 3. Update dispatch sites, i.e., point dispatch entry points to
  // corresponding cloned executables.
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> newEntryPoints;
    SmallVector<IREE::Stream::AffinityAttr> execAffinities;
    // Sanity checks. It should already meet the requirement because they are
    // checked in step 1. This can not be wrapped by an assertion because it
    // could be dropped by compiler.
    if (!affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                     execAffinities)) {
      return failure();
    }

    assert(execAffinities.size() == 1);
    SmallVector<Attribute> operandAttrs = resourceAffinities[dispatchOp];
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      auto info = DispatchSiteInfo(
          execAffinities[0], rewriter.getArrayAttr(operandAttrs), exportOp);
      assert(dispatchSiteToExecutableOp.count(info));

      auto executableOp = dispatchSiteToExecutableOp[info];
      auto newSym = SymbolRefAttr::get(executableOp->getAttrOfType<StringAttr>(
                                           SymbolTable::getSymbolAttrName()),
                                       entryPoint.getNestedReferences());
      newEntryPoints.push_back(newSym);
    });

    rewriter.modifyOpInPlace(dispatchOp, [&] {
      dispatchOp.setEntryPointsAttr(rewriter.getArrayAttr(newEntryPoints));
    });
  }

  // TODO(hanchung): Update encodings in executables.

  return success();
}

// TODO(hanchung): Add "cloneWithEncoding" method to RankedTensorType.
static RankedTensorType cloneWithEncoding(RankedTensorType type,
                                          Attribute encodingAttr) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encodingAttr);
}

static LogicalResult addLayoutsToTensorPhaseOps(
    ModuleOp moduleOp, FunctionOpInterface funcOp,
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  SmallVector<IREE::Stream::AffinityOpInterface> candidates;
  funcOp.walk([&](IREE::Stream::AffinityOpInterface affinityOp) {
    // Only need to update encoding types for ops that have TensorPhaseOp trait.
    if (!affinityOp->hasTrait<OpTrait::IREE::Stream::TensorPhaseOp>()) {
      return;
    }

    // Bail out if the operation does not have an affinity attribute.
    auto affinityAttr = affinityOp.getAffinityAttr();
    if (!affinityAttr) {
      return;
    }
    candidates.push_back(affinityOp);
  });

  if (candidates.empty()) {
    return success();
  }

  IRRewriter rewriter(funcOp.getContext());
  for (auto affinityOp : candidates) {
    auto affinityAttr = affinityOp.getAffinityAttr();
    SetVector<Attribute> layoutResolvers;
    if (failed(resolveLayoutAttr(affinityAttr, moduleOp, layoutResolvers))) {
      return affinityOp.emitError("failed on making layout resolvers");
    }

    // Returns an updated encoding attribute if an encoding attribute is present
    // in the type. Otherwise, returns std::nullopt.
    auto getEncodingWithNewLayouts =
        [=](Type type) -> std::optional<IREE::Encoding::EncodingAttr> {
      auto rankedTensorType = dyn_cast<RankedTensorType>(type);
      if (!rankedTensorType) {
        return std::nullopt;
      }
      auto encodingAttr = IREE::Encoding::getEncodingAttr(rankedTensorType);
      if (!encodingAttr) {
        return std::nullopt;
      }
      SmallVector<Attribute> layouts;
      for (auto attr : layoutResolvers) {
        auto encodingLayoutAttr =
            dyn_cast<IREE::Encoding::EncodingLayoutAttrInterface>(attr);
        if (!encodingLayoutAttr) {
          layouts.push_back(attr);
          continue;
        }
        layouts.push_back(encodingLayoutAttr.getLayout(rankedTensorType));
      }
      return encodingAttr.cloneWithLayouts(layouts);
    };

    // TODO(hanchung): Update other Stream operations.
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case<IREE::Stream::TensorSizeOfOp>([&](auto sizeOfOp) {
              auto encodingType =
                  dyn_cast<RankedTensorType>(sizeOfOp.getEncoding());
              if (!encodingType) {
                return success();
              }
              std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
                  getEncodingWithNewLayouts(encodingType);
              if (!encodingAttr) {
                return success();
              }
              rewriter.modifyOpInPlace(sizeOfOp, [&] {
                sizeOfOp.setEncoding(
                    cloneWithEncoding(encodingType, encodingAttr.value()));
              });
              return success();
            })
            .Default([](auto *op) { return failure(); });

    if (failed(result)) {
      return failure();
    }
  }
  return success();
}
} // namespace

struct SpecializeEncodingsPass
    : public impl::SpecializeEncodingsPassBase<SpecializeEncodingsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto usedDialects = gatherUsedDialectInterfaces<
        IREE::Stream::AffinityAnalysisDialectInterface>(moduleOp);
    if (usedDialects.size() != 1) {
      moduleOp.emitError("expected only one dialect implementing "
                         "AffinityAnalysisDialectInterface");
      return signalPassFailure();
    }

    SymbolTable symbolTable(moduleOp);
    llvm::MapVector<StringRef, IREE::Stream::ExecutableOp> executableOps;
    for (auto executableOp : moduleOp.getOps<IREE::Stream::ExecutableOp>()) {
      executableOps[executableOp.getName()] = executableOp;
    }

    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr =
        usedDialects[0]->makeLayoutAttrResolver(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(addLayoutsToTensorPhaseOps(moduleOp, funcOp,
                                            resolveLayoutAttr))) {
        funcOp.emitError(
            "failed on adding layouts to Stream::TensorPhaseOp with encodings");
        return signalPassFailure();
      }

      if (failed(duplicateExecutablesPerAffinityVariant(
              moduleOp, symbolTable, funcOp, resolveLayoutAttr))) {
        funcOp.emitError("failed on executable duplication");
        return signalPassFailure();
      }
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
