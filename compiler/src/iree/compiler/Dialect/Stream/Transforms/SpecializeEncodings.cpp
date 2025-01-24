// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTraits.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
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

// Returns an updated encoding attribute if the type is a RankedTensorType
// and an EncodingAttr is present. Otherwise, returns std::nullopt. The
// method uses the EncodingLayoutAttrInterface from the EncodingAttr to
// resolve the layouts of the given `type`; returns the new encodings with
// the resolved layouts.
static std::optional<IREE::Encoding::EncodingAttr>
getEncodingWithNewLayouts(Type type,
                          const SetVector<Attribute> &layoutResolvers) {
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

/// Update the bindings of function argumentswith encoding layouts. It only
/// updates the uses when the argument type is stream.binding_type. The bindings
/// are only used by binding subspan ops that return whatever types. Today they
/// are mostly flow tensor type. If the type implements
/// IREE::Encoding::EncodingTypeInterface type interface, the method uses the
/// interface methods to compute the type that has updated encodings (i.e.,
/// encodings with layouts) and updates the type.
static LogicalResult
updateBindingEncodings(FunctionOpInterface funcOp,
                       ArrayRef<Attribute> bindingLayoutAttrs) {
  int idx = 0;
  Region &region = funcOp.getFunctionBody();
  for (auto arg : region.getArguments()) {
    if (!isa<IREE::Stream::BindingType>(arg.getType())) {
      continue;
    }
    auto updatedEncoding =
        dyn_cast<IREE::Encoding::EncodingAttr>(bindingLayoutAttrs[idx++]);
    if (!updatedEncoding) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skip, The binding layout attribute is not EncodingAttr, "
                    "which means that the type does not have encodings.\n");
      continue;
    }
    for (auto user : arg.getUsers()) {
      auto subspanOp = dyn_cast<IREE::Stream::BindingSubspanOp>(user);
      if (!subspanOp) {
        return failure();
      }

      auto encodingTypeInterface =
          dyn_cast<IREE::Encoding::EncodingTypeInterface>(subspanOp.getType());
      if (!encodingTypeInterface) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Can not update the binding type because the type does "
                      "not implement EncodingTypeInterface.\n");
        return failure();
      }
      subspanOp.getResult().setType(
          encodingTypeInterface.updateEncoding(updatedEncoding));
    }
  }
  return success();
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

/// Returns the encoding layout of each binding. They are resolved by the
/// `resolveLayoutAttr` and the corresponding affinity. An empty array attribute
/// indicates that the operand resource does not have an encoding in the tensor
/// type.
static FailureOr<SmallVector<Attribute>> getEncodingLayoutForBindings(
    ModuleOp moduleOp, FunctionOpInterface funcOp,
    ArrayRef<Attribute> operandAffinities,
    IREE::Stream::AffinityAttr resultAffinity,
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  MLIRContext *ctx = funcOp.getContext();
  Region &region = funcOp.getFunctionBody();

  // The size of function arguments could be greater than the number of operand
  // affinities because the function also captures output resources in the
  // arguments.
  SmallVector<ArrayAttr> argsAffinities = llvm::map_to_vector(
      operandAffinities, [](Attribute attr) { return cast<ArrayAttr>(attr); });
  auto resAffinityAttr = ArrayAttr::get(ctx, {cast<Attribute>(resultAffinity)});
  argsAffinities.resize(region.getNumArguments(), resAffinityAttr);

  SmallVector<Attribute> result;
  auto emptyArrayAttr = ArrayAttr::get(ctx, {});
  int idx = 0;
  for (auto arg : region.getArguments()) {
    if (!isa<IREE::Stream::BindingType>(arg.getType())) {
      continue;
    }
    ArrayRef<Attribute> affinities = argsAffinities[idx++].getValue();
    assert(affinities.size() == 1);

    SetVector<Attribute> layoutResolvers;
    if (failed(
            resolveLayoutAttr(cast<IREE::Stream::AffinityAttr>(affinities[0]),
                              moduleOp, layoutResolvers))) {
      return failure();
    }

    for (auto user : arg.getUsers()) {
      auto subspanOp = dyn_cast<IREE::Stream::BindingSubspanOp>(user);
      if (!subspanOp) {
        return failure();
      }
      auto resultType = llvm::dyn_cast<IREE::Flow::DispatchTensorType>(
          subspanOp.getResult().getType());
      if (!resultType) {
        return failure();
      }
      std::optional<IREE::Encoding::EncodingAttr> newEncodingType =
          getEncodingWithNewLayouts(resultType.asRankedTensorType(),
                                    layoutResolvers);
      if (!newEncodingType) {
        result.push_back(emptyArrayAttr);
      } else {
        result.push_back(newEncodingType.value());
      }
    }
  }

  return result;
}

/// Returns the resolved layouts of the bindings for the `dispatchOp`. It
/// gathers the operand resource affinities and the execution affinity, looks at
/// the bindings in executable functions, and resolves the layouts for the
/// tensor types. Returns a failure if it can not figure the layouts.
static FailureOr<SmallVector<Attribute>> getBindingLayoutsForDispatchOp(
    ModuleOp moduleOp, SymbolTable symbolTable,
    IREE::Stream::AffinityAnalysis &affinityAnalysis,
    IREE::Stream::AsyncDispatchOp dispatchOp,
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  SmallVector<IREE::Stream::AffinityAttr> execAffinities;
  if (!affinityAnalysis.tryLookupExecutionAffinity(dispatchOp,
                                                   execAffinities)) {
    return failure();
  }
  if (execAffinities.size() != 1) {
    return failure();
  }

  // We can gather the binding layouts from any of entry_point, because they
  // should all have the same binding types.
  bool done = false;
  FailureOr<SmallVector<Attribute>> result;
  dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
    if (done) {
      return;
    }
    auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
        symbolTable.lookupSymbolIn(moduleOp, entryPoint));
    auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
    SmallVector<Attribute> operandAffinityAttrs =
        getResourceOperandsAffinities(affinityAnalysis, dispatchOp);
    auto funcOp = cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
        executableOp.getInnerModule(), exportOp.getSymName()));
    result =
        getEncodingLayoutForBindings(moduleOp, funcOp, operandAffinityAttrs,
                                     execAffinities[0], resolveLayoutAttr);
    done = true;
  });

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

  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return moduleOp.emitError("failed on running affinity analysis");
  }
  SmallVector<IREE::Stream::AsyncDispatchOp> candidates;
  funcOp.walk(
      [&](IREE::Stream::AsyncDispatchOp op) { candidates.push_back(op); });

  //===--------------------------------------------------------------------===//
  // Gather per-export [binding layouts] map. A function in an executable can be
  // run with different affinities. The function arguments, where the types are
  // `!stream.binding`, are consumed by `stream.binding.subspan` ops, and the op
  // returns a tensor type. The binding layouts indicate the resolved layouts
  // for those tensor types. The map records the mapping between an export op
  // and the possible binding layouts.
  //===--------------------------------------------------------------------===//
  DenseMap<IREE::Stream::ExecutableExportOp, SetVector<ArrayAttr>>
      bindingLayoutSetPerExportOp;

  // Records the binding layouts for a dispatch op.
  llvm::MapVector<IREE::Stream::AsyncDispatchOp, SmallVector<Attribute>>
      dispatchOpBindingLayouts;
  for (auto dispatchOp : candidates) {
    FailureOr<SmallVector<Attribute>> bindingLayoutAttrs =
        getBindingLayoutsForDispatchOp(moduleOp, symbolTable, affinityAnalysis,
                                       dispatchOp, resolveLayoutAttr);
    if (failed(bindingLayoutAttrs)) {
      return failure();
    }
    dispatchOpBindingLayouts[dispatchOp] = bindingLayoutAttrs.value();
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      bindingLayoutSetPerExportOp[exportOp].insert(
          rewriter.getArrayAttr(bindingLayoutAttrs.value()));
    });
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Dump of bindingLayoutSetPerExportOp\n";
    for (auto [exportOp, layoutSet] : bindingLayoutSetPerExportOp) {
      llvm::dbgs() << "  ExportOp: " << exportOp.getSymName() << "\n";
      for (auto [idx, attr] : llvm::enumerate(layoutSet)) {
        llvm::dbgs() << "    binding_layouts #" << idx << ": " << attr << "\n ";
      }
    }
  });

  //===--------------------------------------------------------------------===//
  // Duplicate executables for each unqiue binding layouts.
  //===--------------------------------------------------------------------===//
  // Mapping from [export op, binding layouts] to the executable op. So we can
  // use it to update dispatch sites later on.
  using ExportAndBindingLayouts =
      std::pair<IREE::Stream::ExecutableExportOp, ArrayAttr>;
  DenseMap<ExportAndBindingLayouts, IREE::Stream::ExecutableOp>
      dispatchSiteToExecutableOp;
  for (auto [exportOp, layoutSet] : bindingLayoutSetPerExportOp) {
    auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
    // No need to duplicate the executable if all the uses have the same
    // incoming layouts and produce the result in the same layout.
    if (layoutSet.size() == 1) {
      dispatchSiteToExecutableOp[ExportAndBindingLayouts(
          exportOp, layoutSet[0])] = executableOp;
      continue;
    }

    int64_t dupId = -1;
    for (auto bindingLayoutAttrs : layoutSet) {
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
      dispatchSiteToExecutableOp[ExportAndBindingLayouts(
          exportOp, bindingLayoutAttrs)] = dupOp;
      dupId++;
    }
  }

  //===--------------------------------------------------------------------===//
  // Update dispatch sites, i.e., point dispatch entry points to corresponding
  // duplicated executables.
  //===--------------------------------------------------------------------===//
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> newEntryPoints;
    SmallVector<Attribute> bindingLayoutAttrs =
        dispatchOpBindingLayouts[dispatchOp];
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      auto info = ExportAndBindingLayouts(
          exportOp, rewriter.getArrayAttr(bindingLayoutAttrs));
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

  //===--------------------------------------------------------------------===//
  // Update encoding types for bindings in executables.
  //===--------------------------------------------------------------------===//
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> bindingLayoutAttrs =
        dispatchOpBindingLayouts[dispatchOp];
    bool succeeded = true;
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      if (!succeeded) {
        return;
      }
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      auto executableOp =
          exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
      LLVM_DEBUG(llvm::dbgs() << "Update ExecutableOp: "
                              << executableOp.getSymName() << "\n");
      LLVM_DEBUG({
        llvm::dbgs() << "  binding layouts: [";
        llvm::interleaveComma(bindingLayoutAttrs, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });
      auto func = cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
          executableOp.getInnerModule(), exportOp.getSymName()));
      if (failed(updateBindingEncodings(func, bindingLayoutAttrs))) {
        succeeded = false;
      }
    });

    if (!succeeded) {
      return failure();
    }
  }

  return success();
}

// TODO(hanchung): Add "cloneWithEncoding" method to RankedTensorType.
static RankedTensorType cloneWithEncoding(RankedTensorType type,
                                          Attribute encodingAttr) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encodingAttr);
}

/// Updates the encoding of `sizeOfOp` with resolved layouts.
static LogicalResult
updateTensorSizeOfOp(RewriterBase &rewriter,
                     IREE::Stream::TensorSizeOfOp sizeOfOp,
                     const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(sizeOfOp.getEncoding());
  std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
      getEncodingWithNewLayouts(encodingType, layoutResolvers);
  if (!encodingAttr) {
    return success();
  }
  rewriter.modifyOpInPlace(sizeOfOp, [&] {
    sizeOfOp.setEncoding(cloneWithEncoding(encodingType, encodingAttr.value()));
  });
  return success();
}

/// Returns failure if `op` has encoding. The EncodingAttr has padding
/// semantic, a constant op with such  encoding can not be resolved at this
/// moment.
static LogicalResult
updateTensorConstantOp(RewriterBase &rewriter,
                       IREE::Stream::TensorConstantOp op,
                       const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
  if (!encodingType) {
    return success();
  }
  if (IREE::Encoding::getEncodingAttr(encodingType)) {
    return failure();
  }
  return success();
}

/// Updates the result_encoding for `op`. The op have to define a
/// `result_encoding` parameter.
template <typename OpTy>
static LogicalResult
updateResultEncoding(RewriterBase &rewriter, OpTy op,
                     const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
  std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
      getEncodingWithNewLayouts(encodingType, layoutResolvers);
  if (!encodingAttr) {
    return success();
  }
  rewriter.modifyOpInPlace(op, [&] {
    op.setResultEncoding(cloneWithEncoding(encodingType, encodingAttr.value()));
  });
  return success();
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

    // TODO(hanchung): Update other Stream operations.
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case<IREE::Stream::TensorSizeOfOp>([&](auto op) {
              return updateTensorSizeOfOp(rewriter, op, layoutResolvers);
            })
            .Case<IREE::Stream::TensorEmptyOp, IREE::Stream::TensorSplatOp>(
                [&](auto op) {
                  return updateResultEncoding(rewriter, op, layoutResolvers);
                })
            .Case<IREE::Stream::TensorConstantOp>([&](auto op) {
              return updateTensorConstantOp(rewriter, op, layoutResolvers);
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
