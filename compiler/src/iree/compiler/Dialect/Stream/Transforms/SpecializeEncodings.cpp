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
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
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

/// Updates the bindings of function arguments with encoding layouts. It only
/// updates the uses when the argument type is stream.binding_type. The bindings
/// are only used by binding subspan ops that return whatever types. Today they
/// are mostly flow tensor type. If the type implements
/// IREE::Encoding::EncodingTypeInterface type interface, the method uses the
/// interface methods to compute the type that has updated encodings (i.e.,
/// encodings with layouts) and updates the type.
static LogicalResult
updateBindingEncodings(FunctionOpInterface funcOp,
                       ArrayRef<Attribute> bindingLayoutTypeAttrs) {
  Region &region = funcOp.getFunctionBody();
  for (auto [arg, newTypeAttr] :
       llvm::zip_equal(region.getArguments(), bindingLayoutTypeAttrs)) {
    if (!isa<IREE::Stream::BindingType>(arg.getType())) {
      continue;
    }
    auto newType =
        dyn_cast<RankedTensorType>(cast<TypeAttr>(newTypeAttr).getValue());
    if (!newType) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Skip, the new type is not RankedTensorType.\n");
      continue;
    }
    auto encodingAttr = IREE::Encoding::getEncodingAttr(newType);
    if (!encodingAttr) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip, the binding layout attribute is not EncodingAttr, "
             "which means that the type does not have a valid encoding.\n");
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
          encodingTypeInterface.updateEncoding(encodingAttr));
    }
  }
  return success();
}

/// Duplicates stream.executables based on the operand encodings and result
/// encodings of stream.tensor.dispatch ops. Some executables can be launched by
/// different devices. It can produce wrong codegen artifacts when bindings
/// types are encoded (i.e., the tensor type has an encoding attribute). Because
/// they can result in different layouts, especially when multi-device is
/// involved. E.g., say that device_a and device_b interpret a tensor type with
/// encodings in different layouts, and there is an executable that can be
/// launch with resources from either device_a or device_b. It is confusing what
/// the input layouts for the executable because there are two possibilities. In
/// this case, we have to duplicate the executable with updated encoding, and
/// modify the dispatch to launch proper executable based on resolved encoding
/// layouts.
static LogicalResult duplicateExecutablesPerLayoutVariant(
    ModuleOp moduleOp, SymbolTable symbolTable, FunctionOpInterface funcOp,
    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  MLIRContext *ctx = moduleOp.getContext();
  IRRewriter rewriter(ctx);

  SmallVector<IREE::Stream::TensorDispatchOp> candidates;
  funcOp.walk(
      [&](IREE::Stream::TensorDispatchOp op) { candidates.push_back(op); });

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
  llvm::MapVector<IREE::Stream::TensorDispatchOp, SmallVector<Attribute>>
      dispatchOpBindingLayouts;
  for (auto dispatchOp : candidates) {
    SmallVector<Attribute> bindingLayoutAttrs(
        dispatchOp.getOperandEncodings().getValue());
    llvm::append_range(bindingLayoutAttrs,
                       dispatchOp.getResultEncodings().getValue());
    dispatchOpBindingLayouts[dispatchOp] = bindingLayoutAttrs;
    dispatchOp.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      auto exportOp = cast<IREE::Stream::ExecutableExportOp>(
          symbolTable.lookupSymbolIn(moduleOp, entryPoint));
      bindingLayoutSetPerExportOp[exportOp].insert(
          rewriter.getArrayAttr(bindingLayoutAttrs));
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
    int64_t dupId = -1;
    auto executableOp = exportOp->getParentOfType<IREE::Stream::ExecutableOp>();
    for (ArrayAttr bindingLayoutTypeAttrs : layoutSet) {
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

      // Update the binding encodings within the cloned executable op.
      auto innerFuncOp =
          cast<mlir::FunctionOpInterface>(symbolTable.lookupSymbolIn(
              dupOp.getInnerModule(), exportOp.getSymName()));
      if (failed(updateBindingEncodings(innerFuncOp,
                                        bindingLayoutTypeAttrs.getValue()))) {
        return failure();
      }
      dispatchSiteToExecutableOp[ExportAndBindingLayouts(
          exportOp, bindingLayoutTypeAttrs)] = dupOp;
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
  return success();
}

// TODO(hanchung): Add "cloneWithEncoding" method to RankedTensorType.
static RankedTensorType cloneWithEncoding(RankedTensorType type,
                                          Attribute encodingAttr) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encodingAttr);
}

/// Updates the operand encondings and result encodings for the `dispatchOp`
/// with resolved layouts.
static LogicalResult
updateTensorDispatchOp(RewriterBase &rewriter, ModuleOp moduleOp,
                       IREE::Stream::AffinityAnalysis &affinityAnalysis,
                       IREE::Stream::TensorDispatchOp dispatchOp,
                       const SetVector<Attribute> &resLayoutResolvers,
                       IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr) {
  SmallVector<Type> newOperandEncodings;
  for (auto [operand, typeAttr] :
       llvm::zip_equal(dispatchOp.getMixedOperands(),
                       dispatchOp.getOperandEncodings().getValue())) {
    auto type = cast<TypeAttr>(typeAttr).getValue();
    // Skip if the operand type is not AffinityType.
    if (!isa<IREE::Stream::AffinityTypeInterface>(type)) {
      newOperandEncodings.push_back(type);
      continue;
    }
    SmallVector<IREE::Stream::AffinityAttr> affinityAttrs;
    if (!affinityAnalysis.tryLookupResourceAffinity(operand, affinityAttrs)) {
      return failure();
    }
    if (affinityAttrs.size() != 1) {
      return failure();
    }
    SetVector<Attribute> layoutResolvers;
    if (failed(
            resolveLayoutAttr(affinityAttrs[0], moduleOp, layoutResolvers))) {
      return dispatchOp.emitError("failed on making layout resolvers");
    }

    std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
        getEncodingWithNewLayouts(type, layoutResolvers);
    if (!encodingAttr) {
      newOperandEncodings.push_back(type);
      continue;
    }
    newOperandEncodings.push_back(
        cloneWithEncoding(cast<RankedTensorType>(type), encodingAttr.value()));
  }
  dispatchOp.setOperandEncodingsAttr(
      rewriter.getTypeArrayAttr(newOperandEncodings));

  SmallVector<Type> newResultEncodings;
  for (auto typeAttr : dispatchOp.getResultEncodings().getValue()) {
    auto type = cast<TypeAttr>(typeAttr).getValue();
    // Skip if the result type is not AffinityType.
    if (!isa<IREE::Stream::AffinityTypeInterface>(type)) {
      newResultEncodings.push_back(type);
      continue;
    }

    std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
        getEncodingWithNewLayouts(type, resLayoutResolvers);
    if (!encodingAttr) {
      newResultEncodings.push_back(type);
      continue;
    }
    newResultEncodings.push_back(
        cloneWithEncoding(cast<RankedTensorType>(type), encodingAttr.value()));
  }
  dispatchOp.setResultEncodingsAttr(
      rewriter.getTypeArrayAttr(newResultEncodings));

  return success();
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

/// Updates the target encoding of `op` with resolved layouts.
static LogicalResult
updateTensorFillOp(RewriterBase &rewriter, IREE::Stream::TensorFillOp op,
                   const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getTargetEncoding());
  std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
      getEncodingWithNewLayouts(encodingType, layoutResolvers);
  if (!encodingAttr) {
    return success();
  }
  rewriter.modifyOpInPlace(op, [&] {
    op.setTargetEncoding(cloneWithEncoding(encodingType, encodingAttr.value()));
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

/// Returns a failure if there are encodings in target encoding type or update
/// encoding type.
static LogicalResult updateTensorUpdateOp(RewriterBase &rewriter,
                                          IREE::Stream::TensorUpdateOp op) {
  auto targetEncodingType = dyn_cast<RankedTensorType>(op.getTargetEncoding());
  if (targetEncodingType && targetEncodingType.getEncoding()) {
    return failure();
  }
  auto updateEncodingType = dyn_cast<RankedTensorType>(op.getUpdateEncoding());
  if (updateEncodingType && updateEncodingType.getEncoding()) {
    return failure();
  }
  return success();
}

/// Returns a failure if there are encodings in source encoding type or result
/// encoding type.
static LogicalResult updateTensorCloneOp(RewriterBase &rewriter,
                                         IREE::Stream::TensorCloneOp op) {
  auto sourceEncodingType = dyn_cast<RankedTensorType>(op.getSourceEncoding());
  if (sourceEncodingType && sourceEncodingType.getEncoding()) {
    return failure();
  }
  auto resultEncodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
  if (resultEncodingType && resultEncodingType.getEncoding()) {
    return failure();
  }
  return success();
}

/// Returns a failure if there are encodings in source encoding type or result
/// encoding type.
static LogicalResult updateTensorSliceOp(RewriterBase &rewriter,
                                         IREE::Stream::TensorSliceOp op) {
  auto sourceEncodingType = dyn_cast<RankedTensorType>(op.getSourceEncoding());
  if (sourceEncodingType && sourceEncodingType.getEncoding()) {
    return failure();
  }
  auto resultEncodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
  if (resultEncodingType && resultEncodingType.getEncoding()) {
    return failure();
  }
  return success();
}

/// Updates the source_encoding for `op`. The op has to define a
/// `source_encoding` parameter.
template <typename OpTy>
static LogicalResult
updateSourceEncoding(RewriterBase &rewriter, OpTy op,
                     const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getSourceEncoding());
  std::optional<IREE::Encoding::EncodingAttr> encodingAttr =
      getEncodingWithNewLayouts(encodingType, layoutResolvers);
  if (!encodingAttr) {
    return success();
  }
  rewriter.modifyOpInPlace(op, [&] {
    op.setSourceEncoding(cloneWithEncoding(encodingType, encodingAttr.value()));
  });
  return success();
}

/// Updates the result_encoding for `op`. The op has to define a
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

/// Adds the resolved layouts to all tensor types on stream tensor ops, if
/// encodings are present. Most of stream tensor ops implement
/// AffinityOpInterface, where a stream affinity indicates the kind of
/// enviroment the ops are expected run in. When an encoding is present in the
/// tensor type, the method resolves the layouts, strips outdated information,
/// and adds the resolved layouts to the encodings. The updated encodings should
/// have enough information for other lowering transformations.
/// TODO(hanchung): Add support for stream.tensor.load ops and
/// stream.tensor.store ops. They are not affinity ops, so additional analysis
/// will be needed in the work.
static LogicalResult addLayoutsToTensorPhaseOps(
    ModuleOp moduleOp, IREE::Stream::AffinityAnalysis &affinityAnalysis,
    FunctionOpInterface funcOp,
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

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case<IREE::Stream::TensorDispatchOp>([&](auto op) {
              return updateTensorDispatchOp(rewriter, moduleOp,
                                            affinityAnalysis, op,
                                            layoutResolvers, resolveLayoutAttr);
            })
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
            .Case<IREE::Stream::TensorFillOp>([&](auto op) {
              return updateTensorFillOp(rewriter, op, layoutResolvers);
            })
            .Case<IREE::Stream::TensorCloneOp>(
                [&](auto op) { return updateTensorCloneOp(rewriter, op); })
            .Case<IREE::Stream::TensorSliceOp>(
                [&](auto op) { return updateTensorSliceOp(rewriter, op); })
            .Case<IREE::Stream::TensorUpdateOp>(
                [&](auto op) { return updateTensorUpdateOp(rewriter, op); })
            .Default([](Operation *op) {
              return op->emitOpError("Unhandled stream op");
            });

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

    IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
    if (failed(affinityAnalysis.run())) {
      moduleOp.emitError("failed on running affinity analysis");
      return signalPassFailure();
    }

    IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr =
        usedDialects[0]->makeLayoutAttrResolver(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(addLayoutsToTensorPhaseOps(moduleOp, affinityAnalysis, funcOp,
                                            resolveLayoutAttr))) {
        funcOp.emitError(
            "failed on adding layouts to Stream::TensorPhaseOp with encodings");
        return signalPassFailure();
      }
      if (failed(duplicateExecutablesPerLayoutVariant(
              moduleOp, symbolTable, funcOp, resolveLayoutAttr))) {
        funcOp.emitError("failed on executable duplication");
        return signalPassFailure();
      }
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
