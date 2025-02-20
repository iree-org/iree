// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
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
#include "mlir/IR/BuiltinOps.h"
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

} // namespace

// TODO(hanchung): Add "cloneWithEncoding" method to RankedTensorType.
static RankedTensorType cloneWithEncoding(RankedTensorType type,
                                          Attribute encodingAttr) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encodingAttr);
}

/// Returns the type with updated encoding, if any. Returns the original type if
/// the type is not a RankedTensorType. If it is a RankedTensorType with an
/// unknown encoding, returns the type without the encoding. The method uses
/// `layoutResolvers` to resolve the layouts of the given `type`; returns the
/// new encoding with the resolved layouts.
///
/// There are requirements to get the resolved layouts. Otherwise, the encodings
/// are dropped unconditionally.
///   - All attributes in the `layoutResolvers` must implement
///     EncodingLayoutResolverAttrInterface. Otherwise, there is no way to query
///     layouts.
///   - The encoding on the type must implement
///     SerializableEncodingAttrInterface. Otherwise, there is no way to update
///     encodings.
static Type getTypeWithResolvedEncodingLayouts(
    Type type, const SetVector<Attribute> &layoutResolvers) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (!rankedTensorType) {
    return type;
  }
  auto encodingAttr =
      IREE::Encoding::getSerializableEncodingAttrInterface(rankedTensorType);
  if (!encodingAttr) {
    return IREE::Encoding::dropEncoding(rankedTensorType);
  }
  if (encodingAttr.isSerialized()) {
    return type;
  }
  if (!llvm::all_of(
          layoutResolvers,
          llvm::IsaPred<IREE::Encoding::EncodingLayoutResolverAttrInterface>)) {
    return IREE::Encoding::dropEncoding(rankedTensorType);
  }
  SmallVector<Attribute> layouts;
  for (auto attr : layoutResolvers) {
    auto encodingLayoutAttr =
        cast<IREE::Encoding::EncodingLayoutResolverAttrInterface>(attr);
    Attribute layout = encodingLayoutAttr.getLayout(rankedTensorType);
    if (!layout) {
      // Drop the encoding if the layout is not resolved.
      return IREE::Encoding::dropEncoding(rankedTensorType);
    }
    layouts.push_back(layout);
  }
  Attribute newEncoding = encodingAttr.cloneWithLayouts(layouts);
  assert(isa<IREE::Encoding::SerializableEncodingAttrInterface>(newEncoding));
  return cloneWithEncoding(rankedTensorType, newEncoding);
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
    auto encodingAttr =
        IREE::Encoding::getSerializableEncodingAttrInterface(newType);
    if (!encodingAttr) {
      LLVM_DEBUG(
          llvm::dbgs()
          << "Skip, the binding layout attribute is not "
             "SerializableEncodingAttrInterface, which means that the type "
             "does not have a valid encoding.\n");
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

/// Returns the operands encodings and result encodings from the `dispatchOp` in
/// |operands| + |results| order, i.e., it returns the stripped concatenated
/// operand encodings and result encodings. If a result is tied to an operand,
/// the result encoding is skipped. Because it shares the same binding with the
/// tied operands.
///
/// Example 1:
///
///   %0 = stream.tensor.dispatch ...(%arg0, %c4)
///     : (tensor<4x?xf32, #encoding> in !resource, index)
///     -> tensor<4x?xf32, #encoding> in !resource
///
/// The above dispatch op does not have tied operands. Thus, it returns
///   |#resolved_encoding, whatever_without_encoding, #resolved_encoding|
///
/// Example 2:
///
///   %0 = stream.tensor.dispatch ...(%arg0, %c4) : tensor<4x?xf32, #encoding>
///     -> tensor<4x?xf32, #encoding> in %arg0
///
/// The above dispatch op ties the result to the first operand. Thus, the result
/// encoding is stripped. It returns
///   |#resolved_encoding, whatever_without_encoding|
static SmallVector<Attribute>
getBindingLayoutAttrs(IREE::Stream::TensorDispatchOp dispatchOp) {
  SmallVector<int64_t> tiedOperands(dispatchOp.getNumResults(),
                                    IREE::Util::TiedOpInterface::kUntiedIndex);
  if (std::optional<ArrayAttr> tiedOperandsAttr =
          dispatchOp.getTiedOperands()) {
    tiedOperands =
        llvm::map_to_vector(tiedOperandsAttr.value(), [](Attribute intAttr) {
          return llvm::cast<IntegerAttr>(intAttr).getInt();
        });
  }

  SmallVector<Attribute> result(dispatchOp.getOperandEncodings().getValue());
  for (auto [resultEncoding, tiedOperand] : llvm::zip_equal(
           dispatchOp.getResultEncodings().getValue(), tiedOperands)) {
    if (tiedOperand != IREE::Util::TiedOpInterface::kUntiedIndex) {
      continue;
    }
    result.push_back(resultEncoding);
  }

  return result;
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
static LogicalResult
duplicateExecutablesPerLayoutVariant(ModuleOp moduleOp, SymbolTable symbolTable,
                                     FunctionOpInterface funcOp) {
  MLIRContext *ctx = moduleOp.getContext();
  IRRewriter rewriter(ctx);

  SmallVector<IREE::Stream::TensorDispatchOp> candidates;
  funcOp.walk([&](IREE::Stream::TensorDispatchOp op) {
    // Filter out the cases that are not from the normal pipeline. E.g., custom
    // dispatch could embed hal.executables.
    bool recognizedInput = true;
    op.forEachEntryPointAttr([&](SymbolRefAttr entryPoint) {
      if (!isa<IREE::Stream::ExecutableExportOp>(
              symbolTable.lookupSymbolIn(moduleOp, entryPoint))) {
        recognizedInput = false;
      }
    });
    if (recognizedInput) {
      candidates.push_back(op);
    }
  });

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
    SmallVector<Attribute> bindingLayoutAttrs =
        getBindingLayoutAttrs(dispatchOp);
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

/// Returns all the stream tensor ops that implement AffinityOpInterface, where
/// a stream affinity indicates the kind of enviroment the ops are expected run
/// in.
static SmallVector<IREE::Stream::AffinityOpInterface>
collectStreamTensorOps(FunctionOpInterface funcOp) {
  SmallVector<IREE::Stream::AffinityOpInterface> result;
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
    result.push_back(affinityOp);
  });
  return result;
}

namespace {

// Adds the resolved layouts to all tensor types on stream tensor ops, if
// encodings are present. Most of stream tensor ops implement
// AffinityOpInterface, where a stream affinity indicates the kind of
// enviroment the ops are expected run in. When an encoding is present in the
// tensor type, the method resolves the layouts, strips outdated information,
// and adds the resolved layouts to the encodings. The updated encodings should
// have enough information for other lowering transformations.
// TODO(hanchung): Add support for stream.tensor.load ops and
// stream.tensor.store ops. They are not affinity ops, so additional analysis
// will be needed in the work.
class StreamTensorOpUpdater {
public:
  explicit StreamTensorOpUpdater(ModuleOp moduleOp) : moduleOp(moduleOp) {}
  ~StreamTensorOpUpdater() {}

  // Collects the stream tensor op candidates, and prepares all the needed
  // information for the update. This must be called once before calling `run`.
  // Note that all the ops are unmodified after the execution.
  LogicalResult init();

  // Adds the resolved layouts to all tensor types of `streamOps`, if encodings
  // are present.
  LogicalResult run();

private:
  // Appends the query from the `affinityOp` to `queries`. Note that most of
  // operations only care the execution affinity. There are outliers (e.g.,
  // tensor dispatch op, etc.) that need to resolve affinities for
  // operand resources.
  LogicalResult addQuery(IREE::Stream::AffinityAnalysis &affinityAnalysis,
                         IREE::Stream::AffinityOpInterface affinityOp);

  // The list of the queries that can be used for batch affinity queries. The
  // analysis could be very expensive because it could apply the whole program
  // data flow analysis.
  SmallVector<IREE::Stream::AffinityAndOpPair> queries;

  // The layout resolvers for each query.
  llvm::DenseMap<IREE::Stream::AffinityAndOpPair, SetVector<Attribute>>
      cachedLayoutAttrs;

  // Input moduleOp. The op is not expected to be updated during the query.
  // Because data flow analaysis can be involved. Modifying the IR invalidates
  // the state and may lead to crashes as pointer references into the IR
  // structure are retained.
  ModuleOp moduleOp;

  // The ops that need to be updated.
  SmallVector<IREE::Stream::AffinityOpInterface> streamOps;

  // The layout resolver function, which is used to resolve layouts for
  // encodings. See StreamInterfaces.h for more details.
  IREE::Stream::ResolveLayoutAttrFn resolveLayoutAttr;
};

} // namespace

LogicalResult StreamTensorOpUpdater::init() {
  auto usedDialects = gatherUsedDialectInterfaces<
      IREE::Stream::AffinityAnalysisDialectInterface>(moduleOp);
  if (usedDialects.size() != 1) {
    return moduleOp.emitError("expected only one dialect implementing "
                              "AffinityAnalysisDialectInterface");
  }
  resolveLayoutAttr = usedDialects[0]->makeLayoutAttrResolver(moduleOp);

  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    streamOps.append(collectStreamTensorOps(funcOp));
  }

  return success();
}

LogicalResult StreamTensorOpUpdater::addQuery(
    IREE::Stream::AffinityAnalysis &affinityAnalysis,
    IREE::Stream::AffinityOpInterface affinityOp) {
  queries.emplace_back(affinityOp.getAffinityAttr(), affinityOp);

  if (auto dispatchOp =
          dyn_cast<IREE::Stream::TensorDispatchOp>(affinityOp.getOperation())) {
    for (auto [operand, typeAttr] :
         llvm::zip_equal(dispatchOp.getMixedOperands(),
                         dispatchOp.getOperandEncodings().getValue())) {
      auto type = cast<TypeAttr>(typeAttr).getValue();
      // Skip if the operand type is not AffinityType.
      if (!isa<IREE::Stream::AffinityTypeInterface>(type)) {
        continue;
      }
      SmallVector<IREE::Stream::AffinityAttr> affinityAttrs;
      if (!affinityAnalysis.tryLookupResourceAffinity(operand, affinityAttrs)) {
        return dispatchOp.emitError(
                   "failed to determine resource affinity for operand ")
               << operand;
      }
      for (auto affinity : affinityAttrs) {
        queries.emplace_back(affinity, affinityOp);
      }
    }
  }

  return success();
}

/// Updates the operand encodings and result encodings for the `dispatchOp`
/// with resolved layouts.
static LogicalResult updateTensorDispatchOp(
    RewriterBase &rewriter, ModuleOp moduleOp,
    IREE::Stream::AffinityAnalysis &affinityAnalysis,
    IREE::Stream::TensorDispatchOp dispatchOp,
    const SetVector<Attribute> &resLayoutResolvers,
    llvm::DenseMap<IREE::Stream::AffinityAndOpPair, SetVector<Attribute>>
        &cachedLayoutAttrs) {
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

    IREE::Stream::AffinityAndOpPair key(affinityAttrs[0], dispatchOp);
    assert(cachedLayoutAttrs.contains(key) &&
           "the (affinity, dispatchOp) query is invalid");
    const SetVector<Attribute> &layoutResolvers = cachedLayoutAttrs[key];

    Type newEncodingType =
        getTypeWithResolvedEncodingLayouts(type, layoutResolvers);
    newOperandEncodings.push_back(newEncodingType);
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
    Type newEncodingType =
        getTypeWithResolvedEncodingLayouts(type, resLayoutResolvers);
    newResultEncodings.push_back(newEncodingType);
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
  Type newEncodingType =
      getTypeWithResolvedEncodingLayouts(encodingType, layoutResolvers);
  rewriter.modifyOpInPlace(sizeOfOp,
                           [&] { sizeOfOp.setEncoding(newEncodingType); });
  return success();
}

/// Updates the target encoding of `op` with resolved layouts.
static LogicalResult
updateTensorFillOp(RewriterBase &rewriter, IREE::Stream::TensorFillOp op,
                   const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getTargetEncoding());
  Type newEncodingType =
      getTypeWithResolvedEncodingLayouts(encodingType, layoutResolvers);
  rewriter.modifyOpInPlace(op, [&] { op.setTargetEncoding(newEncodingType); });
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
  if (encodingType.getEncoding()) {
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
  Type newEncodingType =
      getTypeWithResolvedEncodingLayouts(encodingType, layoutResolvers);
  rewriter.modifyOpInPlace(op, [&] { op.setSourceEncoding(newEncodingType); });
  return success();
}

/// Updates the result_encoding for `op`. The op has to define a
/// `result_encoding` parameter.
template <typename OpTy>
static LogicalResult
updateResultEncoding(RewriterBase &rewriter, OpTy op,
                     const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getResultEncoding());
  Type newEncodingType =
      getTypeWithResolvedEncodingLayouts(encodingType, layoutResolvers);
  rewriter.modifyOpInPlace(op, [&] { op.setResultEncoding(newEncodingType); });
  return success();
}

LogicalResult StreamTensorOpUpdater::run() {
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return moduleOp.emitError("failed on running affinity analysis");
  }

  for (auto op : streamOps) {
    if (failed(addQuery(affinityAnalysis, op))) {
      return failure();
    }
  }

  if (failed(resolveLayoutAttr(queries, cachedLayoutAttrs))) {
    return failure();
  }

  IRRewriter rewriter(moduleOp.getContext());
  for (auto affinityOp : streamOps) {
    const SetVector<Attribute> &layoutResolvers =
        cachedLayoutAttrs[IREE::Stream::AffinityAndOpPair(
            affinityOp.getAffinityAttr(), affinityOp)];

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case<IREE::Stream::TensorDispatchOp>([&](auto op) {
              return updateTensorDispatchOp(rewriter, moduleOp,
                                            affinityAnalysis, op,
                                            layoutResolvers, cachedLayoutAttrs);
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

namespace {
struct SpecializeEncodingsPass
    : public impl::SpecializeEncodingsPassBase<SpecializeEncodingsPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    StreamTensorOpUpdater streamTensorOpUpdater(moduleOp);
    if (failed(streamTensorOpUpdater.init())) {
      moduleOp.emitError("failed to initialize StreamTensorOpUpdater");
      return signalPassFailure();
    }
    if (failed(streamTensorOpUpdater.run())) {
      moduleOp.emitError(
          "failed to add layouts to Stream::TensorPhaseOp with encodings");
      return signalPassFailure();
    }

    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(duplicateExecutablesPerLayoutVariant(moduleOp, symbolTable,
                                                      funcOp))) {
        funcOp.emitError("failed on executable duplication");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
