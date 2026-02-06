// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Encoding/Utils/Utils.h"
#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTraits.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Utils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVectorExtras.h"
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

/// Returns true iff the type is a RankedTensorType and it has an encoding that
/// implements SerializableAttr.
static bool isRecognizedEncodingType(Type type) {
  auto rankedTensorType = dyn_cast<RankedTensorType>(type);
  if (!rankedTensorType) {
    return false;
  }
  Attribute encoding = rankedTensorType.getEncoding();
  if (!encoding) {
    return false;
  }
  return isa<IREE::Encoding::SerializableAttr>(encoding);
}

/// Returns the type with updated encoding, if any. Returns the original type if
/// the the encoding type is not recognized or it is already serialized. If it
/// fails to resolve the layout, returns nullptr.
/// The method uses `layoutResolvers` to resolve the layouts of the given
/// `type`; returns the new encoding with the resolved layouts.
///
/// There are requirements to get the resolved layouts. Otherwise, the encodings
/// are dropped.
///   - All attributes in the `layoutResolvers` must implement
///     LayoutResolverAttr. Otherwise, there is no way to query layouts.
///   - The encoding on the type must implement SerializableAttr. Otherwise,
///     there is no way to update encodings.
static Type getTypeWithResolvedEncodingLayouts(
    Type type, const SetVector<Attribute> &layoutResolvers) {
  if (!isRecognizedEncodingType(type)) {
    return type;
  }
  auto rankedTensorType = dyn_cast<RankedTensorType>(type);
  auto encodingAttr = IREE::Encoding::getSerializableAttr(rankedTensorType);
  if (encodingAttr.isSerialized()) {
    return type;
  }
  if (!llvm::all_of(layoutResolvers,
                    llvm::IsaPred<IREE::Encoding::LayoutResolverAttr>)) {
    return rankedTensorType.dropEncoding();
  }
  SmallVector<Attribute> layouts;
  for (auto attr : layoutResolvers) {
    auto encodingLayoutAttr = cast<IREE::Encoding::LayoutResolverAttr>(attr);
    Attribute layout = encodingLayoutAttr.getLayout(rankedTensorType);
    if (!layout) {
      return nullptr;
    }
    layouts.push_back(layout);
  }
  Attribute newEncoding = encodingAttr.cloneWithLayouts(layouts);
  assert(isa<IREE::Encoding::SerializableAttr>(newEncoding));
  return rankedTensorType.cloneWithEncoding(newEncoding);
};

/// Returns true if any of encoding types is a recognized encoding. See
/// `isRecognizedEncodingType` method for the definition.
static bool hasRecognizedEncoding(ModuleOp moduleOp, SymbolTable &symbolTable,
                                  Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case([&](TensorDispatchOp op) {
        if (!recognizeDispatchEntryPoints(moduleOp, symbolTable, op)) {
          return false;
        }
        for (TypeAttr typeAttr : llvm::concat<TypeAttr>(
                 op.getOperandEncodings().template getAsRange<TypeAttr>(),
                 op.getResultEncodings().template getAsRange<TypeAttr>())) {
          if (isRecognizedEncodingType(typeAttr.getValue())) {
            return true;
          }
        }
        return false;
      })
      .Case([&](IREE::Stream::TensorSizeOfOp op) {
        return isRecognizedEncodingType(op.getEncoding());
      })
      .Case<IREE::Stream::TensorEmptyOp, IREE::Stream::TensorSplatOp>(
          [&](auto op) {
            return isRecognizedEncodingType(op.getResultEncoding());
          })
      .Case([&](IREE::Stream::TensorConstantOp op) {
        return isRecognizedEncodingType(op.getResultEncoding());
      })
      .Case([&](IREE::Stream::TensorFillOp op) {
        return isRecognizedEncodingType(op.getTargetEncoding());
      })
      .Case([&](IREE::Stream::TensorCloneOp op) {
        return isRecognizedEncodingType(op.getSourceEncoding()) ||
               isRecognizedEncodingType(op.getResultEncoding());
      })
      .Case([&](IREE::Stream::TensorSliceOp op) {
        return isRecognizedEncodingType(op.getSourceEncoding()) ||
               isRecognizedEncodingType(op.getResultEncoding());
      })
      .Case([&](IREE::Stream::TensorUpdateOp op) {
        return isRecognizedEncodingType(op.getTargetEncoding()) ||
               isRecognizedEncodingType(op.getUpdateEncoding());
      })
      .Case([&](IREE::Stream::TensorEncodeOp op) {
        return isRecognizedEncodingType(op.getSourceEncoding()) ||
               isRecognizedEncodingType(op.getResultEncoding());
      })
      .Default(false);
}

/// Returns all the stream tensor ops that implement AffinityOpInterface, where
/// a stream affinity indicates the kind of enviroment the ops are expected run
/// in.
static SmallVector<IREE::Stream::AffinityOpInterface>
collectStreamTensorOps(ModuleOp moduleOp, SymbolTable &symbolTable,
                       FunctionOpInterface funcOp) {
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

    if (!hasRecognizedEncoding(moduleOp, symbolTable, affinityOp)) {
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
// environment the ops are expected run in. When an encoding is present in the
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

  SmallVector<IREE::Stream::TensorDispatchOp> getTensorDispatchOps() {
    return llvm::map_to_vector(
        llvm::filter_to_vector(streamOps,
                               llvm::IsaPred<IREE::Stream::TensorDispatchOp>),
        llvm::CastTo<IREE::Stream::TensorDispatchOp>);
  }

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
  // Because data flow analysis can be involved. Modifying the IR invalidates
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
    LLVM_DEBUG(llvm::dbgs() << "expected only one dialect implementing "
                               "AffinityAnalysisDialectInterface\n");
    return failure();
  }
  resolveLayoutAttr = usedDialects[0]->makeLayoutAttrResolver(moduleOp);

  SymbolTable symbolTable(moduleOp);
  for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
    streamOps.append(collectStreamTensorOps(moduleOp, symbolTable, funcOp));
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
        return dispatchOp.emitOpError(
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
      return dispatchOp->emitOpError(
          "failed to look up operand resource affinity");
    }

    IREE::Stream::AffinityAndOpPair key(affinityAttrs[0], dispatchOp);
    assert(cachedLayoutAttrs.contains(key) &&
           "the (affinity, dispatchOp) query is invalid");
    const SetVector<Attribute> &layoutResolvers = cachedLayoutAttrs[key];

    Type newEncodingType =
        getTypeWithResolvedEncodingLayouts(type, layoutResolvers);
    if (!newEncodingType) {
      return dispatchOp.emitOpError("failed to resolve recognized layout");
    }
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
    if (!newEncodingType) {
      return dispatchOp.emitOpError("failed to resolve recognized layout");
    }
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
  if (!newEncodingType) {
    return sizeOfOp.emitOpError("failed to resolve recognized layout");
  }
  rewriter.modifyOpInPlace(sizeOfOp,
                           [&] { sizeOfOp.setEncoding(newEncodingType); });
  return success();
}

static bool isUnrecognizedOrSerializedEncodingType(Type type) {
  if (!isRecognizedEncodingType(type)) {
    return true;
  }
  auto rankedTensorType = cast<RankedTensorType>(type);
  return IREE::Encoding::getSerializableAttr(rankedTensorType).isSerialized();
}

/// Updates the target encoding of `op` with resolved layouts.
static LogicalResult
updateTensorFillOp(RewriterBase &rewriter, IREE::Stream::TensorFillOp op,
                   const SetVector<Attribute> &layoutResolvers) {
  auto encodingType = dyn_cast<RankedTensorType>(op.getTargetEncoding());
  Type newEncodingType =
      getTypeWithResolvedEncodingLayouts(encodingType, layoutResolvers);
  if (!newEncodingType) {
    return op.emitOpError("failed to resolve recognized layout");
  }
  rewriter.modifyOpInPlace(op, [&] { op.setTargetEncoding(newEncodingType); });
  return success();
}

/// Returns success iff all the encodings are either unrecognized encoding or
/// serialized encoding.
static LogicalResult
updateTensorConstantOp(RewriterBase &rewriter,
                       IREE::Stream::TensorConstantOp op,
                       const SetVector<Attribute> &layoutResolvers) {
  return success(
      isUnrecognizedOrSerializedEncodingType(op.getResultEncoding()));
}

/// Returns success iff all the encodings are either unrecognized encoding or
/// serialized encoding.
static LogicalResult updateTensorUpdateOp(RewriterBase &rewriter,
                                          IREE::Stream::TensorUpdateOp op) {
  return success(
      isUnrecognizedOrSerializedEncodingType(op.getTargetEncoding()) &&
      isUnrecognizedOrSerializedEncodingType(op.getUpdateEncoding()));
}

/// Returns success iff all the encodings are either unrecognized encoding or
/// serialized encoding.
static LogicalResult updateTensorCloneOp(RewriterBase &rewriter,
                                         IREE::Stream::TensorCloneOp op) {
  return success(
      isUnrecognizedOrSerializedEncodingType(op.getSourceEncoding()) &&
      isUnrecognizedOrSerializedEncodingType(op.getResultEncoding()));
}

/// Returns success iff all the encodings are either unrecognized encoding or
/// serialized encoding.
static LogicalResult updateTensorSliceOp(RewriterBase &rewriter,
                                         IREE::Stream::TensorSliceOp op) {
  return success(
      isUnrecognizedOrSerializedEncodingType(op.getSourceEncoding()) &&
      isUnrecognizedOrSerializedEncodingType(op.getResultEncoding()));
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
  if (!newEncodingType) {
    return op.emitOpError("failed to resolve recognized layout");
  }
  rewriter.modifyOpInPlace(op, [&] { op.setResultEncoding(newEncodingType); });
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
  if (!newEncodingType) {
    return op.emitOpError("failed to resolve recognized layout");
  }
  rewriter.modifyOpInPlace(op, [&] { op.setSourceEncoding(newEncodingType); });
  return success();
}

/// Updates the source encoding and the result encoding of `op` with resolved
/// layouts.
static LogicalResult
updateTensorEncodeOp(RewriterBase &rewriter, IREE::Stream::TensorEncodeOp op,
                     const SetVector<Attribute> &layoutResolvers) {
  if (failed(updateResultEncoding(rewriter, op, layoutResolvers)) ||
      failed(updateSourceEncoding(rewriter, op, layoutResolvers))) {
    return failure();
  }
  return success();
}

LogicalResult StreamTensorOpUpdater::run() {
  IREE::Stream::AffinityAnalysis affinityAnalysis(moduleOp);
  if (failed(affinityAnalysis.run())) {
    return moduleOp.emitError("failed on running affinity analysis");
  }

  for (auto op : streamOps) {
    if (failed(addQuery(affinityAnalysis, op))) {
      return moduleOp->emitError(
          "failed to cache all the queries, it usually means that there are "
          "failures in affinity analysis");
    }
  }

  if (failed(resolveLayoutAttr(queries, cachedLayoutAttrs))) {
    return moduleOp->emitError("failed to resolve layouts for an query");
  }

  IRRewriter rewriter(moduleOp.getContext());
  for (auto affinityOp : streamOps) {
    const SetVector<Attribute> &layoutResolvers =
        cachedLayoutAttrs[IREE::Stream::AffinityAndOpPair(
            affinityOp.getAffinityAttr(), affinityOp)];

    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case([&](IREE::Stream::TensorDispatchOp op) {
              return updateTensorDispatchOp(rewriter, moduleOp,
                                            affinityAnalysis, op,
                                            layoutResolvers, cachedLayoutAttrs);
            })
            .Case([&](IREE::Stream::TensorSizeOfOp op) {
              return updateTensorSizeOfOp(rewriter, op, layoutResolvers);
            })
            .Case<IREE::Stream::TensorEmptyOp, IREE::Stream::TensorSplatOp>(
                [&](auto op) {
                  return updateResultEncoding(rewriter, op, layoutResolvers);
                })
            .Case([&](IREE::Stream::TensorConstantOp op) {
              return updateTensorConstantOp(rewriter, op, layoutResolvers);
            })
            .Case([&](IREE::Stream::TensorFillOp op) {
              return updateTensorFillOp(rewriter, op, layoutResolvers);
            })
            .Case([&](IREE::Stream::TensorEncodeOp op) {
              return updateTensorEncodeOp(rewriter, op, layoutResolvers);
            })
            .Case([&](IREE::Stream::TensorCloneOp op) {
              return updateTensorCloneOp(rewriter, op);
            })
            .Case([&](IREE::Stream::TensorSliceOp op) {
              return updateTensorSliceOp(rewriter, op);
            })
            .Case([&](IREE::Stream::TensorUpdateOp op) {
              return updateTensorUpdateOp(rewriter, op);
            })
            .Default(failure());

    if (failed(result)) {
      return affinityOp->emitOpError(
          "failed to convert unserialized encoding to serialized encoding");
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
      LLVM_DEBUG(
          llvm::dbgs()
          << "failed to initialize StreamTensorOpUpdater, skip the pass.");
      return;
    }

    // Signal a pass failure if any of following steps fails. At this point,
    // we recognize that all the unserialized encodings can be handled by the
    // pass.
    if (failed(streamTensorOpUpdater.run())) {
      moduleOp.emitError(
          "failed to add layouts to Stream::TensorPhaseOp with encodings");
      return signalPassFailure();
    }

    SymbolTable symbolTable(moduleOp);
    for (auto funcOp : moduleOp.getOps<FunctionOpInterface>()) {
      if (failed(duplicateExecutablesPerLayoutVariant(
              moduleOp, symbolTable,
              streamTensorOpUpdater.getTensorDispatchOps()))) {
        funcOp.emitError("failed on executable duplication");
        return signalPassFailure();
      }
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
