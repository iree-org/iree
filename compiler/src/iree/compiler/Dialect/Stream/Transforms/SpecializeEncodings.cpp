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
#include "mlir/IR/PatternMatch.h"
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

      // TODO(hanchung): Duplicate executables and update dispatch ops.
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
