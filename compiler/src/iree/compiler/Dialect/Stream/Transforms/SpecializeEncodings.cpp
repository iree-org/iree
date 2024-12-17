// Copyright 2024 The IREE Authors
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

// TODO(hanchung): Add "cloneWithEncoding" method to RankedTensorType.
static RankedTensorType cloneWithEncoding(RankedTensorType type,
                                          Attribute encoding) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encoding);
}

static LogicalResult
addLayoutsToTensorPhaseOps(ModuleOp moduleOp, FunctionOpInterface funcOp,
                           LayoutAttrSolverFn makeLayoutAttrFn) {
  SmallVector<AffinityOpInterface> candidates;
  funcOp.walk([&](AffinityOpInterface affinityOp) {
    // Only need to update encoding types for ops that have TensorPhaseOp trait.
    if (!affinityOp->hasTrait<OpTrait::IREE::Stream::TensorPhaseOp>()) {
      return;
    }

    // Bail out if the operation does not have an affinity attribute.
    // TODO(hanchung): We should use the default device in this case. However,
    // it is not guaranteed that default device attribute will always be set in
    // the IR. (Is the statement correct?)
    auto affAttr = affinityOp.getAffinityAttr();
    if (!affAttr) {
      return;
    }
    candidates.push_back(affinityOp);
  });

  if (candidates.empty()) {
    return success();
  }

  IRRewriter rewriter(funcOp.getContext());
  for (auto affinityOp : candidates) {
    auto affAttr = affinityOp.getAffinityAttr();
    SetVector<Attribute> layouts;
    if (failed(makeLayoutAttrFn(affAttr, moduleOp, layouts))) {
      affinityOp.emitError("failed on making layouts");
      return failure();
    }

    auto getEncodingWithNewLayouts =
        [=](Type type) -> std::optional<IREE::Encoding::EncodingAttr> {
      auto rankedTensorType = dyn_cast<RankedTensorType>(type);
      if (!rankedTensorType) {
        return std::nullopt;
      }
      auto encoding = IREE::Encoding::getEncodingAttr(rankedTensorType);
      if (!encoding) {
        return std::nullopt;
      }
      SmallVector<Attribute> attrs(layouts.begin(), layouts.end());
      return encoding.cloneWithLayouts(attrs);
    };
    // TODO(hanchung): Update other Stream operations.
    LogicalResult result =
        TypeSwitch<Operation *, LogicalResult>(affinityOp)
            .Case<Stream::TensorSizeOfOp>([&](auto sizeOfOp) {
              auto encodingType =
                  dyn_cast<RankedTensorType>(sizeOfOp.getEncoding());
              if (!encodingType) {
                return success();
              }
              std::optional<IREE::Encoding::EncodingAttr> encoding =
                  getEncodingWithNewLayouts(encodingType);
              if (!encoding.has_value()) {
                return success();
              }
              rewriter.modifyOpInPlace(sizeOfOp, [&] {
                sizeOfOp.setEncoding(
                    cloneWithEncoding(encodingType, encoding.value()));
              });
              return success();
            })
            .Default([](auto *op) { return success(); });

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
    auto usedDialects =
        gatherUsedDialectInterfaces<AffinityAnalysisDialectInterface>(moduleOp);
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

    LayoutAttrSolverFn makeLayoutAttrFn =
        usedDialects[0]->makeLayoutAttrSolver(moduleOp);
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      if (failed(
              addLayoutsToTensorPhaseOps(moduleOp, funcOp, makeLayoutAttrFn))) {
        funcOp.emitError(
            "failed on adding layouts to Stream::TensorPhaseOp with encodings");
        return signalPassFailure();
      }

      // TODO(hanchung): Duplicate executables and update dispatch ops.
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
