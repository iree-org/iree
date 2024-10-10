// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingTypes.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamInterfaces.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_MAKEENCODINGSOLVABLEPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {
// Returns a stably sorted list of dialect interfaces of T for all dialects used
// within the given module.
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

struct MakeEncodingSolvablePass
    : public impl::MakeEncodingSolvablePassBase<MakeEncodingSolvablePass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    auto usedDialects =
        gatherUsedDialectInterfaces<AffinityAnalysisDialectInterface>(moduleOp);
    if (usedDialects.size() != 1) {
      moduleOp.emitError("expected single resolver");
      return signalPassFailure();
    }
    std::function<LogicalResult(AffinityAttr, Operation *,
                                SetVector<Attribute> &)>
        resolver = usedDialects[0]->makeTargetResolver(moduleOp);
    IRRewriter rewriter(&getContext());
    for (auto funcOp : moduleOp.getOps<mlir::FunctionOpInterface>()) {
      SmallVector<AffinityOpInterface> candidates;
      funcOp.walk([&](AffinityOpInterface affinityOp) {
        auto affAttr = affinityOp.getAffinityAttr();
        if (!affAttr) {
          return;
        }
        candidates.push_back(affinityOp);
      });

      for (auto affinityOp : candidates) {
        auto affAttr = affinityOp.getAffinityAttr();
        // TODO: Add implementation for other ops when needed.
        LogicalResult result =
            TypeSwitch<Operation *, LogicalResult>(affinityOp)
                .Case<Stream::TensorSizeOfOp>([&](auto sizeOfOp) {
                  auto encodingType =
                      dyn_cast<RankedTensorType>(sizeOfOp.getEncoding());
                  if (!encodingType) {
                    return success();
                  }
                  auto encoding =
                      llvm::dyn_cast_or_null<IREE::Encoding::EncodingAttr>(
                          encodingType.getEncoding());
                  if (!encoding) {
                    return success();
                  }

                  SetVector<Attribute> vec;
                  // if (failed(resolver(affAttr, sizeOfOp, vec))) {
                  if (failed(resolver(affAttr, moduleOp, vec))) {
                    affinityOp.emitError("failed on getting target resolvers");
                    return failure();
                  }

                  SmallVector<Attribute> targets(vec.begin(), vec.end());
                  rewriter.modifyOpInPlace(sizeOfOp, [&] {
                    auto newEncoding = encoding.cloneWithTargets(targets);
                    sizeOfOp.setEncoding(RankedTensorType::get(
                        encodingType.getShape(), encodingType.getElementType(),
                        newEncoding));
                  });

                  return success();
                })
                .Default([](auto *op) { return success(); });

        if (failed(result)) {
          return signalPassFailure();
        }
      }
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Stream
