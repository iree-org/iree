// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Stream/Analysis/Affinity.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Stream {

#define GEN_PASS_DEF_ANNOTATEAFFINITIESPASS
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// --iree-stream-annotate-affinities
//===----------------------------------------------------------------------===//

static void annotateOp(Operation *op,
                       ArrayRef<IREE::Stream::AffinityAttr> affinities) {
  auto affinityOp = dyn_cast<IREE::Stream::AffinityOpInterface>(op);
  if (!affinityOp || !affinityOp.requiresAffinity()) {
    return;
  }
  if (!affinities.empty()) {
    op->setAttr("stream.affinities",
                ArrayAttr::get(op->getContext(),
                               llvm::to_vector_of<Attribute>(affinities)));
  }
}

static void annotateGlobalOp(IREE::Util::GlobalOpInterface globalOp,
                             AffinityAnalysis &affinityAnalysis) {
  if (!isa<IREE::Stream::AffinityTypeInterface>(globalOp.getGlobalType())) {
    return;
  }
  SmallVector<IREE::Stream::AffinityAttr> affinities;
  if (affinityAnalysis.tryLookupGlobalAffinity(globalOp, affinities)) {
    annotateOp(globalOp, affinities);
  }
}

static void annotateOperandsAndResults(Operation *op,
                                       AffinityAnalysis &affinityAnalysis) {
  auto emptyArray = ArrayAttr::get(op->getContext(), {});
  SmallVector<Attribute> operandAttrs;
  SmallVector<Attribute> operandUsageAttrs;
  for (auto operand : op->getOperands()) {
    if (isa<IREE::Stream::AffinityTypeInterface>(operand.getType())) {
      SmallVector<IREE::Stream::AffinityAttr> affinities;
      if (affinityAnalysis.tryLookupResourceAffinity(operand, affinities)) {
        operandAttrs.push_back(ArrayAttr::get(
            op->getContext(), llvm::to_vector_of<Attribute>(affinities)));
      } else {
        operandAttrs.push_back(emptyArray);
      }
      SmallVector<IREE::Stream::AffinityAttr> usageAffinities;
      if (affinityAnalysis.tryLookupResourceUsageAffinity(operand,
                                                          usageAffinities)) {
        operandUsageAttrs.push_back(ArrayAttr::get(
            op->getContext(), llvm::to_vector_of<Attribute>(usageAffinities)));
      } else {
        operandUsageAttrs.push_back(emptyArray);
      }
    }
  }
  SmallVector<Attribute> resultAttrs;
  SmallVector<Attribute> resultUsageAttrs;
  for (auto result : op->getResults()) {
    if (isa<IREE::Stream::AffinityTypeInterface>(result.getType())) {
      SmallVector<IREE::Stream::AffinityAttr> affinities;
      if (affinityAnalysis.tryLookupResourceAffinity(result, affinities)) {
        resultAttrs.push_back(ArrayAttr::get(
            op->getContext(), llvm::to_vector_of<Attribute>(affinities)));
      } else {
        resultAttrs.push_back(emptyArray);
      }
      SmallVector<IREE::Stream::AffinityAttr> usageAffinities;
      if (affinityAnalysis.tryLookupResourceUsageAffinity(result,
                                                          usageAffinities)) {
        resultUsageAttrs.push_back(ArrayAttr::get(
            op->getContext(), llvm::to_vector_of<Attribute>(usageAffinities)));
      } else {
        resultUsageAttrs.push_back(emptyArray);
      }
    }
  }
  if (!operandAttrs.empty()) {
    op->setAttr("stream.affinities.operands",
                ArrayAttr::get(op->getContext(), operandAttrs));
  }
  if (!operandUsageAttrs.empty()) {
    op->setAttr("stream.affinities.operands.usage",
                ArrayAttr::get(op->getContext(), operandUsageAttrs));
  }
  if (!resultAttrs.empty()) {
    op->setAttr("stream.affinities.results",
                ArrayAttr::get(op->getContext(), resultAttrs));
  }
  if (!resultUsageAttrs.empty()) {
    op->setAttr("stream.affinities.results.usage",
                ArrayAttr::get(op->getContext(), resultUsageAttrs));
  }
}

static void annotateFuncOp(FunctionOpInterface funcOp,
                           AffinityAnalysis &affinityAnalysis) {
  funcOp.walk([&](Operation *op) {
    SmallVector<IREE::Stream::AffinityAttr> affinities;
    if (affinityAnalysis.tryLookupExecutionAffinity(op, affinities)) {
      annotateOp(op, affinities);
    }
    annotateOperandsAndResults(op, affinityAnalysis);
  });
}

struct AnnotateAffinitiesPass
    : public IREE::Stream::impl::AnnotateAffinitiesPassBase<
          AnnotateAffinitiesPass> {
  void runOnOperation() override {
    // Run affinity analysis on the whole module.
    AffinityAnalysis affinityAnalysis(getOperation());
    if (failed(affinityAnalysis.run())) {
      return signalPassFailure();
    }

    // Annotate all ops with derived affinities.
    for (auto &op : getOperation().getOps()) {
      if (op.hasTrait<OpTrait::IREE::Util::ObjectLike>())
        continue;
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOpInterface>(op)) {
        annotateGlobalOp(globalOp, affinityAnalysis);
      } else if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
        annotateFuncOp(funcOp, affinityAnalysis);
      }
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::IREE::Stream
