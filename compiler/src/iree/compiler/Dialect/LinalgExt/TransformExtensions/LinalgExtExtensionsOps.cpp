// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE;

LinalgExt::LinalgExtTransformOpsExtension::LinalgExtTransformOpsExtension() {
  registerTransformOps<
#define GET_OP_LIST
#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.cpp.inc"
      >();
}

void LinalgExt::LinalgExtTransformOpsExtension::init() {}

DiagnosedSilenceableFailure LinalgExt::DecomposeOnlineAttention::applyToOne(
    transform::TransformRewriter &rewriter,
    LinalgExt::OnlineAttentionOp attentionOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  FailureOr<SmallVector<Value>> replacements =
      attentionOp.decomposeOperation(rewriter);
  if (failed(replacements)) {
    attentionOp->emitError("could not decompose online_attention op");
    return DiagnosedSilenceableFailure::definiteFailure();
  }
  for (Value val : replacements.value()) {
    results.push_back(val.getDefiningOp());
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure LinalgExt::ConvertToOnlineAttention::applyToOne(
    transform::TransformRewriter &rewriter, LinalgExt::AttentionOp attentionOp,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  SmallVector<Operation *> ops;
  LinalgExt::convertToOnlineAttention(attentionOp, ops, rewriter);
  for (Operation *op : ops) {
    results.push_back(op);
  }
  return DiagnosedSilenceableFailure::success();
}

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.cpp.inc"
