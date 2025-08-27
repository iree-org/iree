// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.h"
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc"

namespace mlir::iree_compiler::IREE::ROCM {

//===----------------------------------------------------------------------===//
// BuiltinTuningModuleAttr
//===----------------------------------------------------------------------===//

FailureOr<mlir::ModuleOp>
BuiltinTuningModuleAttr::getModule(Operation * /*annotationSite*/) const {
  auto &rocmDialect = cast<ROCMDialect>(getDialect());
  return rocmDialect.getOrLoadBuiltinModule(getBuiltinFilename());
}

//===----------------------------------------------------------------------===//
// UKernelProviderAttr
//===----------------------------------------------------------------------===//

/// Utility function to help create and replace argmax linalg with a ukernel.
static LogicalResult handleArgmaxUkernel(
    RewriterBase &rewriter, StringRef name, DictionaryAttr targetConfiguration,
    Operation *contextualOp, SmallVectorImpl<Value> &inputs,
    SmallVectorImpl<Value> &outputs, SmallVectorImpl<Value> &otherOperands) {
  auto genericOp = dyn_cast<linalg::GenericOp>(contextualOp);
  if (!genericOp) {
    return rewriter.notifyMatchFailure(
        genericOp, "expected a linalg.generic op for argmax");
  }
  // Currently only support 1D reduction, where reduction is on fastest dim.
  // Tiling argmax ukernel is also set to enforce this structure.
  const int kReductionDim = genericOp.getNumLoops() - 1;
  Location loc = genericOp.getLoc();
  Value reductionDimSize = rewriter.create<tensor::DimOp>(
      loc, genericOp.getDpsInputOperand(0)->get(), kReductionDim);
  // `returnsMaxValue` differentiates between the two argmax versions :-
  // 1. Returns only the index of the max value (returnsMaxValue == true)
  // 2. Returns both the max value as well as the corresponding index.
  bool returnsMaxValue = genericOp.getResults()[0].use_empty();
  Value writeMaxValueFlag = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI1Type(), rewriter.getBoolAttr(!returnsMaxValue));
  llvm::append_values(otherOperands, reductionDimSize, writeMaxValueFlag);
  MLIRContext *context = rewriter.getContext();
  auto fnDefAttrs = DictionaryAttr::get(
      context, {{"vm.import.module", StringAttr::get(context, "rocm")}});
  auto ukernelOp = rewriter.create<IREE::Codegen::UKernelGenericOp>(
      loc, contextualOp->getResultTypes(), name, inputs, outputs, otherOperands,
      fnDefAttrs, /*num_strided_outer_dims=*/0);
  if (returnsMaxValue) {
    rewriter.replaceAllUsesWith(genericOp.getResults()[1],
                                ukernelOp.getResults()[1]);
    return success();
  }
  ResultRange origResults = genericOp.getResults();
  ResultRange newResults = ukernelOp.getResults();
  if (origResults.size() != newResults.size()) {
    return rewriter.notifyMatchFailure(genericOp, "result count mismatch");
  }
  rewriter.replaceAllUsesWith(genericOp.getResults()[0],
                              ukernelOp.getResults()[0]);
  rewriter.replaceAllUsesWith(genericOp.getResults()[1],
                              ukernelOp.getResults()[1]);
  return success();
}

std::optional<LogicalResult> UKernelProviderAttr::createAndReplaceWithUkernelOp(
    RewriterBase &rewriter, StringRef name, DictionaryAttr targetConfiguration,
    Operation *contextualOp, SmallVectorImpl<Value> &inputs,
    SmallVectorImpl<Value> &outputs,
    SmallVectorImpl<Value> &otherOperands) const {
  if (name.contains("argmax")) {
    return handleArgmaxUkernel(rewriter, name, targetConfiguration,
                               contextualOp, inputs, outputs, otherOperands);
  }
  // TODO(avarma): Add multi_mfma ukernel support via descriptors.
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Attribute Registration
//===----------------------------------------------------------------------===//

void ROCMDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMAttrs.cpp.inc" // IWYU pragma: keep
      >();
}

} // namespace mlir::iree_compiler::IREE::ROCM
