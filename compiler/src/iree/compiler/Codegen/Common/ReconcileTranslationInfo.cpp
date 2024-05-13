// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//=== ReconcileTranslationInfo.cpp ---------------------------------------===//
//
// While lowering executable target, the pipelines used are run at a
// func-like op granularity. Each of these func-like operations set the
// workgroup size, and subgroup size as required (as part of the
// `TranslationInfo`). Eventually these have to be reconciled and set
// appropriately on the surrounding HAL ops for the host runtime to pick them
// up. In case of inconsistencies, this pass will throw an error.
//===---------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"

namespace mlir::iree_compiler {

namespace {

class ReconcileTranslationInfoPass
    : public ReconcileTranslationInfoBase<ReconcileTranslationInfoPass> {
public:
  void runOnOperation() override;
};
} // namespace

// Reconcile workgroup sizes across all translation infos.
static FailureOr<SmallVector<int64_t>> reconcileWorkgroupSize(
    ArrayRef<IREE::Codegen::TranslationInfoAttr> translationInfos) {
  if (translationInfos.empty()) {
    return SmallVector<int64_t>{};
  }
  SmallVector<int64_t> reconciledWorkgroupSize =
      llvm::to_vector(translationInfos.front().getWorkgroupSize());
  for (auto translationInfo : translationInfos.drop_front()) {
    auto workGroupSize = llvm::to_vector(translationInfo.getWorkgroupSize());
    if (workGroupSize != reconciledWorkgroupSize) {
      return failure();
    }
  }
  return reconciledWorkgroupSize;
}

// Reconcile subgroup size across all translation infos.
static FailureOr<int64_t> reconcileSubgroupSize(
    ArrayRef<IREE::Codegen::TranslationInfoAttr> translationInfos) {
  if (translationInfos.empty()) {
    return int64_t();
  }
  int64_t subgroupSize = translationInfos.front().getSubgroupSize();
  for (auto translationInfo : translationInfos.drop_front()) {
    if (subgroupSize != translationInfo.getSubgroupSize()) {
      return failure();
    }
  }
  return subgroupSize;
}

/// Helper function to retrieve the target-func-attrs value from translation
/// info.
static DictionaryAttr
getTargetFuncAttrs(IREE::Codegen::TranslationInfoAttr translationInfo) {
  auto translationConfig = translationInfo.getConfiguration();
  if (!translationConfig) {
    return nullptr;
  }
  auto attr = translationConfig.getAs<DictionaryAttr>("llvm_func_attrs");
  if (!attr) {
    return nullptr;
  }
  return attr;
}

void ReconcileTranslationInfoPass::runOnOperation() {
  auto variantOp = getOperation();
  auto innerModuleOp = variantOp.getInnerModule();

  auto exportOps = variantOp.getOps<IREE::HAL::ExecutableExportOp>();
  if (!llvm::hasSingleElement(exportOps)) {
    variantOp.emitOpError("reconciliation for multiple export ops unsupported");
    return signalPassFailure();
  }
  auto exportOp = *exportOps.begin();
  Builder builder(&getContext());

  SmallVector<IREE::Codegen::TranslationInfoAttr> translationInfos;
  innerModuleOp->walk([&](FunctionOpInterface funcOp) {
    auto translationInfo = getTranslationInfo(funcOp);
    if (!translationInfo) {
      return;
    }

    translationInfos.push_back(translationInfo);
    // The following is moving the target-func-attrs specification from
    // translation info into the func-like op. This is not the best
    // place to do this, but the intent is after this pass all the
    // lowering configs and translation infos will be deleted.
    DictionaryAttr targetFuncAttrs = getTargetFuncAttrs(translationInfo);
    if (targetFuncAttrs) {
      funcOp->setAttr("llvm_func_attrs", targetFuncAttrs);
    }
  });

  // Reconcile workgroup sizes.
  FailureOr<SmallVector<int64_t>> reconciledWorkgroupSize =
      reconcileWorkgroupSize(translationInfos);
  if (failed(reconciledWorkgroupSize)) {
    exportOp.emitOpError("failed to reconcile workgroup sizes");
    return signalPassFailure();
  }
  if (reconciledWorkgroupSize->size() > 3) {
    exportOp.emitOpError(
        "reconciled workgroup size is greater than 3 (illegal)");
    return signalPassFailure();
  }
  std::array<int64_t, 3> workgroupSize = {1, 1, 1};
  for (auto [index, size] : llvm::enumerate(reconciledWorkgroupSize.value())) {
    workgroupSize[index] = size;
  }
  auto workgroupSizeArrayAttr = builder.getIndexArrayAttr(workgroupSize);
  exportOp.setWorkgroupSizeAttr(workgroupSizeArrayAttr);

  // Reconcile subgroup sizes.
  FailureOr<int64_t> reconciledSubgroupSize =
      reconcileSubgroupSize(translationInfos);
  if (failed(reconciledSubgroupSize)) {
    exportOp.emitOpError("failed to reconcile subgroup size");
    return signalPassFailure();
  }
  if (reconciledSubgroupSize.value() != int64_t()) {
    exportOp.setSubgroupSizeAttr(
        builder.getIndexAttr(reconciledSubgroupSize.value()));
  }

  // Erase all the lowering configs and translation infos.
  innerModuleOp->walk([](Operation *op) {
    if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
      eraseTranslationInfo(funcOp);
    }
    eraseLoweringConfig(op);
  });
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createReconcileTranslationInfoPass() {
  return std::make_unique<ReconcileTranslationInfoPass>();
}

} // namespace mlir::iree_compiler
