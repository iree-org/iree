// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

static const char kConfigAttrName[] = "lowering.config";
static const char kTranslationInfoAttrName[] = "translation.info";

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfigEnums.cpp.inc"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting information needed to lower an executable. These
// are information that are stored as attributes on the
// `hal.executable.entry_point`
//===----------------------------------------------------------------------===//

IREE::HAL::TranslationInfo buildTranslationInfo(
    IREE::HAL::DispatchLoweringPassPipeline passPipeline,
    ArrayRef<int64_t> workloadPerWorkgroup, MLIRContext *context) {
  OpBuilder builder(context);
  auto pipelineAttr =
      IREE::HAL::DispatchLoweringPassPipelineAttr::get(context, passPipeline);
  ArrayAttr workloadPerWorkgroupAttr = nullptr;
  if (!workloadPerWorkgroup.empty()) {
    workloadPerWorkgroupAttr = builder.getI64ArrayAttr(workloadPerWorkgroup);
  }
  return IREE::HAL::TranslationInfo::get(pipelineAttr, workloadPerWorkgroupAttr,
                                         context);
}

IREE::HAL::TranslationInfo getTranslationInfo(
    IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  return entryPointOp->getAttrOfType<IREE::HAL::TranslationInfo>(
      kTranslationInfoAttrName);
}

SmallVector<int64_t> getWorkgroupSize(
    IREE::HAL::ExecutableEntryPointOp entryPointOp) {
  SmallVector<int64_t> workgroupSize;
  if (Optional<ArrayAttr> workgroupSizeAttrList =
          entryPointOp.workgroup_size()) {
    workgroupSize.resize(workgroupSizeAttrList->size());
    for (auto attr : llvm::enumerate(workgroupSizeAttrList.getValue())) {
      workgroupSize[attr.index()] = attr.value().cast<IntegerAttr>().getInt();
    }
  }
  return workgroupSize;
}

void setTranslationInfo(IREE::HAL::ExecutableEntryPointOp entryPointOp,
                        IREE::HAL::TranslationInfo translationInfo,
                        ArrayRef<int64_t> workgroupSize) {
  entryPointOp->setAttr(kTranslationInfoAttrName, translationInfo);
  // The workgroup size is set on the entry point op directly.
  if (!workgroupSize.empty()) {
    MLIRContext *context = entryPointOp->getContext();
    auto indexType = IndexType::get(context);
    auto attrs = llvm::to_vector<4>(
        llvm::map_range(workgroupSize, [&](int64_t v) -> Attribute {
          return IntegerAttr::get(indexType, v);
        }));
    entryPointOp.workgroup_sizeAttr(ArrayAttr::get(context, attrs));
  }
}

//===----------------------------------------------------------------------===//
// Helpers for getting/setting the `hal.lowering.*` attributes that drive the
// linalg-based lowering.
// ===----------------------------------------------------------------------===//

IREE::HAL::LoweringConfig getLoweringConfig(Operation *op) {
  return op->getAttrOfType<IREE::HAL::LoweringConfig>(kConfigAttrName);
}

void setLoweringConfig(Operation *op, IREE::HAL::LoweringConfig config) {
  op->setAttr(kConfigAttrName, config);
}

void eraseLoweringConfig(Operation *op) { op->removeAttr(kConfigAttrName); }

//===----------------------------------------------------------------------===//
// Helpers for accessing values from the LoweringConfig attribute.
//===----------------------------------------------------------------------===//

IREE::HAL::LoweringConfig buildConfigAttr(TileSizesListTypeRef tileSizes,
                                          ArrayRef<int64_t> nativeVectorSize,
                                          MLIRContext *context) {
  OpBuilder builder(context);
  ArrayAttr tileSizesAttr = nullptr;
  if (!tileSizes.empty()) {
    auto attrList = llvm::to_vector<4>(
        llvm::map_range(tileSizes, [&](ArrayRef<int64_t> sizes) -> Attribute {
          return builder.getI64ArrayAttr(sizes);
        }));
    tileSizesAttr = builder.getArrayAttr(attrList);
  }
  ArrayAttr nativeVectorSizeAttr = nullptr;
  if (!nativeVectorSize.empty()) {
    nativeVectorSizeAttr = builder.getI64ArrayAttr(nativeVectorSize);
  }
  return IREE::HAL::LoweringConfig::get(tileSizesAttr, nativeVectorSizeAttr,
                                        /*passPipeline = */ nullptr,
                                        /*workgroupSize = */ nullptr, context);
}

TileSizesListType getTileSizes(IREE::HAL::LoweringConfig config) {
  auto tileSizesAttr = config.tileSizes();
  if (!tileSizesAttr) return {};
  return llvm::to_vector<1>(llvm::map_range(
      tileSizesAttr, [&](Attribute attr) -> SmallVector<int64_t, 4> {
        return llvm::to_vector<4>(
            llvm::map_range(attr.cast<ArrayAttr>(), [&](Attribute intAttr) {
              return intAttr.cast<IntegerAttr>().getInt();
            }));
      }));
}

SmallVector<int64_t, 4> getTileSizes(IREE::HAL::LoweringConfig config,
                                     unsigned level) {
  ArrayAttr tileSizesAttr = config.tileSizes();
  if (!tileSizesAttr || tileSizesAttr.size() <= level) return {};
  return llvm::to_vector<4>(llvm::map_range(
      tileSizesAttr.getValue()[level].cast<ArrayAttr>(),
      [&](Attribute intAttr) { return intAttr.cast<IntegerAttr>().getInt(); }));
}

SmallVector<Value, 4> getTileSizes(OpBuilder &b, Operation *op,
                                   unsigned level) {
  return llvm::to_vector<4>(
      llvm::map_range(getTileSizes(op, level), [&](int64_t t) -> Value {
        return b.create<ConstantIndexOp>(op->getLoc(), t);
      }));
}

SmallVector<int64_t, 4> getNativeVectorSize(IREE::HAL::LoweringConfig config) {
  ArrayAttr nativeVectorSizeAttr = config.nativeVectorSize();
  if (!nativeVectorSizeAttr) return {};
  return llvm::to_vector<4>(llvm::map_range(
      nativeVectorSizeAttr,
      [&](Attribute intAttr) { return intAttr.cast<IntegerAttr>().getInt(); }));
}

}  // namespace iree_compiler
}  // namespace mlir
