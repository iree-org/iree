// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

static const char kConfigAttrName[] = "lowering.config";

#include "iree/compiler/Dialect/HAL/IR/LoweringConfig.cpp.inc"
#include "iree/compiler/Dialect/HAL/IR/LoweringConfigEnums.cpp.inc"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Helpers for getting/setting the `hal.lowering.*` attributes that drive the
// linalg-based lowering.
// ===----------------------------------------------------------------------===//

IREE::HAL::LoweringConfig getLoweringConfig(Operation *op) {
  return op->getAttrOfType<IREE::HAL::LoweringConfig>(kConfigAttrName);
}

bool hasLoweringConfig(Operation *op) {
  return op->hasAttrOfType<IREE::HAL::LoweringConfig>(kConfigAttrName);
}

bool setLoweringConfig(Operation *op, IREE::HAL::LoweringConfig config) {
  if (hasLoweringConfig(op)) return false;
  op->setAttr(kConfigAttrName, config);
  return true;
}

void eraseLoweringConfig(Operation *op) { op->removeAttr(kConfigAttrName); }

//===----------------------------------------------------------------------===//
// Helpers for accessing values from the LoweringConfig attribute.
//===----------------------------------------------------------------------===//

IREE::HAL::LoweringConfig getConfigAttr(TileSizesListTypeRef tileSizes,
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
                                        context);
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
