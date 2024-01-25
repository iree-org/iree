// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./ROCMTargetFeatures.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::iree_compiler::IREE::HAL {

static ArrayAttr getMfmaArrayAttr(MLIRContext *context,
                                  ArrayRef<IREE::GPU::MFMAType> types) {
  SmallVector<Attribute> attrs(types.size(), IREE::GPU::MFMAAttr());
  for (auto [idx, type] : llvm::enumerate(types)) {
    attrs[idx] = IREE::GPU::MFMAAttr::get(context, type);
  }
  return ArrayAttr::get(context, attrs);
}

FailureOr<ArrayAttr> getROCMSupportedMmaAttrs(MLIRContext *context,
                                              StringRef targetArch) {
  if (targetArch == "gfx940") {
    return getMfmaArrayAttr(context, {IREE::GPU::MFMAType::F16_16x16x16_F32,
                                      IREE::GPU::MFMAType::F16_32x32x8_F32});
  }
  return failure();
}

} // namespace mlir::iree_compiler::IREE::HAL
