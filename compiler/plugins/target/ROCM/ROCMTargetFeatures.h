// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETFEATURES_H_
#define IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETFEATURES_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/IR/Module.h"
#include "llvm/Target/TargetMachine.h"

namespace mlir::iree_compiler::IREE::HAL {

// Returns the list of supported mma types (mfma/wmma).
std::optional<ArrayAttr> getROCMSupportedMmaAttrs(MLIRContext *context,
                                                  StringRef targetArch);

} // namespace mlir::iree_compiler::IREE::HAL

#endif // IREE_COMPILER_PLUGINS_TARGET_ROCM_ROCMTARGETFEATURES_H_
