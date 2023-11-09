// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_
#define IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir {
namespace iree_compiler {

IREE::LinalgExt::MaterializeEncodingValueFn
getMaterializeEncodingValueFn(IREE::HAL::ExecutableTargetAttr targetAttr);

void populateMaterializeEncodingIntoPackUnPackPatterns(
    RewritePatternSet &patterns,
    IREE::LinalgExt::MaterializeEncodingConversionTarget &target,
    IREE::LinalgExt::MaterializeEncodingTypeConverter &typeConverter,
    IREE::LinalgExt::MaterializeEncodingValueFn materializeEncodingValueFn);

} // namespace iree_compiler
} // namespace mlir
#endif // IREE_COMPILER_SRC_IREE_COMPILER_CODEGEN_COMMON_ENCODINGINFO_H_
