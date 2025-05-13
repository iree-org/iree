// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_
#define IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_

#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;

namespace mlir::iree_compiler {

struct DistributionConfig {
  SmallVector<int64_t> tileSizes;
  SmallVector<Attribute> mapping;
};

using DistributionConfigFn = function_ref<SmallVector<DistributionConfig>(
    ArrayRef<int64_t>, MLIRContext *)>;

LogicalResult
combineLayoutTransformation(MLIRContext *ctx, FunctionOpInterface funcOp,
                            DistributionConfigFn distributionConfigFn);

} // namespace mlir::iree_compiler
#endif // IREE_COMPILER_CODEGEN_COMMON_COMBINELAYOUTTRANSFORMATION_H_
