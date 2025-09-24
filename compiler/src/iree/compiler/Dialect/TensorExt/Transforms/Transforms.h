// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_TENSOREXT_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_DIALECT_TENSOREXT_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::iree_compiler::IREE::TensorExt {

/// Patterns to fold tensor.extract_slice/insert_slice with
/// iree_tensor_ext.dispatch.tensor.load/store. These patterns may not be
/// canonicalizers, since they might change the parallelism semantics in
/// non-obvious ways.
void populateTensorSliceOpWithDispatchTensorOpFoldingPatterns(
    RewritePatternSet &results, MLIRContext *context);

}; // namespace mlir::iree_compiler::IREE::TensorExt

#endif // IREE_COMPILER_DIALECT_TENSOREXT_TRANSFORMS_TRANSFORMS_H_
