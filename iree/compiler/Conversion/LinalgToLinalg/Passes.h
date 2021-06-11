// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_LINALGTOLINALG_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTOLINALG_PASSES_H_

namespace mlir {
namespace iree_compiler {

/// Creates a pass to convert linalg convolution ops with 1x1 kernels into
/// linalg.matmul
std::unique_ptr<OperationPass<FuncOp>> createConvert1x1ConvToMatmulPass();

std::unique_ptr<OperationPass<FuncOp>> createConvertConv2DToImg2ColPass();

/// Pass to convert a linalg.pad_tensor operation into a linalg.fill +
/// subtensor_insert. This allows lowering the operation into a single kernel.
std::unique_ptr<OperationPass<>> createPadTensorToSubTensorInsertPass();

}  // namespace iree_compiler
}  // namespace mlir
#endif  // IREE_COMPILER_CONVERSION_LINALGTOLINALG_PASSES_H_
