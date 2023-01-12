// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_VMVX_ENCODINGINFO_H_
#define IREE_COMPILER_CODEGEN_VMVX_ENCODINGINFO_H_

#include "iree-dialects/Dialect/LinalgExt/Passes/Passes.h"

namespace mlir {
namespace iree_compiler {
FailureOr<IREE::LinalgExt::MaterializeEncodingValueInfo>
chooseDynamicEncodingInfoVMVXMicrokernels(RankedTensorType tensorType,
                                          OpBuilder &builder, Location loc);
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_VMVX_ENCODINGINFO_H_
