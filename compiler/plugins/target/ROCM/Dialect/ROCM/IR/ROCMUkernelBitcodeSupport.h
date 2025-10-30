// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMUKERNELBITCODESUPPORT_H_
#define IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMUKERNELBITCODESUPPORT_H_

#include "compiler/plugins/target/ROCM/Dialect/ROCM/IR/ROCMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::ROCM {

/// Utility function to help create and replace argmax linalg with a ukernel.
LogicalResult handleArgmaxUkernel(RewriterBase &rewriter, StringRef name,
                                  DictionaryAttr targetConfiguration,
                                  Operation *contextualOp,
                                  ArrayRef<Value> inputs,
                                  ArrayRef<Value> outputs,
                                  SmallVectorImpl<Value> &otherOperands);

/// Utility function to help create and replace inner_tiled with a ukernel.
LogicalResult handleInnerTiledMmaUkernel(RewriterBase &rewriter, StringRef name,
                                         DictionaryAttr targetConfiguration,
                                         Operation *contextualOp,
                                         ArrayRef<Value> inputs,
                                         ArrayRef<Value> outputs,
                                         SmallVectorImpl<Value> &otherOperands);

} // namespace mlir::iree_compiler::IREE::ROCM

#endif // IREE_PLUGINS_TARGET_ROCM_DIALECT_ROCM_ROCMUKERNELBITCODESUPPORT_H_
