// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_CONVERT_TF_STRINGS_TO_STRINGS_H_
#define IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_CONVERT_TF_STRINGS_TO_STRINGS_H_

#include "iree/compiler/Dialect/Modules/Strings/IR/Dialect.h"
#include "iree/compiler/Dialect/Modules/Strings/IR/Types.h"
#include "iree_tf_compiler/dialect/tf_strings/ir/types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFStringsToStringsPass();

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateTFStringsToStringsPatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns);

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TFSTRINGS_TRANSFORMS_TFSTRINGSTOSTRINGS_H_
