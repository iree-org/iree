// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_TFTOTFSTRINGSS_H_
#define IREE_INTEGRATIONS_TFSTRINGS_CONVERSION_TFTOTFSTRINGSS_H_

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_integrations {
namespace tf_strings {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTFToTFStringsPass();

// Adds rewrite patterns for lowering tensorflow operations to tf_strings.
void populateTFToTFStringsPatterns(MLIRContext *ctx,
                                   OwningRewritePatternList &patterns);

}  // namespace tf_strings
}  // namespace iree_integrations
}  // namespace mlir

#endif  // IREE_INTEGRATIONS_TFSTRINGS_TRANSFORMS_TFSTRINGSTOSTRINGS_H_
