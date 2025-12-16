// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGPATTERNS_H_
#define IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGPATTERNS_H_

#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::Encoding {

/// Populates patterns for reifying `iree_encoding.dim` operations.
///
/// Reification traces the producer chain to resolve encoding dim ops:
/// - From ops implementing EncodingDimReificationInterface (set_encoding,
/// tensor.cast)
/// - Through DPS ops (linalg, etc.): forwards query to tied init operand
///
/// Note: External models for EncodingDimReificationInterface (e.g., for
/// tensor.cast) are registered via registerEncodingExternalModels() in
/// ExternalInterfaces/.
void populateEncodingDimReificationPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::Encoding

#endif // IREE_COMPILER_DIALECT_ENCODING_IR_ENCODINGPATTERNS_H_
