// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_PATTERNS_H
#define IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_PATTERNS_H

namespace mlir {

class MLIRContext;
class TypeConverter;
class RewritePatternSet;

namespace iree_compiler {
namespace IREE {
namespace PYDM {

/// Populates patterns to lower from the iree_pydm dialect to the IREE
// standard input dialects for core language structures.
void populatePyDMToIREELoweringPatterns(MLIRContext *context,
                                        TypeConverter &typeConverter,
                                        RewritePatternSet &patterns);

} // namespace PYDM
} // namespace IREE
} // namespace iree_compiler
} // namespace mlir

#endif // IREE_DIALECTS_DIALECT_IREEPYDM_TRANSFORMS_TOIREE_LOWERING_PATTERNS_H
