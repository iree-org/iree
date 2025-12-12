// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations for the IREE PCF dialect ------------===//
//
// Defines transformations that apply to IREE PCF ops for use in multiple
// places.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.h"
#include "mlir/IR/PatternMatch.h"

// Forward declares.
namespace mlir::scf {
class ForallOp;
} // namespace mlir::scf

namespace mlir::iree_compiler::IREE::PCF {

// Helper to convert scf.forall ops to pcf.loop by linearizing/delinearizing
// ids beyond |numIds| into the slowest varying id. Uses
// DeviceMappingAttrInterface to infer the order of ids from slowest to fastest
// varying. If |numIds| <= 0, then no linearization/delinearization is done.
FailureOr<PCF::LoopOp> convertForallToPCF(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::ScopeAttrInterface scope,
                                          int64_t numIds = -1);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_TRANSFORMS_H_
