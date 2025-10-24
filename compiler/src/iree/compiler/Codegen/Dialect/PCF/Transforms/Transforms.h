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
namespace mlir {
class TilingInterface;
} // namespace mlir
namespace mlir::scf {
class ForallOp;
} // namespace mlir::scf

namespace mlir::iree_compiler::IREE::PCF {

// Helper to convert scf.forall ops to pcf.loop.
FailureOr<PCF::LoopOp> convertForallToPCF(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::ScopeAttr scope);

struct ConsumerFusionParams {
  // List of operands in the consumer that are fused along.
  SmallVector<unsigned> operands;
  // List of results of the producer that are fused along.
  SetVector<unsigned> results;
  // List of slices that produce the results. This has two possible
  // interpretations. If |results| > 1, then |slices| == |operands| must hold
  // and each slice corresponds to the sole write to the result consumed by the
  // corresponding operand. If |results| == 1, then each slice is a different
  // write to the same sole result.
  //
  // In the first case, slices[0] is the most dominant slice (and thus the
  // insertion point for the fused op).
  SmallVector<PCF::WriteSliceOp> slices;

  void clear() {
    operands.clear();
    results.clear();
    slices.clear();
  }
};

// Helpers to match a consumer as fusable into a producer. There are two
// supported cases:
//   1. The tilable |target| only consumes a single result of the producer but
//      the produced operand may be constructed out of multiple writes within
//      the producer.
//   1. The tilable |target| consumes multiple results of the producer but only
//      a single writing op constructs each consumed result.
// Populates |params| with the matched information needed to perform a fusion
// upon success. On failure |params| is cleared and a different tilable consumer
// may be matched against.
LogicalResult matchTilableConsumer(RewriterBase &rewriter,
                                   PCF::GenericOp genericOp,
                                   TilingInterface target,
                                   ConsumerFusionParams &params);
LogicalResult matchTilableConsumer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                   TilingInterface target,
                                   ConsumerFusionParams &params);

void fuseTilableConsumer(RewriterBase &rewriter, PCF::GenericOp genericOp,
                         TilingInterface target,
                         const ConsumerFusionParams &params);
void fuseTilableConsumer(RewriterBase &rewriter, PCF::LoopOp loopOp,
                         TilingInterface target,
                         const ConsumerFusionParams &params);

// Pattern set for dropping unused results from scoped ops. Due to memory
// effects this requires cascading operation erasure and is unsuitable for
// a canonicalization pattern.
void populatePCFDropUnusedResultPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_TRANSFORMS_H_
