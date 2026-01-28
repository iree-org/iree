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
namespace mlir::tensor {
class ExtractSliceOp;
} // namespace mlir::tensor

namespace mlir::iree_compiler::IREE::PCF {

// Helper to convert scf.forall ops to pcf.loop by linearizing/delinearizing
// ids beyond |numIds| into the slowest varying id. Uses
// DeviceMappingAttrInterface to infer the order of ids from slowest to fastest
// varying. If |numIds| <= 0, then no linearization/delinearization is done.
FailureOr<PCF::LoopOp> convertForallToPCF(RewriterBase &rewriter,
                                          scf::ForallOp forallOp,
                                          PCF::ScopeAttrInterface scope,
                                          int64_t numIds = -1);

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
};

// Helpers to match a consumer as fusable into a producer. There are two
// supported cases:
//   1. The tilable |target| only consumes a single result of the producer but
//      the produced operand may be constructed out of multiple writes within
//      the producer.
//   1. The tilable |target| consumes multiple results of the producer but only
//      a single writing op constructs each consumed result.
// Populates |params| with the matched information needed to perform a fusion
// upon success. On failure |params| may be partially populated. It is the
// caller's responsibility to pass an empty struct on subsequent matches.
//
// Note that currently multiple producers split across block/region boundaries
// is unsupported. We need to guarantee the existence of a point in
// the control flow of the IR where the fused op is guaranteed to
// produce its original results in their entirety.
//
// For example:
//
//   pcf.write_slice
//   scf.if {
//     %0 = ...
//     pcf.write_slice ...[%0]
//
// This IR is problematic because the most dominated `write_slice` is
// the one inside the `if`, however if we put the fused op there then
// there is no guarantee we actually produce the full original result
// in the aggregate since some of the writes will be masked off.
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

// Fuse a tensor.extract_slice consumer into a pcf.loop producer. This shrinks
// the result to the extracted size and clamps all write_slice ops accordingly.
LogicalResult
fuseExtractSliceIntoProducerLoop(RewriterBase &rewriter, PCF::LoopOp loopOp,
                                 tensor::ExtractSliceOp extractSliceOp);

// Fuse a tensor.extract_slice consumer into a pcf.generic producer. This
// shrinks the result to the extracted size and clamps all write_slice ops
// accordingly.
LogicalResult
fuseExtractSliceIntoProducerGeneric(RewriterBase &rewriter,
                                    PCF::GenericOp genericOp,
                                    tensor::ExtractSliceOp extractSliceOp);

// Composes a pcf.write_slice with a tensor.parallel_insert_slice from an
// scf.forall terminator. The write_slice's destination must be produced by the
// forall op, and the parallel_insert_slice must be inserting into that result.
// Returns the newly created write_slice op on success.
FailureOr<PCF::WriteSliceOp>
composeWriteSliceWithParallelInsert(RewriterBase &rewriter,
                                    PCF::WriteSliceOp writeSliceOp);

} // namespace mlir::iree_compiler::IREE::PCF

#endif // IREE_COMPILER_CODEGEN_DIALECT_PCF_TRANSFORMS_TRANSFORMS_H_
