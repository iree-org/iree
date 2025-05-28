// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Transforms.h - Transformations for the IREE GPU dialect ------------===//
//
// Defines transformations that apply to IREE GPU ops for use in multiple
// places.
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::linalg {
class LinalgOp;
}

namespace mlir::scf {
class ForallOp;
}

namespace mlir::vector {
struct UnrollVectorOptions;
}

namespace mlir::iree_compiler::IREE::GPU {

/// Function to fuse the given producer-consumer pair of forall loops into
/// the single consumer loop. This is managed by inserting an
/// `iree_gpu.barrier_region` at the boundary to synchronize the workers at
/// the fusion point.
///
/// Copy semantics of tensors means that having multiple threads (i.e. in an
/// scf.forall) inserting into a tensor has unclear semantics without an op
/// to separate contexts with different levels of parallelism. scf.forall
/// does this through its terminator and `iree_gpu.barrier_region` does this
/// by keeping code writing to shared memory in a distinct region. This allows
/// us to always default to private memory when bufferizing.
///
/// The mapping attributes of both the producer and consumer `scf.forall` ops
/// must be in a relative descending order, for example:
///  [#gpu.thread<z>, #gpu.thread<y>, #gpu.thread<x>]
/// or
///  [#gpu.thread<linear_dim_1>, #gpu.thread<linear_dim_0>]
LogicalResult fuseForallIntoConsumer(RewriterBase &rewriter,
                                     scf::ForallOp producer,
                                     scf::ForallOp consumer,
                                     SmallVector<Operation *> consumerChain);

/// Function to fuse a collapse shape op into a forall op producer. This
/// rewrite effectively bubbles the collapse_shape op up through the forall
/// output operand, and the block argument inside the forall becomes expanded
/// with the reassociation indices of the collapse. The parallel_insert_slice
/// for the collapsed init will be collapsed, and an expand_shape on the loop
/// init block argument will be added to ensure that types match for users other
/// than the parallel_insert_slice. The following example illustrates a simple
/// case of this transformation:
///
/// ```
/// %forall = scf.forall ... shared_outs(%arg = %dest) -> tensor<4x4xf32> {
///   %user = "some_user" %arg
///   ...
///   scf.in_parallel {
///     tensor.parallel_insert_slice %val into %arg ...
///         tensor<1x4xf32> into tensor<4x4xf32>
///   }
/// }
/// %collapse = tensor.collapse_shape %forall ...
///     tensor<4x4xf32> into tensor<16xf32>
/// ```
/// After the transformation this would become:
/// ```
/// %collapse = tensor.collapse_shape %dest ...
///     tensor<4x4xf32> into tensor<16xf32>
/// %forall = scf.forall ... shared_outs(%arg = %collapse) -> tensor<16xf32> {
///   %expanded_arg = tensor.expand_shape %arg ...
///     tensor<16xf32> to tensor<4x4xf32>
///   %user = "some_user" %expanded_arg
///   ...
///   %collapsed_val = tensor.collapse_shape %val ...
///       tensor<1x4xf32> to tensor<4xf32>
///   scf.in_parallel {
///     tensor.parallel_insert_slice %collapsed_val into %arg ...
///         tensor<4xf32> into tensor<16xf32>
///   }
/// }
/// ```
FailureOr<scf::ForallOp>
fuseCollapseShapeIntoProducerForall(RewriterBase &rewriter,
                                    scf::ForallOp forallOp,
                                    tensor::CollapseShapeOp collapseOp);

/// Function to fuse an extract slice op into a forall op producer. This rewrite
/// effectively bubbles the extract_slice op up through the forall output
/// operand, and the block argument inside the forall becomes the size of the
/// slice. The parallel_insert_slice user of the init block argument will have
/// its source clamped to fit into the sliced destination, and all other uses
/// of the block argument will be replaced with the value of the output operand
/// for the forall outside of the loop body.
/// *NOTE: This can create dynamically zero sized tensors inside the forall
/// body when the source of the parallel_insert_slice is clamped.
///
/// The following example illustrates a simple case of this transformation:
/// ```
/// %forall = scf.forall ... shared_outs(%arg = %dest) -> tensor<16xf32> {
///   %user = "some_user" %arg
///   ...
///   scf.in_parallel {
///     tensor.parallel_insert_slice %val into %arg ...
///         tensor<4xf32> into tensor<16xf32>
///   }
/// }
/// %extract = tensor.extract_slice %forall ...
///     tensor<16xf32> into tensor<?xf32>
/// ```
/// After the transformation this would become:
/// ```
/// %extract = tensor.extract_slice %dest ...
///     tensor<16xf32> into tensor<?xf32>
/// %forall = scf.forall ... shared_outs(%arg = %extract) -> tensor<?xf32> {
///   // The user now has the dest from outside the loop as its operand.
///   %user = "some_user" %dest
///   ...
///   // `%clamped_val` can be dynamically zero sized.
///   %clamped_val = tensor.extract_slice %val ...
///       tensor<4xf32> to tensor<?xf32>
///   scf.in_parallel {
///     tensor.parallel_insert_slice %clamped_val into %arg ...
///         tensor<?xf32> into tensor<?xf32>
///   }
/// }
/// ```
FailureOr<scf::ForallOp>
fuseExtractSliceIntoProducerForall(RewriterBase &rewriter,
                                   scf::ForallOp forallOp,
                                   tensor::ExtractSliceOp extractSliceOp);

// Helper to convert a contraction-like linalg op to an iree_gpu.multi_mma.
FailureOr<IREE::GPU::MultiMmaOp>
convertContractionToMultiMma(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                             IREE::GPU::MmaInterfaceAttr mmaKind);

// Helper to distribute a multi_mma op to lanes.
FailureOr<Operation *> distributeMultiMmaOp(
    RewriterBase &rewriter, IREE::GPU::MultiMmaOp mmaOp,
    std::optional<SmallVector<int64_t>> workgroupSize = std::nullopt);

// Helper to map all scf.forall ops on lanes.
void mapLaneForalls(RewriterBase &rewriter, Operation *funcOp,
                    bool insertBarrier);

// Various populate pattern methods.
void populateIREEGPUDropUnitDimsPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerMultiMmaPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerBarrierRegionPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerValueBarrierPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorUnrollPatterns(
    RewritePatternSet &patterns, const vector::UnrollVectorOptions &options);
// Version of unrolling with a preset configuration.
void populateIREEGPUVectorUnrollPatterns(RewritePatternSet &patterns);
void populateIREEGPUVectorizationPatterns(RewritePatternSet &patterns);
void populateIREEGPULowerGlobalLoadDMAPatterns(RewritePatternSet &patterns);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H_
