// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the Common Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_COMMON_PASSES_H_
#define IREE_COMPILER_CODEGEN_COMMON_PASSES_H_

#include <limits>

#include "iree/compiler/Codegen/Common/PassUtils.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

/// Function to register all dependent dialects for Transform Dialect based
/// passes.
void registerTransformDialectTranslationDependentDialects(
    DialectRegistry &registry);

/// Passes that are done on all backends before target-specific code-generation
/// kicks in.
void addCommonTargetExecutablePreprocessingPasses(
    FunctionLikeNest &funcPassManager, bool useDecomposeSoftmaxFusion = true);

/// Post-bufferization passes run to cleanup the IR
/// (ResolveShapedTypeResultDims, Canonicalization/CSE and
/// CleanupBufferAllocView).
/// TODO(#20912): Enable the injection of assume_alignment ops after the
/// hoisting bug is fixed.
void addIREEPostBufferizationPasses(OpPassManager &funcPassManager,
                                    bool injectAssumeAlignmentOp = true);

using bufferization::BufferizationOptions;
void addIREEComprehensiveBufferizePasses(
    OpPassManager &funcPassManager,
    std::optional<BufferizationOptions::AllocationFn> allocationFn =
        std::nullopt,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn = std::nullopt,
    bool injectAssumeAlignmentOp = true);

/// Populate Encoding to Nop pass and canonicalizer pass to the pipeline.
void addEncodingToNopPasses(FunctionLikeNest &passManager);

/// Populate Encoding to padding pass and canonicalizer pass to the pipeline.
void addEncodingToPaddingPasses(FunctionLikeNest &passManager);

/// Links nested transform dialect tuning specs named sequences into a single
/// entry point. Returns the new named sequence op (inserted into the `module`)
/// that includes the nested tuning specs, or a null op when no nested named
/// sequences were found. The order of inclusion is the same as the order in
/// which these nested tuning specs appear in the IR.
FailureOr<transform::NamedSequenceOp> linkTuningSpecs(ModuleOp module);

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options. See Passes.td for details.
//------------------------------------------------------------------------------

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createConvertToDestinationPassingStylePass(
    bool useWARForCooperativeMatrixCodegen);

std::unique_ptr<Pass> createDecomposeSoftmaxPass(bool useFusion);

std::unique_ptr<Pass> createDecomposeMemrefsPass();

/// Pass to perform linalg on tensor bufferization. The function passed into
/// the pass through the `allocationFn` argument is invoked whenever a new
/// buffer is to be created. The callback will be passed the Values for the
/// dynamic dimensions in the memref type that is to be allocated.  The
/// callback is expected to return a MemRefType Value.  When no `allocationFn`
/// is specified, the default allocator generates an `std.alloc` instruction
/// with the allocated MemRefType having no stride map (i.e. default row-major
/// striding) and default memory space.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createIREEComprehensiveBufferizePass(
    std::optional<BufferizationOptions::AllocationFn> allocationFn,
    std::optional<BufferizationOptions::MemCpyFn> memCpyFn);

/// Create an IREE-specific Transform dialect interpreter pass with all
/// registrations necessary for IREE.
std::unique_ptr<Pass>
createTransformDialectInterpreterPass(StringRef transformSequenceName);

/// Pass to tile and distribute to workgroups.
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createTileAndDistributeToWorkgroupsPass(
    int32_t maxWorkgroupParallelDims,
    linalg::DistributionMethod distributionMethod);

// Pass to tile and distribute using scf.forall with workgroup reordering.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createTileAndDistributeToWorkgroupsWithReordering(bool transposeWorkgroup);

//----------------------------------------------------------------------------//
// CodeGen Common Patterns
//----------------------------------------------------------------------------//

/// Populates patterns with patterns to concretize tensor.pad op's result
/// shape. `numWorkgroups`, if not empty, will be used as bounds for simplifying
/// workgroup ID ops.
void populateConcretizePadResultShapePatterns(
    RewritePatternSet &patterns, ArrayRef<int64_t> numWorkgroups = {});

/// Populates `patterns` with patterns to fold `affine.min` ops in tiled and
/// distributed loops.
void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns, ArrayRef<int64_t> staticNumWorkgroup = {});

/// Populates `patterns` with a very specific pattern that vectorizes a
/// linalg.conv op for a single thread. The linalg.conv should compute on
/// static-sized subviews. To match, output shape must be 1x1xWoxCo, where Co
/// Co is a multiple of 4, and filter shape must be 1x1x4xCo.
void populateLinalgToVectorVectorizeConvPatterns(MLIRContext *context,
                                                 RewritePatternSet &patterns);

/// Populates `patterns` with patterns that vectorize tensor.pad with static
/// result shape by generating control flows to guard against vector transfer
/// read ops to make sure they are in bounds.
///
/// Such conversions are needed for correctness when the tensor.pad op has
/// dynamic low padding values and also beneficial for eventually lowering to
/// hardware targets without native support for vector transfer read ops with
/// out of bound semantics.
void populateVectorizePadPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

/// Collect patterns to fold tensor.extract_slice -> vector.transfer_read and
/// vector.transfer_write -> tensor.insert_slice op chains into vector tranfer
/// read and write ops.
void populateVectorTransferTensorSliceTransforms(RewritePatternSet &patterns,
                                                 PatternBenefit benefit = 1);

void populateDecomposeMemrefsPatterns(RewritePatternSet &patterns);

/// Add a pattern to combine instructions across scf.for boundary. It is common
/// when doing incremental lowering to generate transient ops that cancel each
/// other out. Canonicalization usually clean up those operations. When the
/// value is loop carried, MLIR canonicalization currently doesn't remove the
/// redundant operations.
///
/// The pattern added here provides a workaround to MLIR's limitation, and does
/// ad hoc clean up of instructions found in IREE. Once we have a more general
/// mechanism in MLIR this pattern can be completely removed.
///
/// This pattern does this kind of transformation on scf.for:
/// ```
/// %21 = vector.shape_cast %20 : vector<4xf32> to vector<1x4xf32>
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %21)
///    -> vector<1x4xf32> {
///    [...]
///    %100 = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
///    [...]
///    %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
///    scf.yield %109 : vector<1x4xf32>
///  }
///  %24 = vector.shape_cast %22 : vector<1x4xf32> to vector<4xf32>
/// ```
/// ->
/// ```
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %20)
///    -> vector<4xf32> {
///    [...]
///    scf.yield %108 : vector<4xf32>
///  }
/// ```
void populateForOpInductionVarShapePatterns(RewritePatternSet &,
                                            PatternBenefit = 1);

//----------------------------------------------------------------------------//
// Register CodeGen Common Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/Common/Passes.h.inc" // IWYU pragma: keep

/// Method to register all passes.
void registerCodegenCommonPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_COMMON_PASSES_H_
