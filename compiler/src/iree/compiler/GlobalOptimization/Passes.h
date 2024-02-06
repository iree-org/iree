// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
#define IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_

#include <functional>

#include "iree/compiler/Pipelines/Options.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir::iree_compiler::GlobalOptimization {

/// We have a layer of indirection around the GlobalOptimizationOptions because
/// we also need a reference to the const-eval builder, which is injected
/// in by callers.
struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  GlobalOptimizationOptions options;

  // Hook to populate a constant evaluation pass pipeline. If nullptr, then
  // no passes are added for constant evaluation. This must be injected in
  // because constant-evaluators can depend on the whole compiler, of which
  // this is a part, and we maintain strict optionality for this component.
  std::function<void(OpPassManager &passManager)> buildConstEvalPassPipeline;
};

/// Subset of the overall pass pipeline for optimizing globals and numerics.
/// We may ultimately break this out separately so creating a syntactic
/// distinction to keep that as an option.
void buildGlobalOptimizationPassPipeline(
    OpPassManager &mainPassManager, const TransformOptions &transformOptions);

//===----------------------------------------------------------------------===//
// Input canonicalization and legalization
//===----------------------------------------------------------------------===//

/// Cleans up any numeric narrowing ops inserted by
/// iree-global-opt-infer-numeric-narrowing.
std::unique_ptr<Pass> createCleanupNumericNarrowingPass();

/// Converts linalg convolution ops with 1x1 kernels into linalg.matmul.
std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass();

/// Fuses dequantization and matmul linalg.generic ops
std::unique_ptr<Pass>
createDecomposeConcatPass(bool enableConcatTransposition = false);

/// Demotes inputs (LHS, RHS) of linalg matmul-like ops from f32 to bf16.
std::unique_ptr<Pass> createDemoteContractionInputsToBF16Pass();

/// Detaches elementwise ops from named Linalg ops.
std::unique_ptr<Pass> createDetachElementwiseFromNamedOpsPass();

/// Applies patterns to erase unused linalg operands and remove dead code
/// associated.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createEraseUnusedLinalgOperands();

/// Expands tensor shape dimensions into SSA values across the program.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createExpandTensorShapesPass();

/// Fuses dequantization and matmul linalg.generic ops
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseDequantizationMatmulPass(
    bool enableQuantizedMatmulReassociation = false);

/// Fuses two matmul ops and a linalg.generic Silu op
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createFuseSiluHorizontalMatmulPass();

/// Generalizes some named Linalg ops into `linalg.generic` operations since the
/// compiler can handle that better.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGeneralizeLinalgNamedOpsPass();

/// Infers and inserts util.numeric.optional_narrow ops at points that may be
/// beneficial.
std::unique_ptr<Pass> createInferNumericNarrowingPass();

/// Materializes logical encodings to physical encodings if there is a single
/// device target.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeHomogeneousEncodingsPass();

/// Optimizes numerics given annotations added via
/// iree-global-opt-infer-numeric-narrowing.
std::unique_ptr<Pass> createOptimizeNumericsPass();

/// Propagates linalg.transpose ops to a restricted set of operations.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createPropagateLinalgTransposePass(bool enableAggressivePropagation = false);

/// Performs specialized raisings of various sequences of ops to a
/// representation easier for the compiler to handle.
std::unique_ptr<Pass> createRaiseSpecialOps();

/// Removes tensors that have 0-extents.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createRemoveZeroExtentTensorsPass();

/// Sets encoding for tensors to allow tiled execution of operations.
std::unique_ptr<Pass> createSetEncodingPass();

/// Simplifies tensor pack/unpack ops to reshape ops.
std::unique_ptr<Pass> createSimplifyPackUnpackPass();

void registerGlobalOptimizationPipeline();

} // namespace mlir::iree_compiler::GlobalOptimization

#endif // IREE_COMPILER_GLOBALOPTIMIZATION_PASSES_H_
