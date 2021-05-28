// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CONVERSION_LINALGTOSPIRV_PASSES_H_
#define IREE_COMPILER_CONVERSION_LINALGTOSPIRV_PASSES_H_

#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

/// Pass to tile and vectorize Linalg operations on buffers in a single
/// workgroup.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndVectorizeInOneWorkgroupPass(const SPIRVCodegenOptions &options);

/// Pass to add the synchronizations and attributes needed to lower from PLoops
/// to GPU dialect.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createConvertToGPUPass();

/// Pass to perform the final conversion to SPIR-V dialect.
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass();

/// Pass to convert vector operations to GPU level operations. Instructions of
/// vector size equal to subgroup size are distributed across the subgroup.
std::unique_ptr<OperationPass<FuncOp>> createVectorToGPUPass();

/// Pass to convert vector read/write/arithmetic operations to the corresponding
/// cooperative matrix ops when possible.
std::unique_ptr<FunctionPass> createConvertVectorToCooperativeMatrixPass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>> createVectorizeMemrefLoadStorePass();

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createFoldProcessorIDUsesPass();

/// Pass that materializes new hal.executable.entry_point ops for
/// spv.EntryPoints that are added by other passes.
/// To be removed along with SplitDispatchFunctionPass.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createMaterializeEntryPointsPass();

/// Creates a pass to concretize hal.interface.workgroup.* ops with concrete
/// tiling and distribution scheme.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createConcretizeTileAmongWorkgroupsPass(const SPIRVCodegenOptions &options);

//===----------------------------------------------------------------------===//
// Pipelines
//===----------------------------------------------------------------------===//

/// Populates passes needed to lower a XLA HLO op to SPIR-V dialect via the
/// structured ops path. The pass manager `pm` in here operate on the module
/// within the IREE::HAL::ExecutableOp. The `workGroupSize` can be used to
/// control the work group size used in the code generation and is intended for
/// testing purposes only. The pass pipeline will set an appropriate workgroup
/// size.
void buildSPIRVTransformPassPipeline(OpPassManager &pm,
                                     const SPIRVCodegenOptions &options);

//===----------------------------------------------------------------------===//
// Patterns
//===----------------------------------------------------------------------===//

/// Populates patterns to tile and distribute linalg.copy operations.
void populateTileAndDistributeLinalgCopyPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns);

/// Populates patterns to fold processor ID uses by using processor counts
/// information where possible.
void populateFoldGPUProcessorIDUsesPatterns(MLIRContext *context,
                                            OwningRewritePatternList &patterns);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CONVERSION_LINALGTOSPIRV_PASSES_H_
