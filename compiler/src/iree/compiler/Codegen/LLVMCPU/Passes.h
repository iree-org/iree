// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the LLVMCPU Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_PASSES_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_PASSES_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

class TilingConfig;

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options.
//------------------------------------------------------------------------------

struct LLVMCPUVectorLoweringPassOptions {
  std::string splitVectorTransfersTo = "";
  bool lowerVectorTransposeToAVX2 = false;
  bool enableArmI8mm = false;
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(bool enableReassociateFpReductions);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTilePass(int64_t tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileAndFusePass(int64_t tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseProducerConsumer(int64_t tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseInputOperands(int64_t tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVerifyVectorSizeLegalityPass(
    int64_t maxAllowedNumberOfNativeVectors);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertToLLVMPass(bool reassociateFpReordering);

//------------------------------------------------------------------------------
// LLVMCPU Codegen specific patterns.
//------------------------------------------------------------------------------

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns);

/// Populates `patterns` to convert certain vector.contract ops to special
/// "kernels" written either in SIMD intrinsics or inline assembly.
void populateVectorContractCustomKernelsPatterns(
    IREE::HAL::ExecutableTargetAttr target, RewritePatternSet &patterns);

//----------------------------------------------------------------------------//
// LLVMCPU backend Pass Pipelines.
//----------------------------------------------------------------------------//

struct LLVMCPUPipelineOptions {
  bool decomposePackUnPackOps = true;
  bool useConfiguredVectorSizes = true;
  bool enablePeeling = false;
  bool enableVectorMasking = false;
  bool enableAArch64SSVE = false;
  bool enableAArch64I8mm = false;
  bool lowerToAVX2 = false;
};

/// Populates the passes to lower linalg ops on buffers. Currenly this
/// pipeline is only used for dispatches that just copy data from input
/// interfaces to output interface.
void addCPUBufferOpsTileAndVectorizePipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes to lower ops through data tiling transformations.
void addCPUDataTilingPipeline(OpPassManager &funcPassManager,
                              TilingConfig &tilingConfig,
                              LLVMCPUPipelineOptions &pipelineOpt);

void addCPULinalgExtTileAndVectorizePipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just
/// converts to memrefs
void addCPUDefaultPassPipeline(OpPassManager &funcPassManager);

void addConvTileAndDecomposeExpertPassPipeline(
    OpPassManager &funcPassManager, TilingConfig &tilingConfig,
    LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes needed to multi level tile, fuse and vectorize
/// lowering of linalg ops on tensors to vectors operations.
void addMmt4dTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt);

void addMultiTilingExpertPassPipeline(OpPassManager &funcPassManager,
                                      TilingConfig &tilingConfig,
                                      LLVMCPUPipelineOptions &pipelineOpt);

void addTensorToVectorsPassPipeline(OpPassManager &funcPassManager,
                                    bool lowerToVectors = true);

// Populates the passes needed to do tiling, decomposing, and vectorizing the
// convolution ops.
LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, TilingConfig &tilingConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});

/// Populates the passes needed to do two-level tile + vectorize of linalg ops.
LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, TilingConfig &tilingConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});

/// Populates the passes needed to multi level tile and lowering of linalg ops
/// on tensors to vectors operations.
LogicalResult verifyTensorToVectorsPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});

//----------------------------------------------------------------------------//
// LLVMCPU Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

/// Populates passes needed for preprocessing before codegen lowerings, as well
/// as high level lowering strategy selection.
void buildLLVMCPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager);

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool enableAArch64SME = false);

//----------------------------------------------------------------------------//
// LLVMCPU Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Populates passes needed to link HAL executables across LLVMCPU targets.
void buildLLVMCPULinkingPassPipeline(OpPassManager &modulePassManager);

//----------------------------------------------------------------------------//
// Register LLVMCPU Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc" // IWYU pragma: keep

void registerCodegenLLVMCPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_PASSES_H_
