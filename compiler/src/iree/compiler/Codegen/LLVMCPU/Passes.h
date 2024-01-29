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

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertToLLVMPass(bool reassociateFpReordering = false);

/// Checks CPU backend specific IR constraints (like no stack allocations)
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass(bool failOnOutOfBounds = true);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUEmitVectorizationRemarksPass();

/// Pass to select a lowering strategy for a hal.executable.variant operation.
/// The variant is annotated with the selected strategies, which are
/// subsequently ingested by LLVMCPULowerExecutableTargetPass.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUSelectLoweringStrategyPass();

/// Pass to lower the module an hal.executable.variant operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass();

/// Pass to handel F16 bit operations, but converting f16 operands to F32.
/// Currently this pass is handeling fmaxf conversion from f16 to f32,
/// and then returing a f16 output back after preforming the operation.
/// Can handel more operations if required in future.
std::unique_ptr<Pass> createExpandF16OpToF32Pass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUMmt4dVectorLoweringPass();

/// Pass to perform peeling on non-distributed loops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUPeelPass();

/// Pass to perform SplitReduction transformations of `LinalgOp`s.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(bool enableReassociateFpReductions = false);

/// Synchronizes LLVM linkage with MLIR symbol visibility.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUSynchronizeSymbolVisibilityPass();

/// Pass to tile and fuse TilingInterface ops with given tilingLevel.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileAndFusePass(int64_t tilingLevel = -1);

/// Pass to tile TilingInterface ops with given tilingLevel.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTilePass(int64_t tilingLevel = -1);

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUUnfuseFMAOpsPass();

//------------------------------------------------------------------------------
// Passes to lower Vector ops before conversion to LLVM.
//------------------------------------------------------------------------------

struct LLVMCPUVectorLoweringPassOptions {
  std::string splitVectorTransfersTo = "";
  bool lowerVectorTransposeToAVX2 = false;
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUDropVectorUnitDimsPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVirtualVectorLoweringPass(std::string splitVectorTransfersTo = "");

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorTransferLoweringPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorTransposeLoweringPass(
    bool lowerVectorTransposeToAVX2 = false);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorShapeCastLoweringPass();

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorLoweringPass();
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUVectorLoweringPass(
    const LLVMCPUVectorLoweringPassOptions &options);

/// A pass that converts certain vector.contract ops to custom kernels.
std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createVectorContractCustomKernelsPass();

// Verifies that only supported IR constructs are passed to the compiler (like
// no Linalg transform markers are set).
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgTransformLegalityPass();

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

/// Populates the passes to lower linalg ops on buffers. Currenly this
/// pipeline is only used for dispatches that just copy data from input
/// interfaces to output interface.
void addCPUBufferOpsTileAndVectorizePipeline(OpPassManager &passManager,
                                             TilingConfig &tilingConfig,
                                             bool enableVectorMasking,
                                             bool enableAArch64SSVE = false);

/// Populates the passes to lower ops through data tiling transformations.
void addCPUDataTilingPipeline(OpPassManager &passManager,
                              TilingConfig &tilingConfig,
                              bool enableVectorMasking);

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just
/// converts to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager);

void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager,
                                               TilingConfig &tilingConfig,
                                               bool enableVectorMasking,
                                               bool enableAArch64SSVE = false);

/// Populates the passes needed to multi level tile, fuse and vectorize
/// lowering of linalg ops on tensors to vectors operations.
void addMmt4dTilingExpertPassPipeline(OpPassManager &passManager,
                                      TilingConfig &tilingConfig,
                                      bool enableMicrokernels,
                                      bool lowerToAVX2);

void addMultiTilingExpertPassPipeline(
    OpPassManager &passManager, TilingConfig &tilingConfig, bool enablePeeling,
    bool enableVectorMasking, bool lowerToAVX2, bool enableAArch64SSVE = false);

void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors = true);

/// Transform dialect-based common.
void addTransformDialectPasses(OpPassManager &passManager);

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
void buildLLVMCPUCodegenConfigurationPassPipeline(OpPassManager &passManager);

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager,
                                     bool enableAArch64SME = false);

//----------------------------------------------------------------------------//
// LLVMCPU Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Assigns executable constant ordinals across all LLVMCPU variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignConstantOrdinalsPass();

/// Assigns executable import ordinals across all LLVMCPU variants.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPUAssignImportOrdinalsPass();

/// Links LLVMCPU HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createLLVMCPULinkExecutablesPass();

/// Populates passes needed to link HAL executables across LLVMCPU targets.
void buildLLVMCPULinkingPassPipeline(OpPassManager &passManager);

//----------------------------------------------------------------------------//
// Register LLVMCPU Passes
//----------------------------------------------------------------------------//

void registerCodegenLLVMCPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_PASSES_H_
