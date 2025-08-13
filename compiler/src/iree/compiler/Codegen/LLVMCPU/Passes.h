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

#include <optional>

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//------------------------------------------------------------------------------
// Wrappers that not use tablegen options.
//------------------------------------------------------------------------------

struct LLVMCPUVectorLoweringPassOptions {
  std::string splitVectorTransfersTo = "";
  bool lowerVectorTransposeToAVX2 = false;
  bool enableArmI8mm = false;
  bool enableArmSME = false;
};

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUSplitReductionPass(bool enableReassociateFpReductions);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTilePass(int64_t tilingLevel, bool skipRootOp);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileAndFusePass(int64_t tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseProducerConsumerPass(
    IREE::CPU::TilingLevel tilingLevel);

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createLLVMCPUTileRootAndFuseInputOperandsPass(
    IREE::CPU::TilingLevel tilingLevel);

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
  bool disableDistribution = false;
  bool decomposePackUnPackOps = true;
  bool useConfiguredVectorSizes = true;
  bool enablePeeling = false;
  bool enableVectorMasking = false;
  bool enableAArch64SME = false;
  bool enableAArch64I8mm = false;
  bool lowerToAVX2 = false;
};

/// Populates the passes to lower linalg ops on buffers. Currenly this
/// pipeline is only used for dispatches that just copy data from input
/// interfaces to output interface.
void addCPUBufferOpsTileAndVectorizePipeline(
    OpPassManager &funcPassManager, const LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes to lower ops through data tiling transformations.
void addCPUDataTilingPipeline(OpPassManager &funcPassManager,
                              const LLVMCPUPipelineOptions &pipelineOpt);

void addCPULinalgExtTileAndVectorizePipeline(
    OpPassManager &funcPassManager, const LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes to lower scalars and unknown tensor op (i.e. linalg op
/// that is not specialized by any pipeline). Adds an additional level of tiling
/// and converts to memrefs.
void addCPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               const LLVMCPUPipelineOptions &pipelineOpt);

void addConvTileAndDecomposeExpertPassPipeline(
    OpPassManager &funcPassManager, const LLVMCPUPipelineOptions &pipelineOpt);

/// Populates the passes needed to multi level tile, fuse and vectorize
/// lowering of linalg ops on tensors to vectors operations.
void addMmt4dTilingExpertPassPipeline(
    OpPassManager &funcPassManager, const LLVMCPUPipelineOptions &pipelineOpt);

void addMultiTilingExpertPassPipeline(
    OpPassManager &funcPassManager,
    IREE::Codegen::LoweringConfigAttrInterface loweringConfig,
    const LLVMCPUPipelineOptions &pipelineOpt);

//----------------------------------------------------------------------------//
// LLVMCPU Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

/// Populates passes needed for preprocessing before codegen lowerings, as well
/// as high level lowering strategy selection.
void buildLLVMCPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager);

/// Populates passes needed to lower high level ops, e.g., linalg, vector, etc,
/// to LLVM dialect via the structured ops path. The  `variantPassManager`
/// should operate on the module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool enableAArch64SME = false);

//----------------------------------------------------------------------------//
// LLVMCPU Linking Passes and Pipelines
//----------------------------------------------------------------------------//

/// Populates passes needed to link HAL executables across LLVMCPU targets.
void buildLLVMCPULinkingPassPipeline(
    OpPassManager &modulePassManager,
    std::optional<std::string> target = std::nullopt);

//----------------------------------------------------------------------------//
// Register LLVMCPU Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc" // IWYU pragma: keep

void registerCodegenLLVMCPUPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_PASSES_H_
