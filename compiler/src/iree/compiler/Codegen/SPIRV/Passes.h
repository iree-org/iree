// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the SPIR-V Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_CODEGEN_SPIRV_PASSES_H_
#define IREE_COMPILER_CODEGEN_SPIRV_PASSES_H_

#include "iree/compiler/Codegen/Dialect/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

/// Pass pipeline to lower IREE HAL executables without any tiling and
/// distribution.
void addSPIRVBaseLoweringPassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups and invocations. Each invocation handles a scalar.
void addSPIRVBaseDistributePassPipeline(OpPassManager &pm);

void addSPIRVBaseVectorizePassPipeline(OpPassManager &pm);

void addSPIRVCooperativeMatrixVectorizePassPipeline(OpPassManager &pm,
                                                    unsigned pipelineDepth,
                                                    unsigned storeStage);

void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &pm,
                                                unsigned pipelineDepth,
                                                unsigned storeStage);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing
/// reduction to workgroups and then subgroups.
void addSPIRVSubgroupReducePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables via transform dialect schedules.
void addSPIRVTransformDialectPassPipeline(OpPassManager &pm);

/// Pass pipeline to lower winograd ops. This pipeline follows the
/// SPIRVBaseVectorize pipeline with the following exception:
/// Since the ops are already tiled, we skip tiling and instead
/// just annotate the loops with the spirv distribute attribute.
///
void addSPIRVWinogradVectorizePassPipeline(OpPassManager &pm);

/// Populates passes needed to preprocess the input variant before lowering
/// and select lowering strategies.
void buildSPIRVCodegenConfigurationPassPipeline(OpPassManager &pm);

/// Populates passes needed to lower linalg/arith/math ops to SPIR-V ops via
/// the structured ops path. The pass manager `pm` here operate on the module
/// within the IREE::HAL::ExecutableOp.
void buildSPIRVCodegenPassPipeline(OpPassManager &pm, bool enableFastMath);

/// Pass to perform the final conversion to SPIR-V dialect.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertToSPIRVPass(bool enableFastMath = false, unsigned indexWidth = 32);

/// Annotates the innermost Winograd loops with the spirv distribute attribute.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVAnnotateWinogradLoopsPass();

/// Breaks down large vectors not natively supported by SPIR-V.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVBreakDownLargeVectorPass();

// Uses `tensor.pad` ops as anchors to create separate fast and slow paths
// inside the kernel. The fast path is for inner tiles where we don't need
// padding, while the slow path is for boundary tiles where we do need
// padding.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVCreateFastSlowPathPass();

/// Pass to distribute tiled loop nests to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVDistributePass();

/// Emulates bfloat 16 ops with 32-bit float ops.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVEmulateBf16Pass();

/// Emulates 64-bit integer ops with 32-bit integer ops.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVEmulateI64Pass();

/// Turns static shaped storage buffer subspan ops into dynamic shaped ones.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVEraseStorageBufferStaticShapePass();

/// Pass to perform final vector ops lowering to meet SPIR-V requirements.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFinalVectorLoweringPass();

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFoldProcessorIDUsesPass();

/// Pass to perform initial vector ops lowering to meet SPIR-V requirements.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVInitialVectorLoweringPass();

/// Links SPIR-V HAL executables within the top-level program module.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createSPIRVLinkExecutablesPass();

/// Populates passes needed to link HAL executables across SPIRV targets.
void buildSPIRVLinkingPassPipeline(OpPassManager &passManager);

/// Pass to set the lowering strategy for the target variant.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVSelectLoweringStrategyPass();

/// Main pass to lower executables to scalar + vector code on SPIR-V path.
/// Invokes one of the pass pipelines that translate the executable to
/// scalar + vector code.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass();

/// Pass to map MemRef memory spaces to SPIR-V storage classes.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVMapMemRefStorageClassPass();

/// Pass to tile and distribute Linalg ops with buffer semantics to
/// invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndDistributePass();

/// Pass to promote Linalg ops with buffer semantics to use workgroup memory
/// and then tile to invocations.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileAndPromotePass(bool promoteCMatrix = false,
                              bool skipThreadLevel = false);

/// Pass to tile Linalg ops with tensor semantics to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTilePass();

/// Pass to tile Linalg ops with buffer semantics suitable for lowering to
/// SPIR-V cooperative ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileToCooperativeOpsPass();

/// Converts vector ops to gpu subgroup MMA ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorToGPUSubgroupMMAOpsPass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVVectorizeLoadStore();

/// Pass to do vectorization suitable for lowering to SPIR-V cooperative ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorizeToCooperativeOpsPass();

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups and invocations and vectorizing. Each invocation handles a
/// vector.
LogicalResult verifySPIRVBaseVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing
/// to workgroups and subgroups and then vectorizing to SPIR-V cooperative
/// matrix code.
LogicalResult verifySPIRVCooperativeMatrixVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups, promoting to use workgroup memory, and then tiling and
/// distributing to invocations and vectorizing. Each invocation handles a
/// vector.
LogicalResult verifySPIRVMatmulPromoteVectorizePassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);

//----------------------------------------------------------------------------//
// Register SPIR-V Passes
//----------------------------------------------------------------------------//

void registerCodegenSPIRVPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_SPIRV_PASSES_H_
