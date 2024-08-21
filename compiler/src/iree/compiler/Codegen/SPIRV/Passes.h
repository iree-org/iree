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

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

//===---------------------------------------------------------------------===//
// SPIR-V pass pipelines
//===---------------------------------------------------------------------===//

/// Pass pipeline to lower IREE HAL executables without any tiling and
/// distribution.
void addSPIRVBaseLoweringPassPipeline(OpPassManager &funcPassManager);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing to
/// workgroups and invocations. Each invocation handles a scalar.
void addSPIRVBaseDistributePassPipeline(OpPassManager &funcPassManager);

void addSPIRVBaseVectorizePassPipeline(OpPassManager &funcPassManager);

/// Adds passes to lower vector ops to meet SPIR-V requirements.
void addSPIRVVectorLoweringPasses(OpPassManager &funcPassManager);

void addSPIRVCooperativeMatrixVectorizePassPipeline(
    OpPassManager &funcPassManager, unsigned pipelineDepth,
    unsigned storeStage);

void addSPIRVMatmulPromoteVectorizePassPipeline(OpPassManager &funcPassManager,
                                                unsigned pipelineDepth,
                                                unsigned storeStage);

/// Pass pipeline to lower IREE HAL executables by tiling and distributing
/// reduction to workgroups and then subgroups.
void addSPIRVSubgroupReducePassPipeline(OpPassManager &funcPassManager);

/// Pass pipeline to lower winograd ops. This pipeline follows the
/// SPIRVBaseVectorize pipeline with the following exception:
/// Since the ops are already tiled, we skip tiling and instead
/// just annotate the loops with the spirv distribute attribute.
///
void addSPIRVWinogradVectorizePassPipeline(OpPassManager &funcPassManager);

/// Populates passes needed to preprocess the input variant before lowering
/// and select lowering strategies.
void buildSPIRVCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager);

/// Populates passes needed to lower linalg/arith/math ops to SPIR-V ops via
/// the structured ops path.
void buildSPIRVCodegenPassPipeline(OpPassManager &variantPassManager);

/// Populates passes needed to link HAL executables across SPIRV targets.
void buildSPIRVLinkingPassPipeline(OpPassManager &modulePassManager);

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

//===---------------------------------------------------------------------===//
// Wrappers that not use tablegen options.
//===---------------------------------------------------------------------===//

/// Pass to perform the final conversion to SPIR-V dialect.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertToSPIRVPass(unsigned indexWidth);

//----------------------------------------------------------------------------//
// Registration
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "iree/compiler/Codegen/SPIRV/Passes.h.inc" // IWYU pragma: keep

void registerCodegenSPIRVPasses();

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_SPIRV_PASSES_H_
