// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

namespace mlir::iree_compiler::IREE::GPU {

/// Helper for setting up a data tiled multi_mma config based on the specified
/// target.
LogicalResult setDataTiledMultiMmaLoweringConfig(
    IREE::GPU::TargetAttr target, mlir::FunctionOpInterface entryPoint,
    Operation *op, IREE::GPU::UKernelConfigAttr ukernelConfig);

/// Helper for setting up a convolution config using IGEMM based on the
/// specified target.
/// TODO: Currently this only succeeds if the target supports an mma
/// kind. Add support for a fallback direct lowering path.
LogicalResult
setIGEMMConvolutionLoweringConfig(IREE::GPU::TargetAttr target,
                                  mlir::FunctionOpInterface entryPoint,
                                  Operation *op, bool useDirectLoad = false);

/// Helper for setting up a matmul config based on the specified target.
/// TODO: Currently this only succeeds if the target supports an mma
/// kind. Add support for a fallback direct lowering path.
LogicalResult setMatmulLoweringConfig(IREE::GPU::TargetAttr target,
                                      mlir::FunctionOpInterface entryPoint,
                                      Operation *op,
                                      bool useDirectLoad = false);

/// Helper for setting up a default tile and fuse config for targeting
/// simple thread distribution. Currently restricted to linalg ops.
LogicalResult setTileAndFuseLoweringConfig(IREE::GPU::TargetAttr target,
                                           mlir::FunctionOpInterface entryPoint,
                                           Operation *op);

// Helper for setting tile sizes for scatter.
LogicalResult setScatterLoweringConfig(IREE::GPU::TargetAttr target,
                                       mlir::FunctionOpInterface entryPoint,
                                       Operation *op);

LogicalResult setSortConfig(IREE::GPU::TargetAttr target,
                            mlir::FunctionOpInterface entryPoint,
                            Operation *op);

//===----------------------------------------------------------------------===//
// Pass Pipeline Options
//===----------------------------------------------------------------------===//

using IREE::GPU::ReorderWorkgroupsStrategy;

struct GPUPipelineOptions {
  bool enableReduceSharedMemoryBankConflicts = true;
  bool prefetchSharedMemory = false;
  bool useIgemmConvolution = false;
  bool enableUkernels = false;
  std::optional<ReorderWorkgroupsStrategy> reorderStrategy;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const GPUPipelineOptions &options);

GPUPipelineOptions
getPipelineOptions(FunctionOpInterface funcOp,
                   IREE::Codegen::TranslationInfoAttr translationInfo);

} // namespace mlir::iree_compiler::IREE::GPU

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_TARGETUTILS_CONFIGUTILS_H_
