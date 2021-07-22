// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- Passes.cpp - Pipeline from HLO to Linalg to SPIR-V -----------------===//
//
// Implementation of conversion from XLA-HLO to Linalg to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/MemorySpace.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/StandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

void buildLinalgToSPIRVPassPipeline(OpPassManager &pm,
                                    const SPIRVCodegenOptions &options) {
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Tiling, distribution, vectorization
  //===--------------------------------------------------------------------===//

  // flow.dispatch.workgroups performed abstract tiling and distribution. Make
  // them concrete now since we know the target and settings now.
  pm.addPass(createSPIRVConcretizeWorkgroupTilesPass(options));
  // Tile and distribute to GPU subgroups/invocations and vectorize.
  pm.addPass(createSPIRVTileAndVectorizePass(options));
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());

  // Handle ops that cannot go through the previous tiling, distribution, and
  // vectorization flow. Only perform one level of distribution to map them to
  // GPU global invocation IDs for distribution.
  // TODO(antiagainst): Handle all the cases uniformly and remove this pass.
  pm.addPass(createSPIRVConvertToGPUPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createSPIRVVectorToGPUPass());
  pm.nest<ModuleOp>().addPass(createLowerAffinePass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // Optimizations and cleanups
  //===--------------------------------------------------------------------===//

  // Perform various vector-level cross-op optimizations like load-store
  // forwarding, shape casting and casting op cancelling.
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createOptimizeVectorTransferPass());

  // Fold load/store from/to subview ops into the original memref when possible.
  // In SPIR-V we don't use memref descriptor so it's not possible to handle
  // subview ops.
  pm.nest<ModuleOp>().addPass(memref::createFoldSubViewOpsPass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  // Turn scalar load/store from memrefs into vectorized ones if possible. This
  // gives better memory access patterns, which is very important for perf.
  pm.nest<ModuleOp>().addPass(createSPIRVVectorizeLoadStore());
  // Lower vector ops to SPIR-V cooperative matrix ops. This needs to be done
  // before flattening memref because we still need the multi-dimension
  // structure.
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(
      createSPIRVVectorToCooperativeMatrixPass());

  // Perform optimizations that need to across the scf.for region boundary.
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createForOpCanonicalizationPass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  // Turn multi-dimension memref into one-dimension. This is needed for SPIR-V
  // because we don't use upstream memref descriptors.
  pm.nest<ModuleOp>().addPass(createFlattenMemRefSubspanPass());
  pm.nest<ModuleOp>().addPass(createLowerAffinePass());
  pm.nest<ModuleOp>().addPass(createCanonicalizerPass());
  pm.nest<ModuleOp>().addPass(createCSEPass());

  //===--------------------------------------------------------------------===//
  // SPIR-V conversions
  //===--------------------------------------------------------------------===//

  // Finally convert everything to SPIR-V.
  pm.nest<ModuleOp>().addPass(createConvertToSPIRVPass());

  OpPassManager &spirvModulePM = pm.nest<ModuleOp>().nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
  spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

void buildSPIRVCodegenPassPipeline(OpPassManager &pm,
                                   const SPIRVCodegenOptions &options) {
  pm.nest<ModuleOp>().addPass(createInlinerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCleanupBufferAllocViewPass());

  WorkgroupMemoryAllocationFn allocationFn =
      [](OpBuilder &builder, Location loc, ArrayRef<int64_t> staticShape,
         Type elementType, ArrayRef<Value> dynamicSizes) {
        MemRefType allocType = MemRefType::get(staticShape, elementType, {},
                                               getWorkgroupMemorySpace());
        return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
      };
  addLinalgBufferizePasses(pm.nest<ModuleOp>(), allocationFn);

  buildLinalgToSPIRVPassPipeline(pm, options);
}

}  // namespace iree_compiler
}  // namespace mlir
