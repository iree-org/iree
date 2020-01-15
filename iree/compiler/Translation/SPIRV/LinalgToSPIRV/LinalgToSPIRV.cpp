// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//===- LinalgToSPIRV.cpp - Linalg dialect to SPIR-V dialect----------------===//
//
// Implementation of conversion from Linalg To SPIRV
//
//===----------------------------------------------------------------------===//
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRVPass.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPUPass.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
struct LinalgToSPIRVPassOptions
    : public PassPipelineOptions<LinalgToSPIRVPassOptions> {
  ListOption<int64_t> numWorkGroups{
      *this, "num-workgroups",
      llvm::cl::desc(
          "Number of workgroups in the SPIR-V module for x, followed by y, "
          "followed by z dimension of the dispatch (others will be ignored)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
  ListOption<int64_t> workGroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Workgroup Sizes in the SPIR-V module for x, followed by y, followed "
          "by z dimension of the dispatch (others will be ignored)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
}  // namespace

static void addLinalgToSPIRVPasses(OpPassManager &pm,
                                   const LinalgToSPIRVPassOptions &options) {
  // TODO(ravishankarm): For now only evaluated with 2D tiling. So set the
  // workgroup size and numworkgroups to size 2
  SmallVector<int64_t, 2> numWorkGroups, workGroupSize;
  numWorkGroups.assign(options.numWorkGroups.begin(),
                       options.numWorkGroups.end());
  numWorkGroups.resize(2, 1);
  workGroupSize.assign(options.workGroupSize.begin(),
                       options.workGroupSize.end());
  workGroupSize.resize(2, 1);

  // Linalg to loops.
  pm.addPass(createLinalgTilingPass(workGroupSize));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Loops to GPU.
  pm.addPass(createLoopToGPUPass(numWorkGroups, workGroupSize));
  pm.addPass(createGpuKernelOutliningPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLowerAffinePass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createConvertGPUToSPIRVPass(workGroupSize));

  // SPIR-V passes for lowering attributes.
  OpPassManager &spirvModulePM = pm.nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
}

static PassPipelineRegistration<LinalgToSPIRVPassOptions> linalgToSPIRVPipeline(
    "iree-linalg-to-spirv",
    "Runs the progressive lowering pipeline from Linalg to SPIR-V",
    [](OpPassManager &passManager, const LinalgToSPIRVPassOptions &options) {
      addLinalgToSPIRVPasses(passManager, options);
    });
}  // namespace iree_compiler
}  // namespace mlir
