// Copyright 2020 Google LLC
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

//===- LowerToSPIRV.cpp - Lower from XLA to Linalg to SPIR-V dialect-------===//
//
// Implementation of conversion from XLA-HLO to Linalg to SPIR-V dialect.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/LowerToSPIRV.h"

#include "iree/compiler/Translation/CodegenPasses/Passes.h"
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Passes.h"
#include "mlir/Conversion/GPUToSPIRV/ConvertGPUToSPIRV.h"
#include "mlir/Conversion/LoopsToGPU/LoopsToGPU.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRV.h"
#include "mlir/Conversion/StandardToSPIRV/ConvertStandardToSPIRVPass.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/LoopOps/LoopOps.h"
#include "mlir/Dialect/SPIRV/Passes.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/EDSC/Intrinsics.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// These options are only for testing purposes. For actual execution with IREE,
/// these are computed by IREE/Backends automatically.
struct WorkGroupOptions : public PassPipelineOptions<WorkGroupOptions> {
  ListOption<int64_t> workGroupSize{
      *this, "workgroup-size",
      llvm::cl::desc(
          "Number of workgroups to dispatch for the SPIR-V module; at most "
          "three integers standarding for the x, y, and z dimension; "
          "additional arguments will be ignored (used only for testing)"),
      llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated};
};
}  // namespace

/// Helper function to create a std.constant of index type to initialize the
/// workgroup size as a SSA value.
static void createConstantsInFunc(FuncOp funcOp, ArrayRef<int64_t> intVal,
                                  SmallVectorImpl<Value> &constVal) {
  OpBuilder builder(funcOp.getBody());
  MLIRContext *context = funcOp.getContext();
  for (auto val : intVal) {
    constVal.push_back(builder.create<ConstantOp>(
        funcOp.getLoc(), IntegerAttr::get(IndexType::get(context), val)));
  }
}

namespace {

/// To be able to use the workgroup size from the dispatch function attribute to
/// convert loops to GPU kernel, need to actually implement a pass to retrieve
/// the attribute value from the function and pass it along.
// TODO(ravishankarm): Structure the Loops to GPU pass in MLIR so that we dont
// have to do this. Maybe make it an OpPassBase<loop::ForOp> ?
struct LoopsToGPUPass : public FunctionPass<LoopsToGPUPass> {
  void runOnFunction() override {
    // Get the workgroup size from the attributes.
    FuncOp funcOp = getFunction();
    SmallVector<int64_t, 3> workGroupSizeVec;
    workGroupSizeVec.reserve(3);
    if (failed(getWorkGroupSize(funcOp, workGroupSizeVec))) return;
    ArrayRef<int64_t> workGroupSize = dropTrailingOnes(workGroupSizeVec);

    SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;

    // The number of workgroups passed in the conversion should not be
    // considered while generating the device-side code. To avoid some
    // optimization/folding to kick in, allocate variables for the number of
    // workgroups. This allocated value should not be used within the region of
    // the gpu.launch, but if it is the compilation will fail due to ABI
    // mismatch.
    numWorkGroupsVal.reserve(3);
    OpBuilder builder(funcOp.getBody());
    auto indexType = IndexType::get(funcOp.getContext());
    auto int32Type = IntegerType::get(32, funcOp.getContext());
    auto resultType = MemRefType::get(ArrayRef<int64_t>(), int32Type);
    for (unsigned i = 0, e = workGroupSize.size(); i != e; ++i) {
      auto allocOp = builder.create<AllocOp>(funcOp.getLoc(), resultType);
      auto loadVal = builder.create<LoadOp>(funcOp.getLoc(), allocOp);
      numWorkGroupsVal.push_back(
          builder.create<IndexCastOp>(funcOp.getLoc(), loadVal, indexType));
    }
    workGroupSizeVal.reserve(3);
    createConstantsInFunc(funcOp, workGroupSize, workGroupSizeVal);
    for (Block &block : getFunction())
      for (Operation &op : llvm::make_early_inc_range(block))
        if (auto forOp = dyn_cast<loop::ForOp>(&op))
          if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
                                            workGroupSizeVal)))
            return signalPassFailure();
  }
};

/// To be able to use the workgroup size from the dispatch function attribute to
/// convert GPU kernel into SPIR-V kernel, need to actually implement a pass to
/// retrieve the attribute value from the function and pass it along.
// TODO(ravishankarm): Move this into MLIR core.
struct IREEGPUToSPIRVPass : public ModulePass<IREEGPUToSPIRVPass> {
  void runOnModule() {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getModule();

    auto kernelModules = moduleOp.getOps<gpu::GPUModuleOp>();
    if (kernelModules.empty()) return;
    if (!mlir::has_single_element(kernelModules)) {
      moduleOp.emitError(
          "unhandled multiple gpu modules within a dispatch module");
      return signalPassFailure();
    }
    gpu::GPUModuleOp gpuModule = *kernelModules.begin();
    auto targetAttr = spirv::lookupTargetEnvOrDefault(gpuModule);
    SPIRVTypeConverter typeConverter(targetAttr);
    OwningRewritePatternList patterns;

    populateGPUToSPIRVPatterns(context, typeConverter, patterns);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);

    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(targetAttr);
    target->addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyFullConversion(gpuModule, *target, patterns,
                                   &typeConverter))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

void addLinalgToSPIRVPasses(OpPassManager &pm,
                            ArrayRef<int64_t> workGroupSize) {
  // Linalg to loops.
  pm.addPass(createLinalgTileAndFusePass(workGroupSize));
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(std::make_unique<LoopsToGPUPass>());
  pm.addPass(createIREEGpuKernelOutliningPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLowerAffinePass());

  // GPU to SPIR-V.
  pm.addPass(createLegalizeStdOpsForSPIRVLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(std::make_unique<IREEGPUToSPIRVPass>());

  // SPIR-V passes for lowering attributes.
  OpPassManager &spirvModulePM = pm.nest<spirv::ModuleOp>();
  spirvModulePM.addPass(spirv::createLowerABIAttributesPass());
  spirvModulePM.addPass(createCanonicalizerPass());
  spirvModulePM.addPass(createCSEPass());
  spirvModulePM.addPass(spirv::createUpdateVersionCapabilityExtensionPass());
}

void addHLOToLinalgToSPIRVPasses(OpPassManager &pm,
                                 ArrayRef<int64_t> workGroupSize) {
  addHLOToLinalgOnBuffersPasses(pm);
  addLinalgToSPIRVPasses(pm, workGroupSize);
}

static PassPipelineRegistration<WorkGroupOptions> linalgToSPIRVPipeline(
    "iree-linalg-to-spirv",
    "Runs the progressive lowering pipeline from Linalg to SPIR-V",
    [](OpPassManager &passManager, const WorkGroupOptions &options) {
      SmallVector<int64_t, 2> workGroupSize;
      workGroupSize.assign(options.workGroupSize.begin(),
                           options.workGroupSize.end());
      addLinalgToSPIRVPasses(passManager, workGroupSize);
    });

static PassPipelineRegistration<WorkGroupOptions> xlaToLinalgSPIRVPipeline(
    "iree-xla-to-linalg-to-spirv",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to SPIR-V",
    [](OpPassManager &passManager, const WorkGroupOptions &options) {
      SmallVector<int64_t, 2> workGroupSize;
      workGroupSize.assign(options.workGroupSize.begin(),
                           options.workGroupSize.end());
      addHLOToLinalgToSPIRVPasses(passManager, workGroupSize);
    });
}  // namespace iree_compiler
}  // namespace mlir
