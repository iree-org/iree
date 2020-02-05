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

#include "iree/compiler/Translation/XLAToLinalg/LinalgTensorToBuffer.h"
#include "iree/compiler/Utils/IREECodegenUtils.h"
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
#include "mlir/Dialect/StandardOps/Ops.h"
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

static DenseIntElementsAttr getDenseIntElementsAttrVal(
    Builder *builder, ArrayRef<int64_t> value) {
  SmallVector<int32_t, 3> vector;
  vector.reserve(3);
  for (auto val : value) {
    vector.emplace_back(val);
  }
  vector.resize(3, 1);
  return builder->getI32VectorAttr(vector);
}

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

/// To be able to use the workgroup size from the dispatch function attribute
/// within the linalg tiling pass, need to actually implement a pass to retrieve
/// the attribute value from the function and pass it along.
// TODO(ravishankarm): Move this into Linalg dialect.
struct IREETileLinalgPass : public FunctionPass<IREETileLinalgPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    SmallVector<int64_t, 3> workGroupSize;
    workGroupSize.reserve(3);
    if (failed(getLegacyWorkGroupSize(funcOp, workGroupSize))) {
      return;
    }
    OpBuilder builder(funcOp);
    OperationFolder folder(funcOp.getContext());
    funcOp.walk([&workGroupSize, &builder, &folder](linalg::LinalgOp op) {
      if (!op.hasBufferSemantics()) {
        return;
      }
      SmallVector<int64_t, 3> tileSizes;
      auto nLoops = op.getNumLoops();
      tileSizes.assign(workGroupSize.begin(), workGroupSize.end());
      // Linalg convention is to use 0 for no tiling. If the workgroup size is
      // 1, then dont tile along that dimension. So overriding 1 to 0.
      for (auto &tileSize : tileSizes) {
        if (tileSize == 1) tileSize = 0;
      }
      tileSizes.resize(nLoops, 0);
      auto tiledOp = linalg::tileLinalgOp(builder, op, tileSizes, {}, &folder);
      if (tiledOp) {
        op.erase();
      }
    });
  }
};

/// To be able to use the workgroup size from the dispatch function attribute to
/// convert loops to GPU kernel, need to actually implement a pass to retrieve
/// the attribute value from the function and pass it along.
// TODO(ravishankarm): Structure the Loops to GPU pass in MLIR so that we dont
// have to do this. Maybe make it an OpPassBase<loop::ForOp> ?
struct LoopsToGPUPass : public FunctionPass<LoopsToGPUPass> {
  void runOnFunction() override {
    // Get the workgroup size from the attributes.
    FuncOp funcOp = getFunction();
    SmallVector<int64_t, 3> workGroupSize;
    workGroupSize.reserve(3);
    if (failed(getLegacyWorkGroupSize(funcOp, workGroupSize))) {
      return;
    }
    // TODO(ravishankarm): Currently evaluating only 2D tiling. Generalize this.
    workGroupSize.resize(2);
    // The Loop To GPU pass expects the numWorkGroups only to create the
    // host-side launch operation. We don't care about that, so just pass {1, 1,
    // 1} for that.
    SmallVector<int64_t, 3> numWorkGroups(workGroupSize.size(), 1);
    SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;
    numWorkGroupsVal.reserve(3);
    workGroupSizeVal.reserve(3);
    createConstantsInFunc(funcOp, numWorkGroups, numWorkGroupsVal);
    createConstantsInFunc(funcOp, workGroupSize, workGroupSizeVal);
    for (Block &block : getFunction()) {
      for (Operation &op : llvm::make_early_inc_range(block)) {
        if (auto forOp = dyn_cast<loop::ForOp>(&op)) {
          if (failed(convertLoopToGPULaunch(forOp, numWorkGroupsVal,
                                            workGroupSizeVal))) {
            return signalPassFailure();
          }
        }
      }
    }
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
    FuncOp funcOp = nullptr;
    auto walkResult = moduleOp.walk([&funcOp](FuncOp fOp) -> WalkResult {
      if (fOp.getAttr("iree.executable.export")) {
        if (funcOp) {
          return WalkResult::interrupt();
        }
        funcOp = fOp;
      }
      return WalkResult::advance();
    });
    if (!funcOp || walkResult.wasInterrupted()) {
      moduleOp.emitError("expected a single dispatch function within module");
      return signalPassFailure();
    }
    SmallVector<Operation *, 1> kernelModules;
    OpBuilder builder(context);
    builder.setInsertionPoint(funcOp.getOperation());

    // Clone the GPU module into the funcop to convert into a SPIR-V module.
    funcOp.walk(
        [&builder, &moduleOp, &kernelModules](gpu::LaunchFuncOp gpuLaunchOp) {
          auto kernelModuleName = gpuLaunchOp.getKernelModuleName();
          auto gpuModuleOp =
              moduleOp.lookupSymbol<gpu::GPUModuleOp>(kernelModuleName);
          kernelModules.push_back(builder.clone(*gpuModuleOp.getOperation()));
        });
    SPIRVTypeConverter typeConverter;
    OwningRewritePatternList patterns;
    SmallVector<int64_t, 3> workGroupSize;
    if (failed(getLegacyWorkGroupSize(funcOp, workGroupSize))) {
      return;
    }
    populateGPUToSPIRVPatterns(context, typeConverter, patterns, workGroupSize);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);

    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(
            spirv::lookupTargetEnvOrDefault(funcOp), context);
    target->addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    if (failed(applyFullConversion(kernelModules, *target, patterns,
                                   &typeConverter))) {
      return signalPassFailure();
    }
  }
};

/// Pass to override the workgroup_size attribute value of a dispatch function.
// TODO(ravishankarm): Use a more cohorent strategy than just setting it to {2,
// 2}.
struct UpdateWorkGroupSizePass : FunctionPass<UpdateWorkGroupSizePass> {
  UpdateWorkGroupSizePass(ArrayRef<int64_t> workGroupSize)
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}
  void runOnFunction() {
    FuncOp funcOp = getFunction();
    if (!funcOp.getAttr("iree.executable.export")) {
      return;
    }
    if (workGroupSize.empty()) {
      workGroupSize = {2, 2};
    }
    workGroupSize.resize(3, 1);
    OpBuilder builder(&getContext());
    funcOp.setAttr("iree.executable.workgroup_size",
                   getDenseIntElementsAttrVal(&builder, workGroupSize));
  }

 private:
  SmallVector<int64_t, 3> workGroupSize;
};
}  // namespace

static void addLinalgToSPIRVPasses(OpPassManager &pm) {
  // Linalg to loops.
  pm.addPass(std::make_unique<IREETileLinalgPass>());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createLowerAffinePass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  pm.addPass(std::make_unique<LoopsToGPUPass>());
  pm.addPass(createGpuKernelOutliningPass());
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
}

void addLowerToSPIRVPasses(OpPassManager &pm, ArrayRef<int64_t> workGroupSize) {
  pm.addPass(xla_hlo::createLegalizeHloToLinalgPass());
  pm.addPass(createLinalgTensorToBufferConversionPass());
  pm.addPass(std::make_unique<UpdateWorkGroupSizePass>(workGroupSize));
  addLinalgToSPIRVPasses(pm);
}

static PassPipelineRegistration<WorkGroupOptions> xlaToLinalgSPIRVPipeline(
    "iree-xla-to-linalg-to-spirv",
    "Runs the progressive lowering pipeline from XLA HLO to Linalg to SPIR-V",
    [](OpPassManager &passManager, const WorkGroupOptions &options) {
      SmallVector<int64_t, 2> workGroupSize;
      workGroupSize.assign(options.workGroupSize.begin(),
                           options.workGroupSize.end());
      addLowerToSPIRVPasses(passManager, workGroupSize);
    });
}  // namespace iree_compiler
}  // namespace mlir
