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

#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "iree/compiler/Translation/SPIRV/LinalgToSPIRV/Passes.h"
#include "iree/compiler/Translation/XLAToLinalg/Passes.h"
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

/// Gets the number of outer parallel loops in a linalg operation.
unsigned getNumOuterParallelLoops(linalg::LinalgOp linalgOp) {
  // Find the number of leading parallel loops in the generic op
  unsigned numOuterParallelLoops = 0;
  for (auto iteratorType : linalgOp.iterator_types()) {
    if (iteratorType.cast<StringAttr>().getValue() !=
        getParallelIteratorTypeName()) {
      break;
    }
    numOuterParallelLoops++;
  }
  return numOuterParallelLoops;
}

namespace {

/// To be able to use the workgroup size from the dispatch function attribute
/// within the linalg tiling pass, need to actually implement a pass to retrieve
/// the attribute value from the function and pass it along.
// TODO(ravishankarm): Move this into Linalg dialect.
struct IREETileLinalgPass : public FunctionPass<IREETileLinalgPass> {
  void runOnFunction() override {
    FuncOp funcOp = getFunction();
    SmallVector<int64_t, 3> workGroupSizeVec;
    workGroupSizeVec.reserve(3);
    if (failed(getWorkGroupSize(funcOp, workGroupSizeVec))) return;
    ArrayRef<int64_t> workGroupSize = dropTrailingOnes(workGroupSizeVec);

    OpBuilder builder(funcOp);
    OperationFolder folder(funcOp.getContext());
    Region &body = funcOp.getBody();
    if (!mlir::has_single_element(body)) {
      funcOp.emitError(
          "unhandled dispatch function that doesn't have a single block");
      return signalPassFailure();
    }
    auto linalgOps = body.front().getOps<linalg::LinalgOp>();
    if (!mlir::has_single_element(linalgOps)) {
      funcOp.emitError(
          "unhandled SPIR-V code generation with multiple linalg operations");
      return signalPassFailure();
    }
    linalg::LinalgOp linalgOp = *linalgOps.begin();
    if (!linalgOp.hasBufferSemantics()) {
      linalgOp.emitError(
          "expected linalg op with buffer semantics during SPIR-V "
          "code generation");
      return signalPassFailure();
    }

    unsigned numOuterParallelLoops = getNumOuterParallelLoops(linalgOp);
    if (!numOuterParallelLoops) {
      // There are no outer parallel loops to partition. So just create dummy
      // 1-trip loops that will be "split" across workgroups and workitems.
      builder.setInsertionPoint(linalgOp);
      auto indexType = builder.getIndexType();
      auto loc = linalgOp.getLoc();
      auto zero =
          builder.create<ConstantOp>(loc, builder.getIntegerAttr(indexType, 0));
      auto one =
          builder.create<ConstantOp>(loc, builder.getIntegerAttr(indexType, 1));
      auto outerLoop = builder.create<loop::ForOp>(loc, zero, one, one);
      OpBuilder outerLoopBuilder = outerLoop.getBodyBuilder();
      auto innerLoop =
          outerLoopBuilder.create<loop::ForOp>(loc, zero, one, one);
      OpBuilder innerLoopBuilder = innerLoop.getBodyBuilder();
      innerLoopBuilder.clone(*linalgOp.getOperation());
      linalgOp.erase();
      return;
    }

    // Tile sizes to use are reverse of the workGroupSize.
    SmallVector<int64_t, 3> tileSizes(reverse(workGroupSize));
    // Linalg convention is to use 0 for no tiling. If the workgroup size is
    // 1, then dont tile along that dimension. So overriding 1 to 0.
    for (auto &tileSize : tileSizes)
      if (tileSize == 1) tileSize = 0;
    tileSizes.resize(numOuterParallelLoops, 0);
    if (linalg::tileLinalgOp(builder, linalgOp, tileSizes, {}, &folder)) {
      linalgOp.erase();
      return;
    }
    linalgOp.emitError("unable to tile linalg op for SPIR-V code generation");
    return signalPassFailure();
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
    SmallVector<int64_t, 3> workGroupSizeVec;
    workGroupSizeVec.reserve(3);
    if (failed(getWorkGroupSize(funcOp, workGroupSizeVec))) return;
    ArrayRef<int64_t> workGroupSize = dropTrailingOnes(workGroupSizeVec);

    // For now just use number of workgroups to be [1, 1, 1]. The loop to GPU
    // lowering doesnt use the value of number of workgroups in the codegen
    // itself, but rather only uses this in the gpu.launch op which is
    // irrelevant for IREE.
    // TODO(ravishankarm): Fix the GPU lowering to allow not using gpu.launch at
    // all.
    SmallVector<int64_t, 3> numWorkGroups(workGroupSize.size(), 1);

    SmallVector<Value, 3> numWorkGroupsVal, workGroupSizeVal;
    numWorkGroupsVal.reserve(3);
    workGroupSizeVal.reserve(3);
    createConstantsInFunc(funcOp, numWorkGroups, numWorkGroupsVal);
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

    // Need to get the workgroup size from the original function.
    // TODO(b/150312935): When the usage of attributes on the function is
    // dropped we might not need to do this.
    FuncOp funcOp = nullptr;
    auto walkResult = moduleOp.walk([&funcOp](FuncOp fOp) -> WalkResult {
      if (isDispatchFunction(fOp)) {
        if (funcOp) return WalkResult::interrupt();
        funcOp = fOp;
      }
      return WalkResult::advance();
    });
    if (!funcOp || walkResult.wasInterrupted()) {
      moduleOp.emitError("expected a single dispatch function within module");
      return signalPassFailure();
    }
    SmallVector<int32_t, 3> workGroupSize;
    if (failed(getWorkGroupSize(funcOp, workGroupSize))) return;

    auto kernelModules = moduleOp.getOps<gpu::GPUModuleOp>();
    SPIRVTypeConverter typeConverter;
    OwningRewritePatternList patterns;

    // Set spv.entry_point_abi on each kernel functions to drive SPIR-V CodeGen.
    // This is required because SPIR-V CodeGen's contract.
    // TODO(ravishankarm/antiagainst) : When there is a mirror of the
    // workgroup-size attribute in gpu dialect use that instad.
    StringRef abiAttrName = spirv::getEntryPointABIAttrName();
    auto abiAttr = spirv::getEntryPointABIAttr(workGroupSize, context);
    for (Operation *gpuModule : kernelModules)
      gpuModule->walk([abiAttrName, abiAttr](gpu::GPUFuncOp gpuFunc) {
        gpuFunc.setAttr(abiAttrName, abiAttr);
      });

    populateGPUToSPIRVPatterns(context, typeConverter, patterns);
    populateStandardToSPIRVPatterns(context, typeConverter, patterns);

    std::unique_ptr<ConversionTarget> target =
        spirv::SPIRVConversionTarget::get(
            spirv::lookupTargetEnvOrDefault(funcOp), context);
    target->addDynamicallyLegalOp<FuncOp>([&](FuncOp op) {
      return typeConverter.isSignatureLegal(op.getType());
    });

    SmallVector<Operation *, 1> targetOps(kernelModules.begin(),
                                          kernelModules.end());
    if (failed(applyFullConversion(targetOps, *target, patterns,
                                   &typeConverter))) {
      return signalPassFailure();
    }
  }
};

/// Pass to override the workgroup_size attribute value of a dispatch function.
struct UpdateWorkGroupSizePass : FunctionPass<UpdateWorkGroupSizePass> {
  UpdateWorkGroupSizePass(ArrayRef<int64_t> workGroupSize)
      : workGroupSize(workGroupSize.begin(), workGroupSize.end()) {}
  void runOnFunction() {
    FuncOp funcOp = getFunction();
    if (!isDispatchFunction(funcOp)) return;

    if (workGroupSize.empty()) {
      // By default look at the number of "parallel" loops in the generic op.
      Region &body = funcOp.getBody();
      // Only handle single block functions.
      if (body.getBlocks().size() != 1) {
        funcOp.emitError("unhandled dispatch function with multiple blocks");
        return signalPassFailure();
      }
      Block &block = body.front();
      auto linalgOps = block.getOps<linalg::LinalgOp>();
      if (!mlir::has_single_element(linalgOps)) {
        funcOp.emitError(
            "unhandled SPIR-V code-generation with multiple linalg ops in "
            "dispatch region");
        return signalPassFailure();
      }
      // Find the number of leading parallel loops in the generic op
      unsigned numOuterParallelLoops =
          getNumOuterParallelLoops(*linalgOps.begin());
      workGroupSize.resize(3, 1);
      if (numOuterParallelLoops > 0) {
        workGroupSize[0] = 32;
      }
      if (numOuterParallelLoops > 1) {
        workGroupSize[1] = 4;
      }
      if (numOuterParallelLoops > 2) {
        // Change workGroupsSize[1] such that the total size is equal to 128,
        // which is the minimum gauranteed by Vulkan spec.
        workGroupSize[1] = 2;
        workGroupSize[2] = 2;
      }
      // TODO(ravishankarm): The current code-generation will "serialize" all
      // the inner loops that are more than 3 deep. We can potentially "fold"
      // all the parallel loops so that they all executed on different
      // workitems.
    }
    OpBuilder builder(&getContext());
    funcOp.setAttr("iree.executable.workgroup_size",
                   getDenseIntElementsAttrVal(&builder, workGroupSize));
  }

 private:
  SmallVector<int64_t, 3> workGroupSize;
};
}  // namespace

void addLinalgToSPIRVPasses(OpPassManager &pm,
                            ArrayRef<int64_t> workGroupSize) {
  // Linalg to loops.
  pm.addPass(std::make_unique<UpdateWorkGroupSizePass>(workGroupSize));
  pm.addPass(std::make_unique<IREETileLinalgPass>());
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
}

void addLowerToSPIRVPasses(OpPassManager &pm, ArrayRef<int64_t> workGroupSize) {
  pm.addPass(createXLAToLinalgPass());
  pm.addPass(createLinalgFusionPass());
  pm.addPass(createLinalgTensorToBufferConversionPass());
  addLinalgToSPIRVPasses(pm);
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
      addLowerToSPIRVPasses(passManager, workGroupSize);
    });
}  // namespace iree_compiler
}  // namespace mlir
