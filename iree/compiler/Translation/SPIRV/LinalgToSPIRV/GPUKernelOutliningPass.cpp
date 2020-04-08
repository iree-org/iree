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

//===- GPUKernelOutliningPass.cpp - Generate GPU device-side code ---------===//
//
// Implements a pass to convert a launch operation into a device-side code. Uses
// a separate pass since the pass from core puts the gpu.module at the module
// scope instead of allowing where to put it. Since we dont need the host-side
// aspects of the GPU dialect, a separate pass is used here that only cares
// about the device-side.
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/CodegenUtils/CodegenUtils.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/GPU/Utils.h"
#include "mlir/Dialect/SPIRV/TargetAndABI.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace iree_compiler {

namespace {
// Pattern to get the gpu.GPUModuleOp from the gpu.LaunchOp.
struct ConvertToGPUFuncOp : public OpRewritePattern<gpu::LaunchOp> {
  using OpRewritePattern<gpu::LaunchOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(gpu::LaunchOp launchOp,
                                PatternRewriter &rewriter) const final;
};

// Pass to outline the region of the gpu.LaunchOp.
class IREEGpuKernelOutliningPass
    : public PassWrapper<IREEGpuKernelOutliningPass, OperationPass<ModuleOp>> {
 public:
  void runOnOperation() override;
};
}  // namespace

LogicalResult ConvertToGPUFuncOp::matchAndRewrite(
    gpu::LaunchOp launchOp, PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  auto funcOp = launchOp.getParentOfType<FuncOp>();
  SmallVector<int32_t, 3> workGroupSize;
  if (failed(getWorkGroupSize(funcOp, workGroupSize))) return failure();

  if (failed(sinkOperationsIntoLaunchOp(launchOp))) return failure();

  // The arguments of the funcOp must be the arguments of the launchOp, in the
  // same order.
  SmallVector<Value, 4> arguments(funcOp.args_begin(), funcOp.args_end());
  Optional<StringRef> dispatchFnName = getDispatchFuncName(funcOp);
  if (!dispatchFnName)
    return launchOp.emitError("unable to get dispatch function name");
  gpu::GPUFuncOp gpuFuncOp =
      outlineKernelFunc(launchOp, dispatchFnName.getValue(), arguments);

  // Add the SPIR-V ABI attr here since it is needed for the SPIR-V lowering.
  // TODO(ravishankarm/antiagainst) : When there is a mirror of the
  // workgroup-size attribute in GPU dialect use that instead.
  spirv::EntryPointABIAttr abiAttr = spirv::lookupEntryPointABI(launchOp);
  StringRef abiAttrName = spirv::getEntryPointABIAttrName();
  gpuFuncOp.setAttr(abiAttrName, abiAttr);

  // If any additional arguments are needed, then the launch op cannot be
  // converted.
  if (arguments.size() != gpuFuncOp.getNumArguments()) return failure();

  // Wrap this within a gpu.module
  rewriter.setInsertionPoint(funcOp);
  std::string moduleName = Twine(funcOp.getName(), "_gpumodule").str();
  auto kernelModule =
      rewriter.create<gpu::GPUModuleOp>(funcOp.getLoc(), moduleName);
  SymbolTable symbolTable(kernelModule);
  symbolTable.insert(gpuFuncOp);

  // Set the conversion target attributes on the GPU module.
  auto targetEnvAttrName = spirv::getTargetEnvAttrName();
  kernelModule.setAttr(targetEnvAttrName,
                       spirv::lookupTargetEnvOrDefault(funcOp));

  rewriter.eraseOp(launchOp);
  return success();
}

void IREEGpuKernelOutliningPass::runOnOperation() {
  OwningRewritePatternList patterns;
  ModuleOp moduleOp = getOperation();
  SmallVector<gpu::LaunchOp, 1> gpuLaunchOp;
  moduleOp.walk(
      [&gpuLaunchOp](gpu::LaunchOp op) { gpuLaunchOp.push_back(op); });
  if (!mlir::has_single_element(gpuLaunchOp)) {
    moduleOp.emitError(
        "expected single gpu.launch operation within translation module");
    return signalPassFailure();
  }
  patterns.insert<ConvertToGPUFuncOp>(moduleOp.getContext());
  applyPatternsGreedily(moduleOp.getOperation(), patterns);
}

std::unique_ptr<OperationPass<ModuleOp>> createIREEGpuKernelOutliningPass() {
  return std::make_unique<IREEGpuKernelOutliningPass>();
}
}  // namespace iree_compiler
}  // namespace mlir
