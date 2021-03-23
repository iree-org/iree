// Copyright 2021 Google LLC
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

//===- LinalgTileAndDistributePass.cpp ------------------------------------===//
//
// This pass tiles and distributes linalg operations among multiple workgroups.
//
// NOTE: Deprecated. This pass is used for the first-level tiling in the Linalg
// on buffers path, which is expected to go away soon.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/CodeGenOptionUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/KernelDispatchUtils.h"
#include "iree/compiler/Conversion/LinalgToSPIRV/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-spirv-tile-and-distribute"

namespace mlir {
namespace iree_compiler {
namespace {

/// Returns the distribution options for operations when targeting workgroups.
linalg::LinalgLoopDistributionOptions getWorkgroupDistributionOptions() {
  linalg::LinalgLoopDistributionOptions options;

  options.procInfo = [](OpBuilder &builder, Location loc,
                        ArrayRef<Range> parallelLoopRanges) {
    return getGPUProcessorIdsAndCounts<gpu::BlockIdOp, gpu::GridDimOp>(
        builder, loc, parallelLoopRanges.size());
  };
  options.distributionMethod.assign(
      3, linalg::DistributionMethod::CyclicNumProcsEqNumIters);

  return options;
}

class LinalgTileAndDistributePass
    : public PassWrapper<LinalgTileAndDistributePass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  LinalgTileAndDistributePass(const SPIRVCodegenOptions &options)
      : options(options) {}
  LinalgTileAndDistributePass(const LinalgTileAndDistributePass &that)
      : options(that.options) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, IREE::HAL::HALDialect, linalg::LinalgDialect,
                    scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    IREE::HAL::ExecutableTargetOp targetOp = getOperation();
    ModuleOp module = targetOp.getInnerModule();

    for (FuncOp funcOp : module.getOps<FuncOp>()) {
      if (!isEntryPoint(funcOp)) continue;

      SmallVector<linalg::LinalgOp, 4> linalgOps;
      SmallVector<Operation *, 4> tiledLoops;

      if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
        return signalPassFailure();
      }

      linalg::Aliases aliases;
      linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
      Optional<LaunchConfig> launchConfigOpt =
          initGPULaunchConfig(context, dependenceGraph, options, linalgOps);
      if (!launchConfigOpt) {
        funcOp.emitError("unable to find launch configuration");
        return signalPassFailure();
      }
      LaunchConfig &launchConfig = *launchConfigOpt;

      LLVM_DEBUG({
        llvm::dbgs()
            << "\n--- IREE Linalg tile and distribute configuration ---\n";
        llvm::dbgs() << "@func " << funcOp.getName()
                     << ": # workgroup sizes: [";
        interleaveComma(launchConfig.getWorkgroupSize(), llvm::dbgs());
        llvm::dbgs() << "]\n";
        for (auto op : linalgOps) {
          llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
          TileSizesListTypeRef tileSizes = launchConfig.getTileSizes(op);
          llvm::dbgs() << "{";
          std::string sep = "";
          for (auto &level : enumerate(tileSizes)) {
            llvm::dbgs() << sep << level.index() << " : [";
            sep = ", ";
            interleaveComma(level.value(), llvm::dbgs());
            llvm::dbgs() << "]";
          }
          llvm::dbgs() << "}\n";
        }
      });
      // Annotate the linalg op with the original types.
      for (linalg::LinalgOp op : linalgOps) {
        const char inputTypeAttrName[] = "iree.codegen.original_input_types";
        const char outputTypeAttrName[] = "iree.codegen.original_output_types";

        SmallVector<Type> inputTypes;
        SmallVector<Type> outputTypes;
        for (Type type : op.getInputBufferTypes()) inputTypes.push_back(type);
        for (Type type : op.getOutputBufferTypes()) outputTypes.push_back(type);
        if (!inputTypes.empty()) {
          op->setAttr(inputTypeAttrName,
                      Builder(op).getTypeArrayAttr(inputTypes));
        }
        if (!outputTypes.empty()) {
          op->setAttr(outputTypeAttrName,
                      Builder(op).getTypeArrayAttr(outputTypes));
        }
      }
      TileAndFuseOptions tileAndFuseOptions = {
          getWorkgroupDistributionOptions(), allocateWorkgroupMemory};
      if (failed(tileAndFuseLinalgBufferOps(funcOp, linalgOps, dependenceGraph,
                                            launchConfig,
                                            tileAndFuseOptions)) ||
          failed(
              updateWorkGroupSize(funcOp, launchConfig.getWorkgroupSize()))) {
        return signalPassFailure();
      }
    }
  }

 private:
  SPIRVCodegenOptions options;
};

}  // namespace

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createTileAndDistributeAmongWorkgroupsPass(const SPIRVCodegenOptions &options) {
  return std::make_unique<LinalgTileAndDistributePass>(options);
}

static PassRegistration<LinalgTileAndDistributePass> pass(
    "iree-codegen-spirv-linalg-tile-and-distribute",
    "Tile and distribute Linalg operations on buffers", [] {
      SPIRVCodegenOptions options = getSPIRVCodegenOptionsFromClOptions();
      return std::make_unique<LinalgTileAndDistributePass>(options);
    });

}  // namespace iree_compiler
}  // namespace mlir
