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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/CodegenUtils/MarkerUtils.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-linalg-to-llvm-tile-and-distribute"

namespace mlir {
namespace iree_compiler {

namespace {
class MaterializeCPULaunchConfigurationPass
    : public PassWrapper<MaterializeCPULaunchConfigurationPass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::HAL::HALDialect, AffineDialect,
                    scf::SCFDialect>();
  }

  MaterializeCPULaunchConfigurationPass() = default;
  MaterializeCPULaunchConfigurationPass(
      const MaterializeCPULaunchConfigurationPass &pass) {}
  void runOnOperation() override;
};
}  // namespace

void MaterializeCPULaunchConfigurationPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IREE::HAL::ExecutableTargetOp targetOp = getOperation();
  ModuleOp module = targetOp.getInnerModule();

  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    SmallVector<linalg::LinalgOp, 4> linalgOps;
    SmallVector<Operation *, 4> tiledLoops;
    if (failed(getLinalgOps(funcOp, linalgOps, tiledLoops))) {
      // Nothing to do here. Continue.
      continue;
    }
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOps);
    Optional<LaunchConfig> launchConfigOpt =
        initCPULaunchConfig(context, dependenceGraph, linalgOps);
    if (!launchConfigOpt) {
      // Nothing to do here. Continue.
      continue;
    }
    LaunchConfig &launchConfig = *launchConfigOpt;

    LLVM_DEBUG({
      llvm::dbgs() << "@func " << funcOp.getName() << "\n";
      for (auto op : linalgOps) {
        llvm::dbgs() << "\t" << op.getOperation()->getName() << " : ";
        TileSizesListTypeRef configTileSizes = launchConfig.getTileSizes(op);
        llvm::dbgs() << "{";
        std::string sep = "";
        for (auto &level : enumerate(configTileSizes)) {
          llvm::dbgs() << sep << level.index() << " : [";
          sep = ", ";
          interleaveComma(level.value(), llvm::dbgs());
          llvm::dbgs() << "]";
        }
        llvm::dbgs() << "}\n";
      }
    });

    // Find the root operation for the dispatch region and get the tile sizes.
    Operation *rootOperation =
        launchConfig.getRootOperation(llvm::to_vector<4>(llvm::map_range(
            linalgOps, [](linalg::LinalgOp op) { return op.getOperation(); })));
    if (!rootOperation) {
      // By default just set the number of workgroups to be {1, 1, 1}.
      WorkgroupCountRegionBuilder regionBuilder =
          [](OpBuilder &b, Location loc,
             std::array<Value, 3> workload) -> std::array<Value, 3> {
        Value one = b.create<ConstantIndexOp>(loc, 1);
        return {one, one, one};
      };
      OpBuilder builder(context);
      if (failed(defineWorkgroupCountRegion(builder, funcOp, regionBuilder))) {
        return signalPassFailure();
      }
      launchConfig.finalize(funcOp);
      return;
    }

    ArrayRef<int64_t> rootOperationTileSizes =
        launchConfig.getTileSizes(rootOperation, 0);
    // Only use the tile sizes for parallel loops of the root operation.
    rootOperationTileSizes = rootOperationTileSizes.take_front(
        getNumOuterParallelLoops(rootOperation));

    SmallVector<int64_t, 4> workloadPerWorkgroup =
        llvm::to_vector<4>(llvm::reverse(rootOperationTileSizes));
    if (failed(
            materializeStaticLaunchInformation(funcOp, workloadPerWorkgroup))) {
      funcOp.emitOpError("failed to materialize static launch information");
      return signalPassFailure();
    }

    launchConfig.finalize(funcOp);
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createMaterializeCPULaunchConfigurationPass() {
  return std::make_unique<MaterializeCPULaunchConfigurationPass>();
}

static PassRegistration<MaterializeCPULaunchConfigurationPass> pass(
    "iree-codegen-llvm-materialize-launch-configuration",
    "Tile and distribute Linalg operations on buffers",
    [] { return std::make_unique<MaterializeCPULaunchConfigurationPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
