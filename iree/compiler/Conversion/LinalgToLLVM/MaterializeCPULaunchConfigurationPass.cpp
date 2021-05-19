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
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
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

static const unsigned kNumMaxParallelDims = 3;

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

/// Returns the workgroup size to use for the flow.dispatch.workgroups
/// operation, the `linalgOp` is in. This is obtained from the tile size for the
/// operation at the distributed level.
static SmallVector<int64_t, 4> getDistributedWorkgroupSize(
    linalg::LinalgOp linalgOp, ArrayRef<int64_t> tileSizes) {
  if (tileSizes.empty()) return {};
  tileSizes = tileSizes.take_front(getNumOuterParallelLoops(linalgOp));
  // For now we only distribute the inner-parallel loops.
  if (tileSizes.size() > kNumMaxParallelDims) {
    tileSizes = tileSizes.take_back(kNumMaxParallelDims);
  }
  auto workgroupSize = llvm::to_vector<4>(llvm::reverse(tileSizes));
  return workgroupSize;
}

/// Gets the workgroup size to use based on the workgroups sizes set for each
/// operation in `linalgOps`. Checks that all the ops return a consistent
/// value.
static FailureOr<SmallVector<int64_t, 4>> getDistributedWorkgroupSize(
    ArrayRef<linalg::LinalgOp> linalgOps) {
  Optional<SmallVector<int64_t, 4>> distributedWorkgroupSize;
  for (auto linalgOp : linalgOps) {
    SmallVector<int64_t, 4> opTileSizes = getTileSizes(
        linalgOp, static_cast<unsigned>(TilingLevel::WorkGroupTiles));
    auto opWorkgroupSize = getDistributedWorkgroupSize(linalgOp, opTileSizes);
    if (!distributedWorkgroupSize) {
      distributedWorkgroupSize = std::move(opWorkgroupSize);
      continue;
    }
    if (*distributedWorkgroupSize != opWorkgroupSize) {
      return static_cast<LogicalResult>(linalgOp.emitError(
          "mismatch in workgroup size for this op as compared to other linalg "
          "ops in the dispatch region"));
    }
  }
  return distributedWorkgroupSize ? *distributedWorkgroupSize
                                  : SmallVector<int64_t, 4>{};
}

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
    if (failed(initCPULaunchConfig(linalgOps))) {
      return signalPassFailure();
    }

    // Get the workgroup size to use.
    auto distributedWorkgroupSize = getDistributedWorkgroupSize(linalgOps);
    if (failed(distributedWorkgroupSize)) {
      return signalPassFailure();
    }

    // If workgroup size is empty, serialize the operation by setting the number
    // of workgroups to {1, 1, 1}.
    if (distributedWorkgroupSize->empty()) {
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
      return;
    }

    // Materialize the computed workgroup size.
    if (failed(materializeStaticLaunchInformation(
            funcOp, distributedWorkgroupSize.getValue()))) {
      funcOp.emitOpError("failed to materialize static launch information");
      return signalPassFailure();
    }

    // Apply post distribution canonicalization passes.
    {
      OwningRewritePatternList canonicalization(&getContext());
      AffineMinOp::getCanonicalizationPatterns(canonicalization, context);
      populateAffineMinSCFCanonicalizationPattern(canonicalization);
      IREE::Flow::populateFlowDispatchCanonicalizationPatterns(canonicalization,
                                                               context);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(canonicalization));
    }
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
