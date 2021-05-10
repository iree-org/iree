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

/// Gets the tile sizes to use for the distributed dimensions of the first-level
/// of tiling. This is obtained by taking the tile sizes for the outer parallel
/// loops and filtering out the tile sizes set to 0. If there are multiple
/// linalg ops, verifies all of them are consistent. This is a required property
/// for a dispatch region.
static Optional<SmallVector<int64_t, 4>> getDistributedDimsTileSizes(
    ArrayRef<linalg::LinalgOp> linalgOps) {
  Optional<SmallVector<int64_t, 4>> distributedTileSizes;
  for (auto linalgOp : linalgOps) {
    SmallVector<int64_t, 4> opTileSizes = getTileSizes(linalgOp, 0);
    ArrayRef<int64_t> opTileSizesRef(opTileSizes);
    opTileSizes = llvm::to_vector<4>(llvm::make_filter_range(
        opTileSizesRef.take_front(getNumOuterParallelLoops(linalgOp)),
        [](int64_t t) -> bool { return t; }));
    if (!distributedTileSizes) {
      distributedTileSizes = std::move(opTileSizes);
      continue;
    }
    if (*distributedTileSizes != opTileSizes) {
      return llvm::None;
    }
  }
  if (!distributedTileSizes) return SmallVector<int64_t, 4>{};
  return distributedTileSizes;
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
    auto distributedTileSizes = getDistributedDimsTileSizes(linalgOps);
    if (!distributedTileSizes) {
      return signalPassFailure();
    }

    if (failed(materializeStaticLaunchInformation(
            funcOp, llvm::to_vector<4>(
                        llvm::reverse(distributedTileSizes.getValue()))))) {
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
