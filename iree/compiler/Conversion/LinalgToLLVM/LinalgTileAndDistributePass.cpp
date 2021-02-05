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
#include "iree/compiler/Conversion/CodegenUtils/GetNumWorkgroups.h"
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
struct LinalgTileAndDistributePass
    : public PassWrapper<LinalgTileAndDistributePass,
                         OperationPass<IREE::HAL::ExecutableTargetOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::HAL::HALDialect, AffineDialect,
                    scf::SCFDialect>();
  }
  LinalgTileAndDistributePass() = default;
  LinalgTileAndDistributePass(const LinalgTileAndDistributePass &pass) {}
  void runOnOperation() override;
};
}  // namespace

Optional<Value> allocateThreadLocalMemory(OpBuilder &b, SubViewOp subview,
                                          ArrayRef<Value> boundingSubViewSize,
                                          OperationFolder *folder) {
  // Allocate the memory into the entry block of the parent FuncOp. This better
  // aligns with the semantics of this memory which is available at the entry of
  // the function.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview->getParentOfType<FuncOp>();
  if (!funcOp) {
    subview.emitError("expected op to be within std.func");
    return llvm::None;
  }
  b.setInsertionPointToStart(&(*funcOp.getBody().begin()));
  // The bounding subview size is expected to be constant. This specified the
  // shape of the allocation.
  SmallVector<int64_t, 2> shape = llvm::to_vector<2>(
      llvm::map_range(boundingSubViewSize, [](Value v) -> int64_t {
        APInt value;
        if (matchPattern(v, m_ConstantInt(&value))) return value.getSExtValue();
        return -1;
      }));
  if (llvm::any_of(shape, [](int64_t v) { return v == -1; })) return {};
  MemRefType allocType =
      MemRefType::get(shape, subview.getType().getElementType());
  Value buffer = b.create<AllocaOp>(subview.getLoc(), allocType);
  return buffer;
}

void LinalgTileAndDistributePass::runOnOperation() {
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
        initCPULaunchConfig(context, dependenceGraph, linalgOps);
    if (!launchConfigOpt) {
      funcOp.emitError("unable to find launch configuration");
      return signalPassFailure();
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

    if (tiledLoops.empty()) {
      linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
          [](OpBuilder &builder, Location loc,
             ArrayRef<Range> parallelLoopRanges) {
            auto numParallelDims = parallelLoopRanges.size();
            SmallVector<linalg::ProcInfo, 3> procInfo(numParallelDims);
            for (size_t dim = 0,
                        e = std::min(numParallelDims, static_cast<size_t>(3));
                 dim < e; ++dim) {
              procInfo[numParallelDims - dim - 1] = {
                  builder.createOrFold<IREE::HAL::InterfaceWorkgroupIDOp>(
                      loc, builder.getIndexType(), APInt(64, dim)),
                  builder.createOrFold<IREE::HAL::InterfaceWorkgroupCountOp>(
                      loc, builder.getIndexType(), APInt(64, dim)),
              };
            }
            return procInfo;
          },
          {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
           linalg::DistributionMethod::CyclicNumProcsEqNumIters,
           linalg::DistributionMethod::CyclicNumProcsEqNumIters}};
      TileAndFuseOptions tileAndFuseOptions = {workgroupDistributionOptions,
                                               allocateThreadLocalMemory};
      if (failed(tileAndFuseLinalgBufferOps(funcOp, linalgOps, dependenceGraph,
                                            launchConfig,
                                            tileAndFuseOptions))) {
        return signalPassFailure();
      }
    } else {
      SmallVector<int64_t, 4> defaultWorkloadPerWorkgroup(tiledLoops.size(), 1);
      if (!defaultWorkloadPerWorkgroup.empty()) {
        defaultWorkloadPerWorkgroup.back() = 4;
      }
      Optional<SmallVector<int64_t, 4>> workloadPerWorkgroup =
          launchConfig.getWorkloadPerWorkgroup(tiledLoops.size(),
                                               defaultWorkloadPerWorkgroup);
      if (!workloadPerWorkgroup) {
        funcOp.emitOpError("unable to find workload per workgroup");
        return signalPassFailure();
      }
      if (failed(materializeStaticLaunchInformation(
              funcOp, workloadPerWorkgroup.getValue()))) {
        funcOp.emitOpError("failed to materialize static launch information");
        return signalPassFailure();
      }
    }

    launchConfig.finalize(funcOp);
  }
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableTargetOp>>
createLinalgTileAndDistributePass() {
  return std::make_unique<LinalgTileAndDistributePass>();
}

static PassRegistration<LinalgTileAndDistributePass> pass(
    "iree-codegen-llvm-linalg-tile-and-distribute",
    "Tile and distribute Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndDistributePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
