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
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "iree/compiler/Conversion/Common/Attributes.h"
#include "iree/compiler/Conversion/Common/Transforms.h"
#include "iree/compiler/Conversion/LinalgToLLVM/KernelDispatch.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
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
    : public PassWrapper<LinalgTileAndDistributePass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREEDialect, AffineDialect,
                    scf::SCFDialect>();
  }
  LinalgTileAndDistributePass() = default;
  LinalgTileAndDistributePass(const LinalgTileAndDistributePass &pass) {}
  void runOnOperation() override;
};
}  // namespace

namespace {
template <typename LinalgOpTy>
struct TileToCPUThreads : public linalg::LinalgBaseTilingPattern {
  using Base = linalg::LinalgBaseTilingPattern;
  TileToCPUThreads(MLIRContext *context,
                   const linalg::LinalgDependenceGraph &dependenceGraph,
                   const CPUKernelDispatch &cpuKernelDispatch,
                   linalg::LinalgTilingOptions options,
                   linalg::LinalgMarker marker, PatternBenefit benefit = 1)
      : Base(LinalgOpTy::getOperationName(), context, options, marker, benefit),
        cpuKernelDispatch(cpuKernelDispatch) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Find the parent FuncOp before tiling. If tiling succeeds, the op will be
    // erased.
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    SmallVector<Value, 4> tensorResults;
    if (!funcOp ||
        failed(Base::matchAndRewriteBase(op, rewriter, tensorResults)) ||
        !tensorResults.empty() ||
        (funcOp.getAttr(getNumWorkgroupsFnAttrName()) &&
         failed(createNumWorkgroupsFromResultShape(
             rewriter, cast<linalg::LinalgOp>(op), funcOp,
             getNumWorkgroupsFnAttrName(),
             cpuKernelDispatch.getTileSizes<TilingLevel::WorkGroupTiles>(
                 op))))) {
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
  CPUKernelDispatch cpuKernelDispatch;
};

template <typename LinalgOpTy>
struct TileAndFuseToCPUThreads
    : public linalg::LinalgTileAndFusePattern<LinalgOpTy> {
  using Base = linalg::LinalgTileAndFusePattern<LinalgOpTy>;
  TileAndFuseToCPUThreads(MLIRContext *context,
                          const linalg::LinalgDependenceGraph &dependenceGraph,
                          const CPUKernelDispatch &cpuKernelDispatch,
                          linalg::LinalgTilingOptions tilingOptions,
                          linalg::LinalgMarker marker,
                          PatternBenefit benefit = 1)
      : Base(context, dependenceGraph, tilingOptions,
             linalg::LinalgFusionOptions().setIndicesToFuse({2}), marker,
             marker,
             linalg::LinalgMarker(ArrayRef<Identifier>(),
                                  Identifier::get(getDeleteMarker(), context)),
             benefit),
        dependenceGraph(dependenceGraph),
        cpuKernelDispatch(cpuKernelDispatch) {}

  virtual LogicalResult matchAndRewrite(Operation *op,
                                        PatternRewriter &rewriter) const {
    FuncOp funcOp = op->getParentOfType<FuncOp>();
    linalg::LinalgOp linalgOp = cast<linalg::LinalgOp>(op);
    if (!funcOp || !dependenceGraph.hasDependentOperations(linalgOp) ||
        failed(Base::matchAndRewrite(op, rewriter)) ||
        failed(createNumWorkgroupsFromResultShape(
            rewriter, cast<linalg::LinalgOp>(op), funcOp,
            getNumWorkgroupsFnAttrName().str(),
            cpuKernelDispatch.getTileSizes<TilingLevel::WorkGroupTiles>(op)))) {
      return failure();
    }
    return success();
  }

  const linalg::LinalgDependenceGraph &dependenceGraph;
  CPUKernelDispatch cpuKernelDispatch;
};

}  // namespace

Optional<Value> allocateThreadLocalMemory(OpBuilder &b, SubViewOp subview,
                                          ArrayRef<Value> boundingSubViewSize,
                                          OperationFolder *folder) {
  // Allocate the memory into the entry block of the parent FuncOp. This better
  // aligns with the semantics of this memory which is available at the entry of
  // the function.
  OpBuilder::InsertionGuard guard(b);
  FuncOp funcOp = subview.getParentOfType<FuncOp>();
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
  ModuleOp module = getOperation();

  static linalg::LinalgLoopDistributionOptions workgroupDistributionOptions = {
      [](OpBuilder &builder, Location loc, ArrayRef<Range> parallelLoopRanges) {
        Type indexType = builder.getIndexType();
        auto numParallelDims = parallelLoopRanges.size();
        SmallVector<linalg::ProcInfo, 2> procInfo(numParallelDims);
        for (int dim = 0; dim < numParallelDims; ++dim) {
          std::array<StringRef, 3> dimAttr{"x", "y", "z"};
          StringAttr attr =
              builder.getStringAttr(dimAttr[std::min<unsigned>(dim, 3)]);
          procInfo[numParallelDims - dim - 1] = {
              builder.create<IREE::WorkgroupIdOp>(loc, indexType, attr),
              builder.create<IREE::WorkgroupSizeOp>(loc, indexType, attr)};
        }
        return procInfo;
      },
      {linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters,
       linalg::DistributionMethod::CyclicNumProcsEqNumIters}};

  for (FuncOp funcOp : module.getOps<FuncOp>()) {
    if (!isEntryPoint(funcOp)) continue;

    Region &body = funcOp.getBody();
    if (!llvm::hasSingleElement(body.getBlocks())) {
      funcOp.emitError("unhandled dispatch function with multiple blocks");
      return signalPassFailure();
    }
    Block &block = body.front();
    auto linalgOps = block.getOps<linalg::LinalgOp>();
    if (linalgOps.empty()) continue;

    SmallVector<linalg::LinalgOp, 4> linalgOpsVec =
        llvm::to_vector<4>(llvm::map_range(linalgOps, [](Operation *op) {
          return cast<linalg::LinalgOp>(op);
        }));
    linalg::Aliases aliases;
    linalg::LinalgDependenceGraph dependenceGraph(aliases, linalgOpsVec);
    Optional<LaunchConfig> launchConfigOpt =
        initCPULaunchConfig(context, dependenceGraph, linalgOpsVec);
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

    TileAndFuseOptions tileAndFuseOptions = {workgroupDistributionOptions,
                                             allocateThreadLocalMemory};
    if (failed(tileAndFuseLinalgBufferOps(funcOp, linalgOpsVec, dependenceGraph,
                                          launchConfig, tileAndFuseOptions))) {
      return signalPassFailure();
    }

    launchConfig.finalize(funcOp);
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTileAndDistributePass() {
  return std::make_unique<LinalgTileAndDistributePass>();
}

static PassRegistration<LinalgTileAndDistributePass> pass(
    "iree-codegen-llvm-linalg-tile-and-distribute",
    "Tile and distribute Linalg operations on buffers",
    [] { return std::make_unique<LinalgTileAndDistributePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
