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
#include "iree/compiler/Conversion/CodegenUtils/MatmulCodegenStrategy.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

static llvm::cl::opt<int> wgTileSize(
    "iree-codegen-linalg-to-gpu-wg-tile-size",
    llvm::cl::desc(
        "Specify the size of workgroup tile for matmul vector lowering"),
    llvm::cl::init(32));

static llvm::cl::list<uint32_t> unrollSize(
    "iree-codegen-linalg-to-gpu-unroll-size",
    llvm::cl::desc("Specify the size of the "), llvm::cl::CommaSeparated);

static llvm::cl::opt<bool> enableLICM(
    "iree-codegen-linalg-to-gpu-matmul-licm",
    llvm::cl::desc(
        "If true run LICM and hoisting passes after the staged transforms"),
    llvm::cl::init(true));

namespace mlir {
namespace iree_compiler {

namespace {
struct MatMulTileAndVectorizeGPUPass
    : PassWrapper<MatMulTileAndVectorizeGPUPass, FunctionPass> {
  void runOnFunction() override;
};
}  // namespace

void MatMulTileAndVectorizeGPUPass::runOnFunction() {
  FuncOp fn = getFunction();
  SmallVector<uint32_t, 3> vUnrollSize(unrollSize.begin(), unrollSize.end());
  if (vUnrollSize.size() != 3) signalPassFailure();
  MatmulCodegenStrategy strategy;
  strategy
      .tile<linalg::MatmulOp>(
          linalg::LinalgTilingOptions()
              // TODO(thomasraoux): Enable parallel loops once affine.min
              // canonicalize supports it.
              //.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops)
              .setTileSizes({wgTileSize, wgTileSize, wgTileSize}))
      .setHoistInvariantCode(enableLICM)
      .vectorize<linalg::MatmulOp>()
      .unrollVector<vector::ContractionOp>(
          {vUnrollSize[0], vUnrollSize[1], vUnrollSize[2]});
  strategy.transform(fn);
}

std::unique_ptr<FunctionPass> createMatMulTileAndVectorizeGPUPass() {
  return std::make_unique<MatMulTileAndVectorizeGPUPass>();
}

static PassRegistration<MatMulTileAndVectorizeGPUPass> pass(
    "iree-codegen-linalg-to-gpu-matmul-vectorization-pass",
    "Tile and vectorize linalg.matmul operation",
    [] { return std::make_unique<MatMulTileAndVectorizeGPUPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
