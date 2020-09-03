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

namespace mlir {
namespace iree_compiler {

static llvm::cl::opt<bool> useL1TilesOnly(
    "iree-codegen-linalg-to-llvm-matmul-use-l1-tiles-only",
    llvm::cl::desc("If specified only L1 tiling applies, tiling for reduction "
                   "dim is set l1-tile-size, other dimension are tiled by "
                   "at most l1-reigster-max-tile-size"),
    llvm::cl::init(false));

static llvm::cl::opt<int> l1RegisterMaxTileSize(
    "iree-codegen-linalg-to-llvm-matmul-l1-register-max-tile-size",
    llvm::cl::desc(
        "Max tile size for M & N dimensions when use-l1-tiles-only is set"),
    llvm::cl::init(8));

static llvm::cl::opt<int> l1TileSize(
    "iree-codegen-linalg-to-llvm-matmul-l1-tile-size",
    llvm::cl::desc("Specify the size of L1 tile for matmul vector lowering"),
    llvm::cl::init(4));

static llvm::cl::opt<int> l2TileSize(
    "iree-codegen-linalg-to-llvm-matmul-l2-tile-size",
    llvm::cl::desc("Specify the size of L2 tile for matmul vector lowering"),
    llvm::cl::init(32));

static llvm::cl::opt<int> l3TileSize(
    "iree-codegen-linalg-to-llvm-matmul-l3-tile-size",
    llvm::cl::desc("Specify the size of L3 tile for matmul vector lowering"),
    llvm::cl::init(64));

static llvm::cl::opt<bool> unrollVectorTransfer(
    "iree-codegen-linalg-to-llvm-matmul-unroll-vector-transfer",
    llvm::cl::desc("If true vector transfers operation loop get unrolled."),
    llvm::cl::init(true));

static llvm::cl::opt<std::string> vectorOpLowering(
    "iree-codegen-linalg-to-llvm-matmul-vector-op-lowerig",
    llvm::cl::desc(
        "Select the vector operation for lowering linalg.matmul, options : "
        "{'outer_product', 'vector_contract', 'matrix_internsics'}"),
    llvm::cl::init("outer_product"));

namespace {
struct MatMulTileAndVectorizePass
    : PassWrapper<MatMulTileAndVectorizePass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect, scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnFunction() override;
};
}  // namespace

void MatMulTileAndVectorizePass::runOnFunction() {
  FuncOp fn = getFunction();
  MatmulCodegenStrategy strategy;

  if (useL1TilesOnly) {
    auto matmulOps = llvm::to_vector<1>(fn.getOps<linalg::MatmulOp>());
    if (matmulOps.size() != 1) return;

    auto lhsShapeType =
        matmulOps[0].getInput(0).getType().dyn_cast_or_null<ShapedType>();
    auto rhsShapeType =
        matmulOps[0].getInput(1).getType().dyn_cast_or_null<ShapedType>();

    if (!lhsShapeType || !rhsShapeType) return;

    auto lhsShape = lhsShapeType.getShape();
    auto rhsShape = rhsShapeType.getShape();

    int M = lhsShape[0], N = rhsShape[1];
    auto tileDim = [](int dim, int maxSize) -> int {
      for (int i = maxSize; i > 0; --i) {
        if (dim % i == 0) return i;
      }
      return 1;
    };
    strategy.tile<linalg::MatmulOp>(linalg::LinalgTilingOptions().setTileSizes(
        {tileDim(M, l1RegisterMaxTileSize), tileDim(N, l1RegisterMaxTileSize),
         l1TileSize}));
  } else {
    strategy
        .tile<linalg::MatmulOp>(linalg::LinalgTilingOptions().setTileSizes(
            {l3TileSize, l3TileSize, l3TileSize}))
        .tile<linalg::MatmulOp>(linalg::LinalgTilingOptions().setTileSizes(
            {l2TileSize, l2TileSize, l2TileSize}))
        .tile<linalg::MatmulOp>(linalg::LinalgTilingOptions().setTileSizes(
            {l1TileSize, l1TileSize, l1TileSize}));
  }
  strategy.vectorize<linalg::MatmulOp>().setVectorTransferToSCFOptions(
      VectorTransferToSCFOptions().setUnroll(unrollVectorTransfer));
  if (vectorOpLowering == "outer_product") {
    strategy.setVectorTransformsOptions(
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct));
  } else if (vectorOpLowering == "vector_contract") {
    strategy.setVectorTransformsOptions(
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct));
  } else if (vectorOpLowering == "matrix_internsics") {
    strategy.setVectorTransformsOptions(
        vector::VectorTransformsOptions().setVectorTransformsOptions(
            vector::VectorContractLowering::OuterProduct));
  } else {
    signalPassFailure();
  }
  strategy.setDefaultCPULowering();
  strategy.transform(fn);
}

std::unique_ptr<FunctionPass> createMatMulTileAndVectorizePass() {
  return std::make_unique<MatMulTileAndVectorizePass>();
}

static PassRegistration<MatMulTileAndVectorizePass> pass(
    "iree-codegen-linalg-to-llvm-matmul-vectorization-pass",
    "Tile and vectorize linalg.matmul operation",
    [] { return std::make_unique<MatMulTileAndVectorizePass>(); });

}  // namespace iree_compiler
}  // namespace mlir
