//===- LinalgTensorCodegenStrategyPass.cpp - Test Linalg codegen strategy--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing the Linalg codegen strategy.
//
//===----------------------------------------------------------------------===//

// TODO: avoid copy-pasta but I can't seem to be able to inherit from a pass.
// Will get better once upstreamed to core and it replaces the existing codegen
// strategy.

#include "Transforms.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgTensorCodegenStrategyPass
    : public PassWrapper<LinalgTensorCodegenStrategyPass, FunctionPass> {
  LinalgTensorCodegenStrategyPass() = default;
  LinalgTensorCodegenStrategyPass(const LinalgTensorCodegenStrategyPass &pass) {
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // clang-format off
    registry.insert<AffineDialect,
                    gpu::GPUDialect,
                    linalg::LinalgDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    StandardOpsDialect,
                    tensor::TensorDialect,
                    vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnFunction() override;

  ListOption<int64_t> tileSizes{*this, "tile-sizes",
                                llvm::cl::MiscFlags::CommaSeparated,
                                llvm::cl::desc("Specifies the tile sizes.")};
  Option<bool> promote{
      *this, "promote",
      llvm::cl::desc("Promote the tile into a small aligned memory buffer."),
      llvm::cl::init(false)};
  Option<bool> promoteFullTile{
      *this, "promote-full-tile-pad",
      llvm::cl::desc("Pad the small aligned memory buffer to the tile sizes."),
      llvm::cl::init(false)};
  Option<bool> vectorize{
      *this, "vectorize",
      llvm::cl::desc("Rewrite the linalg op as a vector operation."),
      llvm::cl::init(false)};
  Option<std::string> splitVectorTransfersTo{
      *this, "split-transfers",
      llvm::cl::desc(
          "Split vector transfers between slow (masked) and fast "
          "(unmasked) variants. Possible options are:\n"
          "\tnone: keep unsplit vector.transfer and pay the full price\n"
          "\tlinalg-copy: use linalg.fill + linalg.copy for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfer for"
          " the slow path\n"),
      llvm::cl::init("none")};
  Option<std::string> vectorizeContractionTo{
      *this, "vectorize-contraction-to",
      llvm::cl::desc("the type of vector op to use for linalg contractions"),
      llvm::cl::init("outerproduct")};
  Option<bool> unrollVectorTransfers{
      *this, "unroll-vector-transfers",
      llvm::cl::desc("Enable full unrolling of vector.transfer operations"),
      llvm::cl::init(false)};
  Option<bool> licm{*this, "licm", llvm::cl::desc("Enable LICM."),
                    llvm::cl::init(true)};
  Option<bool> hoistRedundantVectorTransfers{
      *this, "hoist-redundant-vector-transfers",
      llvm::cl::desc("Enable HoistRedundantVectorTransfers"),
      llvm::cl::init(true)};
  Option<bool> vectorTransferPartialRewrite{
      *this, "vector-transfer-partial-rewrite",
      llvm::cl::desc("Enable rewriting of vector.transfer operations into "
                     "full/partial read/writes."),
      llvm::cl::init(true)};
  Option<bool> vectorContractLowering{
      *this, "vector-contract-lowering",
      llvm::cl::desc("Enable lowering of vector contractions."),
      llvm::cl::init(true)};
  Option<bool> vectorToSCFConversion{
      *this, "vector-to-scf-conversion",
      llvm::cl::desc("Enable vector to scf conversions."),
      llvm::cl::init(true)};
  Option<std::string> anchorOpName{
      *this, "anchor-op",
      llvm::cl::desc(
          "Which single linalg op is the anchor for the codegen strategy to "
          "latch on:\n"
          "\tlinalg.matmul: anchor on linalg.matmul\n"
          "\tlinalg.matmul_column_major: anchor on linalg.matmul_column_major\n"
          "\tlinalg.copy: anchor on linalg.copy\n"
          "\tlinalg.fill: anchor on linalg.fill\n"),
      llvm::cl::init("")};
  Option<std::string> anchorFuncOpName{
      *this, "anchor-func",
      llvm::cl::desc(
          "Which single func op is the anchor for the codegen strategy to "
          "latch on."),
      llvm::cl::init("")};

  Option<bool> distribute{
      *this, "distribute",
      llvm::cl::desc("Distribute the linalg op into a TiledGeneric."),
      llvm::cl::init(false)};
  ListOption<int64_t> distributeTileSizes{
      *this, "distribute-tile-sizes", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the tile sizes.")};
  Option<bool> pad{*this, "pad", llvm::cl::desc("Use padding during tiling."),
                   llvm::cl::init(false)};
  Option<int> hoistPadding{
      *this, "hoist-padding",
      llvm::cl::desc("Hoist padding by the number of specified loops."),
      llvm::cl::init(0)};
  Option<bool> vectorizePadding{
      *this, "vectorize-padding",
      llvm::cl::desc("Rewrite linalg.pad_tensor in vector form."),
      llvm::cl::init(false)};
};
}  // end anonymous namespace

static void runStrategy(LinalgTensorCodegenStrategyPass &pass,
                        CodegenStrategy &strategy) {
  strategy.setEnableLICM(pass.licm)
      .setEnableHoistRedundantVectorTransfers(
          pass.hoistRedundantVectorTransfers)
      .setEnableHoistRedundantVectorTransfersOnTensor(
          pass.hoistRedundantVectorTransfers)
      .setEnableVectorTransferPartialRewrite(pass.vectorTransferPartialRewrite)
      .setEnableVectorContractLowering(pass.vectorContractLowering)
      .setEnableVectorToSCFConversion(pass.vectorToSCFConversion)
      .transform(pass.getFunction());
}

static void runGenericStrategy(
    LinalgTensorCodegenStrategyPass &pass, LinalgTilingOptions tilingOptions,
    vector::VectorContractLowering vectorContractLowering,
    vector::VectorTransferSplit vectorTransferSplit) {
  assert(!pass.anchorOpName.empty());
  CodegenStrategy strategy;
  strategy
      .tileIf<LinalgOp>(!pass.tileSizes.empty(), pass.anchorOpName,
                        tilingOptions)
      .promoteIf<LinalgOp>(
          pass.promote, pass.anchorOpName,
          LinalgPromotionOptions()
              .setAlignment(16)
              .setUseFullTileBuffersByDefault(pass.promoteFullTile))
      .vectorizeIf(pass.vectorize, pass.anchorOpName)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(pass.unrollVectorTransfers));
  runStrategy(pass, strategy);
}

template <typename OpType>
static void runStrategy(LinalgTensorCodegenStrategyPass &pass,
                        LinalgTilingOptions tilingOptions,
                        vector::VectorContractLowering vectorContractLowering,
                        vector::VectorTransferSplit vectorTransferSplit) {
  CodegenStrategy strategy;
  strategy.tileIf<OpType>(!pass.tileSizes.empty(), tilingOptions)
      .template promoteIf<OpType>(
          pass.promote,
          LinalgPromotionOptions()
              .setAlignment(16)
              .setUseFullTileBuffersByDefault(pass.promoteFullTile))
      .template vectorizeIf<OpType>(pass.vectorize)
      .setVectorTransformsOptions(
          vector::VectorTransformsOptions()
              .setVectorTransformsOptions(vectorContractLowering)
              .setVectorTransferSplit(vectorTransferSplit))
      .setVectorTransferToSCFOptions(
          VectorTransferToSCFOptions().setUnroll(pass.unrollVectorTransfers));
  runStrategy(pass, strategy);
}

// For now, just assume it is the zero of type.
// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<ConstantOp>(op.getOwner()->getLoc(), t, b.getZeroAttr(t));
}

/// Apply transformations specified as patterns.
void LinalgTensorCodegenStrategyPass::runOnFunction() {
  if (!anchorFuncOpName.empty() && anchorFuncOpName != getFunction().getName())
    return;

  if (distribute && !distributeTileSizes.empty()) {
    LinalgTilingOptions tilingOptions;
    tilingOptions = tilingOptions.setTileSizes(distributeTileSizes);
    if (pad)
      tilingOptions = tilingOptions.setPaddingValueComputationFunction(
          getNeutralOfLinalgOp);
    OwningRewritePatternList patterns;
    patterns.insert<TileAndDistributePattern>(
        TileAndDistributeOptions{tilingOptions},
        LinalgTransformationFilter(
            ArrayRef<Identifier>{},
            {Identifier::get("distributed", getFunction().getContext())})
            .addFilter([](Operation *op) {
              return success(isaContractionOpInterface(dyn_cast<LinalgOp>(op)));
            }));
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
    // Ensure we drop the marker in the end.
    getFunction().walk([](LinalgOp op) {
      op.removeAttr(LinalgTransforms::kLinalgTransformMarker);
    });
  }

  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty()) tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (pad)
    tilingOptions =
        tilingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);

  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          vectorizeContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  if (!anchorOpName.empty()) {
    using RunFnType = decltype(&runGenericStrategy);
    RunFnType runFn = &runGenericStrategy;
    // `linalg::PadTensorOp::getOperationName()` is not a StringLiteral, cannot
    // use StringSwitch.
    if (anchorOpName == CopyOp::getOperationName())
      runFn = &runStrategy<CopyOp>;
    if (anchorOpName == FillOp::getOperationName())
      runFn = &runStrategy<FillOp>;
    if (anchorOpName == PadTensorOp::getOperationName())
      runFn = &runStrategy<PadTensorOp>;
    else if (anchorOpName == MatmulOp::getOperationName())
      runFn = &runStrategy<MatmulOp>;
    else if (anchorOpName == MatmulI8I8I32Op::getOperationName())
      runFn = &runStrategy<MatmulI8I8I32Op>;
    runFn(*this, tilingOptions, vectorContractLowering, vectorTransferSplit);
  }

  // Transforms that do not require anchoring on a given op.
  if (hoistPadding > 0) {
    SmallVector<PadTensorOp> ops;
    getFunction().walk([&](PadTensorOp op) { ops.push_back(op); });
    for (auto op : llvm::reverse(ops))
      (void)hoistPaddingOnTensors(op, hoistPadding);
  }
  if (vectorizePadding) {
    OwningRewritePatternList extraVectorizationPatterns;
    extraVectorizationPatterns.insert<PadTensorOpVectorizationPattern>(
        &getContext());
    (void)applyPatternsAndFoldGreedily(getFunction(),
                                       std::move(extraVectorizationPatterns));
  }
}

namespace mlir {
namespace linalg {
void registerLinalgTensorCodegenStrategyPass() {
  PassRegistration<LinalgTensorCodegenStrategyPass>
      testLinalgCodegenStrategyPass("linalg-tensor-codegen-strategy",
                                    "Linalg Tensor Codegen Strategy.");
}
}  // namespace linalg
}  // namespace mlir
