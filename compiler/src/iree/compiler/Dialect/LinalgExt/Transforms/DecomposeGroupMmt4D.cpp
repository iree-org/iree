// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/CPU/IR/IREECPUTypes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

#define GEN_PASS_DEF_DECOMPOSEGROUPMMT4DPASS
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h.inc"

namespace {

class DecomposeGroupMmt4DPass final
    : public impl::DecomposeGroupMmt4DPassBase<DecomposeGroupMmt4DPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    getOperation().walk([&](GroupMmt4DOp op) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      auto weightsType =
          cast<RankedTensorType>(op.getExpertWeights().getType());
      auto outputType = cast<RankedTensorType>(op.getOutput().getType());
      ArrayRef<int64_t> outputPermutation = op.getOutputPermutation();
      int64_t outerRowDim = outputPermutation[0];
      int64_t innerRowDim = outputPermutation[2];
      int64_t innerTileRows = outputType.getDimSize(innerRowDim);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value upper = arith::ConstantIndexOp::create(rewriter, loc,
                                                   weightsType.getDimSize(0));
      Value tileEnd = arith::AddIOp::create(
          rewriter, loc, op.getRowOffset(),
          arith::MulIOp::create(
              rewriter, loc,
              tensor::DimOp::create(rewriter, loc, op.getInput(), 0),
              tensor::DimOp::create(rewriter, loc, op.getInput(), 2)));

      auto loop = scf::ForOp::create(
          rewriter, loc, zero, upper, one, ValueRange{op.getOutput(), zero},
          [&](OpBuilder &builder, Location bodyLoc, Value expert,
              ValueRange iterArgs) {
            Value output = iterArgs[0];
            Value expertStart = iterArgs[1];
            Value expertEnd = tensor::ExtractOp::create(
                builder, bodyLoc, op.getExpertOffsets(), ValueRange{expert});
            if (!expertEnd.getType().isIndex()) {
              expertEnd = arith::IndexCastOp::create(
                  builder, bodyLoc, builder.getIndexType(), expertEnd);
            }
            Value activeAfterStart = arith::CmpIOp::create(
                builder, bodyLoc, arith::CmpIPredicate::slt, op.getRowOffset(),
                expertEnd);
            Value activeBeforeEnd = arith::CmpIOp::create(
                builder, bodyLoc, arith::CmpIPredicate::slt, expertStart,
                tileEnd);
            Value isActive = arith::AndIOp::create(
                builder, bodyLoc, activeAfterStart, activeBeforeEnd);

            auto active = scf::IfOp::create(
                builder, bodyLoc, TypeRange{output.getType()}, isActive,
                /*withElseRegion=*/true);
            {
              OpBuilder::InsertionGuard guard(builder);
              OpBuilder thenBuilder =
                  OpBuilder::atBlockBegin(&active.getThenRegion().front());
              SmallVector<OpFoldResult> offsets(5, thenBuilder.getIndexAttr(0));
              offsets[0] = expert;
              SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(
                  thenBuilder, bodyLoc, op.getExpertWeights());
              sizes[0] = thenBuilder.getIndexAttr(1);
              SmallVector<OpFoldResult> weightStrides(
                  5, thenBuilder.getIndexAttr(1));
              auto expertWeightsType =
                  weightsType.clone(weightsType.getShape().drop_front());
              Value weights = tensor::ExtractSliceOp::create(
                  thenBuilder, bodyLoc, expertWeightsType,
                  op.getExpertWeights(), offsets, sizes, weightStrides);
              Value mmtInit = tensor::EmptyOp::create(
                  thenBuilder, bodyLoc,
                  tensor::getMixedSizes(thenBuilder, bodyLoc, output),
                  outputType.getElementType());
              Value zeroElement = arith::ConstantOp::create(
                  thenBuilder, bodyLoc,
                  thenBuilder.getZeroAttr(outputType.getElementType()));
              mmtInit = linalg::FillOp::create(thenBuilder, bodyLoc,
                                               zeroElement, mmtInit)
                            .getResult(0);
              Value matmul =
                  linalg::Mmt4DOp::create(
                      thenBuilder, bodyLoc, output.getType(),
                      op.getTransposed() ? ValueRange{weights, op.getInput()}
                                         : ValueRange{op.getInput(), weights},
                      ValueRange{mmtInit})
                      .getResult(0);
              if (auto config = getLoweringConfig(op)) {
                if (auto cpuConfig =
                        dyn_cast<IREE::CPU::LoweringConfigAttr>(config);
                    cpuConfig && op.getTransposed()) {
                  MLIRContext *ctx = cpuConfig.getContext();
                  SmallVector<NamedAttribute> configItems;
                  for (int i : IREE::CPU::getTilingLevelsAsInts()) {
                    if (!cpuConfig.hasTilingLevel(i)) {
                      continue;
                    }
                    auto level = static_cast<IREE::CPU::TilingLevel>(i);
                    auto levelAttr =
                        cast<IREE::Codegen::LoweringConfigTilingLevelAttr>(
                            cpuConfig.getTilingLevelAttr(i));
                    SmallVector<int64_t> tileSizes(levelAttr.getSizes());
                    SmallVector<bool> scalableFlags(
                        levelAttr.getScalableFlags());
                    if (tileSizes.size() == 6) {
                      applyPermutationToVector(tileSizes,
                                               op.getLoopPermutation());
                      scalableFlags.resize(tileSizes.size(), false);
                      applyPermutationToVector(scalableFlags,
                                               op.getLoopPermutation());
                    }
                    configItems.emplace_back(
                        IREE::CPU::getTilingLevelName(level),
                        IREE::CPU::LoweringConfigAttr::getTilingLevelAttr(
                            ctx, tileSizes, scalableFlags));
                  }
                  setLoweringConfig(
                      matmul.getDefiningOp(),
                      IREE::CPU::LoweringConfigAttr::get(ctx, configItems));
                } else {
                  setLoweringConfig(matmul.getDefiningOp(), config);
                }
              }

              SmallVector<AffineMap> indexingMaps(
                  2, thenBuilder.getMultiDimIdentityMap(outputType.getRank()));
              SmallVector<utils::IteratorType> iteratorTypes(
                  outputType.getRank(), utils::IteratorType::parallel);
              auto merge = linalg::GenericOp::create(
                  thenBuilder, bodyLoc, outputType, ValueRange{matmul},
                  ValueRange{output}, indexingMaps, iteratorTypes,
                  [&](OpBuilder &nestedBuilder, Location nestedLoc,
                      ValueRange args) {
                    Value outerRow = linalg::IndexOp::create(
                        nestedBuilder, nestedLoc, outerRowDim);
                    Value innerRow = linalg::IndexOp::create(
                        nestedBuilder, nestedLoc, innerRowDim);
                    Value row = arith::AddIOp::create(
                        nestedBuilder, nestedLoc, op.getRowOffset(),
                        arith::AddIOp::create(
                            nestedBuilder, nestedLoc,
                            arith::MulIOp::create(
                                nestedBuilder, nestedLoc, outerRow,
                                arith::ConstantIndexOp::create(
                                    nestedBuilder, nestedLoc, innerTileRows)),
                            innerRow));
                    Value afterStart = arith::CmpIOp::create(
                        nestedBuilder, nestedLoc, arith::CmpIPredicate::sge,
                        row, expertStart);
                    Value beforeEnd = arith::CmpIOp::create(
                        nestedBuilder, nestedLoc, arith::CmpIPredicate::slt,
                        row, expertEnd);
                    Value belongsToExpert = arith::AndIOp::create(
                        nestedBuilder, nestedLoc, afterStart, beforeEnd);
                    linalg::YieldOp::create(
                        nestedBuilder, nestedLoc,
                        arith::SelectOp::create(nestedBuilder, nestedLoc,
                                                belongsToExpert, args[0],
                                                args[1])
                            .getResult());
                  });
              scf::YieldOp::create(thenBuilder, bodyLoc, merge.getResult(0));
              OpBuilder elseBuilder =
                  OpBuilder::atBlockBegin(&active.getElseRegion().front());
              scf::YieldOp::create(elseBuilder, bodyLoc, output);
            }
            scf::YieldOp::create(builder, bodyLoc,
                                 ValueRange{active.getResult(0), expertEnd});
          });
      rewriter.replaceOp(op, loop.getResult(0));
    });
  }
};

} // namespace
} // namespace mlir::iree_compiler::IREE::LinalgExt
