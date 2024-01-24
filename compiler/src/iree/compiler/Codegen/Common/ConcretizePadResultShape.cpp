// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-concretize-pad-result-shape"

namespace mlir::iree_compiler {

/// Gets the given `attrOrValue` as an index value by creating constant ops
/// for attributes.
static Value getAsIndexValue(OpFoldResult attrOrValue, OpBuilder &builder,
                             Location loc) {
  IntegerAttr attr;
  if (Value val = attrOrValue.dyn_cast<Value>()) {
    if (val.getType().isIndex())
      return val;
    matchPattern(val, m_Constant(&attr));
  } else {
    attr = llvm::cast<IntegerAttr>(attrOrValue.get<Attribute>());
  }
  return builder.createOrFold<arith::ConstantIndexOp>(
      loc, attr.getValue().getSExtValue());
}

namespace {

/// Concretizes tensor.pad op's result shape if its source op implements
/// OffsetSizeAndStrideOpInterface. For example, pad(extract_slice).
struct ConcretizePadResultShape final : public OpRewritePattern<tensor::PadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PadOp padOp,
                                PatternRewriter &rewriter) const override {
    // If the result shape is already static, then nothing to do.
    if (padOp.getResultType().hasStaticShape())
      return failure();

    int rank = padOp.getResultType().getRank();
    SmallVector<int64_t> staticShape;
    staticShape.reserve(rank);

    auto sourceIfxOp = dyn_cast_or_null<OffsetSizeAndStrideOpInterface>(
        padOp.getSource().getDefiningOp());
    if (!sourceIfxOp)
      return failure();

    SmallVector<OpFoldResult> lowPad = padOp.getMixedLowPad();
    SmallVector<OpFoldResult> source = sourceIfxOp.getMixedSizes();
    SmallVector<OpFoldResult> highPad = padOp.getMixedHighPad();

    MLIRContext *context = padOp.getContext();
    Location loc = padOp.getLoc();

    AffineExpr sym0, sym1, sym2;
    bindSymbols(context, sym0, sym1, sym2);
    auto addMap = AffineMap::get(0, 3, {sym0 + sym1 + sym2}, context);

    SmallVector<Value, 3> valueSizes;
    for (int dimIndex = 0; dimIndex < rank; ++dimIndex) {
      valueSizes.clear();
      valueSizes.push_back(getAsIndexValue(lowPad[dimIndex], rewriter, loc));
      valueSizes.push_back(getAsIndexValue(source[dimIndex], rewriter, loc));
      valueSizes.push_back(getAsIndexValue(highPad[dimIndex], rewriter, loc));

      // The pad op's result shape is low padding + source size + high padding.
      // Try to see if we can get a constant number by composing and
      // canonicalizing the result. We use affine mechanisms here because
      // generating arithmetic add ops over dim ops won't work, given they are
      // SSA values that would need invoking other patterns to simplify. We
      // cannot invoke patterns in patterns.
      AffineMap map = addMap;
      affine::fullyComposeAffineMapAndOperands(&map, &valueSizes);
      affine::canonicalizeMapAndOperands(&map, &valueSizes);

      auto cstExpr = dyn_cast<AffineConstantExpr>(map.getResult(0));
      // Specially handle the case where we have both dimensions and symbols and
      // they map to the same value, e.g.:
      //   affine_map<(d0, s0) -> (d0 - s0 + 4)>(%v, %v).
      // Due to the restrictions over dimensions and symbols, the above won't
      // simplify. Try to change dimensions for symbols for such cases.
      if (!cstExpr && llvm::all_equal(valueSizes)) {
        int numDims = map.getNumDims();
        int numSyms = map.getNumSymbols();
        DenseMap<AffineExpr, AffineExpr> dimToSymMap;
        for (int i = 0; i < numDims; ++i) {
          dimToSymMap[rewriter.getAffineDimExpr(i)] =
              rewriter.getAffineSymbolExpr(numSyms + i);
        }
        map = map.replace(dimToSymMap, /*numResultDims=*/0,
                          /*numResultSyms=*/numDims + numSyms);

        affine::canonicalizeMapAndOperands(&map, &valueSizes);
        cstExpr = dyn_cast<AffineConstantExpr>(map.getResult(0));
      }
      if (!cstExpr)
        return failure();

      staticShape.push_back(cstExpr.getValue());
    }

    auto resultType = RankedTensorType::get(
        staticShape, padOp.getResultType().getElementType(),
        padOp.getResultType().getEncoding());

    rewriter.modifyOpInPlace(padOp,
                             [&]() { padOp.getResult().setType(resultType); });
    return success();
  }
};

class ConcretizePadResultShapePass final
    : public ConcretizePadResultShapeBase<ConcretizePadResultShapePass> {
public:
  ConcretizePadResultShapePass() = default;
  ConcretizePadResultShapePass(const ConcretizePadResultShapePass &pass) =
      default;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    {
      RewritePatternSet patterns(context);
      populateConcretizePadResultShapePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After concretizing pad result shape and "
                      "canonicalization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createConcretizePadResultShapePass() {
  return std::make_unique<ConcretizePadResultShapePass>();
}

void populateConcretizePadResultShapePatterns(RewritePatternSet &patterns,
                                              ArrayRef<int64_t> numWorkgroups) {
  MLIRContext *context = patterns.getContext();
  linalg::populateLinalgTilingCanonicalizationPatterns(patterns);
  // Pulling in upstream scf.for and affine.min canonicalization patterns.
  // They work on tiled (but not distributed) loops.
  scf::populateSCFForLoopCanonicalizationPatterns(patterns);
  // Pulling in flow.dispatch.tensor.load op canonicalization patterns.
  // Tiling can generate dim ops taking them as operands.
  IREE::Flow::DispatchTensorLoadOp::getCanonicalizationPatterns(patterns,
                                                                context);
  // Pull in tensor dialect canonicalization patterns to fold tensor.cast
  // into producers when possible.
  context->getLoadedDialect<tensor::TensorDialect>()
      ->getCanonicalizationPatterns(patterns);
  patterns.add<ConcretizePadResultShape>(context);
}

} // namespace mlir::iree_compiler
