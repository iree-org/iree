// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO/CHLO dialects to Linalg dialect.

#include <algorithm>
#include <cstdint>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo-iree/Conversion/LegalizeToLinalgUtils.h"
#include "stablehlo-iree/Conversion/Passes.h"
#include "stablehlo-iree/Conversion/Rewriters.h"
#include "stablehlo-iree/Conversion/TypeConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALG
#include "stablehlo-iree/Conversion/Passes.h.inc"

namespace {
Value getResultValue(Operation *op) { return op->getResult(0); }

ShapedType getHloOpResultType(Operation *op) {
  return llvm::cast<ShapedType>(getResultValue(op).getType());
}

/// Extracts an element from a tensor and optionally converts it to an index
/// type, based on the tensor's pre-type conversion type.
Value extractIndexFromTensor(OpBuilder &builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted = builder.create<tensor::ExtractOp>(loc, tensor, tensorIndex);
  if (extracted.getType().isIndex())
    return extracted;
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

//===----------------------------------------------------------------------===//
// stablehlo.Einsum conversion patterns.
//===----------------------------------------------------------------------===//

// Looks through a set of dimension that has been marked as reduction axes,
// if it is found within the set, then we set it as "reduction", otherwise
// we can label it as "parallel".
SmallVector<utils::IteratorType, 3>
getEinsumLoopsAttrs(const llvm::SmallSetVector<StringRef, 4> &inputInd,
                    const llvm::SmallSetVector<StringRef, 4> &reductionDims) {
  SmallVector<utils::IteratorType, 3> res;
  for (StringRef dim : inputInd) {
    if (!reductionDims.contains(dim)) {
      res.push_back(utils::IteratorType::parallel);
    } else {
      res.push_back(utils::IteratorType::reduction);
    }
  }
  return res;
}

SmallVector<Value, 2>
extractDynamicEinsumSizes(OpBuilder &b, Location loc, Value lhs, Value rhs,
                          const SmallVector<std::string> &lhsLoopVec,
                          const SmallVector<std::string> &rhsLoopVec,
                          const SmallVector<std::string> &outputLoopVec) {
  SmallVector<Value, 2> dynSizes;
  for (const std::string &dimInd : outputLoopVec) {
    Value dimSize;
    const auto *dimIndIt = llvm::find(lhsLoopVec, dimInd);
    if (dimIndIt != lhsLoopVec.end()) {
      // Query from lhs vars.
      auto dimIndPos = dimIndIt - lhsLoopVec.begin();
      auto lhsShape =
          llvm::dyn_cast<RankedTensorType>(lhs.getType()).getShape();
      if (!ShapedType::isDynamic(lhsShape[dimIndPos]))
        continue;
      dimSize = b.create<tensor::DimOp>(loc, lhs, dimIndPos);
    } else {
      // query from rhs vars.
      dimIndIt = std::find(rhsLoopVec.begin(), rhsLoopVec.end(), dimInd);
      auto dimIndPos = dimIndIt - rhsLoopVec.begin();
      auto rhsShape =
          llvm::dyn_cast<RankedTensorType>(rhs.getType()).getShape();
      if (!ShapedType::isDynamic(rhsShape[dimIndPos]))
        continue;
      dimSize = b.create<tensor::DimOp>(loc, rhs, dimIndPos);
    }
    dynSizes.push_back(dimSize);
  }
  return dynSizes;
}

// Adds indices/axes that are missing from output set.
llvm::SmallSetVector<StringRef, 4>
findSummationAxes(const llvm::SmallSetVector<StringRef, 4> &inputSet,
                  const llvm::SmallSetVector<StringRef, 4> &outputSet) {
  llvm::SmallSetVector<StringRef, 4> summationAxes;
  for (StringRef ind : inputSet) {
    if (!outputSet.contains(ind))
      summationAxes.insert(ind);
  }
  return summationAxes;
}

// Given a 1:1 map from std::string -> affine dimension expression
// we can get the affine expression of dimensions that an
// operand will access based on the input_str of einsum_config.
// For example:
// let string_dim_umap = {'a' : d0, 'b' : d1, 'c' : d2}
// for einsum_config "abc,cb->acb"
// first_input_operand will get umap[{"a","b","c"}] -> (d0, d1, d2).
// second_input_operand will get umap[{"c","b"}] -> (d2, d1).
// output_operand will get umap[{"a","c","b"}] -> (d0, d2, d1).
SmallVector<AffineExpr>
getExprFromConfig(const SmallVector<std::string> &loopDims,
                  const DenseMap<StringRef, AffineExpr> &strAffineDimUmap) {
  SmallVector<AffineExpr> exprs;
  for (const auto &dim : loopDims) {
    exprs.push_back(strAffineDimUmap.lookup(dim));
  }
  return exprs;
}

// Convert stablehlo.einsum op into linalg.generic.
// Algorithm in general 3 steps:

// Step1) Dissect entire einsum_config to different operands
// e.g f("abc,cd->abd") = {lhs:["abc"], rhs:["cd"], out:["abd"]}.

// Step2) Split up the string into vector of the elements
// e.g {lhs:["abc"], rhs:["cd"], out:["abd"]} = {lhs:["a","b","c"],
// rhs:["c","d"], out:["a","b","d"]}.

// Step3) Convert the vector into data access
// patern represented by affineMaps with affineDimensions e.g
// {lhs:["a","b","c"], rhs:["c","d"], out:["a","b","d"]} = {lhs:[d0,d1,d2],
// rhs:[d2,d3], out:[d0,d1,d3]}.
struct EinsumToLinalgConverter final
    : OpConversionPattern<mlir::stablehlo::EinsumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::EinsumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto getRank = [](Value v) {
      return llvm::cast<ShapedType>(v.getType()).getRank();
    };
    auto einsumConfig = op.getEinsumConfig();

    // With the assumption of binary input operand and single output
    // get the inputs and output operands' indices.
    // einsum_config = "lhs_loop,rhs_loop->out_loop"
    std::size_t posArrow = einsumConfig.find(kArrow);
    std::size_t posComma = einsumConfig.find(kComma);

    StringRef lhsLoop = einsumConfig.substr(0, posComma);
    StringRef rhsLoop = einsumConfig.substr(
        posComma + kComma.size(), posArrow - (posComma + kComma.size()));
    StringRef outLoop = einsumConfig.substr(posArrow + kArrow.size());

    // Check for Invalid Configs.
    // 1.Check that there is only maximum 2 inputs
    // 2.Check that there is only maximum 1 output
    // 3.Check that there is 1 kArrow
    if (rhsLoop.contains(kComma) || outLoop.contains(kComma) ||
        outLoop.contains(kArrow)) {
      return rewriter.notifyMatchFailure(op, "Invalid einsum config!");
    }

    // Find result type, if on tensors.
    auto resultTy = getTypeConverter()->convertType<RankedTensorType>(
        getHloOpResultType(op));

    // Check result type compatibility.
    if (!resultTy || !resultTy.getElementType().isSignlessIntOrFloat()) {
      return rewriter.notifyMatchFailure(op, "Invalid result type");
    }

    // Convert the representation to vector<string>.
    SmallVector<std::string> lhsEin =
        getEinsumConfigAsVector(lhsLoop, getRank(adaptor.getLhs()));
    SmallVector<std::string> rhsEin =
        getEinsumConfigAsVector(rhsLoop, getRank(adaptor.getRhs()));
    SmallVector<std::string> outEin =
        getEinsumConfigAsVector(outLoop, resultTy.getRank());

    if (!checkBatchHasEqualRank(lhsEin.size(), lhsLoop, rhsEin.size(), rhsLoop,
                                outEin.size(), outLoop)) {
      return rewriter.notifyMatchFailure(
          op, "Invalid elipsis('...') within einsum config!");
    }

    // Find all unique indices in the input and output.
    llvm::SmallSetVector<StringRef, 4> inputInd;
    llvm::SmallSetVector<StringRef, 4> outputInd;

    inputInd.insert(lhsEin.begin(), lhsEin.end());
    inputInd.insert(rhsEin.begin(), rhsEin.end());
    outputInd.insert(outEin.begin(), outEin.end());

    llvm::SmallSetVector<StringRef, 4> reductionAxe =
        findSummationAxes(inputInd, outputInd);

    // Find input/output values and types.
    Location loc = op.getLoc();

    // Prepare init tensor for linalg.generic op.
    auto dynSizes =
        extractDynamicEinsumSizes(rewriter, loc, adaptor.getLhs(),
                                  adaptor.getRhs(), lhsEin, rhsEin, outEin);
    Value output = getEmptyTensor(rewriter, loc, resultTy, dynSizes);
    if (!reductionAxe.empty()) {
      output = fillTensorWithZeros(rewriter, loc, output);
    }

    // Create indexing maps.
    // Create a 1:1 map from f:strDimension -> affineDimension.
    int64_t nloops = inputInd.size();
    DenseMap<StringRef, AffineExpr> strAffineDimUmap;
    for (auto [idx, value] : llvm::enumerate(inputInd)) {
      strAffineDimUmap[value] = rewriter.getAffineDimExpr(idx);
    }

    // From einsum_config of each operand in vector<string>, generate
    // the equivalent vector<AffineExpr>.
    SmallVector<AffineMap> maps;
    for (const SmallVector<std::string> &loopOperand :
         {lhsEin, rhsEin, outEin}) {
      auto exprs = getExprFromConfig(loopOperand, strAffineDimUmap);
      maps.push_back(AffineMap::get(nloops, 0, exprs, rewriter.getContext()));
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? resultTy : TypeRange{}, adaptor.getOperands(), output,
        maps, getEinsumLoopsAttrs(inputInd, reductionAxe),
        [reductionAxe](OpBuilder &b, Location nestedLoc, ValueRange args) {
          Value resultVal =
              b.create<mlir::arith::MulFOp>(nestedLoc, args[0], args[1]);
          if (!reductionAxe.empty()) {
            resultVal =
                b.create<mlir::arith::AddFOp>(nestedLoc, args[2], resultVal);
          }
          b.create<linalg::YieldOp>(nestedLoc, resultVal);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }

private:
  static constexpr StringLiteral kArrow = "->";
  static constexpr StringLiteral kComma = ",";
  static constexpr StringLiteral kEllipsis = "...";

  static bool checkBatchHasEqualRank(size_t lhsRank, StringRef lhsLoop,
                                     size_t rhsRank, StringRef rhsLoop,
                                     size_t outRank, StringRef outLoop);
  static SmallVector<std::string> getEinsumConfigAsVector(StringRef loop,
                                                          size_t operandRank);
};

// Convert the representation from string/vector<char> to vector<string>.
// i.e ("abc") -> {"a", "b", "c"}. For cases with ellipsis with batch rank 3:
// get loop_dim = f("ab...cde") = {"a","b","0","1","2","c","d","e"}
SmallVector<std::string>
EinsumToLinalgConverter::getEinsumConfigAsVector(StringRef loop,
                                                 size_t operandRank) {
  SmallVector<std::string> loopDim;
  size_t preElip = loop.find(kEllipsis);
  bool hasElip = preElip != StringRef::npos;
  if (!hasElip)
    preElip = loop.size();
  // Add the dimension until the end or up to ellipsis if it exist.
  for (int64_t preElipInd = 0; preElipInd < static_cast<int64_t>(preElip);
       preElipInd++) {
    loopDim.push_back(loop.substr(preElipInd, 1).str());
  }
  if (!hasElip)
    return loopDim;
  // Case where Ellipsis presence:
  size_t nonBatchRank = loop.size() - kEllipsis.size();
  size_t batchRank = operandRank - nonBatchRank;
  // Add the batch dimension ("0",...,"N") where N is rank of batch into the
  // loop.
  for (int64_t batchInd = 0; batchInd < static_cast<int64_t>(batchRank);
       batchInd++) {
    loopDim.push_back(std::to_string(batchInd));
  }
  // Add the dimension after ellipsis into the loop.
  int postElip = preElip + kEllipsis.size();
  for (int64_t postElipInd = postElip;
       postElipInd < static_cast<int64_t>(loop.size()); ++postElipInd) {
    loopDim.push_back(loop.substr(postElipInd, 1).str());
  }
  return loopDim;
}

// Returns true if all operand's batch has same rank.
bool EinsumToLinalgConverter::checkBatchHasEqualRank(
    size_t lhsRank, StringRef lhsLoop, size_t rhsRank, StringRef rhsLoop,
    size_t outRank, StringRef outLoop) {
  SmallVector<int, 3> batchRankVec;
  if (lhsRank != lhsLoop.size()) {
    size_t lhsBatchRank = lhsRank - (lhsLoop.size() - kEllipsis.size());
    batchRankVec.push_back(lhsBatchRank);
  }
  if (rhsRank != rhsLoop.size()) {
    size_t rhsBatchRank = rhsRank - (rhsLoop.size() - kEllipsis.size());
    batchRankVec.push_back(rhsBatchRank);
  }
  if (outRank != outLoop.size()) {
    size_t outBatchRank = outRank - (outLoop.size() - kEllipsis.size());
    batchRankVec.push_back(outBatchRank);
  }
  bool batchHasEqualRank = true;

  // Condition is valid if only 1 operand or less have batches.
  if (batchRankVec.size() < 2)
    return batchHasEqualRank;

  if (!llvm::all_equal(batchRankVec))
    return false;

  return batchHasEqualRank;
}

/// Base class for lowering HLO operations that have one operand and one result,
/// and are semantically equivalent to a copy of the input to the output (like
/// transpose, some reshape, etc.). The derived classes need to provide a method
/// `getIndexingMaps` that returns AffineMaps for the index maps of the input
/// and the output.
template <typename Derived, typename OpTy>
struct DataMovementOpConverter : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op)))
      return failure();

    ShapedType resultType = getHloOpResultType(op);
    resultType =
        this->getTypeConverter()->template convertType<ShapedType>(resultType);
    if (!resultType) {
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    }

    SmallVector<AffineMap, 2> indexingMaps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexingMaps.empty())
      return failure();

    int64_t nloops = resultType.getRank();
    Location loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/adaptor.getOperands().front(),
        /*outputBuffers=*/

        ValueRange{getEmptyTensorFor(rewriter, loc, resultType, op,
                                     adaptor.getOperands())},
        indexingMaps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getOperation()->getResults());
    return success();
  }
};

/// Pattern to convert BroadcastOp to Linalg ops.
template <typename OpTy>
struct BroadcastConverter final
    : DataMovementOpConverter<BroadcastConverter<OpTy>, OpTy> {
  using DataMovementOpConverter<BroadcastConverter,
                                OpTy>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcastOp,
                                                   Builder *b) {
    ShapedType inputType =
        llvm::cast<ShapedType>(broadcastOp.getOperand().getType());
    unsigned inputRank = inputType.getRank();
    unsigned nloops = getHloOpResultType(broadcastOp).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned numPrependedDims = llvm::size(broadcastOp.getBroadcastSizes());
    SmallVector<AffineExpr> inputDimExprs;
    inputDimExprs.reserve(inputRank);
    for (unsigned i = 0; i < inputRank; ++i) {
      inputDimExprs.push_back(b->getAffineDimExpr(numPrependedDims + i));
    }

    AffineMap inputMap;
    MLIRContext *context = b->getContext();
    if (inputDimExprs.empty()) {
      // The input is a scalar, i.e. this is a scalar broadcast op.
      inputMap = AffineMap::get(nloops, /*symbolCount=*/0, context);
    } else {
      inputMap =
          AffineMap::get(nloops, /*symbolCount=*/0, inputDimExprs, context);
    }
    return {inputMap, b->getMultiDimIdentityMap(nloops)};
  }
};

struct BroadcastOpToBroadcastConverter final
    : OpConversionPattern<mlir::stablehlo::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    int64_t numPrependedDims = op.getBroadcastSizes().size();
    SmallVector<int64_t> dimensions =
        llvm::to_vector(llvm::seq<int64_t>(0, numPrependedDims));

    Location loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, op.getOperand(), emptyTensor, dimensions,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

struct HloBroadcastInDimConverter final
    : DataMovementOpConverter<HloBroadcastInDimConverter,
                              mlir::stablehlo::BroadcastInDimOp> {
  using DataMovementOpConverter::DataMovementOpConverter;

  static SmallVector<AffineMap, 2>
  getIndexingMaps(mlir::stablehlo::BroadcastInDimOp broadcastOp, Builder *b) {
    ShapedType resultType = getHloOpResultType(broadcastOp);
    auto operandType = cast<ShapedType>(broadcastOp.getOperand().getType());
    unsigned nloops = resultType.getRank();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operandType.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    ArrayRef<int64_t> operandShape = operandType.getShape();
    SmallVector<AffineExpr> dimExprs;
    dimExprs.reserve(nloops);

    for (auto [idx, size] :
         llvm::enumerate(broadcastOp.getBroadcastDimensions())) {
      bool expansionNeeded =
          operandShape[idx] == 1 && resultType.getShape()[size] != 1;
      dimExprs.push_back(expansionNeeded ? b->getAffineConstantExpr(0)
                                         : b->getAffineDimExpr(size));
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dimExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

Value collapseExpandingDims(PatternRewriter &rewriter, Location loc,
                            Value operand, SmallVector<int64_t> &dimensions,
                            llvm::function_ref<bool(int64_t)> isExpandingDim) {
  auto operandTy = llvm::cast<RankedTensorType>(operand.getType());

  SmallVector<ReassociationIndices> reassociationMap;
  ReassociationIndices currentIndices;

  ArrayRef<int64_t> operandShape = operandTy.getShape();
  SmallVector<int64_t> newOperandShape;
  SmallVector<int64_t> newDimensions;

  for (auto [idx, dim] : llvm::enumerate(dimensions)) {
    currentIndices.push_back(idx);

    if (!isExpandingDim(idx)) {
      reassociationMap.push_back(currentIndices);
      currentIndices.clear();
      newOperandShape.push_back(operandShape[idx]);
      newDimensions.push_back(dim);
    }
  }

  if (!reassociationMap.empty()) {
    reassociationMap.back().insert(reassociationMap.back().end(),
                                   currentIndices.begin(),
                                   currentIndices.end());
  }

  if (dimensions.size() != newDimensions.size()) {
    dimensions = newDimensions;

    auto newOperandType =
        RankedTensorType::get(newOperandShape, operandTy.getElementType());
    operand = rewriter.create<tensor::CollapseShapeOp>(
        loc, newOperandType, operand, reassociationMap);
  }
  return operand;
}

// Insert linalg.transpose if broadcasted dimensions are not in sorted order.
// linalg.broadcast does not support implicit transpose, so the input needs to
// be explicitly transposed.
Value transposeBroadcastOperand(PatternRewriter &rewriter, Location loc,
                                Value operand,
                                SmallVector<int64_t> &dimensions) {
  // Do not insert `transpose` is dimensions are already sorted.
  if (llvm::is_sorted(dimensions))
    return operand;

  SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(0, dimensions.size()));
  llvm::sort(permutation, [&](int64_t lhs, int64_t rhs) {
    return dimensions[lhs] < dimensions[rhs];
  });

  auto operandTy = llvm::cast<ShapedType>(operand.getType());
  ArrayRef<int64_t> operandShape = operandTy.getShape();
  SmallVector<int64_t> transposedOperandShape, transposedDimensions;

  for (int64_t index : permutation) {
    transposedOperandShape.push_back(operandShape[index]);
    transposedDimensions.push_back(dimensions[index]);
  }
  dimensions = transposedDimensions;

  return rewriter.create<mlir::stablehlo::TransposeOp>(
      loc,
      RankedTensorType::get(transposedOperandShape, operandTy.getElementType()),
      operand, rewriter.getDenseI64ArrayAttr(permutation));
}

struct BroadcastInDimOpToBroadcastConverter final
    : OpConversionPattern<mlir::stablehlo::BroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<int64_t> broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions());

    Value operand = adaptor.getOperand();
    auto operandTy = llvm::cast<ShapedType>(operand.getType());
    auto resultTy =
        llvm::cast<ShapedType>(typeConverter->convertType(op.getType()));

    ArrayRef<int64_t> operandShape = operandTy.getShape();
    ArrayRef<int64_t> resultShape = resultTy.getShape();

    operand = collapseExpandingDims(
        rewriter, loc, operand, broadcastDimensions, [&](int64_t i) {
          return operandShape[i] == 1 &&
                 resultShape[broadcastDimensions[i]] != 1;
        });
    operand =
        transposeBroadcastOperand(rewriter, loc, operand, broadcastDimensions);

    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    SmallVector<int64_t> addedDimensions;
    for (int64_t dim : llvm::seq<int64_t>(0, resultTy.getRank())) {
      if (!llvm::is_contained(broadcastDimensions, dim))
        addedDimensions.push_back(dim);
    }

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, operand, emptyTensor, addedDimensions,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

// If the input has a static shape we know exactly when the broadcast must
// expand (the dimension is 1, which also trivially expands to 1) or will never
// expand (the dimension is not 1). We can also source the information from the
// optionally provided attributes on statically known broadcasting behavior.
// This means we can lower the broadcast just as we would lower a fully static
// broadcast and go directly to `linalg.generic`.

// This also covers the important case of broadcasting a scalar. Ideally the
// pattern (`stablehlo.constant` -> `stablehlo.dynamic_broadcast_in_dim`) should
// be converted to a tensor dialect op similar to TF's `ConstantLikeOp`.
struct HloDynamicBroadcastInDimConverter final
    : OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value operand = adaptor.getOperand();
    auto operandType = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandType)
      return failure();
    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();

    // Determine dimension expressions based on whether the dimension is
    // expanding (0) or non-expanding (identity), and fail if we cannot decide
    // this.
    SmallVector<AffineExpr> dimExprs(operandType.getRank(), nullptr);

    // Use static type info.
    auto bcastDims =
        llvm::map_to_vector(op.getBroadcastDimensions(),
                            [](int64_t d) { return static_cast<int64_t>(d); });
    for (auto [idx, dim] : llvm::enumerate(operandType.getShape())) {
      if (ShapedType::isDynamic(dim))
        continue;

      bool isExpanding = dim == 1;
      dimExprs[idx] = isExpanding ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(bcastDims[idx]);
    }

    // Use annotated expansion behavior, if available.
    if (auto dims = op.getKnownExpandingDimensions()) {
      for (int i : *dims) {
        dimExprs[i] = rewriter.getAffineConstantExpr(0);
      }
    }
    if (auto dims = op.getKnownNonexpandingDimensions()) {
      for (int i : *dims) {
        dimExprs[i] = rewriter.getAffineDimExpr(bcastDims[i]);
      }
    }

    // Fail if unknown expansion behavior remains.
    if (!llvm::all_of(dimExprs, [](AffineExpr expr) { return expr; }))
      return failure();

    // Materialize `linalg.generic` op.
    Location loc = op.getLoc();
    int64_t nloops = resultType.getRank();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultType, op, adaptor.getOperands());
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, TypeRange{emptyTensor.getType()}, ValueRange{operand},
        /*outputBuffers=*/ValueRange{emptyTensor},
        llvm::ArrayRef({AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0,
                                       dimExprs, rewriter.getContext()),
                        rewriter.getMultiDimIdentityMap(nloops)}),
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        },
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

struct DynamicBroadcastInDimOpToBroadcastConverter final
    : OpConversionPattern<mlir::stablehlo::DynamicBroadcastInDimOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicBroadcastInDimOp op,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value operand = adaptor.getOperand();
    auto operandTy = llvm::dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy)
      return failure();
    auto resultTy =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!resultTy)
      return failure();

    SmallVector<int64_t> broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions());

    SmallVector<std::optional<bool>> expansionBehavior(
        broadcastDimensions.size());

    // Use static type info.
    for (auto [idx, dim] : llvm::enumerate(operandTy.getShape())) {
      if (ShapedType::isDynamic(dim))
        continue;
      expansionBehavior[idx] = (dim == 1);
    }

    // Use annotated expansion behavior, if available.
    if (op.getKnownExpandingDimensions()) {
      auto dims = op.getKnownExpandingDimensions().value();
      for (int it : dims) {
        expansionBehavior[it] = true;
      }
    }
    if (op.getKnownNonexpandingDimensions()) {
      auto dims = op.getKnownNonexpandingDimensions().value();
      for (int it : dims) {
        expansionBehavior[it] = false;
      }
    }

    // Fail if unknown expansion behavior remains.
    if (llvm::any_of(expansionBehavior, [](auto v) { return !v.has_value(); }))
      return failure();

    auto isExpandingDim = [&](int64_t i) {
      return expansionBehavior[i].value();
    };

    // Use attribute information to insert 1s into operand type.
    operand = getBroadcastOperand(rewriter, loc, operand, isExpandingDim);

    auto broadcastResultTy = getBroadcastResultType(
        operand, resultTy, broadcastDimensions, isExpandingDim);

    operand = collapseExpandingDims(rewriter, loc, operand, broadcastDimensions,
                                    isExpandingDim);
    operand =
        transposeBroadcastOperand(rewriter, loc, operand, broadcastDimensions);

    Value emptyTensor = getEmptyTensorFor(rewriter, loc, broadcastResultTy, op,
                                          adaptor.getOperands());

    SmallVector<int64_t> addedDimensions;
    for (int64_t dim : llvm::seq<int64_t>(0, resultTy.getRank())) {
      if (!llvm::is_contained(broadcastDimensions, dim))
        addedDimensions.push_back(dim);
    }

    Value result = rewriter
                       .create<linalg::BroadcastOp>(
                           loc, operand, emptyTensor, addedDimensions,
                           linalg::getPrunedAttributeList(op))
                       .getResults()[0];

    if (resultTy != broadcastResultTy) {
      result = rewriter.create<tensor::CastOp>(loc, resultTy, result);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

private:
  static Value
  getBroadcastOperand(PatternRewriter &rewriter, Location loc, Value operand,
                      llvm::function_ref<bool(int64_t)> isExpandingDim) {
    auto operandTy = llvm::dyn_cast<RankedTensorType>(operand.getType());

    SmallVector<int64_t> updatedOperandShape =
        llvm::to_vector(operandTy.getShape());
    for (auto [idx, dim] : llvm::enumerate(updatedOperandShape)) {
      if (isExpandingDim(idx))
        dim = 1;
    }

    auto updatedOperandTy =
        RankedTensorType::get(updatedOperandShape, operandTy.getElementType());

    if (updatedOperandTy != operandTy) {
      operand = rewriter.create<tensor::CastOp>(loc, updatedOperandTy, operand);
    }

    return operand;
  }

  static ShapedType
  getBroadcastResultType(Value operand, RankedTensorType resultTy,
                         ArrayRef<int64_t> dimensions,
                         llvm::function_ref<bool(int64_t)> isExpandingDim) {
    auto operandShape =
        llvm::cast<RankedTensorType>(operand.getType()).getShape();
    auto broadcastResultShape = llvm::to_vector(resultTy.getShape());

    for (auto [operandIndex, resultIndex] : llvm::enumerate(dimensions)) {
      if (isExpandingDim(operandIndex))
        continue;
      broadcastResultShape[resultIndex] = operandShape[operandIndex];
    }

    return RankedTensorType::get(broadcastResultShape,
                                 resultTy.getElementType());
  }
};

template <typename OpTy>
struct TransposeConverter final
    : DataMovementOpConverter<TransposeConverter<OpTy>, OpTy> {
  using DataMovementOpConverter<TransposeConverter<OpTy>,
                                OpTy>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder *b) {
    auto resultType = llvm::cast<ShapedType>(getHloOpResultType(op));
    int64_t nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultType.getRank());
    for (auto [idx, value] : llvm::enumerate(op.getPermutation())) {
      inputExprs[value] = b->getAffineDimExpr(idx);
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

struct TransposeOpToTransposeConverter final
    : OpConversionPattern<mlir::stablehlo::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultTy)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Location loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    auto permutation = op.getPermutationAttr();

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getOperand(), emptyTensor, permutation,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

struct BitcastConvertConverter final
    : OpConversionPattern<mlir::stablehlo::BitcastConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::BitcastConvertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyHloOpBufferOrTensorSemantics(op)))
      return failure();

    auto inputType =
        llvm::cast<RankedTensorType>(adaptor.getOperand().getType());
    auto outputType =
        getTypeConverter()->convertType<RankedTensorType>(op.getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    Location loc = op.getLoc();

    // Fallback to pointwise conversion if the tensor dimensions are not
    // changing.
    if (inputType.getRank() == outputType.getRank()) {
      return failure();
    }

    auto inputBitWidth = inputType.getElementType().getIntOrFloatBitWidth();
    auto outputBitWidth = outputType.getElementType().getIntOrFloatBitWidth();

    auto maxRank = std::max(inputType.getRank(), outputType.getRank());
    auto identityMap =
        AffineMap::getMultiDimIdentityMap(maxRank, rewriter.getContext());
    AffineMap indexingMaps[] = {
        AffineMap::get(
            /*dimCount=*/maxRank, /*symbolCount=*/0,
            identityMap.getResults().take_front(inputType.getRank()),
            rewriter.getContext()),
        AffineMap::get(
            /*dimCount=*/maxRank, /*symbolCount=*/0,
            identityMap.getResults().take_front(outputType.getRank()),
            rewriter.getContext())};

    Value output =
        getEmptyTensorFor(rewriter, loc, outputType, op, adaptor.getOperands());
    bool isExpansion = inputBitWidth > outputBitWidth;
    bool isContraction = inputBitWidth < outputBitWidth;
    // When combining values we start with a 0 and merge bits into it.
    if (isContraction) {
      output = fillTensorWithZeros(rewriter, loc, output);
    }

    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, outputType, adaptor.getOperand(), output, indexingMaps,
        getParallelAndReductionIterators(maxRank, isContraction ? 1 : 0),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
          auto inIntType = nestedBuilder.getIntegerType(inputBitWidth);
          auto outIntType = nestedBuilder.getIntegerType(outputBitWidth);
          Value innerResult = args.front();
          if (isExpansion) {
            // Expand a big value into multiple small values with shifts.
            auto iotaIndex =
                nestedBuilder.create<linalg::IndexOp>(nestedLoc, maxRank - 1);
            auto iota = nestedBuilder.create<arith::IndexCastOp>(
                nestedLoc, inIntType, iotaIndex);

            auto width = nestedBuilder.create<arith::ConstantOp>(
                nestedLoc,
                nestedBuilder.getIntegerAttr(inIntType, outputBitWidth));
            auto shiftWidth =
                nestedBuilder.create<arith::MulIOp>(nestedLoc, iota, width);
            Value inputCasted = nestedBuilder.create<arith::BitcastOp>(
                nestedLoc, inIntType, args.front());
            Value shifted = nestedBuilder.create<arith::ShRUIOp>(
                nestedLoc, inputCasted, shiftWidth);
            innerResult = nestedBuilder.create<arith::TruncIOp>(
                nestedLoc, outIntType, shifted);
          } else if (isContraction) {
            // Combine multiple small values into one big value.
            auto iotaIndex =
                nestedBuilder.create<linalg::IndexOp>(nestedLoc, maxRank - 1);
            auto iota = nestedBuilder.create<arith::IndexCastOp>(
                nestedLoc, outIntType, iotaIndex);

            auto width = nestedBuilder.create<arith::ConstantOp>(
                nestedLoc,
                nestedBuilder.getIntegerAttr(outIntType, inputBitWidth));
            auto shiftWidth =
                nestedBuilder.create<arith::MulIOp>(nestedLoc, iota, width);
            Value inputCasted = nestedBuilder.create<arith::BitcastOp>(
                nestedLoc, inIntType, args.front());
            Value inputExt = nestedBuilder.create<arith::ExtUIOp>(
                nestedLoc, outIntType, inputCasted);
            Value shifted = nestedBuilder.create<arith::ShLIOp>(
                nestedLoc, inputExt, shiftWidth);
            Value accumulatorCasted = nestedBuilder.create<arith::BitcastOp>(
                nestedLoc, outIntType, args.back());
            innerResult = nestedBuilder.create<arith::OrIOp>(
                nestedLoc, outIntType, shifted, accumulatorCasted);
          }
          innerResult = nestedBuilder.create<arith::BitcastOp>(
              nestedLoc, outputType.getElementType(), innerResult);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, innerResult);
        },
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

// Lowers stablehlo.RealDynamicSliceOp to tensor.extract_slice and other
// arith/tensor dialect ops.
struct RealDynamicSliceConverter final
    : OpConversionPattern<mlir::stablehlo::RealDynamicSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  // Computes size of a slice as
  //   size = ceil((limit - start)/stride)
  static Value computeSize(Location loc, Value start, Value limit, Value stride,
                           ConversionPatternRewriter &b) {
    Value delta = b.create<arith::SubIOp>(loc, limit, start);
    Value ret = b.create<arith::CeilDivUIOp>(loc, delta, stride);
    if (ret.getType().isIndex())
      return ret;
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), ret);
  }

  LogicalResult
  matchAndRewrite(mlir::stablehlo::RealDynamicSliceOp realDynamicSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = realDynamicSliceOp.getLoc();
    auto argType = llvm::dyn_cast<ShapedType>(adaptor.getOperand().getType());
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(realDynamicSliceOp,
                                         "require known-rank args");
    }

    Type dimElementType = getElementTypeOrSelf(adaptor.getStartIndices());
    if (getElementTypeOrSelf(adaptor.getLimitIndices()) != dimElementType ||
        getElementTypeOrSelf(adaptor.getStrides()) != dimElementType) {
      return rewriter.notifyMatchFailure(
          realDynamicSliceOp,
          "requires same element type for all dimension specification");
    }
    Type arithType =
        dimElementType.isIndex() ? rewriter.getI64Type() : dimElementType;
    Type indexType = rewriter.getIndexType();

    auto resultType = llvm::cast<RankedTensorType>(
        this->typeConverter->convertType(realDynamicSliceOp.getType()));
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<OpFoldResult> offsets, sizes, strides;
    SmallVector<Type, 3> clampType(3, arithType);
    for (auto i : llvm::seq<unsigned>(0, argType.getRank())) {
      Value dim = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value start = rewriter.create<tensor::ExtractOp>(
          loc, adaptor.getStartIndices(), dim);
      Value limit = rewriter.create<tensor::ExtractOp>(
          loc, adaptor.getLimitIndices(), dim);
      Value stride =
          rewriter.create<tensor::ExtractOp>(loc, adaptor.getStrides(), dim);

      // Compute i-th dimension size of the result : size[i].
      // If the i-th dimension of the result type is known, we go ahead with it
      // else we compute it using limit, start and stride values.
      int64_t resultDimSize = resultType.getDimSize(i);
      Value size =
          ShapedType::isDynamic(resultDimSize)
              ? computeSize(loc, start, limit, stride, rewriter)
              : rewriter.create<arith::ConstantIndexOp>(loc, resultDimSize);

      // We can now convert start to index.
      if (!start.getType().isIndex())
        start = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), start);

      // Fetch i-th dimension size of the operand and calculate upper bound as
      //   ub = operand_dim[i] - size[i]
      Value operandDimSize =
          rewriter.createOrFold<tensor::DimOp>(loc, adaptor.getOperand(), dim);
      Value upperBound =
          rewriter.createOrFold<arith::SubIOp>(loc, operandDimSize, size);

      // We clamp the start_index to keep it bounded as
      //   0 <= start_index[i] <= ub
      // Clamp does not support index type, so cast to integer type.
      start = rewriter.create<arith::MaxSIOp>(loc, start, zero);
      start = rewriter.create<arith::MinSIOp>(loc, start, upperBound);

      offsets.push_back(start);
      if (ShapedType::isDynamic(resultDimSize))
        sizes.push_back(size);
      else
        sizes.push_back(IntegerAttr::get(indexType, resultDimSize));

      if (!stride.getType().isIndex())
        stride =
            rewriter.createOrFold<arith::IndexCastOp>(loc, indexType, stride);
      strides.push_back(stride);
    }

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        realDynamicSliceOp, resultType, adaptor.getOperand(), offsets, sizes,
        strides);
    return success();
  }
};

// Converts reshape ops that can be proven to be either a collapse of dimensions
// or expansion of dimensions of the operand.
struct ReshapeOpConverter final
    : OpConversionPattern<mlir::stablehlo::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ReshapeOp reshapeOp,
                  mlir::stablehlo::ReshapeOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyHloOpBufferOrTensorSemantics(reshapeOp)))
      return failure();
    Value operand = adaptor.getOperand();
    auto operandType = llvm::cast<ShapedType>(operand.getType());
    Type elemType = operandType.getElementType();
    auto resultType = llvm::cast<ShapedType>(reshapeOp.getType());

    if (!resultType.hasStaticShape())
      return failure();

    // If any of the output dimensions is 0, the tensor has no elements. In that
    // case, we can just replace the reshape with an empty op.
    if (llvm::is_contained(resultType.getShape(), 0)) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
          reshapeOp, resultType.getShape(), elemType);
      return success();
    }

    resultType = getTypeConverter()->convertType<ShapedType>(resultType);
    if (!resultType)
      return rewriter.notifyMatchFailure(reshapeOp, "type conversion failed");

    // Special case where the result is a scalar.
    if (resultType.getRank() == 0 && !operandType.hasStaticShape()) {
      // This means all dimensions of the operand need to be 1. We add a cast to
      // cast the dynamic dimensions to 1.
      auto staticType = RankedTensorType::get(
          llvm::SmallVector<int64_t>(operandType.getRank(), 1), elemType);
      operand = rewriter.create<tensor::CastOp>(reshapeOp.getLoc(), staticType,
                                                operand);
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          reshapeOp, resultType, operand, ArrayRef<ReassociationIndices>{});
      return success();
    }

    // Compute the reassociation maps for the linalg operation. This will
    // succeed if the reshape can be done with a single expand_shape or
    // collapse_shape.
    if (std::optional<SmallVector<ReassociationIndices>> reassociationMap =
            getReassociationIndicesForReshape(operandType, resultType)) {
      if (resultType.getRank() < operandType.getRank()) {
        // We have found a working reassociation map. If the operand is dynamic,
        // we first need to cast all unknown dimensions in the input that get
        // collapsed to a static-sized dimension in the output, to 1.
        SmallVector<int64_t> shape(operandType.getShape().begin(),
                                   operandType.getShape().end());
        for (auto [idx, dims] : llvm::enumerate(*reassociationMap)) {
          // If the result dim is dynamic, we do not mind dynamic entries in the
          // source.
          if (resultType.isDynamicDim(idx))
            continue;
          for (auto targetDim : dims) {
            if (ShapedType::isDynamic(shape[targetDim]))
              shape[targetDim] = 1;
          }
        }
        // Insert a cast if types are not the same (ignoring sparse encoding).
        auto enc = sparse_tensor::getSparseTensorEncoding(operandType);
        auto newOperandType = RankedTensorType::get(shape, elemType, enc);
        if (newOperandType != operandType) {
          operand = rewriter.create<tensor::CastOp>(reshapeOp.getLoc(),
                                                    newOperandType, operand);
        }
        // Generate collapse operation.
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            reshapeOp, resultType, operand, *reassociationMap);
      } else {
        // Generate expand operation.
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            reshapeOp, resultType, operand, *reassociationMap);
      }
      return success();
    }

    Value collapsedOp = operand;
    Location loc = reshapeOp.getLoc();
    auto getIdentityExprs = [&rewriter](int64_t n) {
      SmallVector<AffineExpr> exprs;
      for (int i = 0; i < n; ++i)
        exprs.push_back(rewriter.getAffineDimExpr(i));
      return exprs;
    };
    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions. If there is only a single
    // source dimension, the reduce step can be skipped. TensorCollapseShape
    // expects a different rank of operand and result.
    if (operandType.getRank() != 1) {
      SmallVector<ReassociationExprs> collapsingMap = {
          // Use operand_type here because we need to collapse all operands
          // dimensions.
          getIdentityExprs(operandType.getRank())};

      collapsedOp =
          rewriter.create<tensor::CollapseShapeOp>(loc, operand, collapsingMap);
    }
    // Cast to a known static type if the input has dynamic dimensions.
    int64_t totalElems = resultType.getNumElements();
    auto collapsedType = RankedTensorType::get({totalElems}, elemType);
    collapsedOp =
        rewriter.create<tensor::CastOp>(loc, collapsedType, collapsedOp);
    if (resultType.getRank() == 1) {
      rewriter.replaceOp(reshapeOp, collapsedOp);
    } else {
      SmallVector<ReassociationExprs> expandingMap = {
          // Use resultType here because we need to expand to all result
          // dimensions.
          getIdentityExprs(resultType.getRank())};
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          reshapeOp, resultType, collapsedOp, expandingMap);
    }
    return success();
  }
};

template <typename OpTy>
struct IotaConverter final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using Adaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy iotaOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ShapedType resultShapedType = getHloOpResultType(iotaOp);
    if (!resultShapedType)
      return failure();

    resultShapedType =
        this->getTypeConverter()->template convertType<ShapedType>(
            resultShapedType);
    if (!resultShapedType)
      return rewriter.notifyMatchFailure(iotaOp, "type conversion failed");

    Type resultElementType = resultShapedType.getElementType();

    // Construct the indexing maps needed for linalg.generic ops.
    unsigned nloops = resultShapedType.getRank();

    Location loc = iotaOp.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/
        ArrayRef<Type>{resultShapedType},
        /*inputs=*/ValueRange{},
        /*outputBuffers=*/

        ValueRange{getEmptyTensorFor(rewriter, loc, resultShapedType, iotaOp,
                                     adaptor.getOperands())},
        llvm::ArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
          Value indexOp = nestedBuilder.create<linalg::IndexOp>(
              nestedLoc, iotaOp.getIotaDimension());
          Type unwrappedResultElementType = resultElementType;
          if (auto complexType =
                  llvm::dyn_cast<ComplexType>(unwrappedResultElementType))
            unwrappedResultElementType = complexType.getElementType();
          Value castOp = nestedBuilder.create<arith::IndexCastOp>(
              nestedLoc,
              nestedBuilder.getIntegerType(
                  unwrappedResultElementType.getIntOrFloatBitWidth()),
              indexOp);
          castOp = mlir::stablehlo::StableHloOpToStdScalarOp::mapOpOfType<
              mlir::stablehlo::ConvertOp>(nestedLoc, resultElementType,
                                          castOp.getType(), {castOp},
                                          &nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, castOp);
        },
        linalg::getPrunedAttributeList(iotaOp));
    rewriter.replaceOp(iotaOp, linalgOp.getResultTensors());
    return success();
  }
};

template <typename OpTy>
struct IotaToMapConverter final : OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  using Adaptor = typename OpTy::Adaptor;

  LogicalResult
  matchAndRewrite(OpTy iotaOp, Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ShapedType resultTy = getHloOpResultType(iotaOp);
    if (!resultTy)
      return failure();

    resultTy =
        this->getTypeConverter()->template convertType<ShapedType>(resultTy);
    if (!resultTy)
      return rewriter.notifyMatchFailure(iotaOp, "type conversion failed");

    Location loc = iotaOp.getLoc();
    Value empty = getEmptyTensorFor(rewriter, loc, resultTy, iotaOp,
                                    adaptor.getOperands());

    auto linalgOp = rewriter.create<linalg::MapOp>(
        loc, ValueRange{}, empty,
        [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
          Value index = nestedBuilder.create<linalg::IndexOp>(
              nestedLoc, iotaOp.getIotaDimension());
          index = nestedBuilder.create<arith::IndexCastOp>(
              nestedLoc, nestedBuilder.getI64Type(), index);
          Value result = mlir::stablehlo::StableHloOpToStdScalarOp::mapOpOfType<
              mlir::stablehlo::ConvertOp>(nestedLoc, resultTy.getElementType(),
                                          index.getType(), {ValueRange{index}},
                                          &nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, ValueRange{result});
        },
        linalg::getPrunedAttributeList(iotaOp));
    rewriter.replaceOp(iotaOp, linalgOp.getResult());
    return success();
  }
};

/// Converts stablehlo.concatenate operation to a linalg.generic op.
struct ConcatenateConverter final
    : OpConversionPattern<mlir::stablehlo::ConcatenateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConcatenateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Shortcut the one-operand case, simplifies code below.
    if (adaptor.getOperands().size() == 1) {
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }

    auto resultType = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    uint64_t dim = op.getDimension();
    Location loc = op.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Allocate the output tensor with tensor.empty.
    Value result =
        getEmptyTensorFor(rewriter, loc, resultType, op, adaptor.getOperands());

    // Generate a generic op to gather the elements of the concatenate. This is
    // awkward standalone but allows fusion with other generic ops.
    int64_t nloops = resultType.getRank();
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/ValueRange{}, /*outputBuffers=*/result,
        llvm::ArrayRef(rewriter.getMultiDimIdentityMap(nloops)),
        getNParallelLoopsAttrs(nloops),
        [&](OpBuilder &nestedBuilder, Location loc, ValueRange) {
          OpBuilder b = nestedBuilder;
          Value concatDimSize = zero;
          Value result;

          SmallVector<Value> extractIndices;
          extractIndices.reserve(nloops);
          for (int64_t i = 0; i < nloops; i++) {
            extractIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }

          Value indexOp = b.create<linalg::IndexOp>(loc, dim);
          for (auto [idx, arg] : llvm::enumerate(adaptor.getOperands())) {
            Value newConcatDimSize;
            scf::IfOp ifOp;
            if (idx + 1 != adaptor.getOperands().size()) {
              // Calculate how far along we have iterated along the concatenate
              // dimension. That way we can tell which input to select.
              newConcatDimSize = b.create<arith::AddIOp>(
                  loc, concatDimSize, b.create<tensor::DimOp>(loc, arg, dim));
              Value cmp = b.create<arith::CmpIOp>(loc, rewriter.getI1Type(),
                                                  arith::CmpIPredicate::ult,
                                                  indexOp, newConcatDimSize);
              ifOp = b.create<scf::IfOp>(loc, resultType.getElementType(), cmp,
                                         true);
              if (result) {
                b.create<scf::YieldOp>(loc, ifOp->getResults()[0]);
              } else {
                result = ifOp->getResults()[0];
              }

              b = ifOp.getThenBodyBuilder(b.getListener());
            }

            // Now adjust the index for the concatenated dimension to fit into
            // the selected tensor and do an extract at that position.
            extractIndices[dim] =
                b.create<arith::SubIOp>(loc, indexOp, concatDimSize);
            Value extract =
                b.create<tensor::ExtractOp>(loc, arg, extractIndices);
            b.create<scf::YieldOp>(loc, extract);

            if (ifOp) {
              b = ifOp.getElseBodyBuilder(b.getListener());
              concatDimSize = newConcatDimSize;
            }
          }
          nestedBuilder.create<linalg::YieldOp>(loc, result);
        },
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

struct ConstConverterTensor final
    : OpConversionPattern<mlir::stablehlo::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::ConstantOp constOp, OpAdaptor /*adaptor*/,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacementType =
        getTypeConverter()->convertType<ShapedType>(constOp.getType());
    if (!replacementType)
      return rewriter.notifyMatchFailure(constOp, "type conversion failed");

    ElementsAttr replacementAttr = constOp.getValue();
    if (replacementType == constOp.getType()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, replacementType,
                                                     replacementAttr);
      return success();
    } else {
      auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!denseAttr) {
        return rewriter.notifyMatchFailure(
            constOp,
            "DenseElementsAttr cast failed (only DenseElementsAttr supported)");
      }
      // Signedness conversion.
      // TODO(#15442): Add generic mapping utility, so we aren't limited to
      // supporting only DenseElementsAttr.
      replacementAttr = denseAttr.mapValues(replacementType.getElementType(),
                                            [](const APInt &i) { return i; });
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, replacementType,
                                                     replacementAttr);
      return success();
    }
  }
};

// TODO(b/156787842): Support the lowering for dynamic shapes.
struct ReverseConverter final
    : DataMovementOpConverter<ReverseConverter, mlir::stablehlo::ReverseOp> {
  using DataMovementOpConverter::DataMovementOpConverter;

  static SmallVector<AffineMap, 2>
  getIndexingMaps(mlir::stablehlo::ReverseOp op, Builder *b) {
    auto resultType = llvm::cast<ShapedType>(getHloOpResultType(op));
    int64_t nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.reserve(nloops);
    for (int64_t i = 0; i < nloops; ++i)
      inputExprs.push_back(b->getAffineDimExpr(i));
    for (int i : op.getDimensions()) {
      if (resultType.isDynamicDim(i))
        return {};
      int n = resultType.getShape()[i];
      inputExprs[i] = b->getAffineConstantExpr(n - 1) - inputExprs[i];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

struct SliceConverter final : OpConversionPattern<mlir::stablehlo::SliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::SliceOp sliceOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto argType =
        llvm::dyn_cast<ShapedType>(adaptor.getOperands()[0].getType());
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expects known-rank args");
    }

    SmallVector<OpFoldResult, 3> offsets, sizes, strides;
    auto startIndices = sliceOp.getStartIndices();
    auto limitIndices = sliceOp.getLimitIndices();
    auto sliceStrides = sliceOp.getStrides();

    for (int64_t i = 0, e = argType.getRank(); i < e; ++i) {
      int64_t start = startIndices[i];
      int64_t limit = limitIndices[i];
      int64_t stride = sliceStrides[i];
      offsets.push_back(rewriter.getI64IntegerAttr(start));
      // Say that there are k elements in total, we have condition:
      //   start + (k - 1) * strides <= limit - 1
      // ->
      //   k <= (limit - 1 - start + strides) / strides
      sizes.push_back(
          rewriter.getI64IntegerAttr((limit - 1 - start + stride) / stride));
      strides.push_back(rewriter.getI64IntegerAttr(stride));
    }
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        sliceOp, adaptor.getOperands()[0], offsets, sizes, strides);
    return success();
  }
};

struct DynamicSliceConverter final
    : OpConversionPattern<mlir::stablehlo::DynamicSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicSliceOp dynamicSliceOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = dynamicSliceOp.getLoc();
    auto argType = llvm::dyn_cast<ShapedType>(adaptor.getOperand().getType());
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(dynamicSliceOp,
                                         "require known-rank args");
    }

    auto resultType = getTypeConverter()->convertType<RankedTensorType>(
        dynamicSliceOp.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(dynamicSliceOp,
                                         "type conversion failed");

    SmallVector<OpFoldResult, 3> startIndices, sizes;
    auto originalStartIndexType = llvm::cast<ShapedType>(
        dynamicSliceOp.getStartIndices().front().getType());
    for (auto [idx, start, size] : llvm::enumerate(
             adaptor.getStartIndices(), dynamicSliceOp.getSliceSizes())) {
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      // By stablehlo.DynamicSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - size_indices[i])`
      Value startIndex =
          extractIndexFromTensor(rewriter, loc, start, originalStartIndexType);

      Value mn = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      Value mx =
          rewriter.createOrFold<tensor::DimOp>(loc, adaptor.getOperand(), idx);
      mx = rewriter.createOrFold<arith::SubIOp>(
          loc, mx, rewriter.create<arith::ConstantIndexOp>(loc, size));

      startIndex = rewriter.create<arith::MaxSIOp>(loc, startIndex, mn);
      startIndex = rewriter.create<arith::MinSIOp>(loc, startIndex, mx);

      startIndices.push_back(startIndex);
    }

    int64_t rank = argType.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        dynamicSliceOp, resultType, adaptor.getOperand(), startIndices, sizes,
        strides);
    return success();
  }
};

struct DynamicUpdateSliceConverter final
    : OpConversionPattern<mlir::stablehlo::DynamicUpdateSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto operandType =
        llvm::dyn_cast<RankedTensorType>(adaptor.getOperand().getType());
    if (!operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    auto updateType =
        llvm::dyn_cast<RankedTensorType>(adaptor.getUpdate().getType());
    if (!updateType || !updateType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    // We do not have to clamp sizes because the semantic of `update`
    // guarantees that it is always in the bounds. See
    // https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
    SmallVector<OpFoldResult, 3> sizes;
    for (int64_t size : updateType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(size));
    }

    SmallVector<OpFoldResult, 3> startIndices;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (auto [idx, start] : llvm::enumerate(adaptor.getStartIndices())) {
      // By stablehlo.DynamicUpdateSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - update.dimension_size[i])`
      Value startIndex = extractIndexFromTensor(
          rewriter, loc, start,
          cast<ShapedType>(op.getStartIndices()[idx].getType()));
      Value ub = rewriter.create<arith::ConstantIndexOp>(
          loc, operandType.getDimSize(idx) - updateType.getDimSize(idx));

      startIndex = rewriter.create<arith::MaxSIOp>(loc, startIndex, zero);
      startIndex = rewriter.create<arith::MinSIOp>(loc, startIndex, ub);
      startIndices.push_back(startIndex);
    }

    int64_t rank = operandType.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getUpdate(), adaptor.getOperand(), startIndices, sizes,
        strides);
    return success();
  }
};

struct MapOpToGenericConverter final
    : OpConversionPattern<mlir::stablehlo::MapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::MapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyHloOpBufferOrTensorSemantics(op)))
      return failure();

    auto resultType = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    assert(op.getDimensions().size() == resultType.getRank() &&
           "Expected a pointwise map");

    Location loc = op.getLoc();
    Value output =
        getEmptyTensorFor(rewriter, loc, resultType, op, adaptor.getOperands());
    SmallVector<AffineMap> indexingMaps(
        op.getNumOperands() + 1,
        rewriter.getMultiDimIdentityMap(resultType.getRank()));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, adaptor.getOperands(), output, indexingMaps,
        getNParallelLoopsAttrs(resultType.getRank()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. We scalarize the operands and add a
    // scalar operand representing the output tensor.
    Region &region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getComputation(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(op.getNumOperands() +
                                                          1);

    for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
      Type convertedTy = getTypeConverter()->convertType(
          cast<ShapedType>(operand.getType()).getElementType());
      if (!convertedTy)
        return rewriter.notifyMatchFailure(op,
                                           "operand type conversion failed");

      signatureConverter.addInputs(idx, convertedTy);
    }
    signatureConverter.addInputs(resultType.getElementType());

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct MapOpToMapConverter final : OpConversionPattern<mlir::stablehlo::MapOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::MapOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (failed(verifyHloOpBufferOrTensorSemantics(op)))
      return failure();

    auto resultType = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");
    assert(op.getDimensions().size() == resultType.getRank() &&
           "Expected a pointwise map");

    Location loc = op.getLoc();
    Value operand0 = adaptor.getOperands()[0];
    SmallVector<Value> coercedOperands = {operand0};
    for (Value operand : llvm::drop_begin(adaptor.getOperands(), 1)) {
      coercedOperands.push_back(coerceTensorShape(
          rewriter, loc, cast<TypedValue<ShapedType>>(operand),
          cast<ShapedType>(operand0.getType())));
    }
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, operand0),
        resultType.getElementType());

    auto linalgOp = rewriter.create<linalg::MapOp>(
        loc, coercedOperands, output,
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. We scalarize the operands and add a
    // scalar operand representing the output tensor.
    Region &region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getComputation(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(op.getNumOperands());

    for (auto [idx, operand] : llvm::enumerate(op.getOperands())) {
      Type convertedTy = getTypeConverter()->convertType(
          cast<ShapedType>(operand.getType()).getElementType());
      if (!convertedTy)
        return rewriter.notifyMatchFailure(op,
                                           "operand type conversion failed");
      signatureConverter.addInputs(idx, convertedTy);
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    auto result = rewriter.createOrFold<tensor::CastOp>(loc, resultType,
                                                        linalgOp.getResults());
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// This lowering encompasses the full range of the Gather operation and
/// therefore is very general and just loops over the output and calculate the
/// corresponding input index. It follows the explanation at
/// https://www.tensorflow.org/xla/operation_semantics#gather. The compiler
/// should be able to optimize that a bit, but in order to get efficient
/// lowerings, special-cases of gather should be extracted in separate
/// lowerings, and ideally encapsulated as separate ops or canonicalization
/// patterns.
struct GatherConversion final : OpConversionPattern<mlir::stablehlo::GatherOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::GatherOp gatherOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();

    Value startIndices = adaptor.getStartIndices();
    Value operand = adaptor.getOperand();

    auto resultType =
        getTypeConverter()->convertType<RankedTensorType>(gatherOp.getType());
    RankedTensorType startIndicesType =
        dyn_cast<RankedTensorType>(startIndices.getType());
    // We could actually deal with an unranked result by inferring the result
    // rank, but the current reifyReturnTypes doesn't support unranked either.
    if (!resultType || !startIndicesType) {
      return rewriter.notifyMatchFailure(gatherOp,
                                         "unranked start indices or result");
    }

    int64_t resultRank = resultType.getRank();
    // slice_sizes has to have the same size as operand.rank, and doing it this
    // way permits an unranked operand.
    int64_t operandRank = gatherOp.getSliceSizes().size();

    int64_t indexVectorDim = gatherOp.getDimensionNumbers().getIndexVectorDim();

    ArrayRef<int64_t> offsetDims =
        gatherOp.getDimensionNumbers().getOffsetDims();
    ArrayRef<int64_t> collapsedSliceDims =
        gatherOp.getDimensionNumbers().getCollapsedSliceDims();
    ArrayRef<int64_t> startIndexMap =
        gatherOp.getDimensionNumbers().getStartIndexMap();

    // We'll need these later and creating them on demand we end up with
    // duplicates, which also makes lit tests really hard to write.
    SmallVector<Value> constants;
    for (int64_t i = 0, e = std::max({resultRank, operandRank, int64_t{2}});
         i < e; ++i) {
      constants.push_back(
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(i)));
    }

    Value emptyOp = getEmptyTensorFor(rewriter, loc, resultType, gatherOp,
                                      adaptor.getOperands());

    ValueRange ins;
    SmallVector<AffineMap, 1> indexingMaps(
        {rewriter.getMultiDimIdentityMap(resultRank)});
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/resultType,
        /*inputs=*/ins,
        /*outputs=*/emptyOp, indexingMaps, getNParallelLoopsAttrs(resultRank),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(gatherOp));

    // Now populate the linalg generic region
    Region &region = linalgOp.getRegion();
    Block *block = rewriter.createBlock(&region, region.end());
    block->addArguments(resultType.getElementType(), loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    // Dimensions in the result that aren't offset dimensions are called batch.
    SmallVector<int64_t> batchDims;
    for (int64_t dim = 0; dim < resultRank; ++dim) {
      if (!llvm::is_contained(offsetDims, dim)) {
        batchDims.push_back(dim);
      }
    }

    // Same as with the constants. Creating these all up front is easier than
    // potentially getting duplicates later.
    SmallVector<Value> linalgIndices;
    for (int64_t i = 0; i < resultRank; ++i) {
      linalgIndices.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    }

    // Now the complicated part. For a given output dimension we build up an
    // index into the input. It's composed of two parts: the index coming from
    // start_indices, and the offset from that index along the offset
    // dimensions. Everything includes dimension shuffling and remapping as well
    // because of the way gather is defined to allow for any-layout input by
    // adding more attributes.

    // The base gather index (`G` in the documentation) points to a place in
    // start_indices along the batch dimensions.
    SmallVector<Value> gatherIndex;
    for (int64_t dim : batchDims) {
      gatherIndex.push_back(linalgIndices[dim]);
    }

    SmallVector<Value> indexFromStartIndices;
    for (size_t i = 0, e = startIndexMap.size(); i != e; ++i) {
      // The index along the index_vector dimension of start_indices varies.
      // Basically indexFromStartIndices indexes into a "row" along
      // index_vector_dim, where the row is selected by the current output
      // index.
      // But if index_vector_dim is equal to start_indices.rank, then
      // start_indices gets a trailing 1 dimension added. So the row we're
      // extracting always has length 1 and the index into it is always 0, so we
      // just use the gather index directly
      SmallVector<Value> gCombine(gatherIndex);
      if (indexVectorDim != startIndicesType.getRank()) {
        assert(indexVectorDim <= static_cast<int64_t>(gCombine.size()));
        gCombine.insert(gCombine.begin() + indexVectorDim, constants[i]);
      }

      indexFromStartIndices.push_back(extractIndexFromTensor(
          rewriter, loc, startIndices, gatherOp.getStartIndices().getType(),
          gCombine));
    }

    // But then start indices are shuffled by the start index map. To make a
    // full index into the operand, all missing indices are zeroes.
    SmallVector<Value> remappedIndexFromIndices(operandRank, constants[0]);
    for (auto [idx, value] : llvm::enumerate(startIndexMap)) {
      remappedIndexFromIndices[value] = indexFromStartIndices[idx];
    }

    // Now we construct the index based on the offset. First we need to remap
    // the offset dimensions by dropping the collapsed indices.
    SmallVector<unsigned> remappedOffsetDims;
    for (int64_t i = 0; i < operandRank; ++i) {
      if (!llvm::is_contained(collapsedSliceDims, i)) {
        remappedOffsetDims.push_back(static_cast<unsigned>(i));
      }
    }

    assert(remappedOffsetDims.size() == offsetDims.size());

    // Clamp out of bounds indices.
    for (int i = 0, operandIndexDim = 0; i < operandRank; ++i) {
      // Compute the size of the output shape dimension corresponding to this
      // index dimension. If it's collapsed set it to 1.
      Value outputDimSize = constants[1];
      if (!llvm::is_contained(collapsedSliceDims, i)) {
        outputDimSize = rewriter.createOrFold<tensor::DimOp>(
            loc, emptyOp, offsetDims[operandIndexDim++]);
      }

      // If this is a skipped dimension, we're done and don't have to clamp.
      if (remappedIndexFromIndices[i] == constants[0])
        continue;

      Value operandDimSize =
          rewriter.createOrFold<tensor::DimOp>(loc, operand, i);
      Value largestValidIndex = rewriter.createOrFold<arith::SubIOp>(
          loc, operandDimSize, outputDimSize);

      // Clamp indices to [0, i, operand_dim-output_dim].
      Value clamp = rewriter.create<arith::MinSIOp>(
          loc,
          rewriter.create<arith::MaxSIOp>(loc, constants[0],
                                          remappedIndexFromIndices[i]),
          largestValidIndex);
      remappedIndexFromIndices[i] = clamp;
    }

    // For the (remapped) offset dimensions, the index is the current index in
    // the output. As before this is expanded to a full index into the operand
    // by using zeros for the missing indices.
    SmallVector<Value> indexFromOffset(operandRank, constants[0]);
    for (auto [remappedOffsetDim, offsetDim] :
         llvm::zip_equal(remappedOffsetDims, offsetDims)) {
      indexFromOffset[remappedOffsetDim] = linalgIndices[offsetDim];
    }

    // Now we add together our two indices to get the final index into the
    // operand.
    SmallVector<Value> combinedIndex;
    for (int64_t i = 0; i < operandRank; ++i)
      combinedIndex.push_back(rewriter.createOrFold<arith::AddIOp>(
          loc, rewriter.getIndexType(), remappedIndexFromIndices[i],
          indexFromOffset[i]));

    Value extractOperand;
    if (isa<RankedTensorType>(operand.getType())) {
      extractOperand = operand;
    } else {
      // Cannot extract from unranked tensors, cast to ranked first.
      SmallVector<int64_t> dims(operandRank, ShapedType::kDynamic);
      auto type = RankedTensorType::get(
          dims, cast<TensorType>(operand.getType()).getElementType());
      extractOperand = rewriter.create<tensor::CastOp>(loc, type, operand);
    }
    Value element =
        rewriter.create<tensor::ExtractOp>(loc, extractOperand, combinedIndex);
    rewriter.create<linalg::YieldOp>(loc, element);

    rewriter.replaceOp(gatherOp, linalgOp.getResults());

    return success();
  }
};

/// Converts xla-hlo.select_and_scatter op to a sequence of linalg.generics ops.
/// The current version computes the scattered index and populates the correct
/// value for each tile. It does not currently handle overlapping tiles.
struct SelectAndScatterNoOverlapConverter final
    : OpConversionPattern<mlir::stablehlo::SelectAndScatterOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::SelectAndScatterOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value source = op.getSource();
    Value operand = op.getOperand();
    Value init = op.getInitValue();

    auto sourceTy = llvm::dyn_cast<RankedTensorType>(source.getType());
    auto operandTy = llvm::dyn_cast<RankedTensorType>(operand.getType());
    auto initTy = llvm::dyn_cast<RankedTensorType>(init.getType());
    auto resultTy = llvm::dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!sourceTy || !operandTy || !initTy || !resultTy)
      return rewriter.notifyMatchFailure(op, "inputs/outputs must be ranked");

    auto indexETy = b.getI32Type();
    auto srcETy = operandTy.getElementType();
    auto destETy = initTy.getElementType();

    const int64_t rank = sourceTy.getRank();

    llvm::SmallVector<int64_t> pad(rank * 2, 0);
    if (op.getPadding().has_value())
      pad = llvm::to_vector(op.getPaddingAttr().getValues<int64_t>());

    // TODO(suderman): Add support for padding.
    if (llvm::any_of(pad, [](int64_t p) { return p != 0; }))
      return rewriter.notifyMatchFailure(op, "non-zero padding values found.");

    if (!op.getWindowStrides().has_value())
      return rewriter.notifyMatchFailure(op, "no window strides found");

    if (!op.getWindowDimensions().has_value())
      return rewriter.notifyMatchFailure(op, "no window dimensions found");

    auto strides = llvm::to_vector(op.getWindowStrides().value());
    auto window = llvm::to_vector(op.getWindowDimensions().value());

    if (static_cast<int64_t>(strides.size()) != operandTy.getRank() ||
        static_cast<int64_t>(window.size()) != operandTy.getRank())
      return rewriter.notifyMatchFailure(
          op, "stride/window length should equal operand rank");

    // The current version cannot handle overlapped regions.
    for (int i = 0, s = strides.size(); i < s; ++i) {
      if (strides[i] < window[i])
        return rewriter.notifyMatchFailure(
            op, "overlapping windows are not supported");
    }

    // If the window only contains a single element, this lowering will be
    // problematic. Ultimately we should handle this with a canonicalizer.
    if (llvm::all_of(window, [](auto sz) { return sz == 1; })) {
      return rewriter.notifyMatchFailure(op,
                                         "unary window size is not supported");
    }

    // The first linalg.generic operation computes the relevant index over
    // window for the defined stablehlo.select_and_scatter. This involves
    // iterating over the window of the operand a computing the index.
    // Rather than storing N indices we compute the row major identifier
    // in the window, to specify which location should be scattered to.

    // Output specifies the `rank` parallel iterators that correspond to
    // output values.
    SmallVector<AffineExpr> outputExprs;
    for (int i = 0, s = rank; i < s; ++i)
      outputExprs.push_back(b.getAffineDimExpr(i));

    // For the output we need to define the reduction across the window
    // width and height. This includes applying striding behavior and
    // adding the additional reduction iterators. We skip length-1 dimensions
    // as the reduction is degenerate.
    SmallVector<int64_t> filteredWindows, filteredStrides;
    SmallVector<AffineExpr> sourceExprs(outputExprs);
    SmallVector<AffineExpr> windowExprs;
    for (int i = 0, s = rank; i < s; ++i) {
      sourceExprs[i] = sourceExprs[i] * strides[i];
      if (strides[i] != 1) {
        auto expr = b.getAffineDimExpr(windowExprs.size() + sourceExprs.size());
        sourceExprs[i] = sourceExprs[i] + expr;
        windowExprs.push_back(expr);
        filteredWindows.push_back(window[i]);
        filteredStrides.push_back(strides[i]);
      }
    }

    // Determine the total number of AffineExprs and construct the IndexingMaps
    // for the windowed reduction operation.
    const int64_t reduceExprCount = windowExprs.size() + sourceExprs.size();
    SmallVector<AffineMap, 2> reduceIndexingMaps;
    reduceIndexingMaps.push_back(AffineMap::get(reduceExprCount,
                                                /*symbolCount=*/0, sourceExprs,
                                                rewriter.getContext()));
    reduceIndexingMaps.push_back(AffineMap::get(reduceExprCount,
                                                /*symbolCount=*/0, windowExprs,
                                                rewriter.getContext()));
    auto reduceOutMap =
        AffineMap::get(reduceExprCount,
                       /*symbolCount=*/0, outputExprs, rewriter.getContext());
    reduceIndexingMaps.push_back(reduceOutMap);
    reduceIndexingMaps.push_back(reduceOutMap);

    // Output sizes should match the dimensions of the `source` tensor, even if
    // dynamic.
    SmallVector<Value> reduceDynSizes;
    for (int i = 0, s = rank; i < s; ++i)
      if (sourceTy.isDynamicDim(i))
        reduceDynSizes.push_back(b.create<tensor::DimOp>(source, i));

    Value reduceValueEmpty =
        b.create<tensor::EmptyOp>(sourceTy.getShape(), destETy, reduceDynSizes);
    Value reduceIndexEmpty = b.create<tensor::EmptyOp>(
        sourceTy.getShape(), indexETy, reduceDynSizes);

    // We initialize indices to -1 which indicates no matching destination.
    Value negativeOne = b.create<arith::ConstantOp>(b.getI32IntegerAttr(-1));
    reduceIndexEmpty =
        b.create<linalg::FillOp>(negativeOne, reduceIndexEmpty).getResult(0);

    // We only care to match the reduction dimensions.
    Value windowEmpty = b.create<tensor::EmptyOp>(filteredWindows, srcETy);

    auto reduceGeneric = b.create<linalg::GenericOp>(
        /*resultTensors=*/ArrayRef<Type>{reduceValueEmpty.getType(),
                                         reduceIndexEmpty.getType()},
        /*inputs=*/ValueRange{operand, windowEmpty},
        /*outputs=*/ValueRange{reduceValueEmpty, reduceIndexEmpty},
        reduceIndexingMaps,
        getParallelAndReductionIterators(reduceExprCount, windowExprs.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // First we clone in the selection block.
    auto &reduceRegion = reduceGeneric.getRegion();
    rewriter.setInsertionPoint(reduceGeneric);
    rewriter.cloneRegionBefore(op.getSelect(), reduceRegion,
                               reduceRegion.end());

    // This includes convert `stablehlo` scalar-tensor regions to `linalg`
    // scalars.
    TypeConverter::SignatureConversion reduceSignConverter(4);
    reduceSignConverter.addInputs(0, srcETy);
    reduceSignConverter.addInputs(srcETy);
    reduceSignConverter.addInputs(1, destETy);
    reduceSignConverter.addInputs(indexETy);
    rewriter.applySignatureConversion(&reduceRegion, reduceSignConverter,
                                      getTypeConverter());

    // Grab the terminator and use the turned value to now select the
    // correct index and value.
    auto &reduceBlock = reduceRegion.front();
    auto *reduceTerminator = reduceBlock.getTerminator();
    Value selectPred = reduceTerminator->getOperand(0);
    Value selectInVal = reduceBlock.getArgument(0);
    Value selectOutVal = reduceBlock.getArgument(2);
    Value selectOutIdx = reduceBlock.getArgument(3);

    b.setInsertionPoint(reduceTerminator);

    // The predicate operates on scalar-tensors, so we need to extract the
    // value for `linalg` operations. Tensor-ops are cleaned up by other
    // rewriters.
    selectPred = b.create<tensor::ExtractOp>(rewriter.getI1Type(), selectPred,
                                             ValueRange{});

    // We select if either the selection function returns `true` or the
    // current reduction index is `-1`, e.g. no index has been selected yet.
    Value selectNegOne = b.create<arith::CmpIOp>(arith::CmpIPredicate::eq,
                                                 selectOutIdx, negativeOne);
    selectPred = b.create<arith::OrIOp>(selectPred, selectNegOne);

    // We compute a unique idx for each element in the window.
    Value computedIdx = b.create<linalg::IndexOp>(rank);
    for (int i = 1, s = filteredStrides.size(); i < s; ++i) {
      Value width = b.create<arith::ConstantIndexOp>(filteredStrides[i]);
      Value idx = b.create<linalg::IndexOp>(rank + i);
      computedIdx = b.create<arith::MulIOp>(width, computedIdx);
      computedIdx = b.create<arith::AddIOp>(computedIdx, idx);
    }
    computedIdx = b.create<arith::IndexCastOp>(indexETy, computedIdx);

    // Using the selection predicate track the value and selected
    // identifier for the future scattering.
    Value selectedIdx =
        b.create<arith::SelectOp>(selectPred, computedIdx, selectOutIdx);
    Value selectedValue =
        b.create<arith::SelectOp>(selectPred, selectInVal, selectOutVal);
    b.create<linalg::YieldOp>(ValueRange{selectedValue, selectedIdx});

    // Original terminator is an stablehlo.return we no longer need.
    rewriter.eraseOp(reduceTerminator);
    b.setInsertionPoint(op);

    Value reduceIndex = reduceGeneric.getResult(1);
    ShapedType reduceIndexTy = llvm::cast<ShapedType>(reduceIndex.getType());

    // For the second generic we restricted to only cases where there are
    // no window overlaps. This guarantees that each source value is scattered
    // within its own unique window. We can broadcast to this window size and
    // populate only the relative location.
    llvm::SmallVector<int64_t> broadcastShape;
    llvm::SmallVector<Value> broadcastDynDims;
    llvm::SmallVector<AffineExpr> broadcastExprs;
    for (int i = 0, s = reduceIndexTy.getRank(); i < s; ++i) {
      int64_t broadcast = strides[i];
      if (sourceTy.isDynamicDim(i))
        broadcastDynDims.push_back(b.create<tensor::DimOp>(source, i));

      broadcastExprs.push_back(b.getAffineDimExpr(broadcastShape.size()));
      broadcastShape.push_back(sourceTy.getDimSize(i));
      if (broadcast > 1) {
        broadcastShape.push_back(broadcast);
      }
    }

    // We broadcast the values of our input tensors across the stride-tiling
    // size.
    Value scatterEmpty = b.create<tensor::EmptyOp>(
        broadcastShape, resultTy.getElementType(), broadcastDynDims);
    Value initScalar = b.create<tensor::ExtractOp>(initTy.getElementType(),
                                                   init, ValueRange{});
    Value scatterFill =
        b.create<linalg::FillOp>(initScalar, scatterEmpty).getResult(0);

    // Both the indices and values are broadcasted using the same indexing map.
    // Output fully parallel.
    auto scatterInputMap =
        AffineMap::get(broadcastShape.size(), /*symbolCount=*/0, broadcastExprs,
                       b.getContext());
    SmallVector<AffineMap> scatterIndexingMaps;
    scatterIndexingMaps.push_back(scatterInputMap);
    scatterIndexingMaps.push_back(scatterInputMap);
    scatterIndexingMaps.push_back(
        b.getMultiDimIdentityMap(broadcastShape.size()));

    auto scatterGeneric = b.create<linalg::GenericOp>(
        /*resultTensors=*/ArrayRef<Type>{scatterFill.getType()},
        /*inputs=*/ValueRange{reduceIndex, source},
        /*outputs=*/ValueRange{scatterFill}, scatterIndexingMaps,
        getNParallelLoopsAttrs(broadcastShape.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Clone the scattering combination logic and perform the tensor-to-scalar
    // conversion.
    auto &scatterRegion = scatterGeneric.getRegion();
    b.setInsertionPoint(scatterGeneric);
    rewriter.cloneRegionBefore(op.getScatter(), scatterRegion,
                               scatterRegion.end());

    TypeConverter::SignatureConversion scatterSignConverter(4);
    scatterSignConverter.addInputs(indexETy);
    scatterSignConverter.addInputs(0, sourceTy.getElementType());
    scatterSignConverter.addInputs(1, sourceTy.getElementType());
    rewriter.applySignatureConversion(&scatterRegion, scatterSignConverter,
                                      getTypeConverter());

    auto &scatterBlock = scatterRegion.front();
    auto scatterTerminator = scatterBlock.getTerminator();
    b.setInsertionPoint(scatterTerminator);

    Value scatterInputIdx = scatterBlock.getArgument(0);
    Value scatterOutputVal = scatterBlock.getArgument(2);
    Value scatterUpdate = b.create<tensor::ExtractOp>(
        sourceTy.getElementType(), scatterTerminator->getOperand(0),
        ValueRange{});

    // Compute the index of the tiled region to determine if it was selected.
    Value id = b.create<arith::ConstantIndexOp>(0);
    int64_t dim = 0;
    for (int i = 0, s = strides.size(); i < s; ++i) {
      if (strides[i] > 1) {
        Value idx = b.create<linalg::IndexOp>(++dim);
        Value tileSz = b.create<arith::ConstantIndexOp>(strides[i]);
        id = b.create<arith::MulIOp>(id, tileSz);
        id = b.create<arith::AddIOp>(id, idx);
      }
      ++dim;
    }

    // Check whether the computed id matches the to-scatter id, then select and
    // yield.
    id = b.create<arith::IndexCastOp>(indexETy, id);
    auto scatterPred = b.create<arith::CmpIOp>(
        b.getI1Type(), arith::CmpIPredicate::eq, id, scatterInputIdx);
    scatterUpdate =
        b.create<arith::SelectOp>(scatterPred, scatterUpdate, scatterOutputVal);

    b.create<linalg::YieldOp>(scatterUpdate);
    rewriter.eraseOp(scatterTerminator);
    b.setInsertionPoint(op);

    // We now need to collapse the tiles back into their
    // source dimensions. We collapse any of the broadcast regions together.
    int64_t collapseDim = 0;
    SmallVector<ReassociationIndices> reassociationMap;
    for (int i = 0, s = window.size(); i < s; ++i) {
      SmallVector<int64_t, 2> dims = {collapseDim};
      if (strides[i] > 1)
        dims.push_back(collapseDim + 1);

      reassociationMap.push_back(ReassociationIndices(dims));
      collapseDim += dims.size();
    }

    Value collapse = b.create<tensor::CollapseShapeOp>(
        scatterGeneric.getResult(0), reassociationMap);
    auto collapseTy = llvm::cast<ShapedType>(collapse.getType());

    // After collapsing it it possible that the target may need to be padded.
    auto zero = b.createOrFold<arith::ConstantIndexOp>(0);
    SmallVector<int64_t> padShape;
    SmallVector<OpFoldResult> padLow, padHigh;
    padLow.resize(operandTy.getRank(), zero);

    for (int i = 0, s = rank; i < s; ++i) {
      int64_t size = std::max(resultTy.getDimSize(i), collapseTy.getDimSize(i));
      if (operandTy.isDynamicDim(i) || collapseTy.isDynamicDim(i))
        size = ShapedType::kDynamic;
      padShape.push_back(size);

      Value in = b.create<tensor::DimOp>(collapse, i);
      Value out = b.create<tensor::DimOp>(operand, i);
      Value diff = b.create<arith::SubIOp>(out, in);
      Value pad = b.createOrFold<arith::MaxSIOp>(diff, zero);
      padHigh.push_back(pad);
    }

    Value padded = b.create<tensor::PadOp>(collapseTy.clone(padShape), collapse,
                                           padLow, padHigh, initScalar);

    // The result may exceed the target size, slice if necessary.
    SmallVector<OpFoldResult> sliceSizes;
    SmallVector<OpFoldResult> sliceOffsets(operandTy.getRank(),
                                           b.getIndexAttr(0));
    SmallVector<OpFoldResult> sliceStrides(operandTy.getRank(),
                                           b.getIndexAttr(1));
    for (int i = 0, s = operandTy.getRank(); i < s; ++i) {
      OpFoldResult dim = b.getIndexAttr(operandTy.getDimSize(i));
      if (operandTy.isDynamicDim(i))
        dim = b.createOrFold<tensor::DimOp>(operand, i);
      sliceSizes.push_back(dim);
    }

    rewriter.setInsertionPoint(op);
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, padded, sliceOffsets, sliceSizes, sliceStrides);

    return success();
  }
};

// Decomposes a pad with negative edge padding into a pad without negative edge
// padding and a tensor.extract_slice.
struct PadOpNegativePaddingConversion final
    : OpConversionPattern<mlir::stablehlo::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<int64_t> padLow;
    SmallVector<int64_t> padHigh;
    SmallVector<OpFoldResult> sliceStarts;

    bool hasNegativePadding = false;
    for (int64_t low : op.getEdgePaddingLow()) {
      if (low >= 0) {
        padLow.push_back(low);
        sliceStarts.push_back(rewriter.getIndexAttr(0));
      } else {
        padLow.push_back(0);
        sliceStarts.push_back(rewriter.getIndexAttr(-low));
        hasNegativePadding = true;
      }
    }

    for (int64_t high : op.getEdgePaddingHigh()) {
      if (high >= 0) {
        padHigh.push_back(high);
      } else {
        padHigh.push_back(-high);
        hasNegativePadding = true;
      }
    }

    // If there's no negative edge padding we're done.
    if (!hasNegativePadding)
      return failure();

    // Create a new pad op with the positive values.
    Value pad = rewriter.create<mlir::stablehlo::PadOp>(
        op.getLoc(), adaptor.getOperand(), adaptor.getPaddingValue(),
        rewriter.getDenseI64ArrayAttr(padLow),
        rewriter.getDenseI64ArrayAttr(padHigh), op.getInteriorPadding());

    // Then slice according to the negative edge padding. Static shapes only for
    // now.
    if (!op.getType().hasStaticShape())
      return failure();
    SmallVector<OpFoldResult> sizes(
        llvm::map_range(op.getType().getShape(), [&](int64_t dim) {
          return rewriter.getIndexAttr(dim);
        }));
    SmallVector<OpFoldResult> strides(sliceStarts.size(),
                                      rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, pad, sliceStarts,
                                                        sizes, strides);
    return success();
  }
};

/// Converts stablehlo.pad operation to tensor.pad or tensor.insert_slice.
struct PadOpConversion final : OpConversionPattern<mlir::stablehlo::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType =
        getTypeConverter()->convertType<ShapedType>(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    // Negative edge padding is decomposed separately.
    auto isNegative = [](int64_t intVal) { return intVal < 0; };
    if (llvm::any_of(op.getEdgePaddingLow(), isNegative) ||
        llvm::any_of(op.getEdgePaddingHigh(), isNegative))
      return failure();

    Value paddingVal = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    auto i64ToFoldResult = [&](const int64_t &i) -> OpFoldResult {
      return rewriter.getIntegerAttr(rewriter.getI64Type(), i);
    };

    // If there is no interior padding lower to tensor.pad directly.
    if (llvm::all_of(op.getInteriorPadding(),
                     [](const int64_t &i) { return i == 0; })) {
      auto padTensorOp = rewriter.create<tensor::PadOp>(
          loc, resultType, adaptor.getOperand(),
          llvm::map_to_vector(op.getEdgePaddingLow(), i64ToFoldResult),
          llvm::map_to_vector(op.getEdgePaddingHigh(), i64ToFoldResult),
          paddingVal);
      rewriter.replaceOp(op, padTensorOp.getResult());
      return success();
    }

    // We have interior padding, which can be lowered to tensor.insert_slice.
    // Start by filling a result-sized tensor with the pad value.
    auto emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultType, op, adaptor.getOperands());
    auto fill =
        rewriter.create<linalg::FillOp>(loc, paddingVal, emptyTensor).result();

    // Get sizes of the original operand.
    auto operandType = llvm::cast<ShapedType>(adaptor.getOperand().getType());
    auto sizes = llvm::map_to_vector(
        llvm::seq<int64_t>(0, operandType.getRank()),
        [&](int64_t dim) -> OpFoldResult {
          if (!operandType.isDynamicDim(dim))
            return rewriter.getIndexAttr(operandType.getDimSize(dim));
          return rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), dim)
              .getResult();
        });
    // Map interior padding to strides.
    auto strides = llvm::map_to_vector(
        op.getInteriorPadding(), [&](const int64_t &stride) -> OpFoldResult {
          return rewriter.getIntegerAttr(rewriter.getI64Type(), stride + 1);
        });

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getOperand(), fill,
        llvm::map_to_vector(op.getEdgePaddingLow(), i64ToFoldResult), sizes,
        strides);
    return success();
  }
};

/// Converts xla-hlo.torch_index_select op to a linalg.generic op.
struct TorchIndexSelectOpConversion final
    : OpConversionPattern<mlir::stablehlo::TorchIndexSelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::TorchIndexSelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int axis = static_cast<int>(op.getDim());
    int batch = static_cast<int>(op.getBatchDims());
    auto indexShapedType = llvm::cast<ShapedType>(adaptor.getIndex().getType());
    int numIndices = static_cast<int>(indexShapedType.getRank());
    auto operandShapedType =
        llvm::cast<ShapedType>(adaptor.getOperand().getType());
    if (axis < 0)
      axis += static_cast<int>(operandShapedType.getRank());
    if (batch < 0)
      batch += numIndices;

    Location loc = op.getLoc();
    auto resultType = getTypeConverter()->convertType<ShapedType>(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "type conversion failed");

    int rank = static_cast<int>(resultType.getRank());

    // The output shape is
    //   `params[:axis] + indices[batch_dims:] + params[axis + 1:]`
    SmallVector<Value> dynSizes;
    for (int i = 0; i < rank; ++i) {
      if (!resultType.isDynamicDim(i))
        continue;
      if (i < axis) {
        dynSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), i));
      } else if (i < (axis + numIndices - batch)) {
        int idx = i - axis + batch;
        dynSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.getIndex(), idx));
      } else {
        int idx = i - (axis + numIndices - batch) + axis + 1;
        dynSizes.push_back(
            rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), idx));
      }
    }

    // Generate dummy tensor to preserve slice shape information.
    SmallVector<int64_t> sliceShape;
    SmallVector<Value> dynSliceSizes;
    SmallVector<AffineExpr> sliceExprs;
    ArrayRef<int64_t> resultShape = resultType.getShape();
    for (int i = 0; i < axis; ++i) {
      sliceExprs.push_back(rewriter.getAffineDimExpr(i));
      sliceShape.push_back(resultShape[i]);
      if (!resultType.isDynamicDim(i))
        continue;
      dynSliceSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), i));
    }
    for (int i = axis + numIndices - batch; i < rank; ++i) {
      sliceExprs.push_back(rewriter.getAffineDimExpr(i));
      sliceShape.push_back(resultShape[i]);
      if (!resultType.isDynamicDim(i))
        continue;
      int idx = i - (axis + numIndices - batch) + axis + 1;
      dynSliceSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), idx));
    }

    // Setup AffineMap for operand tensor.
    SmallVector<AffineExpr> exprs;
    for (int i = 0; i < batch; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    for (int i = 0, e = numIndices - batch; i < e; ++i) {
      exprs.push_back(rewriter.getAffineDimExpr(axis + i));
    }

    SmallVector<AffineMap, 2> indexingMaps;
    indexingMaps.emplace_back(
        AffineMap::get(rank, /*symbolCount=*/0, exprs, rewriter.getContext()));
    indexingMaps.emplace_back(AffineMap::get(
        rank, /*symbolCount=*/0, sliceExprs, rewriter.getContext()));
    indexingMaps.emplace_back(rewriter.getMultiDimIdentityMap(rank));

    Value sliceOp = rewriter.create<tensor::EmptyOp>(
        loc, sliceShape, resultType.getElementType(), dynSliceSizes);

    Value emptyOp = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType(), dynSizes);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/ArrayRef<Type>{resultType},
        /*inputs=*/ValueRange{adaptor.getIndex(), sliceOp},
        /*outputs=*/emptyOp, indexingMaps, getNParallelLoopsAttrs(rank),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    SmallVector<Type> bodyArgTypes;
    SmallVector<Value, 2> linalgOpArgs = {adaptor.getIndex(), sliceOp};
    // Add a block to the region.
    auto *region = &linalgOp.getRegion();
    auto *block = rewriter.createBlock(region, region->end());
    for (auto blockArgs : linalgOpArgs) {
      bodyArgTypes.push_back(
          llvm::cast<ShapedType>(blockArgs.getType()).getElementType());
    }
    block->addArguments(bodyArgTypes,
                        SmallVector<Location>(bodyArgTypes.size(), loc));
    block->addArguments(resultType.getElementType(), loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    Value castedValue = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), block->getArgument(0));

    SmallVector<Value> indices;
    for (int i = 0; i < axis; ++i) {
      indices.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    }
    indices.push_back(castedValue);
    for (int i = axis + numIndices - batch; i < rank; ++i) {
      indices.push_back(rewriter.create<linalg::IndexOp>(loc, i));
    }
    Value res =
        rewriter.create<tensor::ExtractOp>(loc, adaptor.getOperand(), indices);
    rewriter.create<linalg::YieldOp>(loc, res);

    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct SetDimensionSizeConverter final
    : OpConversionPattern<mlir::stablehlo::SetDimensionSizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::stablehlo::SetDimensionSizeOp setDimensionSizeOp,
                  OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // We can lower SetDimensionSize to tensor extract. This turns into a
    // regular dynamic shape. Note that the bounds annotation is still around
    // but may be no longer valid depending on choices made by bufferization.
    Location loc = setDimensionSizeOp.getLoc();
    auto resultType = dyn_cast<RankedTensorType>(setDimensionSizeOp.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(setDimensionSizeOp,
                                         "expected a ranked tensor");

    SmallVector<OpFoldResult> offsets(resultType.getRank(),
                                      rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(resultType.getRank(),
                                      rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes(llvm::map_range(
        resultType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));
    Value dimensionSize =
        rewriter.create<tensor::ExtractOp>(loc, setDimensionSizeOp.getSize());
    sizes[setDimensionSizeOp.getDimension()] =
        rewriter
            .create<arith::IndexCastOp>(loc, rewriter.getIndexType(),
                                        dimensionSize)
            .getResult();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        setDimensionSizeOp, resultType, adaptor.getOperand(), offsets, sizes,
        strides);
    return success();
  }
};

struct ConvertStableHloToLinalg final
    : impl::ConvertStableHloToLinalgBase<ConvertStableHloToLinalg> {
  using ConvertStableHloToLinalgBase::ConvertStableHloToLinalgBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    scf::SCFDialect, complex::ComplexDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    MLIRContext &ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<
        bufferization::BufferizationDialect, arith::ArithDialect,
        complex::ComplexDialect, linalg::LinalgDialect, math::MathDialect,
        tensor::TensorDialect, sparse_tensor::SparseTensorDialect,
        scf::SCFDialect, shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = createStableHloToLinalgTypeConverter();
    ModuleOp module = getOperation();

    populateStableHloToLinalgConversionPatterns(&ctx, *typeConverter, &patterns,
                                                this->enablePrimitiveOps);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

void populateStableHloToLinalgConversionPatterns(MLIRContext *context,
                                                 TypeConverter &typeConverter,
                                                 RewritePatternSet *patterns,
                                                 bool enablePrimitiveOps) {
  // clang-format off
  patterns->add<
      BitcastConvertConverter,
      ConcatenateConverter,
      ConstConverterTensor,
      EinsumToLinalgConverter,
      GatherConversion,
      RealDynamicSliceConverter,
      ReshapeOpConverter,
      ReverseConverter,
      SetDimensionSizeConverter,
      SliceConverter,
      DynamicSliceConverter,
      DynamicUpdateSliceConverter,
      PadOpConversion,
      PadOpNegativePaddingConversion,
      TorchIndexSelectOpConversion,
      SelectAndScatterNoOverlapConverter
      >(typeConverter, context);

  detail::populatePointwiseStableHloToLinalgConversionPatterns(
      context, typeConverter, patterns, enablePrimitiveOps);

  if (enablePrimitiveOps) {
    patterns->add<
      BroadcastInDimOpToBroadcastConverter,
      BroadcastOpToBroadcastConverter,
      DynamicBroadcastInDimOpToBroadcastConverter,
      IotaToMapConverter<mlir::stablehlo::IotaOp>,
      IotaToMapConverter<mlir::stablehlo::DynamicIotaOp>,
      MapOpToMapConverter,
      TransposeOpToTransposeConverter
    >(typeConverter, context);
  } else {
    patterns->add<
      BroadcastConverter<mlir::stablehlo::BroadcastOp>,
      IotaConverter<mlir::stablehlo::IotaOp>,
      IotaConverter<mlir::stablehlo::DynamicIotaOp>,
      HloBroadcastInDimConverter,
      HloDynamicBroadcastInDimConverter,
      MapOpToGenericConverter,
      TransposeConverter<mlir::stablehlo::TransposeOp>
    >(typeConverter, context);
  }

  // clang-format on

  detail::populateStableHloConvolutionToLinalgConversionPatterns(
      context, typeConverter, patterns);
  detail::populateStableHloDotProdToLinalgConversionPatterns(
      context, typeConverter, patterns);
  detail::populateStableHloRandomToLinalgConversionPatterns(
      context, typeConverter, patterns);
  detail::populateStableHloReductionToLinalgConversionPatterns(
      context, typeConverter, patterns, enablePrimitiveOps);
  detail::populateScalarHloToArithConversionPatterns(
      context, typeConverter, patterns, isInBodyOfLinalgOps);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(*patterns);
}

std::unique_ptr<TypeConverter> createStableHloToLinalgTypeConverter() {
  return std::make_unique<LinalgTypeConverter>();
}

} // namespace mlir::iree_compiler::stablehlo
