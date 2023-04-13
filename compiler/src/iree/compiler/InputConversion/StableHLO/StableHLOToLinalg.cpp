// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Implements logic for lowering StableHLO/CHLO dialects to Linalg dialect.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>

#include "iree/compiler/InputConversion/StableHLO/LegalizeToLinalgUtils.h"
#include "iree/compiler/InputConversion/StableHLO/Passes.h"
#include "iree/compiler/InputConversion/StableHLO/Rewriters.h"
#include "iree/compiler/InputConversion/StableHLO/TypeConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace mlir::iree_compiler::stablehlo {

#define GEN_PASS_DEF_CONVERTSTABLEHLOTOLINALG
#include "iree/compiler/InputConversion/StableHLO/Passes.h.inc"

namespace {

namespace stablehlo = mlir::stablehlo;

Value getResultValue(Operation* op) { return op->getResult(0); }

ShapedType getHloOpResultType(Operation* op) {
  return getResultValue(op).getType().cast<ShapedType>();
}

SmallVector<int64_t, 4> extract1DVector(DenseIntElementsAttr elements) {
  SmallVector<int64_t, 4> ret;
  for (const APInt& element : elements) {
    ret.push_back(element.getLimitedValue());
  }
  return ret;
}

/// Returns a permutation AffineMap that puts all reduction dimensions to the
/// last. The order of parallel loops and reduction loops are all sorted. E.g.,
/// if `rank` is 4 and `reductionDims` is {1, 3}, then
/// "(d0, d1, d2, d3) -> (d0, d2, d1, d3)" is used. The inverse permutation of
/// the AffineMap is returned.
AffineMap getTransposeMapForReduction(MLIRContext* context, int rank,
                                      ArrayRef<int64_t> reductionDims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reductionDims) s.insert(dim);

  SmallVector<unsigned, 4> permutation;
  for (int i = 0; i < rank; ++i)
    if (!s.count(i)) permutation.push_back(i);
  for (auto dim : reductionDims) permutation.push_back(dim);

  auto map = AffineMap::getPermutationMap(permutation, context);
  return inversePermutation(map);
}

/// Returns true if the given `attr` is a splat of the given `value`.
bool isSplatValue(DenseIntElementsAttr attr, uint64_t value) {
  return attr.isSplat() && attr.getSplatValue<uint64_t>() == value;
}

/// Extracts an element from a tensor and optionally converts it to an index
/// type, based on the tensor's pre-type conversion type.
Value extractIndexFromTensor(OpBuilder& builder, Location loc, Value tensor,
                             ShapedType originalType,
                             ArrayRef<Value> tensorIndex = {}) {
  Value extracted = builder.create<tensor::ExtractOp>(loc, tensor, tensorIndex);
  if (extracted.getType().isIndex()) return extracted;
  return originalType.getElementType().isUnsignedInteger()
             ? builder.createOrFold<arith::IndexCastUIOp>(
                   loc, builder.getIndexType(), extracted)
             : builder.createOrFold<arith::IndexCastOp>(
                   loc, builder.getIndexType(), extracted);
}

//===----------------------------------------------------------------------===//
// stablehlo.RngOp conversion patterns.
//===----------------------------------------------------------------------===//

// Pass to lower from rng to stateless pseudo RNG with LCG
// algorithm
struct RngUniformConversion : public OpConversionPattern<stablehlo::RngOp> {
  using OpConversionPattern<stablehlo::RngOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::RngOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // We only handle uniform distributions
    if (op.getRngDistribution() !=
        ::mlir::stablehlo::RngDistribution::UNIFORM) {
      return failure();
    }
    // TODO(raikonenfnu): Handle other element types as well.
    auto minTy = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    auto maxTy = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    if (!minTy.getElementType().dyn_cast<FloatType>() ||
        !maxTy.getElementType().dyn_cast<FloatType>()) {
      return rewriter.notifyMatchFailure(
          op, "expected min/max for rng op to be FloatType");
    }
    auto targetTy = this->typeConverter->convertType(op.getResult().getType())
                        .cast<ShapedType>();
    if (!targetTy) {
      return rewriter.notifyMatchFailure(
          op, "expected target shape of rng op to be ShapedType");
    }
    auto loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, targetTy, op, adaptor.getOperands());
    // Creates index map using target matrix's rank.
    auto targetRank = targetTy.getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        2, AffineMap::get(targetRank, /*symbolCount=*/0,
                          SmallVector<AffineExpr>({}), rewriter.getContext()));
    indexingMaps.push_back(rewriter.getMultiDimIdentityMap(targetRank));
    const int kInitialSeed = 0;
    // Generic region with LCG Algorithm that make use of element index from:
    // https://reviews.llvm.org/D101364
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensors=*/targetTy,
        /*inputs=*/
        ValueRange{adaptor.getOperands()[0], adaptor.getOperands()[1]},
        /*outputs=*/emptyTensor, indexingMaps,
        getParallelAndReductionIterators(/*nLoops=*/targetRank,
                                         /*nReduction=*/0),
        [&](OpBuilder& b, Location loc, ValueRange args) {
          llvm::SmallVector<Value> updateVec = {b.create<arith::ConstantOp>(
              loc, b.getI32IntegerAttr(kInitialSeed))};
          Value multiplier =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
          Value incrementStep =
              b.create<arith::ConstantOp>(loc, b.getI32IntegerAttr(12345));
          // For output matrix with rank N:
          // temp1 = (cast(I32, index(D.0)) + seed) * mult + incr
          // ...
          // tempN = (cast(I32, index(D.(N))) + tempN_1) * mult + incr
          for (int i = 0; i < targetRank; i++) {
            Value update = updateVec.back();
            Value ind = b.create<linalg::IndexOp>(loc, i);
            Value castInd =
                b.create<arith::IndexCastOp>(loc, b.getI32Type(), ind);
            Value addRes = b.create<arith::AddIOp>(loc, castInd, update);
            Value multRes = b.create<arith::MulIOp>(loc, addRes, multiplier);
            Value incRes = b.create<arith::AddIOp>(loc, multRes, incrementStep);
            updateVec.push_back(incRes);
          }
          // Scaling = (max - min) * const(F64, 2.3283064E-10)
          // which is derived from rand(min,max) = rand()/(RAND_MAX/(max-min)).
          Value epsilon = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(args[0].getType(), 2.3283064E-10));
          Value range = b.create<arith::SubFOp>(loc, args[1], args[0]);
          Value scale = b.create<arith::MulFOp>(loc, range, epsilon);
          // Res = cast(T, cast(F64, tempN) * scaling + min)
          Value updateCast = b.create<arith::UIToFPOp>(
              loc, targetTy.getElementType(), updateVec.back());
          Value scaleUpdate = b.create<arith::MulFOp>(loc, updateCast, scale);
          Value res = b.create<arith::AddFOp>(loc, scaleUpdate, args[0]);
          b.create<linalg::YieldOp>(loc, res);
        },
        linalg::getPrunedAttributeList(op));
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// stablehlo.Einsum conversion patterns.
//===----------------------------------------------------------------------===//

// Looks through a set of dimension that has been marked as reduction axes,
// if it is found within the set, then we set it as "reduction", otherwise
// we can label it as "parallel".
SmallVector<utils::IteratorType, 3> getEinsumLoopsAttrs(
    const llvm::SmallSetVector<StringRef, 4>& inputInd,
    const llvm::SmallSetVector<StringRef, 4>& reductionDims) {
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

SmallVector<Value, 2> extractDynamicEinsumSizes(
    OpBuilder& b, Location loc, Value lhs, Value rhs,
    const SmallVector<std::string>& lhsLoopVec,
    const SmallVector<std::string>& rhsLoopVec,
    const SmallVector<std::string>& outputLoopVec) {
  SmallVector<Value, 2> dynSizes;
  for (const std::string& dimInd : outputLoopVec) {
    Value dimSize;
    const auto* dimIndIt =
        std::find(lhsLoopVec.begin(), lhsLoopVec.end(), dimInd);
    if (dimIndIt != lhsLoopVec.end()) {
      // Query from lhs vars.
      auto dimIndPos = dimIndIt - lhsLoopVec.begin();
      auto lhsShape = lhs.getType().dyn_cast<RankedTensorType>().getShape();
      if (lhsShape[dimIndPos] != ShapedType::kDynamic) continue;
      dimSize = b.create<tensor::DimOp>(loc, lhs, dimIndPos);
    } else {
      // query from rhs vars.
      dimIndIt = std::find(rhsLoopVec.begin(), rhsLoopVec.end(), dimInd);
      auto dimIndPos = dimIndIt - rhsLoopVec.begin();
      auto rhsShape = rhs.getType().dyn_cast<RankedTensorType>().getShape();
      if (rhsShape[dimIndPos] != ShapedType::kDynamic) continue;
      dimSize = b.create<tensor::DimOp>(loc, rhs, dimIndPos);
    }
    dynSizes.push_back(dimSize);
  }
  return dynSizes;
}

// Adds indices/axes that are missing from output set.
llvm::SmallSetVector<StringRef, 4> findSummationAxes(
    const llvm::SmallSetVector<StringRef, 4>& inputSet,
    const llvm::SmallSetVector<StringRef, 4>& outputSet) {
  llvm::SmallSetVector<StringRef, 4> summationAxes;
  for (StringRef ind : inputSet) {
    if (!outputSet.contains(ind)) summationAxes.insert(ind);
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
SmallVector<AffineExpr> getExprFromConfig(
    const SmallVector<std::string>& loopDims,
    const DenseMap<StringRef, AffineExpr>& strAffineDimUmap) {
  SmallVector<AffineExpr> exprs;
  for (const auto& dim : loopDims) {
    exprs.push_back(strAffineDimUmap.lookup(dim));
  }
  return exprs;
}

// Convert mhlo.einsum op into linalg.generic.
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
class EinsumToLinalgConverter
    : public OpConversionPattern<stablehlo::EinsumOp> {
 public:
  using OpConversionPattern<stablehlo::EinsumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::EinsumOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto getRank = [](Value v) {
      return v.getType().cast<ShapedType>().getRank();
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
    if (rhsLoop.find(kComma) != std::string::npos ||
        outLoop.find(kComma) != std::string::npos ||
        outLoop.find(kArrow) != std::string::npos) {
      return rewriter.notifyMatchFailure(op, "Invalid einsum config!");
    }

    // Find result type, if on tensors.
    auto resultTy = this->typeConverter->convertType(getHloOpResultType(op))
                        .dyn_cast<RankedTensorType>();

    // Check result type compatibility.
    if (!resultTy || !(resultTy.getElementType().isSignlessIntOrFloat())) {
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
    auto loc = op.getLoc();

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
    for (const auto& it : llvm::enumerate(inputInd)) {
      strAffineDimUmap[it.value()] = rewriter.getAffineDimExpr(it.index());
    }

    // From einsum_config of each operand in vector<string>, generate
    // the equivalent vector<AffineExpr>.
    SmallVector<AffineMap, 4> maps;
    for (const SmallVector<std::string>& loopOperand :
         {lhsEin, rhsEin, outEin}) {
      auto exprs = getExprFromConfig(loopOperand, strAffineDimUmap);
      maps.push_back(AffineMap::get(nloops, 0, exprs, rewriter.getContext()));
    }

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, resultTy ? resultTy : TypeRange{}, adaptor.getOperands(), output,
        maps, getEinsumLoopsAttrs(inputInd, reductionAxe),
        [reductionAxe](OpBuilder& b, Location nestedLoc, ValueRange args) {
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
  static constexpr StringRef kArrow = "->";
  static constexpr StringRef kComma = ",";
  static constexpr StringRef kEllipsis = "...";

  static bool checkBatchHasEqualRank(size_t lhsRank, StringRef lhsLoop,
                                     size_t rhsRank, StringRef rhsLoop,
                                     size_t outRank, StringRef outLoop);
  static SmallVector<std::string> getEinsumConfigAsVector(StringRef loop,
                                                          size_t operandRank);
};

// Definition of util const member variables.
constexpr StringRef EinsumToLinalgConverter::kArrow;
constexpr StringRef EinsumToLinalgConverter::kComma;
constexpr StringRef EinsumToLinalgConverter::kEllipsis;

// Convert the representation from string/vector<char> to vector<string>.
// i.e ("abc") -> {"a", "b", "c"}. For cases with ellipsis with batch rank 3:
// get loop_dim = f("ab...cde") = {"a","b","0","1","2","c","d","e"}
SmallVector<std::string> EinsumToLinalgConverter::getEinsumConfigAsVector(
    StringRef loop, size_t operandRank) {
  SmallVector<std::string> loopDim;
  size_t preElip = loop.find(kEllipsis);
  bool hasElip = preElip != std::string::npos;
  if (!hasElip) preElip = loop.size();
  // Add the dimension until the end or up to ellipsis if it exist.
  for (int64_t preElipInd = 0; preElipInd < static_cast<int64_t>(preElip);
       preElipInd++) {
    loopDim.push_back(loop.substr(preElipInd, 1).str());
  }
  if (!hasElip) return loopDim;
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
       postElipInd < static_cast<int64_t>(loop.size()); postElipInd++) {
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
  if (batchRankVec.size() < 2) return batchHasEqualRank;
  if (!std::equal(batchRankVec.begin() + 1, batchRankVec.end(),
                  batchRankVec.begin()) &&
      batchRankVec.size() > 1)
    batchHasEqualRank = false;
  return batchHasEqualRank;
}

/// Base class for lowering HLO operations that have one operand and one result,
/// and are semantically equivalent to a copy of the input to the output (like
/// transpose, some reshape, etc.). The derived classes need to provide a method
/// `getIndexingMaps` that returns AffineMaps for the index maps of the input
/// and the output.
template <typename Derived, typename OpTy>
class DataMovementOpConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) return failure();
    auto resultType = getHloOpResultType(op);
    resultType = this->typeConverter->convertType(resultType)
                     .template cast<ShapedType>();

    SmallVector<AffineMap, 2> indexingMaps =
        Derived::getIndexingMaps(op, &rewriter);
    if (indexingMaps.empty()) return failure();

    auto nloops = resultType.getRank();
    auto loc = op.getLoc();
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTensorTypes=*/resultType,
        /*inputs=*/adaptor.getOperands().front(),
        /*outputBuffers=*/

        ValueRange{getEmptyTensorFor(rewriter, loc, resultType, op,
                                     adaptor.getOperands())},
        indexingMaps, getNParallelLoopsAttrs(nloops),
        [&](OpBuilder& nestedBuilder, Location /*nested_loc*/,
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
class BroadcastConverter
    : public DataMovementOpConverter<BroadcastConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<BroadcastConverter,
                                OpTy>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy broadcastOp,
                                                   Builder* b) {
    ShapedType inputType =
        broadcastOp.getOperand().getType().template cast<ShapedType>();
    unsigned inputRank = inputType.getRank();
    unsigned nloops = getHloOpResultType(broadcastOp).getRank();

    // BroadcastOp prepends the dimensions in the `broadcast_sizes` attribute to
    // the input's dimensions.
    unsigned numPrependedDims = llvm::size(broadcastOp.getBroadcastSizes());
    SmallVector<AffineExpr, 4> inputDimExprs;
    inputDimExprs.reserve(inputRank);
    for (unsigned i = 0; i < inputRank; ++i) {
      inputDimExprs.push_back(b->getAffineDimExpr(numPrependedDims + i));
    }

    AffineMap inputMap;
    MLIRContext* context = b->getContext();
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

class BroadcastOpToBroadcastConverter
    : public OpConversionPattern<stablehlo::BroadcastOp> {
  using OpConversionPattern<stablehlo::BroadcastOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::BroadcastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultTy = typeConverter->convertType(op.getType()).cast<ShapedType>();

    int64_t numPrependedDims = op.getBroadcastSizes().size();
    SmallVector<int64_t> dimensions =
        llvm::to_vector(llvm::seq<int64_t>(0, numPrependedDims));

    auto loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(
        op, op.getOperand(), emptyTensor, dimensions,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

class HloBroadcastInDimConverter
    : public DataMovementOpConverter<HloBroadcastInDimConverter,
                                     stablehlo::BroadcastInDimOp> {
 public:
  using DataMovementOpConverter<
      HloBroadcastInDimConverter,
      stablehlo::BroadcastInDimOp>::DataMovementOpConverter;

  static SmallVector<AffineMap, 2> getIndexingMaps(
      stablehlo::BroadcastInDimOp broadcastOp, Builder* b) {
    auto resultType = getHloOpResultType(broadcastOp);
    auto operandType =
        broadcastOp.getOperand().getType().template cast<ShapedType>();
    unsigned nloops = resultType.getRank();

    // The input is a scalar, i.e. this is a scalar broadcast op.
    if (operandType.getRank() == 0) {
      return {AffineMap::get(nloops, /*symbolCount=*/0, b->getContext()),
              b->getMultiDimIdentityMap(nloops)};
    }

    auto operandShape = operandType.getShape();
    SmallVector<AffineExpr, 4> dimExprs;
    dimExprs.reserve(nloops);

    if (broadcastOp.getBroadcastDimensions()) {
      for (const auto& broadcastDim :
           enumerate(broadcastOp.getBroadcastDimensions().getValues<APInt>())) {
        int size = broadcastDim.value().getSExtValue();
        bool expansionNeeded = operandShape[broadcastDim.index()] == 1 &&
                               resultType.getShape()[size] != 1;
        dimExprs.push_back(expansionNeeded ? b->getAffineConstantExpr(0)
                                           : b->getAffineDimExpr(size));
      }
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, dimExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

Value collapseExpandingDims(PatternRewriter& rewriter, Location loc,
                            Value operand, SmallVector<int64_t>& dimensions,
                            llvm::function_ref<bool(int64_t)> isExpandingDim) {
  auto operandTy = operand.getType().cast<RankedTensorType>();

  SmallVector<ReassociationIndices> reassociationMap;
  ReassociationIndices currentIndices;

  ArrayRef<int64_t> operandShape = operandTy.getShape();
  SmallVector<int64_t> newOperandShape;
  SmallVector<int64_t> newDimensions;

  for (const auto& [idx, dim] : llvm::enumerate(dimensions)) {
    currentIndices.push_back(idx);

    if (!isExpandingDim(idx)) {
      reassociationMap.push_back(currentIndices);
      currentIndices.clear();
      newOperandShape.push_back(operandShape[idx]);
      newDimensions.push_back(dim);
    }
  }

  if (!reassociationMap.empty())
    reassociationMap.back().insert(reassociationMap.back().end(),
                                   currentIndices.begin(),
                                   currentIndices.end());

  if (dimensions.size() != newDimensions.size()) {
    dimensions = newDimensions;

    auto newOperandType =
        RankedTensorType::get(newOperandShape, operandTy.getElementType());
    operand = rewriter.create<tensor::CollapseShapeOp>(
        loc, newOperandType, operand, reassociationMap);
  }
  return operand;
}

// Insert linalg.transpose if broadcasted dimensions are not in sorded order.
// linalg.broadcast does not support implicit transpose, so the input needs to
// be explicitely transposed.
Value transposeBroadcastOperand(PatternRewriter& rewriter, Location loc,
                                Value operand,
                                SmallVector<int64_t>& dimensions) {
  // Do not insert `transpose` is dimensions are already sorted.
  if (llvm::is_sorted(dimensions)) return operand;

  SmallVector<int64_t> permutation =
      llvm::to_vector(llvm::seq<int64_t>(0, dimensions.size()));
  llvm::sort(permutation, [&](int64_t lhs, int64_t rhs) {
    return dimensions[lhs] < dimensions[rhs];
  });

  auto operandTy = operand.getType().cast<ShapedType>();
  ArrayRef<int64_t> operandShape = operandTy.getShape();
  SmallVector<int64_t> transposedOperandShape, transposedDimensions;

  for (int64_t index : permutation) {
    transposedOperandShape.push_back(operandShape[index]);
    transposedDimensions.push_back(dimensions[index]);
  }
  dimensions = transposedDimensions;

  return rewriter.create<stablehlo::TransposeOp>(
      loc,
      RankedTensorType::get(transposedOperandShape, operandTy.getElementType()),
      operand, rewriter.getI64VectorAttr(permutation));
}

class BroadcastInDimOpToBroadcastConverter
    : public OpConversionPattern<stablehlo::BroadcastInDimOp> {
 public:
  using OpConversionPattern<stablehlo::BroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::BroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();

    SmallVector<int64_t> broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>());

    Value operand = adaptor.getOperand();
    auto operandTy = operand.getType().cast<ShapedType>();
    auto resultTy = typeConverter->convertType(op.getType()).cast<ShapedType>();

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
// optionally provided attrbibutes on statically known broadcasting behavior.
// This means we can lower the broadcast just as we would lower a fully static
// broadcast and go directly to `linalg.generic`.

// This also covers the important case of broadcasting a scalar. Ideally the
// pattern (`mhlo.constant` -> `mhlo.dynamic_broadcast_in_dim`) should be
// converted to a tensor dialect op similar to TF's `ConstantLikeOp`.
class HloDynamicBroadcastInDimConverter
    : public OpConversionPattern<stablehlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern<
      stablehlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Value operand = adaptor.getOperand();
    auto operandType = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandType) return failure();
    auto resultType =
        typeConverter->convertType(op.getType()).dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

    // Determine dimension expressions based on whether the dimension is
    // expanding (0) or non-expanding (identity), and fail if we cannot decide
    // this.
    SmallVector<AffineExpr> dimExprs(operandType.getRank(), nullptr);

    // Use static type info.
    auto bcastDims = llvm::to_vector(
        llvm::map_range(op.getBroadcastDimensions(), [](const APInt& d) {
          return static_cast<int64_t>(d.getLimitedValue());
        }));
    for (const auto& it : llvm::enumerate(operandType.getShape())) {
      if (ShapedType::isDynamic(it.value())) continue;
      bool isExpanding = it.value() == 1;
      dimExprs[it.index()] =
          isExpanding ? rewriter.getAffineConstantExpr(0)
                      : rewriter.getAffineDimExpr(bcastDims[it.index()]);
    }

    // Use annotated expansion behavior, if available.
    if (op.getKnownExpandingDimensions()) {
      for (const auto& it :
           op.getKnownExpandingDimensions()->getValues<APInt>()) {
        auto i = it.getLimitedValue();
        dimExprs[i] = rewriter.getAffineConstantExpr(0);
      }
    }
    if (op.getKnownNonexpandingDimensions()) {
      for (const auto& it :
           op.getKnownNonexpandingDimensions()->getValues<APInt>()) {
        auto i = it.getLimitedValue();
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
        [&](OpBuilder& nestedBuilder, Location /*nested_loc*/,
            ValueRange args) {
          nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
        },
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

class DynamicBroadcastInDimOpToBroadcastConverter
    : public OpConversionPattern<stablehlo::DynamicBroadcastInDimOp> {
 public:
  using OpConversionPattern<
      stablehlo::DynamicBroadcastInDimOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DynamicBroadcastInDimOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();

    Value operand = adaptor.getOperand();
    auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
    if (!operandTy) return failure();
    auto resultTy =
        typeConverter->convertType(op.getType()).dyn_cast<RankedTensorType>();
    if (!resultTy) return failure();

    SmallVector<int64_t> broadcastDimensions =
        llvm::to_vector(op.getBroadcastDimensions().getValues<int64_t>());

    SmallVector<std::optional<bool>> expansionBehavior(
        broadcastDimensions.size());

    // Use static type info.
    for (const auto& [idx, dim] : llvm::enumerate(operandTy.getShape())) {
      if (ShapedType::isDynamic(dim)) continue;
      expansionBehavior[idx] = (dim == 1);
    }

    // Use annotated expansion behavior, if available.
    if (op.getKnownExpandingDimensions()) {
      for (const auto& it :
           op.getKnownExpandingDimensions()->getValues<int64_t>()) {
        expansionBehavior[it] = true;
      }
    }
    if (op.getKnownNonexpandingDimensions()) {
      for (const auto& it :
           op.getKnownNonexpandingDimensions()->getValues<int64_t>()) {
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
  static Value getBroadcastOperand(
      PatternRewriter& rewriter, Location loc, Value operand,
      llvm::function_ref<bool(int64_t)> isExpandingDim) {
    auto operandTy = operand.getType().dyn_cast<RankedTensorType>();

    SmallVector<int64_t> updatedOperandShape =
        llvm::to_vector(operandTy.getShape());
    for (size_t i = 0; i < updatedOperandShape.size(); ++i) {
      if (isExpandingDim(i)) updatedOperandShape[i] = 1;
    }

    auto updatedOperandTy =
        RankedTensorType::get(updatedOperandShape, operandTy.getElementType());

    if (updatedOperandTy != operandTy) {
      operand = rewriter.create<tensor::CastOp>(loc, updatedOperandTy, operand);
    }

    return operand;
  }

  static ShapedType getBroadcastResultType(
      Value operand, RankedTensorType resultTy, ArrayRef<int64_t> dimensions,
      llvm::function_ref<bool(int64_t)> isExpandingDim) {
    auto operandShape = operand.getType().cast<RankedTensorType>().getShape();
    auto broadcastResultShape = llvm::to_vector(resultTy.getShape());

    for (auto [operandIndex, resultIndex] : llvm::enumerate(dimensions)) {
      if (isExpandingDim(operandIndex)) continue;
      broadcastResultShape[resultIndex] = operandShape[operandIndex];
    }

    return RankedTensorType::get(broadcastResultShape,
                                 resultTy.getElementType());
  }
};

template <typename OpTy>
class TransposeConverter
    : public DataMovementOpConverter<TransposeConverter<OpTy>, OpTy> {
 public:
  using DataMovementOpConverter<TransposeConverter<OpTy>,
                                OpTy>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(OpTy op, Builder* b) {
    auto resultType = getHloOpResultType(op).template cast<ShapedType>();
    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.resize(resultType.getRank());
    for (const auto& permutation : llvm::enumerate(op.getPermutation())) {
      inputExprs[permutation.value().getZExtValue()] =
          b->getAffineDimExpr(permutation.index());
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class TransposeOpToTransposeConverter
    : public OpConversionPattern<stablehlo::TransposeOp> {
  using OpConversionPattern<stablehlo::TransposeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::TransposeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto resultTy = typeConverter->convertType(op.getType()).cast<ShapedType>();

    auto loc = op.getLoc();
    Value emptyTensor =
        getEmptyTensorFor(rewriter, loc, resultTy, op, adaptor.getOperands());

    auto permutation = rewriter.getDenseI64ArrayAttr(
        llvm::to_vector(op.getPermutation().getValues<int64_t>()));

    rewriter.replaceOpWithNewOp<linalg::TransposeOp>(
        op, adaptor.getOperand(), emptyTensor, permutation,
        linalg::getPrunedAttributeList(op));
    return success();
  }
};

class BitcastConvertConverter
    : public OpConversionPattern<stablehlo::BitcastConvertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::BitcastConvertOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) return failure();

    auto inputType = adaptor.getOperand().getType().cast<RankedTensorType>();
    auto outputType =
        typeConverter->convertType(op.getType()).cast<RankedTensorType>();
    auto loc = op.getLoc();

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
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange args) {
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

// Lowers mhlo.RealDynamicSliceOp to tensor.extract_slice and other
// arith/tensor dialect ops.
class RealDynamicSliceConverter
    : public OpConversionPattern<stablehlo::RealDynamicSliceOp> {
 public:
  using OpConversionPattern<stablehlo::RealDynamicSliceOp>::OpConversionPattern;

  // Computes size of a slice as
  //   size = ceil((limit - start)/stride)
  static Value computeSize(Location loc, Value start, Value limit, Value stride,
                           ConversionPatternRewriter& b) {
    Value delta = b.create<arith::SubIOp>(loc, limit, start);
    Value ret = b.create<arith::CeilDivUIOp>(loc, delta, stride);
    if (ret.getType().isIndex()) return ret;
    return b.create<arith::IndexCastOp>(loc, b.getIndexType(), ret);
  }

  LogicalResult matchAndRewrite(
      stablehlo::RealDynamicSliceOp realDynamicSliceOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = realDynamicSliceOp.getLoc();
    auto argType = adaptor.getOperand().getType().dyn_cast<ShapedType>();
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

    auto resultType =
        this->typeConverter->convertType(realDynamicSliceOp.getType())
            .cast<RankedTensorType>();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<OpFoldResult, 4> offsets, sizes, strides;
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
class ReshapeOpConverter : public OpConversionPattern<stablehlo::ReshapeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ReshapeOp reshapeOp, stablehlo::ReshapeOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(reshapeOp))) return failure();
    auto operand = adaptor.getOperand();
    auto operandType = operand.getType().cast<ShapedType>();
    auto elemType = operandType.getElementType();
    auto resultType = reshapeOp.getType().cast<ShapedType>();

    if (!resultType.hasStaticShape()) return failure();

    // If any of the output dimensions is 0, the tensor has no elements. In that
    // case, we can just replace the reshape with an empty op.
    if (llvm::is_contained(resultType.getShape(), 0)) {
      rewriter.replaceOpWithNewOp<tensor::EmptyOp>(
          reshapeOp, resultType.getShape(), elemType);
      return success();
    }

    resultType = typeConverter->convertType(resultType).cast<ShapedType>();

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
        for (const auto& map : llvm::enumerate(*reassociationMap)) {
          // If the result dim is dynamic, we do not mind dynamic entries in the
          // source.
          if (resultType.isDynamicDim(map.index())) continue;
          for (auto targetDim : map.value()) {
            if (shape[targetDim] == ShapedType::kDynamic) shape[targetDim] = 1;
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
      SmallVector<AffineExpr, 4> exprs;
      for (int i = 0; i < n; ++i) exprs.push_back(rewriter.getAffineDimExpr(i));
      return exprs;
    };
    // Otherwise, we need to first reduce all source dimensions into one and
    // then expand to the destination dimensions. If there is only a single
    // source dimension, the reduce step can be skipped. TensorCollapseShape
    // expects a different rank of operand and result.
    if (operandType.getRank() != 1) {
      SmallVector<ReassociationExprs, 4> collapsingMap = {
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
      SmallVector<ReassociationExprs, 4> expandingMap = {
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
class IotaConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy iotaOp, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType resultShapedType = getHloOpResultType(iotaOp);
    if (!resultShapedType) return failure();
    resultShapedType = this->typeConverter->convertType(resultShapedType)
                           .template dyn_cast<ShapedType>();

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
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
          Value indexOp = nestedBuilder.create<linalg::IndexOp>(
              nestedLoc, iotaOp.getIotaDimension());
          Type unwrappedResultElementType = resultElementType;
          if (auto complexType =
                  unwrappedResultElementType.dyn_cast<ComplexType>())
            unwrappedResultElementType = complexType.getElementType();
          Value castOp = nestedBuilder.create<arith::IndexCastOp>(
              nestedLoc,
              nestedBuilder.getIntegerType(
                  unwrappedResultElementType.getIntOrFloatBitWidth()),
              indexOp);
          castOp = stablehlo::StableHloOpToStdScalarOp::mapOpOfType<
              stablehlo::ConvertOp>(nestedLoc, resultElementType,
                                    castOp.getType(), {castOp}, &nestedBuilder);
          nestedBuilder.create<linalg::YieldOp>(nestedLoc, castOp);
        },
        linalg::getPrunedAttributeList(iotaOp));
    rewriter.replaceOp(iotaOp, linalgOp.getResultTensors());
    return success();
  }
};

template <typename OpTy>
class IotaToMapConverter : public OpConversionPattern<OpTy> {
 public:
  using OpConversionPattern<OpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      OpTy iotaOp, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    ShapedType resultTy = getHloOpResultType(iotaOp);
    if (!resultTy) return failure();
    resultTy = this->typeConverter->convertType(resultTy)
                   .template dyn_cast<ShapedType>();

    Location loc = iotaOp.getLoc();
    Value empty = getEmptyTensorFor(rewriter, loc, resultTy, iotaOp,
                                    adaptor.getOperands());

    auto linalgOp = rewriter.create<linalg::MapOp>(
        loc, ValueRange{}, empty,
        [&](OpBuilder& nestedBuilder, Location nestedLoc, ValueRange /*args*/) {
          Value index = nestedBuilder.create<linalg::IndexOp>(
              nestedLoc, iotaOp.getIotaDimension());
          index = nestedBuilder.create<arith::IndexCastOp>(
              nestedLoc, nestedBuilder.getI64Type(), index);
          Value result = stablehlo::StableHloOpToStdScalarOp::mapOpOfType<
              stablehlo::ConvertOp>(nestedLoc, resultTy.getElementType(),
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
struct ConcatenateConverter
    : public OpConversionPattern<stablehlo::ConcatenateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ConcatenateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Shortcut the one-operand case, simplifies code below.
    if (adaptor.getOperands().size() == 1) {
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }

    auto resultType = this->typeConverter->convertType(op.getResult().getType())
                          .dyn_cast<RankedTensorType>();
    if (!resultType) return failure();

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
        [&](OpBuilder& nestedBuilder, Location loc, ValueRange) {
          OpBuilder b = nestedBuilder;
          Value concatDimSize = zero;
          Value result;

          SmallVector<Value, 4> extractIndices;
          extractIndices.reserve(nloops);
          for (int64_t i = 0; i < nloops; i++) {
            extractIndices.push_back(b.create<linalg::IndexOp>(loc, i));
          }

          Value indexOp = b.create<linalg::IndexOp>(loc, dim);
          for (const auto& it : llvm::enumerate(adaptor.getOperands())) {
            Value arg = it.value();
            Value newConcatDimSize;
            scf::IfOp ifOp;
            if (it.index() != (adaptor.getOperands().size() - 1)) {
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

class ConstConverterTensor : public OpConversionPattern<stablehlo::ConstantOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ConstantOp constOp, OpAdaptor /*adaptor*/,
      ConversionPatternRewriter& rewriter) const final {
    auto valueAttr = constOp.getValue().cast<DenseElementsAttr>();
    auto type =
        typeConverter->convertType(constOp.getType()).cast<ShapedType>();
    if (type != constOp.getType()) {
      // Signedness conversion.
      valueAttr = valueAttr.mapValues(type.getElementType(),
                                      [](const APInt& i) { return i; });
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(constOp, type, valueAttr);
    return success();
  }
};

// TODO(b/156787842): Support the lowering for dynamic shapes.
class ReverseConverter
    : public DataMovementOpConverter<ReverseConverter, stablehlo::ReverseOp> {
 public:
  using DataMovementOpConverter<ReverseConverter,
                                stablehlo::ReverseOp>::DataMovementOpConverter;
  static SmallVector<AffineMap, 2> getIndexingMaps(stablehlo::ReverseOp op,
                                                   Builder* b) {
    auto resultType = getHloOpResultType(op).cast<ShapedType>();
    auto nloops = resultType.getRank();
    SmallVector<AffineExpr, 2> inputExprs;
    inputExprs.reserve(nloops);
    for (int i = 0; i < nloops; ++i)
      inputExprs.push_back(b->getAffineDimExpr(i));
    for (auto dim : op.getDimensions()) {
      int i = dim.getZExtValue();
      if (resultType.isDynamicDim(i)) return {};
      int n = resultType.getShape()[i];
      inputExprs[i] = b->getAffineConstantExpr(n - 1) - inputExprs[i];
    }
    return {
        AffineMap::get(nloops, /*symbolCount=*/0, inputExprs, b->getContext()),
        b->getMultiDimIdentityMap(nloops)};
  }
};

class SliceConverter : public OpConversionPattern<stablehlo::SliceOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::SliceOp sliceOp, typename stablehlo::SliceOp::Adaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto argType = adaptor.getOperands()[0].getType().dyn_cast<ShapedType>();
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(sliceOp, "expects known-rank args");
    }

    SmallVector<OpFoldResult, 3> offsets, sizes, strides;
    for (int i = 0, e = argType.getRank(); i < e; ++i) {
      auto start = sliceOp.getStartIndices().getValues<int64_t>()[i];
      auto limit = sliceOp.getLimitIndices().getValues<int64_t>()[i];
      auto stride = sliceOp.getStrides().getValues<int64_t>()[i];
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

class DynamicSliceConverter
    : public OpConversionPattern<stablehlo::DynamicSliceOp> {
 public:
  using OpConversionPattern<stablehlo::DynamicSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DynamicSliceOp dynamicSliceOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = dynamicSliceOp.getLoc();
    auto argType = adaptor.getOperand().getType().dyn_cast<ShapedType>();
    if (!argType || !argType.hasRank()) {
      return rewriter.notifyMatchFailure(dynamicSliceOp,
                                         "require known-rank args");
    }

    SmallVector<OpFoldResult, 3> startIndices, sizes;
    Type originalStartIndexType =
        dynamicSliceOp.getStartIndices().front().getType();
    for (const auto& en : llvm::enumerate(
             llvm::zip(adaptor.getStartIndices(),
                       dynamicSliceOp.getSliceSizes().getValues<int64_t>()))) {
      int64_t size = std::get<1>(en.value());
      sizes.push_back(rewriter.getI64IntegerAttr(size));

      // By mhlo.DynamicSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - size_indices[i])`
      Value startIndex = extractIndexFromTensor(
          rewriter, loc, std::get<0>(en.value()), originalStartIndexType);

      Value mn = rewriter.create<arith::ConstantIndexOp>(loc, 0);

      Value mx = rewriter.createOrFold<tensor::DimOp>(loc, adaptor.getOperand(),
                                                      en.index());
      mx = rewriter.createOrFold<arith::SubIOp>(
          loc, mx, rewriter.create<arith::ConstantIndexOp>(loc, size));

      startIndex = rewriter.create<arith::MaxSIOp>(loc, startIndex, mn);
      startIndex = rewriter.create<arith::MinSIOp>(loc, startIndex, mx);

      startIndices.push_back(startIndex);
    }

    int64_t rank = argType.getRank();
    SmallVector<OpFoldResult, 3> strides(rank, rewriter.getI64IntegerAttr(1));

    auto resultType = this->typeConverter->convertType(dynamicSliceOp.getType())
                          .cast<RankedTensorType>();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        dynamicSliceOp, resultType, adaptor.getOperand(), startIndices, sizes,
        strides);
    return success();
  }
};

class DynamicUpdateSliceConverter
    : public OpConversionPattern<stablehlo::DynamicUpdateSliceOp> {
 public:
  using OpConversionPattern<
      stablehlo::DynamicUpdateSliceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::DynamicUpdateSliceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto loc = op.getLoc();
    auto operandType =
        adaptor.getOperand().getType().dyn_cast<RankedTensorType>();
    if (!operandType || !operandType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    auto updateType =
        adaptor.getUpdate().getType().dyn_cast<RankedTensorType>();
    if (!updateType || !updateType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "require static ranked type for operand");
    }

    // We do not have to clamp sizes because the semantic of `update`
    // guarantees that it is always in the bounds. See
    // https://www.tensorflow.org/xla/operation_semantics#dynamicupdateslice
    SmallVector<OpFoldResult, 3> sizes;
    for (auto size : updateType.getShape()) {
      sizes.push_back(rewriter.getIndexAttr(size));
    }

    SmallVector<OpFoldResult, 3> startIndices;
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    for (const auto& en : llvm::enumerate(adaptor.getStartIndices())) {
      // By mhlo.DynamicUpdateSlice definition:
      //   `start_indices[i] = clamp(start_indices[i],
      //       0, operand.dimension_size[i] - update.dimension_size[i])`
      Value startIndex =
          extractIndexFromTensor(rewriter, loc, en.value(),
                                 op.getStartIndices()[en.index()].getType());
      Value ub = rewriter.create<arith::ConstantIndexOp>(
          loc, operandType.getDimSize(en.index()) -
                   updateType.getDimSize(en.index()));

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

class MapOpToGenericConverter : public OpConversionPattern<stablehlo::MapOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) return failure();

    auto resultType =
        typeConverter->convertType(op.getType()).cast<ShapedType>();
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
    Region& region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getComputation(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(op.getNumOperands() +
                                                          1);

    for (const auto& it : llvm::enumerate(op.getOperation()->getOperands())) {
      signatureConverter.addInputs(
          it.index(),
          typeConverter->convertType(
              it.value().getType().cast<ShapedType>().getElementType()));
    }
    signatureConverter.addInputs(resultType.getElementType());

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

class MapOpToMapConverter : public OpConversionPattern<stablehlo::MapOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::MapOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (failed(verifyHloOpBufferOrTensorSemantics(op))) return failure();

    auto resultType =
        typeConverter->convertType(op.getType()).cast<ShapedType>();
    assert(op.getDimensions().size() == resultType.getRank() &&
           "Expected a pointwise map");

    Location loc = op.getLoc();
    Value operand0 = adaptor.getOperands()[0];
    SmallVector<Value> coercedOperands = {operand0};
    for (Value operand : llvm::drop_begin(adaptor.getOperands(), 1)) {
      coercedOperands.push_back(coerceTensorShape(
          rewriter, loc, cast<TypedValue<ShapedType>>(operand),
          operand0.getType()));
    }
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, tensor::getMixedSizes(rewriter, loc, operand0),
        resultType.getElementType());

    auto linalgOp = rewriter.create<linalg::MapOp>(
        loc, coercedOperands, output,
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. We scalarize the operands and add a
    // scalar operand representing the output tensor.
    Region& region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getComputation(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(op.getNumOperands());

    for (const auto& it : llvm::enumerate(op.getOperation()->getOperands())) {
      signatureConverter.addInputs(
          it.index(),
          typeConverter->convertType(
              it.value().getType().cast<ShapedType>().getElementType()));
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    auto result = rewriter.createOrFold<tensor::CastOp>(loc, resultType,
                                                        linalgOp.getResults());
    rewriter.replaceOp(op, result);
    return success();
  }
};

SmallVector<Value, 8> getReduceOpEmptyTensorDynSizes(
    OpBuilder& b, Location loc, Value arg, ShapedType resultType,
    ArrayRef<int64_t> reductionDims) {
  llvm::SmallSetVector<int, 4> s;
  for (auto dim : reductionDims) s.insert(dim);

  SmallVector<unsigned, 4> parallelDims;
  SmallVector<Value, 8> dynShape;
  int rank = arg.getType().cast<RankedTensorType>().getRank();
  for (int i = 0, j = 0; i < rank; ++i) {
    if (s.count(i)) continue;
    if (!resultType.isDynamicDim(j++)) continue;
    dynShape.push_back(b.create<tensor::DimOp>(loc, arg, i));
  }

  return dynShape;
}

class ReduceRegionReturnOpConversion
    : public OpConversionPattern<stablehlo::ReturnOp> {
 public:
  using OpConversionPattern<stablehlo::ReturnOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    if (!isInBodyOfLinalgOps(op)) {
      return failure();
    }
    SmallVector<Value, 4> operands(adaptor.getOperands());
    for (size_t i = 0; i < operands.size(); ++i) {
      if (operands[i].getType().isa<ShapedType>()) {
        auto loc = operands[i].getLoc();
        operands[i] = rewriter.create<tensor::ExtractOp>(loc, operands[i]);
      }
    }
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, operands);
    return success();
  }
};

class ReduceOpToGenericConverter
    : public OpConversionPattern<stablehlo::ReduceOp> {
 public:
  using OpConversionPattern<stablehlo::ReduceOp>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();

    int numOperands = static_cast<int>(adaptor.getInputs().size());

    if (llvm::any_of(adaptor.getInputs(), [](Value v) {
          return !v.getType().isa<RankedTensorType>();
        })) {
      return rewriter.notifyMatchFailure(op, "expects known-rank args");
    }
    auto srcRank =
        adaptor.getInputs()[0].getType().cast<ShapedType>().getRank();

    SmallVector<int64_t, 4> reductionDims = extract1DVector(op.getDimensions());

    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    SmallVector<Value> outputs;
    SmallVector<AffineMap, 3> indexingMaps;
    for (auto [operand, initValue, resultType] :
         llvm::zip(adaptor.getInputs(), adaptor.getInitValues(), resultTypes)) {
      // Check if init_value is constant. If so, inline the value into the
      // region.
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);

      SmallVector<Value, 8> dynShape = getReduceOpEmptyTensorDynSizes(
          rewriter, loc, operand, resultType, reductionDims);
      auto emptyTensor = getEmptyTensor(rewriter, loc, resultType, dynShape);
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor).result();
      outputs.push_back(filledTensor);
    }

    // Prepare indexing maps for linalg generic op. The elements are for src
    // and dst. Transpose `src` to make the reduction loops be the innermost,
    // because it's easier to fully utilize processors.
    indexingMaps.append(
        numOperands, getTransposeMapForReduction(rewriter.getContext(),
                                                 (int)srcRank, reductionDims));

    // The indexing map of `dst` should drop the reduction loops. Since the
    // reduction loops now are all in the innermost, drops
    // `reduction_dims.size()` dimensions. We don't need an inverse
    // permutation here because they are the same.
    SmallVector<AffineExpr, 4> exprs;
    for (int i = 0, e = srcRank - reductionDims.size(); i < e; ++i)
      exprs.push_back(rewriter.getAffineDimExpr(i));
    indexingMaps.append(numOperands,
                        AffineMap::get(srcRank, /*symbolCount=*/0, exprs,
                                       rewriter.getContext()));

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, /*resultTensorTypes=*/resultTypes, adaptor.getInputs(),
        /*outputBuffers=*/ValueRange{outputs}, indexingMaps,
        getParallelAndReductionIterators(srcRank, reductionDims.size()),
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    // Convert the signature of the body. The reduce op region apply function
    // has a signature (lhs, rhs) -> output, all of the same tensor type t.
    // This is converted to a function with the same signature but with
    // element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
    // be converted to "(f32, f32, f32)".
    Region& region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getBody(), region, region.end());
    TypeConverter::SignatureConversion signatureConverter(numOperands * 2);

    // The mhlo ReduceOp requires that the seed be used as a LHS operand inside
    // the region, and the seed is encoded in linalg in the intial out value, so
    // modify the signature of the block and the value mappings, so the output
    // args will correlate with the original LHS and the inputs correlate with
    // the original RHS.
    for (const auto& [idx, val] : llvm::enumerate(op.getInputs())) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + numOperands,
          // type for the new operand number 'idx'.
          typeConverter->convertType(
              val.getType().cast<ShapedType>().getElementType()));
    }
    for (const auto& [idx, val] : llvm::enumerate(op.getInitValues())) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for the new operand number 'idx' + 'numOperands'.
          typeConverter->convertType(
              val.getType().cast<ShapedType>().getElementType()));
    }

    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct ReduceOpToReduceConverter
    : public OpConversionPattern<stablehlo::ReduceOp> {
  using OpConversionPattern<stablehlo::ReduceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::ReduceOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto reductionDims =
        llvm::to_vector(op.getDimensions().getValues<int64_t>());
    // mhlo.reduce doesn't specify the order of the reduction dimensions.
    llvm::sort(reductionDims);

    auto toRankedTensor = [](Value v) -> RankedTensorType {
      return v.getType().dyn_cast<RankedTensorType>();
    };

    SmallVector<Value> outputs;
    SmallVector<RankedTensorType> operandTypes, initTypes;
    SmallVector<Type> resultTypes;
    if (failed(typeConverter->convertTypes(op.getResultTypes(), resultTypes)))
      return failure();

    Location loc = op.getLoc();
    for (auto [operand, initValue, resultType] :
         llvm::zip(adaptor.getInputs(), adaptor.getInitValues(), resultTypes)) {
      auto initType = toRankedTensor(initValue);
      if (!initType)
        return rewriter.notifyMatchFailure(op,
                                           "expects known-rank init values");
      initTypes.push_back(initType);
      auto operandType = toRankedTensor(operand);
      if (!operandType)
        return rewriter.notifyMatchFailure(op, "expects known-rank operands");
      operandTypes.push_back(operandType);
      initValue = rewriter.createOrFold<tensor::ExtractOp>(loc, initValue);
      auto tensorResultType = resultType.cast<RankedTensorType>();
      // For linalg.reduce, the result type's dimensions must match the input's
      // dimensions, whereas MHLO allows replacing static dimensions with
      // dynamic ones.
      SmallVector<int64_t> resultShape;
      SmallVector<Value, 8> dynShape;
      for (auto [index, dim] :
           llvm::enumerate(operand.getType().cast<ShapedType>().getShape())) {
        if (!llvm::is_contained(reductionDims, index)) {
          resultShape.push_back(dim);
          if (ShapedType::isDynamic(dim)) {
            dynShape.push_back(
                rewriter.create<tensor::DimOp>(loc, operand, index));
          }
        }
      }

      Value emptyTensor = rewriter.create<tensor::EmptyOp>(
          loc, resultShape, tensorResultType.getElementType(), dynShape);
      Value filledTensor =
          rewriter.create<linalg::FillOp>(loc, initValue, emptyTensor).result();
      outputs.push_back(filledTensor);
    }

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getInputs(), outputs, reductionDims,
        /*bodyBuild=*/nullptr, linalg::getPrunedAttributeList(op));

    Region& region = linalgOp.getRegion();
    rewriter.inlineRegionBefore(op.getBody(), region, region.end());

    // Convert the signature of the body. The reduce op 'computation' region
    // apply function has a signature with tensor types, this is converted to a
    // function with element types. E.g. the signature "(tensor<f32>,
    // tensor<f32>) -> tensor<f32>" will be converted to "(f32, f32) -> f32".
    // Also, we need to swap the operands of the function. The mhlo.reduce op
    // expects the init values to be the first parameters of the apply function,
    // while the linalg.reduction op expects the init values as the last
    // parameters of the 'combiner' region apply function.
    TypeConverter::SignatureConversion signatureConverter(
        linalgOp.getNumDpsInputs() * 2);
    assert(linalgOp.getNumDpsInputs() == linalgOp.getNumDpsInits());
    for (const auto& [idx, val] : llvm::enumerate(operandTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx + linalgOp.getNumDpsInputs(),
          // type for new operand number 'idx'.
          typeConverter->convertType(val.getElementType()));
    }
    for (const auto& [idx, val] : llvm::enumerate(initTypes)) {
      signatureConverter.addInputs(
          /*origInputNo=*/idx,
          // type for new operand number 'idx' + linalgOp.getNumInputs()
          typeConverter->convertType(val.getElementType()));
    }
    rewriter.applySignatureConversion(&region, signatureConverter,
                                      getTypeConverter());

    // Cast the result to the correct type.
    SmallVector<Value> results;
    for (auto [result, resultType] :
         llvm::zip(linalgOp.getResults(), resultTypes)) {
      results.push_back(
          rewriter.createOrFold<tensor::CastOp>(loc, resultType, result));
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

/// Converts xla-hlo.select_and_scatter op to a sequence of linalg.generics ops.
/// The current version computes the scattered index and populates the correct
/// value for each tile. It does not currently handle overlapping tiles.
struct SelectAndScatterNoOverlapConverter
    : public OpConversionPattern<stablehlo::SelectAndScatterOp> {
  using OpConversionPattern<stablehlo::SelectAndScatterOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::SelectAndScatterOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    Location loc = op.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    Value source = op.getSource();
    Value operand = op.getOperand();
    Value init = op.getInitValue();

    auto sourceTy = source.getType().dyn_cast<RankedTensorType>();
    auto operandTy = operand.getType().dyn_cast<RankedTensorType>();
    auto initTy = init.getType().dyn_cast<RankedTensorType>();
    auto resultTy = op.getResult().getType().dyn_cast<RankedTensorType>();
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

    auto strides =
        llvm::to_vector(op.getWindowStridesAttr().getValues<int64_t>());
    auto window =
        llvm::to_vector(op.getWindowDimensionsAttr().getValues<int64_t>());

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
    // window for the defined mhlo.select_and_scatter. This involves
    // iterating over the window of the operand a computing the index.
    // Rather than storing N indices we compute the row major identifier
    // in the window, to specify which location should be scattered to.

    // Output specifies the `rank` parallel iterators that correspond to
    // output values.
    SmallVector<AffineExpr, 4> outputExprs;
    for (int i = 0, s = rank; i < s; ++i)
      outputExprs.push_back(b.getAffineDimExpr(i));

    // For the output we need to define the reduction across the window
    // width and height. This includes applying striding behavior and
    // adding the additional reduction iterators. We skip length-1 dimensions
    // as the reduction is degenerate.
    SmallVector<int64_t> filteredWindows, filteredStrides;
    SmallVector<AffineExpr, 4> sourceExprs(outputExprs);
    SmallVector<AffineExpr, 4> windowExprs;
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
    SmallVector<Value, 4> reduceDynSizes;
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
    auto& reduceRegion = reduceGeneric.getRegion();
    rewriter.setInsertionPoint(reduceGeneric);
    rewriter.cloneRegionBefore(op.getSelect(), reduceRegion,
                               reduceRegion.end());

    // This includes convert `mhlo` scalar-tensor regions to `linalg` scalars.
    TypeConverter::SignatureConversion reduceSignConverter(4);
    reduceSignConverter.addInputs(0, srcETy);
    reduceSignConverter.addInputs(srcETy);
    reduceSignConverter.addInputs(1, destETy);
    reduceSignConverter.addInputs(indexETy);
    rewriter.applySignatureConversion(&reduceRegion, reduceSignConverter,
                                      getTypeConverter());

    // Grab the terminator and use the turned value to now select the
    // correct index and value.
    auto& reduceBlock = reduceRegion.front();
    auto* reduceTerminator = reduceBlock.getTerminator();
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

    // Original terminator is an mhlo.return we no longer need.
    rewriter.eraseOp(reduceTerminator);
    b.setInsertionPoint(op);

    Value reduceIndex = reduceGeneric.getResult(1);
    ShapedType reduceIndexTy = reduceIndex.getType().cast<ShapedType>();

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
    auto& scatterRegion = scatterGeneric.getRegion();
    b.setInsertionPoint(scatterGeneric);
    rewriter.cloneRegionBefore(op.getScatter(), scatterRegion,
                               scatterRegion.end());

    TypeConverter::SignatureConversion scatterSignConverter(4);
    scatterSignConverter.addInputs(indexETy);
    scatterSignConverter.addInputs(0, sourceTy.getElementType());
    scatterSignConverter.addInputs(1, sourceTy.getElementType());
    rewriter.applySignatureConversion(&scatterRegion, scatterSignConverter,
                                      getTypeConverter());

    auto& scatterBlock = scatterRegion.front();
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
      if (strides[i] > 1) dims.push_back(collapseDim + 1);

      reassociationMap.push_back(ReassociationIndices(dims));
      collapseDim += dims.size();
    }

    Value collapse = b.create<tensor::CollapseShapeOp>(
        scatterGeneric.getResult(0), reassociationMap);
    auto collapseTy = collapse.getType().cast<ShapedType>();

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
struct PadOpNegativePaddingConversion
    : public OpConversionPattern<stablehlo::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<int64_t, 4> padLow;
    SmallVector<int64_t, 4> padHigh;
    SmallVector<OpFoldResult, 4> sliceStarts;

    bool hasNegativePadding = false;
    for (int64_t low : op.getEdgePaddingLow().getValues<int64_t>()) {
      if (low >= 0) {
        padLow.push_back(low);
        sliceStarts.push_back(rewriter.getIndexAttr(0));
      } else {
        padLow.push_back(0);
        sliceStarts.push_back(rewriter.getIndexAttr(-low));
        hasNegativePadding = true;
      }
    }

    for (int64_t high : op.getEdgePaddingHigh().getValues<int64_t>()) {
      if (high >= 0) {
        padHigh.push_back(high);
      } else {
        padHigh.push_back(-high);
        hasNegativePadding = true;
      }
    }

    // If there's no negative edge padding we're done.
    if (!hasNegativePadding) return failure();

    // Create a new pad op with the positive values.
    Value pad = rewriter.create<stablehlo::PadOp>(
        op.getLoc(), adaptor.getOperand(), adaptor.getPaddingValue(),
        rewriter.getI64TensorAttr(padLow), rewriter.getI64TensorAttr(padHigh),
        op.getInteriorPadding());

    // Then slice according to the negative edge padding. Static shapes only for
    // now.
    if (!op.getType().hasStaticShape()) return failure();
    SmallVector<OpFoldResult, 4> sizes(llvm::map_range(
        op.getType().getShape(),
        [&](int64_t dim) { return rewriter.getIndexAttr(dim); }));
    SmallVector<OpFoldResult, 4> strides(sliceStarts.size(),
                                         rewriter.getIndexAttr(1));
    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, pad, sliceStarts,
                                                        sizes, strides);
    return success();
  }
};

/// Converts mhlo.pad operation to tensor.pad or tensor.insert_slice.
struct PadOpConversion : public OpConversionPattern<stablehlo::PadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::PadOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getResult().getType());

    // Negative edge padding is decomposed separately.
    auto isNegative = [](const APInt& intVal) { return intVal.isNegative(); };
    if (llvm::any_of(op.getEdgePaddingLow().getValues<APInt>(), isNegative) ||
        llvm::any_of(op.getEdgePaddingHigh().getValues<APInt>(), isNegative))
      return failure();

    Value paddingVal = rewriter.createOrFold<tensor::ExtractOp>(
        loc, adaptor.getPaddingValue());

    SmallVector<OpFoldResult, 4> low(
        op.getEdgePaddingLow().getValues<IntegerAttr>());

    // If there is no interior padding lower to tensor.pad directly.
    if (llvm::all_of(op.getInteriorPadding().getValues<APInt>(),
                     [](const APInt& intVal) { return intVal.isZero(); })) {
      SmallVector<OpFoldResult, 4> high(
          op.getEdgePaddingHigh().getValues<IntegerAttr>());
      auto padTensorOp = rewriter.create<tensor::PadOp>(
          loc, resultType, adaptor.getOperand(), low, high, paddingVal);
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
    auto operandType = adaptor.getOperand().getType().cast<ShapedType>();
    auto sizes = llvm::to_vector<4>(llvm::map_range(
        llvm::seq<int64_t>(0, operandType.getRank()),
        [&](int64_t dim) -> OpFoldResult {
          if (!operandType.isDynamicDim(dim))
            return rewriter.getIndexAttr(operandType.getDimSize(dim));
          return rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), dim)
              .getResult();
        }));
    // Map interior padding to strides.
    auto strides = llvm::to_vector<4>(
        llvm::map_range(op.getInteriorPadding().getValues<IntegerAttr>(),
                        [&](IntegerAttr stride) -> OpFoldResult {
                          return rewriter.getIntegerAttr(stride.getType(),
                                                         stride.getValue() + 1);
                        }));

    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, adaptor.getOperand(), fill, low, sizes, strides);
    return success();
  }
};

/// Converts xla-hlo.torch_index_select op to a linalg.generic op.
struct TorchIndexSelectOpConversion
    : public OpConversionPattern<stablehlo::TorchIndexSelectOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::TorchIndexSelectOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    int axis = static_cast<int>(op.getDim());
    int batch = static_cast<int>(op.getBatchDims());
    auto indexShapedType = adaptor.getIndex().getType().cast<ShapedType>();
    int numIndices = static_cast<int>(indexShapedType.getRank());
    auto operandShapedType = adaptor.getOperand().getType().cast<ShapedType>();
    if (axis < 0) axis += static_cast<int>(operandShapedType.getRank());
    if (batch < 0) batch += numIndices;

    Location loc = op.getLoc();
    auto resultType = this->typeConverter->convertType(op.getResult().getType())
                          .cast<ShapedType>();
    int rank = static_cast<int>(resultType.getRank());

    // The output shape is
    //   `params[:axis] + indices[batch_dims:] + params[axis + 1:]`
    SmallVector<Value, 4> dynSizes;
    for (int i = 0; i < rank; ++i) {
      if (!resultType.isDynamicDim(i)) continue;
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
    SmallVector<Value, 4> dynSliceSizes;
    SmallVector<AffineExpr, 4> sliceExprs;
    auto resultShape = resultType.getShape();
    for (int i = 0; i < axis; ++i) {
      sliceExprs.push_back(rewriter.getAffineDimExpr(i));
      sliceShape.push_back(resultShape[i]);
      if (!resultType.isDynamicDim(i)) continue;
      dynSliceSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), i));
    }
    for (int i = axis + numIndices - batch; i < rank; ++i) {
      sliceExprs.push_back(rewriter.getAffineDimExpr(i));
      sliceShape.push_back(resultShape[i]);
      if (!resultType.isDynamicDim(i)) continue;
      int idx = i - (axis + numIndices - batch) + axis + 1;
      dynSliceSizes.push_back(
          rewriter.create<tensor::DimOp>(loc, adaptor.getOperand(), idx));
    }

    // Setup AffineMap for operand tensor.
    SmallVector<AffineExpr, 4> exprs;
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

    SmallVector<Type, 4> bodyArgTypes;
    SmallVector<Value, 2> linalgOpArgs = {adaptor.getIndex(), sliceOp};
    // Add a block to the region.
    auto* region = &linalgOp.getRegion();
    auto* block = rewriter.createBlock(region, region->end());
    for (auto blockArgs : linalgOpArgs) {
      bodyArgTypes.push_back(
          blockArgs.getType().cast<ShapedType>().getElementType());
    }
    block->addArguments(bodyArgTypes,
                        SmallVector<Location>(bodyArgTypes.size(), loc));
    block->addArguments(resultType.getElementType(), loc);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(block);

    Value castedValue = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), block->getArgument(0));

    SmallVector<Value, 4> indices;
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

class SetDimensionSizeConverter
    : public OpConversionPattern<stablehlo::SetDimensionSizeOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::SetDimensionSizeOp setDimensionSizeOp, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // We can lower SetDimensionSize to tensor extract. This turns into a
    // regular dynamic shape. Note that the bounds annotation is still around
    // but may be no longer valid depending on choices made by bufferization.
    Location loc = setDimensionSizeOp.getLoc();
    auto resultType = setDimensionSizeOp.getType().cast<RankedTensorType>();

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

struct ConvertStableHloToLinalg
    : impl::ConvertStableHloToLinalgBase<ConvertStableHloToLinalg> {
  using ConvertStableHloToLinalgBase::ConvertStableHloToLinalgBase;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    scf::SCFDialect, complex::ComplexDialect, math::MathDialect,
                    memref::MemRefDialect, shape::ShapeDialect>();
  }

  void runOnOperation() override {
    MLIRContext& ctx = getContext();
    RewritePatternSet patterns(&ctx);
    ConversionTarget target(ctx);
    target.addLegalDialect<
        bufferization::BufferizationDialect, arith::ArithDialect,
        complex::ComplexDialect, linalg::LinalgDialect, math::MathDialect,
        tensor::TensorDialect, sparse_tensor::SparseTensorDialect,
        scf::SCFDialect, shape::ShapeDialect>();

    target.addLegalOp<UnrealizedConversionCastOp>();

    auto typeConverter = createStableHloToLinalgTypeConverter();
    auto func = getOperation();
    // TODO: Port `populateScalarHloToArithmeticConversionPatterns`.
    populateStableHloToLinalgConversionPatterns(&ctx, *typeConverter, &patterns,
                                                this->enablePrimitiveOps);
    if (failed(applyPartialConversion(func, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

void populateStableHloToLinalgConversionPatterns(MLIRContext* context,
                                                 TypeConverter& typeConverter,
                                                 RewritePatternSet* patterns,
                                                 bool enablePrimitiveOps) {
  // clang-format off
  patterns->add<
      BitcastConvertConverter,
      ConcatenateConverter,
      ConstConverterTensor,
      EinsumToLinalgConverter,
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
      SelectAndScatterNoOverlapConverter,  // TODO(#12678): Add tests.
      ReduceRegionReturnOpConversion       // TODO(#12678): Add tests.
      >(typeConverter, context);

  detail::populatePointwiseStableHloToLinalgConversionPatterns(
      context, typeConverter, patterns, enablePrimitiveOps);

  if (enablePrimitiveOps) {
    patterns->add<
      BroadcastInDimOpToBroadcastConverter,
      BroadcastOpToBroadcastConverter,
      DynamicBroadcastInDimOpToBroadcastConverter,
      IotaToMapConverter<stablehlo::IotaOp>,
      IotaToMapConverter<stablehlo::DynamicIotaOp>,
      MapOpToMapConverter,        // TODO(#12678): Add tests.
      ReduceOpToReduceConverter,  // TODO(#12678): Add tests.
      TransposeOpToTransposeConverter
    >(typeConverter, context);
  } else {
    patterns->add<
      BroadcastConverter<stablehlo::BroadcastOp>,
      IotaConverter<stablehlo::IotaOp>,
      IotaConverter<stablehlo::DynamicIotaOp>,
      HloBroadcastInDimConverter,
      HloDynamicBroadcastInDimConverter,
      MapOpToGenericConverter,     // TODO(#12678): Add tests.
      ReduceOpToGenericConverter,  // TODO(#12678): Add tests.
      TransposeConverter<stablehlo::TransposeOp>
    >(typeConverter, context);
  }

  // clang-format on

  // TODO(#12678): Handle the convolution and reduce_window ops.

  detail::populateStableHloDotProdToLinalgConversionPatterns(
      context, typeConverter, patterns);
  linalg::populateEraseUnusedOperandsAndResultsPatterns(*patterns);
}

std::unique_ptr<TypeConverter> createStableHloToLinalgTypeConverter() {
  return std::make_unique<LinalgTypeConverter>();
}

}  // namespace mlir::iree_compiler::stablehlo
