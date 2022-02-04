// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// Returns true if `contractionOp` is of the form
//   matrix * transposed_matrix.
// That is, if there are 2 parallel iterators, say M and N, 1 additive reduction
// iterator, say K, and the indexing maps are {{M, K}, {N, K}, {M, N}}.
static bool isMatrixTimesMatrixTransposed(vector::ContractionOp contractionOp) {
  // Check that the reduction is additive.
  if (contractionOp.kind() != vector::CombiningKind::ADD) {
    return false;
  }
  // Check that there are 2 parallel and 1 reduction iterators.
  auto iteratorTypes = contractionOp.iterator_types().getValue();
  if (iteratorTypes.size() != 3) {
    return false;
  }
  SmallVector<int, 3> parallel_iterators;
  SmallVector<int, 3> reduction_iterators;
  for (int i = 0; i < 3; i++) {
    if (isParallelIterator(iteratorTypes[i])) {
      parallel_iterators.push_back(i);
    } else if (isReductionIterator(iteratorTypes[i])) {
      reduction_iterators.push_back(i);
    } else {
      return false;
    }
  }
  if (parallel_iterators.size() != 2 || reduction_iterators.size() != 1) {
    return false;
  }
  // Give the found iterators some idiomatic names.
  const int MIter = parallel_iterators[0];
  const int NIter = parallel_iterators[1];
  const int KIter = reduction_iterators[0];
  // Check that there are 3 indexing maps.
  auto indexingMaps = contractionOp.indexing_maps().getValue();
  if (indexingMaps.size() != 3) {
    return false;
  }
  // Check that the indexing maps have the expected form.
  const int expectedMapResults[3][2] = {
      {MIter, KIter}, {NIter, KIter}, {MIter, NIter}};
  for (int m = 0; m < 3; ++m) {
    auto map = indexingMaps[m].cast<AffineMapAttr>().getValue();
    if (map.getNumDims() != 3 || map.getNumResults() != 2) {
      return false;
    }
    for (int r = 0; r < 2; ++r) {
      int actualMapResult =
          map.getResults()[r].cast<AffineDimExpr>().getPosition();
      if (actualMapResult != expectedMapResults[m][r]) {
        return false;
      }
    }
  }
  return true;
}

// Returns true if `contractionOp` is of the form
//   matrix * transposed_matrix
// where matrix is a vector<{mSize}x{kSize}xType>, and
// transposed_matrix is a vector<{nSize}x{kSize}xType>
static bool isMatrixTimesMatrixTransposedOfGivenShape(
    vector::ContractionOp contractionOp, int64_t mSize, int64_t kSize,
    int64_t nSize) {
  if (!isMatrixTimesMatrixTransposed(contractionOp)) {
    return false;
  }
  VectorType lhsType = contractionOp.lhs().getType().cast<VectorType>();
  VectorType rhsType = contractionOp.rhs().getType().cast<VectorType>();
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  if (lhsShape[0] != mSize || lhsShape[1] != kSize || rhsShape[0] != nSize ||
      rhsShape[1] != kSize) {
    return false;
  }
  return true;
}

// Checks that the Value `extResult` is defined by an arith::ExtSIOp promoting
// from `extSrcType` to `extDstType`, and returns the input of the ExtSIOp.
// Note that this only looks at the immediately defining operation, so we likely
// want to have earlier passes that sink widening operations as far down as
// possible, which is probably just good regardless.
static Value getExtSIInput(Type extSrcType, Type extDstType, Value extResult) {
  auto extSIOp = extResult.getDefiningOp<arith::ExtSIOp>();
  if (!extSIOp) {
    return nullptr;
  }
  Value extInput = extSIOp.getIn();
  if (extInput.getType().cast<VectorType>().getElementType() != extSrcType) {
    return nullptr;
  }
  return extInput;
}

// Helper to create a 1D, contiguous slice of a 1D vector.
static Value extract1DSlice(PatternRewriter &rewriter, Location loc,
                            VectorType dstVecType, Value input, int position) {
  assert(input.getType().cast<VectorType>().getRank() == 1);
  assert(dstVecType.getRank() == 1);
  std::array<int64_t, 1> offsets{position};
  std::array<int64_t, 1> strides{1};
  return rewriter.create<vector::ExtractStridedSliceOp>(
      loc, input, offsets, dstVecType.getShape(), strides);
}

// Helper to flatten a N-dimensional vector to a 1D vector.
static Value flatten(PatternRewriter &rewriter, Location loc, Value vector) {
  VectorType inputVecType = vector.getType().cast<VectorType>();
  VectorType dstType = VectorType::get(inputVecType.getNumElements(),
                                       inputVecType.getElementType());
  return rewriter.create<vector::ShapeCastOp>(loc, dstType, vector);
}

/// Converts matrix-times-matrix-transposed vector.contracts with
/// lhs and rhs inputs defined by arith.extsi promoting from i8 to i32,
///
///     %lhs_i32 = arith.extsi %lhs_i8 : i8 to i32
///     %rhs_i32 = arith.extsi %rhs_i8 : i8 to i32
///     %result = vector.contract [...]
///                 %lhs_i32 : vector<8x4xi32>,
///                 %rhs_i32 : vector<8x4xi32>,
///                 %acc_i32 : vector<8x8xi32>,
///                 [...]
///
/// To vector ops reading directly from the %lhs_i8 and %rhs_i8 values
/// (bypassing the existing arith.extsi) and passing that to a llvm.inline_asm
/// block implementing the matrix multiplication arithmetic using Aarch64
/// dot-product instructions (sdot).
struct MMT_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm
    : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractionOp,
                                PatternRewriter &rewriter) const override {
    // Check if `contractionOp` matches, and obtain the un-promoted i8 input
    // LHS and RHS vectors, `lhsI8` and `rhsI8`.
    if (!isMatrixTimesMatrixTransposedOfGivenShape(contractionOp, 8, 4, 8)) {
      return failure();
    }
    Type I8Type = rewriter.getIntegerType(8);
    Type I32Type = rewriter.getIntegerType(32);
    VectorType accType = contractionOp.acc().getType().cast<VectorType>();
    if (accType.getElementType() != I32Type) {
      return failure();
    }
    Value lhsI8 = getExtSIInput(I8Type, I32Type, contractionOp.lhs());
    Value rhsI8 = getExtSIInput(I8Type, I32Type, contractionOp.rhs());
    if (!lhsI8 || !rhsI8) {
      return failure();
    }

    // `contractionOp` matches, start rewriting it. We only reference
    // the `lhsI8` and `rhsI8` values obtained above as the inputs of the
    // arith.extsi, so this rewrite will leave the existing arith.extsi without
    // any user (unless something else was using them), so they may be
    // removed by another transformation.
    Location loc = contractionOp.getLoc();
    // Flatten the inputs to 1D vectors.
    Value flatLhsI8 = flatten(rewriter, loc, lhsI8);
    Value flatRhsI8 = flatten(rewriter, loc, rhsI8);
    Value flatAcc = flatten(rewriter, loc, contractionOp.acc());

    // Create the 1D input vectors of 16 bytes each that are directly what
    // the target SIMD instructions will want.
    SmallVector<Value> lhsVec;
    SmallVector<Value> rhsVec;
    VectorType vector16xi8Type = VectorType::get({16}, I8Type);
    for (int position = 0; position < 8 * 4; position += 16) {
      lhsVec.push_back(
          extract1DSlice(rewriter, loc, vector16xi8Type, flatLhsI8, position));
      rhsVec.push_back(
          extract1DSlice(rewriter, loc, vector16xi8Type, flatRhsI8, position));
    }
    SmallVector<Value> accVec;
    VectorType int32x4Type = VectorType::get({4}, I32Type);
    for (int position = 0; position < 8 * 8; position += 4) {
      accVec.push_back(
          extract1DSlice(rewriter, loc, int32x4Type, flatAcc, position));
    }

    // Create the inline asm op's operands list.
    SmallVector<Value> asmOperands;
    // First the inputs operands.
    asmOperands.append(lhsVec);
    asmOperands.append(rhsVec);
    // Then the input-output operands.
    asmOperands.append(accVec);
    SmallVector<Type> asmOutputOperandTypes(
        llvm::map_range(accVec, [](Value v) { return v.getType(); }));

    // Create the inline asm op.
    auto returnType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                       asmOutputOperandTypes);
    auto dialectAttr = LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                                 LLVM::AsmDialect::AD_ATT);
    // The LLVM inline asm syntax is documented here:
    // https://llvm.org/docs/LangRef.html#inline-assembler-expressions
    LLVM::InlineAsmOp asmOp = rewriter.create<LLVM::InlineAsmOp>(
        loc, returnType, asmOperands,
        R"ASM(
            sdot $0.4s, $18.16b, $16.4b[0]
            sdot $1.4s, $19.16b, $16.4b[0]
            sdot $2.4s, $18.16b, $16.4b[1]
            sdot $3.4s, $19.16b, $16.4b[1]
            sdot $4.4s, $18.16b, $16.4b[2]
            sdot $5.4s, $19.16b, $16.4b[2]
            sdot $6.4s, $18.16b, $16.4b[3]
            sdot $7.4s, $19.16b, $16.4b[3]
            sdot $8.4s, $18.16b, $17.4b[0]
            sdot $9.4s, $19.16b, $17.4b[0]
            sdot $10.4s, $18.16b, $17.4b[1]
            sdot $11.4s, $19.16b, $17.4b[1]
            sdot $12.4s, $18.16b, $17.4b[2]
            sdot $13.4s, $19.16b, $17.4b[2]
            sdot $14.4s, $18.16b, $17.4b[3]
            sdot $15.4s, $19.16b, $17.4b[3]
          )ASM",
        "=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,=w,w,w,w,w,0,1,2,3,4,5,6,"
        "7,8,9,10,11,12,13,14,15",
        /*has_side_effects=*/false, /*is_align_stack=*/false, dialectAttr,
        /*operand_attrs=*/ArrayAttr());

    // Extract result vectors from the asm op.
    SmallVector<Value, 16> resVec;
    for (int i = 0; i < 16; ++i) {
      resVec.push_back(rewriter.create<LLVM::ExtractValueOp>(
          loc, int32x4Type, asmOp.getRes(), rewriter.getI64ArrayAttr({i})));
    }

    // Insert the result vectors of size 4 into the overall result vector of
    // size 64, still 1D.
    VectorType int32x64xType = VectorType::get({64}, I32Type);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, int32x64xType, DenseIntElementsAttr::get(int32x64xType, 0));
    for (int i = 0; i < 16; ++i) {
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resVec[i], result, std::array<int64_t, 1>{4 * i},
          std::array<int64_t, 1>{1});
    }

    // Cast the result from 1D to 2D and replace the original vector.contract.
    VectorType int32x8x8xType = VectorType::get({8, 8}, I32Type);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(contractionOp,
                                                     int32x8x8xType, result);
    return success();
  }
};

/// Converts matrix-times-matrix-transposed vector.contracts with
/// lhs and rhs inputs defined by arith.extsi promoting from i8 to i32,
///
///     %lhs_i32 = arith.extsi %lhs_i8 : i8 to i32
///     %rhs_i32 = arith.extsi %rhs_i8 : i8 to i32
///     %result = vector.contract [...]
///                 %lhs_i32 : vector<8x4xi32>,
///                 %rhs_i32 : vector<8x4xi32>,
///                 %acc_i32 : vector<8x8xi32>,
///                 [...]
///
/// To vector ops reading directly from the %lhs_i8 and %rhs_i8 values
/// (bypassing the existing arith.extsi) and passing that to a llvm.inline_asm
/// block implementing the matrix multiplication arithmetic using Aarch64
/// dot-product instructions (sdot).
/// It matches the same patterns as MMT_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm
struct MMT_8x4x8_i8i8i32_Aarch64Dotprod_Intrinsics
    : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractionOp,
                                PatternRewriter &rewriter) const override {
    if (!isMatrixTimesMatrixTransposedOfGivenShape(contractionOp, 8, 4, 8)) {
      return failure();
    }

    Type I8Type = rewriter.getIntegerType(8);
    Type I32Type = rewriter.getIntegerType(32);

    auto acc = contractionOp.acc();
    auto lhs = contractionOp.lhs();
    auto rhs = contractionOp.rhs();
    if (acc.getType().cast<VectorType>().getElementType() != I32Type) {
      return failure();
    }

    Value inLhs = getExtSIInput(I8Type, I32Type, lhs);
    Value inRhs = getExtSIInput(I8Type, I32Type, rhs);

    if (!inLhs || !inRhs) return failure();

    auto loc = contractionOp.getLoc();

    auto int32x4VType = VectorType::get({4}, I32Type);

    std::array<Value, 16> accChunks;
    {
      int idx = 0;
      for (int row = 0; row < 8; ++row) {
        auto accRow = rewriter.create<vector::ExtractOp>(
            loc, acc, ArrayRef<int64_t>{row});
        for (int col = 0; col < 8; col += 4) {
          auto accChunk = rewriter.create<vector::ExtractStridedSliceOp>(
              loc, accRow, ArrayRef<int64_t>{col}, ArrayRef<int64_t>{4},
              ArrayRef<int64_t>{1});
          assert(accChunk.getType() == int32x4VType);
          accChunks[idx++] = accChunk;
        }
      }
    }

    auto int8x4x4VType = VectorType::get({4, 4}, rewriter.getIntegerType(8));
    auto extract4x4 = [&](Value in, int rowOffset, int colOffset) {
      auto chunk = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, in, ArrayRef<int64_t>{rowOffset, colOffset},
          ArrayRef<int64_t>{4, 4}, ArrayRef<int64_t>{1, 1});
      assert(chunk.getType() == int8x4x4VType);
      return chunk;
    };

    std::array<Value, 2> lhsHalves = {extract4x4(inLhs, 0, 0),
                                      extract4x4(inLhs, 4, 0)};
    std::array<Value, 2> rhsHalves = {extract4x4(inRhs, 0, 0),
                                      extract4x4(inRhs, 4, 0)};

    auto int8Zero4x4 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(int8x4x4VType));
    auto sdot = [&](Value acc, Value a, Value b, int64_t lane) -> Value {
      auto bReplicatedLane = rewriter.create<vector::ShuffleOp>(
          loc, b, int8Zero4x4, ArrayRef<int64_t>{lane, lane, lane, lane});

      return rewriter.create<arm_neon::Sdot2dOp>(loc, int32x4VType, acc, a,
                                                 bReplicatedLane);
    };

    std::array<Value, 16> dstChunks;
    {
      int idx = 0;
      for (Value lhs : lhsHalves) {
        for (int lane = 0; lane < 4; ++lane) {
          for (Value rhs : rhsHalves) {
            dstChunks[idx] = sdot(accChunks[idx], rhs, lhs, lane);
            ++idx;
          }
        }
      }
    }

    // Put the results back in the accumulator
    {
      int idx = 0;
      for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 8; col += 4) {
          acc = rewriter.create<vector::InsertStridedSliceOp>(
              loc, dstChunks[idx++], acc, ArrayRef<int64_t>{row, col},
              ArrayRef<int64_t>{1});
        }
      }
    }
    rewriter.replaceOp(contractionOp, {acc});
    return success();
  }
};

class VectorContractCustomKernelsPass
    : public VectorContractCustomKernelsBase<VectorContractCustomKernelsPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect>();
    if (target_info.has(CustomKernelTargetFeature::Intrinsics)) {
      registry.insert<arm_neon::ArmNeonDialect>();
    }
  }
  LogicalResult initializeOptions(StringRef options) override {
    if (failed(Pass::initializeOptions(options))) {
      return failure();
    }
    if (failed(ParseCustomKernelsTargetInfo(arch, features, target_info))) {
      return failure();
    }
    if (intrinsics) {
      target_info.add(CustomKernelTargetFeature::Intrinsics);
    }
    return success();
  }
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateVectorContractCustomKernelsPatterns(target_info, patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }

 private:
  CustomKernelsTargetInfo target_info;
};

}  // namespace

void populateVectorContractCustomKernelsPatterns(
    const CustomKernelsTargetInfo &target_info, RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  if (target_info.has(CustomKernelTargetFeature::Aarch64Dotprod)) {
    if (target_info.has(CustomKernelTargetFeature::Intrinsics)) {
      patterns.insert<MMT_8x4x8_i8i8i32_Aarch64Dotprod_Intrinsics>(context);
    } else {
      patterns.insert<MMT_8x4x8_i8i8i32_Aarch64Dotprod_InlineAsm>(context);
    }
  }
}

std::unique_ptr<OperationPass<FuncOp>> createVectorContractCustomKernelsPass() {
  return std::make_unique<VectorContractCustomKernelsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
