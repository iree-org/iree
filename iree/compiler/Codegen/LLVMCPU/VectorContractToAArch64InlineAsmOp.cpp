// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/LLVM/LLVMTargetOptions.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

static bool isMatrixTimesMatrixTransposed(vector::ContractionOp contractionOp) {
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
  const int M = parallel_iterators[0];
  const int N = parallel_iterators[1];
  const int K = reduction_iterators[0];
  auto indexingMaps = contractionOp.indexing_maps().getValue();
  if (indexingMaps.size() != 3) {
    return false;
  }
  const int expectedMapResults[3][2] = {{M, K}, {N, K}, {M, N}};
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

static Value getExtInput(Type extSrcType, Type extDstType, Value extResult) {
  if (extResult.getType().cast<VectorType>().getElementType() != extDstType) {
    return nullptr;
  }
  auto extSIOp = extResult.getDefiningOp<arith::ExtSIOp>();
  if (!extSIOp) {
    return nullptr;
  }
  Value extInput = extSIOp.in();
  if (extInput.getType().cast<VectorType>().getElementType() != extSrcType) {
    return nullptr;
  }
  return extInput;
}

// WIP WIP WIP WIP WIP
// This currently gives very bad generated code. Working on it.
//
// From the 1D |input| vector, extract the segment
//   [position * S .. (position+1) * S - 1]
// where S is the size of the 1D vector type |dstVecType|
static Value extractChunk(PatternRewriter &rewriter, Location loc,
                          VectorType dstVecType, Value input, int position) {
  Value shapeCastOp = input.getDefiningOp<vector::ShapeCastOp>();

  /*if (shapeCastOp) {
    Value transferReadOp = shapeCastOp.getDefiningOp<vector::TransferReadOp>();
    if (transferReadOp) {
      fprintf(stderr, "transfer read!\n");
      Value chunkTransferReadOp = rewriter.create<vector::TransferReadOp>(loc, ...);
      Value chunkShapeCastOp = rewriter.create<vector::ShapeCastOp>(loc, ...);
      return chunkShapeCastOp;
    }
  }*/

  VectorType inputVecType = input.getType().cast<VectorType>();
  auto inputShape = inputVecType.getShape();
  auto dstShape = dstVecType.getShape();
  assert(inputShape.size() == 1);
  assert(dstShape.size() == 1);
  if (1) {
    // Try vector::ExtractStridedSliceOp ?
    SmallVector<int64_t> offsets{position * dstShape[0]};
    SmallVector<int64_t> sizes{dstShape[0]};
    SmallVector<int64_t> strides{1};

    return rewriter.create<vector::ExtractStridedSliceOp>(loc, input, offsets,
                                                          sizes, strides);
  } else {
    // Try casting to a 2D shape just so we can get the desired chunk with just
    // a vector::ExtractOp ?
    Type elemType = dstVecType.getElementType();
    VectorType input2DType =
        VectorType::get({inputShape[0] / dstShape[0], dstShape[0]}, elemType);
    Value input2D =
        rewriter.create<vector::ShapeCastOp>(loc, input2DType, input);
    auto posAttr = rewriter.getI64ArrayAttr(position);
    return rewriter.create<vector::ExtractOp>(loc, dstVecType, input2D,
                                              posAttr);
  }
}

/// Converts a vector.contract computing A*B^T where A and B are 8x4 matrices
/// of int32's that are themselves the result of an arith.extsi promoting from
/// i8, into an inline asm op using ARM NEON dotprod instructions.
struct ConvertVectorContract8x4x8_i8i8i32_ToAArch64InlineAsmPattern
    : public OpRewritePattern<vector::ContractionOp> {
 public:
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractionOp,
                                PatternRewriter &rewriter) const override {
    auto lhsType = contractionOp.lhs().getType().cast<VectorType>();
    auto rhsType = contractionOp.rhs().getType().cast<VectorType>();
    auto accType = contractionOp.acc().getType().cast<VectorType>();
    auto lhsShape = lhsType.getShape();
    auto rhsShape = rhsType.getShape();
    if (IREE::HAL::getLLVMTargetOptionsFromFlags().targetTriple.rfind(
            "aarch64-", 0) != 0) {
      return failure();
    }
    if (lhsShape[0] != 8 || lhsShape[1] != 4 || rhsShape[0] != 8 ||
        rhsShape[1] != 4) {
      return failure();
    }
    if (!isMatrixTimesMatrixTransposed(contractionOp)) {
      return failure();
    }

    Type I8Type = rewriter.getIntegerType(8);
    Type I32Type = rewriter.getIntegerType(32);

    if (accType.getElementType() != I32Type) {
      return failure();
    }

    Value inLhs = getExtInput(I8Type, I32Type, contractionOp.lhs());
    Value inRhs = getExtInput(I8Type, I32Type, contractionOp.rhs());

    if (!inLhs || !inRhs) return failure();

    auto loc = contractionOp.getLoc();

    auto int32x4VType = VectorType::get({4}, I32Type);

    SmallVector<Value> dstVec;
    for (int i = 0; i < 16; ++i) {
      SmallVector<int64_t> offsets{i / 2, (i % 2) * 4};
      SmallVector<int64_t> sizes{1, 4};
      SmallVector<int64_t> strides{1, 1};
      Value flatAcc = rewriter.create<vector::ShapeCastOp>(
          loc, VectorType::get({8 * 8}, I32Type), contractionOp.acc());
      dstVec.push_back(extractChunk(rewriter, loc, int32x4VType, flatAcc, i));
    }

    auto flatVectorType = VectorType::get({32}, I8Type);
    auto flatHalfVectorType = VectorType::get({16}, I8Type);

    auto lhs = rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, inLhs);
    auto rhs = rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, inRhs);

    auto lhs0 = extractChunk(rewriter, loc, flatHalfVectorType, lhs, 0);
    auto lhs1 = extractChunk(rewriter, loc, flatHalfVectorType, lhs, 1);
    auto rhs0 = extractChunk(rewriter, loc, flatHalfVectorType, rhs, 0);
    auto rhs1 = extractChunk(rewriter, loc, flatHalfVectorType, rhs, 1);

    auto returnType = LLVM::LLVMStructType::getLiteral(rewriter.getContext(),
                                                       {
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                           int32x4VType,
                                                       });

    /// TODO(ataei): We an have a formatter like the c++ inline asm to
    /// ssa-values to string names which will make the inline-assembly
    /// statements more redable e.g :
    /// sdot ${dstVec_0}.4s, ${lhs}.16b,${rhs}.4b[0]
    auto packedResult = rewriter.create<LLVM::InlineAsmOp>(
        loc, returnType,
        ArrayRef<Value>({
            lhs0,       lhs1,       rhs0,       rhs1,       dstVec[0],
            dstVec[1],  dstVec[2],  dstVec[3],  dstVec[4],  dstVec[5],
            dstVec[6],  dstVec[7],  dstVec[8],  dstVec[9],  dstVec[10],
            dstVec[11], dstVec[12], dstVec[13], dstVec[14], dstVec[15],
        }),
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
        false, false,
        LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                  LLVM::AsmDialect::AD_ATT));

    auto resVec =
        llvm::to_vector<16>(llvm::map_range(llvm::seq<int>(0, 16), [&](int i) {
          return rewriter.create<LLVM::ExtractValueOp>(
              loc, int32x4VType, packedResult.res(),
              rewriter.getI64ArrayAttr({i}));
        }));

    auto int32x8x8xVType = VectorType::get({8, 8}, I32Type);

    Value result;
    result = rewriter.create<arith::ConstantOp>(
        loc, int32x8x8xVType, DenseIntElementsAttr::get(int32x8x8xVType, 0));
    for (int i = 0; i < 16; ++i) {
      auto int32x1x4VType = VectorType::get({1, 4}, I32Type);
      Value resVec2D =
          rewriter.create<vector::ShapeCastOp>(loc, int32x1x4VType, resVec[i]);
      SmallVector<int64_t> offsets{i / 2, (i % 2) * 4};
      SmallVector<int64_t> strides{1, 1};
      result = rewriter.create<vector::InsertStridedSliceOp>(
          loc, resVec2D, result, offsets, strides);
    }
    rewriter.replaceOp(contractionOp, {result});
    return success();
  }
};

}  // namespace

namespace {
struct VectorToAArch64InlineAsmPass
    : public VectorToAArch64InlineAsmBase<VectorToAArch64InlineAsmPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect, LLVM::LLVMDialect>();
  }
  void runOnOperation() override;
};
}  // namespace

void populateVectorContractToAArch64InlineAsm(
    OwningRewritePatternList &patterns, MLIRContext *context) {
  patterns.insert<ConvertVectorContract8x4x8_i8i8i32_ToAArch64InlineAsmPattern>(
      context);
}

void VectorToAArch64InlineAsmPass::runOnOperation() {
  MLIRContext *context = &getContext();
  OwningRewritePatternList patterns(context);
  populateVectorContractToAArch64InlineAsm(patterns, context);

  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<FuncOp>>
createVectorToAArch64InlineAssemblyPass() {
  return std::make_unique<VectorToAArch64InlineAsmPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
