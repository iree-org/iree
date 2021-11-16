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

/// Converts 8x4x8 vector contraction with matmul(A_, B) semantics to AArch64
/// inline assembly using aarch64 4-sdot instructions. Each sdot instruction
/// performas a single matrix-vector product and to compute matmul(A, B) with
/// matrix-vector products B is transposed.
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

    Value inLhs = contractionOp.lhs();
    Value inRhs = contractionOp.rhs();

    auto I8Type = rewriter.getIntegerType(8);
    auto I32Type = rewriter.getIntegerType(32);

    if (accType.getElementType() != I32Type) {
      return failure();
    }

    auto getI8Value = [&](Value v) -> Value {
      if (auto parentOp = v.getDefiningOp<arith::ExtSIOp>()) {
        if (parentOp.in().getType().cast<VectorType>().getElementType() !=
            I8Type) {
          return nullptr;
        } else {
          return parentOp.in();
        }
      }
      return nullptr;
    };
    if (lhsType.getElementType() != I8Type) {
      inLhs = getI8Value(inLhs);
    }
    if (rhsType.getElementType() != I8Type) {
      inRhs = getI8Value(inRhs);
    }

    if (!inLhs || !inRhs) return failure();

    auto loc = contractionOp.getLoc();

    auto int32x4VType = VectorType::get({4}, I32Type);

    SmallVector<Value> dstVec;
    for (int i = 0; i < 16; ++i) {
      SmallVector<int64_t> offsets{i / 2, (i % 2) * 4};
      SmallVector<int64_t> sizes{1, 4};
      SmallVector<int64_t> strides{1, 1};

      dstVec.push_back(rewriter.create<vector::ShapeCastOp>(
          loc, int32x4VType,
          rewriter.create<vector::ExtractStridedSliceOp>(
              loc, contractionOp.acc(), offsets, sizes, strides)));
    }

    auto flatVectorType = VectorType::get({32}, I8Type);

    auto lhs = rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, inLhs);

    auto rhs = rewriter.create<vector::ShapeCastOp>(loc, flatVectorType, inRhs);

    auto extract = [&](Value input, int64_t offset) {
      SmallVector<int64_t> offsets{offset};
      SmallVector<int64_t> sizes{16};
      SmallVector<int64_t> strides{1};

      return rewriter.create<vector::ExtractStridedSliceOp>(loc, input, offsets,
                                                            sizes, strides);
    };
    auto lhs0 = extract(lhs, 0);
    auto lhs1 = extract(lhs, 16);
    auto rhs0 = extract(rhs, 0);
    auto rhs1 = extract(rhs, 16);

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
