// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Conversion/PassDetail.h"
#include "iree/compiler/Conversion/Passes.h"
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

/// Converts 4x4x4 vector contraction with matmul(A_, B) semantics to AArch64
/// inline assembly using aarch64 4-sdot instructions. Each sdot instruction
/// performas a single matrix-vector product and to compute matmul(A, B) with
/// matrix-vector products B is transposed.
struct ConvertVectorContract4x4x4_i8i8i32_ToAArch64InlineAsmPattern
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
    if (lhsShape[0] != 4 || lhsShape[1] != 4 || rhsShape[0] != 4 ||
        rhsShape[1] != 4)
      return failure();

    Value inLhs = contractionOp.lhs();
    Value inRhs = contractionOp.rhs();

    auto I8Type = rewriter.getIntegerType(8);
    auto I32Type = rewriter.getIntegerType(32);

    if (accType.getElementType() != I32Type) {
      return failure();
    }

    auto getI8Value = [&](Value v) -> Value {
      if (auto parentOp = v.getDefiningOp<SignExtendIOp>()) {
        if (parentOp.value().getType().cast<VectorType>().getElementType() !=
            I8Type) {
          return nullptr;
        } else {
          return parentOp.value();
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

    SmallVector<Value> dstVec;
    for (int i = 0; i < 4; ++i) {
      dstVec.push_back(
          rewriter.create<vector::ExtractOp>(loc, contractionOp.acc(), i));
    }

    auto flattnedVectorType = VectorType::get({16}, I8Type);

    auto lhs =
        rewriter.create<vector::ShapeCastOp>(loc, flattnedVectorType, inLhs);

    auto inRhsTransposed = rewriter.create<vector::TransposeOp>(
        loc, inRhs, ArrayRef<int64_t>({1, 0}));

    auto rhs = rewriter.create<vector::ShapeCastOp>(loc, flattnedVectorType,
                                                    inRhsTransposed);

    auto int32x4VType = VectorType::get({4}, I32Type);

    auto returnType = LLVM::LLVMStructType::getLiteral(
        rewriter.getContext(),
        {int32x4VType, int32x4VType, int32x4VType, int32x4VType});

    /// TODO(ataei): We an have a formatter like the c++ inline asm to
    /// ssa-values to string names which will make the inline-assembly
    /// statements more redable e.g :
    /// sdot ${dstVec_0}.4s, ${lhs}.16b,${rhs}.4b[0]
    auto packedResult = rewriter.create<LLVM::InlineAsmOp>(
        loc, returnType,
        ArrayRef<Value>({lhs, rhs, dstVec[0], dstVec[1], dstVec[2], dstVec[3]}),
        R"ASM(
            sdot $0.4s, $4.16b, $5.4b[0]
            sdot $1.4s, $4.16b, $5.4b[1]
            sdot $2.4s, $4.16b, $5.4b[2]
            sdot $3.4s, $4.16b, $5.4b[3]
          )ASM",
        "=w,=w,=w,=w,w,w,0,1,2,3", false, false,
        LLVM::AsmDialectAttr::get(rewriter.getContext(),
                                  LLVM::AsmDialect::AD_ATT));

    auto resVec =
        llvm::to_vector<4>(llvm::map_range(llvm::seq<int>(0, 4), [&](int i) {
          return rewriter.create<LLVM::ExtractValueOp>(
              loc, int32x4VType, packedResult.res(),
              rewriter.getI64ArrayAttr({i}));
        }));

    auto int32x4x4xVType = VectorType::get({4, 4}, I32Type);

    Value result;
    result = rewriter.create<ConstantOp>(
        loc, int32x4x4xVType, DenseIntElementsAttr::get(int32x4x4xVType, 0));
    for (int i = 0; i < 4; ++i) {
      result = rewriter.create<vector::InsertOp>(loc, resVec[i], result,
                                                 ArrayRef<int64_t>({i}));
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
  patterns.insert<ConvertVectorContract4x4x4_i8i8i32_ToAArch64InlineAsmPattern>(
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
