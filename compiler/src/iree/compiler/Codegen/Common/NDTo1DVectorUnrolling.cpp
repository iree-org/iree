// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===- NDTo1DVectorUnrolling.cpp - Description ----------------------------===//
//
// TODO: Write docs.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "iree-nd-to-1d-vector-unrolling"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_NDTO1DVECTORUNROLLINGPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

struct UnrollVectorTypeConverter final : public TypeConverter {
  UnrollVectorTypeConverter() {
    // Allow all other types.
    addConversion([](Type type) -> std::optional<Type> { return type; });

    // Convert n-D Vector to 1-D Vector.
    addConversion([](VectorType type, SmallVectorImpl<Type> &types)
                      -> std::optional<LogicalResult> {
      if (type.getRank() <= 1) {
        types.push_back(type);
        return success();
      }
      int64_t innerDim = type.getShape().back();
      int64_t num1DVectors = type.getNumElements() / innerDim;
      Type innerDimVector = VectorType::get({innerDim}, type.getElementType());
      for (int64_t i = 0; i < num1DVectors; i++) {
        types.push_back(innerDimVector);
      }
      return success();
    });

    addSourceMaterialization([](OpBuilder &builder, VectorType targetType,
                                ValueRange inputs, Location loc) -> Value {
      // Create a poison vector of the target type and insert the input vectors.
      Value result = builder.create<ub::PoisonOp>(loc, targetType);
      SmallVector<int64_t> iteratorSpace(targetType.getShape().drop_back());
      for (auto [input, idx] : llvm::zip_equal(
               inputs, StaticTileOffsetRange(
                           iteratorSpace,
                           SmallVector<int64_t>(iteratorSpace.size(), 1)))) {
        result = builder.create<vector::InsertOp>(loc, input, result, idx);
      }
      return result;
    });

    addTargetMaterialization([](OpBuilder &builder, TypeRange targetTypes,
                                ValueRange sources, Location loc,
                                Type originalType) -> SmallVector<Value> {
      Value source = sources[0];
      SmallVector<Value> results;
      VectorType originalTypeVec = cast<VectorType>(originalType);
      SmallVector<int64_t> iteratorSpace(
          originalTypeVec.getShape().drop_back());
      for (SmallVector<int64_t> idx : StaticTileOffsetRange(
               iteratorSpace, SmallVector<int64_t>(iteratorSpace.size(), 1))) {
        results.push_back(builder.create<vector::ExtractOp>(loc, source, idx));
      }
      return results;
    });
  }
};

struct UnrollElementwiseOps final
    : public OpTraitConversionPattern<OpTrait::Elementwise> {
  using Base::Base;

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<ValueRange> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!OpTrait::hasElementwiseMappableTraits(op) ||
        op->getNumResults() != 1) {
      return failure();
    }
    // Check if the result type is an n-D vector.
    VectorType dstVecTy = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!dstVecTy || dstVecTy.getRank() <= 1) {
      return failure();
    }

    SmallVector<Type> convertedTypes;
    if (failed(getTypeConverter()->convertType(op->getResult(0).getType(),
                                               convertedTypes))) {
      return failure();
    }

    // 1:N conversion passes a list of operands, with each operand being a
    // ValueRange of the N new values, we invert the list to be a list of
    // N new lists, each being a list of operands for the unrolled op.
    assert(llvm::all_of(operands,
                        [&](ValueRange r) {
                          return r.size() == convertedTypes.size();
                        }) &&
           "expected all operands to have the same number of values as the "
           "number of converted types.");
    SmallVector<SmallVector<Value>> opOperands;
    for (auto i : llvm::seq<int64_t>(convertedTypes.size())) {
      SmallVector<Value> newOperands;
      for (ValueRange operandList : operands) {
        newOperands.push_back(operandList[i]);
      }
      opOperands.push_back(newOperands);
    }

    // Iterate over all input value ranges and clone the op.
    SmallVector<Value> results;
    for (auto [inputs, convertedTy] :
         llvm::zip_equal(opOperands, convertedTypes)) {
      Operation *clonedOp = clone(rewriter, op, convertedTy, inputs);
      clonedOp->dump();
      results.push_back(clonedOp->getResult(0));
    }
    rewriter.replaceOpWithMultiple(op, {results});
    return success();
  }
};

struct NDTo1DVectorUnrollingPass final
    : impl::NDTo1DVectorUnrollingPassBase<NDTo1DVectorUnrollingPass> {

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    UnrollVectorTypeConverter typeConverter;
    ConversionTarget target(*ctx);

    scf::populateSCFStructuralTypeConversionTarget(typeConverter, target);
    patterns.add<UnrollElementwiseOps>(typeConverter, ctx);
    // vector.transfer_read/vector.transfer_write lowerings.
    vector::populateVectorMaskMaterializationPatterns(
        patterns, /*force32BitVectorIndices=*/true);
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorTransferLoweringPatterns(patterns,
                                                   /*maxTransferRank=*/1);
    auto vectorTransferToSCFOptions =
        VectorTransferToSCFOptions().enableFullUnroll();
    populateVectorToSCFConversionPatterns(patterns, vectorTransferToSCFOptions);
    // vector.shape_cast.
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    // vector.multi_reduction.
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerReduction);

    // Mark scf operations with N-D vectors as illegal.
    scf::populateSCFStructuralTypeConversions(typeConverter, patterns);
    // Mark vector.transfer_read/vector.transfer_write illegal.
    target.addIllegalOp<vector::TransferReadOp>();
    target.addIllegalOp<vector::TransferWriteOp>();
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) -> std::optional<bool> {
          if (OpTrait::hasElementwiseMappableTraits(op) &&
              op->getNumResults() == 1) {
            // Elementwise ops are legal if all their result types are legal.
            return typeConverter.isLegal(op->getResultTypes());
          }
          return true;
        });
    target.addIllegalOp<vector::ShapeCastOp>();
    target.addIllegalOp<vector::MultiDimReductionOp>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  };
};

} // namespace

} // namespace mlir::iree_compiler
