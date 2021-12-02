// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// Op Conversion Patterns
//===----------------------------------------------------------------------===//

/// Converts vector transfer ops to SPIR-V cooperative matrix load/store ops.
struct ConvertVectorTransferOp final
    : public OpInterfaceConversionPattern<VectorTransferOpInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult matchAndRewrite(
      VectorTransferOpInterface op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Don't support masked load/store.
    if (op.getMaskType()) return failure();

    // Expect inbound access.
    if (op.in_bounds()) {
      auto inBounds = op.in_bounds()->getAsValueRange<BoolAttr>();
      if (!llvm::all_of(inBounds, [](bool v) { return v; })) return failure();
    }

    // Expect transfers over memrefs.
    auto memrefType = op.getShapedType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();

    // Expect 2-D vectors.
    auto vectorType = op.getVectorType();
    if (vectorType.getRank() != 2) return failure();

    // TODO: Use coloumn major with transposed transfer ops.
    if (!op.permutation_map().isMinorIdentity()) return failure();

    int64_t offset = 0;
    SmallVector<int64_t, 2> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return failure();
    auto stride = strides[0];
    if (ShapedType::isDynamicStrideOrOffset(stride)) return failure();

    auto loc = op.getLoc();

    auto i32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, i32Type, IntegerAttr::get(i32Type, stride));
    auto coloumnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));

    Type matType = typeConverter->convertType(vectorType);

    if (auto readOp = dyn_cast<vector::TransferReadOp>(*op)) {
      vector::TransferReadOp::Adaptor adaptor(operands,
                                              op->getAttrDictionary());
      Value bufferPtr = spirv::getElementPtr(
          *getTypeConverter<SPIRVTypeConverter>(), memrefType, adaptor.source(),
          adaptor.indices(), loc, rewriter);
      rewriter.replaceOpWithNewOp<spirv::CooperativeMatrixLoadNVOp>(
          op, matType, bufferPtr, strideValue, coloumnMajor,
          spirv::MemoryAccessAttr());
      return success();
    }

    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(*op)) {
      vector::TransferWriteOp::Adaptor adaptor(operands,
                                               op->getAttrDictionary());
      Value bufferPtr = spirv::getElementPtr(
          *getTypeConverter<SPIRVTypeConverter>(), memrefType, adaptor.source(),
          adaptor.indices(), loc, rewriter);
      rewriter.create<spirv::CooperativeMatrixStoreNVOp>(
          loc, bufferPtr, adaptor.vector(), strideValue, coloumnMajor,
          spirv::MemoryAccessAttr());
      rewriter.eraseOp(op);
      return success();
    }

    return failure();
  }
};

/// Converts vector.contract ops to SPIR-V cooperative matrix multiple-add ops.
struct ConvertVectorContractOp final
    : public OpConversionPattern<vector::ContractionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, OpAdaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!llvm::empty(contractOp.masks())) return failure();

    // Check that this is a matmul operation.
    auto iterators = contractOp.iterator_types().getValue();
    if (iterators.size() != 3 || !isParallelIterator(iterators[0]) ||
        !isParallelIterator(iterators[1]) || !isReductionIterator(iterators[2]))
      return failure();
    if (contractOp.kind() != vector::CombiningKind::ADD) return failure();

    // Column major matmuls should have been lowered to transpose + contract
    // by this point. Transpose can be handled by load/store operations.
    if (!isRowMajorMatmul(contractOp.indexing_maps())) return failure();

    rewriter.replaceOpWithNewOp<spirv::CooperativeMatrixMulAddNVOp>(
        contractOp, operands.acc().getType(), operands.lhs(), operands.rhs(),
        operands.acc());
    return success();
  }
};

/// Converts splat vector constants to constant SPIR-V cooperative matrix ops.
struct ConvertConstantMatrix final
    : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      arith::ConstantOp op, OpAdaptor operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only convert 2-D vector constants.
    auto vectorType = op.getType().dyn_cast<VectorType>();
    if (!vectorType || vectorType.getRank() != 2) return failure();

    // Only convert splat integer/float vectors.
    auto values = op.getValue().dyn_cast<DenseIntOrFPElementsAttr>();
    if (!values || !values.isSplat()) return failure();
    Attribute value = values.getSplatValue<Attribute>();

    auto elementType = values.getType().getElementType();
    Value splatValue = rewriter.create<spirv::ConstantOp>(
        op.getLoc(), typeConverter->convertType(elementType), value);

    auto matType = typeConverter->convertType(vectorType);
    rewriter.replaceOpWithNewOp<spirv::CompositeConstructOp>(op, matType,
                                                             splatValue);
    return success();
  }
};

/// Converts elementwise ops to SPIR-V cooperative matrix elementwise ops.
template <typename SrcOpType, typename DstOpType>
struct ConvertElementwiseOp final : public OpConversionPattern<SrcOpType> {
  using OpConversionPattern<SrcOpType>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      SrcOpType op, typename SrcOpType::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // All operands should be of cooperative matrix types.
    for (Value operand : adaptor.getOperands()) {
      if (!operand.getType().isa<spirv::CooperativeMatrixNVType>())
        return failure();
    }

    // Only support ops with one result.
    if (op->getNumResults() != 1) return failure();

    auto matType = this->typeConverter->convertType(op.getType());
    rewriter.replaceOpWithNewOp<DstOpType>(op, matType, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Main Pass
//===----------------------------------------------------------------------===//

struct SPIRVVectorToCooperativeOpsPass final
    : public SPIRVVectorToCooperativeOpsBase<SPIRVVectorToCooperativeOpsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    FuncOp funcOp = getOperation();

    spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(funcOp);
    SPIRVTypeConverter typeConverter(targetAttr);

    // Inject conversion rules for 2-D vector types to cooperative matrix types.
    //
    // Note that we don't perform legality check here; we just directly convert.
    // Legality check is expected to be done when deciding the whole pass
    // pipeline is feasible and also in SPIR-V ConversionTarget.
    typeConverter.addConversion(
        [&typeConverter](VectorType type) -> Optional<Type> {
          if (type.getRank() != 2) return llvm::None;

          Type elementType = typeConverter.convertType(type.getElementType());
          return spirv::CooperativeMatrixNVType::get(
              elementType, spirv::Scope::Subgroup, type.getDimSize(0),
              type.getDimSize(1));
        });

    // Inject another conversion rule for MemRef types.
    //
    // This is for consistency purpose: we will run FlattenMemRefSubspanPass
    // later. That pass flattens all MemRefs into 1-D unknown-sized ones before
    // invoking upstream SPIR-V type converter. So in the end all MemRefs will
    // be converted into SPIR-V runtime arrays. But here if we don't inject the
    // following rule, we'll convert MemRefs into constant-sized arrays. That
    // would cause consistency issues. It's a bit unfortunate to have this; it's
    // a result of performing cooperative matrix conversions earlier (it needs
    // to be done before FlattenMemRefSubspanPass because we need 2-D MemRefs)
    // and conversions spreading across upstream and IREE repos..
    typeConverter.addConversion(
        [&typeConverter](MemRefType type) -> Optional<Type> {
          if (!type.hasStaticShape()) return llvm::None;
          // In IREE all MemRefs are originated from subspan ops, which should
          // have identity layout.
          if (!type.getLayout().isIdentity()) return llvm::None;
          auto flattenedType =
              MemRefType::get(ShapedType::kDynamicSize, type.getElementType(),
                              AffineMap(), type.getMemorySpace());
          return typeConverter.convertType(flattenedType);
        });

    // Add unrealized conversion cast ops to bridge type conversions: we are
    // only converting the cooperative matrix subset; the rest needs to be done
    // at a later stage.
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    RewritePatternSet patterns(context);
    patterns.add<
        ConvertConstantMatrix, ConvertVectorContractOp, ConvertVectorTransferOp,
        // See SPV_NV_cooperative_matrix for supported element wise ops.
        ConvertElementwiseOp<arith::AddFOp, spirv::FAddOp>,
        ConvertElementwiseOp<arith::AddIOp, spirv::IAddOp>,
        ConvertElementwiseOp<arith::SubFOp, spirv::FSubOp>,
        ConvertElementwiseOp<arith::SubIOp, spirv::ISubOp>,
        ConvertElementwiseOp<arith::DivFOp, spirv::FDivOp>,
        ConvertElementwiseOp<arith::DivSIOp, spirv::SDivOp>,
        ConvertElementwiseOp<arith::DivUIOp, spirv::UDivOp>,
        ConvertElementwiseOp<arith::NegFOp, spirv::FNegateOp>>(typeConverter,
                                                               context);

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    target->addLegalOp<UnrealizedConversionCastOp>();
    target->addIllegalDialect<vector::VectorDialect>();

    if (failed(applyPartialConversion(funcOp, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createSPIRVVectorToCooperativeOpsPass() {
  return std::make_unique<SPIRVVectorToCooperativeOpsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
