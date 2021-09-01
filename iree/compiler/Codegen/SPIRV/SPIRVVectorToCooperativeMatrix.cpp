// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace mlir {
namespace iree_compiler {

namespace {

bool isLegalVectorContract(vector::ContractionOp contract) {
  if (llvm::size(contract.masks()) != 0) return false;
  VectorType lhsType = contract.lhs().getType().cast<VectorType>();
  VectorType rhsType = contract.rhs().getType().cast<VectorType>();
  VectorType accType = contract.acc().getType().cast<VectorType>();

  std::tuple<int, int, int> dim(lhsType.getDimSize(0), rhsType.getDimSize(1),
                                lhsType.getDimSize(1));
  // Check if the matrix type can be supported as a cooperative matrix.
  // Currently we have hardcoded checks for what Turing hardware supports.
  // TODO(thomasraoux): Add device information to be able to query what the
  // device supports.
  if (lhsType.getElementType().isInteger(8) &&
      rhsType.getElementType().isInteger(8) &&
      accType.getElementType().isInteger(32) &&
      (dim == std::make_tuple(8, 8, 32) || dim == std::make_tuple(16, 16, 32) ||
       dim == std::make_tuple(16, 8, 32)))
    return true;

  if (lhsType.getElementType().isF16() && rhsType.getElementType().isF16() &&
      (accType.getElementType().isF16() || accType.getElementType().isF32()) &&
      (dim == std::make_tuple(8, 8, 16) || dim == std::make_tuple(16, 16, 16) ||
       dim == std::make_tuple(16, 8, 16)))
    return true;

  return false;
}

bool supportsCooperativeMatrix(Operation *op) {
  if (isa<vector::TransferReadOp, vector::TransferWriteOp, scf::ForOp,
          scf::YieldOp>(op))
    return true;
  if (isa<vector::ContractionOp>(op) &&
      isLegalVectorContract(cast<vector::ContractionOp>(op)))
    return true;
  // We only support minimal set of operations right now. We can trivially
  // extend to ALU instructions supporting Cooperative Matrix in SPIR-V spec.
  // We also need to extend to control flow operations, Alloca, etc...
  // TODO(thomasraoux): extend support to more complex chain of instructions.
  return false;
}

class CooperativeMatrixAnalysis {
 public:
  explicit CooperativeMatrixAnalysis(mlir::Operation *op) {
    spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(op);
    spirv::TargetEnv targetEnv(targetEnvAttr);
    if (!targetEnv.allows(spirv::Capability::CooperativeMatrixNV) ||
        !targetEnv.allows(spirv::Extension::SPV_NV_cooperative_matrix))
      return;

    op->walk([&](Operation *op) {
      auto contract = dyn_cast<vector::ContractionOp>(op);
      if (contract == nullptr) return;
      auto hasVectorDest = [](Operation *op) {
        if (isa<ConstantOp, memref::AllocOp>(op)) return false;
        for (auto resultType : op->getResultTypes()) {
          if (resultType.isa<VectorType>()) return true;
        }
        if (op->getNumResults() == 0) return true;
        return false;
      };
      auto dependentOps = getSlice(op, hasVectorDest, hasVectorDest);
      for (auto *dependeOp : dependentOps) {
        // If any instruction cannot use cooperative matrix drop the whole
        // chaine. In the future we can introduce "bitcast" type of conversion
        // to allow the same value to be used as both cooperative matrix as well
        // as an array.
        if (!supportsCooperativeMatrix(dependeOp)) return;
      }
      // All the dependent instruction can use cooperative matrix type. We can
      // mark the whole chain of operations as using cooperative matrix.
      usesCooperativeMatrix.insert(op);
      usesCooperativeMatrix.insert(dependentOps.begin(), dependentOps.end());
    });
  }

  // Returns true if the operation should be lowered using operations on
  // cooperative matrix type.
  bool usesCooperativeMatrixType(mlir::Operation *op) const {
    return usesCooperativeMatrix.count(op);
  }

 private:
  llvm::DenseSet<mlir::Operation *> usesCooperativeMatrix;
};

/// Convert subgroup level vector transfer to SPIR-V cooperative
/// matrix load/store if those are supported.
/// TODO(thomasraoux): Move to MLIR core once this is stable.
template <typename OpTy>
class TransferToCoopMatLoadStore final : public OpConversionPattern<OpTy> {
 public:
  TransferToCoopMatLoadStore(
      MLIRContext *context, SPIRVTypeConverter &converter,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : OpConversionPattern<OpTy>(converter, context),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

  LogicalResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!cooperativeMatrixAnalysis.usesCooperativeMatrixType(op))
      return failure();

    if (op.mask()) return failure();

    auto memrefType = op.getShapedType().template dyn_cast<MemRefType>();
    if (!memrefType) return failure();

    auto vecType = op.getVectorType();
    if (vecType.getRank() != 2) return failure();

    // TODO(thomasraoux): use coloumn major operand when TransfertRead +
    // TransposeOp.
    if (!op.permutation_map().isMinorIdentity()) return failure();
    if (op.in_bounds() &&
        llvm::any_of(op.in_bounds()->template cast<ArrayAttr>(),
                     [](mlir::Attribute dimInBounds) {
                       return !dimInBounds.cast<BoolAttr>().getValue();
                     }))
      return failure();

    int64_t offset = 0;
    SmallVector<int64_t, 2> strides;
    if (failed(getStridesAndOffset(memrefType, strides, offset)))
      return failure();
    auto stride = strides[0];
    if (BaseMemRefType::isDynamicStrideOrOffset(stride)) return failure();

    auto loc = op.getLoc();
    typename OpTy::Adaptor adaptor(operands, op->getAttrDictionary());
    auto matType = spirv::CooperativeMatrixNVType::get(
        vecType.getElementType(), spirv::Scope::Subgroup, vecType.getDimSize(0),
        vecType.getDimSize(1));
    Value ptr = spirv::getElementPtr(
        *this->template getTypeConverter<SPIRVTypeConverter>(), memrefType,
        adaptor.source(), adaptor.indices(), loc, rewriter);
    auto int32Type = rewriter.getI32Type();
    auto strideValue = rewriter.create<spirv::ConstantOp>(
        loc, int32Type, IntegerAttr::get(int32Type, stride));
    auto coloumnMajor = rewriter.create<spirv::ConstantOp>(
        loc, rewriter.getI1Type(), rewriter.getBoolAttr(false));
    replaceTransferOp(op, adaptor, matType, ptr, strideValue, coloumnMajor,
                      rewriter);
    return success();
  }

 private:
  /// Generates the right load/store instruction and replaces the transfer op.
  void replaceTransferOp(OpTy originalOp, typename OpTy::Adaptor newInputs,
                         Type matType, Value bufferPtr, Value strideValue,
                         Value coloumnMajor,
                         ConversionPatternRewriter &rewriter) const;

  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

template <>
void TransferToCoopMatLoadStore<vector::TransferReadOp>::replaceTransferOp(
    vector::TransferReadOp originalOp,
    vector::TransferReadOp::Adaptor newInputs, Type matType, Value bufferPtr,
    Value strideValue, Value coloumnMajor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.replaceOpWithNewOp<spirv::CooperativeMatrixLoadNVOp>(
      originalOp, matType, bufferPtr, strideValue, coloumnMajor,
      spirv::MemoryAccessAttr());
}

template <>
void TransferToCoopMatLoadStore<vector::TransferWriteOp>::replaceTransferOp(
    vector::TransferWriteOp originalOp,
    vector::TransferWriteOp::Adaptor newInputs, Type matType, Value bufferPtr,
    Value strideValue, Value coloumnMajor,
    ConversionPatternRewriter &rewriter) const {
  rewriter.create<spirv::CooperativeMatrixStoreNVOp>(
      originalOp.getLoc(), bufferPtr, newInputs.vector(), strideValue,
      coloumnMajor, spirv::MemoryAccessAttr());
  rewriter.eraseOp(originalOp);
}

/// Converts subgroup level vector contract to SPIR-V cooperative
/// matrix matmuladd.
class VectorContractToCoopMatmul final
    : public OpConversionPattern<vector::ContractionOp> {
 public:
  VectorContractToCoopMatmul(
      MLIRContext *context, SPIRVTypeConverter &converter,
      const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis)
      : OpConversionPattern(converter, context),
        cooperativeMatrixAnalysis(cooperativeMatrixAnalysis) {}

  LogicalResult matchAndRewrite(
      vector::ContractionOp contractOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (!cooperativeMatrixAnalysis.usesCooperativeMatrixType(contractOp))
      return failure();

    if (llvm::size(contractOp.masks()) != 0) return failure();
    // Check that this is a matmul operation.
    auto iteratorTypes = contractOp.iterator_types().getValue();
    if (!isParallelIterator(iteratorTypes[0]) ||
        !isParallelIterator(iteratorTypes[1]) ||
        !isReductionIterator(iteratorTypes[2]))
      return failure();
    // Coloumn major matmul should have been lowered to Transpose+contract
    // by this point. Transpose can be handled by load/stoore operations.
    if (!isRowMajorMatmul(contractOp.indexing_maps())) return failure();

    vector::ContractionOp::Adaptor adaptor(operands);
    auto loadA = adaptor.lhs();
    auto loadB = adaptor.rhs();
    auto loadC = adaptor.acc();
    rewriter.replaceOpWithNewOp<spirv::CooperativeMatrixMulAddNVOp>(
        contractOp, loadC.getType(), loadA, loadB, loadC);
    return success();
  }

 private:
  const CooperativeMatrixAnalysis &cooperativeMatrixAnalysis;
};

struct SPIRVVectorToCooperativeMatrixPass final
    : public SPIRVVectorToCooperativeMatrixBase<
          SPIRVVectorToCooperativeMatrixPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(funcOp);
    SPIRVTypeConverter typeConverter(targetAttr);

    typeConverter.addConversion(
        [&typeConverter](MemRefType type) -> Optional<Type> {
          if (!type.hasStaticShape()) return llvm::None;
          auto flattenType =
              MemRefType::get(ShapedType::kDynamicSize, type.getElementType(),
                              type.getAffineMaps(), type.getMemorySpace());
          return typeConverter.convertType(flattenType);
        });

    typeConverter.addConversion([](VectorType type) -> Optional<Type> {
      if (type.getRank() != 2) return llvm::None;
      return spirv::CooperativeMatrixNVType::get(
          type.getElementType(), spirv::Scope::Subgroup, type.getDimSize(0),
          type.getDimSize(1));
    });

    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    typeConverter.addSourceMaterialization(addUnrealizedCast);
    typeConverter.addTargetMaterialization(addUnrealizedCast);

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto &analysis = getAnalysis<CooperativeMatrixAnalysis>();
    patterns.add<TransferToCoopMatLoadStore<vector::TransferReadOp>,
                 TransferToCoopMatLoadStore<vector::TransferWriteOp>,
                 VectorContractToCoopMatmul>(context, typeConverter, analysis);

    std::unique_ptr<ConversionTarget> target =
        SPIRVConversionTarget::get(targetAttr);
    target->addLegalOp<UnrealizedConversionCastOp>();

    if (failed(applyPartialConversion(funcOp, *target, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
createSPIRVVectorToCooperativeMatrixPass() {
  return std::make_unique<SPIRVVectorToCooperativeMatrixPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
