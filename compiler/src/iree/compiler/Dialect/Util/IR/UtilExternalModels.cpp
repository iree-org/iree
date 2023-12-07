// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilExternalModels.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir::iree_compiler::IREE::Util {

namespace {

// Since all details of the interface are provided via default implementations,
// we can just have one templated external model to apply per op, vs one
// explicit model per op.
struct GenericNumericCastExternalModel {
  template <typename OpTy>
  struct ExternalModel
      : public NumericCastOpInterface::ExternalModel<ExternalModel<OpTy>,
                                                     OpTy> {};

  template <typename OpTy>
  static void add(MLIRContext *ctx) {
    OpTy::template attachInterface<ExternalModel<OpTy>>(*ctx);
  }

  template <typename OpTy1, typename OpTy2, typename... More>
  static void add(MLIRContext *ctx) {
    add<OpTy1>(ctx);
    add<OpTy2, More...>(ctx);
  }
};

struct InsertSliceOpTiedOpInterface
    : public TiedOpInterface::ExternalModel<InsertSliceOpTiedOpInterface,
                                            tensor::InsertSliceOp> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto insertSliceOp = cast<tensor::InsertSliceOp>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        insertSliceOp.getDest());
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    return {1}; // dest
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    return {1}; // dest
  }
};

template <typename OpTy>
struct LinalgOpTiedOpInterface
    : public TiedOpInterface::ExternalModel<LinalgOpTiedOpInterface<OpTy>,
                                            OpTy> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(
        linalgOp.getDpsInits()[resultIndex]);
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<OpTy>(op);
    return {linalgOp.getDpsInitsMutable()[resultIndex].getOperandNumber()};
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    SmallVector<int64_t> result;
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      result.push_back(*getTiedResultOperandIndex(op, i));
    return result;
  }
};

/// Helper structure that iterates over all LinalgOps in `OpTys` and registers
/// the `TiedOpInterface` with each of them.
template <typename... Ops>
struct LinalgOpTiedOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (void)std::initializer_list<int>{
        0, (Ops::template attachInterface<LinalgOpTiedOpInterface<Ops>>(*ctx),
            0)...};
  }
};

struct GlobalOpInterfaceExternalModel
    : public GlobalOpInterface::ExternalModel<GlobalOpInterfaceExternalModel,
                                              ml_program::GlobalOp> {
  static void add(MLIRContext *ctx) {
    ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(*ctx);
  }

  Attribute getGlobalInitialValue(::mlir::Operation *op) const {
    return cast<ml_program::GlobalOp>(op).getValueAttr();
  }
  void setGlobalInitialValue(::mlir::Operation *op, Attribute value) const {
    if (value) {
      cast<ml_program::GlobalOp>(op).setValueAttr(value);
    } else {
      cast<ml_program::GlobalOp>(op).removeValueAttr();
    }
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                  ml_program::MLProgramDialect, tensor::TensorDialect>();

  registry.addExtension(+[](MLIRContext *ctx,
                            ml_program::MLProgramDialect *dialect) {
    ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, arith::ArithDialect *dialect) {
    GenericNumericCastExternalModel::add<
        arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
        arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp, arith::TruncFOp,
        arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp>(ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::InsertSliceOp::attachInterface<InsertSliceOpTiedOpInterface>(*ctx);
  });

  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    // Register all Linalg structured ops. `LinalgOp` is an interface and it is
    // not possible to attach an external interface to an existing interface.
    // Therefore, attach the `TiedOpInterface` to all ops one-by-one.
    LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
  });

  // TODO(matthias-springer): Use a helper instead of listing all ops. This is
  // tricky because LinalgExtOps.td includes YieldOp.
  registry.addExtension(+[](MLIRContext *ctx,
                            LinalgExt::IREELinalgExtDialect *dialect) {
    LinalgExt::ScatterOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::ScatterOp>>(*ctx);
    LinalgExt::SortOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::SortOp>>(*ctx);
    LinalgExt::FftOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::FftOp>>(*ctx);
    LinalgExt::ScanOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::ScanOp>>(*ctx);
    LinalgExt::ReverseOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::ReverseOp>>(*ctx);
    LinalgExt::TopkOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::TopkOp>>(*ctx);
    LinalgExt::WinogradInputTransformOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::WinogradInputTransformOp>>(*ctx);
    LinalgExt::WinogradOutputTransformOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::WinogradOutputTransformOp>>(*ctx);
    LinalgExt::AttentionOp::attachInterface<
        LinalgOpTiedOpInterface<LinalgExt::AttentionOp>>(*ctx);
  });
}

} // namespace mlir::iree_compiler::IREE::Util
