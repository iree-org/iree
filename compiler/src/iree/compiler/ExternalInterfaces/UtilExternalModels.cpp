// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/UtilExternalModels.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUDialect.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingDialect.h"
#include "iree/compiler/Dialect/Encoding/IR/EncodingOps.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MLProgram/IR/MLProgram.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

namespace mlir::iree_compiler {

namespace {

//===----------------------------------------------------------------------===//
// InferIntDivisibilityOpInterface
//===----------------------------------------------------------------------===//

struct ArithConstantInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithConstantInferIntDivisibilityOpInterface, arith::ConstantOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto constOp = cast<arith::ConstantOp>(op);
    auto constAttr = llvm::dyn_cast_or_null<IntegerAttr>(constOp.getValue());
    if (constAttr) {
      const APInt &value = constAttr.getValue();
      uint64_t udiv = value.getZExtValue();
      uint64_t sdiv = std::abs(value.getSExtValue());
      setResultDivs(constOp.getResult(),
                    IREE::Util::ConstantIntDivisibility(udiv, sdiv));
    }
  }
};

struct ArithMulIInferIntDivisibilityOpInterface
    : public IREE::Util::InferIntDivisibilityOpInterface::ExternalModel<
          ArithMulIInferIntDivisibilityOpInterface, arith::MulIOp> {

  void inferResultDivisibility(
      Operation *op, ArrayRef<IREE::Util::IntegerDivisibility> argDivs,
      IREE::Util::SetIntDivisibilityFn setResultDivs) const {
    auto mulOp = cast<arith::MulIOp>(op);
    APInt intVal;
    if (!matchPattern(mulOp.getLhs(), m_ConstantInt(&intVal))) {
      if (!matchPattern(mulOp.getRhs(), m_ConstantInt(&intVal))) {
        return;
      }
    }

    uint64_t udiv = intVal.getZExtValue();
    uint64_t sdiv = std::abs(intVal.getSExtValue());
    setResultDivs(mulOp.getResult(),
                  IREE::Util::ConstantIntDivisibility(udiv, sdiv));
  }
};

//===----------------------------------------------------------------------===//
// GlobalOpInterface
//===----------------------------------------------------------------------===//

struct GlobalOpInterfaceExternalModel
    : public IREE::Util::GlobalOpInterface::ExternalModel<
          GlobalOpInterfaceExternalModel, ml_program::GlobalOp> {
  Attribute getGlobalInitialValue(Operation *op) const {
    return cast<ml_program::GlobalOp>(op).getValueAttr();
  }
  void setGlobalInitialValue(Operation *op, Attribute value) const {
    if (value) {
      cast<ml_program::GlobalOp>(op).setValueAttr(value);
    } else {
      cast<ml_program::GlobalOp>(op).removeValueAttr();
    }
  }

  IREE::Util::InliningPolicyAttrInterface
  getGlobalInliningPolicy(Operation *op) const {
    if (op->hasAttr("noinline"))
      return IREE::Util::InlineNeverAttr::get(op->getContext());
    return {};
  }
  void
  setGlobalInliningPolicy(Operation *op,
                          IREE::Util::InliningPolicyAttrInterface value) const {
    if (isa_and_nonnull<IREE::Util::InlineNeverAttr>(value)) {
      op->setAttr("noinline", UnitAttr::get(op->getContext()));
    } else {
      op->removeAttr("noinline");
    }
  }

  IREE::Util::GlobalLoadOpInterface createLoadOp(Operation *op, Location loc,
                                                 OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    if (globalOp.getIsMutable()) {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          builder
              .create<ml_program::GlobalLoadOp>(
                  loc, globalOp.getType(), FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    } else {
      return cast<IREE::Util::GlobalLoadOpInterface>(
          builder
              .create<ml_program::GlobalLoadConstOp>(
                  loc, globalOp.getType(), FlatSymbolRefAttr::get(globalOp))
              .getOperation());
    }
  }

  IREE::Util::GlobalStoreOpInterface createStoreOp(Operation *op, Location loc,
                                                   Value value,
                                                   OpBuilder &builder) const {
    auto globalOp = cast<ml_program::GlobalOp>(op);
    return cast<IREE::Util::GlobalStoreOpInterface>(
        builder
            .create<ml_program::GlobalStoreOp>(
                loc, FlatSymbolRefAttr::get(globalOp), value)
            .getOperation());
  }
};

//===----------------------------------------------------------------------===//
// NumericCastOpInterface
//===----------------------------------------------------------------------===//

// Since all details of the interface are provided via default implementations,
// we can just have one templated external model to apply per op, vs one
// explicit model per op.
struct GenericNumericCastExternalModel {
  template <typename OpTy>
  struct ExternalModel
      : public IREE::Util::NumericCastOpInterface::ExternalModel<
            ExternalModel<OpTy>, OpTy> {};

  template <typename OpTy>
  static void add(MLIRContext *context) {
    OpTy::template attachInterface<ExternalModel<OpTy>>(*context);
  }

  template <typename OpTy1, typename OpTy2, typename... More>
  static void add(MLIRContext *context) {
    add<OpTy1>(context);
    add<OpTy2, More...>(context);
  }
};

//===----------------------------------------------------------------------===//
// TiedOpInterface
//===----------------------------------------------------------------------===//

struct InsertSliceOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          InsertSliceOpTiedOpInterface, tensor::InsertSliceOp> {
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
    : public IREE::Util::TiedOpInterface::ExternalModel<
          LinalgOpTiedOpInterface<OpTy>, OpTy> {
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
  static void registerOpInterface(MLIRContext *context) {
    (void)std::initializer_list<int>{
        0,
        (Ops::template attachInterface<LinalgOpTiedOpInterface<Ops>>(*context),
         0)...};
  }
};

// TODO(Max191): Remove this interface once GPU data tiling stops using early
// materialization. This only exists for handling multi_mma ops before dispatch
// workgroups are created, which only happens with early materialization.
struct MultiMmaOpTiedOpInterface
    : public IREE::Util::TiedOpInterface::ExternalModel<
          MultiMmaOpTiedOpInterface, IREE::GPU::MultiMmaOp> {
  Value getTiedResult(Operation *op, unsigned resultIndex) const {
    auto linalgOp = cast<IREE::GPU::MultiMmaOp>(op);
    return IREE::Util::TiedOpInterface::findTiedBaseValue(linalgOp.getAcc());
  }

  ::std::optional<unsigned>
  getTiedResultOperandIndex(Operation *op, unsigned resultIndex) const {
    return {2}; // acc
  }

  SmallVector<int64_t> getTiedResultOperandIndices(Operation *op) const {
    return {2}; // acc
  }
};

//===----------------------------------------------------------------------===//
// HoistableOpInterface
//===----------------------------------------------------------------------===//

template <typename OpTy>
struct UnhoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          UnhoistableOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return false; }
  bool isHoistableLeafOp(Operation *) const { return false; }
};

template <typename OpTy>
struct HoistableNonLeafOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableNonLeafOpInterface<OpTy>, OpTy> {
  bool isHoistableLeafOp(Operation *) const { return false; }
};

// The default interface is always hoistable. This acts as an override
// for other default hoistability checks as the interface is checked
// first.
template <typename OpTy>
struct AlwaysHoistableOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          AlwaysHoistableOpInterface<OpTy>, OpTy> {};

template <typename OpTy>
struct HoistableLinalgOpInterface
    : public IREE::Util::HoistableOpInterface::ExternalModel<
          HoistableLinalgOpInterface<OpTy>, OpTy> {
  bool isHoistableOp(Operation *) const { return true; }

  /// FillOp and broadcasting ops are not hoistableLeaf ops, since it is
  /// typically better to fuse them with their consumers.
  bool isHoistableLeafOp(Operation *op) const {
    auto genericOp = llvm::dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return !isa<linalg::FillOp>(op);
    // Generally, we prefer to not hoist broadcasts.
    // Detect op that only broadcast input as fusing them makes the new
    // op cheaper.
    if (genericOp.getNumParallelLoops() == genericOp.getNumLoops() &&
        isa<linalg::YieldOp>(genericOp.getBody()->front())) {
      for (OpOperand *opOperand : genericOp.getDpsInputOperands()) {
        AffineMap indexingMap = genericOp.getMatchingIndexingMap(opOperand);
        if (indexingMap.isProjectedPermutation() &&
            indexingMap.getNumDims() != indexingMap.getNumResults()) {
          return false;
        }
      }
    }
    return !linalg::isaFillOpInterface(genericOp).has_value();
  }
  bool isAtomicallyHoistableOp(Operation *) const { return true; }
  bool isOperandHoistable(Operation *, OpOperand *) const { return true; }
};

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated Hoistable___OpInterface.
template <typename... Ops>
struct UnhoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<UnhoistableOpInterface<Ops>>(*context), ...);
  }
};

template <typename... Ops>
struct HoistableNonLeafOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableNonLeafOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct AlwaysHoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<AlwaysHoistableOpInterface<Ops>>(*context),
     ...);
  }
};

template <typename... Ops>
struct HoistableLinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *context) {
    (Ops::template attachInterface<HoistableLinalgOpInterface<Ops>>(*context),
     ...);
  }
};

} // namespace

void registerUtilExternalModels(DialectRegistry &registry) {
  // Must ensure that any dependent dialects are registered.
  registry.insert<arith::ArithDialect>();
  registry.insert<linalg::LinalgDialect>();
  registry.insert<ml_program::MLProgramDialect>();
  registry.insert<tensor::TensorDialect>();

  registry.addExtension(
      +[](MLIRContext *context, ml_program::MLProgramDialect *dialect) {
        ml_program::GlobalOp::attachInterface<GlobalOpInterfaceExternalModel>(
            *context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            arith::ArithDialect *dialect) {
    GenericNumericCastExternalModel::add<
        arith::BitcastOp, arith::ExtFOp, arith::ExtUIOp, arith::ExtSIOp,
        arith::FPToSIOp, arith::FPToUIOp, arith::IndexCastOp, arith::TruncFOp,
        arith::TruncIOp, arith::SIToFPOp, arith::UIToFPOp>(context);
    arith::ConstantOp::attachInterface<
        ArithConstantInferIntDivisibilityOpInterface>(*context);
    arith::MulIOp::attachInterface<ArithMulIInferIntDivisibilityOpInterface>(
        *context);
  });

  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        tensor::InsertSliceOp::attachInterface<InsertSliceOpTiedOpInterface>(
            *context);
      });

  registry.addExtension(+[](MLIRContext *context,
                            IREE::GPU::IREEGPUDialect *dialect) {
    IREE::GPU::MultiMmaOp::attachInterface<MultiMmaOpTiedOpInterface>(*context);
  });

  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Register all Linalg structured ops. `LinalgOp` is an interface and it
        // is not possible to attach an external interface to an existing
        // interface. Therefore, attach the `TiedOpInterface` to all ops
        // one-by-one.
        LinalgOpTiedOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
      });

  // TODO(matthias-springer): Use a helper instead of listing all ops. This is
  // tricky because LinalgExtOps.td includes YieldOp.
  registry.addExtension(+[](MLIRContext *context,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    IREE::LinalgExt::ScatterOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::ScatterOp>>(*context);
    IREE::LinalgExt::SortOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::SortOp>>(*context);
    IREE::LinalgExt::FftOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::FftOp>>(*context);
    IREE::LinalgExt::ScanOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::ScanOp>>(*context);
    IREE::LinalgExt::TopkOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::TopkOp>>(*context);
    IREE::LinalgExt::WinogradInputTransformOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::WinogradInputTransformOp>>(
        *context);
    IREE::LinalgExt::WinogradFilterTransformOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::WinogradFilterTransformOp>>(
        *context);
    IREE::LinalgExt::WinogradOutputTransformOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::WinogradOutputTransformOp>>(
        *context);
    IREE::LinalgExt::Im2colOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::Im2colOp>>(*context);
    IREE::LinalgExt::AttentionOp::attachInterface<
        LinalgOpTiedOpInterface<IREE::LinalgExt::AttentionOp>>(*context);
  });

  // Hoistable Op Interface registration.

  // Register hoistable type interfaces for LinalgExt ops.
  registry.addExtension(
      +[](MLIRContext *context, IREE::Encoding::IREEEncodingDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::Encoding::SetEncodingOp>::registerOpInterface(context);
      });
  // Register hoistable type interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(
      +[](MLIRContext *context, linalg::LinalgDialect *dialect) {
        // Structured op implementations and a handful of pure ops are included.
        // Notably: IndexOp is not included because it establishes a hidden
        // dependency to the iterator and is non-const.

        // Register all LinalgOps ops. `LinalgOp` is an interface and it is
        // not possible to attach an external interface to an existing
        // interface. Therefore, attach the `HoistableLinalgOpInterface` to all
        // ops one-by-one.
        HoistableLinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
            >::registerOpInterface(context);
        UnhoistableOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
            >::registerOpInterface(context);
      });
  // Register hoistable type interfaces for tensor ops.
  registry.addExtension(
      +[](MLIRContext *context, tensor::TensorDialect *dialect) {
        // Never hoist empty and other pure metadata ops as a leaf. It's fine to
        // hoist them as a part of a larger constant tree that does actual work.
        HoistableNonLeafOpInterfaceHelper<
            tensor::EmptyOp, tensor::ExpandShapeOp, tensor::CollapseShapeOp,
            tensor::ExtractSliceOp>::registerOpInterface(context);
        // Cases of trivial pack/unpack should be handled as canonicalizations
        // before we get here, thus we're safe to always hoist.
        AlwaysHoistableOpInterfaceHelper<
            tensor::PadOp, tensor::PackOp,
            tensor::UnPackOp>::registerOpInterface(context);
      });
}

} // namespace mlir::iree_compiler
