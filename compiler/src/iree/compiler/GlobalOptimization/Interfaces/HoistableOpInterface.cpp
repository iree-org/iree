// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/GlobalOptimization/Interfaces/HoistableOpInterface.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace mlir {
namespace iree_compiler {

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
  bool isHoistableLeafOp(Operation *op) const {
    auto genericOp = llvm::dyn_cast<linalg::GenericOp>(op);
    if (!genericOp)
      return true;
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
    return true;
  }
  bool isAtomicallyHoistableOp(Operation *) const { return true; }
  bool isOperandHoistable(Operation *op, OpOperand *operand) const {
    linalg::LinalgOp linalgOp = llvm::cast<linalg::LinalgOp>(op);
    // For linalg ops, we only want to hoist inputs.
    return operand->getOperandNumber() < linalgOp.getNumDpsInputs();
  }
};

//===----------------------------------------------------------------------===//
// Interface Registration
//===----------------------------------------------------------------------===//

/// Helper structures that iterates over all Op types in `OpTys` and registers
/// the associated Hoistable___OpInterface.
template <typename... Ops>
struct UnhoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<UnhoistableOpInterface<Ops>>(*ctx), ...);
  }
};

template <typename... Ops>
struct HoistableNonLeafOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<HoistableNonLeafOpInterface<Ops>>(*ctx),
     ...);
  }
};

template <typename... Ops>
struct AlwaysHoistableOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<AlwaysHoistableOpInterface<Ops>>(*ctx), ...);
  }
};

template <typename... Ops>
struct HoistableLinalgOpInterfaceHelper {
  static void registerOpInterface(MLIRContext *ctx) {
    (Ops::template attachInterface<HoistableLinalgOpInterface<Ops>>(*ctx), ...);
  }
};

void registerHoistableOpInterfaces(DialectRegistry &registry) {
  // Register hoistable type interfaces for LinalgExt ops.
  registry.addExtension(
      +[](MLIRContext *ctx, IREE::LinalgExt::IREELinalgExtDialect *dialect) {
        UnhoistableOpInterfaceHelper<
            IREE::LinalgExt::SetEncodingOp,
            IREE::LinalgExt::UpperBoundTileSizeOp>::registerOpInterface(ctx);
      });
  // Register hoistable type interfaces for linalg ops.
  // We have a specific allow-list for Linalg ops because we want to consider
  // new additions carefully.
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    // Structured op implementations and a handful of pure ops are included.
    // Notably: IndexOp is not included because it establishes a hidden
    // dependency to the iterator and is non-const.

    // Register all LinalgOps ops. `LinalgOp` is an interface and it is
    // not possible to attach an external interface to an existing interface.
    // Therefore, attach the `HoistableLinalgOpInterface` to all ops one-by-one.
    HoistableLinalgOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >::registerOpInterface(ctx);
    UnhoistableOpInterfaceHelper<
#define GET_OP_LIST
#include "mlir/Dialect/Linalg/IR/LinalgOps.cpp.inc"
        >::registerOpInterface(ctx);
  });
  // Register hoistable type interfaces for tensor ops.
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    // Never hoist empty and other pure metadata ops as a leaf. It's fine to
    // hoist them as a part of a larger constant tree that does actual work.
    HoistableNonLeafOpInterfaceHelper<
        tensor::EmptyOp, tensor::ExpandShapeOp,
        tensor::CollapseShapeOp>::registerOpInterface(ctx);
    // Cases of trivial pack/unpack should be handled as canonicalizations
    // before we get here, thus we're safe to always hoist.
    AlwaysHoistableOpInterfaceHelper<
        tensor::PadOp, tensor::PackOp,
        tensor::UnPackOp>::registerOpInterface(ctx);
  });
}

} // namespace iree_compiler
} // namespace mlir
