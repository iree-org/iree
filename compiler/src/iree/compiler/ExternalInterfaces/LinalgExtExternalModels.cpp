// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ExternalInterfaces/LinalgExtExternalModels.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace mlir::iree_compiler {
namespace {
// Used to register the LinalgFusionOpInterface with the linalg ops.
template <typename ConcreteType>
struct LinalgFusionOpInterfaceAdapter
    : public IREE::LinalgExt::LinalgFusionOpInterface::ExternalModel<
          LinalgFusionOpInterfaceAdapter<ConcreteType>, ConcreteType> {
public:
  SmallVector<AffineMap> getIndexingMapsForOperands(mlir::Operation *op) const {
    auto maps = llvm::cast<ConcreteType>(op)
                    .getIndexingMaps()
                    .template getAsValueRange<AffineMapAttr>();
    return {maps.begin(),
            maps.end() - llvm::cast<ConcreteType>(op).getNumResults()};
  }

  SmallVector<AffineMap> getIndexingMapsForResults(mlir::Operation *op) const {
    auto maps = llvm::cast<ConcreteType>(op)
                    .getIndexingMaps()
                    .template getAsValueRange<AffineMapAttr>();
    return {maps.end() - llvm::cast<ConcreteType>(op).getNumResults(),
            maps.end()};
  }

  // Forward all the interface methods to the corresponding linalg op.
  unsigned getNumParallelLoops(mlir::Operation *op) const {
    return (llvm::cast<ConcreteType>(op).getNumParallelLoops());
  }

  unsigned getNumLoops(mlir::Operation *op) const {
    return (llvm::cast<ConcreteType>(op).getNumLoops());
  }

  FailureOr<SmallVector<int64_t>>
  getStaticLoopRanges(mlir::Operation *op) const {
    return SmallVector<int64_t>(
        llvm::cast<ConcreteType>(op).getStaticLoopRanges());
  }

  AffineMap getIndexingMapMatchingResult(mlir::Operation *op,
                                         OpResult result) const {
    return (llvm::cast<ConcreteType>(op).getIndexingMapMatchingResult(result));
  }

  AffineMap getMatchingIndexingMap(mlir::Operation *op,
                                   OpOperand *operand) const {
    return (llvm::cast<ConcreteType>(op).getMatchingIndexingMap(operand));
  }

  SmallVector<AffineMap> getIndexingMapsArray(mlir::Operation *op) const {
    auto inputMaps = getIndexingMapsForOperands(op);
    llvm::append_range(inputMaps, getIndexingMapsForResults(op));
    return inputMaps;
  }
};

struct SoftmaxFusionOpInterfaceAdapter
    : public IREE::LinalgExt::LinalgFusionOpInterface::ExternalModel<
          SoftmaxFusionOpInterfaceAdapter, linalg::SoftmaxOp> {
public:
  SmallVector<AffineMap> getIndexingMapsForOperands(mlir::Operation *op) const {
    Builder b(op->getContext());
    return llvm::to_vector(llvm::map_range(
        llvm::cast<linalg::SoftmaxOp>(op).getDpsInputs(),
        [&b](Value operand) -> AffineMap {
          auto rank = cast<ShapedType>(operand.getType()).getRank();
          return b.getMultiDimIdentityMap(rank);
        }));
  }

  SmallVector<AffineMap> getIndexingMapsForResults(mlir::Operation *op) const {
    Builder b(op->getContext());
    return llvm::to_vector(llvm::map_range(
        llvm::cast<linalg::SoftmaxOp>(op).getDpsInits(),
        [&b](Value operand) -> AffineMap {
          auto rank = cast<ShapedType>(operand.getType()).getRank();
          return b.getMultiDimIdentityMap(rank);
        }));
  }

  FailureOr<SmallVector<int64_t>> getStaticLoopRanges(Operation *op) const {
    auto softmaxOp = cast<linalg::SoftmaxOp>(op);
    // Softmax loop range is the input shape.
    return SmallVector<int64_t>(softmaxOp.getInputOperandType().getShape());
  }

  AffineMap getIndexingMapMatchingResult(mlir::Operation *op,
                                         OpResult result) const {
    return getIndexingMapsForResults(op)[result.getResultNumber()];
  }

  AffineMap getMatchingIndexingMap(mlir::Operation *op,
                                   OpOperand *operand) const {
    return getIndexingMapsForOperands(op)[operand->getOperandNumber()];
  }

  SmallVector<AffineMap> getIndexingMapsArray(mlir::Operation *op) const {
    auto inputMaps = getIndexingMapsForOperands(op);
    llvm::append_range(inputMaps, getIndexingMapsForResults(op));
    return inputMaps;
  }
};

template <typename... Args>
void registerOpsWithLinalgExtOpInterface(mlir::MLIRContext *context) {
  (Args::template attachInterface<LinalgFusionOpInterfaceAdapter<Args>>(
       *context),
   ...);
}

} // namespace

void registerLinalgExtExternalModels(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    ctx->loadDialect<mlir::linalg::LinalgDialect>();

#define GET_OP_LIST
    registerOpsWithLinalgExtOpInterface<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
    linalg::SoftmaxOp::attachInterface<SoftmaxFusionOpInterfaceAdapter>(*ctx);
  });
}

} // namespace mlir::iree_compiler
