// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::iree_compiler::IREE::LinalgExt;

#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc" // IWYU pragma: keep

// Used to control inlining behavior.
namespace {
struct IREELinalgExtInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
    // Sure!
    return true;
  }
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }

  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const final {
    // Sure!
    return true;
  }
};

// Used to register the LinalgFusionOpInterface with the linalg ops.
template <typename ConcreteType>
struct LinalgFusionOpInterfaceAdapter
    : public LinalgFusionOpInterface::ExternalModel<
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

  SmallVector<int64_t, 4> getStaticLoopRanges(mlir::Operation *op) const {
    return (llvm::cast<ConcreteType>(op).getStaticLoopRanges());
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
    : public LinalgFusionOpInterface::ExternalModel<
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
} // namespace

struct IREELinalgExtDialectOpAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;
};

template <typename... Args>
static void registerOpsWithLinalgExtOpInterface(mlir::MLIRContext *context) {
  (Args::template attachInterface<LinalgFusionOpInterfaceAdapter<Args>>(
       *context),
   ...);
}

void IREELinalgExtDialect::initialize() {
  mlir::MLIRContext *context = getContext();
  context->loadDialect<mlir::linalg::LinalgDialect>();

#define GET_OP_LIST
  declarePromisedInterfaces<LinalgFusionOpInterface,
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
                            >();

#define GET_OP_LIST
  registerOpsWithLinalgExtOpInterface<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(context);
  linalg::SoftmaxOp::attachInterface<SoftmaxFusionOpInterfaceAdapter>(*context);

  addInterfaces<IREELinalgExtInlinerInterface>();

  addInterfaces<IREELinalgExtDialectOpAsmInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtAttrs.cpp.inc"
      >();

#define GET_OP_LIST
  addOperations<
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.cpp.inc"
      >();
}

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.cpp.inc"
