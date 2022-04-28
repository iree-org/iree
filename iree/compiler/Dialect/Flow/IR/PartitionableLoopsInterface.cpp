// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.cpp.inc"  // IWYU pragma: export
// clang-format on

namespace mlir {
namespace iree_compiler {

/// Filters out dimensions in `parallelLoops` that have unit range in
/// `loopRanges`.
static llvm::SmallVector<unsigned> pruneUnitTripParallelLoops(
    llvm::ArrayRef<unsigned> parallelLoops,
    llvm::ArrayRef<int64_t> loopRanges) {
  return llvm::to_vector(llvm::make_filter_range(
      parallelLoops,
      [&loopRanges](unsigned loopDim) { return loopRanges[loopDim] != 1; }));
}

/// Returns the partitionable loops for all Linalg ops.
llvm::SmallVector<unsigned> getPartitionableLoopsImpl(
    linalg::LinalgOp linalgOp, unsigned maxNumPartitionedLoops) {
  llvm::SmallVector<unsigned> parallelLoops;
  linalgOp.getParallelDims(parallelLoops);
  // Get the static loop ranges.
  llvm::Optional<llvm::SmallVector<int64_t, 4>> staticLoopRanges =
      linalgOp.getStaticLoopRanges();
  if (staticLoopRanges) {
    parallelLoops =
        pruneUnitTripParallelLoops(parallelLoops, *staticLoopRanges);
  }
  // TODO(ravishankarm): For now the outer parallel loops are dropped. This is
  // a pragmatic choice for now but might need to be revisited.
  if (parallelLoops.size() > maxNumPartitionedLoops) {
    parallelLoops = llvm::to_vector(llvm::ArrayRef<unsigned>(parallelLoops)
                                        .take_back(maxNumPartitionedLoops));
  }
  return parallelLoops;
}

static llvm::SmallVector<llvm::StringRef> getIteratorTypesFromAttr(
    ArrayAttr iteratorTypesAttr) {
  return llvm::to_vector(llvm::map_range(iteratorTypesAttr, [](Attribute attr) {
    return attr.cast<StringAttr>().getValue();
  }));
}

namespace IREE {
namespace Flow {

/// External model implementation for all LinalgOps.
struct LinalgOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          LinalgOpPartitionableLoops, linalg::LinalgOp> {
  unsigned getNumLoops(Operation *op) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return linalgOp.getNumLoops();
  }

  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, unsigned maxNumPartitionedLoops) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return getPartitionableLoopsImpl(linalgOp, maxNumPartitionedLoops);
  }

  llvm::SmallVector<llvm::StringRef> getIteratorTypes(Operation *op) const {
    return getIteratorTypesFromAttr(
        cast<linalg::LinalgOp>(op).iterator_types());
  }
};

/// External model implementation for linalg::Mmt4DOp.
struct Mmt4DOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          Mmt4DOpPartitionableLoops, linalg::Mmt4DOp> {
  unsigned getNumLoops(Operation *op) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return linalgOp.getNumLoops();
  }

  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, unsigned maxNumPartitionedLoops) const {
    return {0, 1};
  }

  llvm::SmallVector<StringRef> getIteratorTypes(Operation *op) const {
    return getIteratorTypesFromAttr(
        cast<linalg::LinalgOp>(op).iterator_types());
  }
};

/// External model implementation for all operations that implement the
/// `TiledOpInterface`.
struct TiledOpInterfacePartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          TiledOpInterfacePartitionableLoops, LinalgExt::TiledOpInterface> {
  unsigned getNumLoops(Operation *op) const {
    auto tiledOp = cast<LinalgExt::TiledOpInterface>(op);
    return tiledOp.getLoopIteratorTypes().size();
  }

  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, unsigned maxNumPartitionedLoops) const {
    // For now just return the loops that are returned by the
    // `TiledOpInterface`. This needs to be further pruned to remove unit-dim
    // loops, but that needs the interface to return the static sizes of the
    // loops.
    auto tiledOp = cast<LinalgExt::TiledOpInterface>(op);
    return tiledOp.getPartitionableLoops(maxNumPartitionedLoops);
  }

  llvm::SmallVector<StringRef> getIteratorTypes(Operation *op) const {
    auto tiledOp = cast<LinalgExt::TiledOpInterface>(op);
    return tiledOp.getLoopIteratorTypes();
  }
};

/// Partitionable loop interface for `tensor.extract_slice` operation. Needed
/// only to build the workload during dispatch region formation.
/// TODO(ravishankarm): Drop this ExternalModel once the use of
/// PartitionableLoopsInterface is dropped from Flow.
struct TensorExtractOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          TensorExtractOpPartitionableLoops, tensor::ExtractSliceOp> {
  unsigned getNumLoops(Operation *op) const {
    return cast<tensor::ExtractSliceOp>(op).getType().getRank();
  }

  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, unsigned maxNumPartitionedLoops) const {
    auto sliceOp = cast<tensor::ExtractSliceOp>(op);
    auto partitionableLoops =
        llvm::to_vector(llvm::seq<unsigned>(0, sliceOp.getType().getRank()));
    if (partitionableLoops.size() > maxNumPartitionedLoops) {
      return llvm::to_vector(ArrayRef<unsigned>(partitionableLoops)
                                 .take_back(maxNumPartitionedLoops));
    }
    return partitionableLoops;
  }

  llvm::SmallVector<StringRef> getIteratorTypes(Operation *op) const {
    auto sliceOp = cast<tensor::ExtractSliceOp>(op);
    return llvm::SmallVector<StringRef>(sliceOp.getType().getRank(),
                                        getParallelIteratorTypeName());
  }
};

/// Partitionable loop interface for `tensor.insert_slice` operation. Needed
/// only to build the workload during dispatch region formation.
/// TODO(ravishankarm): Drop this ExternalModel once the use of
/// PartitionableLoopsInterface is dropped from Flow.
struct TensorInsertOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          TensorInsertOpPartitionableLoops, tensor::InsertSliceOp> {
  unsigned getNumLoops(Operation *op) const {
    return cast<tensor::InsertSliceOp>(op).getSourceType().getRank();
  }

  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, unsigned maxNumPartitionedLoops) const {
    auto sliceOp = cast<tensor::InsertSliceOp>(op);
    auto partitionableLoops = llvm::to_vector(
        llvm::seq<unsigned>(0, sliceOp.getSourceType().getRank()));
    if (partitionableLoops.size() > maxNumPartitionedLoops) {
      return llvm::to_vector(ArrayRef<unsigned>(partitionableLoops)
                                 .take_back(maxNumPartitionedLoops));
    }
    return partitionableLoops;
  }

  llvm::SmallVector<StringRef> getIteratorTypes(Operation *op) const {
    auto sliceOp = cast<tensor::InsertSliceOp>(op);
    return llvm::SmallVector<StringRef>(sliceOp.getType().getRank(),
                                        getParallelIteratorTypeName());
  }
};

/// Registers the `LinalgOpPartitionableLoops` model for all Linalg ops. This
/// needs to be done on a op-by-op basis since registration is on an op-by-op
/// basis.
template <typename OpTy>
static void registerInterfaceForLinalgOps(MLIRContext *ctx) {
  OpTy::template attachInterface<LinalgOpPartitionableLoops>(*ctx);
}

/// Specializations of the registration method to use a different external model
/// instead of the generic external model for Linalg ops.
template <>
void registerInterfaceForLinalgOps<linalg::Mmt4DOp>(MLIRContext *ctx) {
  linalg::Mmt4DOp::attachInterface<Mmt4DOpPartitionableLoops>(*ctx);
}

/// Registers the external models for all Linalg operations.
template <typename OpTy1, typename OpTy2, typename... More>
static void registerInterfaceForLinalgOps(MLIRContext *ctx) {
  registerInterfaceForLinalgOps<OpTy1>(ctx);
  registerInterfaceForLinalgOps<OpTy2, More...>(ctx);
}

/// Registers the `TiledOpInterfacePartitionableLoops` model for operations.
template <typename OpTy>
static void registerInterfaceForTiledOpInterfaceOps(MLIRContext *ctx) {
  OpTy ::template attachInterface<TiledOpInterfacePartitionableLoops>(*ctx);
}

/// Registers the external models for all TiledOpInterface operations.
template <typename OpTy1, typename OpTy2, typename... More>
static void registerInterfaceForTiledOpInterfaceOps(MLIRContext *ctx) {
  registerInterfaceForTiledOpInterfaceOps<OpTy1>(ctx);
  registerInterfaceForTiledOpInterfaceOps<OpTy2, More...>(ctx);
}

void registerPartitionableLoopsInterfaceModels(DialectRegistry &registry) {
  registry.insert<linalg::LinalgDialect>();

  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    registerInterfaceForLinalgOps<
        // clang-format off
  
  // This is copy-pasted from LinalgStructuredOps.cpp.inc. In theory you could
  // just include that generated file here, but that cause errors with bazel. 
  // The required generated header is not exposed correctly.
  // Copy paste is fine for now.

  ::mlir::linalg::BatchMatmulOp,
  ::mlir::linalg::BatchMatvecOp,
  ::mlir::linalg::Conv1DNwcWcfOp,
  ::mlir::linalg::Conv1DOp,
  ::mlir::linalg::Conv2DNchwFchwOp,
  ::mlir::linalg::Conv2DNhwcHwcfOp,
  ::mlir::linalg::Conv2DNhwcHwcfQOp,
  ::mlir::linalg::Conv2DOp,
  ::mlir::linalg::Conv3DNdhwcDhwcfOp,
  ::mlir::linalg::Conv3DOp,
  ::mlir::linalg::DepthwiseConv1DNwcWcOp,
  ::mlir::linalg::DepthwiseConv2DNhwcHwcOp,
  ::mlir::linalg::DepthwiseConv2DNhwcHwcQOp,
  ::mlir::linalg::DepthwiseConv2DNhwcHwcmOp,
  ::mlir::linalg::DepthwiseConv2DNhwcHwcmQOp,
  ::mlir::linalg::DotOp,
  ::mlir::linalg::FillOp,
  ::mlir::linalg::FillRng2DOp,
  ::mlir::linalg::GenericOp,
  ::mlir::linalg::MatmulOp,
  ::mlir::linalg::MatmulUnsignedOp,
  ::mlir::linalg::MatvecOp,
  ::mlir::linalg::Mmt4DOp,
  ::mlir::linalg::PoolingNchwMaxOp,
  ::mlir::linalg::PoolingNdhwcMaxOp,
  ::mlir::linalg::PoolingNdhwcMinOp,
  ::mlir::linalg::PoolingNdhwcSumOp,
  ::mlir::linalg::PoolingNhwcMaxOp,
  ::mlir::linalg::PoolingNhwcMaxUnsignedOp,
  ::mlir::linalg::PoolingNhwcMinOp,
  ::mlir::linalg::PoolingNhwcMinUnsignedOp,
  ::mlir::linalg::PoolingNhwcSumOp,
  ::mlir::linalg::QuantizedBatchMatmulOp,
  ::mlir::linalg::QuantizedMatmulOp,
  ::mlir::linalg::VecmatOp
        // clang-format on
        >(ctx);
  });

  registry.insert<LinalgExt::IREELinalgExtDialect>();

  registry.addExtension(
      +[](MLIRContext *ctx, LinalgExt::IREELinalgExtDialect *dialect) {
        registerInterfaceForTiledOpInterfaceOps<
            LinalgExt::FftOp, LinalgExt::ReverseOp, LinalgExt::ScanOp,
            LinalgExt::ScatterOp, LinalgExt::SortOp>(ctx);
      });
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::ExtractSliceOp::attachInterface<TensorExtractOpPartitionableLoops>(
        *ctx);
    tensor::InsertSliceOp::attachInterface<TensorInsertOpPartitionableLoops>(
        *ctx);
  });
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
