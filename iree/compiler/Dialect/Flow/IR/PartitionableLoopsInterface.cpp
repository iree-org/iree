// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree-dialects/Dialect/LinalgExt/IR/TiledOpInterface.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"

// clang-format off
#include "iree/compiler/Dialect/Flow/IR/PartitionableLoopsInterface.cpp.inc"  // IWYU pragma: export
// clang-format on

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
    mlir::linalg::LinalgOp linalgOp, unsigned maxNumPartitionedLoops) {
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

namespace mlir {
namespace iree_compiler {
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
};

/// Registers the `LinalgOpPartitionableLoops` model for all Linalg ops. This
/// needs to be done on a op-by-op basis since registration is on an op-by-op
/// basis.
template <typename OpTy>
static void registerInterfaceForLinalgOps(DialectRegistry &registry) {
  registry.addOpInterface<OpTy, LinalgOpPartitionableLoops>();
}

/// Specializations of the registration method to use a different external model
/// instead of the generic external model for Linalg ops.
template <>
void registerInterfaceForLinalgOps<linalg::Mmt4DOp>(DialectRegistry &registry) {
  registry.addOpInterface<linalg::Mmt4DOp, Mmt4DOpPartitionableLoops>();
}

/// Registers the external models for all Linalg operations.
template <typename OpTy1, typename OpTy2, typename... More>
static void registerInterfaceForLinalgOps(DialectRegistry &registry) {
  registerInterfaceForLinalgOps<OpTy1>(registry);
  registerInterfaceForLinalgOps<OpTy2, More...>(registry);
}

/// Registers the `TiledOpInterfacePartitionableLoops` model for operations.
template <typename OpTy>
static void registerInterfaceForTiledOpInterfaceOps(DialectRegistry &registry) {
  registry.addOpInterface<OpTy, TiledOpInterfacePartitionableLoops>();
}

/// Registers the external models for all TiledOpInterface operations.
template <typename OpTy1, typename OpTy2, typename... More>
static void registerInterfaceForTiledOpInterfaceOps(DialectRegistry &registry) {
  registerInterfaceForTiledOpInterfaceOps<OpTy1>(registry);
  registerInterfaceForTiledOpInterfaceOps<OpTy2, More...>(registry);
}

void registerPartitionableLoopsInterfaceModels(DialectRegistry &registry) {
  // clang-format off
  registerInterfaceForLinalgOps<
  
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
  ::mlir::linalg::CopyOp,
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
  ::mlir::linalg::SoftPlus2DOp,
  ::mlir::linalg::VecmatOp
  >(registry);
  // clang-format on

  registerInterfaceForTiledOpInterfaceOps<
      LinalgExt::FftOp, LinalgExt::ReverseOp, LinalgExt::ScanOp,
      LinalgExt::ScatterOp, LinalgExt::SortOp, tensor::ExtractSliceOp,
      tensor::InsertSliceOp>(registry);
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
