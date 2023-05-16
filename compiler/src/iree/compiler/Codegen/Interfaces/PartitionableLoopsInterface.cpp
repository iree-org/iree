// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.h"

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"

// clang-format off
#include "iree/compiler/Codegen/Interfaces/PartitionableLoopsInterface.cpp.inc"  // IWYU pragma: export
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
    linalg::LinalgOp linalgOp, std::optional<unsigned> maxNumPartitionedLoops) {
  llvm::SmallVector<unsigned> parallelLoops;
  linalgOp.getParallelDims(parallelLoops);
  // Get the static loop ranges.
  llvm::SmallVector<int64_t, 4> staticLoopRanges =
      linalgOp.getStaticLoopRanges();
  parallelLoops = pruneUnitTripParallelLoops(parallelLoops, staticLoopRanges);
  // TODO(ravishankarm): For now the outer parallel loops are dropped. This is
  // a pragmatic choice for now but might need to be revisited.
  if (maxNumPartitionedLoops.has_value() &&
      parallelLoops.size() > maxNumPartitionedLoops.value()) {
    parallelLoops =
        llvm::to_vector(llvm::ArrayRef(parallelLoops)
                            .take_back(maxNumPartitionedLoops.value()));
  }
  return parallelLoops;
}

static llvm::SmallVector<utils::IteratorType> getIteratorTypesFromAttr(
    ArrayAttr iteratorTypesAttr) {
  return llvm::to_vector(llvm::map_range(iteratorTypesAttr, [](Attribute attr) {
    return utils::symbolizeIteratorType(attr.cast<StringAttr>().getValue())
        .value();
  }));
}

/// External model implementation for all LinalgOps.
template <typename OpTy>
struct LinalgOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          LinalgOpPartitionableLoops<OpTy>, OpTy> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    auto linalgOp = cast<linalg::LinalgOp>(op);
    return getPartitionableLoopsImpl(linalgOp, maxNumPartitionedLoops);
  }
};

/// External model implementation for linalg::Mmt4DOp.
struct Mmt4DOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          Mmt4DOpPartitionableLoops, linalg::Mmt4DOp> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    return {0, 1};
  }
};

/// External model implementation for all operations to make only
/// the outer parallel loops as partitionable.
template <typename OpTy>
struct OuterParallelAsPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          OuterParallelAsPartitionableLoops<OpTy>, OpTy> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    // For now just return the loops that are returned by the
    // `TiledOpInterface`. This needs to be further pruned to remove unit-dim
    // loops, but that needs the interface to return the static sizes of the
    // loops.
    SmallVector<unsigned> partitionableLoops;
    auto interfaceOp = cast<TilingInterface>(op);
    for (auto [index, iteratorType] :
         llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
      if (iteratorType != utils::IteratorType::parallel) {
        break;
      }
      partitionableLoops.push_back(index);
    }
    if (!maxNumPartitionedLoops.has_value() ||
        partitionableLoops.size() <= maxNumPartitionedLoops.value()) {
      return partitionableLoops;
    }
    partitionableLoops.erase(
        partitionableLoops.begin(),
        std::next(partitionableLoops.begin(),
                  partitionableLoops.size() - maxNumPartitionedLoops.value()));
    return partitionableLoops;
  }
};

/// External model implementation for operations that are to be executed
/// sequentially.
template <typename OpTy>
struct NoPartitionableLoops : public PartitionableLoopsInterface::ExternalModel<
                                  NoPartitionableLoops<OpTy>, OpTy> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    return {};
  }
};

/// External model implementation for specifying partitionable loops of FftOp.
struct FftOpPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          FftOpPartitionableLoops, IREE::LinalgExt::FftOp> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    auto fftOp = cast<IREE::LinalgExt::FftOp>(op);
    auto range = llvm::seq<unsigned>(0, fftOp.getOperandRank());
    SmallVector<unsigned> partitionableLoops(range.begin(), range.end());
    // Indices matter for coeff computation.
    if (!fftOp.hasCoeff()) {
      partitionableLoops.pop_back();
    }
    if (!maxNumPartitionedLoops.has_value() ||
        partitionableLoops.size() <= maxNumPartitionedLoops.value()) {
      return partitionableLoops;
    }
    partitionableLoops.erase(
        partitionableLoops.begin(),
        std::next(partitionableLoops.begin(),
                  partitionableLoops.size() - maxNumPartitionedLoops.value()));
    return partitionableLoops;
  }
};

/// External model implementation for making all parallel loops as
/// partitionable.
template <typename OpTy>
struct AllParallelAsPartitionableLoops
    : public PartitionableLoopsInterface::ExternalModel<
          AllParallelAsPartitionableLoops<OpTy>, OpTy> {
  llvm::SmallVector<unsigned> getPartitionableLoops(
      Operation *op, std::optional<unsigned> maxNumPartitionedLoops) const {
    SmallVector<unsigned> partitionableLoops;
    auto interfaceOp = cast<OpTy>(op);
    for (auto iteratorType :
         llvm::enumerate(interfaceOp.getLoopIteratorTypes())) {
      if (iteratorType.value() != utils::IteratorType::parallel) {
        continue;
      }
      partitionableLoops.push_back(iteratorType.index());
    }
    if (!maxNumPartitionedLoops.has_value() ||
        partitionableLoops.size() <= maxNumPartitionedLoops.value()) {
      return partitionableLoops;
    }
    partitionableLoops.erase(
        partitionableLoops.begin(),
        std::next(partitionableLoops.begin(),
                  partitionableLoops.size() - maxNumPartitionedLoops.value()));
    return partitionableLoops;
  }
};

/// Registers the `LinalgOpPartitionableLoops` model for all Linalg ops. This
/// needs to be done on a op-by-op basis since registration is on an op-by-op
/// basis.
template <typename OpTy>
static void registerInterfaceForLinalgOps(MLIRContext *ctx) {
  OpTy::template attachInterface<LinalgOpPartitionableLoops<OpTy>>(*ctx);
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

void registerPartitionableLoopsInterfaceModels(DialectRegistry &registry) {
  registry.insert<linalg::LinalgDialect>();

#define GET_OP_LIST
  registry.addExtension(+[](MLIRContext *ctx, linalg::LinalgDialect *dialect) {
    registerInterfaceForLinalgOps<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
        >(ctx);
  });

  registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();

  registry.addExtension(+[](MLIRContext *ctx,
                            IREE::LinalgExt::IREELinalgExtDialect *dialect) {
    IREE::LinalgExt::FftOp::attachInterface<FftOpPartitionableLoops>(*ctx);
    IREE::LinalgExt::ScanOp::attachInterface<
        AllParallelAsPartitionableLoops<IREE::LinalgExt::ScanOp>>(*ctx);
    IREE::LinalgExt::ScatterOp::attachInterface<
        OuterParallelAsPartitionableLoops<IREE::LinalgExt::ScatterOp>>(*ctx);
    IREE::LinalgExt::SortOp::attachInterface<
        AllParallelAsPartitionableLoops<IREE::LinalgExt::SortOp>>(*ctx);
    IREE::LinalgExt::ReverseOp::attachInterface<
        OuterParallelAsPartitionableLoops<IREE::LinalgExt::ReverseOp>>(*ctx);
    IREE::LinalgExt::TopkOp::attachInterface<
        AllParallelAsPartitionableLoops<IREE::LinalgExt::TopkOp>>(*ctx);
    IREE::LinalgExt::WinogradInputTransformOp::attachInterface<
        AllParallelAsPartitionableLoops<
            IREE::LinalgExt::WinogradInputTransformOp>>(*ctx);
    IREE::LinalgExt::WinogradOutputTransformOp::attachInterface<
        AllParallelAsPartitionableLoops<
            IREE::LinalgExt::WinogradOutputTransformOp>>(*ctx);
    IREE::LinalgExt::SoftmaxOp::attachInterface<
        AllParallelAsPartitionableLoops<IREE::LinalgExt::SoftmaxOp>>(*ctx);
    IREE::LinalgExt::AttentionOp::attachInterface<
        AllParallelAsPartitionableLoops<IREE::LinalgExt::AttentionOp>>(*ctx);
  });
  registry.addExtension(+[](MLIRContext *ctx, tensor::TensorDialect *dialect) {
    tensor::PackOp::attachInterface<
        OuterParallelAsPartitionableLoops<tensor::PackOp>>(*ctx);
    tensor::PadOp::attachInterface<
        OuterParallelAsPartitionableLoops<tensor::PadOp>>(*ctx);
    tensor::UnPackOp::attachInterface<
        OuterParallelAsPartitionableLoops<tensor::UnPackOp>>(*ctx);
  });
}

}  // namespace iree_compiler
}  // namespace mlir
