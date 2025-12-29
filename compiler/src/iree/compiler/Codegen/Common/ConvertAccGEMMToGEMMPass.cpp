// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- ConvertAccGEMMtoGEMMpass.cpp ----------------------------------===//
//
// Converts Accumulating GEMM to GEMM + elementwise add.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/MatchUtils.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_CONVERTACCGEMMTOGEMMPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

// Get the `iree_tensor_ext.dispatch.tensor.load` that this value is
// populated with. This could potentially walk the use-def chain to get the load
// operation, but for now it just returns the load op if that is the defining
// operation for `v`.
template <typename LoadOpTy>
std::optional<LoadOpTy> getLoadOp(Value v) {
  auto loadOp = v.getDefiningOp<LoadOpTy>();
  if (loadOp) {
    return loadOp;
  }
  return std::nullopt;
}

// Get the `iree_tensor_ext.dispatch.tensor.store` that this value is
// populated writes to. This could potentially walk the use-def chain of DPS
// init operands to get the store operation, but for now it just returns the
// store op if the result has a single use and that use is the store op.
template <typename StoreOpTy>
std::optional<StoreOpTy> getStoreOp(Value v) {
  if (v.getNumUses() != 1) {
    return std::nullopt;
  }
  auto storeOp = dyn_cast<StoreOpTy>(*(v.getUsers().begin()));
  if (storeOp) {
    return storeOp;
  }
  return std::nullopt;
}

// Check if the init value of the operation is read/write from the same buffer.
// If not, it is invalid to use an accumulating GEMM operation, and convert it
// to a non-accumulating GEMM.
static bool isValidInPlaceAccumulatingOp(DestinationStyleOpInterface dpsOp) {
  assert(dpsOp.getNumDpsInits() == 1 &&
         "expected op to have a single outs operand");
  OpOperand *initValue = dpsOp.getDpsInitOperand(0);

  // Case 1. Check for the case when reading/writing from the same buffer
  // through `iree_codegen.load_from_buffer`/`iree_codegen.store_to_buffer`.
  {
    std::optional<IREE::Codegen::LoadFromBufferOp> loadOp =
        getLoadOp<IREE::Codegen::LoadFromBufferOp>(initValue->get());
    std::optional<IREE::Codegen::StoreToBufferOp> storeOp =
        getStoreOp<IREE::Codegen::StoreToBufferOp>(dpsOp->getResult(0));
    if (loadOp && storeOp && loadOp->getBuffer() == storeOp->getBuffer()) {
      return true;
    }
  }

  // Case 2. If the `outs` operand is from a read-write buffer, and the result
  // is writing into the same buffer, do not convert to a non-accumulating gemm.
  // This currently would only work for very simple cases, but could be
  // generalized further.
  {
    std::optional<IREE::TensorExt::DispatchTensorLoadOp> initLoadOp =
        getLoadOp<IREE::TensorExt::DispatchTensorLoadOp>(initValue->get());
    std::optional<IREE::TensorExt::DispatchTensorStoreOp> resultStoreOp =
        getStoreOp<IREE::TensorExt::DispatchTensorStoreOp>(dpsOp->getResult(0));
    if (initLoadOp && resultStoreOp && initLoadOp->getSource() &&
        resultStoreOp->getTarget()) {
      // Check that the source and the result have a read/write tag. If they
      // don't then its really a bug in the way the dispatch is formed, but check
      // here for safety.
      if (initLoadOp->getSourceType().getAccess() ==
          IREE::TensorExt::TensorAccess::ReadWrite) {
        return true;
      }
    }
  }
  return false;
}

static bool accGemmToGemmPrecondition(Operation *op) {
  if (auto innerTiledOp = dyn_cast<IREE::Codegen::InnerTiledOp>(op)) {
    return isa<IREE::GPU::MmaInterfaceAttr, IREE::GPU::ScaledMMAAttr,
               IREE::GPU::DataTiledMMAInterfaceAttr>(innerTiledOp.getKind());
  }
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp) {
    return false;
  }
  if (!linalg::isaContractionOpInterface(linalgOp) &&
      !isa<linalg::ConvolutionOpInterface>(*linalgOp) &&
      !IREE::LinalgExt::isaScaledContractionOpInterface(linalgOp)) {
    return false;
  }
  if (!linalgOp.hasPureTensorSemantics()) {
    return false;
  }
  if (isValidInPlaceAccumulatingOp(
          cast<DestinationStyleOpInterface>(linalgOp.getOperation()))) {
    return false;
  }

  return linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0))
      .isProjectedPermutation();
}

static bool isFuncArgument(Value v) {
  auto blockArg = dyn_cast<BlockArgument>(v);
  if (!blockArg) {
    return false;
  }
  return isa<func::FuncOp, IREE::Util::FuncOp>(
      blockArg.getParentBlock()->getParentOp());
}

static void convertAccGemmToGemm(RewriterBase &rewriter,
                                 DestinationStyleOpInterface dpsOp) {
  SmallVector<OpOperand *> outputOperands =
      llvm::to_vector(llvm::make_pointer_range(dpsOp.getDpsInitsMutable()));
  Value outputOperand = outputOperands.front()->get();
  auto outsDefiningOp = outputOperand.getDefiningOp();
  // If, not a function argument, and not DispatchTensorLoadOp/LoadFromBufferOp
  // then do nothing.
  if (!isFuncArgument(outputOperand) &&
      !isa_and_nonnull<IREE::TensorExt::DispatchTensorLoadOp,
                       IREE::Codegen::LoadFromBufferOp>(outsDefiningOp)) {
    return;
  }
  auto outputType = cast<RankedTensorType>(outputOperand.getType());
  if (!outputType.getElementType().isIntOrFloat()) {
    return;
  }
  auto elementType = outputType.getElementType();

  Location loc = dpsOp.getLoc();
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(dpsOp);

  int64_t outputRank = outputType.getRank();
  SmallVector<utils::IteratorType> iterators(outputRank,
                                             utils::IteratorType::parallel);
  SmallVector<AffineMap> maps(3, rewriter.getMultiDimIdentityMap(outputRank));

  // Create a zero tensor as the new output tensor operand to the Linalg
  // contraction op.
  SmallVector<OpFoldResult> mixedSizes =
      tensor::getMixedSizes(rewriter, loc, outputOperand);
  Value initOp =
      tensor::EmptyOp::create(rewriter, loc, mixedSizes, elementType);
  Value zero = arith::ConstantOp::create(rewriter, loc,
                                         rewriter.getZeroAttr(elementType));
  Value fill = linalg::FillOp::create(rewriter, loc, zero, initOp).result();

  // Update the contraction op to use the new zero tensor as output operand.
  rewriter.modifyOpInPlace(dpsOp, [&]() { dpsOp.setDpsInitOperand(0, fill); });

  // Create a generic op to add back the original output tensor operand.
  rewriter.setInsertionPointAfter(dpsOp);
  auto genericOp = linalg::GenericOp::create(
      rewriter, loc, outputType, ValueRange{dpsOp->getResult(0), outputOperand},
      ValueRange{initOp}, maps, iterators,
      [&](OpBuilder &b, Location nestedLoc, ValueRange args) {
        Value result;
        if (isa<FloatType>(elementType)) {
          result = arith::AddFOp::create(b, nestedLoc, args[0], args[1]);
        } else {
          result = arith::AddIOp::create(b, nestedLoc, args[0], args[1]);
        }
        linalg::YieldOp::create(b, nestedLoc, result);
      });
  dpsOp->getResult(0).replaceAllUsesExcept(genericOp->getResult(0), genericOp);
}

namespace {

struct ConvertAccGEMMToGEMMPass final
    : impl::ConvertAccGEMMToGEMMPassBase<ConvertAccGEMMToGEMMPass> {
  void runOnOperation() override {
    FunctionOpInterface funcOp = getOperation();
    SmallVector<Operation *> candidates = llvm::filter_to_vector(
        llvm::make_pointer_range(funcOp.getFunctionBody().getOps()),
        accGemmToGemmPrecondition);
    IRRewriter rewriter(&getContext());
    for (Operation *candidate : candidates) {
      convertAccGemmToGemm(rewriter,
                           cast<DestinationStyleOpInterface>(candidate));
    }
  }
};

} // namespace
} // namespace mlir::iree_compiler
