// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-spirv-erase-storage-buffer-static-shape"

namespace mlir {
namespace iree_compiler {

namespace {

class EraseStorageBufferStaticShapePass final
    : public SPIRVEraseStorageBufferStaticShapeBase<
          EraseStorageBufferStaticShapePass> {
  void runOnOperation() override;
};

/// Returns true if the given `subspanOp` is from a 1-D static shaped storage
/// buffer.
bool is1DStaticShapedStorageBuffer(
    IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  auto type = subspanOp.getType().dyn_cast<MemRefType>();
  if (!type) return false;
  auto attr =
      type.getMemorySpace().dyn_cast_or_null<IREE::HAL::DescriptorTypeAttr>();
  if (!attr) return false;
  return type.hasStaticShape() && type.getRank() == 1 &&
         attr.getValue() == IREE::HAL::DescriptorType::StorageBuffer;
}

/// Rewrites a subspan op with 1-D static shape into dynamic shape.
/// e.g.,
///
/// ```mlir
///  hal.interface.binding.subspan set(0) binding(0) offset(%offset)
///      : memref<16xf32>
/// ```
///
/// is re-written to
///
/// ```mlir
///  hal.interface.binding.subspan set(0) binding(0) offset(%offset)
///      : memref<?xf32>{%c16}
/// ```
IREE::HAL::InterfaceBindingSubspanOp rewriteStorageBufferSubspanOp(
    RewriterBase &rewriter, IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
  assert(is1DStaticShapedStorageBuffer(subspanOp));
  LLVM_DEBUG({
    llvm::dbgs() << "Rewriting subspan op: ";
    subspanOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(subspanOp);

  auto oldType = subspanOp.getType().cast<MemRefType>();
  auto newType =
      MemRefType::get({ShapedType::kDynamic}, oldType.getElementType(),
                      oldType.getLayout(), oldType.getMemorySpace());

  SmallVector<Value, 1> dynamicDims;
  assert(subspanOp.getDynamicDims().empty());
  dynamicDims.push_back(rewriter.create<arith::ConstantIndexOp>(
      subspanOp.getLoc(), oldType.getNumElements()));

  auto newOp = rewriter.create<IREE::HAL::InterfaceBindingSubspanOp>(
      subspanOp.getLoc(), newType, subspanOp.getSetAttr(),
      subspanOp.getBindingAttr(), subspanOp.getDescriptorTypeAttr(),
      subspanOp.getByteOffset(), dynamicDims, subspanOp.getAlignmentAttr(),
      subspanOp.getDescriptorFlagsAttr());

  LLVM_DEBUG({
    llvm::dbgs() << "Rewritten to: ";
    newOp->print(llvm::dbgs(), OpPrintingFlags().assumeVerified());
    llvm::dbgs() << "\n";
  });
  return newOp;
}

}  // namespace

void EraseStorageBufferStaticShapePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // Collect all storage buffer subspan ops with 1-D static shapes. We only need
  // to handle such cases here--high-D static shapes are expected to be flattend
  // into 1-D by a previous pass.
  SmallVector<IREE::HAL::InterfaceBindingSubspanOp> subspanOps;
  funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    if (is1DStaticShapedStorageBuffer(subspanOp)) {
      subspanOps.push_back(subspanOp);
    }
  });

  IRRewriter rewriter(funcOp.getContext());
  for (auto subspanOp : subspanOps) {
    auto newSubspanOp = rewriteStorageBufferSubspanOp(rewriter, subspanOp);
    replaceMemrefUsesAndPropagateType(rewriter, subspanOp.getLoc(), subspanOp,
                                      newSubspanOp);
  }

  {
    RewritePatternSet patterns(&getContext());
    populateRemoveDeadMemAllocPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<mlir::OperationPass<func::FuncOp>>
createSPIRVEraseStorageBufferStaticShapePass() {
  return std::make_unique<EraseStorageBufferStaticShapePass>();
}

}  // namespace iree_compiler
}  // namespace mlir
