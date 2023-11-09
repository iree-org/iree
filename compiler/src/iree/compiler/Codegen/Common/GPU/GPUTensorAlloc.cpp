// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/LinalgOpInfo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-codegen-gpu-tensor-alloc"

namespace mlir {
namespace iree_compiler {

// For optimal performance we always want to copy 128 bits
static constexpr int copyVectorNumBits = 128;

/// Filter to decide which contract ops need allocations.
static bool contractOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;

  if (!linalg::isaContractionOpInterface(linalgOp)) {
    return false;
  }

  // The workgroup specialization already makes static shapes available for the
  // main tile part and makes the partial tile computation small, so promoting
  // to shared memory for the partial tile actually hurts the performance.
  if (linalgOp.hasDynamicShape())
    return false;

  // Check if the shape is tile-distributable. The leading dimension must be a
  // multiple of the target vector size, which is 128b / the element bit width.
  auto isTileDistributable = [&](Value v) {
    ShapedType ty = llvm::cast<ShapedType>(v.getType());
    unsigned bitWidth = ty.getElementTypeBitWidth();
    int targetVectorSize = copyVectorNumBits / bitWidth;
    return ty.getShape().back() % targetVectorSize == 0;
  };

  if (!llvm::all_of(linalgOp.getDpsInputs(), isTileDistributable)) {
    return false;
  }

  if (!llvm::all_of(linalgOp.getDpsInits(), isTileDistributable)) {
    return false;
  }

  SmallVector<unsigned> dims;
  linalgOp.getParallelDims(dims);
  SmallVector<int64_t> shapes = linalgOp.getStaticLoopRanges();
  // Don't promote vector*matrix kind of case.
  int numNonUnitParallelLoop = 0;
  for (unsigned parallelDim : dims) {
    if (shapes[parallelDim] != 1) {
      numNonUnitParallelLoop++;
    }
  }
  return numNonUnitParallelLoop > 1 && linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

/// Filter to decide which transpose ops need allocations.
static bool transposeOpFilter(Operation *op) {
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp)
    return false;
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);
  return opInfo.isTranspose();
}

namespace {
/// Swaps bufferization.alloc_tensor with the copied linalg op result when the
/// linalg op does not use the output initial value during calculation.
///
/// This converts the following IR:
/// ```
/// %linalg = linalg ... ins(...) outs(...)
/// %val = bufferization.alloc_tensor() copy(%linalg)
/// ```
/// Into
/// ```
/// %alloc = bufferization.alloc_tensor()
/// %val = linalg ... ins(...) outs(%alloc)
/// ```
struct SwapAllocTensorPattern final
    : OpRewritePattern<bufferization::AllocTensorOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(bufferization::AllocTensorOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (!allocOp.getCopy())
      return failure();
    auto linalgOp = allocOp.getCopy().getDefiningOp<linalg::LinalgOp>();
    if (!linalgOp)
      return failure();

    // Make sure we don't use the initial values for the linalg output we are
    // copying during the tensor allocation.
    unsigned resultNumber = cast<OpResult>(allocOp.getCopy()).getResultNumber();
    OpOperand *initOperand = linalgOp.getDpsInitOperand(resultNumber);
    if (linalgOp.payloadUsesValueFromOperand(initOperand))
      return failure();

    rewriter.setInsertionPoint(linalgOp);
    std::optional<Attribute> memorySpace = allocOp.getMemorySpace();
    auto newAllocOp = rewriter.create<bufferization::AllocTensorOp>(
        allocOp.getLoc(), allocOp.getType(), allocOp.getDynamicSizes(),
        /*copy=*/Value(),
        memorySpace ? cast<IntegerAttr>(*memorySpace) : IntegerAttr());
    rewriter.updateRootInPlace(linalgOp, [&]() {
      linalgOp->setOperand(linalgOp.getNumDpsInputs() + resultNumber,
                           newAllocOp);
    });
    rewriter.replaceOp(allocOp, linalgOp->getResult(resultNumber));

    return failure();
  }
};

struct GPUTensorAllocPass : public GPUTensorAllocBase<GPUTensorAllocPass> {
private:
  GPUPromoteSharedMemPattern promoteSharedMemPattern =
      GPUPromoteSharedMemPattern::ContractionOpPattern;

public:
  GPUTensorAllocPass(GPUPromoteSharedMemPattern promoteSharedMemPattern)
      : promoteSharedMemPattern(promoteSharedMemPattern) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, scf::SCFDialect>();
  }
  void runOnOperation() override {
    auto funcOp = getOperation();

    // Tile the reduction first to reduce the alloc size.
    if (failed(
            tileReductionToSerialLoops(funcOp, /*fuseInputProducer=*/true))) {
      return signalPassFailure();
    }

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After tiling to serial loops ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    SmallVector<Operation *> opsToPromote;
    funcOp.walk([&](Operation *op) {
      switch (promoteSharedMemPattern) {
      case GPUPromoteSharedMemPattern::ContractionOpPattern:
        if (contractOpFilter(op))
          opsToPromote.push_back(op);
        break;
      case GPUPromoteSharedMemPattern::TransposeOpPattern:
        if (transposeOpFilter(op))
          opsToPromote.push_back(op);
        break;
      }
    });
    for (Operation *op : opsToPromote) {
      OpBuilder builder(op);
      auto linalgOp = cast<linalg::LinalgOp>(op);
      bufferization::BufferizationOptions options;
      switch (promoteSharedMemPattern) {
      case GPUPromoteSharedMemPattern::ContractionOpPattern:
        // Promote all the input operands
        for (auto operand : linalgOp.getDpsInputOperands()) {
          FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
              builder, op->getLoc(), operand->get(), options);
          if (failed(ret)) {
            return signalPassFailure();
          }
          Value v = ret.value();
          operand->set(v);
        }
        break;

      case GPUPromoteSharedMemPattern::TransposeOpPattern:
        LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

        for (auto operand : opInfo.getTransposeOperands()) {
          FailureOr<Value> ret = bufferization::allocateTensorForShapedValue(
              builder, op->getLoc(), operand->get(), options);
          if (failed(ret)) {
            return signalPassFailure();
          }
          Value v = ret.value();
          operand->set(v);
        }
        break;
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "// --- After promotion ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // Move tensor allocations earlier and use them for linalg init operands
    // when possible. This change cleans up the IR to avoid bufferization
    // creating extra buffers in later stages.
    {
      MLIRContext *context = &getContext();
      RewritePatternSet patterns(context);
      patterns.add<SwapAllocTensorPattern>(context);
      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createGPUTensorAlloc(GPUPromoteSharedMemPattern promoteSharedMemPattern) {
  return std::make_unique<GPUTensorAllocPass>(promoteSharedMemPattern);
}

} // namespace iree_compiler
} // namespace mlir
