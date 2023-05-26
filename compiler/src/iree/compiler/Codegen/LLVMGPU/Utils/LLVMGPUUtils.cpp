// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"

using namespace mlir;

#define DEBUG_TYPE "llvm-gpu-utils"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir {
namespace iree_compiler {

static bool isContiguousStore(Operation* write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write)) {
    if (!transferWrite.getPermutationMap().isMinorIdentity() ||
        !transferWrite.isDimInBounds(0) || transferWrite.getMask()) {
      return false;
    }
    return true;
  }
  if (isa<vector::StoreOp>(write)) {
    return true;
  }
  return false;
}

static bool isContiguousRead(Operation* read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read)) {
    if (!transferRead.isDimInBounds(0) ||
        !transferRead.getPermutationMap().isMinorIdentity()) {
      return false;
    }
    return true;
  }
  if (isa<vector::LoadOp>(read)) {
    return true;
  }
  return false;
}

static Value getMemrefOperand(Operation* op) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(op)) {
    return transferWrite.getSource();
  }
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(op)) {
    return transferRead.getSource();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(op)) {
    return storeOp.getBase();
  }
  if (auto loadOp = dyn_cast<vector::LoadOp>(op)) {
    return loadOp.getBase();
  }
  return Value();
}

struct MaskResult {
  vector::CreateMaskOp maskOp;
  vector::ExtractOp maybeExtractOp;
};
static MaskResult getMask(Operation* op) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(op);
  if (!transferRead || !transferRead.getMask()) return MaskResult{};
  vector::ExtractOp maybeExtractOp =
      transferRead.getMask().getDefiningOp<vector::ExtractOp>();
  auto maskOp =
      maybeExtractOp
          ? maybeExtractOp.getVector().getDefiningOp<vector::CreateMaskOp>()
          : transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  if (maybeExtractOp) {
    if (maybeExtractOp.getPosition().size() + 1 !=
        llvm::cast<VectorType>(maskOp->getResultTypes().front()).getRank()) {
      LDBG("----mask through extract unexpected position size -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    if (maybeExtractOp.getPosition().size() != 1) {
      LDBG("----only mask through 2-D -> 1-D extract supported atm -> Skip: "
           << maybeExtractOp);
      return MaskResult{};
    }
    LDBG("----mask through extract: " << maybeExtractOp);
  }
  return MaskResult{maskOp, maybeExtractOp};
}

static Value getMaskValue(RewriterBase& rewriter, Operation* op) {
  MaskResult maskResult = getMask(op);
  if (!maskResult.maskOp) return Value();
  Value count = maskResult.maskOp->getOperands().back();
  vector::ExtractOp maybeExtractOp = maskResult.maybeExtractOp;
  if (maybeExtractOp) {
    assert(maybeExtractOp.getPosition().size() == 1 && "expected single pos");
    int64_t sliceNum =
        llvm::cast<IntegerAttr>(maybeExtractOp.getPosition()[0]).getInt();
    // TODO: to support >2-D mask + extract, and all the cmp.
    Location loc = op->getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value cmp = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt,
        rewriter.create<arith::ConstantIndexOp>(loc, sliceNum),
        maskResult.maskOp->getOperands().front());
    count = rewriter.create<arith::SelectOp>(loc, cmp, count, zero);
  }
  return count;
}

static Value getValueStored(Operation* writeOp) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(writeOp)) {
    return transferWrite.getValue();
  }
  if (auto storeOp = dyn_cast<vector::StoreOp>(writeOp)) {
    return storeOp.getValueToStore();
  }
  return Value();
}

static Operation::operand_range getIndices(Operation* op) {
  if (auto vectorReadOp = dyn_cast<vector::LoadOp>(op))
    return vectorReadOp.getIndices();
  if (auto vectorStoreOp = dyn_cast<vector::StoreOp>(op))
    return vectorStoreOp.getIndices();
  if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op))
    return transferReadOp.getIndices();
  if (auto transferWriteOp = dyn_cast<vector::TransferWriteOp>(op))
    return transferWriteOp.getIndices();
  llvm_unreachable("unsupported op type");
}

void createAsyncGroups(RewriterBase& rewriter, func::FuncOp funcOp,
                       bool useMMASync) {
  LDBG("Start asyncGroups: useMMASync=" << useMMASync);
  llvm::SmallSetVector<Operation*, 16> copyToSharedMem;
  // Look for all the copy that can be converted to async copy ops.
  funcOp.walk([&](Operation* writeOp) {
    if (!isContiguousStore(writeOp)) {
      return WalkResult::advance();
    }
    LDBG("--candidate writeOp: " << writeOp);
    Value vectorVal = getValueStored(writeOp);
    if (llvm::cast<VectorType>(vectorVal.getType()).getRank() != 1) {
      LDBG("----writeOp is not an inbounds 1-D minor identity -> Skip");
      return WalkResult::advance();
    }
    Value memrefOperand = getMemrefOperand(writeOp);
    if (!hasSharedMemoryAddressSpace(
            llvm::cast<MemRefType>(memrefOperand.getType()))) {
      LDBG("----address space is not workgroup -> Skip");
      return WalkResult::advance();
    }
    Operation* readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp)) {
      LDBG("----no readOp defining the writeOp -> Skip");
      return WalkResult::advance();
    }

    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (transferRead.getMask()) {
        auto paddingCst =
            transferRead.getPadding().getDefiningOp<arith::ConstantFloatOp>();
        if (!paddingCst || !paddingCst.value().isZero()) {
          LDBG("----read padding value is not 0.f -> Skip");
          return WalkResult::advance();
        }
        auto maskResult = getMask(transferRead);
        if (!maskResult.maskOp) {
          LDBG("----read mask is not a vector.create_mask op -> Skip: "
               << transferRead.getMask());
          return WalkResult::advance();
        }
      }
    }

    VectorType vecType = llvm::cast<VectorType>(vectorVal.getType());
    if (!((vecType.getElementType().isF32() && vecType.getNumElements() <= 4) ||
          (vecType.getElementType().isF16() &&
           vecType.getNumElements() <= 8))) {
      LDBG("----readOp is not (<=4)xf32 or (<=8)xf16 -> Skip");
      return WalkResult::advance();
    }

    LDBG("--writeOp can be made async -> SUCCESS");
    copyToSharedMem.insert(writeOp);
    return WalkResult::advance();
  });

  while (!copyToSharedMem.empty()) {
    SmallVector<Operation*> group;
    Operation* writeOp = *copyToSharedMem.begin();
    // Start a group with the first write.
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation* nextNode = writeOp;
    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation* readOp = nextNode;
        Value memrefOperand = getMemrefOperand(readOp);
        if (!hasSharedMemoryAddressSpace(
                llvm::cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }
    // emit the group.
    SmallVector<Value> tokens;
    for (Operation* writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = getValueStored(writeOp);
      Operation* readOp = vectorVal.getDefiningOp();
      Value storeBase = getMemrefOperand(writeOp);
      Value loadBase = getMemrefOperand(readOp);
      Value mask = getMaskValue(rewriter, readOp);
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(),
          nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()), storeBase,
          getIndices(writeOp), loadBase, getIndices(readOp),
          rewriter.getIndexAttr(
              llvm::cast<VectorType>(vectorVal.getType()).getNumElements()),
          mask,
          /*bypassL1=*/useMMASync ? rewriter.getUnitAttr() : UnitAttr());
      tokens.push_back(token);
    }
    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        funcOp.getLoc(), nvgpu::DeviceAsyncTokenType::get(funcOp.getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(funcOp.getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation* writeOp : group) rewriter.eraseOp(writeOp);
  }
}

void reorderTranspose(IRRewriter& rewriter, func::FuncOp funcOp) {
  SmallVector<vector::TransposeOp> transposeOps;
  funcOp.walk([&](Operation* op) {
    if (auto transposeOp = dyn_cast<vector::TransposeOp>(op)) {
      Operation* definingOp = transposeOp.getVector().getDefiningOp();
      if (OpTrait::hasElementwiseMappableTraits(definingOp)) {
        transposeOps.push_back(transposeOp);
      }
    }
    return WalkResult::advance();
  });

  for (auto transposeOp : transposeOps) {
    OpBuilder::InsertionGuard g(rewriter);
    Operation* op = transposeOp.getVector().getDefiningOp();
    rewriter.setInsertionPoint(op);
    SmallVector<int64_t> perm;
    transposeOp.getTransp(perm);
    SmallVector<Value> transposedOperands;
    for (auto operand : op->getOperands()) {
      Value transposed =
          rewriter.create<vector::TransposeOp>(op->getLoc(), operand, perm);
      transposedOperands.push_back(transposed);
    }
    SmallVector<Type> resultTypes{transposedOperands.front().getType()};
    Operation* newOp =
        rewriter.create(op->getLoc(), op->getName().getIdentifier(),
                        transposedOperands, resultTypes, op->getAttrs());
    rewriter.replaceAllUsesWith(transposeOp.getResult(), newOp->getResult(0));
  }
}

}  // namespace iree_compiler
}  // namespace mlir
