// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-codegen-fission-transfer-ops-in-control-flow"

#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE << "]: ")

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_FISSIONTRANSFEROPSINCONTROLFLOWPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

namespace {

// Note: this should exists in mlir/lib/Dialect/IR/GPUDialect.cpp
bool isPrivateAddressSpace(Attribute memorySpace) {
  if (!memorySpace)
    return false;
  if (auto gpuAttr = llvm::dyn_cast<gpu::AddressSpaceAttr>(memorySpace))
    return gpuAttr.getValue() == gpu::GPUDialect::getPrivateAddressSpace();
  return false;
}

void replaceIterationVariable(Operation *op, Value iterArg, Value constant) {
  for (auto &operand : op->getOpOperands()) {
    if (operand.get() == iterArg) {
      operand.set(constant);
    }
  }
}

SetVector<Operation *> collectBackwardSliceInControlFlow(
    Operation *op, Operation *parentOp) {
  BackwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) {
    return parentOp == op->getParentOp();
  };
  SetVector<Operation *> slice;
  getBackwardSlice(op, &slice, options);
  return slice;
}

void cloneSliceIntoLoop(IRRewriter &rewriter, SetVector<Operation *> &slice, scf::ForOp &newLoop, IRMapping &mapping) {  
  rewriter.setInsertionPointToStart(newLoop.getBody());  
  for (Operation *op : slice) {  
    rewriter.clone(*op, mapping);  
  }  
}  
  
scf::ForOp createNewLoop(IRRewriter &rewriter, scf::ForOp forOp, Location loc) {  
  return rewriter.create<scf::ForOp>(  
      loc, forOp.getLowerBound(), forOp.getUpperBound(),  
      forOp.getStep(), forOp.getRegionIterArgs());  
}  
  
memref::AllocaOp createAlloca(IRRewriter &rewriter, vector::TransferWriteOp writeOp) {  
  auto allocaType = cast<MemRefType>(writeOp.getBase().getType());  
  auto allocaTypeNoStride =  
      MemRefType::Builder(allocaType.getShape(), allocaType.getElementType());  
  return rewriter.create<memref::AllocaOp>(writeOp.getLoc(), allocaTypeNoStride);  
}  
  
void splitTransferOpsFromControlFlow(IRRewriter &rewriter,  
                                     vector::TransferReadOp readOp,  
                                     vector::TransferWriteOp writeOp,  
                                     scf::ForOp forOp) {  
  rewriter.setInsertionPoint(forOp);
  memref::AllocaOp alloca = createAlloca(rewriter, writeOp);  
  
  SetVector<Operation *> readSlice =  
      collectBackwardSliceInControlFlow(readOp, forOp);  
 
  // Read loop  
  scf::ForOp readLoop = createNewLoop(rewriter, forOp, readOp.getLoc());  
  IRMapping mapping;  
  mapping.map(forOp.getInductionVar(), readLoop.getInductionVar());  // Map induction variables  
  cloneSliceIntoLoop(rewriter, readSlice, readLoop, mapping);  
  
  Operation *lastRead = rewriter.clone(*readOp, mapping);  
  auto newTransferWrite = writeOp.clone();    
  newTransferWrite->setOperand(0, lastRead->getResult(0));  
  newTransferWrite->setOperand(1, alloca);  

  Operation *terminator = readLoop.getBody()->getTerminator();
  readLoop.getBody()->getOperations().insert(
      terminator->getIterator(), newTransferWrite);
  readLoop.dump();
  
  // Write loop  
  scf::ForOp writeLoop = createNewLoop(rewriter, forOp, readOp.getLoc());  
  rewriter.setInsertionPoint(writeLoop);  
  
  auto newReadOp = readOp.clone();  
  newReadOp->setOperand(0, alloca);  
  writeLoop.getBody()->getOperations().insert(  
      writeLoop.getBody()->begin(), newReadOp);  

  SetVector<Operation *> writeSlice =  
      collectBackwardSliceInControlFlow(writeOp, forOp);
  IRMapping writeMapping;
  writeMapping.map(forOp.getInductionVar(), writeLoop.getInductionVar());
  cloneSliceIntoLoop(rewriter, writeSlice, writeLoop, writeMapping);
  auto lastWrite =
    rewriter.clone(*writeOp, writeMapping);
  lastWrite->setOperand(0, newReadOp->getResult(0));
  writeLoop.dump();
  
  //for (Operation &op : writeLoop.getBody()->getOperations()) {  
  //  for (auto &operand : op.getOpOperands()) {  
  //    if (operand.get() == readOp.getResult()) {  
  //      operand.set(newReadOp);  
  //    }  
  //  }  
  //}  
  
  // Erase original ops  
  //rewriter.eraseOp(readOp);  
  //rewriter.eraseOp(forOp);  
}  


//void splitTransferOpsFromControlFlow(IRRewriter &rewriter,
//                                     vector::TransferReadOp readOp,
//                                     vector::TransferWriteOp writeOp,
//                                     scf::ForOp forOp) {
//  rewriter.setInsertionPoint(forOp);
//  auto allocaType = cast<MemRefType>(writeOp.getBase().getType());
//  // Get rid of stride and offset information in the alloca type.
//  auto allocaTypeNoStride = MemRefType::Builder(allocaType.getShape(), allocaType.getElementType());
//  auto alloca = rewriter.create<memref::AllocaOp>(  
//              readOp.getLoc(), allocaTypeNoStride
//              );
//  SetVector<Operation *> readSlice =
//      collectBackwardSliceInControlFlow(readOp, forOp);
//  auto readLoop = rewriter.create<scf::ForOp>(
//      readOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
//      forOp.getStep(), forOp.getRegionIterArgs());
//  IRMapping mapping;
//  mapping.map(forOp.getInductionVar(), readLoop.getInductionVar());
//  Operation *terminator = readLoop.getBody()->getTerminator();
//
//  rewriter.setInsertionPointToStart(readLoop.getBody());
//  for (Operation *op : readSlice) {
//    rewriter.clone(*op, mapping);
//  }
//  Operation *lastRead = rewriter.clone(*readOp, mapping);
//  //rewriter.setInsertionPoint(terminator);
//  auto newTransferWrite = writeOp.clone();
//  newTransferWrite->setOperand(0, lastRead->getResult(0));
//  // Replace the base of the transfer_write with the alloca.
//  newTransferWrite->setOperand(1, alloca);
//  readLoop.getBody()->getOperations().insert(
//      terminator->getIterator(), newTransferWrite);
//  //auto endOp = 
//  //auto writeLoop = forOp.clone();
//  rewriter.setInsertionPoint(forOp);
//  auto newReadOp = readOp.clone();
//  newReadOp->setOperand(0, alloca);
//  forOp.getBody()->getOperations().insert(
//      forOp.getBody()->begin(), newReadOp);
//  for (Operation &op : forOp.getBody()->getOperations()) {
//    for (auto &operand : op.getOpOperands()) {
//      if (operand.get() == readOp.getResult()) {
//        operand.set(newReadOp);
//      }
//    }
//  }
//  rewriter.eraseOp(readOp);
//  //DBGS() << "Read loop: " << "\n";
//  //readLoop.dump();
//  //DBGS() << "Write loop: " << "\n";
//  //forOp.dump();
//
//  //forOp.erase();
//}

void hoistTransferReadAndDependencies(scf::ForOp forOp) {
  OpBuilder builder(forOp.getContext());
  //std::vector<Operation *> dependencies;
  DenseSet<Operation *> visited;
  Value iterArg = forOp.getInductionVar();

  //SetVector<Operation *> slice;
  SetVector<Operation *> dependencies;
  BackwardSliceOptions options;
  options.inclusive = true;
  options.filter = [&](Operation *op) {
    return forOp == op->getParentOp();
  };
  // Find transfer_read operations and relevant dependencies within the loop.
  forOp.walk([&](vector::TransferReadOp readOp) {
    getBackwardSlice(readOp.getOperation(), &dependencies, options);
  });

  builder.setInsertionPoint(forOp);
  Value zero = builder.create<arith::ConstantIndexOp>(forOp.getLoc(), 0);

  // Hoist dependencies in the order they were collected.
  for (Operation *dep : dependencies) {
    replaceIterationVariable(dep, iterArg, zero);
    dep->moveBefore(forOp);
  }
}

struct FissionTarget{
  Operation *parent;
  vector::TransferReadOp readOp;
  vector::TransferWriteOp writeOp;
};

static FailureOr<FissionTarget> processReadOp(vector::TransferReadOp readOp) {
  auto parentOp = readOp->getParentOp();
  //auto parentOp = forOp;
  if (!parentOp || !isa<scf::ForOp, scf::IfOp, scf::WhileOp>(parentOp)) {
    return failure();
  }

  auto base = readOp.getBase();
  auto addrspace = cast<MemRefType>(base.getType()).getMemorySpace();
  if (gpu::GPUDialect::isWorkgroupMemoryAddressSpace(addrspace) ||
      isPrivateAddressSpace(addrspace)) {
    return failure();
  }

  ForwardSliceOptions options;
  options.inclusive = false;
  options.filter = [&](Operation *op) {
    return parentOp == op->getParentOp();
  };
  SetVector<Operation *> slice;
  getForwardSlice(readOp.getOperation(), &slice, options);

  bool hasWriteOp = false;
  FissionTarget fissionTarget;
  for (Operation *op : slice) {
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto writeBase = writeOp.getBase();
      auto writeAddrspace =
          cast<MemRefType>(writeBase.getType()).getMemorySpace();
      if (isPrivateAddressSpace(writeAddrspace)) {
        // Only consider transfer_write ops that are in the same address space.
        fissionTarget = {parentOp, readOp, writeOp};
        hasWriteOp = true;
        // DBGS() << "Fissioning target readOp: " << readOp
        //        << " and writeOp: " << writeOp
        //        //<< " in parent operation: " <<
        //        cast<scf::ForOp>(target.parent)
        //        << "\n";
      }
    }
  }
  if (!hasWriteOp) {
    return failure();
  }
  return fissionTarget;
}

static FailureOr<SmallVector<FissionTarget>> populateFissionTargets(
  scf::ForOp forOp) {

  SmallVector<FissionTarget> fissionTargets;
  forOp->walk([&](Operation *op) {
    if (op->getParentOp() != forOp) {
      return;
    }

    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      auto result = processReadOp(readOp);
      if (failed(result)) {
        return;
      }
      fissionTargets.push_back(result.value());
      for (const FissionTarget &target : fissionTargets) {
        DBGS() << "Fissioning target readOp: " << target.readOp
               << " and writeOp: "
               << target.writeOp
               //       << " in parent operation: " <<
               //       cast<scf::ForOp>(target.parent)
               << "\n";
      }
    }
  });
  return fissionTargets;
}

struct FissionTransferOpsInControlFlowPass final
    : impl::FissionTransferOpsInControlFlowPassBase<
          FissionTransferOpsInControlFlowPass> {
public:
  void runOnOperation() override;
};

} // namespace

void FissionTransferOpsInControlFlowPass::runOnOperation() {
  FunctionOpInterface funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());

  SmallVector<scf::ForOp> loops;
  funcOp.walk([&loops](scf::ForOp forOp) { loops.push_back(forOp); });

  SmallVector<FissionTarget> fissionTargets;
  for (scf::ForOp forOp : loops) {
    auto result = populateFissionTargets(forOp);
    if (failed(result)) {
      continue;
    }
    fissionTargets.insert(fissionTargets.end(), result.value().begin(),
                          result.value().end());
  }

  for (const FissionTarget &target : fissionTargets) {
    if (isa<scf::ForOp>(target.parent)) {
      splitTransferOpsFromControlFlow(rewriter, target.readOp, target.writeOp,
                                      cast<scf::ForOp>(target.parent));
    }
  }
}

} // namespace mlir::iree_compiler
