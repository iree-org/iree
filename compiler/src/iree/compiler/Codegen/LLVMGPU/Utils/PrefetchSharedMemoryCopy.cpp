// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/MathExtras.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "iree-codegen-llvmgpu-prefetch-shared-memory-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

llvm::cl::opt<bool> clLLVMGPUEnablePrefetchingIGLP0(
    "iree-llvmgpu-prefetching-use-iglp-0",
    llvm::cl::desc(
        "Interleave DS and MFMA instructions for small GEMM kernels."),
    llvm::cl::init(false));

llvm::cl::opt<bool> clLLVMGPUEnablePrefetchingIGLP1(
    "iree-llvmgpu-prefetching-use-iglp-1",
    llvm::cl::desc(
        "Interleave DS and MFMA instructions for single wave GEMM kernels."),
    llvm::cl::init(false));

namespace mlir::iree_compiler {

namespace {

/// Returns true if the given `memrefType` has a memory space meant for global
/// memory, that is, the default numeric address space 0 or a HAL descriptor
/// type address space.
bool hasDefaultOrHALAddressSpace(MemRefType memrefType) {
  using IREE::HAL::DescriptorType;

  Attribute addrSpace = memrefType.getMemorySpace();
  if (!addrSpace)
    return true;

  if (auto intAttr = dyn_cast<IntegerAttr>(memrefType.getMemorySpace())) {
    return intAttr.getInt() == 0;
  }

  if (auto dtAttr = dyn_cast<IREE::HAL::DescriptorTypeAttr>(addrSpace)) {
    return dtAttr.getValue() == DescriptorType::StorageBuffer ||
           dtAttr.getValue() == DescriptorType::UniformBuffer;
  }

  return false;
}

/// Returns the underlying index if the given value is a constant index.
std::optional<int64_t> getConstantIndex(Value value) {
  if (!isa<IndexType>(value.getType()))
    return std::nullopt;

  APInt val;
  if (!matchPattern(value, m_ConstantInt(&val)))
    return std::nullopt;

  return val.getSExtValue();
}

static void setPriority(Location loc, OpBuilder &builder, int priority) {
  auto asmDialectAttr =
      LLVM::AsmDialectAttr::get(builder.getContext(), LLVM::AsmDialect::AD_ATT);
  std::string str = "s_setprio " + std::to_string(priority);
  const char *asmStr = str.c_str();
  const char *constraints = "";
  builder.create<LLVM::InlineAsmOp>(
      loc,
      /*resultTypes=*/TypeRange(), /*operands=*/ValueRange(),
      /*asm_string=*/asmStr, constraints, /*has_side_effects=*/true,
      /*is_align_stack=*/false, /*asm_dialect=*/asmDialectAttr,
      /*operand_attrs=*/ArrayAttr());
}

class LoopPrefetcher {
public:
  /// Creates an instance that plans the given scf.for |op| to be ready for
  /// prefetching. Returns failure if unable to support the given |op|.
  static FailureOr<LoopPrefetcher> get(scf::ForOp op) {
    if (!op.getOps<scf::ForOp>().empty()) {
      LDBG("Loop prefetcher does not support nested loops yet");
      return failure();
    }

    LoopPrefetcher prefetcher;
    prefetcher.mapping = SmallVector<IRMapping>(4);
    prefetcher.forOp = op;
    prefetcher.lb = prefetcher.ub = prefetcher.step = 0;
    prefetcher.singleStage = true;

    if (failed(prefetcher.initializeLoopInfo())) {
      LDBG("Failed to initialize loop info (unsupported loop)");
      return failure();
    }

    if (failed(prefetcher.initializeStages())) {
      LDBG("Failed to initialize stage info (unsupported loop)");
      return failure();
    }

    return prefetcher;
  }

  // Emits the prologue before the main pipelined loop and returns the read
  // results to be passed to the main loop as initial loop carried values, and
  // their useages by corresponding writes in the main loop.
  std::tuple<SmallVector<Value>, SmallVector<Value>>
  emitPrologue(RewriterBase &rewriter) {
    Location loc = forOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, lb + step);
    SmallVector<Value> iterArgs;
    SmallVector<Value> readResults;
    SmallVector<Value> writeArgs;

    if (singleStage) {
      // If we only prefetch one step ahead, we can directly write in the
      // prologue and use the shared memory to communicate data instead of the
      // loop carried values.
      // Read (0)
      emitRead(mapping[0], rewriter, zero);
      // Write(0)
      emitWrite(mapping[0], rewriter, zero);
      // Read (1)
      iterArgs = emitRead(mapping[1], rewriter, one);
      emitBarrier(loc, rewriter);
      return {iterArgs, writeArgs};
    }

    // Read(0).
    iterArgs = emitRead(mapping[0], rewriter, zero);
    // Read(1).
    readResults = emitRead(mapping[1], rewriter, one);
    llvm::append_range(iterArgs, readResults);

    // Collect the values to be used as write args.
    for (Operation *op : readStage) {
      if (auto transferReadOp = dyn_cast<vector::TransferReadOp>(op)) {
        for (Operation *user : transferReadOp.getResult().getUsers()) {
          if (auto writeOp = dyn_cast<vector::TransferWriteOp>(user)) {
            writeArgs.push_back(writeOp.getVector());
          }
        }
      }
    }

    return {iterArgs, writeArgs};
  }

  /// Emits the main pipelined loop structure.
  scf::ForOp createKernelLoop(RewriterBase &rewriter,
                              SmallVector<Value> &newIterArgs,
                              SmallVector<Value> &writeArgs) {
    Location loc = forOp.getLoc();
    int64_t newUpperBound = (ub - 2 * step);
    auto newUb = rewriter.create<arith::ConstantIndexOp>(loc, newUpperBound);

    // Keep original iter args and then add some for what's being loaded to
    // registers.
    auto iterArgs = llvm::to_vector_of<Value>(forOp.getInitArgs());
    llvm::append_range(iterArgs, newIterArgs);

    Value newStep = forOp.getStep();
    if (!singleStage) {
      newStep = rewriter.create<arith::AddIOp>(loc, newStep, newStep);
    }
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                newUb, newStep, iterArgs);

    // When there are no iter args, the loop body terminator will be created.
    // Since we always create it below, remove the terminator if it was created.
    if (!newForOp.getBody()->empty())
      rewriter.eraseOp(newForOp.getBody()->getTerminator());

    if (singleStage)
      return newForOp;

    SmallVector<Value> targetValues(writeArgs.size());
    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getRegionIterArg(i + 1);

    createWriteMappings(writeArgs, targetValues, mapping[0]);

    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getRegionIterArg(i + e + 1);

    createWriteMappings(writeArgs, targetValues, mapping[1]);
    return newForOp;
  }

  /// Emits contents into the main pipelined loop structure.
  void createKernel(RewriterBase &rewriter, scf::ForOp newForOp) {
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    Location loc = forOp.getLoc();

    if (clLLVMGPUEnablePrefetchingIGLP0) {
      // Set iglp opt 0 if flag enabled.
      StringAttr intrinsic = rewriter.getStringAttr("llvm.amdgcn.iglp.opt");
      Value iglpOpt = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 0);
      rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange(), intrinsic, ValueRange(iglpOpt));
    } else if (clLLVMGPUEnablePrefetchingIGLP1) {
      // Set iglp opt 1 if flag enabled.
      StringAttr intrinsic = rewriter.getStringAttr("llvm.amdgcn.iglp.opt");
      Value iglpOpt = rewriter.create<LLVM::ConstantOp>(loc, rewriter.getI32Type(), 1);
      rewriter.create<LLVM::CallIntrinsicOp>(loc, TypeRange(), intrinsic, ValueRange(iglpOpt));
    }

    Value indVar = newForOp.getInductionVar();
    Value increment = rewriter.create<arith::ConstantIndexOp>(loc, step);
    Value iPlusOne = rewriter.create<arith::AddIOp>(loc, indVar, increment);
    Value iPlusTwo = rewriter.create<arith::AddIOp>(loc, iPlusOne, increment);
    Value iPlusThree = rewriter.create<arith::AddIOp>(loc, iPlusTwo, increment);

    for (int i = 0; i < 3; ++i) {
      for (auto [idx, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
        mapping[i].map(arg, newForOp.getRegionIterArgs()[idx]);
      }
      if (i == 1) {
        // Map write vectors to iter args of new for loop.
        int index = forOp.getRegionIterArgs().size();
        for (Operation *op : writeStage) {
          if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
            mapping[i].map(writeOp.getVector(),
                           newForOp.getRegionIterArgs()[index++]);
          }
        }
      }
    }

    SmallVector<Value> readRegisters, moreRegisters;
    if (singleStage) {
      emitCompute(mapping[0], rewriter, indVar);
      emitBarrier(loc, rewriter);
      setPriority(loc, rewriter, 1);
      emitWrite(mapping[1], rewriter, iPlusOne);
      readRegisters = emitRead(mapping[2], rewriter, iPlusTwo);
      setPriority(loc, rewriter, 0);
      emitBarrier(loc, rewriter);
      updateYieldWithMultipleMappings(mapping[2], mapping[0], readRegisters,
                                      rewriter);
      return;
    }

    emitWrite(mapping[0], rewriter, indVar);
    readRegisters = emitRead(mapping[2], rewriter, iPlusTwo);
    emitBarrier(loc, rewriter);
    auto computeResults = emitCompute(mapping[0], rewriter, indVar);
    mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
    emitBarrier(loc, rewriter);
    emitWrite(mapping[1], rewriter, iPlusOne);
    moreRegisters = emitRead(mapping[3], rewriter, iPlusThree);
    emitBarrier(loc, rewriter);
    emitCompute(mapping[0], rewriter, iPlusOne);
    emitBarrier(loc, rewriter);
    readRegisters.append(moreRegisters.begin(), moreRegisters.end());
    updateYield(mapping[0], readRegisters, rewriter);
  }

  // Emits the epilogue after the main pipelined loop and returns the final
  // results to replace the original loop results main loop.
  SmallVector<Value> emitEpilogue(RewriterBase &rewriter, scf::ForOp newForOp,
                                  SmallVector<Value> &writeArgs) {
    rewriter.setInsertionPointAfter(newForOp);
    Location loc = forOp.getLoc();
    Value nMinusTwo =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 2 * step);
    Value nMinusOne =
        rewriter.create<arith::ConstantIndexOp>(loc, ub - 1 * step);

    // Map iter_args to results of newForOp.
    for (unsigned i = 0, e = forOp->getNumResults(); i != e; ++i) {
      mapping[0].map(forOp.getRegionIterArg(i), newForOp.getResult(i));
    }

    // Map write inputs to results of newForLoop.
    int index = forOp->getNumResults();
    for (Operation *op : writeStage) {
      if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
        mapping[0].map(writeOp.getVector(), newForOp.getResult(index++));
      }
    }

    if (singleStage) {
      SmallVector<Value> computeResults;
      // Compute(N-2)
      computeResults = emitCompute(mapping[0], rewriter, nMinusTwo);
      mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
      emitBarrier(loc, rewriter);
      // Write(N-1)
      emitWrite(mapping[0], rewriter, nMinusOne);
      emitBarrier(loc, rewriter);
      // Compute(N-1)
      return emitCompute(mapping[0], rewriter, nMinusOne);
    }

    SmallVector<Value> targetValues(writeArgs.size());
    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getResult(i + 1);

    createWriteMappings(writeArgs, targetValues, mapping[2]);

    for (size_t i = 0, e = writeArgs.size(); i != e; ++i)
      targetValues[i] = newForOp.getResult(i + e + 1);

    createWriteMappings(writeArgs, targetValues, mapping[3]);

    emitWrite(mapping[2], rewriter, nMinusTwo);
    emitBarrier(loc, rewriter);
    SmallVector<Value> computeResults =
        emitCompute(mapping[0], rewriter, nMinusTwo);
    mapping[0].map(forOp.getRegionIterArg(0), computeResults[0]);
    emitBarrier(loc, rewriter);
    emitWrite(mapping[3], rewriter, nMinusOne);
    emitBarrier(loc, rewriter);
    computeResults = emitCompute(mapping[0], rewriter, nMinusOne);
    return computeResults;
  }

private:
  LogicalResult initializeLoopInfo() {
    std::optional<int64_t> lbCst = getConstantIndex(forOp.getLowerBound());
    std::optional<int64_t> ubCst = getConstantIndex(forOp.getUpperBound());
    std::optional<int64_t> stepCst = getConstantIndex(forOp.getStep());
    if (!lbCst || !ubCst || !stepCst)
      return failure();

    lb = *lbCst;
    ub = *ubCst;
    step = *stepCst;

    int64_t numIters = mlir::ceilDiv(ub - lb, step);
    if (numIters <= 2)
      return failure();

    return success();
  }

  void getValueDependencies(Operation *op, DenseSet<Operation *> &dependencies,
                            bool noTransferReads = false) {
    if (!op || dependencies.contains(op)) {
      return;
    }

    if (noTransferReads && isa<vector::TransferReadOp>(op)) {
      return;
    }

    if (!forOp->isProperAncestor(op)) {
      return;
    }

    dependencies.insert(op);
    op->walk([&](Operation *nested) {
      for (Value val : nested->getOperands()) {
        getValueDependencies(val.getDefiningOp(), dependencies,
                             noTransferReads);
      }
    });
  }

  // We only support loops whose bodies can be divided into 3 stages (read,
  // write, compute). If there are any remaning ops with side effects (except
  // for gpu.barrier), the loop is not supported.
  LogicalResult initializeStages() {
    DenseSet<Operation *> readDependencies;
    DenseSet<Operation *> writeDependencies;
    DenseSet<Operation *> computeDependencies;

    for (Operation &op : forOp.getBody()->getOperations()) {
      if (auto read = dyn_cast<vector::TransferReadOp>(op)) {
        MemRefType type = dyn_cast<MemRefType>(read.getSource().getType());
        if (!gpu::GPUDialect::hasWorkgroupMemoryAddressSpace(type))
          getValueDependencies(read, readDependencies);
      } else if (auto write = dyn_cast<vector::TransferWriteOp>(op)) {
        getValueDependencies(write, writeDependencies,
                             /*noTransferReads=*/true);
      } else if (auto compute = dyn_cast<scf::YieldOp>(op)) {
        getValueDependencies(compute, computeDependencies);
      }
    }

    // Restore the original order.
    for (auto &op : forOp.getBody()->getOperations()) {
      bool hasStage = false;
      if (readDependencies.contains(&op)) {
        readStage.push_back(&op);
        hasStage = true;
      }
      if (writeDependencies.contains(&op)) {
        writeStage.push_back(&op);
        hasStage = true;
      }
      if (computeDependencies.contains(&op)) {
        computeStage.push_back(&op);
        hasStage = true;
      }

      // Ops with a stage will be cloned over to the new for op, while barriers
      // will be re-written.
      if (!hasStage && !isa<gpu::BarrierOp>(op)) {
        if (!isPure(&op)) {
          LDBG("Found a non-pure loop body op not assigned to any stage "
               "(unsupported loop): "
               << op);
          return failure();
        }
      }
    }

    // Modify yield op to take all read results as operands
    SmallVector<Value> readResults;
    for (Operation *op : readStage) {
      if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
        readResults.push_back(readOp.getResult());
      }
    }
    IRRewriter rewriter(forOp.getContext());
    for (auto [idx, op] : llvm::enumerate(computeStage)) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        SmallVector<Value> newOperands = yieldOp->getOperands();
        llvm::append_range(newOperands, readResults);
        rewriter.setInsertionPoint(yieldOp);
        Operation *newYield =
            rewriter.create<scf::YieldOp>(op->getLoc(), newOperands);
        computeStage[idx] = newYield;
        break;
      }
    }

    LLVM_DEBUG({
      // Stages cannot have overlapping operations.
      llvm::dbgs() << "--- Read Stage ---\n";
      for (Operation *op : readStage)
        llvm::dbgs() << *op << "\n";
      llvm::dbgs() << "--- Write Stage ---\n";
      for (Operation *op : writeStage)
        llvm::dbgs() << *op << "\n";
      llvm::dbgs() << "--- Compute Stage ---\n";
      for (Operation *op : computeStage)
        llvm::dbgs() << *op << "\n";
    });

    return success();
  }

  /// Clones |op| and call |callback| on the cloned op's oeprands as well as any
  /// operands of nested ops that 1) aren't defined within the new op or 2) are
  /// block arguments.
  static Operation *
  cloneAndUpdateOperands(RewriterBase &rewriter, Operation *op,
                         function_ref<void(OpOperand *newOperand)> callback) {
    Operation *clone = rewriter.clone(*op);
    for (OpOperand &operand : clone->getOpOperands())
      callback(&operand);
    return clone;
  }

  /// Creates all read stage ops for a loop iteration with |rewriter| and maps
  /// the original loop induction variable to |iv| in |mapping|.
  SmallVector<Value> emitRead(IRMapping &mapping, RewriterBase &rewriter,
                              Value iv) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), iv);

    SmallVector<Value> results;
    for (Operation *op : readStage) {
      // Clone the current read stage op and updates all its operands to
      // reference newly created ops.
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      if (isa<vector::TransferReadOp>(newOp)) {
        llvm::append_range(results, newOp->getResults());
      }

      // Update read stage op results mapping.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }
    return results;
  }

  /// Creates all write stage ops for a loop iteration with |rewriter| and maps
  /// the original loop induction variable to |iv| in |mapping|.
  void emitWrite(IRMapping &mapping, RewriterBase &rewriter, Value iv) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), iv);

    for (Operation *op : writeStage) {
      // Clone the current read stage op and updates all its operands to
      // reference newly created ops.
      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });

      // If a mapping for any results already exists, move on, otherwise,
      // add a new mapping.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        if (!mapping.contains(op->getResult(i))) {
          mapping.map(op->getResult(i), newOp->getResult(i));
        }
      }
    }
  }

  /// Creates a gpu.barrier op with |rewriter|.
  void emitBarrier(Location loc, RewriterBase &rewriter) {
    rewriter.create<gpu::BarrierOp>(loc);
  }

  /// Creates all compute stage ops for a loop iteration with |rewriter| and
  /// maps the original loop induction variable to |iv| in |mapping|.
  SmallVector<Value> emitCompute(IRMapping &mapping, RewriterBase &rewriter,
                                 Value indVar) {
    // Map the original loop induction variable to |iv| for later op rewrites.
    mapping.map(forOp.getInductionVar(), indVar);

    SmallVector<Value> results;
    for (Operation *op : computeStage) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        // On yield, return the operands of the yield converted.
        results =
            llvm::map_to_vector<4>(yieldOp.getOperands(), [&](Value operand) {
              return mapping.lookup(operand);
            });
        break;
      }

      Operation *newOp =
          cloneAndUpdateOperands(rewriter, op, [&](OpOperand *newOperand) {
            if (mapping.contains(newOperand->get())) {
              newOperand->set(mapping.lookup(newOperand->get()));
            }
          });
      results = newOp->getResults();

      // Map compute operations to new compute operations.
      for (unsigned i = 0, e = op->getNumResults(); i != e; ++i) {
        mapping.map(op->getResult(i), newOp->getResult(i));
      }
    }

    return results;
  }

  /// Maps values in |srcValues| to |targetValues| in |mapping|.
  static void createWriteMappings(SmallVector<Value> &srcValues,
                                  SmallVector<Value> &targetValues,
                                  IRMapping &mapping) {
    for (auto [src, tgt] : llvm::zip_equal(srcValues, targetValues)) {
      mapping.map(src, tgt);
    }
  }

  void updateYieldWithMultipleMappings(IRMapping &mapping0, IRMapping &mapping1,
                                       SmallVector<Value> &readValues,
                                       RewriterBase &rewriter) {
    for (Operation *op : computeStage) {
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        cloneAndUpdateOperands(rewriter, yield, [&](OpOperand *newOperand) {
          if (mapping0.contains(newOperand->get())) {
            newOperand->set(mapping0.lookup(newOperand->get()));
          }
          if (mapping1.contains(newOperand->get())) {
            newOperand->set(mapping1.lookup(newOperand->get()));
          }
        });

        break;
      }
    }
  }

  void updateYield(IRMapping &mapping, SmallVector<Value> &readValues,
                   RewriterBase &rewriter) {
    for (Operation *op : computeStage) {
      if (auto yield = dyn_cast<scf::YieldOp>(op)) {
        cloneAndUpdateOperands(rewriter, yield, [&](OpOperand *newOperand) {
          if (mapping.contains(newOperand->get())) {
            newOperand->set(mapping.lookup(newOperand->get()));
          }
        });

        break;
      }
    }
  }

private:
  // Mapping for each pipelined stage.
  SmallVector<IRMapping, 4> mapping;
  // The original scf.for loop to prefetch shared memory copy from.
  scf::ForOp forOp;
  // Original static loop range and step.
  int64_t lb, ub, step;
  // Whether we only prefetch one single step ahead.
  bool singleStage;

  // Ops in the original scf.for loop that belongs to different classes.
  SmallVector<Operation *> readStage;
  SmallVector<Operation *> writeStage;
  SmallVector<Operation *> computeStage;
};

} // namespace

FailureOr<scf::ForOp> prefetchSharedMemoryCopy(RewriterBase &rewriter,
                                               scf::ForOp forOp) {
  rewriter.setInsertionPoint(forOp);

  auto prefetcherOr = LoopPrefetcher::get(forOp);
  if (failed(prefetcherOr))
    return failure();
  LoopPrefetcher &prefetcher = *prefetcherOr;

  auto [iterArgs, writeArgs] = prefetcher.emitPrologue(rewriter);
  scf::ForOp newForOp =
      prefetcher.createKernelLoop(rewriter, iterArgs, writeArgs);
  prefetcher.createKernel(rewriter, newForOp);

  SmallVector<Value> results =
      prefetcher.emitEpilogue(rewriter, newForOp, writeArgs);
  for (int i = results.size() - 1; i >= forOp->getNumResults(); --i)
    results.pop_back();
  rewriter.replaceOp(forOp, results);
  return newForOp;
}

} // namespace mlir::iree_compiler
