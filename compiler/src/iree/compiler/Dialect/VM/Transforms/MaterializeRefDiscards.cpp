// Copyright 2025 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// TODO: This pass inserts discards based on SSA value liveness, but discards
// operate on registers after allocation. When two values share a register
// (non-overlapping intervals), a discard for the earlier value can kill the
// later value. Currently fixed by extending live intervals to include discards
// (LiveIntervals.cpp), but this wastes registers. The correct fix is to insert
// discards post-regalloc based on register liveness.

#include "iree/compiler/Dialect/VM/Analysis/ValueLiveness.h"
#include "iree/compiler/Dialect/VM/IR/VMOps.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::VM {

#define GEN_PASS_DEF_MATERIALIZEREFDISCARDSPASS
#include "iree/compiler/Dialect/VM/Transforms/Passes.h.inc"

// Materializes vm.discard.refs operations at ref death points.
// Uses edge-based placement for clean semantics:
// - For refs dying on control flow edges, insert discards on those edges
// - For refs dying mid-block (last use within block), insert after last use
// - For unused refs (no uses), insert immediately after definition/block entry
class MaterializeRefDiscardsPass
    : public IREE::VM::impl::MaterializeRefDiscardsPassBase<
          MaterializeRefDiscardsPass> {
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      if (failed(processFunction(funcOp))) {
        return signalPassFailure();
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Escaping Ref Detection
  //===--------------------------------------------------------------------===//

  // Collects all refs that escape the function (via return).
  // These must NOT be discarded within the function.
  llvm::DenseSet<Value> collectEscapingRefs(FuncOp funcOp) {
    llvm::DenseSet<Value> escaping;
    for (Block &block : funcOp.getBlocks()) {
      for (Operation &op : block) {
        if (auto returnOp = dyn_cast<IREE::VM::ReturnOp>(&op)) {
          for (Value operand : returnOp.getOperands()) {
            if (isa<IREE::VM::RefType>(operand.getType())) {
              escaping.insert(operand);
            }
          }
        }
      }
    }
    return escaping;
  }

  //===--------------------------------------------------------------------===//
  // Terminator Operand Analysis
  //===--------------------------------------------------------------------===//

  // Returns true if value is a MOVE operand of a terminator (but not
  // forwarded). MOVE operands transfer ownership to the callee/op, so we must
  // NOT discard. Note: Forwarded operands (branch args) may also be marked as
  // MOVE, but forwarding is handled separately by isForwardedOnEdge - this
  // function only identifies true MOVE semantics (like vm.call.yieldable args
  // to callee).
  bool isTerminatorMoveOperand(Value value, Operation *terminator) {
    auto refMoveOp = dyn_cast<IREE::VM::RefMoveInterface>(terminator);
    if (!refMoveOp) {
      return false;
    }

    // Check if value is forwarded to any successor - if so, it's not a "pure"
    // MOVE to callee, it's a forward to successor block.
    if (auto branchOp = dyn_cast<BranchOpInterface>(terminator)) {
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        auto operands = branchOp.getSuccessorOperands(i);
        if (llvm::is_contained(operands.getForwardedOperands(), value)) {
          return false; // Forwarded, not a callee MOVE
        }
      }
    }

    // Check if value is a MOVE operand (callee takes ownership)
    for (OpOperand &operand : terminator->getOpOperands()) {
      if (operand.get() == value && isa<IREE::VM::RefType>(value.getType()) &&
          refMoveOp.isRefOperandMovable(operand.getOperandNumber())) {
        return true;
      }
    }
    return false;
  }

  // Returns true if value is used by the terminator (any operand position).
  bool isTerminatorOperand(Value value, Operation *terminator) {
    for (OpOperand &operand : terminator->getOpOperands()) {
      if (operand.get() == value) {
        return true;
      }
    }
    return false;
  }

  //===--------------------------------------------------------------------===//
  // Edge-Based Discard Insertion
  //===--------------------------------------------------------------------===//

  // Returns true if value is passed as a branch argument on the edge
  // Pred->Succ. Such values get MOVE semantics and shouldn't be discarded.
  bool isForwardedOnEdge(Value value, Block *pred, Block *succ) {
    Operation *terminator = pred->getTerminator();
    auto branchOp = dyn_cast<BranchOpInterface>(terminator);
    if (!branchOp) {
      return false;
    }

    for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
      if (terminator->getSuccessor(i) == succ) {
        auto operands = branchOp.getSuccessorOperands(i);
        if (llvm::is_contained(operands.getForwardedOperands(), value)) {
          return true;
        }
      }
    }
    return false;
  }

  // Inserts a discard on the edge Pred->Succ for the given values.
  // For values that are terminator operands, we must insert AFTER the
  // terminator (in the successor), never before it.
  void insertDiscardOnEdge(OpBuilder &builder, Block *pred, Block *succ,
                           ValueRange values, Location loc) {
    Operation *terminator = pred->getTerminator();
    unsigned numPredSuccessors = terminator->getNumSuccessors();
    unsigned numSuccPredecessors = std::distance(
        succ->getPredecessors().begin(), succ->getPredecessors().end());

    // Check if any value is a terminator operand (non-MOVE, non-forwarded).
    // Such values require discard AFTER the terminator executes.
    bool hasTerminatorOperand = false;
    for (Value value : values) {
      if (isTerminatorOperand(value, terminator)) {
        hasTerminatorOperand = true;
        break;
      }
    }

    // Only use before-terminator insertion if no values are terminator
    // operands.
    if (numPredSuccessors == 1 && !hasTerminatorOperand) {
      // Single successor, no terminator operands: insert before terminator.
      builder.setInsertionPoint(terminator);
      IREE::VM::DiscardRefsOp::create(builder, loc, values);
    } else if (numSuccPredecessors == 1) {
      // Single predecessor: insert at start of successor.
      builder.setInsertionPointToStart(succ);
      IREE::VM::DiscardRefsOp::create(builder, loc, values);
    } else {
      // Critical edge: need to split.
      // Create a new block between pred and succ.
      Block *newBlock = new Block();
      newBlock->insertBefore(succ);

      // Get operands from terminator for the edge being split.
      Operation *terminator = pred->getTerminator();
      auto branchOp = cast<BranchOpInterface>(terminator);

      // Find the successor index for this edge.
      unsigned succIndex = 0;
      for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
        if (terminator->getSuccessor(i) == succ) {
          succIndex = i;
          break;
        }
      }

      // Get the operands being passed to succ on this edge.
      SuccessorOperands succOperands = branchOp.getSuccessorOperands(succIndex);
      SmallVector<Value> operandValues(succOperands.getForwardedOperands());
      unsigned producedCount = succOperands.getProducedOperandCount();

      // Add block arguments to newBlock to receive the forwarded operands.
      for (Value operand : operandValues) {
        newBlock->addArgument(operand.getType(), operand.getLoc());
      }

      // Add block arguments for produced operands (e.g., vm.call.yieldable
      // results). These are created by the terminator at runtime and must be
      // forwarded through the new block to the original successor.
      for (unsigned i = 0; i < producedCount; ++i) {
        // Produced operands come after forwarded operands in succ's arguments.
        Type type = succ->getArgument(operandValues.size() + i).getType();
        newBlock->addArgument(type, loc);
      }

      // Update predecessor's terminator to go to new block instead of succ.
      // The operands stay the same - they'll now be passed to newBlock.
      terminator->setSuccessor(newBlock, succIndex);

      // Insert discard in new block.
      builder.setInsertionPointToStart(newBlock);
      IREE::VM::DiscardRefsOp::create(builder, loc, values);

      // Add unconditional branch to original successor, forwarding block args.
      SmallVector<Value> branchOperands;
      for (BlockArgument arg : newBlock->getArguments()) {
        branchOperands.push_back(arg);
      }
      IREE::VM::BranchOp::create(builder, loc, succ, branchOperands);
    }
  }

  //===--------------------------------------------------------------------===//
  // Main Processing
  //===--------------------------------------------------------------------===//

  LogicalResult processFunction(FuncOp funcOp) {
    // Skip empty functions.
    if (funcOp.getBlocks().empty()) {
      return success();
    }

    // Compute liveness information.
    ValueLiveness liveness;
    if (failed(liveness.recalculate(funcOp))) {
      return failure();
    }
    llvm::DenseSet<Value> escapingRefs = collectEscapingRefs(funcOp);

    OpBuilder builder(funcOp.getContext());

    // Collect all refs in the function in deterministic order.
    // Walk blocks and operations in order and insert into SetVector, which
    // maintains insertion order for deterministic iteration.
    llvm::SetVector<Value> allRefs;
    for (Block &block : funcOp.getBlocks()) {
      for (BlockArgument arg : block.getArguments()) {
        if (isa<IREE::VM::RefType>(arg.getType())) {
          allRefs.insert(arg);
        }
      }
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          if (isa<IREE::VM::RefType>(result.getType())) {
            allRefs.insert(result);
          }
        }
      }
    }

    // Phase 1: Edge-based discards.
    // Collect refs dying on each edge, then insert batched discards.
    // This includes:
    // a) Refs live-out from pred but not live-in to succ (and not forwarded)
    // b) Refs forwarded on SOME edges but not ALL (partial-edge deaths)
    //
    // For case (b), consider: vm.cond_br %c, ^loop(%ref), ^exit
    // The ref is forwarded to ^loop but not to ^exit. It needs a discard on
    // the ^exit edge. However, %ref won't be in liveOuts because it's only
    // used as a branch operand (forwarded transfer).

    // Collect refs dying on each edge (pred, succ pair).
    // Use a vector to maintain deterministic ordering.
    SmallVector<std::tuple<Block *, Block *, SmallVector<Value>>> edgeDiscards;
    llvm::DenseMap<std::pair<Block *, Block *>, size_t> edgeToIndex;

    for (Block &block : funcOp.getBlocks()) {
      Operation *terminator = block.getTerminator();
      for (Block *succ : block.getSuccessors()) {
        auto succLiveIns = liveness.getBlockLiveIns(succ);
        auto liveOuts = liveness.getBlockLiveOuts(&block);

        SmallVector<Value> dyingRefs;
        for (Value ref : allRefs) {
          if (escapingRefs.count(ref)) {
            continue;
          }

          // Check if ref should be discarded on this edge.
          bool isInLiveOuts = llvm::is_contained(liveOuts, ref);
          bool isForwardedOnAny = false;
          if (auto branchOp = dyn_cast<BranchOpInterface>(terminator)) {
            for (unsigned i = 0; i < terminator->getNumSuccessors(); ++i) {
              if (isForwardedOnEdge(ref, &block, terminator->getSuccessor(i))) {
                isForwardedOnAny = true;
                break;
              }
            }
          }

          // Skip if ref is neither in liveOuts nor forwarded on any edge.
          if (!isInLiveOuts && !isForwardedOnAny) {
            continue;
          }

          // Skip if ref is live-in to successor.
          if (llvm::is_contained(succLiveIns, ref)) {
            continue;
          }

          // Skip if ref is forwarded on this specific edge.
          if (isForwardedOnEdge(ref, &block, succ)) {
            continue;
          }

          // Skip if ref is a MOVE operand of the terminator.
          // MOVE operands transfer ownership to the callee, so we must NOT
          // discard them - the callee takes responsibility for the ref.
          if (isTerminatorMoveOperand(ref, terminator)) {
            continue;
          }

          // Ref dies on this edge.
          dyingRefs.push_back(ref);
        }

        if (!dyingRefs.empty()) {
          edgeDiscards.push_back({&block, succ, std::move(dyingRefs)});
        }
      }
    }

    // Insert batched discards for each edge.
    for (auto &[pred, succ, refs] : edgeDiscards) {
      Location loc = pred->getTerminator()->getLoc();
      insertDiscardOnEdge(builder, pred, succ, refs, loc);
    }

    // Phase 2: Mid-block discards.
    // Collect refs dying mid-block (last use is not at block end and ref is not
    // live-out), grouped by insertion point for batching.
    for (Block &block : funcOp.getBlocks()) {
      // Group refs by their insertion point (the op after which to insert).
      // Key is the op, value is the list of refs dying after that op.
      SmallVector<std::pair<Operation *, SmallVector<Value>>> midBlockDiscards;
      llvm::DenseMap<Operation *, size_t> opToIndex;

      for (Operation &op : block) {
        if (isa<IREE::VM::DiscardRefsOp>(&op)) {
          continue;
        }

        for (OpOperand &operand : op.getOpOperands()) {
          Value value = operand.get();
          if (!isa<IREE::VM::RefType>(value.getType())) {
            continue;
          }

          // Skip escaping refs.
          if (escapingRefs.count(value)) {
            continue;
          }

          // Check if this is the last use and value doesn't escape via
          // live-outs.
          if (liveness.isLastValueUse(value, &op, operand.getOperandNumber())) {
            auto liveOuts = liveness.getBlockLiveOuts(&block);
            if (!llvm::is_contained(liveOuts, value)) {
              // For terminators, don't insert mid-block discards for their
              // operands. Terminator ref operands fall into three categories:
              // 1. Forwarded (branch args) → successor block takes ownership
              // 2. MOVE (vm.call.yieldable args) → callee takes ownership
              // 3. Non-MOVE, non-forwarded → Phase 1 inserts edge discards in
              //    successors (after terminator executes)
              // In all cases, a mid-block discard would kill the value before
              // the terminator can use it.
              if (op.hasTrait<OpTrait::IsTerminator>()) {
                continue;
              }

              // Skip refs that are MOVE operands of RefMoveInterface
              // operations. When an operand is movable and this is its last
              // use, the MOVE bit will be set by the register allocator and
              // ownership transfers to the operation (e.g., vm.call,
              // vm.call.variadic). Inserting a discard would be incorrect as
              // the ref is consumed by the operation.
              if (auto refMoveOp = dyn_cast<IREE::VM::RefMoveInterface>(&op)) {
                if (refMoveOp.isRefOperandMovable(operand.getOperandNumber())) {
                  continue;
                }
              }

              // Group by insertion point.
              auto it = opToIndex.find(&op);
              if (it == opToIndex.end()) {
                opToIndex[&op] = midBlockDiscards.size();
                midBlockDiscards.push_back({&op, {value}});
              } else {
                midBlockDiscards[it->second].second.push_back(value);
              }
            }
          }
        }
      }

      // Insert batched mid-block discards.
      for (auto &[op, values] : midBlockDiscards) {
        if (op->hasTrait<OpTrait::IsTerminator>()) {
          builder.setInsertionPoint(op);
        } else {
          builder.setInsertionPointAfter(op);
        }
        IREE::VM::DiscardRefsOp::create(builder, op->getLoc(), values);
      }
    }

    // Phase 3: Unused refs.
    // Insert discards for refs that have no uses at all.
    for (Block &block : funcOp.getBlocks()) {
      // Unused block arguments.
      SmallVector<Value> unusedBlockArgs;
      for (BlockArgument arg : block.getArguments()) {
        if (!isa<IREE::VM::RefType>(arg.getType())) {
          continue;
        }
        if (arg.use_empty() && !escapingRefs.count(arg)) {
          unusedBlockArgs.push_back(arg);
        }
      }
      if (!unusedBlockArgs.empty()) {
        builder.setInsertionPointToStart(&block);
        IREE::VM::DiscardRefsOp::create(builder, block.front().getLoc(),
                                        unusedBlockArgs);
      }

      // Unused results - batch by defining op.
      SmallVector<std::pair<Operation *, SmallVector<Value>>> unusedResults;
      llvm::DenseMap<Operation *, size_t> opToResultIndex;
      for (Operation &op : block) {
        for (Value result : op.getResults()) {
          if (!isa<IREE::VM::RefType>(result.getType())) {
            continue;
          }
          if (result.use_empty() && !escapingRefs.count(result)) {
            auto it = opToResultIndex.find(&op);
            if (it == opToResultIndex.end()) {
              opToResultIndex[&op] = unusedResults.size();
              unusedResults.push_back({&op, {result}});
            } else {
              unusedResults[it->second].second.push_back(result);
            }
          }
        }
      }
      for (auto &[op, results] : unusedResults) {
        builder.setInsertionPointAfter(op);
        IREE::VM::DiscardRefsOp::create(builder, op->getLoc(), results);
      }
    }

    return success();
  }
};

} // namespace mlir::iree_compiler::IREE::VM
