// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Generalizes executables that differ only in constant values so that
// DeduplicateExecutables can merge them: within a group of executables whose
// dispatch functions are structurally equal up to the values of index and f32
// constants and of the static offsets of their tensor loads and stores (into
// globals),
// every such value that is not the same in all of the group becomes a new
// operand of the dispatch, passed by each call site. A typical case is a
// per-layer KV cache read or write at the layer's constant offset into one
// buffer: one executable per layer becomes one executable for all of them, the
// offset a push constant.

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/bit.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::Flow {

#define GEN_PASS_DEF_GENERALIZEEXECUTABLESPASS
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h.inc"

namespace {

// A value of an executable that may be generalized: an arith.constant of index
// or f32 type (e.g. a layer's quantization scale), or one static offset of a
// dispatch tensor load or store. (Integer constants of other types compute
// values, e.g. a position's wrap-around and clamping; a backend may need them
// constant, so they stay in the signature.)
struct Slot {
  Operation *op;
  int offset = -1; // -1: the constant; else the offset's position
  int64_t value;   // (an f32's bits)
  Type type;
};

struct Signature {
  OwningOpRef<Operation *> normalizedFunc;
  llvm::hash_code hash = {};
  SmallVector<Slot> slots;
};

static bool isScalarGeneralizableConstant(Operation *op) {
  auto c = dyn_cast<arith::ConstantOp>(op);
  if (!c) {
    return false;
  }
  if (isa<IndexType>(c.getType())) {
    return isa<IntegerAttr>(c.getValue());
  }
  return c.getType().isF32() && isa<FloatAttr>(c.getValue());
}

static int64_t constantBits(arith::ConstantOp c) {
  if (auto f = dyn_cast<FloatAttr>(c.getValue())) {
    return int64_t(f.getValue().bitcastToAPInt().getZExtValue());
  }
  return cast<IntegerAttr>(c.getValue()).getValue().getSExtValue();
}

template <typename OpT, typename NormalizedOpT>
static void normalizeOffsets(OpT op, NormalizedOpT normalizedOp,
                             SmallVector<Slot> &slots) {
  SmallVector<int64_t> normalizedOffsets;
  for (auto [i, v] : llvm::enumerate(op.getStaticOffsets())) {
    if (ShapedType::isDynamic(v)) {
      normalizedOffsets.push_back(v);
    } else {
      slots.push_back(
          {op.getOperation(), int(i), v, IndexType::get(op.getContext())});
      normalizedOffsets.push_back(0);
    }
  }
  normalizedOp.setStaticOffsets(normalizedOffsets);
}

// Clones |funcOp| and normalizes the values that this pass may promote. The
// clone can then be compared with MLIR's structural equivalence utilities;
// the original values and operations are retained as slots in walk order.
static Signature buildSignature(func::FuncOp funcOp) {
  Signature sig;
  sig.normalizedFunc = funcOp->clone();
  SmallVector<Operation *> originalOps;
  SmallVector<Operation *> normalizedOps;
  funcOp.getBody().walk<WalkOrder::PreOrder>(
      [&](Operation *op) { originalOps.push_back(op); });
  cast<func::FuncOp>(*sig.normalizedFunc)
      .getBody()
      .walk<WalkOrder::PreOrder>(
          [&](Operation *op) { normalizedOps.push_back(op); });
  assert(originalOps.size() == normalizedOps.size());
  for (auto [op, normalizedOp] : llvm::zip(originalOps, normalizedOps)) {
    if (isScalarGeneralizableConstant(op)) {
      auto c = cast<arith::ConstantOp>(op);
      sig.slots.push_back({op, -1, constantBits(c), c.getType()});
      Attribute value = c.getType().isF32()
                            ? Attribute(FloatAttr::get(c.getType(), 0.0))
                            : Attribute(IntegerAttr::get(c.getType(), 0));
      normalizedOp->setAttr("value", value);
    } else if (auto load =
                   dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(op)) {
      normalizeOffsets(
          load, cast<IREE::TensorExt::DispatchTensorLoadOp>(normalizedOp),
          sig.slots);
    } else if (auto store =
                   dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(op)) {
      normalizeOffsets(
          store, cast<IREE::TensorExt::DispatchTensorStoreOp>(normalizedOp),
          sig.slots);
    }
  }
  func::FuncOp normalizedFunc = cast<func::FuncOp>(*sig.normalizedFunc);
  sig.hash =
      llvm::hash_value(normalizedFunc.getFunctionType().getAsOpaquePointer());
  normalizedFunc.getBody().walk([&](Operation *op) {
    sig.hash = llvm::hash_combine(sig.hash,
                                  OperationEquivalence::computeHash(
                                      op, OperationEquivalence::ignoreHashValue,
                                      OperationEquivalence::ignoreHashValue,
                                      OperationEquivalence::IgnoreLocations));
  });
  return sig;
}

static bool areEquivalent(const Signature &lhs, const Signature &rhs) {
  func::FuncOp lhsFunc = cast<func::FuncOp>(*lhs.normalizedFunc);
  func::FuncOp rhsFunc = cast<func::FuncOp>(*rhs.normalizedFunc);
  return lhsFunc.getFunctionType() == rhsFunc.getFunctionType() &&
         OperationEquivalence::isRegionEquivalentTo(
             &lhsFunc.getBody(), &rhsFunc.getBody(),
             OperationEquivalence::IgnoreLocations);
}

template <typename OpT>
static void makeOffsetDynamic(OpT op, int pos, Value v) {
  SmallVector<OpFoldResult> mixed = op.getMixedOffsets();
  mixed[pos] = v;
  SmallVector<int64_t> staticOffsets;
  SmallVector<Value> dynamicOffsets;
  dispatchIndexOpFoldResults(mixed, dynamicOffsets, staticOffsets);
  op.setStaticOffsets(staticOffsets);
  op.getOffsetsMutable().assign(dynamicOffsets);
}

// Whether |v| is a global's contents (a load of it, possibly through dispatches
// that update it in place): its binding then has the same offset at every call
// site, so a generalized offset into it is the only dynamic part of the
// address. An offset into a transient tensor is left alone (its binding offset
// may be dynamic already).
static bool isGlobalBacked(Value v) {
  for (int depth = 0; depth < 4096; ++depth) {
    Operation *op = v.getDefiningOp();
    if (!op) {
      return false;
    }
    if (isa<IREE::Util::GlobalLoadOpInterface>(op)) {
      return true;
    }
    auto tied = dyn_cast<IREE::Util::TiedOpInterface>(op);
    if (!tied) {
      return false;
    }
    Value operand = tied.getTiedResultOperand(v);
    if (!operand) {
      return false;
    }
    v = operand;
  }
  return false;
}

struct Member {
  IREE::Flow::ExecutableOp executableOp;
  func::FuncOp funcOp;
  SmallVector<IREE::Flow::DispatchOp> callers;
  Signature sig;
};

} // namespace

class GeneralizeExecutablesPass
    : public IREE::Flow::impl::GeneralizeExecutablesPassBase<
          GeneralizeExecutablesPass> {
  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    // The executables with one export and one function, and their callers.
    llvm::MapVector<StringRef, Member> members;
    for (auto executableOp : moduleOp.getOps<IREE::Flow::ExecutableOp>()) {
      auto exports = executableOp.getOps<IREE::Flow::ExecutableExportOp>();
      auto innerModule = executableOp.getInnerModule();
      if (!innerModule || !llvm::hasSingleElement(exports)) {
        continue;
      }
      auto funcs = innerModule.getOps<func::FuncOp>();
      if (!llvm::hasSingleElement(funcs)) {
        continue;
      }
      members[executableOp.getSymName()] = {
          executableOp, *funcs.begin(), {}, {}};
    }
    moduleOp.walk([&](IREE::Flow::DispatchOp dispatchOp) {
      SmallVector<SymbolRefAttr> refs =
          llvm::to_vector(dispatchOp.getEntryPointRefs());
      if (refs.size() != 1) {
        for (auto ref : refs) {
          members.erase(ref.getRootReference().getValue());
        }
        return;
      }
      auto it = members.find(refs.front().getRootReference().getValue());
      if (it != members.end()) {
        it->second.callers.push_back(dispatchOp);
      }
    });

    // Hash normalized functions into candidate buckets, then confirm full
    // structural equivalence. Hash equality alone is never sufficient.
    llvm::MapVector<size_t, SmallVector<size_t>> groupsByHash;
    SmallVector<SmallVector<Member *>> groups;
    for (auto &entry : members) {
      Member &m = entry.second;
      if (m.callers.empty()) {
        continue;
      }
      m.sig = buildSignature(m.funcOp);
      auto &candidateGroups = groupsByHash[size_t(m.sig.hash)];
      auto it = llvm::find_if(candidateGroups, [&](size_t group) {
        return areEquivalent(groups[group].front()->sig, m.sig);
      });
      if (it == candidateGroups.end()) {
        candidateGroups.push_back(groups.size());
        groups.push_back({&m});
      } else {
        groups[*it].push_back(&m);
      }
    }

    for (SmallVector<Member *> &group : groups) {
      if (group.size() < 2) {
        continue;
      }
      size_t nslots = group.front()->sig.slots.size();
      SmallVector<size_t> varying;
      for (size_t s = 0; s < nslots; ++s) {
        int64_t v = group.front()->sig.slots[s].value;
        for (Member *m : group) {
          if (m->sig.slots[s].value != v) {
            varying.push_back(s);
            break;
          }
        }
      }
      if (varying.empty()) {
        continue; // already identical: deduplication merges them
      }
      // an offset may be generalized only into a global's contents
      bool ok = true;
      for (size_t s : varying) {
        for (Member *m : group) {
          Slot &slot = m->sig.slots[s];
          if (slot.offset < 0) {
            continue;
          }
          Value t;
          if (auto load =
                  dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(slot.op)) {
            t = load.getSource();
          } else {
            t = cast<IREE::TensorExt::DispatchTensorStoreOp>(slot.op)
                    .getTarget();
          }
          auto arg = dyn_cast<BlockArgument>(t);
          unsigned nops = m->callers.front().getArguments().size();
          if (!arg || arg.getOwner() != &m->funcOp.front() ||
              arg.getArgNumber() >= nops) {
            ok = false;
            break;
          }
          for (IREE::Flow::DispatchOp caller : m->callers) {
            ok &= isGlobalBacked(caller.getArguments()[arg.getArgNumber()]);
          }
          if (!ok) {
            break;
          }
        }
        if (!ok) {
          break;
        }
      }
      if (!ok) {
        continue;
      }
      // the new operands go after each member's operands (before its untied
      // results): all of a member's callers must pass the same number
      bool consistent = llvm::all_of(group, [](Member *m) {
        size_t n = m->callers.front().getArguments().size();
        return llvm::all_of(m->callers, [&](IREE::Flow::DispatchOp c) {
          return c.getArguments().size() == n;
        });
      });
      if (!consistent) {
        continue;
      }
      for (Member *m : group) {
        func::FuncOp funcOp = m->funcOp;
        unsigned pos = m->callers.front().getArguments().size();
        SmallVector<Value> newArgs;
        for (size_t s : varying) {
          Slot &slot = m->sig.slots[s];
          (void)funcOp.insertArgument(pos + newArgs.size(), slot.type, {},
                                      funcOp.getLoc());
          Value arg = funcOp.getArgument(pos + newArgs.size());
          newArgs.push_back(arg);
          if (slot.offset < 0) {
            slot.op->getResult(0).replaceAllUsesWith(arg);
          } else if (auto load =
                         dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(
                             slot.op)) {
            makeOffsetDynamic(load, slot.offset, arg);
          } else {
            makeOffsetDynamic(
                cast<IREE::TensorExt::DispatchTensorStoreOp>(slot.op),
                slot.offset, arg);
          }
        }
        for (size_t s : varying) {
          Slot &slot = m->sig.slots[s];
          if (slot.offset < 0 && slot.op->use_empty()) {
            slot.op->erase();
          }
        }
        for (IREE::Flow::DispatchOp caller : m->callers) {
          OpBuilder b(caller);
          SmallVector<Value> values;
          for (size_t s : varying) {
            Slot &slot = m->sig.slots[s];
            TypedAttr attr =
                slot.type.isF32()
                    ? TypedAttr(b.getF32FloatAttr(
                          llvm::bit_cast<float>(uint32_t(slot.value))))
                    : TypedAttr(b.getIntegerAttr(slot.type, slot.value));
            values.push_back(
                arith::ConstantOp::create(b, caller.getLoc(), slot.type, attr));
          }
          caller.getArgumentsMutable().append(values);
        }
      }
    }
  }
};

} // namespace mlir::iree_compiler::IREE::Flow
