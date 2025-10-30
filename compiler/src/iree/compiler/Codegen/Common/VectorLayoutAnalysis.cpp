// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Common/Transforms.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"

#include <cassert>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"

#define DEBUG_TYPE "iree-codegen-vector-layout-analysis"

namespace mlir::iree_compiler {

using namespace IREE::VectorExt;

struct LayoutInfo {
  /// Given a value, propagate its layout information forward through its
  /// users.
  void propagateLayoutForward(Value val);
  /// Given a value, propagate its layout information backward through its
  /// defining operation.
  void propagateLayoutBackward(Value val);

  void setLayoutIfUnset(Value val, VectorLayoutInterface layout) {
    if (!isa<ShapedType>(val.getType())) {
      // Don't set layouts on non-shaped types. This would anyway be an empty
      // layout.
      return;
    }
    if (hasLayout(val)) {
      return;
    }
    layouts[val] = layout;
    forward.push(val);
    backward.push(val);
  }
  void setLayoutOrClone(OpOperand *val, VectorLayoutInterface layout);
  VectorLayoutInterface getLayout(Value val) const {
    return layouts.lookup(val);
  }
  bool hasLayout(Value val) const { return layouts.contains(val); }

  llvm::MapVector<Value, VectorLayoutInterface> layouts;
  std::queue<Value> forward;
  std::queue<Value> backward;
};

void LayoutInfo::propagateLayoutForward(Value val) {
  LDBG() << "Propagating layout forward for value: " << val << "\n";
  VectorLayoutInterface layout = getLayout(val);
  for (OpOperand &use : val.getUses()) {
    unsigned operandIdx = use.getOperandNumber();
    Operation *user = use.getOwner();

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      Value arg = forOp.getTiedLoopRegionIterArg(&use);
      Value result = forOp.getTiedLoopResult(&use);
      setLayoutIfUnset(arg, layout);
      setLayoutIfUnset(result, layout);
      continue;
    }

    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      Operation *parentOp = yieldOp->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
        Value arg = forOp.getRegionIterArg(operandIdx);
        Value result = forOp->getResult(operandIdx);
        setLayoutIfUnset(arg, layout);
        setLayoutIfUnset(result, layout);
        continue;
      }
      if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
        Value thenArg = ifOp.getThenRegion().getArgument(operandIdx);
        Value elseArg = ifOp.getElseRegion().getArgument(operandIdx);
        Value result = ifOp->getResult(operandIdx);
        setLayoutIfUnset(thenArg, layout);
        setLayoutIfUnset(elseArg, layout);
        setLayoutIfUnset(result, layout);
        continue;
      }
    }

    if (auto yieldOp = dyn_cast<vector::YieldOp>(user)) {
      Operation *parentOp = cast<vector::MaskOp>(yieldOp->getParentOp());
      Value result = parentOp->getResult(operandIdx);
      setLayoutIfUnset(result, layout);
    }

    if (OpTrait::hasElementwiseMappableTraits(user)) {
      for (OpResult result : user->getOpResults()) {
        setLayoutIfUnset(result, layout);
      }
      continue;
    }

    if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(user)) {
      if (multiReduce.getSource() == val) {
        if (auto maskOp =
                dyn_cast<vector::MaskOp>(multiReduce->getParentOp())) {
          // We shouldn't have to do this... but vector.mask is badly designed
          // and there is no mapping from the mask operand to the operation.
          // TODO: Open vector.mask before vector distribute.
          setLayoutOrClone(&maskOp.getMaskMutable(), layout);
        }
        SmallVector<bool> reductionMask = multiReduce.getReductionMask();
        VectorLayoutInterface reduceLayout = layout.project(reductionMask);
        setLayoutIfUnset(multiReduce.getResult(), reduceLayout);
        continue;
      }
      if (multiReduce.getAcc() == val) {
        setLayoutIfUnset(multiReduce.getResult(), layout);
        continue;
      }
    }

    if (auto transpose = dyn_cast<vector::TransposeOp>(user)) {
      if (transpose.getVector() == val) {
        setLayoutIfUnset(transpose.getResult(),
                         layout.permute(transpose.getPermutation()));
        continue;
      }
    }

    if (auto contract = dyn_cast<vector::ContractionOp>(user)) {
      if (contract.getAcc() == val) {
        setLayoutIfUnset(contract.getResult(), layout);
        continue;
      }
      if (contract.getLhs() == val || contract.getRhs() == val) {
        if (contract->hasAttr("iree.amdgpu.mma")) {
          // Intrinsic ops have fixed layouts, do not try to infer them through
          // maps.
          // TODO: Move to iree_gpu.multi_mma ops.
          continue;
        }
        if (auto maskOp = dyn_cast<vector::MaskOp>(contract->getParentOp())) {
          // We shouldn't have to do this... but vector.mask is badly designed
          // and there is no mapping from the mask operand to the operation.
          // TODO: Open vector.mask before vector distribute.
          AffineMap map = contract.getMatchingIndexingMap(&use);
          if (map.isPermutation()) {
            setLayoutOrClone(&maskOp.getMaskMutable(),
                             layout.apply(inversePermutation(map)));
          }
        }
        // If lhs, rhs layout is known, infer result layout.
        VectorLayoutInterface lhsLayout = getLayout(contract.getLhs());
        VectorLayoutInterface rhsLayout = getLayout(contract.getRhs());
        if (lhsLayout && rhsLayout) {
          AffineMap lhsMap = contract.getIndexingMapsArray()[0];
          AffineMap rhsMap = contract.getIndexingMapsArray()[1];
          AffineMap resMap = contract.getIndexingMapsArray()[2];
          VectorLayoutInterface resLayout = lhsLayout.getRecombinedLayout(
              {lhsLayout, rhsLayout}, {lhsMap, rhsMap}, resMap);
          setLayoutIfUnset(contract.getResult(), resLayout);
        }
        continue;
      }
    }

    if (auto gather = dyn_cast<vector::GatherOp>(user)) {
      setLayoutIfUnset(gather.getResult(), layout);
      continue;
    }

    if (auto write = dyn_cast<vector::TransferWriteOp>(user)) {
      if (!write.getMask()) {
        continue;
      }
      OpOperand &mask = write.getMaskMutable()[0];
      AffineMap maskMap =
          inversePermutation(compressUnusedDims(write.getPermutationMap()));
      setLayoutOrClone(&mask, layout.apply(maskMap));
      continue;
    }
  }
}

void LayoutInfo::propagateLayoutBackward(Value val) {
  LDBG() << "Propagating layout backward for value: " << val << "\n";
  VectorLayoutInterface layout = getLayout(val);
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    Operation *parent = val.getParentBlock()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      OpOperand *yielded = forOp.getTiedLoopYieldedValue(blockArg);
      OpOperand *init = forOp.getTiedLoopInit(blockArg);
      setLayoutOrClone(yielded, layout);
      setLayoutOrClone(init, layout);
    }
    return;
  }

  Operation *defOp = val.getDefiningOp();
  if (OpTrait::hasElementwiseMappableTraits(defOp)) {
    for (OpOperand &operand : defOp->getOpOperands()) {
      setLayoutOrClone(&operand, layout);
    }
    return;
  }

  if (auto toLayout = dyn_cast<ToLayoutOp>(defOp)) {
    setLayoutOrClone(&toLayout.getInputMutable(), layout);
    return;
  }

  if (auto multiReduce = dyn_cast<vector::MultiDimReductionOp>(defOp)) {
    setLayoutOrClone(&multiReduce.getAccMutable(), layout);
    return;
  }

  if (auto transpose = dyn_cast<vector::TransposeOp>(defOp)) {
    setLayoutOrClone(
        &transpose.getVectorMutable(),
        layout.permute(invertPermutationVector(transpose.getPermutation())));
    return;
  }

  if (auto broadcast = dyn_cast<vector::BroadcastOp>(defOp)) {
    // Ensure that there are no broadcasted unit dims as we do not know how to
    // handle them as of now.
    assert(broadcast.computeBroadcastedUnitDims().empty() &&
           "Stretching in broadcasting not implemented yet.");
    if (!isa<VectorType>(broadcast.getSourceType())) {
      return;
    }
    int64_t numBroadcastedDims =
        broadcast.getResultVectorType().getRank() -
        cast<VectorType>(broadcast.getSourceType()).getRank();
    SmallVector<bool> reductionMask(layout.getRank(), false);
    std::fill(reductionMask.begin(), reductionMask.begin() + numBroadcastedDims,
              true);
    setLayoutOrClone(&broadcast.getSourceMutable(),
                     layout.project(reductionMask));
    return;
  }

  if (auto contract = dyn_cast<vector::ContractionOp>(defOp)) {
    // TODO: We could determine lhs/rhs layout if we know one of them, but
    // NYI for now.
    setLayoutOrClone(&contract.getAccMutable(), layout);
    return;
  }

  if (auto gather = dyn_cast<vector::GatherOp>(defOp)) {
    OpOperand &indices = gather.getIndicesMutable();
    OpOperand &mask = gather.getMaskMutable();
    OpOperand &passthru = gather.getPassThruMutable();
    setLayoutOrClone(&indices, layout);
    setLayoutOrClone(&mask, layout);
    setLayoutOrClone(&passthru, layout);
    return;
  }

  if (auto read = dyn_cast<vector::TransferReadOp>(defOp)) {
    if (!read.getMask()) {
      return;
    }
    OpOperand &mask = read.getMaskMutable()[0];
    AffineMap maskMap =
        inversePermutation(compressUnusedDims(read.getPermutationMap()));
    setLayoutOrClone(&mask, layout.apply(maskMap));
    return;
  }

  if (auto gather = dyn_cast<TransferGatherOp>(defOp)) {
    AffineMap sourceMap =
        inverseAndBroadcastProjectedPermutation(gather.getPermutationMap());
    VectorLayoutInterface sourceLayout = layout.apply(sourceMap);
    for (auto [map, operand] : llvm::zip_equal(gather.getIndexedMapsArray(),
                                               gather.getIndexVecsMutable())) {
      setLayoutOrClone(&operand, sourceLayout.apply(map));
    }
    if (gather.getMask()) {
      OpOperand &mask = gather.getMaskMutable()[0];
      AffineMap maskMap =
          inversePermutation(compressUnusedDims(gather.getPermutationMap()));
      setLayoutOrClone(&mask, layout.apply(maskMap));
    }
    return;
  }
}

void LayoutInfo::setLayoutOrClone(OpOperand *val,
                                  VectorLayoutInterface layout) {
  if (!isa<ShapedType>(val->get().getType())) {
    // Don't set layouts on non-shaped types. This would anyway be an empty
    // layout.
    return;
  }
  // Always clone constant like ops and set the layout on them.
  OpBuilder b(val->getOwner());
  if (Operation *defOp = val->get().getDefiningOp()) {
    bool isConstantLike = defOp->hasTrait<OpTrait::ConstantLike>();
    bool isDuplicatable =
        isa<vector::StepOp, vector::CreateMaskOp, vector::ConstantMaskOp>(
            defOp);
    if (isConstantLike || isDuplicatable) {
      b.setInsertionPoint(defOp);
      Operation *cloned = b.clone(*defOp);
      val->set(cloned->getResult(0));
      layouts[cloned->getResult(0)] = layout;
      return;
    }
  }

  if (!hasLayout(val->get())) {
    layouts[val->get()] = layout;
    forward.push(val->get());
    backward.push(val->get());
    return;
  }

  // Otherwise, create a to_layout op to change the layout.
  Value v = val->get();
  Value layourtedV = ToLayoutOp::create(b, v.getLoc(), v, layout);
  val->set(layourtedV);
  layouts[layourtedV] = layout;
  return;
}

LogicalResult propagateVectorLayoutInfo(
    Operation *root, llvm::MapVector<Value, VectorLayoutInterface> &layouts) {
  LayoutInfo info;
  // Initialize propagation info with to_layout operations;
  root->walk([&](ToLayoutOp toLayout) {
    LDBG() << "Initializing layout from to_layout op: " << toLayout << "\n";
    info.setLayoutIfUnset(toLayout.getResult(), toLayout.getLayout());
  });
  // Propgate all layout information until fixpoint. Give priority to
  // forward propagation and only do backward propagation when there is no
  // forward propagation work left.
  while (!info.forward.empty() || !info.backward.empty()) {
    SmallVector<Value> changed;
    if (!info.forward.empty()) {
      Value val = info.forward.front();
      info.forward.pop();
      info.propagateLayoutForward(val);
    } else {
      Value val = info.backward.front();
      info.backward.pop();
      info.propagateLayoutBackward(val);
    }
  }
  layouts = std::move(info.layouts);
  return success();
}

#define GEN_PASS_DEF_TESTVECTORLAYOUTANALYSISPASS
#include "iree/compiler/Codegen/Common/Passes.h.inc"

struct TestVectorLayoutAnalysisPass final
    : impl::TestVectorLayoutAnalysisPassBase<TestVectorLayoutAnalysisPass> {
  void runOnOperation() override {
    Operation *root = getOperation();
    llvm::MapVector<Value, VectorLayoutInterface> layouts;
    if (failed(propagateVectorLayoutInfo(root, layouts))) {
      root->emitError("Layout Analysis Failed");
      return signalPassFailure();
    }

    root->walk([&](Operation *op) {
      if (isa<ToLayoutOp>(op)) {
        return;
      }

      for (OpResult result : op->getOpResults()) {
        if (layouts.contains(result)) {
          op->emitRemark("layout of result #")
              << result.getResultNumber() << " is " << layouts[result];
        }
      }
    });
  }
};
}; // namespace mlir::iree_compiler
