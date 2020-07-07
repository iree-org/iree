// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/Flow/Transforms/DispatchConfig.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Utils/DispatchUtils.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

#define DEBUG_TYPE "iree-detail"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {
// TODO(laurenzo): Every one of these should have better support and removed
// from this exclusion list eventually.
bool isUnsupportedFusionOp(Operation *op) {
  return isa<mhlo::DotOp>(op) || isa<mhlo::ConvOp>(op) ||
         isa<mhlo::ReduceOp>(op) || isa<mhlo::PadOp>(op) ||
         isa<mhlo::ReduceWindowOp>(op) || isa<mhlo::TorchIndexSelectOp>(op) ||
         isa<mhlo::SliceOp>(op) || isa<mhlo::ConcatenateOp>(op);
}

// Allowlist of ops that materialize to a an index-permuted copy of some kind
// if they exist standalone. Generally we try to avoid anchoring on these,
// letting them fuse into more meaningful ops as possible.
bool isIndexOp(Operation *op) {
  // TODO(laurenzo): Curate this list more specifically (or have a better
  // mechanism for determining).
  return isa<Shape::RankedBroadcastInDimOp>(op) ||
         isa<mhlo::BroadcastInDimOp>(op) ||
         isa<mhlo::DynamicBroadcastInDimOp>(op) ||
         isa<mhlo::DynamicReshapeOp>(op) || isa<mhlo::DynamicSliceOp>(op) ||
         isa<mhlo::ReshapeOp>(op) || isa<mhlo::SliceOp>(op) ||
         isa<mhlo::TransposeOp>(op);
}
}  // namespace

//------------------------------------------------------------------------------
// OpDispatchPolicy
//------------------------------------------------------------------------------

bool OpDispatchPolicy::isDispatchable(Operation *op) {
  if (FlowDialect::isDialectOp(op)) {
    // Ignore things we've already produced as they should only relate to
    // sequencer operations.
    LLVM_DEBUG(llvm::dbgs() << "  NOT DISPATCHABLE (Flow Dialect): "
                            << op->getName() << "\n");
    return false;
  } else if (op->isKnownTerminator()) {
    // Currently we skip all terminators as we want to leave them in the block
    // to keep it valid. Future folding passes may take care of them if they are
    // worth bringing into the dispatch region.
    LLVM_DEBUG(llvm::dbgs() << "  NOT DISPATCHABLE (Known Terminator): "
                            << op->getName() << "\n");
    return false;
  } else if (auto callOp = dyn_cast<CallOp>(op)) {
    bool dispatchable = dispatchability.isDispatchable(callOp.getCallee());
    LLVM_DEBUG(llvm::dbgs()
               << "  " << (dispatchable ? "" : "NOT ")
               << "DISPATCHABLE (Call): " << op->getName() << "\n");
    return dispatchable;
  } else if (isa<CallIndirectOp>(op)) {
    // Indirect calls are not supported in dispatch code.
    LLVM_DEBUG(llvm::dbgs() << "  NOT DISPATCHABLE (Call Indirect): "
                            << op->getName() << "\n");
    return false;
  } else if (isa<ConstantOp>(op)) {
    // Constants are handled in the RematerializeDispatchConstants pass.
    // We do that independently so that we can more easily see the use of
    // constants across all dispatches instead of just on an individual basis
    // as we do here.
    LLVM_DEBUG(llvm::dbgs()
               << "  NOT DISPATCHABLE (Constant): " << op->getName() << "\n");
    return false;
  } else if (op->getNumResults() &&
             !op->getResult(0).getType().isa<ShapedType>()) {
    // We don't put scalar manipulation into dispatch regions.
    LLVM_DEBUG(llvm::dbgs()
               << "  NOT DISPATCHABLE (Non Shaped): " << op->getName() << "\n");
    return false;
  } else if (!isOpOfKnownDialect(op)) {
    // Probably a custom op.
    LLVM_DEBUG(llvm::dbgs() << "  NOT DISPATCHABLE (Unknown Dialect): "
                            << op->getName() << "\n");
    return false;
  }
  LLVM_DEBUG(llvm::dbgs() << "  DISPATCHABLE: " << op->getName() << "\n");
  return true;
}

bool OpDispatchPolicy::isIdentityMetadata(Operation *op) {
  return isa<Shape::TieShapeOp>(op);
}

int OpDispatchPolicy::getAnchorBenefit(Operation *op) {
  if (isUnsupportedFusionOp(op)) {
    return 100;
  }

  if (isa<Shape::TieShapeOp>(op) || isa<Shape::MakeRankedShapeOp>(op)) {
    // Cannot anchor.
    return 0;
  } else if (isIndexOp(op)) {
    // We generally do not want to form anchors around ops that just do a copy
    // (perhaps with an affine map) except as a last resort.
    return 1;
  } else if (isa<mhlo::SelectOp>(op)) {
    // TODO(#2050): In a number of cases, this makes it less likely to split
    // a DR across a compare/select boundary. Remove this once i1 is legalized
    // properly.
    return 15;
  } else {
    // Most dispatchable ops can anchor but are a fairly low benefit.
    return 10;
  }
}

OpDispatchPolicy::FusionType OpDispatchPolicy::fuseInput(Operation *anchorOp,
                                                         Operation *inputOp) {
  if (inputOp->isKnownTerminator()) return FusionType::DISABLED;

  if (isIdentityMetadata(inputOp)) {
    // Shape ties must always be duplicated into the region and remain in their
    // original position. This should apply to any such "metadata" ops.
    return FusionType::CLONE_INTO;
  }
  if (isUnsupportedFusionOp(anchorOp) || isUnsupportedFusionOp(inputOp)) {
    return FusionType::DISABLED;
  }

  // By default for operands, they are duplicated into the dispatch region.
  // Typically at the initial fusion stage, there is not a sufficient cost
  // model to determine whether it is more beneficial to fuse or materialize,
  // so the bias is towards fusion and leaving inter-region analysis to a later
  // phase.
  return FusionType::CLONE_INTO;
}

OpDispatchPolicy::FusionType OpDispatchPolicy::fuseOutput(Operation *anchorOp,
                                                          Operation *outputOp) {
  if (outputOp->isKnownTerminator() || outputOp->getNumResults() == 0) {
    return FusionType::DISABLED;
  }
  if (isIdentityMetadata(outputOp)) {
    return FusionType::MOVE_INTO;
  }
  if (isUnsupportedFusionOp(anchorOp) || isUnsupportedFusionOp(outputOp)) {
    return FusionType::DISABLED;
  }

  // Generally, it is hard to reason locally about the legality of fusing an
  // output, since additional analysis may need to be done to determine
  // workload compatibility (especially with dynamic shapes involved). As
  // such, we do as little as possible here and instead rely on optimization
  // passes to merge compatible regions.
  return FusionType::DISABLED;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
