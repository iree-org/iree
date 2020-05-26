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
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

#define DEBUG_TYPE "iree-dispatch-config"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

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
  if (isa<Shape::TieShapeOp>(op) || isa<Shape::MakeRankedShapeOp>(op)) {
    // Cannot anchor.
    return 0;
  } else if (isa<xla_hlo::DotOp>(op) || isa<xla_hlo::ConvOp>(op) ||
             isa<xla_hlo::ReduceOp>(op) || isa<xla_hlo::PadOp>(op) ||
             isa<xla_hlo::ReduceWindowOp>(op)) {
    // High benefit anchor.
    // TODO(GH-1605): Remove xla_hlo::PadOp from the check.
    return 10;
  } else {
    // Most dispatchable ops can anchor but are a fairly low benefit.
    return 1;
  }
}

OpDispatchPolicy::FusionType OpDispatchPolicy::fuseInput(Operation *anchorOp,
                                                         Operation *inputOp) {
  if (inputOp->isKnownTerminator()) return FusionType::DISABLED;

  if (isa<xla_hlo::DotOp>(inputOp) || isa<xla_hlo::ConvOp>(inputOp) ||
      isa<xla_hlo::ReduceOp>(inputOp) || isa<xla_hlo::PadOp>(inputOp) ||
      isa<xla_hlo::ReduceWindowOp>(inputOp)) {
    // TODO(GH-1605): Remove xla_hlo::PadOp from the check.
    return FusionType::DISABLED;
  } else if (isIdentityMetadata(inputOp)) {
    // Shape ties must always be duplicated into the region and remain in their
    // original position. This should apply to any such "metadata" ops.
    return FusionType::CLONE_INTO;
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

  if (isa<xla_hlo::DotOp>(outputOp) || isa<xla_hlo::ConvOp>(outputOp) ||
      isa<xla_hlo::ReduceOp>(outputOp) || isa<xla_hlo::PadOp>(outputOp) ||
      isa<xla_hlo::ReduceWindowOp>(outputOp)) {
    // TODO(GH-1605): Remove xla_hlo::PadOp from the check.
    return FusionType::DISABLED;
  } else if (isIdentityMetadata(outputOp)) {
    return FusionType::MOVE_INTO;
  }

  // Generally allow fusions of any output op that has the same shape.
  // This is very simplified and needs to be extended to infer shape
  // equivalence wrt dynamic shapes and other conditions that are legal to
  // fuse.
  assert(anchorOp->getNumResults() > 0);
  Type anchorType = anchorOp->getResult(0).getType();
  Type outputType = outputOp->getResult(0).getType();
  auto shapedType = anchorType.dyn_cast<ShapedType>();
  if (shapedType.hasStaticShape() && anchorType == outputType) {
    return FusionType::MOVE_INTO;
  }

  return FusionType::DISABLED;
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
