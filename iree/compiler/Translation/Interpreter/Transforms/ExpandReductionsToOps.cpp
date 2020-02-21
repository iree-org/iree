// Copyright 2019 Google LLC
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

#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Translation/Interpreter/IR/HLDialect.h"
#include "iree/compiler/Translation/Interpreter/IR/HLOps.h"
#include "iree/compiler/Translation/Interpreter/Transforms/ConversionUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/MemRefUtils.h"
#include "iree/compiler/Translation/Interpreter/Utils/OpCreationUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

LogicalResult convertReductionOp(FuncOp entryPoint, FuncOp applyFunc,
                                 Operation *elementOp, OpBuilder &builder) {
  // Ensure that this op is pass-through and does not interact with any other
  // ops within the function.
  // TODO(b/139313439): support fused reductions.
  for (auto operand : elementOp->getOperands()) {
    if (operand.getDefiningOp() != nullptr) {
      return elementOp->emitOpError()
             << "Fused reductions are not supported (operand not sourced from "
                "block args)";
    }
  }
  for (auto result : elementOp->getResults()) {
    for (auto *user : result.getUsers()) {
      if (!user->isKnownTerminator()) {
        return elementOp->emitOpError() << "Fused reductions are not supported "
                                           "(result used by non-terminator)";
      }
    }
  }

  // Determine the index of the args we care about. We'll use these to match up
  // the operands of the entry point with our application.
  // Our arguments are expanded tuples like <lhs0, rhs0>, <lhs1, rhs1>, so this
  // index gets the offset * 2.
  auto &applyEntryBlock = applyFunc.getBlocks().front();
  int setIndex = std::distance(applyEntryBlock.args_begin(),
                               llvm::find(applyEntryBlock.getArguments(),
                                          elementOp->getOperand(0))) /
                 2;

  // Map to the args from the entry point.
  auto &entryPointEntryBlock = entryPoint.getBlocks().front();
  Value srcArg = entryPointEntryBlock.getArgument(setIndex);
  Value initArg = entryPointEntryBlock.getArgument(
      applyFunc.getNumArguments() / 2 + setIndex);
  Value dstArg =
      entryPointEntryBlock.getArgument(applyFunc.getNumArguments() + setIndex);
  auto dstType = dstArg.getType().cast<ShapedType>();
  Type elementType = dstType.getElementType();
  auto loc = elementOp->getLoc();
  auto dimensionAttr = entryPoint.getAttrOfType<IntegerAttr>(
      "iree.executable.reduction.dimension");

  Operation *expandedOp = nullptr;
  if (isa<IREEInterp::HL::AddFOp>(elementOp) ||
      isa<IREEInterp::HL::AddIOp>(elementOp)) {
    if (elementType.isa<FloatType>()) {
      expandedOp = builder.create<IREEInterp::HL::ReduceSumFOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    } else {
      expandedOp = builder.create<IREEInterp::HL::ReduceSumIOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    }
  } else if (isa<IREEInterp::HL::MinFOp>(elementOp) ||
             isa<IREEInterp::HL::MinISOp>(elementOp) ||
             isa<IREEInterp::HL::MinIUOp>(elementOp)) {
    if (elementType.isa<FloatType>()) {
      expandedOp = builder.create<IREEInterp::HL::ReduceMinFOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    } else {
      expandedOp = builder.create<IREEInterp::HL::ReduceMinIOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    }
  } else if (isa<IREEInterp::HL::MaxFOp>(elementOp) ||
             isa<IREEInterp::HL::MaxISOp>(elementOp) ||
             isa<IREEInterp::HL::MaxIUOp>(elementOp)) {
    if (elementType.isa<FloatType>()) {
      expandedOp = builder.create<IREEInterp::HL::ReduceMaxFOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    } else {
      expandedOp = builder.create<IREEInterp::HL::ReduceMaxIOp>(
          loc, dstType, srcArg, initArg, dimensionAttr);
    }
  }
  if (!expandedOp) {
    return elementOp->emitOpError()
           << "No matching expanded reduction op for elemental op";
  }
  llvm::SmallVector<int64_t, 4> zeroOffset(dstType.getRank(), 0);
  auto zeroIndices = createArrayConstant(builder, loc, zeroOffset);
  auto lengths = createArrayConstant(builder, loc, dstType.getShape());
  builder.create<IREEInterp::HL::CopyOp>(
      loc, expandedOp->getResult(0), zeroIndices, dstArg, zeroIndices, lengths);

  return success();
}

// Replaces the given elemental |funcOp| with a widened reduction.
LogicalResult expandReductionFunction(FuncOp entryFunc) {
  if (!entryFunc.empty()) {
    return entryFunc.emitError()
           << "Function has already been expanded or has existing contents";
  } else if (!entryFunc.getAttr("iree.executable.reduction.dimension")) {
    return entryFunc.emitError() << "Windowed reductions are not yet supported";
  }
  auto applySym = entryFunc.getAttrOfType<FlatSymbolRefAttr>(
      "iree.executable.reduction.apply");
  if (!applySym) {
    return entryFunc.emitError() << "No reduction application function defined";
  }
  auto applyFunc = entryFunc.getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
      applySym.getValue());
  if (!applyFunc) {
    return entryFunc.emitError()
           << "Unable to find apply function " << applySym;
  }

  auto *entryBlock = entryFunc.addEntryBlock();
  OpBuilder builder(entryBlock);

  if (applyFunc.getBlocks()
          .front()
          .walk([&](Operation *op) {
            if (!op->isKnownTerminator()) {
              if (failed(
                      convertReductionOp(entryFunc, applyFunc, op, builder))) {
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          })
          .wasInterrupted()) {
    return applyFunc.emitError() << "Unable to convert apply func";
  }

  builder.create<mlir::ReturnOp>(builder.getUnknownLoc());

  // Remove the apply function as we have inlined it.
  applyFunc.erase();
  entryFunc.removeAttr("iree.executable.reduction.apply");
  entryFunc.removeAttr("iree.executable.reduction.dimension");

  return success();
}

// Limited lowering of reductions to fat reduce_* ops.
//
// The specific subset this supports is:
//   * 'min', 'max', and 'add' computations, with function names matching the
//      computation
//   * one op per reduction (no fusions yet).
// Note: computations and shapes are not validated.
//
// TODO(b/139410773): Implement more generally, supporting custom computations.
class ExpandReductionsToOpsPass : public ModulePass<ExpandReductionsToOpsPass> {
 public:
  void runOnModule() override {
    auto module = getModule();
    SmallVector<FuncOp, 4> reductionFuncs;
    for (auto funcOp : module.getOps<FuncOp>()) {
      if (funcOp.getAttr("iree.executable.reduction.apply")) {
        reductionFuncs.push_back(funcOp);
      }
    }
    for (auto funcOp : reductionFuncs) {
      if (failed(expandReductionFunction(funcOp))) {
        return signalPassFailure();
      }
    }
  }
};

}  // namespace

std::unique_ptr<OpPassBase<ModuleOp>> createExpandReductionsToOpsPass() {
  return std::make_unique<ExpandReductionsToOpsPass>();
}

static PassRegistration<ExpandReductionsToOpsPass> pass(
    "iree-expand-reductions-to-ops",
    "Expands IREE reduction functions to their interpreter ops");

}  // namespace iree_compiler
}  // namespace mlir
