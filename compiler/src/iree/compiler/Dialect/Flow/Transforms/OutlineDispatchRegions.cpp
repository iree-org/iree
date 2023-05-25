// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <utility>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {
namespace {

static int64_t costOfDomain(ArrayRef<int64_t> domain) {
  int64_t product = 1;
  for (int64_t size : domain) {
    if (size == mlir::ShapedType::kDynamic) return INT64_MAX;
    product *= size;
  }
  return product;
};

// Estimates the evaluation cost of a linalg op using a heuristic cost model.
static int64_t estimateLinalgOpCost(linalg::LinalgOp op) {
  // For linalg ops we know the iteration domain, so return the number
  // of iterations of the iteration domain (or INT64_MAX for dynamic.)
  int64_t cost = costOfDomain(op.getStaticLoopRanges());
  LLVM_DEBUG(llvm::dbgs() << "// " << op->getName() << " cost: " << cost
                          << "\n");
  return cost;
}

static TensorType getMainTensorForLinalgExtOp(Operation *op) {
  TensorType main;
  auto operandTypes = llvm::to_vector(op->getOperandTypes());
  auto resultTypes = llvm::to_vector(op->getResultTypes());
  for (Type t : llvm::concat<Type>(operandTypes, resultTypes)) {
    auto tensorType = llvm::dyn_cast<TensorType>(t);
    if (!tensorType) continue;
    if (!main) {
      main = tensorType;
    } else if (costOfDomain(tensorType.getShape()) >
               costOfDomain(main.getShape())) {
      main = tensorType;
    }
  }
  return main;
}

// Estimates the evaluation cost of a LinalgExt op using a heuristic cost
// model.
static int64_t estimateLinalgExtOpCost(Operation *op) {
  TensorType mainTensor = getMainTensorForLinalgExtOp(op);
  // Use the cost of the biggest tensor of the LinalgExt op as an approximation.
  // This is a very, very coarse approximation.
  auto cost = mainTensor ? costOfDomain(mainTensor.getShape()) : 1;
  // Multiply by a semi-arbitrarily chosen factor to capture that LinalgExt ops
  // are "somewhat more expensive" than simply traversing the main tensor.
  // This is something like the extra log(N) factor for a sort or FFT, or
  // the amount of work done by a softmax vs a cheap elementwise on a tensor
  // of the same shape.
  cost *= 10;
  LLVM_DEBUG(llvm::dbgs() << "// " << op->getName() << " cost: " << cost
                          << "\n");
  return cost;
}

// Returns a string like "512xDx128" representing loop ranges.
static std::string loopRangesToString(ArrayRef<int64_t> loopRanges) {
  std::string outputString;
  llvm::raw_string_ostream sstream(outputString);
  llvm::interleave(
      loopRanges,
      [&](int64_t loopRange) {
        // Note: normally we'd use '?', but that isn't a valid character for
        // function names on a variety of targets, so we stick to [a-Z0-9_]
        // characters.
        sstream << (ShapedType::isDynamic(loopRange) ? "D"
                                                     : llvm::itostr(loopRange));
      },
      [&] { sstream << "x"; });
  return outputString;
}

static std::string operandTypeToString(Value operandValue) {
  auto operandType = operandValue.getType();
  std::string outputString;
  llvm::raw_string_ostream sstream(outputString);
  if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
    shapedType.getElementType().print(sstream);
  } else {
    operandType.print(sstream);
  }
  return outputString;
}

// Returns a string like "f32xi32xf16" representing a linalg op's types for each
// operands. Will collapse to single type if all match.
static std::string getLinalgDataTypes(linalg::LinalgOp op) {
  std::string firstToken = "";
  bool allTokensSame = true;
  SmallVector<std::string, 4> datatypeTokens;

  for (Value operandValue : op->getOperands()) {
    datatypeTokens.push_back(operandTypeToString(operandValue));
    if (firstToken.empty()) {
      firstToken = operandTypeToString(operandValue);
    } else if (allTokensSame) {
      allTokensSame = firstToken == operandTypeToString(operandValue);
    }
  }

  if (allTokensSame) {
    return firstToken;
  } else {
    std::string outputString;
    llvm::raw_string_ostream sstream(outputString);
    llvm::interleave(
        datatypeTokens, [&](std::string token) { sstream << token; },
        [&] { sstream << "x"; });
    return outputString;
  }
}

/// Returns the op name without dialect name. E.g., it returns "set_encoding" if
/// the input operation is iree_linalg_ext.set_encoding.
static std::string getOpNameWithoutDialectName(Operation *op) {
  auto opName =
      op->getName().getStringRef().drop_until([](char c) { return c == '.'; });
  if (opName.starts_with(".")) opName = opName.drop_front();
  return opName.str();
}

static std::string summarizeLinalgOp(linalg::LinalgOp op) {
  auto opName = op->getName().getStringRef();
  if (!opName.consume_front("linalg.")) return "";
  std::string opLoopRanges = loopRangesToString(op.getStaticLoopRanges());
  std::string opTypes = opLoopRanges.empty() ? "" : getLinalgDataTypes(op);
  return opName.str() + (opLoopRanges.empty() ? "" : "_" + opLoopRanges) +
         (opTypes.empty() ? "" : "_" + opTypes);
}

static std::string summarizeLinalgExtOp(Operation *op) {
  auto opName = op->getName().getStringRef();
  if (!opName.consume_front("iree_linalg_ext.")) return "";
  std::string suffix = "";
  if (TensorType mainTensor = getMainTensorForLinalgExtOp(op)) {
    llvm::raw_string_ostream sstream(suffix);
    sstream << "_";
    sstream << loopRangesToString(mainTensor.getShape());
    sstream << "x";
    mainTensor.getElementType().print(sstream);
    sstream.flush();
  }
  return opName.str() + suffix;
}

// Summarizes the contents of a dispatch into a short string.
// This uses heuristics to aid developer debugging.
static std::string summarizeDispatchWorkgroupsOp(
    DispatchWorkgroupsOp regionOp) {
  // The goal here is to build a relatively concise description that gives
  // enough information to developers to see roughly what sort of computation a
  // dispatch region performs. Multiple approaches are valid here, depending on
  // what a developer wants to highlight.
  //
  // Currently, this uses a cost model to estimate which individual operation
  // is the most computationally expensive, then a summary is generated which
  // includes some of that operation's parameters.
  //
  // Other metrics to determine which single op is the "best" or which list of
  // ops is most interesting (e.g. to highlight large data movements) could be
  // used instead.

  Operation *bestOp = NULL;
  const int64_t kMinEstimatedCost = -1;
  int64_t bestEstimatedCost = kMinEstimatedCost;
  regionOp.getBodyRegion().walk([&](Operation *op) {
    TypeSwitch<Operation *>(op)
        .Case<linalg::LinalgOp>([&](auto op) {
          int64_t estimatedCost = estimateLinalgOpCost(op);
          if (estimatedCost < bestEstimatedCost) return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Case<IREE::LinalgExt::SetEncodingOp, IREE::LinalgExt::UnsetEncodingOp>(
            [&](auto op) {
              // SetEncoding/UnsetEncoding is the bestOp only if there are no
              // other operations.
              int64_t estimatedCost = kMinEstimatedCost + 1;
              if (estimatedCost < bestEstimatedCost) return;
              bestEstimatedCost = estimatedCost;
              bestOp = op;
              LLVM_DEBUG(llvm::dbgs()
                         << "// new best op: '" << bestOp->getName()
                         << "', cost: " << bestEstimatedCost << "\n");
            })
        .Case<IREE::LinalgExt::LinalgExtOp>([&](auto op) {
          int64_t estimatedCost = estimateLinalgExtOpCost(op);
          if (estimatedCost < bestEstimatedCost) return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Default([&](Operation *op) {
          // No cost estimation implemented, skip.
        });
  });
  if (!bestOp) return "";

  std::string bestSummary = "";
  TypeSwitch<Operation *>(bestOp)
      .Case<linalg::LinalgOp>(
          [&](auto op) { bestSummary = summarizeLinalgOp(op); })
      .Case<IREE::LinalgExt::SetEncodingOp>([&](auto op) {
        auto opName = getOpNameWithoutDialectName(op);
        auto encoding = stringifyEnum(op.getResultTensorEncoding());
        ArrayRef<int64_t> shape = op.getSourceType().getShape();
        bestSummary =
            opName + "_" + encoding.str() + "_" + loopRangesToString(shape);
        ;
      })
      .Case<IREE::LinalgExt::UnsetEncodingOp>([&](auto op) {
        auto opName = getOpNameWithoutDialectName(op);
        auto encoding = stringifyEnum(op.getSourceTensorEncoding());
        ArrayRef<int64_t> shape = op.getResultType().getShape();
        bestSummary =
            opName + "_" + encoding.str() + "_" + loopRangesToString(shape);
      })
      .Case<IREE::LinalgExt::LinalgExtOp>(
          [&](auto op) { bestSummary = summarizeLinalgExtOp(op); })
      .Default([&](Operation *op) {
        // No summarization implemented, default to the op's name.
        bestSummary = op->getName().getStringRef().str();
      });

  LLVM_DEBUG(llvm::dbgs() << "// best op summary: '" << bestSummary << "'\n");
  return bestSummary;
}

// Creates a flow.executable out of a set of functions, pulling in all other
// functions reachable by the provided functions.
static ExecutableOp createExecutable(Location loc, StringRef executableName,
                                     ArrayRef<mlir::func::FuncOp> funcOps,
                                     ModuleOp parentModuleOp) {
  assert(!funcOps.empty() && "must have at least one entry function");

  // Create the executable that will contain the outlined region.
  // NOTE: this will get uniquified if we have multiple in the same block.
  OpBuilder parentModuleBuilder(&parentModuleOp.getBody()->back());
  auto executableOp =
      parentModuleBuilder.create<IREE::Flow::ExecutableOp>(loc, executableName);

  // Create the inner ModuleOp that contains the original functions. We need
  // to provide this shim as some ops (like std.call) look for the
  // containing module to provide symbol resolution.
  OpBuilder executableBuilder(executableOp);
  executableBuilder.setInsertionPointToStart(&executableOp.getBlock());
  auto innerModule = executableBuilder.create<mlir::ModuleOp>(loc);
  for (auto funcOp : funcOps) {
    innerModule.push_back(funcOp);
  }

  // Copy all reachable functions into the executable.
  // Linker passes may dedupe these later on.
  OpBuilder innerModuleBuilder = OpBuilder::atBlockEnd(innerModule.getBody());
  innerModuleBuilder.setInsertionPoint(innerModule.getBody(),
                                       ++innerModule.getBody()->begin());

  return executableOp;
}

// Converts a dispatch region op into a dispatch op to the outlined region.
static LogicalResult convertToDispatchOp(DispatchWorkgroupsOp regionOp,
                                         ExecutableOp executableOp,
                                         ExecutableExportOp exportOp) {
  // Insert at the same place as the original region.
  OpBuilder builder(regionOp);

  // Create the dispatch op to the executable function.
  // Note that we copy the tied operand indices from the workgroups op - it
  // lines up 1:1 with the dispatch once we've outlined things.
  auto dispatchOp = builder.create<DispatchOp>(
      regionOp.getLoc(), exportOp, regionOp.getWorkload(),
      regionOp.getResultTypes(), regionOp.getResultDims(),
      regionOp.getArguments(), regionOp.getArgumentDims(),
      regionOp.getTiedOperandsAttr());
  dispatchOp->setDialectAttrs(regionOp->getDialectAttrs());

  // Replace uses of the existing results with the new results.
  for (int i = 0; i < regionOp.getNumResults(); ++i) {
    regionOp.getResult(i).replaceAllUsesWith(dispatchOp.getResult(i));
  }

  // Erase original region.
  regionOp.erase();

  return success();
}

// Converts a dispatch region body to a free-floating function.
static mlir::func::FuncOp createWorkgroupFunc(Location loc,
                                              StringRef functionName,
                                              Region &region) {
  // Build function type matching the region signature.
  auto functionType = FunctionType::get(
      region.getContext(), region.getArgumentTypes(), /*results=*/{});

  // Clone region into the function body.
  auto funcOp = mlir::func::FuncOp::create(loc, functionName, functionType);
  IRMapping mapping;
  region.cloneInto(&funcOp.getFunctionBody(), mapping);

  // Replace flow.return with std.return.
  // NOTE: in the dispatch workgroups case the return should have no values.
  for (auto &block : funcOp.getBlocks()) {
    if (auto returnOp = dyn_cast<IREE::Flow::ReturnOp>(block.back())) {
      OpBuilder builder(returnOp);
      builder.create<mlir::func::ReturnOp>(
          returnOp.getLoc(), llvm::to_vector<4>(returnOp.getOperands()));
      returnOp.erase();
    }
  }

  return funcOp;
}

// Outlines a dispatch region into a flow.executable and replaces the region op
// with a dispatch to that outlined executable.
static LogicalResult outlineDispatchWorkgroupsOp(
    std::string executableOpName, std::string exportOpName,
    DispatchWorkgroupsOp regionOp) {
  // Convert the region to a free-floating function.
  auto workgroupFuncOp = createWorkgroupFunc(regionOp.getLoc(), exportOpName,
                                             regionOp.getWorkgroupBody());
  if (!workgroupFuncOp) {
    return failure();
  }

  // Create the executable with the region cloned into it.
  auto parentFuncOp = regionOp->getParentOfType<FunctionOpInterface>();
  auto executableOp =
      createExecutable(regionOp.getLoc(), executableOpName, {workgroupFuncOp},
                       parentFuncOp->getParentOfType<mlir::ModuleOp>());
  executableOp.getOperation()->moveBefore(parentFuncOp);
  executableOp.setPrivate();

  // Add an export pointing at the entry point function.
  OpBuilder builder(executableOp.getBody());
  auto exportOp = builder.create<ExecutableExportOp>(
      regionOp.getLoc(), workgroupFuncOp.getName(),
      SymbolRefAttr::get(workgroupFuncOp));
  if (!regionOp.getWorkgroupCount().empty())
    exportOp.getWorkgroupCount().takeBody(regionOp.getWorkgroupCount());

  // Move over the workgroup count region, if present.
  if (!regionOp.getWorkgroupCount().empty()) {
    exportOp.getWorkgroupCount().takeBody(regionOp.getWorkgroupCount());
  }

  // Finally convert the dispatch region into a dispatch to the outlined func.
  return convertToDispatchOp(regionOp, executableOp, exportOp);
}

}  // namespace

class OutlineDispatchRegionsPass
    : public OutlineDispatchRegionsBase<OutlineDispatchRegionsPass> {
 public:
  OutlineDispatchRegionsPass() = default;

  void runOnOperation() override {
    // Convert each dispatch region into a flow.executable + dispatch op.
    int initializerCount = 0;
    for (auto it :
         llvm::enumerate(getOperation().getOps<FunctionOpInterface>())) {
      FunctionOpInterface op = it.value();
      Operation *operation = op;

      // Generate a nice name if possible.
      std::string namePrefix;
      if (auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(operation)) {
        namePrefix = funcOp.getName().str();
      } else if (llvm::isa<IREE::Util::InitializerOp>(operation)) {
        namePrefix =
            std::string("_initializer_") + std::to_string(initializerCount++);
      } else {
        namePrefix =
            std::string("_function_like_") + std::to_string(it.index());
      }

      auto &bodyRegion = op.getFunctionBody();
      // Outline all of the dispatch regions ops in this function.
      auto dispatchWorkgroupsOps =
          llvm::to_vector<8>(bodyRegion.getOps<DispatchWorkgroupsOp>());
      for (int i = 0; i < dispatchWorkgroupsOps.size(); ++i) {
        std::string executableOpName =
            (namePrefix + "_dispatch_" + llvm::Twine(i)).str();
        // Add a summary of the op as a suffix, if one can be generated.
        // Note: the executable names omit this suffix so their names are more
        // predictable.
        LLVM_DEBUG(llvm::dbgs()
                   << "//--- summarizing '" << executableOpName << "' ---//\n");
        std::string opSummary =
            summarizeDispatchWorkgroupsOp(dispatchWorkgroupsOps[i]);
        LLVM_DEBUG(llvm::dbgs()
                   << "//--- opSummary: '" << opSummary << "' ---//\n\n");
        std::string opSuffix = opSummary.empty() ? "" : "_" + opSummary;
        std::string exportOpName = executableOpName + opSuffix;
        if (failed(outlineDispatchWorkgroupsOp(executableOpName, exportOpName,
                                               dispatchWorkgroupsOps[i]))) {
          return signalPassFailure();
        }
      }
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>>
createOutlineDispatchRegionsPass() {
  return std::make_unique<OutlineDispatchRegionsPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
