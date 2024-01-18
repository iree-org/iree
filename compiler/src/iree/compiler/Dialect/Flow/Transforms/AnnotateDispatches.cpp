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
#include "iree/compiler/Utils/StringUtils.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-dispatch"

namespace mlir::iree_compiler::IREE::Flow {
namespace {

static int64_t costOfDomain(ArrayRef<int64_t> domain) {
  int64_t product = 1;
  for (int64_t size : domain) {
    if (ShapedType::isDynamic(size))
      return INT64_MAX;
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
    if (!tensorType)
      continue;
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

// Estimates the evaluation cost of a Linalg::Softmax op using a heuristic cost
// model similar to LinalgExt ops.
static int64_t estimateLinalgSoftmaxOpCost(Operation *op) {
  return estimateLinalgExtOpCost(op);
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
  SmallVector<std::string> datatypeTokens;

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
  if (opName.starts_with("."))
    opName = opName.drop_front();
  return opName.str();
}

static std::string summarizeLinalgOp(linalg::LinalgOp op) {
  std::string prefix;

  // Check if the op is a transpose and mark it as such for a better summary.
  {
    // Check if the body only contains a yield.
    bool hasOnlyYield = op.getBlock()->without_terminator().empty();

    // Check if the indexing maps are only permutations.
    bool hasOnlyPermutation = true;
    for (AffineMap map : op.getIndexingMapsArray()) {
      if (!map.isPermutation()) {
        hasOnlyPermutation = false;
      }
    }

    if (hasOnlyYield && hasOnlyPermutation) {
      prefix = "transpose";
    }
  }

  if (prefix.empty()) {
    // By default, use the op name as prefix.
    auto opName = op->getName().getStringRef();
    if (!opName.consume_front("linalg."))
      return "";
    prefix = opName.str();
  }

  std::string opLoopRanges = loopRangesToString(op.getStaticLoopRanges());
  std::string opTypes = opLoopRanges.empty() ? "" : getLinalgDataTypes(op);
  return prefix + (opLoopRanges.empty() ? "" : "_" + opLoopRanges) +
         (opTypes.empty() ? "" : "_" + opTypes);
}

static std::string summarizeLinalgExtOp(Operation *op) {
  auto opName = op->getName().getStringRef();
  // Currently, this utility is also invoked by Linalg::SoftmaxOp.
  if (!(opName.consume_front("iree_linalg_ext.") ||
        opName.consume_front("linalg.")))
    return "";
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
static std::string summarizeDispatchRegion(Region &region) {
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
  // Collect TilingInterface ops for better heuristic names.
  SmallVector<Operation *> tileableOps;
  region.walk([&](Operation *op) {
    if (isa<TilingInterface>(op)) {
      tileableOps.push_back(op);
    }
    TypeSwitch<Operation *>(op)
        .Case<linalg::SoftmaxOp>([&](auto op) {
          int64_t estimatedCost = estimateLinalgSoftmaxOpCost(op);
          if (estimatedCost < bestEstimatedCost)
            return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Case<linalg::LinalgOp>([&](auto op) {
          int64_t estimatedCost = estimateLinalgOpCost(op);
          if (estimatedCost < bestEstimatedCost)
            return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Case<IREE::LinalgExt::SetEncodingOp, IREE::LinalgExt::UnsetEncodingOp,
              tensor::PackOp, tensor::UnPackOp>([&](auto op) {
          // SetEncoding/UnsetEncoding/PackOp/UnPackOp is the bestOp only if
          // there are no other operations.
          int64_t estimatedCost = kMinEstimatedCost + 1;
          if (estimatedCost < bestEstimatedCost)
            return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Case<IREE::LinalgExt::LinalgExtOp>([&](auto op) {
          int64_t estimatedCost = estimateLinalgExtOpCost(op);
          if (estimatedCost < bestEstimatedCost)
            return;
          bestEstimatedCost = estimatedCost;
          bestOp = op;
          LLVM_DEBUG(llvm::dbgs() << "// new best op: '" << bestOp->getName()
                                  << "', cost: " << bestEstimatedCost << "\n");
        })
        .Default([&](Operation *op) {
          // No cost estimation implemented, skip.
        });
  });

  if (!bestOp) {
    std::string bestSummary = "";
    // Check if there is a possible slow memory copy as a dispatch. The current
    // heuristic is to check if a dispatch.tensor.store stores a tensor that is
    // directly loaded from a dispatch.tensor.load.
    region.walk([&](IREE::Flow::DispatchTensorStoreOp storeOp) {
      Value input = storeOp.getValue();
      if (auto loadOp =
              input.getDefiningOp<IREE::Flow::DispatchTensorLoadOp>()) {
        bestSummary = "slow_memcpy";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return bestSummary;
  }

  std::string bestSummary = "";
  TypeSwitch<Operation *>(bestOp)
      .Case<linalg::SoftmaxOp>(
          [&](auto op) { bestSummary = summarizeLinalgExtOp(op); })
      .Case<linalg::LinalgOp>(
          [&](auto op) { bestSummary = summarizeLinalgOp(op); })
      .Case<tensor::PackOp, tensor::UnPackOp>([&](auto op) {
        auto opName = getOpNameWithoutDialectName(op);
        bestSummary = opName + "_" + operandTypeToString(op.getSource());
      })
      .Case<IREE::LinalgExt::SetEncodingOp>([&](auto op) {
        auto opName = getOpNameWithoutDialectName(op);
        auto encoding = op.getResultType()
                            .getEncoding()
                            .template cast<IREE::LinalgExt::EncodingAttr>();
        auto user = stringifyEnum(encoding.getUser().getValue());
        auto role = stringifyEnum(encoding.getRole().getValue());
        ArrayRef<int64_t> shape = op.getSourceType().getShape();
        bestSummary = opName + "_" + user.str() + "_" + role.str() + "_" +
                      loopRangesToString(shape);
        ;
      })
      .Case<IREE::LinalgExt::UnsetEncodingOp>([&](auto op) {
        auto opName = getOpNameWithoutDialectName(op);
        auto encoding = op.getSourceType()
                            .getEncoding()
                            .template cast<IREE::LinalgExt::EncodingAttr>();
        auto user = stringifyEnum(encoding.getUser().getValue());
        auto role = stringifyEnum(encoding.getRole().getValue());
        ArrayRef<int64_t> shape = op.getResultType().getShape();
        bestSummary = opName + "_" + user.str() + "_" + role.str() + "_" +
                      loopRangesToString(shape);
      })
      .Case<IREE::LinalgExt::LinalgExtOp>(
          [&](auto op) { bestSummary = summarizeLinalgExtOp(op); })
      .Default([&](Operation *op) {
        // No summarization implemented, default to the op's name.
        bestSummary = op->getName().getStringRef().str();
      });

  // Add heuristic hint to dispatch name if the unpack op is the first op and
  // the pack op is the last op.
  if (!tileableOps.empty()) {
    if (!isa<tensor::UnPackOp>(bestOp) &&
        isa<tensor::UnPackOp>(tileableOps.front())) {
      bestSummary = "unpack_" + bestSummary;
    }
    if (!isa<tensor::PackOp>(bestOp) &&
        isa<tensor::PackOp>(tileableOps.back())) {
      bestSummary = bestSummary + "_pack";
    }
  }

  // Sanitize the string so that it contains only C literal-compatible chars.
  bestSummary = sanitizeSymbolName(bestSummary);

  LLVM_DEBUG(llvm::dbgs() << "// best op summary: '" << bestSummary << "'\n");
  return bestSummary;
}

} // namespace

class AnnotateDispatchesPass
    : public AnnotateDispatchesBase<AnnotateDispatchesPass> {
public:
  AnnotateDispatchesPass() = default;

  void runOnOperation() override {
    DenseMap<Attribute, SymbolRefAttr> entryPointRefReplacements;
    for (auto executableOp :
         getOperation().getBody()->getOps<IREE::Flow::ExecutableOp>()) {
      auto innerModuleOp = executableOp.getInnerModule();
      if (!innerModuleOp)
        continue;
      for (auto exportOp :
           executableOp.getBlock().getOps<ExecutableExportOp>()) {
        auto oldSymbolRefAttr = SymbolRefAttr::get(
            &getContext(), executableOp.getName(),
            {SymbolRefAttr::get(&getContext(), exportOp.getSymName())});

        auto funcOp = innerModuleOp.lookupSymbol<FunctionOpInterface>(
            exportOp.getFunctionRef());
        if (!funcOp)
          continue; // extern module, maybe
        std::string summary = summarizeDispatchRegion(funcOp.getFunctionBody());
        if (summary.empty())
          continue; // unable to tell

        std::string newName = funcOp.getName().str() + "_" + summary;

        exportOp.setSymName(newName);
        exportOp.setFunctionRef(newName);
        funcOp.setName(newName);

        auto newSymbolRefAttr =
            SymbolRefAttr::get(&getContext(), executableOp.getName(),
                               {SymbolRefAttr::get(&getContext(), newName)});
        entryPointRefReplacements[oldSymbolRefAttr] = newSymbolRefAttr;
      }
    }

    // Replace each usage of an entry point with its original symbol name with a
    // new symbol name.
    for (auto funcLikeOp : getOperation().getOps<FunctionOpInterface>()) {
      funcLikeOp->walk([&](IREE::Flow::DispatchOp dispatchOp) {
        SmallVector<Attribute> replacementRefs;
        for (auto originalRef : dispatchOp.getEntryPointRefs()) {
          auto it = entryPointRefReplacements.find(originalRef);
          if (it != entryPointRefReplacements.end()) {
            replacementRefs.push_back(it->second);
          } else {
            replacementRefs.push_back(originalRef);
          }
        }
        dispatchOp.setEntryPointsAttr(
            ArrayAttr::get(dispatchOp.getContext(), replacementRefs));
      });
    }
  }
};

std::unique_ptr<OperationPass<mlir::ModuleOp>> createAnnotateDispatchesPass() {
  return std::make_unique<AnnotateDispatchesPass>();
}

} // namespace mlir::iree_compiler::IREE::Flow
