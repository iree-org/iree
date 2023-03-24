// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace ABI {

static constexpr int64_t kUnspecifiedDim = -1;
static constexpr int64_t kTiedDim = -2;

struct StreamableFunc {
  // Converted func op.
  IREE::Flow::FuncOp funcOp;
  // Parsed tied operand indices.
  SmallVector<int64_t> tiedOperands;
  // Total number of dynamic result dims required.
  int requiredResultDims = 0;
  // Optional custom shape calculation function.
  SymbolRefAttr resultDimsFunc;
  // For each result specifies which call arguments the dynamic dimensions are
  // sourced from. For example, if a result with tensor<?xf32> takes its dim[0]
  // from call arg 2. Only valid if there's no shape calculation function.
  // May contain values of kTiedDim if the dimension matches the equivalent
  // rank dimension of a tied operand.
  SmallVector<SmallVector<int64_t>> resultDimArgs;
};

// Returns true if |funcOp| is a valid result dimension calculation function.
static LogicalResult verifyResultDimsFunc(FunctionType functionType,
                                          int requiredResultDims,
                                          FunctionOpInterface calculateFuncOp) {
  // Check arguments match the function exactly.
  if (functionType.getNumInputs() != calculateFuncOp.getNumArguments()) {
    return calculateFuncOp.emitOpError()
           << "must match the signature of the function using it exactly; "
              "argument count mismatch";
  }
  for (auto [callerType, calleeType] : llvm::zip_equal(
           functionType.getInputs(), calculateFuncOp.getArgumentTypes())) {
    if (callerType != calleeType) {
      return calculateFuncOp.emitOpError()
             << "must match the signature of the function using it exactly; "
                "argument type mismatch (expected "
             << callerType << ", have " << calleeType << ")";
    }
  }

  // We only need dynamic dimensions.
  if (calculateFuncOp.getNumResults() != requiredResultDims) {
    return calculateFuncOp.emitOpError()
           << "must return the exact number of dynamic tensor dimensions as "
              "the function using it (expected "
           << requiredResultDims << ", have " << calculateFuncOp.getNumResults()
           << ")";
  }
  if (!llvm::all_of(calculateFuncOp.getResultTypes(),
                    [](Type type) { return type.isIndex(); })) {
    return calculateFuncOp.emitOpError()
           << "must return only index types matching the dynamic tensor "
              "dimensions";
  }

  return success();
}

// Converts a func.func with the iree.abi.streamable attribute into a flow.func
// and fixes all func.call ops to be flow.call across the module.
static Optional<StreamableFunc> convertStreamableFunc(
    mlir::ModuleOp moduleOp, func::FuncOp funcOp, SymbolTable &symbolTable) {
  OpBuilder moduleBuilder(funcOp);
  auto functionType = funcOp.getFunctionType();

  StreamableFunc streamableFunc;

  // Because streamable ops are asynchronous they must be able to declare their
  // result shapes before they execute so memory can be allocated.
  for (auto resultType : functionType.getResults()) {
    if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
      streamableFunc.requiredResultDims += shapedType.getNumDynamicDims();
    }
  }

  // Check to see if there's a custom result shape calculation function. This
  // will override any of the logic we try to do for dimension setting.
  streamableFunc.resultDimsFunc =
      funcOp->getAttrOfType<SymbolRefAttr>("iree.abi.result_dims");
  if (streamableFunc.resultDimsFunc) {
    auto calculateFuncOp =
        symbolTable.lookupNearestSymbolFrom<FunctionOpInterface>(
            funcOp, streamableFunc.resultDimsFunc);
    if (!calculateFuncOp) {
      funcOp.emitOpError()
          << "cannot find the referenced result shape calculation function "
          << streamableFunc.resultDimsFunc;
      return std::nullopt;
    }
    if (failed(verifyResultDimsFunc(functionType,
                                    streamableFunc.requiredResultDims,
                                    calculateFuncOp))) {
      return std::nullopt;
    }
  }

  // Exclude the attrs used by this pass but leave the rest. Later stages of
  // lowering may have some of their own they need to pass-through.
  SmallVector<NamedAttribute> funcAttrs;
  for (auto attr : funcOp->getDialectAttrs()) {
    if (attr.getName() == "iree.abi.streamable" ||
        attr.getName() == "iree.abi.result_dims") {
      continue;
    }
    funcAttrs.push_back(attr);
  }

  SmallVector<DictionaryAttr> funcArgAttrs;
  for (auto [i, argType] : llvm::enumerate(functionType.getInputs())) {
    // No arg attrs today, just pass-through. Note that we have to handle null.
    if (auto oldArgAttrs = funcOp.getArgAttrDict(i)) {
      funcArgAttrs.push_back(oldArgAttrs);
    } else {
      funcArgAttrs.push_back(moduleBuilder.getDictionaryAttr({}));
    }
  }

  streamableFunc.tiedOperands.resize(functionType.getNumResults(),
                                     IREE::Util::TiedOpInterface::kUntiedIndex);
  SmallVector<DictionaryAttr> funcResAttrs;
  for (auto [i, resultType] : llvm::enumerate(functionType.getResults())) {
    // Tensor results need to have their dynamic dimensions specified.
    // If the result is tied we can default to using the operand dims and
    // otherwise require the user to specify where the dims are. This could get
    // arbitrarily complex (up to and including calling a function to compute
    // dims).
    SmallVector<int64_t> dynamicDimArgs;
    auto shapedType = resultType.dyn_cast<ShapedType>();
    if (shapedType) {
      // Initialize dynamic dim args - we'll verify that they all get covered.
      dynamicDimArgs.resize(shapedType.getNumDynamicDims(), kUnspecifiedDim);
    }

    SmallVector<NamedAttribute> newResAttrs;
    if (auto oldResAttrs = funcOp.getResultAttrDict(i)) {
      // First check if the result is tied to an argument.
      // We can use this to source the initial set of dynamic dimensions.
      if (auto tiedAttr = oldResAttrs.getAs<IntegerAttr>("iree.abi.tied")) {
        streamableFunc.tiedOperands[i] = tiedAttr.getInt();
        if (!streamableFunc.resultDimsFunc &&
            shapedType == functionType.getInput(i)) {
          // Tied types match and we can infer the shape from that. This may
          // have false positives (e.g. in the case of ?x? that gets transposed)
          // but in the more common case of read/write in-place operations this
          // makes this much easier.
          dynamicDimArgs.assign(shapedType.getNumDynamicDims(), kTiedDim);
        }
      }

      // If the user has manually specified the dimensions then override the
      // tied dims (if they were set at all).
      if (auto dimsAttr = oldResAttrs.getAs<ArrayAttr>("iree.abi.dims")) {
        if (streamableFunc.resultDimsFunc) {
          funcOp.emitOpError()
              << "cannot have both an explicit result shape "
                 "calculation function and arg dims reference (on result "
              << i << ")";
          return std::nullopt;
        }
        if (dimsAttr.size() != shapedType.getNumDynamicDims()) {
          funcOp.emitOpError()
              << "result " << i
              << " dynamic dimension mismatch; attribute specifies "
              << dimsAttr.size() << " dimensions but tensor has "
              << shapedType.getNumDynamicDims() << " dynamic dimensions";
          return std::nullopt;
        }
        for (auto [j, value] :
             llvm::enumerate(dimsAttr.getAsValueRange<IntegerAttr>())) {
          dynamicDimArgs[j] = value.getSExtValue();
        }
      }

      // Pass-through all other attrs we don't care about.
      for (auto resAttr : oldResAttrs) {
        if (resAttr.getName() == "iree.abi.tied" ||
            resAttr.getName() == "iree.abi.dims") {
          continue;
        }
        newResAttrs.push_back(resAttr);
      }
    }
    funcResAttrs.push_back(moduleBuilder.getDictionaryAttr(newResAttrs));

    // Ensure all result dims are set or we have a calculation function that can
    // set them.
    if (!streamableFunc.resultDimsFunc) {
      for (auto dim : dynamicDimArgs) {
        if (dim == kUnspecifiedDim) {
          funcOp.emitOpError()
              << "missing dynamic dimensions on result " << i
              << "; must provide via iree.abi.dims, a matching typed tied "
                 "operand, or with a custom result shape calculation function";
          return std::nullopt;
        }
      }
    }
    streamableFunc.resultDimArgs.push_back(std::move(dynamicDimArgs));
  }

  // Create the new streamable flow.func op at the same place as the original.
  streamableFunc.funcOp = moduleBuilder.create<IREE::Flow::FuncOp>(
      funcOp.getLoc(), funcOp.getName(), functionType,
      moduleBuilder.getIndexArrayAttr(streamableFunc.tiedOperands), funcAttrs,
      funcArgAttrs, funcResAttrs);

  // Swap out the symbol in the symbol table.
  symbolTable.erase(funcOp);
  symbolTable.insert(streamableFunc.funcOp);

  return streamableFunc;
}

static LogicalResult convertStreamableCall(StreamableFunc &streamableFunc,
                                           func::CallOp callOp) {
  OpBuilder builder(callOp);

  // Capture all argument dynamic dimensions.
  SmallVector<Value> argDims;
  for (auto arg : callOp.getOperands()) {
    if (arg.getType().isa<ShapedType>()) {
      llvm::append_range(argDims, IREE::Util::buildDynamicDimsForValue(
                                      callOp.getLoc(), arg, builder));
    }
  }

  // Capture all result dynamic dimensions.
  SmallVector<Value> resultDims;
  if (streamableFunc.resultDimsFunc) {
    // Call the custom shape calculation function, if specified.
    // It should return the required number of dynamic dimensions.
    SmallVector<Type> resultDimTypes(streamableFunc.requiredResultDims,
                                     builder.getIndexType());
    auto calculateCallOp = builder.create<func::CallOp>(
        callOp.getLoc(), streamableFunc.resultDimsFunc, resultDimTypes,
        callOp.getOperands());
    llvm::append_range(resultDims, calculateCallOp.getResults());
  } else {
    // Get the shape dimensions from existing call arguments or tied operands.
    for (auto [i, resultType] : llvm::enumerate(callOp.getResultTypes())) {
      if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
        const auto &resultDimArgs = streamableFunc.resultDimArgs[i];
        if (resultDimArgs.empty()) continue;
        if (resultDimArgs.front() == kTiedDim) {
          // Source from a tied operand. Types must match exactly.
          assert(streamableFunc.tiedOperands[i] !=
                     IREE::Util::TiedOpInterface::kUntiedIndex &&
                 "tied dims must be from tied operands");
          auto sourceArg = callOp.getOperand(streamableFunc.tiedOperands[i]);
          assert(sourceArg.getType() == resultType &&
                 "only valid to infer result shapes from identically typed "
                 "tied operands");
          llvm::append_range(resultDims,
                             IREE::Util::buildDynamicDimsForValue(
                                 callOp.getLoc(), sourceArg, builder));
        } else {
          // Source from call arguments.
          for (int64_t j : resultDimArgs) {
            resultDims.push_back(callOp.getOperand(j));
          }
        }
      }
    }
  }

  // Replace the original func.call with the new flow.call.
  auto streamableCallOp = builder.create<IREE::Flow::CallOp>(
      callOp.getLoc(), callOp.getCalleeAttr(), callOp.getResultTypes(),
      resultDims, callOp.getOperands(), argDims,
      streamableFunc.funcOp.getTiedOperandsAttr());
  streamableCallOp->setDialectAttrs(callOp->getDialectAttrs());
  callOp.replaceAllUsesWith(streamableCallOp.getResults());
  callOp.erase();

  return success();
}

static LogicalResult convertStreamableCalls(
    mlir::ModuleOp moduleOp,
    DenseMap<StringRef, StreamableFunc> &streamableFuncs) {
  auto walkResult = moduleOp.walk([&](func::CallOp callOp) {
    auto it = streamableFuncs.find(callOp.getCallee());
    if (it != streamableFuncs.end()) {
      if (failed(convertStreamableCall(it->second, callOp))) {
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return walkResult.wasInterrupted() ? failure() : success();
}

class ConvertStreamableOpsPass
    : public PassWrapper<ConvertStreamableOpsPass, OperationPass<ModuleOp>> {
 public:
  ConvertStreamableOpsPass() = default;
  ConvertStreamableOpsPass(const ConvertStreamableOpsPass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, mlir::tensor::TensorDialect,
                    IREE::Flow::FlowDialect>();
  }

  StringRef getArgument() const override {
    return "iree-abi-convert-streamable-ops";
  }

  StringRef getDescription() const override {
    return "Converts streamable ops in input dialects into their IREE dialect "
           "forms.";
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Gather functions that need wrapping.
    SmallVector<func::FuncOp> originalFuncOps;
    for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
      // Ignore functions already marked as having their ABI goo handled.
      if (funcOp->hasAttr("iree.abi.streamable")) {
        if (!funcOp.isExternal()) {
          funcOp.emitOpError()
              << "only external streamable calls are supported today";
          return signalPassFailure();
        }
        originalFuncOps.push_back(funcOp);
      }
    }

    SymbolTable symbolTable(moduleOp);
    DenseMap<StringRef, StreamableFunc> streamableFuncs;

    // Convert all function declarations identified as streamable.
    for (auto originalFuncOp : originalFuncOps) {
      auto streamableFuncOr =
          convertStreamableFunc(moduleOp, originalFuncOp, symbolTable);
      if (!streamableFuncOr.has_value()) return signalPassFailure();
      auto streamableFunc = std::move(streamableFuncOr).value();
      streamableFuncs[streamableFunc.funcOp.getName()] =
          std::move(streamableFunc);
    }

    // Convert all calls to those streamable func ops.
    if (failed(convertStreamableCalls(moduleOp, streamableFuncs))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertStreamableOpsPass() {
  return std::make_unique<ConvertStreamableOpsPass>();
}

static PassRegistration<ConvertStreamableOpsPass> pass;

}  // namespace ABI
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
