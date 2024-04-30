// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Preprocessing/Common/Utilities.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace mlir::iree_compiler;

/// Get strides for row-major oredering of a tensor with the given `shape`.
static SmallVector<int64_t> getStridesFromShape(ArrayRef<int64_t> shape) {
  if (shape.empty()) {
    return {};
  }
  SmallVector<int64_t> strides(shape.size(), ShapedType::kDynamic);
  strides.back() = 1;
  for (int i = strides.size() - 1; i > 0; --i) {
    if (ShapedType::isDynamic(shape[i])) {
      break;
    }
    strides[i - 1] = strides[i] * shape[i];
  }
  return strides;
}

// Get the `memref` type for a `tensor` type.
static MemRefType getMemRefTypeFor(MLIRContext *context,
                                   RankedTensorType tensorType) {
  SmallVector<int64_t> strides = getStridesFromShape(tensorType.getShape());
  MemRefLayoutAttrInterface layoutAttr =
      StridedLayoutAttr::get(context,
                             /*offset=*/ShapedType::kDynamic, strides);
  return MemRefType::get(tensorType.getShape(), tensorType.getElementType(),
                         layoutAttr);
}

// Generates the external function call type that corresponds to the
// matched list.
static FunctionType getExternalFunctionCallType(MLIRContext *context,
                                                Location loc,
                                                TypeRange inputTypes,
                                                TypeRange resultTypes,
                                                TypeRange otherOperandTypes) {
  SmallVector<Type> externalCallArgTypes;
  // Conversion from tensor types to call arg types.
  auto convertTensorTypeToCallArgTypes = [&](RankedTensorType tensorType) {
    auto memRefType = getMemRefTypeFor(context, tensorType);
    externalCallArgTypes.push_back(
        MemRefType::get(ArrayRef<int64_t>{}, memRefType.getElementType()));
    externalCallArgTypes.push_back(IndexType::get(context));
  };

  // Conversion from input type to call arg types.
  auto convertInputTypeToCallArgTypes = [&](Type inputType) {
    if (inputType.isIntOrFloat()) {
      externalCallArgTypes.push_back(inputType);
      return;
    }

    auto tensorType = cast<RankedTensorType>(inputType);
    convertTensorTypeToCallArgTypes(tensorType);
    return;
  };

  llvm::for_each(inputTypes, convertInputTypeToCallArgTypes);
  llvm::for_each(resultTypes, [&](Type t) {
    convertTensorTypeToCallArgTypes(cast<RankedTensorType>(t));
  });
  llvm::for_each(otherOperandTypes, convertInputTypeToCallArgTypes);

  return FunctionType::get(context, externalCallArgTypes,
                           /*results=*/TypeRange{});
}

// Returns the base pointer and offset from the given binding.
std::pair<Value, Value>
getBasePtrAndOffsetForTensor(PatternRewriter &rewriter, Location loc,
                             RankedTensorType tensorType, Value value,
                             Value bindingOffset, ValueRange dynamicDims) {
  auto memrefType = getMemRefTypeFor(rewriter.getContext(), tensorType);
  Value memrefVal = rewriter.create<IREE::Stream::BindingSubspanOp>(
      loc, memrefType, value, bindingOffset, dynamicDims);
  auto extractMetadataOp =
      rewriter.create<IREE::Codegen::ExtractStridedMetadataOp>(loc, memrefVal);
  return std::make_pair<Value, Value>(extractMetadataOp.getResult(0),
                                      extractMetadataOp.getResult(1));
}

// Create the entry point function to marshal IREEs ABI and call the external
// function.
static func::FuncOp
createEntryPointFn(PatternRewriter &rewriter, Operation *rootOp,
                   StringRef entryPointFnName, func::FuncOp externalFn,
                   TypeRange inputTypes, TypeRange resultTypes,
                   TypeRange otherOperandTypes) {
  MLIRContext *context = rewriter.getContext();
  Location loc = rootOp->getLoc();

  // The ABI is
  // - !stream.binding for all tensor type operands
  // - !stream.binding for all tensor type results
  // - all scalar operands
  // - values of dynamic dimensions of all tensor operands and results.
  SmallVector<Type> entryPointInputTypes;
  SmallVector<Type> entryPointScalarInputTypes;
  int64_t totalNumDynamicDims = 0;
  auto bindingType = IREE::Stream::BindingType::get(context);

  // Method to process tensor types.
  auto processTensorType = [&](RankedTensorType tensorType) {
    entryPointInputTypes.push_back(bindingType);
    totalNumDynamicDims += tensorType.getNumDynamicDims();
  };
  // Method to process input types.
  auto processInputType = [&](Type type) {
    if (type.isIntOrFloat()) {
      entryPointScalarInputTypes.push_back(type);
      return;
    }
    auto tensorType = cast<RankedTensorType>(type);
    processTensorType(tensorType);
  };

  llvm::for_each(inputTypes, processInputType);
  llvm::for_each(resultTypes,
                 [&](Type t) { processTensorType(cast<RankedTensorType>(t)); });
  llvm::for_each(otherOperandTypes, processInputType);

  int64_t numTensorOperands = (int64_t)entryPointInputTypes.size();
  int64_t numScalarOperands = (int64_t)entryPointScalarInputTypes.size();
  entryPointInputTypes.append(entryPointScalarInputTypes);
  entryPointInputTypes.append(totalNumDynamicDims, rewriter.getIndexType());

  auto entryPointFnType = FunctionType::get(context, entryPointInputTypes,
                                            /*results=*/TypeRange{});
  auto entryPointFn =
      rewriter.create<func::FuncOp>(loc, entryPointFnName, entryPointFnType);
  Region &body = entryPointFn.getBody();
  SmallVector<Location> locs(entryPointInputTypes.size(), loc);
  rewriter.createBlock(&body, body.begin(), entryPointInputTypes, locs);

  auto entryPointArgs = entryPointFn.getArguments();
  auto tensorArgs = entryPointArgs.take_front(numTensorOperands);
  auto scalarArgs = entryPointArgs.slice(numTensorOperands, numScalarOperands);
  auto dynamicDimArgs = entryPointArgs.take_back(totalNumDynamicDims);
  SmallVector<Value> callOperands;
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

  // Method to marshal tensor types into call operands.
  auto marshalTensorTypes = [&](RankedTensorType tensorType) {
    int64_t numDynamicDims = tensorType.getNumDynamicDims();
    auto dynamicDims = dynamicDimArgs.take_front(numDynamicDims);
    auto [basePtr, offset] = getBasePtrAndOffsetForTensor(
        rewriter, loc, tensorType, tensorArgs.front(), zero, dynamicDims);
    callOperands.push_back(basePtr);
    callOperands.push_back(offset);
    tensorArgs = tensorArgs.drop_front();
    dynamicDimArgs = dynamicDimArgs.drop_front(numDynamicDims);
  };
  // Method to marshal input types into call operands.
  auto marshalInputTypes = [&](Type type) {
    if (type.isIntOrFloat()) {
      callOperands.push_back(scalarArgs.front());
      scalarArgs = scalarArgs.drop_front();
      return;
    }
    marshalTensorTypes(cast<RankedTensorType>(type));
  };

  llvm::for_each(inputTypes, marshalInputTypes);
  llvm::for_each(resultTypes, [&](Type t) {
    marshalTensorTypes(cast<RankedTensorType>(t));
  });
  llvm::for_each(otherOperandTypes, marshalInputTypes);

  rewriter.create<func::CallOp>(loc, externalFn, callOperands);
  rewriter.create<func::ReturnOp>(loc, /*operands=*/ValueRange{});
  return entryPointFn;
}

static std::string operandTypeToString(Type operandType) {
  std::string outputString;
  llvm::raw_string_ostream sstream(outputString);
  if (auto shapedType = dyn_cast<ShapedType>(operandType)) {
    shapedType.getElementType().print(sstream);
  } else {
    operandType.print(sstream);
  }
  return outputString;
}

// Make a unique function name based on the base name and input/output types.
static std::string makeUniqueFuncName(std::string externalFnName,
                                      SmallVector<Type> inputTypes,
                                      SmallVector<Type> resultTypes,
                                      SmallVector<Type> otherOperandTypes) {
  SmallVector<std::string> datatypeTokens;
  for (Type inputType : inputTypes) {
    datatypeTokens.push_back(operandTypeToString(inputType));
  }
  for (Type resultType : resultTypes) {
    datatypeTokens.push_back(operandTypeToString(resultType));
  }
  for (Type otherOperandType : otherOperandTypes) {
    datatypeTokens.push_back(operandTypeToString(otherOperandType));
  }
  std::string outputString;
  llvm::raw_string_ostream sstream(outputString);
  llvm::interleave(
      datatypeTokens, [&](std::string token) { sstream << token; },
      [&] { sstream << "_"; });
  return externalFnName + "_" + outputString;
}

// Generate the `hal.executable` that calls into the external function.
// Return the nested symbol reference to the entry point function generated.
static SymbolRefAttr
createStreamExecutableOp(PatternRewriter &rewriter, Operation *rootOp,
                         StringRef externalFnName, TypeRange inputTypes,
                         TypeRange resultTypes, TypeRange otherOperandTypes) {
  auto moduleOp = rootOp->getParentOfType<ModuleOp>();
  assert(moduleOp && "found op without surrounding module");

  Block *body = moduleOp.getBody();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(body);

  // Create the hal.executable to marshal calling the external function.
  Location loc = rootOp->getLoc();

  std::string uniqueExternalFnName = makeUniqueFuncName(
      externalFnName.str(), inputTypes, resultTypes, otherOperandTypes);
  std::string executableOpName = uniqueExternalFnName + "_executable";
  auto executableOp =
      rewriter.create<IREE::Stream::ExecutableOp>(loc, executableOpName);
  executableOp.setPrivate();
  Block &executableOpBody = executableOp.getBlock();
  rewriter.setInsertionPointToStart(&executableOpBody);

  // Create the dispatch inner module.
  auto innerModule = rewriter.create<ModuleOp>(loc);
  Block *moduleBody = innerModule.getBody();
  rewriter.setInsertionPointToStart(moduleBody);

  // Create a private function call which is the external function call.
  MLIRContext *context = rewriter.getContext();
  FunctionType externalFnCallType = getExternalFunctionCallType(
      context, loc, inputTypes, resultTypes, otherOperandTypes);
  func::FuncOp externalFnCall =
      rewriter.create<func::FuncOp>(loc, externalFnName, externalFnCallType);
  externalFnCall.setPrivate();
  externalFnCall->setAttr("llvm.bareptr", rewriter.getBoolArrayAttr(true));

  // Create the executable entry point function.
  std::string entryPointName = externalFnName.str() + "_entry_point";
  func::FuncOp entryFn =
      createEntryPointFn(rewriter, rootOp, entryPointName, externalFnCall,
                         inputTypes, resultTypes, otherOperandTypes);

  // Create the export operation.
  rewriter.setInsertionPoint(innerModule);
  auto exportOp = rewriter.create<IREE::Stream::ExecutableExportOp>(
      loc, entryPointName, FlatSymbolRefAttr::get(context, entryPointName));

  // Create the body of the export operation.
  // TODO(MaheshRavishankar): This represents the number of workgroups to use.
  // Ideally this is somehow exposed to the rewrite mechanism to get the
  // workload and the number of workgroups.
  Region &exportOpRegion = exportOp.getRegion();
  Block *exportOpBody =
      rewriter.createBlock(&exportOpRegion, exportOpRegion.begin());
  rewriter.setInsertionPointToStart(exportOpBody);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  rewriter.create<IREE::Stream::ReturnOp>(loc, ValueRange{one, one, one});
  return SymbolRefAttr::get(rewriter.getStringAttr(executableOpName),
                            SymbolRefAttr::get(entryFn));
}

// Create the `flow.dispatch` op calling into the executable.
static IREE::Flow::DispatchOp
createFlowDispatchOp(PatternRewriter &rewriter, SymbolRefAttr exportOp,
                     Operation *rootOp, TypeRange resultTypes,
                     ValueRange resultDynamicDims, ValueRange operands) {
  Location loc = rootOp->getLoc();
  SmallVector<Value> operandsVec = llvm::to_vector(operands);
  SmallVector<Value> operandDynamicDims;

  // Get the dynamic dims for the operands.
  for (auto operand : operands) {
    auto tensorType = dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorType)
      continue;

    for (auto [index, shape] : llvm::enumerate(tensorType.getShape())) {
      if (!ShapedType::isDynamic(shape))
        continue;

      Value dim = rewriter.create<tensor::DimOp>(loc, operand, index);
      operandDynamicDims.push_back(dim);
    }
  }

  // Append all the dynamic dims to the operands.
  operandsVec.append(operandDynamicDims);
  operandsVec.append(resultDynamicDims.begin(), resultDynamicDims.end());

  // Insert the `flow.dispatch`.
  auto dispatchOp = rewriter.create<IREE::Flow::DispatchOp>(
      loc, exportOp,
      /*workload=*/ValueRange{}, resultTypes, resultDynamicDims, operandsVec,
      operandDynamicDims, /*tiedOperands=*/nullptr);
  return dispatchOp;
}

// Get the values for dynamic shape of results of `rootOp`.
static FailureOr<SmallVector<Value>>
getDynamicResultDims(PatternRewriter &rewriter, ValueRange givenResultDims) {
  // Prune the given dimensions to get just the dynamic dims.
  SmallVector<Value> dynamicResultDims;
  SmallVector<OpFoldResult> mixedValues = getAsOpFoldResult(givenResultDims);
  for (auto ofr : mixedValues) {
    auto value = dyn_cast<Value>(ofr);
    if (!value)
      continue;
    dynamicResultDims.push_back(value);
  }
  return dynamicResultDims;
}

// Check that the operand types and result type satisfy some constants
// - All operands must be scalar type or tensor type.
// - All results must be tensor type.
static LogicalResult checkOperandAndResultTypes(Operation *rootOp,
                                                TypeRange inputTypes,
                                                TypeRange resultTypes,
                                                TypeRange otherOperandTypes) {
  if (llvm::any_of(inputTypes, [](Type type) {
        return !type.isIntOrFloat() && !type.isa<RankedTensorType>();
      })) {
    return rootOp->emitOpError("operand types of external function can be "
                               "`int*`, `float*` or `tensor`");
  }

  if (llvm::any_of(resultTypes,
                   [](Type type) { return !type.isa<RankedTensorType>(); })) {
    return rootOp->emitOpError("result types of external function can only be "
                               "`int*`, `float*` or `tensor`s");
  }

  if (llvm::any_of(otherOperandTypes, [](Type type) {
        return !type.isIntOrFloat() && !isa<RankedTensorType>(type);
      })) {
    return rootOp->emitOpError("operand types of external function can be "
                               "`int*`, `float*` or `tensor`");
  }
  return success();
}

// Constraint function to check that a tensor has a given element type
LogicalResult mlir::iree_compiler::checkTensorElementType(
    PatternRewriter &rewriter, Type operandType, Type elementType) {
  auto tensorType = dyn_cast<RankedTensorType>(operandType);
  return success(tensorType && tensorType.getElementType() == elementType);
}

// Rewrite function to rewrite a matched DAG into a flow.dispatch. Conceptually,
// the matched DAG at the tensor level gets replaced by a function
//
// ```
//   <results> = <external fn>(<input operands>, <initial value of results>,
//   <other operands>)
// ```
//
// `<other operands>` is handled same as `<input operands>`. The split is to
// allow freedom for where the result buffers are passed in through the ABI.
// `<results>` and `<initial values of result>` get tied to the same `memref`.
// So conceptually, at a `memref` level the DAG gets replaced by
//
// ```
//   <external fn>(<input operands>, <result operands in-out>, <other operands>)
// ```
//
// Each buffer object (input or output) is passed as a `pointer, offset` pair
// and value at location `index` is expected to be accessed as `pointer[offset +
// index]` (note: `offset` is number of elements)
//
//
// The operands to this are
// - `rootOp` is the root of the matched DAG. This op will be erased after the
// call.
// - `externalFnName` the name of the function that is provided externally
//   (using a plugin).
// - `inputOperands` are values that are captures as the part of the match
//   and are inputs to the match.
// - `replacedValues` are the values that are captured as part of the match
//   and are replaced by the `flow.dispatch`. The `flow.dispatch` returns
//   as many values as `replacedValues` (and of same type).
// - `replacedValuesShape` are the values for the dynamic dimensions of all the
// `tensor` values in `replacedValues`.
//   For matches that could be static or dynamic, it should be assumed that the
//   shape is dynamic and the value needs to be passed to the rewrite function.
// - `otherOperands` same as `inputOperands`, but kept separate to allow
// flexibility of where the
//   results are passed through the ABI boundary.
LogicalResult mlir::iree_compiler::rewriteAsFlowDispatch(
    PatternRewriter &rewriter, Operation *rootOp, Attribute externalFnName,
    ValueRange inputOperands, ValueRange replacedValues,
    ValueRange replacedValuesShapes, ValueRange otherOperands) {
  auto getType = [](Value v) { return v.getType(); };
  auto inputTypes = llvm::map_to_vector(inputOperands, getType);
  SmallVector<Type> resultTypes = llvm::map_to_vector(replacedValues, getType);
  auto otherOperandTypes = llvm::map_to_vector(otherOperands, getType);

  if (failed(checkOperandAndResultTypes(rootOp, inputTypes, resultTypes,
                                        otherOperandTypes))) {
    return rewriter.notifyMatchFailure(rootOp,
                                       "unhandled operand/result types");
  }
  StringAttr externalFnNameAttr = dyn_cast<StringAttr>(externalFnName);

  std::string uniqueExternalFnName =
      makeUniqueFuncName(externalFnNameAttr.getValue().str(), inputTypes,
                         resultTypes, otherOperandTypes);
  if (!externalFnNameAttr) {
    return rewriter.notifyMatchFailure(
        rootOp, "expected string attribute for external fn name");
  }

  // Get the dynamic result dimensions.
  FailureOr<SmallVector<Value>> dynamicResultDims =
      getDynamicResultDims(rewriter, replacedValuesShapes);
  if (failed(dynamicResultDims)) {
    return rewriter.notifyMatchFailure(
        rootOp, "failed to get dynamic result dimensions");
  }

  // Parent of the rootOp is the funcOp containing it and we want the moduleOp
  // that contains the funcOp. Then we can look at the symbol table to make sure
  // the function we are creating doesnt already exist.
  auto moduleOp = rootOp->getParentOp()->getParentOp();

  SymbolTable symbolTableModule(moduleOp);

  std::string executableOpName = uniqueExternalFnName + "_executable";
  std::string entryPointName =
      externalFnNameAttr.getValue().str() + "_entry_point";
  SymbolRefAttr entryPointFnRef;
  if (auto symbol = symbolTableModule.lookup(executableOpName)) {
    entryPointFnRef = SymbolRefAttr::get(
        SymbolTable::getSymbolName(symbol),
        SymbolRefAttr::get(rewriter.getStringAttr(entryPointName)));
  } else {
    entryPointFnRef = createStreamExecutableOp(
        rewriter, rootOp, externalFnNameAttr.getValue(), inputTypes,
        resultTypes, otherOperandTypes);
  }
  SmallVector<Value> operands = llvm::to_vector(inputOperands);
  operands.append(otherOperands.begin(), otherOperands.end());
  IREE::Flow::DispatchOp dispatchOp =
      createFlowDispatchOp(rewriter, entryPointFnRef, rootOp, resultTypes,
                           dynamicResultDims.value(), operands);
  assert(
      dispatchOp.getNumResults() == replacedValues.size() &&
      "expected dispatch op to return replacements for all specified values");

  for (auto [origValue, replacement] :
       llvm::zip_equal(replacedValues, dispatchOp->getResults())) {
    rewriter.replaceAllUsesWith(origValue, replacement);
  }
  rewriter.eraseOp(rootOp);

  return success();
} // namespace mlir::iree_compiler
