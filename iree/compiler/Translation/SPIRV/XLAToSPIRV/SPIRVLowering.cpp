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

//===- SPIRVLowering.cpp ---------------------------------------*- C++//-*-===//
//
// SPIR-V Code-generation for XLA-HLO Ops within IREE Dispatch functions
//
//===----------------------------------------------------------------------===//
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/SPIRVLowering.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SPIRV/SPIRVLowering.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// SPIR-V codegen implementation
//===----------------------------------------------------------------------===//

Value genPointerOffset(OpBuilder &builder, Location loc,
                       TensorIndexToScalarValueMap &valueCache,
                       const AffineMap &indexMap, ArrayRef<int64_t> shape,
                       const Value &buffer) {
  auto varPtrType =
      buffer.getType().cast<spirv::PointerType>().getPointeeType();
  // The variable has to be a struct type with a single element.
  auto varStructType = varPtrType.cast<spirv::StructType>();
  assert(varStructType.getNumElements() == 1 &&
         "expected variable type to be a spv.ptr of spv.struct with a single "
         "element");
  auto varType = varStructType.getElementType(0);

  SmallVector<Value, 2> accessIndices;
  auto i32Type = builder.getIntegerType(32);
  auto zero = spirv::ConstantOp::getZero(i32Type, loc, &builder);
  accessIndices.push_back(zero);
  if (varType.isa<spirv::ArrayType>()) {
    Value linearizedIndex =
        valueCache.getAccessIndicesForIndexMap(builder, loc, indexMap, shape);
    if (!linearizedIndex) {
      return nullptr;
    }
    accessIndices.push_back(linearizedIndex);
  }
  return builder.create<spirv::AccessChainOp>(loc, buffer, accessIndices);
}

/// Returns the type of the global variable to use for an argument of the
/// dispatch function.
static spirv::PointerType convertArgTypeToVariableType(Location loc,
                                                       ShapedType argType) {
  if (!argType.isa<MemRefType>()) {
    emitError(loc, "unhandled arg type ")
        << argType << " while lowering to SPIR-V";
    return nullptr;
  }
  auto elementType = argType.getElementType();
  if (!elementType.isSignlessIntOrFloat()) {
    emitError(loc, "unhandled element type ")
        << elementType << " while lowering to SPIR-V";
    return nullptr;
  }
  if (!argType.hasStaticShape()) {
    // TODO(ravishankarm): Handle dynamic shapes.
    emitError(loc, "unhandled dynamic shape of argument");
    return nullptr;
  }
  int64_t stride = elementType.getIntOrFloatBitWidth() / 8;
  if (argType.getRank()) {
    elementType = spirv::ArrayType::get(
        elementType, argType.getNumElements(),
        static_cast<spirv::ArrayType::LayoutInfo>(stride));
  }
  return spirv::PointerType::get(
      spirv::StructType::get(elementType,
                             static_cast<spirv::StructType::LayoutInfo>(0)),
      spirv::StorageClass::StorageBuffer);
}

namespace detail {
LogicalResult SPIRVCodegenImpl::createEntryFn(
    OpBuilder &builder, FuncOp &fn, TensorIndexToScalarValueMap &valueCache) {
  auto entryFnAttr = fn.getAttrOfType<spirv::EntryPointABIAttr>(
      spirv::getEntryPointABIAttrName());
  if (!entryFnAttr)
    return fn.emitError(
        "expected spv.entry_point_abi attribute on dispatch function "
        "implementation");

  auto loc = fn.getLoc();

  // Add ABI attributes to function arguments and return values.
  auto fnType = fn.getType();
  SmallVector<Type, 2> entryFnArgTypes;
  SmallVector<spirv::InterfaceVarABIAttr, 2> entryFnArgAttrs;
  entryFnArgTypes.reserve(fnType.getNumInputs());
  entryFnArgAttrs.reserve(fnType.getNumInputs());
  for (auto argType : enumerate(fnType.getInputs())) {
    auto argMemRefType = argType.value().dyn_cast<MemRefType>();
    if (!argMemRefType) {
      return fn.emitError("expected arguments to be of memref types");
    }
    auto convertedArgType = convertArgTypeToVariableType(loc, argMemRefType);
    if (!convertedArgType) {
      return failure();
    }
    entryFnArgTypes.emplace_back(convertedArgType);
    spirv::InterfaceVarABIAttr abiAttr =
        fn.getArgAttrOfType<spirv::InterfaceVarABIAttr>(
            argType.index(), spirv::getInterfaceVarABIAttrName());
    if (!abiAttr) {
      // TODO(ravishankarm): Default ABI if no ABI is specified. This is just to
      // make writing tests easily. Within IREE translation the attributes will
      // be set already.
      abiAttr = spirv::getInterfaceVarABIAttr(0, argType.index(), {},
                                              builder.getContext());
    }
    entryFnArgAttrs.emplace_back(abiAttr);
  }

  // TODO(ravishankarm) : Handle return types. The SPIR-V ABI attribute lowering
  // needs to support attributes on return values.
  if (fnType.getNumResults()) {
    return fn.emitError("unimplemented handling for return types");
  }

  auto entryFnType = builder.getFunctionType(entryFnArgTypes, ArrayRef<Type>());
  Optional<StringRef> entryFnName = getDispatchFuncName(fn);
  if (!entryFnName) return fn.emitError("unable to get dispatch function name");
  auto entryFn =
      builder.create<spirv::FuncOp>(loc, entryFnName.getValue(), entryFnType);
  entryFn.addEntryBlock();

  if (failed(setABIAttrs(entryFn, entryFnAttr, entryFnArgAttrs))) {
    return failure();
  }
  // Set the argument to buffer mapping
  for (auto arg : enumerate(fn.getArguments())) {
    valueCache.setBufferForArgument(arg.value(),
                                    entryFn.getArgument(arg.index()));
  }

  // Start a scope to create an insertion guard to reset the builder once the
  // function is lowered.
  {
    OpBuilder::InsertionGuard funcInsertGuard(builder);
    builder.setInsertionPointToStart(&entryFn.getBody().front());

    if (failed(lowerFunction(builder, fn, entryFn, valueCache))) {
      return failure();
    }
  }

  return success();
}

/// Creates the spv.loop operation given the lb, ub and step.
// TODO(ravishankarm): Move this into SPIR-V dialect as a utility function.
static Value createLoop(OpBuilder &builder, Location loc, Value lb, Value ub,
                        Value step) {
  auto loopControl = builder.getI32IntegerAttr(
      static_cast<uint32_t>(spirv::LoopControl::None));
  auto loopOp = builder.create<spirv::LoopOp>(loc, loopControl);
  loopOp.addEntryAndMergeBlock();
  // Header block.
  auto header = new Block();
  // Insert the header.
  loopOp.body().getBlocks().insert(std::next(loopOp.body().begin(), 1), header);
  // Insert the body.
  Block *body = new Block();
  loopOp.body().getBlocks().insert(std::next(loopOp.body().begin(), 2), body);
  // Header block arg is the induction variable.
  BlockArgument newIndVar = header->addArgument(lb.getType());

  // Emit the entry code.
  Block *entry = loopOp.getEntryBlock();
  builder.setInsertionPointToEnd(entry);
  builder.create<spirv::BranchOp>(loc, header, lb);

  // Emit the header code.
  builder.setInsertionPointToEnd(header);
  auto cmpOp = builder.create<spirv::SLessThanOp>(loc, builder.getI1Type(),
                                                  newIndVar, ub);
  Block *merge = loopOp.getMergeBlock();
  builder.create<spirv::BranchConditionalOp>(
      loc, cmpOp, body, ArrayRef<Value>(), merge, ArrayRef<Value>());

  // Emit the continue/latch block.
  Block *continueBlock = loopOp.getContinueBlock();
  builder.setInsertionPointToEnd(continueBlock);
  Value updatedIndVar =
      builder.create<spirv::IAddOp>(loc, newIndVar.getType(), newIndVar, step);
  builder.create<spirv::BranchOp>(loc, header, updatedIndVar);
  builder.setInsertionPointToStart(body);
  return newIndVar;
}

LogicalResult SPIRVCodegenImpl::createLaunchLoop(
    OpBuilder &builder, Operation *op, ArrayRef<Value> domainSize,
    SmallVectorImpl<Value> &dimValues) {
  if (domainSize.empty()) {
    return success();
  }
  auto loc = op->getLoc();
  Value ub = nullptr;
  if (domainSize.size() >= 3) {
    ub = domainSize[2];
    for (auto i : llvm::seq<unsigned>(3, domainSize.size())) {
      ub = builder.create<spirv::IMulOp>(loc, ub.getType(), ub, domainSize[i]);
    }
  } else {
    ub = domainSize.back();
  }
  auto dim = std::min<unsigned>(domainSize.size() - 1, 2);
  Value gID = spirv::getBuiltinVariableValue(
      op, spirv::BuiltIn::GlobalInvocationId, builder);
  Value lb = builder.create<spirv::CompositeExtractOp>(loc, gID, dim);
  Value numWorkGroupsID = spirv::getBuiltinVariableValue(
      op, spirv::BuiltIn::NumWorkgroups, builder);
  Value numWorkGroups =
      builder.create<spirv::CompositeExtractOp>(loc, numWorkGroupsID, dim);
  // Get the workgroup size from the entry_point_abi attr.
  DenseIntElementsAttr workGroupSizeAttr = spirv::lookupLocalWorkGroupSize(op);
  auto workGroupSizeVal = workGroupSizeAttr.getValue<APInt>(dim).getSExtValue();
  auto integerType = IntegerType::get(32, builder.getContext());
  auto workGroupSize = builder.create<spirv::ConstantOp>(
      loc, integerType, builder.getI32IntegerAttr(workGroupSizeVal));
  Value step = builder.create<spirv::IMulOp>(loc, numWorkGroups.getType(),
                                             numWorkGroups, workGroupSize);

  Value inductionVar = createLoop(builder, loc, lb, ub, step);

  if (domainSize.size() > 3) {
    for (auto i : llvm::seq<unsigned>(2, domainSize.size())) {
      dimValues.push_back(builder.create<spirv::SModOp>(
          loc, inductionVar.getType(), inductionVar, domainSize[i]));
      inductionVar = builder.create<spirv::SDivOp>(loc, inductionVar.getType(),
                                                   inductionVar, domainSize[i]);
    }
  } else {
    dimValues.insert(dimValues.begin(), inductionVar);
  }
  return createLaunchLoop(
      builder, op,
      domainSize.drop_back((domainSize.size() > 3 ? domainSize.size() - 2 : 1)),
      dimValues);
}

LogicalResult SPIRVCodegenImpl::initArgValues(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value origArg) {
  SmallVector<AffineMap, 4> indices;
  getIndexMapsForValue(origArg, indices);
  for (auto indexMap : indices) {
    if (!loadArgValueAtIndex(builder, loc, valueCache, origArg, indexMap)) {
      return failure();
    }
  }
  return success();
}

LogicalResult SPIRVCodegenImpl::initSymbolValues(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value origArg) {
  // Add values corresponding to the symbol numbers.
  SmallVector<std::pair<AffineMap, unsigned>, 2> symbolInfo;
  getSymbolNumberForTensorIndex(origArg.cast<BlockArgument>(), symbolInfo);
  for (auto element : symbolInfo) {
    // Load the value at the index.
    auto val =
        loadArgValueAtIndex(builder, loc, valueCache, origArg, element.first);
    if (!val) {
      return failure();
    }
    // Convert to i32.
    auto valIntType = val.getType().dyn_cast<IntegerType>();
    if (!valIntType) {
      return emitError(loc, "expected symbol value to be integer type, got ")
             << valIntType;
    }
    if (valIntType.getWidth() != 32) {
      auto i32Type = builder.getIntegerType(32);
      val = builder.create<spirv::SConvertOp>(loc, i32Type, val);
    }
    valueCache.setSymbolValue(element.second, val);
  }
  return success();
}

Value SPIRVCodegenImpl::loadArgValueAtIndex(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value origArg, AffineMap indexMap) {
  Value val = valueCache.getValueAtIndex(origArg, indexMap);
  if (val) {
    return val;
  }
  auto var = valueCache.getBufferForArgument(origArg);
  if (!var) {
    emitError(loc, "unable to find buffer for tensor argument");
    return nullptr;
  }
  auto ptr =
      genPointerOffset(builder, loc, valueCache, indexMap,
                       origArg.getType().cast<ShapedType>().getShape(), var);
  val = builder.create<spirv::LoadOp>(loc, ptr,
                                      /*memory_access =*/nullptr,
                                      /*alignment = */ nullptr);
  valueCache.setValueAtIndex(origArg, indexMap, val);
  return val;
}

LogicalResult SPIRVCodegenImpl::lowerFunction(
    OpBuilder &builder, FuncOp fn, spirv::FuncOp entryFn,
    TensorIndexToScalarValueMap &valueCache) {
  SmallVector<int64_t, 3> launchSize;
  if (failed(getLaunchSize(fn, launchSize))) {
    return failure();
  }
  // TODO(ravishankarm): The launch size is for now obtained from static shapes,
  // but they will be dynamic soon. So just create constants for now and adapt
  // later for dynamic shapes.
  SmallVector<Value, 3> launchSizeVal, dimValues;
  launchSizeVal.reserve(launchSize.size());
  auto integerType = IntegerType::get(32, builder.getContext());
  for (auto size : launchSize) {
    launchSizeVal.push_back(builder.create<spirv::ConstantOp>(
        fn.getLoc(), integerType, builder.getI32IntegerAttr(size)));
  }
  if (launchSizeVal.size() < 3)
    launchSizeVal.resize(
        3, spirv::ConstantOp::getOne(integerType, fn.getLoc(), &builder));
  if (failed(createLaunchLoop(builder, entryFn, launchSizeVal, dimValues))) {
    return failure();
  }
  assert(launchSizeVal.size() == dimValues.size());
  for (auto id : enumerate(dimValues)) {
    valueCache.setDimValue(id.index(), id.value());
  }

  for (auto arg : fn.getArguments()) {
    // Load values of the argument at all indices needed for computation
    // within the dispatch function.
    if (failed(initSymbolValues(builder, fn.getLoc(), valueCache, arg))) {
      return failure();
    }
  }

  for (auto arg : fn.getArguments()) {
    if (fn.getArgAttrOfType<UnitAttr>(arg.getArgNumber(),
                                      "iree.executable.reduction.output")) {
      continue;
    }
    // Load values of the argument at all indices needed for computation
    // within the dispatch function.
    if (failed(initArgValues(builder, fn.getLoc(), valueCache, arg))) {
      return failure();
    }
  }

  for (auto &block : fn) {
    for (auto &op : block) {
      // Lower individual operations.
      if (failed(lowerOperation(builder, valueCache, &op))) {
        return failure();
      }
    }
  }
  builder.setInsertionPointToEnd(&(entryFn.getBody().back()));
  builder.create<spirv::ReturnOp>(fn.getLoc());
  return success();
}
}  // namespace detail

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

static LogicalResult lowerSplatConstant(
    Operation *op, OpBuilder &builder, AffineMap index,
    TensorIndexToScalarValueMap &valueCache) {
  auto constOp = cast<ConstantOp>(op);
  auto attr = constOp.value().dyn_cast<DenseElementsAttr>();
  auto resultType = constOp.getResult().getType();
  Type resultElemType;
  if (resultType.isSignlessIntOrFloat()) {
    resultElemType = resultType;
  } else if (auto shapedType = resultType.dyn_cast<ShapedType>()) {
    resultElemType = shapedType.getElementType();
  } else {
    return op->emitError("unhandled result type of constant : ") << resultType;
  }
  Attribute constVal = attr.getSplatValue();
  auto spirvConstOp =
      builder.create<spirv::ConstantOp>(op->getLoc(), resultElemType, constVal);
  valueCache.setValueAtIndex(constOp.getResult(), index,
                             spirvConstOp.getResult());
  return success();
}

static LogicalResult lowerNonSplatConstant(
    Operation *op, OpBuilder &builder, AffineMap index,
    TensorIndexToScalarValueMap &valueCache) {
  auto constOp = cast<ConstantOp>(op);
  auto loc = constOp.getLoc();
  auto argType = constOp.getType().dyn_cast<ShapedType>();
  auto elementType = argType.getElementType();

  if (!argType.hasStaticShape()) {
    return constOp.emitError("expected static shaped tensor");
  }
  if (!elementType.isSignlessIntOrFloat()) {
    return op->emitError("unhandled element type of the result :")
           << elementType;
  }

  // Build the array type.
  int64_t stride = elementType.getIntOrFloatBitWidth() / 8;
  Attribute spirvConstAttr;
  if (argType.getRank()) {
    // Create a DenseElementsAttr that is a linearized version of a the
    // attribute in the original operation.
    // TODO: This is copying the values temporarily (the old one will be
    // deleted), but potentially can "move" the data here.
    auto tensorType =
        RankedTensorType::get(argType.getNumElements(), elementType);
    if (elementType.isSignlessInteger()) {
      auto values =
          constOp.value().cast<DenseElementsAttr>().getValues<APInt>();
      SmallVector<APInt, 4> valuesVec(values.begin(), values.end());
      spirvConstAttr = DenseElementsAttr::get(tensorType, valuesVec);
    } else {
      auto values =
          constOp.value().cast<DenseElementsAttr>().getValues<APFloat>();
      SmallVector<APFloat, 4> valuesVec(values.begin(), values.end());
      spirvConstAttr = DenseElementsAttr::get(tensorType, valuesVec);
    }
    elementType = spirv::ArrayType::get(
        elementType, argType.getNumElements(),
        static_cast<spirv::ArrayType::LayoutInfo>(stride));
  } else {
    spirvConstAttr = constOp.value();
  }
  auto pointerType =
      spirv::PointerType::get(elementType, spirv::StorageClass::Function);
  auto spirvConstOp =
      builder.create<spirv::ConstantOp>(loc, elementType, spirvConstAttr);
  auto spirvVarOp = builder.create<spirv::VariableOp>(
      loc, pointerType,
      builder.getI32IntegerAttr(
          static_cast<int32_t>(spirv::StorageClass::Function)),
      ArrayRef<Value>(spirvConstOp.getResult()));

  Value accessIndex = valueCache.getAccessIndicesForIndexMap(
      builder, loc, index, argType.getShape());
  if (!accessIndex) {
    return failure();
  }
  auto valPtr =
      builder.create<spirv::AccessChainOp>(loc, spirvVarOp, accessIndex);
  auto val =
      builder.create<spirv::LoadOp>(loc, valPtr, /*memory_access=*/nullptr,
                                    /*alignment=*/nullptr);
  valueCache.setValueAtIndex(op->getResult(0), index, val.getResult());
  return success();
}

LogicalResult ConstantOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index, ArrayRef<Value>,
    TensorIndexToScalarValueMap &valueCache) const {
  auto constOp = cast<ConstantOp>(op);
  auto attr = constOp.value().dyn_cast<DenseElementsAttr>();
  if (!attr || !attr.isSplat()) {
    return lowerNonSplatConstant(op, builder, index, valueCache);
  }
  return lowerSplatConstant(op, builder, index, valueCache);
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//
LogicalResult CmpIOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value> operands, TensorIndexToScalarValueMap &valueCache) const {
  if (operands.size() != 2) {
    return op->emitError("expected two operands in spir-v lowering of CmpIOp");
  }
  Operation *spirvOp = nullptr;
  auto opInfo = op->getAttrOfType<IntegerAttr>(CmpIOp::getPredicateAttrName());
  if (!opInfo) {
    return op->emitError("expected CmpIOp to contain ")
           << CmpIOp::getPredicateAttrName() << " attribute";
  }
  auto boolType = builder.getI1Type();
  auto predicateVal = static_cast<CmpIPredicate>(opInfo.getInt());

#define DISPATCH(caseLabel, opName)                                       \
  case caseLabel:                                                         \
    spirvOp = builder.create<opName>(op->getLoc(), boolType, operands[0], \
                                     operands[1]);                        \
    break;

  // Handle i1 type differently because SPIR-V arithmatic ops by default don't
  // support i1 types.
  if (operands[0].getType().cast<IntegerType>().getWidth() == 1) {
    switch (predicateVal) {
      DISPATCH(CmpIPredicate::eq, spirv::LogicalEqualOp);
      DISPATCH(CmpIPredicate::ne, spirv::LogicalNotEqualOp);
      default:
        return op->emitError(
            "unhandled predicate attribute for SPIR-V lowering of i1 type");
    }
  } else {
    switch (predicateVal) {
      DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
      DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
      DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
      DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
      DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
      DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);
      default:
        return op->emitError(
            "unhandled predicate attribute for SPIR-V lowering");
    }
  }

#undef DISPATCH

  valueCache.setValueAtIndex(op->getResult(0), index, spirvOp->getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
LogicalResult CmpFOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value> operands, TensorIndexToScalarValueMap &valueCache) const {
  if (operands.size() != 2) {
    return op->emitError("expected two operands in spir-v lowering of CmpFOp");
  }
  Operation *spirvOp = nullptr;
  auto opInfo = op->getAttrOfType<IntegerAttr>(CmpFOp::getPredicateAttrName());
  if (!opInfo) {
    return op->emitError("expected CmpFOp to contain ")
           << CmpFOp::getPredicateAttrName() << " attribute";
  }
  auto boolType = builder.getI1Type();
  auto predicateVal = static_cast<CmpFPredicate>(opInfo.getInt());
  switch (predicateVal) {
#define DISPATCH(caseLabel, opName)                                       \
  case caseLabel:                                                         \
    spirvOp = builder.create<opName>(op->getLoc(), boolType, operands[0], \
                                     operands[1]);                        \
    break;

    DISPATCH(CmpFPredicate::OEQ, spirv::FOrdEqualOp);
    DISPATCH(CmpFPredicate::OGE, spirv::FOrdGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::OGT, spirv::FOrdGreaterThanOp);
    DISPATCH(CmpFPredicate::OLE, spirv::FOrdLessThanEqualOp);
    DISPATCH(CmpFPredicate::OLT, spirv::FOrdLessThanOp);
    DISPATCH(CmpFPredicate::ONE, spirv::FOrdNotEqualOp);
    DISPATCH(CmpFPredicate::UEQ, spirv::FUnordEqualOp);
    DISPATCH(CmpFPredicate::UGE, spirv::FUnordGreaterThanEqualOp);
    DISPATCH(CmpFPredicate::UGT, spirv::FUnordGreaterThanOp);
    DISPATCH(CmpFPredicate::ULE, spirv::FUnordLessThanEqualOp);
    DISPATCH(CmpFPredicate::ULT, spirv::FUnordLessThanOp);
    DISPATCH(CmpFPredicate::UNE, spirv::FUnordNotEqualOp);

#undef DISPATCH

    default:
      return op->emitError("unhandled predicate attribute for SPIR-V lowering");
  }
  valueCache.setValueAtIndex(op->getResult(0), index, spirvOp->getResult(0));
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
