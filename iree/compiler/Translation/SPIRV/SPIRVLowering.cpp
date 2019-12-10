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
#include "iree/compiler/Translation/SPIRV/SPIRVLowering.h"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// SPIR-V codegen implementation
//===----------------------------------------------------------------------===//

Value *genPointerOffset(OpBuilder &builder, Location loc,
                        TensorIndexToScalarValueMap &valueCache,
                        AffineMap indexMap, spirv::GlobalVariableOp var) {
  auto basePtr = builder.create<spirv::AddressOfOp>(loc, var);
  auto varPtrType = var.type().cast<spirv::PointerType>().getPointeeType();
  // The variable has to be a struct type with a single element.
  assert(varPtrType.isa<spirv::StructType>() &&
         "expected variable type to be a spv.ptr<spv.struct<...>>");
  auto varStructType = varPtrType.cast<spirv::StructType>();
  assert(varStructType.getNumElements() == 1 &&
         "expected variable type to be a spv.ptr of spv.struct with a single "
         "element");
  auto varType = varStructType.getElementType(0);

  SmallVector<Value *, 2> accessIndices;
  /// For scalar values, the index-map computed with already map to the 0-th
  /// element. For arrays, they map to the position accessed. So just for arrays
  /// we need to add an extra 0 to index into the struct.
  auto i32Type = builder.getIntegerType(32);
  if (varType.isa<spirv::ArrayType>()) {
    auto zero = builder.create<spirv::ConstantOp>(loc, i32Type,
                                                  builder.getI32IntegerAttr(0));
    accessIndices.push_back(zero);
  }
  for (auto indexExpr : indexMap.getResults()) {
    accessIndices.push_back(valueCache.getAffineExprValue(
        builder.saveInsertionPoint(), loc, indexExpr));
  }
  return builder.create<spirv::AccessChainOp>(loc, basePtr, accessIndices);
}

/// Returns the type of the global variable to use for an argument of the
/// dispatch function.
static spirv::PointerType convertArgTypeToVariableType(Location loc,
                                                       MemRefType argType) {
  auto elementType = argType.getElementType();
  if (!elementType.isIntOrFloat()) {
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
  for (auto dim : reverse(argType.getShape())) {
    if (dim <= 0) {
      emitError(loc, "expected tensor dimensions to be non-zero");
      return nullptr;
    }
    elementType = spirv::ArrayType::get(
        elementType, dim, static_cast<spirv::ArrayType::LayoutInfo>(stride));
    stride *= dim;
  }
  return spirv::PointerType::get(
      spirv::StructType::get(elementType,
                             static_cast<spirv::StructType::LayoutInfo>(0)),
      spirv::StorageClass::StorageBuffer);
}

/// Returns a spirv::GlobalVariable op for a give argument of type `argType` and
/// argument idnex `argIndex`.
static spirv::GlobalVariableOp createGlobalVariableForArg(Location loc,
                                                          OpBuilder &builder,
                                                          unsigned argIndex,
                                                          Type argType,
                                                          StringRef varName) {
  auto argMemrefType = argType.dyn_cast<MemRefType>();
  if (!argMemrefType) {
    emitError(loc, "unhandled non-memref type argument in SPIR-V lowering");
    return nullptr;
  }
  auto varType = convertArgTypeToVariableType(loc, argMemrefType);
  if (!varType) {
    return nullptr;
  }
  auto var = builder.create<spirv::GlobalVariableOp>(loc, varType, varName, 0,
                                                     argIndex);
  return var;
}

static spirv::GlobalVariableOp createGlobalInvocationID(OpBuilder &builder,
                                                        Location loc) {
  auto i32Type = builder.getIntegerType(32);
  auto idType = VectorType::get(3, i32Type);
  auto ptrIdType = spirv::PointerType::get(idType, spirv::StorageClass::Input);
  return builder.create<spirv::GlobalVariableOp>(
      loc, ptrIdType, "globalInvocationID", spirv::BuiltIn::GlobalInvocationId);
}

namespace detail {
LogicalResult SPIRVCodegenImpl::createEntryFn(
    OpBuilder &builder, FuncOp &fn, TensorIndexToScalarValueMap &valueCache) {
  auto loc = fn.getLoc();

  // Convert functions arguments and return values to
  // spirv::GlobalVariables. All global variables are given a descriptor set
  // of 0 and binding is the argument number.
  auto fnType = fn.getType();
  for (auto argType : enumerate(fnType.getInputs())) {
    auto &var = inputArgToVariable[fn.getArgument(argType.index())];
    auto varName =
        fn.getName().str() + "_arg_" + std::to_string(argType.index());
    var = createGlobalVariableForArg(loc, builder, argType.index(),
                                     argType.value(), varName);
    if (!var) {
      return failure();
    }
  }
  resultIndexToVariable.resize(fnType.getNumResults());
  for (auto resType : enumerate(fnType.getResults())) {
    auto &var = resultIndexToVariable[resType.index()];
    auto varName =
        fn.getName().str() + "_result_" + std::to_string(resType.index());
    var = createGlobalVariableForArg(loc, builder,
                                     resType.index() + fnType.getNumInputs(),
                                     resType.value(), varName);
    if (!var) {
      return failure();
    }
  }

  // Create the Global invocation ID.
  auto globalInvocationID = createGlobalInvocationID(builder, fn.getLoc());
  interface.push_back(builder.getSymbolRefAttr(globalInvocationID.sym_name()));

  auto entryFnType =
      builder.getFunctionType(ArrayRef<Type>(), ArrayRef<Type>());
  auto entryFn = builder.create<FuncOp>(loc, fn.getName(), entryFnType,
                                        ArrayRef<NamedAttribute>());

  // Start a scope to create an insertion guard to reset the builder once the
  // function is lowered.
  {
    assert(globalInvocationIDs.empty());
    OpBuilder::InsertionGuard funcInsertGuard(builder);
    builder.setInsertionPointToStart(entryFn.addEntryBlock());

    auto globalInvocationIDPtr =
        builder.create<spirv::AddressOfOp>(loc, globalInvocationID);
    auto id = builder.create<spirv::LoadOp>(loc, globalInvocationIDPtr, nullptr,
                                            nullptr);

    auto id_x = builder.create<spirv::CompositeExtractOp>(loc, id, 0);
    globalInvocationIDs.push_back(id_x);
    auto id_y = builder.create<spirv::CompositeExtractOp>(loc, id, 1);
    globalInvocationIDs.push_back(id_y);
    auto id_z = builder.create<spirv::CompositeExtractOp>(loc, id, 2);
    globalInvocationIDs.push_back(id_z);

    for (auto id : enumerate(globalInvocationIDs)) {
      valueCache.setDimValue(id.index(), id.value());
    }

    if (failed(lowerFunction(builder, fn, entryFn, valueCache))) {
      return failure();
    }
  }

  // Create the entry point instructions for the entry function.
  if (failed(createEntryPoint(fn, builder, loc, entryFn))) {
    return failure();
  }
  return success();
}

LogicalResult SPIRVCodegenImpl::createEntryPoint(FuncOp fn, OpBuilder &builder,
                                                 Location loc, FuncOp entryFn) {
  builder.create<spirv::EntryPointOp>(loc, spirv::ExecutionModel::GLCompute,
                                      entryFn, interface);

  // Get the workgroup size.
  SmallVector<int32_t, 3> workGroupSize;
  if (failed(getWorkGroupSize(fn, workGroupSize))) {
    return failure();
  }
  builder.create<spirv::ExecutionModeOp>(
      loc, entryFn, spirv::ExecutionMode::LocalSize, workGroupSize);
  interface.clear();
  return success();
}

LogicalResult SPIRVCodegenImpl::createLaunchGuard(OpBuilder &builder,
                                                  FuncOp fn) {
  // First check that the global invocation id is in bounds.
  SmallVector<int64_t, 3> launchSize;
  if (failed(getLaunchSize(fn, launchSize))) {
    return failure();
  }
  auto loc = fn.getLoc();
  auto i1Type = builder.getI1Type();
  auto i32Type = builder.getIntegerType(32);
  Value *condn = spirv::ConstantOp::getOne(i1Type, loc, &builder);
  for (auto launchDim : enumerate(launchSize)) {
    if (launchDim.value() == 1) {
      continue;
    }
    Value *id = getGlobalInvocationID(launchDim.index());
    auto extent = builder.create<spirv::ConstantOp>(
        loc, i32Type, builder.getI32IntegerAttr(launchDim.value()));
    auto check = builder.create<spirv::SLessThanOp>(loc, i1Type, id, extent);
    condn = builder.create<spirv::LogicalAndOp>(loc, i1Type, condn, check);
  }
  auto selectionOp =
      builder.create<spirv::SelectionOp>(loc, spirv::SelectionControl::None);
  selectionOp.addMergeBlock();

  // Create the header block and then blocks.
  auto headerBlock = builder.createBlock(&selectionOp.body(),
                                         std::prev(selectionOp.body().end()));
  auto thenBlock = builder.createBlock(&selectionOp.body(),
                                       std::prev(selectionOp.body().end()));

  // Add branch to the header block.
  builder.setInsertionPointToEnd(headerBlock);
  builder.create<spirv::BranchConditionalOp>(
      loc, condn, thenBlock, ArrayRef<Value *>(), selectionOp.getMergeBlock(),
      ArrayRef<Value *>());

  // Add branch to merge block in the then block.
  builder.setInsertionPointToEnd(thenBlock);
  auto branchOp =
      builder.create<spirv::BranchOp>(loc, selectionOp.getMergeBlock());
  builder.setInsertionPoint(branchOp);
  return success();
}

Value *SPIRVCodegenImpl::getGlobalInvocationID(unsigned dim) {
  if (dim < globalInvocationIDs.size()) {
    return globalInvocationIDs[dim];
  }
  return nullptr;
}

LogicalResult SPIRVCodegenImpl::initArgValues(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value *origArg) {
  SmallVector<AffineMap, 4> indices;
  index_computation_attribute::getIndexMapsForValue(origArg, indices);
  for (auto indexMap : indices) {
    if (!loadArgValueAtIndex(builder, loc, valueCache, origArg, indexMap)) {
      return failure();
    }
  }
  return success();
}

LogicalResult SPIRVCodegenImpl::initSymbolValues(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value *origArg) {
  // Add values corresponding to the symbol numbers.
  SmallVector<std::pair<AffineMap, unsigned>, 2> symbolInfo;
  index_computation_attribute::getSymbolNumberForTensorIndex(
      cast<BlockArgument>(origArg), symbolInfo);
  for (auto element : symbolInfo) {
    // Load the value at the index.
    auto val =
        loadArgValueAtIndex(builder, loc, valueCache, origArg, element.first);
    if (!val) {
      return failure();
    }
    valueCache.setSymbolValue(element.second, val);
  }
  return success();
}

Value *SPIRVCodegenImpl::loadArgValueAtIndex(
    OpBuilder &builder, Location loc, TensorIndexToScalarValueMap &valueCache,
    Value *origArg, AffineMap indexMap) {
  Value *val = valueCache.getValueAtIndex(origArg, indexMap);
  if (val) {
    return val;
  }
  auto var = inputArgToVariable.lookup(origArg);
  if (!var) {
    emitError(loc, "undefined SPIR-V global variable for tensor argument");
    return nullptr;
  }
  auto ptr = genPointerOffset(builder, loc, valueCache, indexMap, var);
  val = builder.create<spirv::LoadOp>(loc, ptr,
                                      /*memory_access =*/nullptr,
                                      /*alignment = */ nullptr);
  valueCache.setValueAtIndex(origArg, indexMap, val);
  return val;
}

LogicalResult SPIRVCodegenImpl::lowerFunction(
    OpBuilder &builder, FuncOp fn, FuncOp entryFn,
    TensorIndexToScalarValueMap &valueCache) {
  if (failed(createLaunchGuard(builder, fn))) {
    return failure();
  }

  for (auto arg : fn.getArguments()) {
    // Load values of the argument at all indices needed for computation
    // within the dispatch function.
    if (failed(initSymbolValues(builder, fn.getLoc(), valueCache, arg))) {
      return failure();
    }
  }

  for (auto arg : fn.getArguments()) {
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
LogicalResult ConstantOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index, ArrayRef<Value *>,
    TensorIndexToScalarValueMap &valueCache) const {
  auto constOp = cast<ConstantOp>(op);
  auto attr = constOp.value().dyn_cast<DenseElementsAttr>();
  if (!attr || !attr.isSplat()) {
    return lowerNonSplatConstant(op, builder, index, valueCache);
  }
  return lowerSplatConstant(op, builder, index, valueCache);
}

LogicalResult ConstantOpSPIRVLowering::lowerSplatConstant(
    Operation *op, OpBuilder &builder, AffineMap index,
    TensorIndexToScalarValueMap &valueCache) const {
  auto constOp = cast<ConstantOp>(op);
  auto attr = constOp.value().dyn_cast<DenseElementsAttr>();
  auto resultType = constOp.getResult()->getType();
  Type resultElemType;
  if (resultType.isIntOrFloat()) {
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

LogicalResult ConstantOpSPIRVLowering::lowerNonSplatConstant(
    Operation *op, OpBuilder &builder, AffineMap index,
    TensorIndexToScalarValueMap &valueCache) const {
  auto constOp = cast<ConstantOp>(op);
  auto loc = constOp.getLoc();
  auto argType = constOp.getType().dyn_cast<ShapedType>();
  auto elementType = argType.getElementType();

  if (!argType.hasStaticShape()) {
    return constOp.emitError("expected static shaped tensor");
  }

  // Build the array type.
  int64_t stride = elementType.getIntOrFloatBitWidth() / 8;
  for (auto dim : reverse(argType.getShape())) {
    elementType = spirv::ArrayType::get(
        elementType, dim, static_cast<spirv::ArrayType::LayoutInfo>(stride));
    stride *= dim;
  }

  auto pointerType =
      spirv::PointerType::get(elementType, spirv::StorageClass::Function);
  auto spirvConstOp =
      builder.create<spirv::ConstantOp>(loc, elementType, constOp.value());
  auto spirvVarOp = builder.create<spirv::VariableOp>(
      loc, pointerType,
      builder.getI32IntegerAttr(
          static_cast<int32_t>(spirv::StorageClass::Function)),
      ArrayRef<Value *>(spirvConstOp.getResult()));

  SmallVector<Value *, 2> accessIndices;
  for (auto indexExpr : index.getResults()) {
    accessIndices.push_back(valueCache.getAffineExprValue(
        builder.saveInsertionPoint(), loc, indexExpr));
  }
  auto valPtr =
      builder.create<spirv::AccessChainOp>(loc, spirvVarOp, accessIndices);
  auto val =
      builder.create<spirv::LoadOp>(loc, valPtr, /*memory_access=*/nullptr,
                                    /*alignment=*/nullptr);
  valueCache.setValueAtIndex(op->getResult(0), index, val.getResult());
  return success();
}

//===----------------------------------------------------------------------===//
// CmpIOp
//===----------------------------------------------------------------------===//
LogicalResult CmpIOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, TensorIndexToScalarValueMap &valueCache) const {
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
  switch (predicateVal) {
#define DISPATCH(caseLabel, opName)                                       \
  case caseLabel:                                                         \
    spirvOp = builder.create<opName>(op->getLoc(), boolType, operands[0], \
                                     operands[1]);                        \
    break;

    DISPATCH(CmpIPredicate::eq, spirv::IEqualOp);
    DISPATCH(CmpIPredicate::ne, spirv::INotEqualOp);
    DISPATCH(CmpIPredicate::slt, spirv::SLessThanOp);
    DISPATCH(CmpIPredicate::sle, spirv::SLessThanEqualOp);
    DISPATCH(CmpIPredicate::sgt, spirv::SGreaterThanOp);
    DISPATCH(CmpIPredicate::sge, spirv::SGreaterThanEqualOp);

#undef DISPATCH

    default:
      return op->emitError("unhandled predicate attribute for SPIR-V lowering");
  }
  valueCache.setValueAtIndex(op->getResult(0), index, spirvOp->getResult(0));
  return success();
}

//===----------------------------------------------------------------------===//
// CmpFOp
//===----------------------------------------------------------------------===//
LogicalResult CmpFOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, AffineMap index,
    ArrayRef<Value *> operands, TensorIndexToScalarValueMap &valueCache) const {
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

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//
LogicalResult ReturnOpSPIRVLowering::lowerOperation(
    Operation *op, OpBuilder &builder, TensorIndexToScalarValueMap &valueCache,
    DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
    ArrayRef<spirv::GlobalVariableOp> outputBuffers) const {
  auto returnOp = cast<ReturnOp>(op);
  if (returnOp.getNumOperands() != 1) {
    return returnOp.emitError(
        "unhandled lowering of return statement with multiple returns");
  }
  auto returnTensor = returnOp.getOperand(0);
  SmallVector<AffineMap, 1> indices;
  index_computation_attribute::getIndexMapsForValue(returnTensor, indices);
  if (indices.size() != 1) {
    return returnOp.emitError(
        "expected to compute a single element of the return tensor");
  }
  assert(outputBuffers.size() == 1 && "Expected a single output buffer");
  auto var = outputBuffers[0];
  auto ptr =
      genPointerOffset(builder, returnOp.getLoc(), valueCache, indices[0], var);
  auto scalarVal = valueCache.getValueAtIndex(returnTensor, indices[0]);
  builder.create<spirv::StoreOp>(returnOp.getLoc(), ptr, scalarVal,
                                 /*memory_access = */ nullptr,
                                 /*alignment = */ nullptr);
  builder.create<spirv::ReturnOp>(returnOp.getLoc());
  return success();
}
}  // namespace iree_compiler
}  // namespace mlir
