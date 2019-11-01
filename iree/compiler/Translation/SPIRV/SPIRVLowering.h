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

//===- SPIRVLowering.h -----------------------------------------*- C++//-*-===//
//
// SPIR-V Code-generation for tensor operations within IREE Dispatch functions
//
//===----------------------------------------------------------------------===//
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_SPIRVLOWERING_H
#define IREE_COMPILER_TRANSLATION_SPIRV_SPIRVLOWERING_H

#include "iree/compiler/Translation/SPIRV/AffineExprCodegen.h"
#include "iree/compiler/Translation/SPIRV/IREECodegenUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Support/StringExtras.h"

namespace mlir {
namespace iree_compiler {

class ValueCache {
 public:
  Value *getOperandDstValue(Value *value, AffineMap index) {
    return convertedValueMap.lookup(value).lookup(index);
  }

  void setOperandDstValue(Value *value, AffineMap index, Value *scalar) {
    convertedValueMap[value][index] = scalar;
  }

 private:
  DenseMap<Value *, DenseMap<AffineMap, Value *>> convertedValueMap;
};

/// Base class for lowering tensor operations in the dispatch function to SPIR-V
/// op.
class SPIRVLowering {
 public:
  virtual ~SPIRVLowering() = default;
  virtual StringRef getOpName() = 0;
  /// This method (in the derived class) should generate the scalar operation
  /// corresponding the the tensor operation `op` to generate the value of the
  /// result tensor at a particular `index`. The scalar value of the operands
  /// needed to compute this value is passed in within `operands`. The methods
  /// have to insert the scalar result value of the generated operation into the
  /// `valueCache`.
  virtual LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                                       AffineMap index,
                                       ArrayRef<Value *> operands,
                                       AffineExprCodegen &affineExprCodegen,
                                       ValueCache &valueCache) const {
    return failure();
  }

  /// This method (in the derived class) should generate the scalar operations
  /// corresponding to the tensor operation `op`. This should be implemented
  /// when the `op` has no result value, typically store operations and return
  /// operations.
  virtual LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineExprCodegen &affineExprCodegen,
      ValueCache &valueCache,
      DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
      ArrayRef<spirv::GlobalVariableOp> outputBuffers) const {
    return failure();
  }
};

/// Base class that gets the opName for the operation.
template <typename OpTy>
class SPIRVOpLowering : public SPIRVLowering {
 public:
  using SPIRVLowering::SPIRVLowering;
  virtual ~SPIRVOpLowering<OpTy>() {}
  StringRef getOpName() override { return OpTy::getOperationName(); }
};

/// SPIR-V lowering for ConstantOp.
class ConstantOpSPIRVLowering final : public SPIRVOpLowering<ConstantOp> {
 public:
  using SPIRVOpLowering<ConstantOp>::SPIRVOpLowering;
  LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                               AffineMap index, ArrayRef<Value *> operands,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache) const override;
};

/// SPIR-V lowering for CmpFOp.
class CmpFOpSPIRVLowering final : public SPIRVOpLowering<CmpFOp> {
 public:
  using SPIRVOpLowering<CmpFOp>::SPIRVOpLowering;

  LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                               AffineMap index, ArrayRef<Value *> operands,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache) const override;
};

/// SPIR-V lowering for Min/Max operations.
template <typename OpTy, typename CmpOpTy, typename CmpFOpTy>
class CmpSelectOpSPIRVLowering final : public SPIRVOpLowering<OpTy> {
 public:
  using SPIRVOpLowering<OpTy>::SPIRVOpLowering;
  LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                               AffineMap index, ArrayRef<Value *> operands,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache) const override {
    if (op->getNumOperands() != 2) {
      return op->emitError(
          "unhandled SPIR-V lowering for more than 2 operands");
    }
    assert(operands.size() == op->getNumOperands() &&
           "expected as many operands for the replacement as the original "
           "instruction");
    auto cmpSelectOp = cast<OpTy>(op);
    auto result = cmpSelectOp.getResult();
    auto resultTy = result->getType().template dyn_cast<ShapedType>();
    if (!resultTy) {
      return op->emitError(
          "unhandled lowering of operations that don't return a "
          "ShapedType");
    }
    auto elementTy = resultTy.getElementType();
    auto boolTy = builder.getI1Type();
    Operation *cmpOp = nullptr;
    if (elementTy.template isa<FloatType>()) {
      cmpOp = builder.create<CmpFOpTy>(op->getLoc(), boolTy, operands,
                                       ArrayRef<NamedAttribute>());
    } else {
      cmpOp = builder.create<CmpOpTy>(op->getLoc(), boolTy, operands,
                                      ArrayRef<NamedAttribute>());
    }
    auto selectOp = builder.create<spirv::SelectOp>(
        op->getLoc(), operands[0]->getType(), cmpOp->getResult(0), operands[0],
        operands[1]);
    valueCache.setOperandDstValue(op->getResult(0), index,
                                  selectOp.getResult());
    return success();
  }
};

/// This class is the general template used to emit scalar instruction
/// corresponding for point-wise operations. Assumes that the original
/// instruction has a single result value of type ShapedType.
/// TODO(ravishankarm) : In XLA-HLO, the same operations is used for
/// integer/float tensor operations. So allow this op to take an additional op
/// type as a template parameter to handle such cases. Find a better way to do
/// this.
template <typename OpTy, typename ReplacementOpTy,
          typename FloatOpTy = ReplacementOpTy>
class SPIRVPwOpLowering final : public SPIRVOpLowering<OpTy> {
 public:
  using SPIRVOpLowering<OpTy>::SPIRVOpLowering;

  LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                               AffineMap index,
                               ArrayRef<Value *> scalarOperands,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache) const override {
    // TODO(ravishankarm) : This check should really be a static_assert. See if
    // that can be changed.
    if (op->getNumOperands() == 0) {
      return op->emitError("expected op to have at least one operand");
    }
    auto pwOp = cast<OpTy>(op);
    auto result = pwOp.getResult();
    auto resultType = result->getType().template dyn_cast<ShapedType>();
    if (!resultType) {
      return op->emitError(
          "unhandled lowering of operations that don't return a "
          "ShapedType");
    }
    auto elementType = resultType.getElementType();
    Operation *scalarOp = nullptr;
    if (elementType.template isa<IntegerType>()) {
      scalarOp = builder
                     .create<ReplacementOpTy>(op->getLoc(), elementType,
                                              scalarOperands,
                                              ArrayRef<NamedAttribute>())
                     .getOperation();
    } else {
      scalarOp =
          builder
              .create<FloatOpTy>(op->getLoc(), elementType, scalarOperands,
                                 ArrayRef<NamedAttribute>())
              .getOperation();
    }
    if (!scalarOp) {
      return op->emitError("unable to lower operation");
    }
    valueCache.setOperandDstValue(pwOp.getResult(), index,
                                  scalarOp->getResult(0));
    return success();
  }
};

/// This class is the general template used to emit scalar instruction for index
/// transformation instructions like transpose. Assumes a single result value
/// and a single operand
template <typename OpTy>
class SPIRVIndexOpLowering final : public SPIRVOpLowering<OpTy> {
 public:
  using SPIRVOpLowering<OpTy>::SPIRVOpLowering;

  LogicalResult lowerOperation(Operation *op, OpBuilder &builder,
                               AffineMap index,
                               ArrayRef<Value *> scalarOperands,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache) const override {
    if (op->getNumOperands() != 1) {
      return op->emitError(
          "unhandled lowering of index transformation operation with multiple "
          "operands");
    }
    auto indexOp = cast<OpTy>(op);
    valueCache.setOperandDstValue(indexOp.getResult(), index,
                                  scalarOperands[0]);
    return success();
  }
};

/// Generates spv.AccessChain instruction to get the pointer value at a given
/// location of a spv.globalVariable.
inline Value *genPointerOffset(OpBuilder &builder, Location loc,
                               AffineExprCodegen &affineExprCodegen,
                               AffineMap indexMap,
                               spirv::GlobalVariableOp &var) {
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
    accessIndices.push_back(affineExprCodegen.getValue(
        indexExpr, builder.saveInsertionPoint(), loc));
  }
  return builder.create<spirv::AccessChainOp>(loc, basePtr, accessIndices);
}

/// Lower return statements during SPIR-V codegeneration.
class ReturnOpSPIRVLowering : public SPIRVOpLowering<ReturnOp> {
 public:
  using SPIRVOpLowering<ReturnOp>::SPIRVOpLowering;

  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineExprCodegen &affineExprCodegen,
      ValueCache &valueCache,
      DenseMap<Value *, spirv::GlobalVariableOp> &inputBuffers,
      ArrayRef<spirv::GlobalVariableOp> outputBuffers) const override;
};

/// Class to drive the SPIRV code-generation.
template <typename... Ts>
class SPIRVCodegen {
  using OpCodegenListT = llvm::StringMap<std::unique_ptr<SPIRVLowering>>;

 public:
  explicit SPIRVCodegen() { insert(); }

  LogicalResult codegen(spirv::ModuleOp &spirvModule, FuncOp &fn,
                        AffineExprCodegen &affineExprCodegen,
                        ValueCache &valueCache) {
    if (fn.getBlocks().size() != 1) {
      return emitError(
          fn.getLoc(),
          "unimplemeneted handling multiple blocks within a function");
    }

    OpBuilder builder(spirvModule.body());
    // Create the entry function and generate global invocation ID. Creates a
    // global variable for all inputs and output tensors.
    return createEntryFn(builder, fn, affineExprCodegen, valueCache);
  }

 private:
  /// Returns the type of the SPIR-V GlobalVariable for an argument of the
  /// dispatch function.
  LogicalResult convertArgTypeToVariableType(Location loc, MemRefType argType,
                                             spirv::PointerType &varType) {
    auto elementType = argType.getElementType();
    if (!elementType.isIntOrFloat()) {
      return emitError(loc, "unhandled element type ")
             << elementType << " while lowering to SPIR-V";
    }
    if (argType.hasStaticShape()) {
      int64_t stride = elementType.getIntOrFloatBitWidth() / 8;
      for (auto dim : reverse(argType.getShape())) {
        if (dim <= 0) {
          return emitError(loc, "expected tensor dimensions to be non-zero");
        }
        elementType = spirv::ArrayType::get(
            elementType, dim,
            static_cast<spirv::ArrayType::LayoutInfo>(stride));
        stride *= dim;
      }
    }
    varType = spirv::PointerType::get(
        spirv::StructType::get(elementType,
                               static_cast<spirv::StructType::LayoutInfo>(0)),
        spirv::StorageClass::StorageBuffer);
    return success();
  }

  /// Inserts a spv.globalVariable for every argument in the function signature.
  LogicalResult createGlobalVariableForArg(Location loc, OpBuilder &builder,
                                           unsigned argIndex, Type argType,
                                           StringRef varName,
                                           spirv::GlobalVariableOp &var) {
    auto descriptorSetAttrName = convertToSnakeCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingAttrName =
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
    auto argMemrefType = argType.dyn_cast<MemRefType>();
    if (!argMemrefType) {
      return emitError(loc,
                       "unhandled non-memref type argument in SPIR-V lowering");
    }
    spirv::PointerType varType;
    if (failed(convertArgTypeToVariableType(loc, argMemrefType, varType))) {
      return failure();
    }
    var = builder.create<spirv::GlobalVariableOp>(
        loc, TypeAttr::get(varType), builder.getStringAttr(varName), nullptr);
    // Set descriptor_set to 0.
    var.setAttr(descriptorSetAttrName, builder.getI32IntegerAttr(0));
    // Set binding to argument number.
    var.setAttr(bindingAttrName, builder.getI32IntegerAttr(argIndex));

    return success();
  }

  /// Helper method to create the entry function. Creates global variables for
  /// all inputs and outputs. Inserts the spv.EntryPoint operations as well.
  LogicalResult createEntryFn(OpBuilder &builder, FuncOp &fn,
                              AffineExprCodegen &affineExprCodegen,
                              ValueCache &valueCache) {
    auto loc = fn.getLoc();

    // Convert functions arguments and return values to
    // spirv::GlobalVariables. All global variables are given a descriptor set
    // of 0 and binding is the argument number.
    auto fnType = fn.getType();
    auto descriptorSetAttrName = convertToSnakeCase(
        stringifyDecoration(spirv::Decoration::DescriptorSet));
    auto bindingAttrName =
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::Binding));
    for (auto argType : enumerate(fnType.getInputs())) {
      auto &var = inputArgToVariable[fn.getArgument(argType.index())];
      auto varName =
          fn.getName().str() + "_arg_" + std::to_string(argType.index());
      if (failed(createGlobalVariableForArg(loc, builder, argType.index(),
                                            argType.value(), varName, var))) {
        return failure();
      }
    }
    resultIndexToVariable.resize(fnType.getNumResults());
    for (auto resType : enumerate(fnType.getResults())) {
      auto &var = resultIndexToVariable[resType.index()];
      auto varName =
          fn.getName().str() + "_result_" + std::to_string(resType.index());
      if (failed(createGlobalVariableForArg(
              loc, builder, resType.index() + fnType.getNumInputs(),
              resType.value(), varName, var))) {
        return failure();
      }
    }

    auto entryFnType =
        builder.getFunctionType(ArrayRef<Type>(), ArrayRef<Type>());
    auto entryFn = builder.create<FuncOp>(loc, fn.getName(), entryFnType,
                                          ArrayRef<NamedAttribute>());

    // Start a scope to create an insertion guard to reset the builder once the
    // function is lowered.
    {
      OpBuilder::InsertionGuard funcInsertGuard(builder);
      builder.setInsertionPointToStart(entryFn.addEntryBlock());

      // Create the Global invocation ID.
      if (failed(createGlobalInvocationID(builder, fn.getLoc(),
                                          affineExprCodegen))) {
        return failure();
      }

      if (failed(lowerFunction(builder, fn, entryFn, affineExprCodegen,
                               valueCache))) {
        return failure();
      }
    }

    // Create the entry point instructions for the entry function.
    if (failed(createEntryPoint(fn, builder, loc, entryFn))) {
      return failure();
    }
    return success();
  }

  /// Creates the global variable for GlobalInvocationID, and gets the ID at x,
  /// y and z dimensions.
  LogicalResult createGlobalInvocationID(OpBuilder &builder, Location loc,
                                         AffineExprCodegen &affineExprCodegen) {
    auto moduleOp = builder.getInsertionBlock()
                        ->getParentOp()
                        ->getParentOfType<spirv::ModuleOp>();
    OpBuilder moduleBuilder(moduleOp.body());
    auto i32Type = builder.getIntegerType(32);
    auto idType = VectorType::get(3, i32Type);
    auto ptrIdType =
        spirv::PointerType::get(idType, spirv::StorageClass::Input);
    auto globalInvocationID = moduleBuilder.create<spirv::GlobalVariableOp>(
        loc, TypeAttr::get(ptrIdType),
        builder.getStringAttr("globalInvocationID"), nullptr);
    globalInvocationID.setAttr(
        convertToSnakeCase(stringifyDecoration(spirv::Decoration::BuiltIn)),
        builder.getStringAttr(
            spirv::stringifyBuiltIn(spirv::BuiltIn::GlobalInvocationId)));
    interface.push_back(
        builder.getSymbolRefAttr(globalInvocationID.sym_name()));

    auto globalInvocationIDPtr = builder.create<spirv::AddressOfOp>(
        loc, ptrIdType,
        builder.getSymbolRefAttr(globalInvocationID.getOperation()));
    auto id = builder.create<spirv::LoadOp>(loc, idType, globalInvocationIDPtr,
                                            nullptr, nullptr);

    auto id_x = builder.create<spirv::CompositeExtractOp>(
        loc, i32Type, id, builder.getArrayAttr(builder.getI32IntegerAttr(0)));
    globalInvocationIDs.push_back(id_x);
    auto id_y = builder.create<spirv::CompositeExtractOp>(
        loc, i32Type, id, builder.getArrayAttr(builder.getI32IntegerAttr(1)));
    globalInvocationIDs.push_back(id_y);
    auto id_z = builder.create<spirv::CompositeExtractOp>(
        loc, i32Type, id, builder.getArrayAttr(builder.getI32IntegerAttr(2)));
    globalInvocationIDs.push_back(id_z);
    affineExprCodegen.setDimDstValue(0, id_x);
    affineExprCodegen.setDimDstValue(1, id_y);
    affineExprCodegen.setDimDstValue(2, id_z);
    return success();
  }

  /// Method to load the values of globalVariables corresponding to the
  /// arguments of the dispatch function at all indices needed within the
  /// dispatch function.
  LogicalResult initArgValues(OpBuilder &builder, Location loc,
                              AffineExprCodegen &affineExprCodegen,
                              ValueCache &valueCache, Value *origArg) {
    for (auto indexMap : affineExprCodegen.getIndices(origArg)) {
      auto var = inputArgToVariable.lookup(origArg);
      if (!var) {
        return emitError(
            loc, "undefined SPIR-V global variable for tensor argument");
      }
      auto ptr =
          genPointerOffset(builder, loc, affineExprCodegen, indexMap, var);
      auto elementType =
          ptr->getType().template cast<spirv::PointerType>().getPointeeType();
      auto val = builder.create<spirv::LoadOp>(loc, elementType, ptr,
                                               /*memory_access =*/nullptr,
                                               /*alignment = */ nullptr);
      valueCache.setOperandDstValue(origArg, indexMap, val);
    }
    return success();
  }

  /// Adds the spv.EntryPointOp and records all the interface variables used in
  /// the entryFn.
  LogicalResult createEntryPoint(FuncOp fn, OpBuilder &builder, Location loc,
                                 FuncOp entryFn) {
    builder.create<spirv::EntryPointOp>(
        loc,
        builder.getI32IntegerAttr(
            static_cast<int32_t>(spirv::ExecutionModel::GLCompute)),
        builder.getSymbolRefAttr(entryFn), builder.getArrayAttr(interface));

    // Get the workgroup size.
    SmallVector<int32_t, 3> workGroupSize;
    if (failed(getWorkGroupSize(fn, workGroupSize))) {
      return failure();
    }
    builder.create<spirv::ExecutionModeOp>(
        loc, builder.getSymbolRefAttr(entryFn),
        builder.getI32IntegerAttr(
            static_cast<int32_t>(spirv::ExecutionMode::LocalSize)),
        builder.getI32ArrayAttr(workGroupSize));
    interface.clear();
    return success();
  }

  /// Creates the spirv::SelectionOp that checks that the global invocation ID
  /// of the workitem is less than the launch bounds.
  LogicalResult createLaunchGuard(OpBuilder &builder, FuncOp fn) {
    // First check that the global invocation id is in bounds.
    SmallVector<int64_t, 3> launchSize;
    if (failed(getLaunchSize(fn, launchSize))) {
      return failure();
    }
    auto loc = fn.getLoc();
    auto i1Type = builder.getI1Type();
    auto i32Type = builder.getIntegerType(32);
    Value *condn = builder.create<spirv::ConstantOp>(loc, i1Type,
                                                     builder.getBoolAttr(true));
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
    auto selectionControl = builder.getI32IntegerAttr(
        static_cast<uint32_t>(spirv::SelectionControl::None));
    auto selectionOp =
        builder.create<spirv::SelectionOp>(loc, selectionControl);
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

  /// Lowers the body of the function in the original dialect to SPIR-V dialect.
  LogicalResult lowerFunction(OpBuilder &builder, FuncOp fn, FuncOp entryFn,
                              AffineExprCodegen &affineExprCodegen,
                              ValueCache &valueCache) {
    if (failed(createLaunchGuard(builder, fn))) {
      return failure();
    }

    for (auto arg : fn.getArguments()) {
      // Load values of the argument at all indices needed for computation
      // within the dispatch function.
      if (failed(initArgValues(builder, fn.getLoc(), affineExprCodegen,
                               valueCache, arg))) {
        return failure();
      }
    }

    for (auto &block : fn) {
      for (auto &op : block) {
        // Lower individual operations.
        if (failed(
                lowerOperation(builder, affineExprCodegen, valueCache, &op))) {
          return failure();
        }
      }
    }
    builder.setInsertionPointToEnd(&(entryFn.getBody().back()));
    builder.create<spirv::ReturnOp>(fn.getLoc());
    return success();
  }

  /// Dispatches the lowering of tensor operation to SPIR-V scalar
  /// operation.
  LogicalResult lowerOperation(OpBuilder &builder,
                               AffineExprCodegen &affineExprCodegen,
                               ValueCache &valueCache, Operation *op) {
    auto opName = op->getName().getStringRef();
    if (!opCodegenList.count(opName)) {
      return op->emitError("unhandled codegen");
    }
    if (op->getNumResults() > 1) {
      return op->emitError("unhandled codegen for multiple result values");
    }

    // Zero return case.
    if (!op->getNumResults()) {
      return opCodegenList[opName]->lowerOperation(
          op, builder, affineExprCodegen, valueCache, inputArgToVariable,
          resultIndexToVariable);
    }

    // Single return case.
    auto resultTensor = op->getResult(0);
    auto indices = affineExprCodegen.getIndices(resultTensor);
    for (auto &index : indices) {
      auto operandIndices =
          affineExprCodegen.getOperandIndices(resultTensor, index);
      SmallVector<Value *, 2> scalarOperands;
      for (auto arg : llvm::enumerate(op->getOperands())) {
        auto scalarArg = valueCache.getOperandDstValue(
            arg.value(), operandIndices[arg.index()]);
        if (!scalarArg) {
          return op->emitError("argument ")
                 << arg.index() << " has no scalar value";
        }
        scalarOperands.push_back(scalarArg);
      }
      if (failed(opCodegenList[opName]->lowerOperation(
              op, builder, index, scalarOperands, affineExprCodegen,
              valueCache))) {
        return failure();
      }
    }
    return success();
  }

  void insert() {
    std::vector<std::unique_ptr<SPIRVLowering>> objs;
    using dummy = int[];
    (void)dummy{0, (objs.emplace_back(std::make_unique<Ts>()), 0)...};
    for (auto &elem : objs) {
      StringRef opName = elem->getOpName();
      opCodegenList.try_emplace(opName, std::move(elem));
    }
  }

  /// Gets the global invocation ID along a particular dimension (0 -> x, 1 ->
  /// y, 2 -> z).
  Value *getGlobalInvocationID(unsigned dim) {
    if (dim < globalInvocationIDs.size()) {
      return globalInvocationIDs[dim];
    }
    return nullptr;
  }

  /// List of classes that implement the operation lowering from tensor
  /// operations to SPIR-V.
  OpCodegenListT opCodegenList;

  /// I/O interface for the entry function containing global variables that are
  /// used by the entire function call tree.
  SmallVector<Attribute, 4> interface;

  /// Mapping from argument of the dispatch function in tensor dialect to the
  /// corresponding spv.globalVariable.
  DenseMap<Value *, spirv::GlobalVariableOp> inputArgToVariable;

  /// List of spv.globalVariables created for tensors returned by the dispatch
  /// function in tensor dialects.
  SmallVector<spirv::GlobalVariableOp, 1> resultIndexToVariable;

  /// GlobalInvocationID variable.
  SmallVector<Value *, 3> globalInvocationIDs;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_SPIRVLOWERING_H
