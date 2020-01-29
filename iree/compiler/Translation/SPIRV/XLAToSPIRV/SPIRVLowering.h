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
#ifndef IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_SPIRVLOWERING_H
#define IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_SPIRVLOWERING_H

#include "iree/compiler/Translation/SPIRV/IndexComputation/IndexComputationAttribute.h"
#include "iree/compiler/Translation/SPIRV/XLAToSPIRV/TensorIndexToScalarValueMap.h"
#include "iree/compiler/Utils/IREECodegenUtils.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/Support/StringExtras.h"

namespace mlir {
namespace iree_compiler {

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
  virtual LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands, TensorIndexToScalarValueMap &valueCache) const {
    return failure();
  }

  /// This method (in the derived class) should generate the scalar operations
  /// corresponding to the tensor operation `op`. This should be implemented
  /// when the `op` has no result value, typically store operations and return
  /// operations.
  virtual LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder,
      TensorIndexToScalarValueMap &valueCache) const {
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
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// SPIR-V lowering for CmpIOp.
class CmpIOpSPIRVLowering final : public SPIRVOpLowering<CmpIOp> {
 public:
  using SPIRVOpLowering<CmpIOp>::SPIRVOpLowering;

  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// SPIR-V lowering for CmpFOp.
class CmpFOpSPIRVLowering final : public SPIRVOpLowering<CmpFOp> {
 public:
  using SPIRVOpLowering<CmpFOp>::SPIRVOpLowering;

  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override;
};

/// SPIR-V lowering for Min/Max operations.
template <typename OpTy, typename CmpOpTy, typename CmpFOpTy>
class CmpSelectOpSPIRVLowering final : public SPIRVOpLowering<OpTy> {
 public:
  using SPIRVOpLowering<OpTy>::SPIRVOpLowering;
  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> operands,
      TensorIndexToScalarValueMap &valueCache) const override {
    if (op->getNumOperands() != 2) {
      return op->emitError(
          "unhandled SPIR-V lowering for more than 2 operands");
    }
    assert(operands.size() == op->getNumOperands() &&
           "expected as many operands for the replacement as the original "
           "instruction");
    auto cmpSelectOp = cast<OpTy>(op);
    auto result = cmpSelectOp.getResult();
    auto resultTy = result.getType().template dyn_cast<ShapedType>();
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
        op->getLoc(), operands[0].getType(), cmpOp->getResult(0), operands[0],
        operands[1]);
    valueCache.setValueAtIndex(op->getResult(0), index, selectOp.getResult());
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

  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> scalarOperands,
      TensorIndexToScalarValueMap &valueCache) const override {
    // TODO(ravishankarm) : This check should really be a static_assert. See if
    // that can be changed.
    if (op->getNumOperands() == 0) {
      return op->emitError("expected op to have at least one operand");
    }
    auto pwOp = cast<OpTy>(op);
    auto result = pwOp.getResult();
    auto resultType = result.getType().template dyn_cast<ShapedType>();
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
    valueCache.setValueAtIndex(pwOp.getResult(), index, scalarOp->getResult(0));
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

  LogicalResult lowerOperation(
      Operation *op, OpBuilder &builder, AffineMap index,
      ArrayRef<Value> scalarOperands,
      TensorIndexToScalarValueMap &valueCache) const override {
    if (op->getNumOperands() != 1) {
      return op->emitError(
          "unhandled lowering of index transformation operation with multiple "
          "operands");
    }
    auto indexOp = cast<OpTy>(op);
    valueCache.setValueAtIndex(indexOp.getResult(), index, scalarOperands[0]);
    return success();
  }
};

/// Generates spv.AccessChain instruction to get the pointer value at a given
/// location of a spv.globalVariable.
Value genPointerOffset(OpBuilder &builder, Location loc,
                       TensorIndexToScalarValueMap &valueCache,
                       const AffineMap &indexMap, ArrayRef<int64_t> shape,
                       const Value &buffer);

namespace detail {
/// Implementation class for generating SPIR-V kernel.
class SPIRVCodegenImpl {
 public:
  virtual ~SPIRVCodegenImpl() {}

 protected:
  /// Creates the entry function for a given dispatch function `fn`.
  LogicalResult createEntryFn(OpBuilder &builder, FuncOp &fn,
                              TensorIndexToScalarValueMap &valueCache);

  /// Mapping from argument of the dispatch function in tensor dialect to the
  /// corresponding spv.globalVariable.
  DenseMap<Value, spirv::GlobalVariableOp> inputArgToVariable;

  /// List of spv.globalVariables created for tensors returned by the dispatch
  /// function in tensor dialects.
  SmallVector<spirv::GlobalVariableOp, 1> resultIndexToVariable;

 private:
  /// Adds the spv.EntryPointOp and records all the interface variables used in
  /// the entryFn.
  LogicalResult createEntryPoint(FuncOp fn, OpBuilder &builder, Location loc,
                                 FuncOp entryFn);

  /// Creates the spirv::SelectionOp that checks that the global invocation ID
  /// of the workitem is less than the launch bounds.
  LogicalResult createLaunchGuard(OpBuilder &builder, FuncOp fn);

  /// Method to load the values of globalVariables corresponding to the
  /// arguments of the dispatch function at all indices needed within the
  /// dispatch function.
  LogicalResult initArgValues(OpBuilder &builder, Location loc,
                              TensorIndexToScalarValueMap &valueCache,
                              Value origArg);

  /// Method to load values that correspond to symbols used in affine maps.
  LogicalResult initSymbolValues(OpBuilder &builder, Location loc,
                                 TensorIndexToScalarValueMap &valueCache,
                                 Value origArg);

  /// Gets the global invocation ID along a particular dimension (0 -> x, 1 ->
  /// y, 2 -> z).
  Value getGlobalInvocationID(unsigned dim);

  /// Method to load the values of an argument at an index.
  Value loadArgValueAtIndex(OpBuilder &builder, Location loc,
                            TensorIndexToScalarValueMap &valueCache,
                            Value origArg, AffineMap indexMap);

  /// Lowers the body of the function in the original dialect to SPIR-V dialect.
  LogicalResult lowerFunction(OpBuilder &builder, FuncOp fn, FuncOp entryFn,
                              TensorIndexToScalarValueMap &valueCache);

  /// Method to lower the operations within the dispatch function.
  virtual LogicalResult lowerOperation(OpBuilder &builder,
                                       TensorIndexToScalarValueMap &valueCache,
                                       Operation *op) = 0;

  /// I/O interface for the entry function containing global variables that are
  /// used by the entire function call tree.
  SmallVector<Attribute, 4> interface;

  /// GlobalInvocationID variable.
  SmallVector<Value, 3> globalInvocationIDs;
};
}  // namespace detail

/// Class to drive the SPIRV code-generation.
template <typename... Ts>
class SPIRVCodegen : public detail::SPIRVCodegenImpl {
  using OpCodegenListT = llvm::StringMap<std::unique_ptr<SPIRVLowering>>;

 public:
  explicit SPIRVCodegen() { insert(); }

  LogicalResult codegen(spirv::ModuleOp &spirvModule, FuncOp &fn) {
    if (fn.getBlocks().size() != 1) {
      return emitError(
          fn.getLoc(),
          "unimplemeneted handling multiple blocks within a function");
    }

    OpBuilder builder(spirvModule.body());
    // Create the entry function and generate global invocation ID. Creates a
    // global variable for all inputs and output tensors.
    TensorIndexToScalarValueMap valueCache;
    return createEntryFn(builder, fn, valueCache);
  }

 private:
  /// Dispatches the lowering of tensor operation to SPIR-V scalar
  /// operation.
  LogicalResult lowerOperation(OpBuilder &builder,
                               TensorIndexToScalarValueMap &valueCache,
                               Operation *op) override {
    auto opName = op->getName().getStringRef();
    if (!opCodegenList.count(opName)) {
      return op->emitError("unhandled codegen");
    }
    if (op->getNumResults() > 1) {
      return op->emitError("unhandled codegen for multiple result values");
    }

    // Zero return case.
    if (!op->getNumResults()) {
      return opCodegenList[opName]->lowerOperation(op, builder, valueCache);
    }

    // Single return case.
    auto resultTensor = op->getResult(0);
    SmallVector<AffineMap, 4> indices;
    getIndexMapsForValue(resultTensor, indices);
    for (auto &index : indices) {
      // TODO(ravishankarm): This is a WAR till the XLA to SPIR-V lowering is
      // updated to handle the generality expressed in IREE::IndexAttr.
      SmallVector<SmallVector<AffineMap, 1>, 2> operandIndices2D;
      getIndexMapsForOperands(op, index, operandIndices2D);
      SmallVector<AffineMap, 2> operandIndices;
      for (auto &operandIndexList : operandIndices2D) {
        assert(operandIndexList.size() == 1 &&
               "unhandled multiple indices per operand");
        operandIndices.push_back(operandIndexList[0]);
      }
      SmallVector<Value, 2> scalarOperands;
      for (auto arg : llvm::enumerate(op->getOperands())) {
        auto scalarArg = valueCache.getValueAtIndex(
            arg.value(), operandIndices[arg.index()]);
        if (!scalarArg) {
          return op->emitError("argument ")
                 << arg.index() << " has no scalar value";
        }
        scalarOperands.push_back(scalarArg);
      }
      if (failed(opCodegenList[opName]->lowerOperation(
              op, builder, index, scalarOperands, valueCache))) {
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

  /// List of classes that implement the operation lowering from tensor
  /// operations to SPIR-V.
  OpCodegenListT opCodegenList;
};

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_TRANSLATION_SPIRV_XLATOSPIRV_SPIRVLOWERING_H
