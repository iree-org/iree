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

#include "iree/compiler/Dialect/Shape/Utils/TypeConversion.h"

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"

namespace mlir {
namespace iree_compiler {
namespace Shape {

//===----------------------------------------------------------------------===//
// TypeExpander
//===----------------------------------------------------------------------===//

LogicalResult TypeExpander::expandFunctionSignature(FuncOp funcOp,
                                                    OpBuilder &builder) const {
  SmallVector<DictionaryAttr, 4> origArgAttrs;
  SmallVector<DictionaryAttr, 4> origResultAttrs;
  funcOp.getAllArgAttrs(origArgAttrs);
  funcOp.getAllResultAttrs(origResultAttrs);

  SmallVector<Type, 4> argTypes;
  SmallVector<DictionaryAttr, 4> argAttrs;
  SmallVector<Type, 4> resultTypes;
  SmallVector<DictionaryAttr, 4> resultAttrs;

  // Convert arguments.
  auto funcType = funcOp.getType();
  for (unsigned origI = 0, e = funcType.getNumInputs(); origI < e; ++origI) {
    SmallVector<Type, 4> convertedTypes;
    Type sourceType = funcType.getInput(origI);
    if (failed(convertType(sourceType, convertedTypes))) {
      return funcOp.emitError() << "unsupported function argument type ("
                                << origI << "): " << sourceType;
    }

    for (unsigned j = 0; j < convertedTypes.size(); ++j) {
      argTypes.push_back(convertedTypes[j]);
      if (j == 0)
        argAttrs.push_back(origArgAttrs[origI]);
      else
        argAttrs.push_back({});
    }
  }

  // Convert results.
  for (unsigned origI = 0, e = funcType.getNumResults(); origI < e; ++origI) {
    SmallVector<Type, 4> convertedTypes;
    Type sourceType = funcType.getResult(origI);
    if (failed(convertType(sourceType, convertedTypes))) {
      return funcOp.emitError() << "unsupported function result type (" << origI
                                << "): " << sourceType;
    }

    for (unsigned j = 0; j < convertedTypes.size(); ++j) {
      resultTypes.push_back(convertedTypes[j]);
      if (j == 0)
        resultAttrs.push_back(origResultAttrs[origI]);
      else
        resultAttrs.push_back({});
    }
  }

  // Update function.
  auto newFuncType =
      FunctionType::get(funcOp.getContext(), argTypes, resultTypes);
  funcOp.setType(newFuncType);
  funcOp.setAllArgAttrs(argAttrs);
  funcOp.setAllResultAttrs(resultAttrs);

  // Expand the entry block, if exists.
  if (!funcOp.empty() && !funcOp.getBlocks().empty()) {
    auto &entryBlock = funcOp.getBlocks().front();
    if (failed(expandBlockSignature(funcOp.getLoc(), &entryBlock, builder))) {
      return failure();
    }
  }

  return success();
}

LogicalResult TypeExpander::expandBlockSignature(Location loc, Block *block,
                                                 OpBuilder &builder) const {
  builder.setInsertionPointToStart(block);
  for (unsigned i = 0; i < block->getNumArguments();) {
    SmallVector<Type, 4> convertedTypes;
    SmallVector<Value, 4> convertedArgs;
    auto origArg = block->getArgument(i);
    Type origArgType = origArg.getType();

    if (failed(convertType(origArgType, convertedTypes))) {
      return emitError(loc)
             << "unsupported block argument type " << origArgType;
    }

    // Identity conversion. Just advance.
    if (convertedTypes.size() == 1 && convertedTypes.front() == origArgType) {
      ++i;
      continue;
    }

    // Remove argument.
    if (convertedTypes.empty()) {
      if (!origArg.use_empty()) {
        return emitError(loc)
               << "cannot remove block argument that has uses: " << origArgType;
      }
      block->eraseArgument(i);
      continue;
    }

    // Splice new arguments after current.
    for (unsigned j = 0; j < convertedTypes.size(); ++j) {
      convertedArgs.push_back(
          block->insertArgument(i + j + 1, convertedTypes[j]));
    }

    // Cast.
    Value newArg = castToSource(loc, origArgType, convertedArgs, builder);
    if (!newArg) {
      return failure();
    }

    origArg.replaceAllUsesWith(newArg);
    block->eraseArgument(i);
    i += convertedArgs.size();
  }

  return success();
}

LogicalResult TypeExpander::expandSourceValuesToTarget(
    Location loc, ArrayRef<Value> sourceValues,
    SmallVectorImpl<Value> &targetValues, OpBuilder &builder) const {
  auto *context = builder.getContext();
  auto emitError = [loc, context]() {
    return context->getDiagEngine().emit(loc, DiagnosticSeverity::Error);
  };

  for (auto sourceValue : sourceValues) {
    SmallVector<Type, 4> convertedTypes;
    SmallVector<Value, 4> convertedValues;
    auto sourceType = sourceValue.getType();
    if (failed(convertType(sourceType, convertedTypes))) {
      emitError() << "unsupported type: " << sourceType;
      return failure();
    }
    if (failed(castToTarget(loc, sourceValue, convertedTypes, convertedValues,
                            builder))) {
      return failure();
    }
    assert(convertedTypes.size() == convertedValues.size());
    targetValues.append(convertedValues.begin(), convertedValues.end());
  }
  return success();
}

LogicalResult TypeExpander::expandGenericReturnLikeTerminator(
    Operation *op, OpBuilder &builder) const {
  builder.setInsertionPoint(op);
  OperationState opState(op->getLoc(), op->getName());
  opState.addAttributes(op->getAttrs());

  SmallVector<Value, 4> origOperands(op->getOperands());
  if (failed(expandSourceValuesToTarget(op->getLoc(), origOperands,
                                        opState.operands, builder))) {
    return failure();
  }

  // Clone the op.
  builder.createOperation(opState);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// DynamicShapeTypeExpander
//===----------------------------------------------------------------------===//

namespace {

class DynamicShapeTypeExpander : public TypeExpander {
  static RankedTensorType getDynamicRankedTensorType(Type t) {
    auto rankedTensorType = t.dyn_cast<RankedTensorType>();
    if (!rankedTensorType) return nullptr;
    if (rankedTensorType.hasStaticShape()) return nullptr;
    return rankedTensorType;
  }

  LogicalResult convertType(Type sourceType,
                            SmallVectorImpl<Type> &targetTypes) const override {
    if (auto rankedTensorType = getDynamicRankedTensorType(sourceType)) {
      auto rankedShapeType = RankedShapeType::get(rankedTensorType.getShape(),
                                                  sourceType.getContext());
      targetTypes.push_back(sourceType);
      targetTypes.push_back(rankedShapeType);
      return success();
    } else if (sourceType.isa<UnrankedTensorType>()) {
      // Not yet supported.
      return failure();
    }

    // Fallback to pass-through.
    targetTypes.push_back(sourceType);
    return success();
  }

  Value castToSource(Location loc, Type sourceType,
                     ArrayRef<Value> targetValues,
                     OpBuilder &builder) const override {
    if (auto rankedTensorType = getDynamicRankedTensorType(sourceType)) {
      assert(targetValues.size() == 2);
      return builder.create<TieShapeOp>(loc, sourceType, targetValues[0],
                                        targetValues[1]);
    } else if (targetValues.size() == 1) {
      // Default to no cast required.
      assert(targetValues.size() == 1);
      return targetValues.front();
    }

    return nullptr;
  }

  LogicalResult castToTarget(Location loc, Value sourceValue,
                             ArrayRef<Type> targetTypes,
                             SmallVectorImpl<Value> &targetValues,
                             OpBuilder &builder) const override {
    if (targetTypes.size() == 2 && targetTypes.back().isa<RankedShapeType>()) {
      auto shape = builder.create<GetRankedShapeOp>(loc, targetTypes.back(),
                                                    sourceValue);
      targetValues.push_back(sourceValue);
      targetValues.push_back(shape);
      return success();
    } else if (targetTypes.size() == 1) {
      targetValues.push_back(sourceValue);
      return success();
    }

    return failure();
  }
};

}  // namespace

const TypeExpander &getDynamicShapeTypeExpander() {
  static DynamicShapeTypeExpander instance;
  return instance;
}

//===----------------------------------------------------------------------===//
// ShapeToPrimitiveTypeExpander
//===----------------------------------------------------------------------===//

namespace {

class ShapeToPrimitiveTypeExpander : public TypeExpander {
  LogicalResult convertType(Type sourceType,
                            SmallVectorImpl<Type> &targetTypes) const override {
    if (auto rsType = sourceType.dyn_cast<RankedShapeType>()) {
      auto dimType = IndexType::get(sourceType.getContext());
      for (int i = 0; i < rsType.getNumDynamicDims(); ++i) {
        targetTypes.push_back(dimType);
      }
      return success();
    }

    // Fallback to pass-through.
    targetTypes.push_back(sourceType);
    return success();
  }

  Value castToSource(Location loc, Type sourceType,
                     ArrayRef<Value> targetValues,
                     OpBuilder &builder) const override {
    if (auto rsType = sourceType.dyn_cast<RankedShapeType>()) {
      if (targetValues.empty()) {
        return builder.create<ConstRankedShapeOp>(loc, rsType);
      } else {
        return builder.create<MakeRankedShapeOp>(loc, rsType, targetValues);
      }
    } else if (targetValues.size() == 1) {
      // Default to no cast required.
      assert(targetValues.size() == 1);
      return targetValues.front();
    }

    return nullptr;
  }

  LogicalResult castToTarget(Location loc, Value sourceValue,
                             ArrayRef<Type> targetTypes,
                             SmallVectorImpl<Value> &targetValues,
                             OpBuilder &builder) const override {
    if (auto rsType = sourceValue.getType().dyn_cast<RankedShapeType>()) {
      // Only dynamic dims are materialized
      assert(targetTypes.size() == rsType.getNumDynamicDims());
      for (int dim = 0, e = rsType.getRank(); dim < e; ++dim) {
        if (rsType.isDimDynamic(dim)) {
          targetValues.push_back(
              builder.create<RankedDimOp>(loc, sourceValue, dim));
        }
      }
      assert(targetValues.size() == targetTypes.size());
      return success();
    } else if (targetTypes.size() == 1) {
      targetValues.push_back(sourceValue);
      return success();
    }

    return failure();
  }
};

}  // namespace

const TypeExpander &getShapeToPrimitiveTypeExpander() {
  static ShapeToPrimitiveTypeExpander instance;
  return instance;
}

}  // namespace Shape
}  // namespace iree_compiler
}  // namespace mlir
