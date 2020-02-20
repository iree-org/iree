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

#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

namespace {

class DynamicDimsTypeConverter : public TypeConverter {
 public:
  DynamicDimsTypeConverter() {
    // Add a default conversion to allow partial type conversions.
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type, SmallVectorImpl<Type> &results) {
      if (type.hasStaticShape()) {
        results.push_back(type);
        return success();
      }

      // Dimension is hard-coded to 32bits currently but better decisions
      // are possible in some situations.
      auto dimType = IndexType::get(type.getContext());
      auto shapeType = Shape::RankedShapeType::get(type.getShape(), dimType);
      // Expand tensor<?...x*> -> (tensor<...>, ranked_shape<...,index>)
      results.push_back(type);
      results.push_back(shapeType);
      return success();
    });
  }

  Operation *materializeConversion(PatternRewriter &rewriter, Type resultType,
                                   ArrayRef<Value> inputs,
                                   Location loc) override {
    // Adds a conversion from (%0 = tensor<...>, %1 = ranked_shape<...>) inputs
    // to: shape.tie_shape %0, %1
    assert(inputs.size() == 2);
    return rewriter.create<Shape::TieShapeOp>(loc, resultType, inputs[0],
                                              inputs[1]);
  }
};

class FuncOpConversion : public OpConversionPattern<FuncOp> {
 public:
  FuncOpConversion(DynamicDimsTypeConverter &typeConverter,
                   MLIRContext *context)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  PatternMatchResult matchAndRewrite(
      FuncOp fnOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto fnType = fnOp.getType();

    // TODO(laurenzo): Need to handle all terminators so conservatively
    // limiting to single block functions until implemented.
    if (fnOp.getBody().getBlocks().size() != 1) {
      fnOp.emitWarning()
          << "dynamic shape conversion only supported for single block "
          << "functions (currently)";
      return matchFailure();
    }

    // Convert function arguments.
    TypeConverter::SignatureConversion signatureConverter(
        fnType.getNumInputs());
    for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i) {
      if (failed(typeConverter.convertSignatureArg(i, fnType.getInput(i),
                                                   signatureConverter))) {
        return matchFailure();
      }
    }

    // Convert function results.
    SmallVector<Type, 1> convertedResultTypes;
    if (failed(typeConverter.convertTypes(fnType.getResults(),
                                          convertedResultTypes))) {
      return matchFailure();
    }

    // Replace function.
    auto newFnOp = rewriter.cloneWithoutRegions(fnOp);
    rewriter.inlineRegionBefore(fnOp.getBody(), newFnOp.getBody(),
                                newFnOp.end());
    newFnOp.setType(rewriter.getFunctionType(
        signatureConverter.getConvertedTypes(), convertedResultTypes));
    rewriter.applySignatureConversion(&newFnOp.getBody(), signatureConverter);
    rewriter.eraseOp(fnOp);

    // Rewrite the terminator to match the result type conversion that was
    // performed.
    auto terminator = newFnOp.getBody().front().getTerminator();
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(terminator);
    SmallVector<Value, 4> newTerminatorOperands;
    for (unsigned i = 0, e = terminator->getNumOperands(); i < e; ++i) {
      auto operand = terminator->getOperand(i);
      SmallVector<Type, 2> expandedTypes;
      if (failed(typeConverter.convertType(operand.getType(), expandedTypes))) {
        continue;
      }

      // Non-conversion
      if (expandedTypes.size() == 1) {
        newTerminatorOperands.push_back(operand);
        continue;
      }
      assert(expandedTypes.size() == 2 &&
             "type converter should expand 1 -> 2");

      // Expand (tensor<...>) to (tensor<...>, ranked_shape<...>)
      auto shape = rewriter.create<Shape::GetRankedShapeOp>(
          terminator->getLoc(), expandedTypes[1], operand);
      newTerminatorOperands.push_back(operand);
      newTerminatorOperands.push_back(shape);
    }

    // Clone the terminator (assumed to be 'return'-like) with modified
    // operands.
    OperationState terminatorState(terminator->getLoc(), terminator->getName());
    terminatorState.addOperands(newTerminatorOperands);
    rewriter.createOperation(terminatorState);
    rewriter.eraseOp(terminator);

    rewriter.restoreInsertionPoint(ip);
    return matchSuccess();
  }

 private:
  DynamicDimsTypeConverter &typeConverter;
};

bool isLegallyShapedSignatureType(Type thisType, Type nextType) {
  if (!thisType.isa<TensorType>()) return true;  // Legal: Don't care.
  auto rankedType = thisType.dyn_cast<RankedTensorType>();
  if (!rankedType) return false;  // Illegal: Non-ranked tensor
  if (rankedType.getNumDynamicDims() == 0) return true;  // Legal: Static shape

  // At this point, the type is ranked and has dynamic dims. Validate.
  auto rankedShapeType = nextType.dyn_cast_or_null<Shape::RankedShapeType>();
  if (!rankedShapeType) return false;  // Illegal: No following shape.

  // Are dims equal.
  auto thisDims = rankedType.getShape();
  auto shapeDims = rankedShapeType.getAllDims();
  if (!thisDims.equals(shapeDims)) return false;  // Illegal: Mismatched shape.
  return true;  // Legal: dynamic tensor followed by matching shape.
}

// Determines whether a function is "legally shaped", which means that its
// shaped inputs/results are either a) statically shaped or b) followed by
// an appropriate (ranked_shape) argument/result with corresponding
// dims.
bool isLegallyShapedFunction(FuncOp fnOp) {
  auto fnType = fnOp.getType();
  // Validate arguments.
  for (unsigned i = 0, e = fnType.getNumInputs(); i < e; ++i) {
    Type type = fnType.getInput(i);
    Type nextType = (i + 1 < e) ? fnType.getInput(i + 1) : nullptr;
    if (!isLegallyShapedSignatureType(type, nextType)) return false;
  }
  // Validate results.
  return true;
}

class ExpandFunctionDynamicDimsPass
    : public ModulePass<ExpandFunctionDynamicDimsPass> {
  void runOnModule() override {
    ConversionTarget target(getContext());
    target.addDynamicallyLegalOp<FuncOp>(isLegallyShapedFunction);
    target.markOpRecursivelyLegal<FuncOp>();

    OwningRewritePatternList patterns;
    DynamicDimsTypeConverter typeConverter;
    patterns.insert<FuncOpConversion>(typeConverter, &getContext());

    if (failed(applyPartialConversion(getModule(), target, patterns,
                                      &typeConverter))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

// For any function which contains dynamic dims in its inputs or results,
// rewrites it so that the dynamic dims are passed in/out.
std::unique_ptr<OpPassBase<ModuleOp>> createExpandFunctionDynamicDimsPass() {
  return std::make_unique<ExpandFunctionDynamicDimsPass>();
}

static PassRegistration<ExpandFunctionDynamicDimsPass> pass(
    "iree-shape-expand-function-dynamic-dims",
    "Expands dynamic dimensions in function signatures.");

}  // namespace iree_compiler
}  // namespace mlir
