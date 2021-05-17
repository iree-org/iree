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

#include <memory>
#include <utility>

#include "iree/compiler/Conversion/Common/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace {

/// Any fp32 derived type is illegal.
static bool isIllegalType(Type type) {
  if (type.isF32()) return true;
  if (auto ptrType = type.dyn_cast<IREE::PtrType>()) {
    return isIllegalType(ptrType.getTargetType());
  }
  if (auto shapedType = type.dyn_cast<ShapedType>()) {
    return isIllegalType(shapedType.getElementType());
  }
  return false;
}

class F32ToF16ConversionTarget : public ConversionTarget {
 public:
  using ConversionTarget::ConversionTarget;

 protected:
  // Operations are legal if they don't contain any illegal type.
  bool isDynamicallyLegal(Operation *op) const override {
    if (auto varOp = dyn_cast<IREE::Flow::VariableOp>(op)) {
      return !isIllegalType(varOp.type());
    }
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      for (Type type : funcOp.getType().getInputs()) {
        if (isIllegalType(type)) return false;
      }
      for (Type type : funcOp.getType().getResults()) {
        if (isIllegalType(type)) return false;
      }
    }
    for (Type type : op->getResultTypes()) {
      if (isIllegalType(type)) return false;
    }
    for (Type type : op->getOperandTypes()) {
      if (isIllegalType(type)) return false;
    }
    return true;
  }
};

class FloatTypeConverter : public TypeConverter {
 public:
  static Type convertTensor(RankedTensorType type) {
    if (!type.getElementType().isF32()) return type;
    auto newType = RankedTensorType::get(type.getShape(),
                                         Float16Type::get(type.getContext()));
    return newType;
  }
  explicit FloatTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([&](FloatType type) {
      if (type.isF32()) return FloatType::getF16(type.getContext());
      return type;
    });
    addConversion(convertTensor);
    addConversion([&](IREE::PtrType ptrType) {
      if (auto tensorType =
              ptrType.getTargetType().dyn_cast<RankedTensorType>()) {
        return IREE::PtrType::get(convertTensor(tensorType));
      }
      return ptrType;
    });
  }
};

// Generic pattern to convert FP32 values and attributes to FP16.
class GenericTypeConvert : public ConversionPattern {
 public:
  GenericTypeConvert(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<NamedAttribute, 4> newAttr;
    convertAttributes(op->getAttrs(), rewriter, newAttr);
    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);
    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttr, op->getSuccessors());
    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }
    Operation *newOp = rewriter.createOperation(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

 protected:
  static void convertAttributes(ArrayRef<NamedAttribute> attrs,
                                ConversionPatternRewriter &rewriter,
                                SmallVectorImpl<NamedAttribute> &newAttrs) {
    for (auto attr : attrs) {
      if (auto fpAttr = attr.second.dyn_cast<DenseFPElementsAttr>()) {
        std::vector<llvm::APFloat> args;
        if (!fpAttr.getType().getElementType().isF32()) continue;
        for (llvm::APFloat f : fpAttr.getFloatValues()) {
          bool losesInfo;
          f.convert(APFloat::IEEEhalf(), APFloat::rmTowardZero, &losesInfo);
          args.push_back(f);
        }
        auto tensorType = RankedTensorType::get(fpAttr.getType().getShape(),
                                                rewriter.getF16Type());
        newAttrs.push_back(std::make_pair(
            attr.first, DenseElementsAttr::get(tensorType, args)));
      } else if (auto typeAttr = attr.second.dyn_cast<TypeAttr>()) {
        if (isIllegalType(typeAttr.getValue())) {
          if (auto tensorType =
                  typeAttr.getValue().dyn_cast<RankedTensorType>()) {
            Type newType = RankedTensorType::get(tensorType.getShape(),
                                                 rewriter.getF16Type());
            newAttrs.push_back(
                std::make_pair(attr.first, TypeAttr::get(newType)));
          }
        }
      } else {
        newAttrs.push_back(attr);
      }
    }
  }
};

struct ConvertF32ToF16Pass
    : public PassWrapper<ConvertF32ToF16Pass, OperationPass<ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp moduleOp = getOperation();

    FloatTypeConverter converter;
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<GenericTypeConvert>(context, converter);
    populateFuncOpTypeConversionPattern(patterns, converter);
    F32ToF16ConversionTarget target(*context);
    target.markUnknownOpDynamicallyLegal();
    if (failed(applyFullConversion(moduleOp, target, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createDemoteF32ToF16Pass() {
  return std::make_unique<ConvertF32ToF16Pass>();
}

static PassRegistration<ConvertF32ToF16Pass> pass(
    "iree-convert-f32-to-f16",
    "Convert f32 operations and values into equivalent f16 ones.");

}  // namespace iree_compiler
}  // namespace mlir
