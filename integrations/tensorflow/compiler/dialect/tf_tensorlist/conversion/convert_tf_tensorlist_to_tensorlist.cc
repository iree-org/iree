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

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/conversion/convert_tf_tensorlist_to_tensorlist.h"

#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_dialect.h"
#include "integrations/tensorflow/compiler/dialect/tf_tensorlist/ir/tf_tensorlist_ops.h"
#include "integrations/tensorflow/compiler/dialect/utils/conversion_utils.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListOps.h"
#include "iree/compiler/Dialect/Modules/TensorList/IR/TensorListTypes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace tf_tensorlist {

namespace {

class TensorListTypeConverter : public TypeConverter {
 public:
  TensorListTypeConverter() {
    // Required to covert any unknown or already converted types.
    addConversion([](Type type) { return type; });
    addConversion([](tf_tensorlist::TensorListType type) {
      return IREE::TensorList::TensorListType::get(type.getContext());
    });
  }
};

// Populates conversion patterns from the tensor-based custom dialect ops to the
// HAL buffer-based ones.
void populateTensorListToHALPatterns(MLIRContext *context,
                                     OwningRewritePatternList &patterns,
                                     TypeConverter &typeConverter);

/*
// Exposes conversion patterns that transition tensors to buffers during the
// Flow->HAL dialect lowering. This is only required if the dialect has ops that
// use tensor types.
class TFTensorListToHALConversionInterface
    : public HALConversionDialectInterface {
 public:
  TFTensorListToHALConversionInterface(Dialect *dialect)
      : HALConversionDialectInterface(dialect) {
    dialect->getContext()->loadDialect<IREE::TensorList::TensorListDialect>();
  }

  void setupConversionTarget(ConversionTarget &target,
                             OwningRewritePatternList &patterns,
                             TypeConverter &typeConverter) const override {
    target.addLegalDialect<IREE::TensorList::TensorListDialect>();
    populateTensorListToHALPatterns(getDialect()->getContext(), patterns,
                                    typeConverter);
  }

  LogicalResult convertType(Type type,
                            SmallVectorImpl<Type> &results) const override {
    if (type.isa<tf_tensorlist::TensorListType>()) {
      results.push_back(
          IREE::TensorList::TensorListType::get(type.getContext()));
      return success();
    }
    return failure();
  }
};

Value getBufferView(Operation *srcOp, Value srcOperand, Value dstOperand,
                    ConversionPatternRewriter &rewriter) {
  auto operand = IREE::HAL::TensorRewriteAdaptor::getChecked(
      srcOp->getLoc(), srcOperand, dstOperand, rewriter);
  if (!operand.hasValue()) {
    srcOp->emitOpError() << "unable to create adaptor for operand";
    return nullptr;
  }
  auto bufferView = operand->getBufferView();
  if (!bufferView) {
    srcOp->emitOpError() << "unable to get buffer view for operand";
    return nullptr;
  }

  return bufferView;
}

class ReserveOpConversion : public OpConversionPattern<tf_tensorlist::Reserve> {
 public:
  ReserveOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Reserve reserveOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto elementTy = reserveOp.element_type();
    auto element_value = IREE::HAL::getElementTypeValue(elementTy).getValue();

    auto operand0 = getBufferView(reserveOp, reserveOp.getOperand(0),
                                  newOperands[0], rewriter);
    auto operand1 = getBufferView(reserveOp, reserveOp.getOperand(1),
                                  newOperands[1], rewriter);

    if (!operand0 || !operand1) {
      return failure();
    }

    rewriter.replaceOpWithNewOp<IREE::TensorList::Reserve>(
        reserveOp,
        IREE::TensorList::TensorListType::get(reserveOp.getContext()), operand0,
        operand1, rewriter.getI32IntegerAttr(element_value));
    return success();
  }
};

class ConcatOpConversion : public OpConversionPattern<tf_tensorlist::Concat> {
 public:
  ConcatOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Concat concatOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(concatOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(concatOp.getLoc(), device)
            .getResult();

    auto newConcatOp = rewriter.createOrFold<IREE::TensorList::Concat>(
        concatOp.getLoc(),
        IREE::HAL::BufferViewType::get(rewriter.getContext()), allocator,
        newOperands[0]);

    auto bufferOp = rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
        newConcatOp.getLoc(), newConcatOp);

    rewriter.replaceOp(concatOp, bufferOp);
    return success();
  }
};

class StackOpConversion : public OpConversionPattern<tf_tensorlist::Stack> {
 public:
  StackOpConversion(MLIRContext *ctx, TypeConverter &converter)
      : OpConversionPattern(ctx) {}

  LogicalResult matchAndRewrite(
      tf_tensorlist::Stack stackOp, llvm::ArrayRef<Value> newOperands,
      ConversionPatternRewriter &rewriter) const override {
    auto device =
        rewriter.createOrFold<IREE::HAL::ExSharedDeviceOp>(stackOp.getLoc());
    auto allocator =
        rewriter.create<IREE::HAL::DeviceAllocatorOp>(stackOp.getLoc(), device)
            .getResult();

    auto operand1 =
        getBufferView(stackOp, stackOp.getOperand(1), newOperands[1], rewriter);
    if (!operand1) return failure();

    auto newStackOp = rewriter.createOrFold<IREE::TensorList::Stack>(
        stackOp.getLoc(), IREE::HAL::BufferViewType::get(rewriter.getContext()),
        allocator, newOperands[0], operand1);

    auto bufferOp = rewriter.createOrFold<IREE::HAL::BufferViewBufferOp>(
        stackOp.getLoc(), newStackOp);

    rewriter.replaceOp(stackOp, bufferOp);
    return success();
  }
};
*/

void populateTFTensorListToTensorListPatterns(
    MLIRContext *context, OwningRewritePatternList &patterns) {
  patterns.insert<
      OpConversion<tf_tensorlist::Reserve, IREE::TensorList::ReserveTensor>>(
      context);
  patterns
      .insert<OpConversion<tf_tensorlist::GetItem, IREE::TensorList::GetItem>>(
          context);
  patterns
      .insert<OpConversion<tf_tensorlist::SetItem, IREE::TensorList::SetItem>>(
          context);
  patterns.insert<
      OpConversion<tf_tensorlist::FromTensor, IREE::TensorList::FromTensor>>(
      context);
  patterns.insert<
      OpConversion<tf_tensorlist::Concat, IREE::TensorList::ConcatTensor>>(
      context);
  patterns.insert<
      OpConversion<tf_tensorlist::Stack, IREE::TensorList::StackTensor>>(
      context);
}

class ConvertTFTensorlistToTensorlistPass
    : public ConversionPass<ConvertTFTensorlistToTensorlistPass,
                            TensorListTypeConverter> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tf_tensorlist::TFTensorListDialect,
                    IREE::TensorList::TensorListDialect, StandardOpsDialect>();
  }

  void Setup(ConversionTarget &target,
             OwningRewritePatternList &patterns) override {
    target.addIllegalDialect<tf_tensorlist::TFTensorListDialect>();
    target.addLegalDialect<IREE::TensorList::TensorListDialect>();
    populateTFTensorListToTensorListPatterns(&this->getContext(), patterns);
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFTensorListToTensorListPass() {
  return std::make_unique<ConvertTFTensorlistToTensorlistPass>();
}

static PassRegistration<ConvertTFTensorlistToTensorlistPass> pass(
    "iree-tf-tensorlist-convert-to-tensorlist",
    "Converts TF string ops to the IREE tf_strings dialect");

}  // namespace tf_tensorlist
}  // namespace iree_compiler
}  // namespace mlir
