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

#include "iree/compiler/Dialect/IREE/IR/IREETypes.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/VMLA/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VMLA/Conversion/HLOToVMLA/ConvertHLOToVMLA.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLADialect.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLAOps.h"
#include "iree/compiler/Dialect/VMLA/IR/VMLATypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/BitVector.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct VMLAConvOpConverter : public OpConversionPattern<mhlo::ConvOp> {
  using OpConversionPattern::OpConversionPattern;
  VMLAConvOpConverter(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      mhlo::ConvOp op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op.dimension_numbers()) {
      const auto dimensionNumbers = op.dimension_numbers();
      const int inputSpatialRank =
          std::distance(dimensionNumbers.input_spatial_dimensions().begin(),
                        dimensionNumbers.input_spatial_dimensions().end());

      if (inputSpatialRank != 2) {
        op.emitWarning() << "Only lowering 2D conv is supported";
        return failure();
      }
      // Input storage order is N,spatial_dims...,Ci.
      if (dimensionNumbers.input_batch_dimension().getInt() != 0 ||
          dimensionNumbers.input_feature_dimension().getInt() !=
              (inputSpatialRank + 1)) {
        op.emitWarning()
            << "Could not lower conv op due to inconsistant storage type";
        return failure();
      }

      const int kernelSpatialRank =
          std::distance(dimensionNumbers.kernel_spatial_dimensions().begin(),
                        dimensionNumbers.kernel_spatial_dimensions().end());
      // Filter storage order is spatial_dims...,C, Co.
      if (dimensionNumbers.kernel_input_feature_dimension().getInt() !=
              kernelSpatialRank ||
          dimensionNumbers.kernel_output_feature_dimension().getInt() !=
              (kernelSpatialRank + 1))
        return failure();

      const int outputSpatialRank =
          std::distance(dimensionNumbers.output_spatial_dimensions().begin(),
                        dimensionNumbers.output_spatial_dimensions().end());
      // Output storage order is N,spatial_dims..,Co.
      if (dimensionNumbers.output_batch_dimension().getInt() != 0 ||
          dimensionNumbers.output_feature_dimension().getInt() !=
              (outputSpatialRank + 1))
        return failure();

      if (inputSpatialRank != outputSpatialRank ||
          inputSpatialRank != kernelSpatialRank)
        return failure();

      auto inputSpatialDim =
          dimensionNumbers.input_spatial_dimensions().begin();
      auto kernelSpatialDim =
          dimensionNumbers.kernel_spatial_dimensions().begin();
      auto outputSpatialDim =
          dimensionNumbers.output_spatial_dimensions().begin();
      // Check spatial dims are ordred correctly.
      for (int i = 0; i < inputSpatialRank; ++i) {
        const int dim = i + 1;
        if ((*inputSpatialDim++).getZExtValue() != dim ||
            (*outputSpatialDim++).getZExtValue() != dim ||
            (*kernelSpatialDim++).getZExtValue() != i)
          return failure();
      }
    }

    auto inputShape = VMLAConversionTarget::getTensorShape(
        op.getLoc(), op.lhs(), typeConverter, rewriter);
    auto filterShape = VMLAConversionTarget::getTensorShape(
        op.getLoc(), op.rhs(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        op.getLoc(), op.getResult(), typeConverter, rewriter);

    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        op.getLoc(), op.getResult(), typeConverter, rewriter);

    auto lhsType =
        TypeAttr::get(op.lhs().getType().cast<ShapedType>().getElementType());
    auto rhsType =
        TypeAttr::get(op.lhs().getType().cast<ShapedType>().getElementType());

    SmallVector<int32_t, 4> windowStrides{1, 1};
    SmallVector<int32_t, 4> padding{0, 0, 0, 0};
    SmallVector<int32_t, 4> lhsDilation{1, 1};
    SmallVector<int32_t, 4> rhsDilation{1, 1};
    int32_t featureGroupCount = op.feature_group_count().getZExtValue();
    int32_t batchGroupCount = op.batch_group_count().getZExtValue();

    auto fill_optional = [](auto filed, SmallVector<int32_t, 4> *vec) {
      if (filed.hasValue()) {
        int index = 0;
        for (auto attribute : filed.getValue()) {
          (*vec)[index++] = attribute.getZExtValue();
        }
      }
    };

    fill_optional(op.window_strides(), &windowStrides);
    fill_optional(op.padding(), &padding);
    fill_optional(op.lhs_dilation(), &lhsDilation);
    fill_optional(op.rhs_dilation(), &rhsDilation);

    // Lower only what VMLA runtime supports.
    if (rhsDilation[0] != 1 || rhsDilation[1] != 1) {
      op.emitWarning() << "De-convoution isn't supported";
      return failure();
    }

    if (batchGroupCount != 1) {
      op.emitWarning() << "Batch group convoution isn't supported";
      return failure();
    }

    rewriter.create<IREE::VMLA::ConvOp>(
        op.getLoc(), op.lhs(), inputShape, op.rhs(), filterShape, dst, dstShape,
        rewriter.getI32VectorAttr(windowStrides),
        rewriter.getI32VectorAttr(padding),
        rewriter.getI32VectorAttr(lhsDilation),
        rewriter.getI32VectorAttr(rhsDilation),
        rewriter.getI32IntegerAttr(featureGroupCount),
        rewriter.getI32IntegerAttr(batchGroupCount), lhsType, rhsType, rhsType);

    rewriter.replaceOp(op, dst);

    return success();
  }
  TypeConverter &typeConverter;
};

}  // namespace

void populateHLOConvToVMLAPatterns(MLIRContext *context,
                                   OwningRewritePatternList &patterns,
                                   TypeConverter &typeConverter) {
  patterns.insert<VMLAConvOpConverter>(context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
