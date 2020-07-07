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

// Converts a simple mhlo.reduce op that performs independent individual
// computations into a set of mhlo.reduce ops. This is an intermediate
// conversion that may make it possible to use the much faster builtin VMLA
// reduction ops.
//
// Only supports single dimensional reductions and assumes that unrolling has
// been performed prior to conversion.
struct SplitIndependentReductionOpConversion
    : public OpConversionPattern<mhlo::ReduceOp> {
  SplitIndependentReductionOpConversion(MLIRContext *context,
                                        TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.dimensions().getNumElements() > 1) {
      srcOp.emitOpError() << "multi-dimensional reductions must be unrolled";
      return failure();
    } else if (srcOp.body().getBlocks().size() > 1) {
      // Control flow within the computation is not supported; bail to fallback.
      return failure();
    }
    auto &block = srcOp.body().getBlocks().front();
    mhlo::ReduceOp::Adaptor newOperands(operands);
    SmallVector<Value, 4> setResults;
    for (auto &op : block) {
      if (op.isKnownTerminator()) {
        continue;
      } else if (op.getOperands().size() != 2) {
        // Only binary ops are supported for builtins.
        return failure();
      }

      // Determine which argument set this op is acting on. For the builtins we
      // only support ops that act within a single set.
      // Our arguments are expanded tuples like <lhs0, lhs1>, <rhs0, rhs1>, so
      // this index gets the set offset.
      int opSetIndex =
          std::distance(block.args_begin(),
                        llvm::find(block.getArguments(), op.getOperand(0)));

      for (auto operand : op.getOperands()) {
        if (operand.getDefiningOp() != nullptr) {
          // Operand comes from another op within the block; unsupported.
          return failure();
        }
        int operandSetIndex =
            std::distance(block.args_begin(),
                          llvm::find(block.getArguments(), operand)) %
            newOperands.operands().size();
        if (operandSetIndex != opSetIndex) {
          // Operand is not coming from the same set as the other operands of
          // this op; unsupported.
          return failure();
        }
      }
      for (auto result : op.getResults()) {
        for (auto *user : result.getUsers()) {
          if (!user->isKnownTerminator()) {
            // Result is not directly returned from the block; unsupported.
            return failure();
          }
        }
      }

      // Create the new op for this set.
      Value operandArg = srcOp.operands()[opSetIndex];
      Value initArg = srcOp.init_values()[opSetIndex];
      auto splitOp = rewriter.create<mhlo::ReduceOp>(
          op.getLoc(), ValueRange{operandArg}, ValueRange{initArg},
          srcOp.dimensionsAttr());
      auto *splitBlock = new Block();
      splitOp.body().getBlocks().push_back(splitBlock);
      OpBuilder splitBuilder = OpBuilder::atBlockEnd(splitBlock);
      BlockAndValueMapping mapping;
      for (auto operand : op.getOperands()) {
        mapping.map(operand, splitBlock->addArgument(operand.getType()));
      }
      Operation *splitComputeOp = splitBuilder.clone(op, mapping);
      splitBuilder.create<mhlo::ReturnOp>(
          srcOp.getLoc(), ValueRange{*splitComputeOp->getResults().begin()});
      setResults.push_back(*splitOp.getResults().begin());
    }

    rewriter.replaceOp(srcOp, setResults);
    return success();
  }

  TypeConverter &typeConverter;
};

// Converts an mhlo.reduce with a single op to a builtin reduce op.
// This is meant to pair with the SplitIndependentReductionOpConversion that
// tries to unfuse/divide combined reductions. If this cannot match then the
// fallback path will be used and a VM loop will be emitted (slower, but can
// perform any reduction).
//
// Only supports single dimensional reductions and assumes that unrolling has
// been performed prior to conversion.
struct BuiltinReduceOpConversion : public OpConversionPattern<mhlo::ReduceOp> {
  BuiltinReduceOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context, /*benefit=*/1000),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.dimensions().getNumElements() > 1) {
      srcOp.emitOpError() << "multi-dimensional reductions must be unrolled";
      return failure();
    } else if (srcOp.body().getBlocks().size() > 1) {
      // Control flow within the computation is not supported; bail to fallback.
      return failure();
    } else if (srcOp.body().front().getOperations().size() > 2) {
      // Require splitting first.
      return failure();
    }

    auto operand = operands[0];
    auto operandShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.operands()[0], typeConverter, rewriter);
    auto initValue = operands[1];
    auto initValueShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.init_values()[0], typeConverter, rewriter);
    int dimension = srcOp.dimensions().getValue<IntegerAttr>({0}).getInt();
    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResults()[0], typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.getResults()[0], typeConverter, rewriter);
    auto elementType =
        srcOp.operands()[0].getType().cast<ShapedType>().getElementType();

    auto &computeOp = *srcOp.body().front().begin();
    if (isa<mlir::AddIOp>(computeOp) || isa<mlir::AddFOp>(computeOp) ||
        isa<mhlo::AddOp>(computeOp)) {
      rewriter.create<IREE::VMLA::ReduceSumOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape,
          rewriter.getI32IntegerAttr(dimension), dst, dstShape,
          TypeAttr::get(elementType));
    } else if (isa<mhlo::MinOp>(computeOp)) {
      rewriter.create<IREE::VMLA::ReduceMinOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape,
          rewriter.getI32IntegerAttr(dimension), dst, dstShape,
          TypeAttr::get(elementType));
    } else if (isa<mhlo::MaxOp>(computeOp)) {
      rewriter.create<IREE::VMLA::ReduceMaxOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape,
          rewriter.getI32IntegerAttr(dimension), dst, dstShape,
          TypeAttr::get(elementType));
    } else {
      computeOp.emitRemark() << "unsupported builtin reduction operation";
      return failure();
    }

    rewriter.replaceOp(srcOp, {dst});
    return success();
  }

  TypeConverter &typeConverter;
};

// Converts a generic mhlo.reduce to a VM loop.
//
// Only supports single dimensional reductions and assumes that unrolling has
// been performed prior to conversion.
struct GenericReduceOpConversion : public OpConversionPattern<mhlo::ReduceOp> {
  GenericReduceOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context), typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      mhlo::ReduceOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.dimensions().getNumElements() > 1) {
      srcOp.emitOpError() << "multi-dimensional reductions must be unrolled";
      return failure();
    }

    // TODO(benvanik): emit VM loop around computation.
    srcOp.emitOpError() << "generic reduction lowering not yet implemented";
    return failure();
  }

  TypeConverter &typeConverter;
};

struct BuiltinPoolingOpConversion
    : public OpConversionPattern<mhlo::ReduceWindowOp> {
  BuiltinPoolingOpConversion(MLIRContext *context, TypeConverter &typeConverter)
      : OpConversionPattern(context, /*benefit=*/1000),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(
      mhlo::ReduceWindowOp srcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (srcOp.body().getBlocks().size() > 1) {
      // Control flow within the computation is not supported; bail to fallback.
      return failure();
    } else if (srcOp.body().front().getOperations().size() > 2) {
      // Require splitting first.
      return failure();
    }

    auto operand = operands[0];
    auto operandShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.operand(), typeConverter, rewriter);
    auto initValue = operands[1];
    auto initValueShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.init_value(), typeConverter, rewriter);
    auto dst = VMLAConversionTarget::allocateOutputBuffer(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto dstShape = VMLAConversionTarget::getTensorShape(
        srcOp.getLoc(), srcOp.getResult(), typeConverter, rewriter);
    auto elementType =
        srcOp.operand().getType().cast<ShapedType>().getElementType();

    SmallVector<int32_t, 4> windowDimensions;
    for (const auto &value : srcOp.window_dimensions().getIntValues())
      windowDimensions.push_back(value.getSExtValue());
    int rank = windowDimensions.size();
    SmallVector<int32_t, 4> windowStrides(rank, 1);
    SmallVector<int32_t, 4> padding(rank, 0);
    for (unsigned i = 0; i < rank; ++i) {
      if (srcOp.window_strides())
        windowStrides[i] = srcOp.window_stridesAttr().getValue<int64_t>(i);
      if (srcOp.padding())
        padding[i] = srcOp.paddingAttr().getValue<int64_t>({i, 0});
    }

    auto &computeOp = *srcOp.body().front().begin();
    if (isa<mlir::AddIOp>(computeOp) || isa<mlir::AddFOp>(computeOp) ||
        isa<mhlo::AddOp>(computeOp)) {
      rewriter.create<IREE::VMLA::PoolingSumOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape, dst,
          dstShape, TypeAttr::get(elementType),
          rewriter.getI32VectorAttr(windowDimensions),
          rewriter.getI32VectorAttr(windowStrides),
          rewriter.getI32VectorAttr(padding));
    } else if (isa<mhlo::MinOp>(computeOp)) {
      rewriter.create<IREE::VMLA::PoolingMinOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape, dst,
          dstShape, TypeAttr::get(elementType),
          rewriter.getI32VectorAttr(windowDimensions),
          rewriter.getI32VectorAttr(windowStrides),
          rewriter.getI32VectorAttr(padding));
    } else if (isa<mhlo::MaxOp>(computeOp)) {
      rewriter.create<IREE::VMLA::PoolingMaxOp>(
          srcOp.getLoc(), operand, operandShape, initValue, initValueShape, dst,
          dstShape, TypeAttr::get(elementType),
          rewriter.getI32VectorAttr(windowDimensions),
          rewriter.getI32VectorAttr(windowStrides),
          rewriter.getI32VectorAttr(padding));
    } else {
      computeOp.emitRemark() << "unsupported builtin reduction operation";
      return failure();
    }

    rewriter.replaceOp(srcOp, {dst});
    return success();
  }

  TypeConverter &typeConverter;
};

}  // namespace

void populateHLOReductionToVMLAPatterns(MLIRContext *context,
                                        OwningRewritePatternList &patterns,
                                        TypeConverter &typeConverter) {
  patterns.insert<SplitIndependentReductionOpConversion>(context,
                                                         typeConverter);
  patterns.insert<BuiltinReduceOpConversion>(context, typeConverter);
  patterns.insert<BuiltinPoolingOpConversion>(context, typeConverter);
  patterns.insert<GenericReduceOpConversion>(context, typeConverter);
}

}  // namespace iree_compiler
}  // namespace mlir
