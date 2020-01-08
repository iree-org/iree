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

//===- ReductionFunctionLowering.cpp ---------------------------*- C++//-*-===//
//
// Lowering for reduction function body
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/SPIRV/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/SPIRVOps.h"
#include "mlir/Dialect/StandardOps/Ops.h"
#include "mlir/IR/Function.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Type converter for legalization of reduction apply function.
class SPIRVReductionTypeConverter : public TypeConverter {
 public:
  Type convertType(Type t) override;
};

/// Base class for legalization of operations within the reduction apply
/// function (and the function itself).
template <typename OpTy>
class SPIRVReductionConversion : public OpConversionPattern<OpTy> {
 public:
  SPIRVReductionConversion(MLIRContext *context,
                           SPIRVReductionTypeConverter &typeConverter,
                           PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit),
        typeConverter(typeConverter) {}

 protected:
  SPIRVReductionTypeConverter &typeConverter;
};

/// The apply function has a signature (lhs, rhs) -> output, all of the same
/// type t. This is converted to a function with the signature (t, !spv.ptr<t,
/// StorageBuffer>) -> (), where the first argument is the update, the second
/// argument is the buffer which contains the result of the reduction.
// TODO(ravishankarm): This is assuming storage class is StorageBuffer. This
// needs to be generalized.
class ReductionApplyFnConversion final
    : public SPIRVReductionConversion<FuncOp> {
 public:
  using SPIRVReductionConversion<FuncOp>::SPIRVReductionConversion;

  PatternMatchResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

/// Return operation conversion. Just converts ReturnOp to
/// spirv::ReturnOp.
// TODO: This can be moved into DRR.
template <typename ReturnOpTy>
class ReturnOpConversion final : public SPIRVReductionConversion<ReturnOpTy> {
 public:
  using SPIRVReductionConversion<ReturnOpTy>::SPIRVReductionConversion;

  PatternMatchResult matchAndRewrite(
      ReturnOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spirv::ReturnOp>(op);
    return this->matchSuccess();
  }
};

/// Operations within the apply function need to be converted to a atomic
/// update.
template <typename OpTy, typename ReplacementOpTy>
class ReductionOpConversion final : public SPIRVReductionConversion<OpTy> {
 public:
  using SPIRVReductionConversion<OpTy>::SPIRVReductionConversion;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

Type SPIRVReductionTypeConverter::convertType(Type t) {
  if (spirv::SPIRVDialect::isValidType(t)) {
    return t;
  }
  if (auto tensorType = t.dyn_cast<RankedTensorType>()) {
    if (tensorType.getRank() == 0) {
      return tensorType.getElementType();
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Apply fn conversion.
//===----------------------------------------------------------------------===//

PatternMatchResult ReductionApplyFnConversion::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  if (fnType.getNumInputs() != 2 || fnType.getNumResults() != 1) {
    return matchFailure();
  }
  if (fnType.getInput(0) != fnType.getInput(1) ||
      fnType.getInput(0) != fnType.getResult(0)) {
    return matchFailure();
  }
  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  auto convertedType = typeConverter.convertType(fnType.getInput(0));
  if (!convertedType) {
    return matchFailure();
  }
  signatureConverter.addInputs(0, convertedType);
  signatureConverter.addInputs(
      1, spirv::PointerType::get(convertedType,
                                 spirv::StorageClass::StorageBuffer));
  auto newFn = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFn.getBody(), newFn.end());
  newFn.setType(rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                         llvm::None));
  rewriter.applySignatureConversion(&newFn.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

template <typename OpTy, typename ReplacementOpTy>
PatternMatchResult
ReductionOpConversion<OpTy, ReplacementOpTy>::matchAndRewrite(
    OpTy op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  if (operands.size() != 2) {
    return this->matchFailure();
  }
  // One of the replacement operands will be a pointer type, and another a value
  // type.
  Value ptr = operands[0];
  Value value = operands[1];
  if (!ptr.getType().isa<spirv::PointerType>()) std::swap(ptr, value);
  if (!ptr.getType().isa<spirv::PointerType>()) return this->matchFailure();
  rewriter.replaceOpWithNewOp<ReplacementOpTy>(
      op, ptr.getType().cast<spirv::PointerType>().getPointeeType(), ptr,
      spirv::Scope::Device, spirv::MemorySemantics::AcquireRelease, value);
  return this->matchSuccess();
}

//===----------------------------------------------------------------------===//
// Pattern builder
//===----------------------------------------------------------------------===//
LogicalResult lowerReductionApplyFunction(MLIRContext *context,
                                          ArrayRef<Operation *> fns) {
  OwningRewritePatternList patterns;
  SPIRVReductionTypeConverter typeConverter;
  patterns
      .insert<ReductionApplyFnConversion,
              ReductionOpConversion<xla_hlo::MinOp, spirv::AtomicSMinOp>,
              ReductionOpConversion<xla_hlo::MaxOp, spirv::AtomicSMaxOp>,
              ReductionOpConversion<AddIOp, spirv::AtomicIAddOp>,
              ReturnOpConversion<IREE::ReturnOp>, ReturnOpConversion<ReturnOp>>(
          context, typeConverter);
  ConversionTarget target(*context);
  target.addLegalDialect<spirv::SPIRVDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return typeConverter.isSignatureLegal(op.getType()); });
  if (failed(applyPartialConversion(fns, target, patterns))) {
    return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass for invoking the conversion.
//===----------------------------------------------------------------------===//

namespace {
// Pass to invoke the reduction fn lowering from command line.
class ReduceFnSPIRVLoweringPass final
    : public OperationPass<ReduceFnSPIRVLoweringPass, ModuleOp> {
 private:
  void runOnOperation() override;
};
}  // namespace

void ReduceFnSPIRVLoweringPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *context = &getContext();
  if (failed(lowerReductionApplyFunction(context, module.getOperation()))) {
    return signalPassFailure();
  }
}

static PassRegistration<ReduceFnSPIRVLoweringPass> pass(
    "iree-spirv-reduction-fn-lowering",
    "Convert the reduction apply function within reduction dispatches to "
    "SPIR-V");

}  // namespace iree_compiler
}  // namespace mlir
