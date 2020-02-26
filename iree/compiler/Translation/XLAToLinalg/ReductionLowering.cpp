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

//===- ReductionLowering.cpp ------------------------------------*-C++//-*-===//
//
// Lower reduction dispatch regions to Linalg.
//
//===----------------------------------------------------------------------===//

#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/StandardTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tensorflow/compiler/mlir/xla/ir/hlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/map_xla_to_scalar_op.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Pass to lower the reduction dispatch functions to Linalg.
struct HLOReductionToLinalgPass : public ModulePass<HLOReductionToLinalgPass> {
  void runOnModule() override;
};

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the `reductionDim`-th element.
// TODO(hanchung): Use helpers in StructuredOpsUtils.h instead of hardcoded
// strings once the build system is set up.
ArrayAttr getNLoopsAttrs(Builder b, unsigned nLoops, int reductionDim) {
  SmallVector<Attribute, 3> attrs(nLoops, b.getStringAttr("parallel"));
  attrs[reductionDim] = b.getStringAttr("reduction");
  return b.getArrayAttr(attrs);
}

/// Base class for legalization of operations within the reduction apply
/// function (and the function itself).
template <typename OpTy>
class ApplyFunctionConversion : public OpConversionPattern<OpTy> {
 public:
  ApplyFunctionConversion(MLIRContext *context, TypeConverter &converter,
                          PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), converter(converter) {}

 protected:
  TypeConverter &converter;
};

/// The apply function has a signature (lhs, rhs) -> output, all of the same
/// tensor type t. This is converted to a function with the same signature but
/// with element types. E.g., "(tensor<f32>, tensor<f32>) -> tensor<f32>" will
/// be converted to "(f32, f32) -> f32".
struct ReductionApplyFnConversion final : ApplyFunctionConversion<FuncOp> {
  using ApplyFunctionConversion<FuncOp>::ApplyFunctionConversion;

  PatternMatchResult matchAndRewrite(
      FuncOp funcOp, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override;
};

}  // namespace

//===----------------------------------------------------------------------===//
// Reduction entry function body
//===----------------------------------------------------------------------===//

/// Adds a body with linalg ops for the reduction entry function `fn`. `fn` is
/// assumed to be empty. The apply function and the dimension to reduce must be
/// set as attributes on `fn`.
static LogicalResult addReductionEntryFnBody(OpBuilder &builder, FuncOp fn) {
  if (fn.getNumArguments() != 3) return failure();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(fn.addEntryBlock());
  auto src = fn.getArgument(0);
  auto initVal = fn.getArgument(1);
  auto dst = fn.getArgument(2);

  auto srcArgType = src.getType().template cast<ShapedType>();
  unsigned nInputRank = srcArgType.getRank();
  if (!nInputRank) return failure();

  auto loc = fn.getLoc();
  int reductionDim =
      fn.getAttrOfType<IntegerAttr>("iree.executable.reduction.dimension")
          .getInt();

  // Prepare indexing maps for linalg generic op. The elements are for src,
  // initial value and dst, respectively.
  SmallVector<Attribute, 2> indexingMaps;
  indexingMaps.emplace_back(
      AffineMapAttr::get(builder.getMultiDimIdentityMap(nInputRank)));
  indexingMaps.emplace_back(AffineMapAttr::get(AffineMap::get(
      nInputRank, /*symbolCount=*/0, {builder.getAffineConstantExpr(0)})));
  SmallVector<AffineExpr, 4> exprs;
  for (int i = 0; i < nInputRank; i++) {
    if (i == reductionDim) continue;
    exprs.push_back(builder.getAffineDimExpr(i));
  }
  indexingMaps.emplace_back(
      AffineMapAttr::get(AffineMap::get(nInputRank, /*symbolCount=*/0, exprs)));

  SmallVector<Type, 2> resultTypes = {};
  SmallVector<Value, 2> linalgOpArgs = {src, initVal, dst};
  auto linalgOp = builder.create<linalg::IndexedGenericOp>(
      loc, resultTypes, linalgOpArgs,
      builder.getI64IntegerAttr(2),  // args_in
      builder.getI64IntegerAttr(1),  // args_out
      builder.getArrayAttr(indexingMaps),
      getNLoopsAttrs(builder, nInputRank, reductionDim),
      /*doc=*/nullptr, /*fun=*/nullptr, /*library_call=*/nullptr);

  // Add a block to the region.
  auto *region = &linalgOp.region();
  auto *block = builder.createBlock(region, region->end());
  Type indexType = builder.getIndexType();
  Type elemType = srcArgType.getElementType();
  for (int i = 0; i < nInputRank; ++i) block->addArguments(indexType);
  for (int i = 0; i < linalgOp.getNumInputsAndOutputs(); ++i)
    block->addArguments(elemType);
  auto numArgs = block->getNumArguments();
  auto blockSrcArg = block->getArgument(numArgs - 3);
  auto blockInitArg = block->getArgument(numArgs - 2);
  auto blockDstArg = block->getArgument(numArgs - 1);
  auto zero = builder.create<ConstantOp>(loc, indexType,
                                         builder.getIntegerAttr(indexType, 0));
  auto cond = builder.create<CmpIOp>(loc, CmpIPredicate::eq,
                                     block->getArgument(reductionDim),
                                     zero.getResult());
  auto applyFn =
      fn.getAttrOfType<FlatSymbolRefAttr>("iree.executable.reduction.apply");
  auto lhs = builder.create<SelectOp>(loc, cond, blockInitArg, blockDstArg);
  auto rhs = blockSrcArg;
  auto result =
      builder.create<CallOp>(loc, applyFn, elemType, ValueRange{lhs, rhs});
  builder.create<linalg::YieldOp>(loc, result.getResult(0));

  builder.setInsertionPointAfter(linalgOp);
  builder.create<ReturnOp>(loc);

  return success();
}

//===----------------------------------------------------------------------===//
// Apply function conversion
//===----------------------------------------------------------------------===//

PatternMatchResult ReductionApplyFnConversion::matchAndRewrite(
    FuncOp funcOp, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  auto fnType = funcOp.getType();
  if (fnType.getNumInputs() != 2 || fnType.getNumResults() != 1)
    return matchFailure();
  if (fnType.getInput(0) != fnType.getInput(1) ||
      fnType.getInput(0) != fnType.getResult(0))
    return matchFailure();

  TypeConverter::SignatureConversion signatureConverter(fnType.getNumInputs());
  auto convertedType = converter.convertType(fnType.getInput(0));
  if (!convertedType) return matchFailure();
  signatureConverter.addInputs(0, convertedType);
  signatureConverter.addInputs(1, convertedType);
  auto newFn = rewriter.cloneWithoutRegions(funcOp);
  rewriter.inlineRegionBefore(funcOp.getBody(), newFn.getBody(), newFn.end());
  newFn.setType(rewriter.getFunctionType(signatureConverter.getConvertedTypes(),
                                         convertedType));
  rewriter.applySignatureConversion(&newFn.getBody(), signatureConverter);
  rewriter.eraseOp(funcOp);
  return matchSuccess();
}

//===----------------------------------------------------------------------===//
// ReductionOp
//===----------------------------------------------------------------------===//

/// Operations within the apply function need to be converted to standard ops.
template <typename OpTy>
struct ReductionOpConversion final : public ApplyFunctionConversion<OpTy> {
  using ApplyFunctionConversion<OpTy>::ApplyFunctionConversion;

  PatternMatchResult matchAndRewrite(
      OpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (operands.size() != 2) return this->matchFailure();
    SmallVector<Type, 1> resultElemTypes = {operands[0].getType()};
    Value opResult = xla_lhlo::MapXlaOpToStdScalarOp<OpTy>(op, resultElemTypes,
                                                           operands, &rewriter);
    rewriter.replaceOp(op, opResult);
    return this->matchSuccess();
  }
};

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

static LogicalResult getApplyFunctions(ModuleOp module,
                                       SmallVectorImpl<Operation *> &applyFns) {
  auto fns = module.getOps<FuncOp>();
  SymbolTable table(module);
  for (auto funcOp : fns) {
    if (!funcOp.getAttr("iree.executable.reduction")) continue;
    auto applyFnSymRef = funcOp.template getAttrOfType<FlatSymbolRefAttr>(
        "iree.executable.reduction.apply");
    auto applyFn = table.lookup<FuncOp>(applyFnSymRef.getValue());
    if (!applyFn) return module.emitError("can't find the apply function");
    applyFns.push_back(applyFn.getOperation());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pattern builder
//===----------------------------------------------------------------------===//

static LogicalResult lowerReductionApplyFnToLinalg(MLIRContext *context,
                                                   ArrayRef<Operation *> fns) {
  TypeConverter converter;
  converter.addConversion([](Type type) {
    return type.isSignlessIntOrFloat() ? type : Optional<Type>();
  });
  converter.addConversion([](RankedTensorType type) {
    return type.getRank() == 0 ? type.getElementType() : Optional<Type>();
  });

  OwningRewritePatternList patterns;
  patterns.insert<ReductionApplyFnConversion,
                  ReductionOpConversion<xla_hlo::AddOp> >(context, converter);
  ConversionTarget target(*context);
  target.addLegalDialect<linalg::LinalgDialect, StandardOpsDialect>();
  target.addDynamicallyLegalOp<FuncOp>(
      [&](FuncOp op) { return converter.isSignatureLegal(op.getType()); });
  if (failed(applyFullConversion(fns, target, patterns))) return failure();
  return success();
}

static LogicalResult lowerReductionEntryFnToLinalg(MLIRContext *context,
                                                   ModuleOp module) {
  OpBuilder builder(module.getBodyRegion());
  for (auto fn : module.getOps<FuncOp>()) {
    if (!fn.getAttr("iree.executable.reduction") || !fn.empty()) continue;
    if (failed(addReductionEntryFnBody(builder, fn))) return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass for invoking the conversion
//===----------------------------------------------------------------------===//

void HLOReductionToLinalgPass::runOnModule() {
  MLIRContext *context = &getContext();
  auto module = getModule();

  if (failed(lowerReductionEntryFnToLinalg(context, module)))
    return signalPassFailure();

  SmallVector<Operation *, 1> applyFns;
  if (failed(getApplyFunctions(module, applyFns))) return signalPassFailure();

  if (failed(lowerReductionApplyFnToLinalg(context, applyFns)))
    return signalPassFailure();
}

static PassRegistration<HLOReductionToLinalgPass> pass(
    "iree-hlo-reduction-to-linalg",
    "Convert the reduction dispatch functions to Linalg");

std::unique_ptr<Pass> createHLOReductionToLinalgPass() {
  return std::make_unique<HLOReductionToLinalgPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
