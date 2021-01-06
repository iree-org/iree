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

#include "iree/compiler/Conversion/CodegenUtils/FunctionUtils.h"
#include "iree/compiler/Conversion/LinalgToLLVM/Passes.h"
#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/IREE/IR/IREEDialect.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

static constexpr int kIndexPackedBuffer = 0;
static constexpr int kIndexPushConstant = 1;
static constexpr int kIndexWorkGroupId = 2;
static constexpr int kIndexWorkGroupCount = 3;
static constexpr int kIndexWorkGroupSize = 4;

// Convert to an LLVMFuncOp form with LLVM types. This implements an ABI that is
// compatible with IREE and which cannot be represented in std atm.
// Since it cannot be represented in std, this is an IREE-specific conversion.
// The conversion rewrites the argument-less and return-less function:
//
// ```
//    func @foo() { ... }
// ```
//
// into:
//
// ```
//    llvm.func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//                  %push_constant: !llvm.ptr<i32>,
//                  workgroup_id[3]: !llvm.ptr<i32>,
//                  workgroup_count[3]: !llvm.ptr<i32>,
//                  workgroup_size[3]: !llvm.ptr<i32>)
// ```
class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType fnType = funcOp.getType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    // func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
    //          %push_constant: !llvm.ptr<i32>,
    //          workgroup_id[3]: !llvm.ptr<i32>,
    //          workgroup_count[3]: !llvm.ptr<i32>,
    //          workgroup_size[3]: !llvm.ptr<i32>)
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    auto packedBuffersArgsTy = LLVM::LLVMPointerType::get(
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 8)));
    auto pushConstantArgTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 32));
    auto xyzTy =
        LLVM::LLVMPointerType::get(LLVM::LLVMIntegerType::get(context, 32));
    signatureConverter.addInputs(
        {packedBuffersArgsTy, pushConstantArgTy, xyzTy, xyzTy, xyzTy});
    SmallVector<LLVM::LLVMType, 5> llvmInputTypes{
        packedBuffersArgsTy, pushConstantArgTy, xyzTy, xyzTy, xyzTy};

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp.getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == mlir::impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, funcAttrs);

    // Copy all of funcOp's operations into newFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
                                newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
                                           &signatureConverter)))
      return failure();

    rewriter.eraseOp(funcOp);
    return success();
  }
};

// Assumes the enclosing FuncOp has been converted to IREE-compatible ABI:
// ```
//    llvm.func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//                  %push_constant: !llvm.ptr<i32>,
//                  workgroup_id[3]: !llvm.ptr<i32>,
//                  workgroup_count[3]: !llvm.ptr<i32>,
//                  workgroup_size[3]: !llvm.ptr<i32>)
// ```
//
// Rewrites hal.interface.subspan into a subview into the proper buffer
// extracted from `packed_buffer_args`.
class ConvertHALInterfaceBindingSubspanToView : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceBindingSubspanToView(MLIRContext *context,
                                                   LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceBindingSubspanOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto parentFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!parentFuncOp) return failure();

    assert(parentFuncOp.getNumArguments() > 0 &&
           parentFuncOp.getArgument(kIndexPackedBuffer)
               .getType()
               .isa<LLVM::LLVMType>());

    Location loc = op->getLoc();
    auto interfaceBindingSubspanOp =
        cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
    IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(operands);
    MemRefType memrefType =
        interfaceBindingSubspanOp.getResult().getType().dyn_cast<MemRefType>();
    if (!memrefType) return failure();

    // Fetch the interface binding op and extract the buffer index from void**.
    auto symbol = SymbolTable::lookupNearestSymbolFrom(
        op, interfaceBindingSubspanOp.binding());
    auto interfaceBindingOp = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    Value bufferIndex =
        rewriter.create<ConstantIndexOp>(loc, interfaceBindingOp.binding());
    Value llvmBufferIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, typeConverter->convertType(bufferIndex.getType()), bufferIndex);
    Value llvmBufferBasePtrAddr = rewriter.create<LLVM::GEPOp>(
        loc, parentFuncOp.getArgument(kIndexPackedBuffer).getType(),
        parentFuncOp.getArgument(kIndexPackedBuffer), llvmBufferIndex);
    Value llvmBufferBasePtr =
        rewriter.create<LLVM::LoadOp>(loc, llvmBufferBasePtrAddr);

    // Base memref is memref<?xi8>.
    MemRefType baseMemRefType =
        MemRefType::get(/*shape*/ {-1}, rewriter.getIntegerType(8),
                        /*layoutMap*/ {}, memrefType.getMemorySpace());
    // Just create a descriptor and set the allocatedPtr and alignedPtr.
    // The size is deemed unimportant at this point.
    Value zeroIndex = rewriter.create<ConstantIndexOp>(loc, 0);
    Value llvmZeroIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, typeConverter->convertType(zeroIndex.getType()), zeroIndex);
    Value oneIndex = rewriter.create<ConstantIndexOp>(loc, 0);
    Value llvmOneIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, typeConverter->convertType(oneIndex.getType()), oneIndex);
    auto llvmBaseDesc = MemRefDescriptor::undef(
        rewriter, loc, typeConverter->convertType(baseMemRefType));
    llvmBaseDesc.setAllocatedPtr(rewriter, loc, llvmBufferBasePtr);
    llvmBaseDesc.setAlignedPtr(rewriter, loc, llvmBufferBasePtr);
    llvmBaseDesc.setOffset(rewriter, loc, llvmZeroIndex);
    // TODO(nicolasvasilache): get proper dynamic size if needed.
    llvmBaseDesc.setSize(rewriter, loc, 0, llvmOneIndex);
    llvmBaseDesc.setStride(rewriter, loc, 0, llvmOneIndex);
    // Dialect cast to the `baseMemRefType` so we can take the view in std land.
    Value baseDesc =
        rewriter.create<LLVM::DialectCastOp>(loc, baseMemRefType, llvmBaseDesc);

    // TODO(nicolasvasilache): adapt the following to extract dynamic sizes from
    // push constant.
    SmallVector<Value, 4> dynamicSizes;
#if 0
    // Go grab dynamic sizes from special buffer location.
    Value pushConstantIndex = rewriter.create<ConstantIndexOp>(loc, 0);
        //adaptor.binding().cast<IREE::HAL::InterfaceBindingOp>().);
    Value llvmPushConstantIndex = rewriter.create<LLVM::DialectCastOp>(
        loc,
        typeConverter->convertType(pushConstantIndex.getType()),
        pushConstantIndex);
    Value llvmPushConstantrBasePtr = rewriter.create<LLVM::GEPOp>(
        loc,
        parentFuncOp.getArgument(kIndexPushConstant).getType(),
        parentFuncOp.getArgument(kIndexPushConstant),
        llvmPushConstantIndex);
    rewriter.create<LLVM::LoadOp>(loc, llvmPushConstantrBasePtr);
#endif

    // Construct view memref from base memref in std land and cast to LLVM.
    Value view = rewriter.create<ViewOp>(
        loc, op->getResultTypes().front(), baseDesc,
        interfaceBindingSubspanOp.byte_offset(), dynamicSizes);
    rewriter.replaceOpWithNewOp<LLVM::DialectCastOp>(
        op, typeConverter->convertType(view.getType()), view);

    return success();
  }
};

// ArgIndex hardcodes knowledge of argument position of `Op` in the
// IREE-compatible ABI.
template <typename Op, int ArgIndex>
class ConvertWorkgroupInfoOpPattern : public ConvertToLLVMPattern {
 public:
  explicit ConvertWorkgroupInfoOpPattern(MLIRContext *context,
                                         LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(Op::getOperationName(), context, converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto newFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    // Bail until enclosing FuncOp has been converted.
    if (!newFuncOp) return failure();

    Location loc = op->getLoc();
    auto xyzArgument = newFuncOp.getArgument(ArgIndex);

    Value dimIndex = rewriter.create<ConstantIndexOp>(
        loc, op->getAttrOfType<IntegerAttr>("dimension").getInt());
    Value llvmDimIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, typeConverter->convertType(dimIndex.getType()), dimIndex);

    auto dimPtr = rewriter.createOrFold<LLVM::GEPOp>(
        loc, typeConverter->convertType(xyzArgument.getType()), xyzArgument,
        llvmDimIndex);
    auto dimValue = rewriter.createOrFold<LLVM::LoadOp>(loc, dimPtr);

    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(
        op, typeConverter->convertType(op->getResult(0).getType()), dimValue);
    return success();
  }
};

struct ConvertToLLVMPass
    : public PassWrapper<ConvertToLLVMPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override;
};

}  // namespace

void ConvertToLLVMPass::runOnOperation() {
  // Run Vector -> Vector transformations ahead of conversion to LLVM.
  {
    OwningRewritePatternList patterns;
    vector::populateVectorToVectorCanonicalizationPatterns(patterns,
                                                           &getContext());
    vector::populateVectorSlicesLoweringPatterns(patterns, &getContext());
    vector::populateVectorContractLoweringPatterns(patterns, &getContext());
    applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }

  auto module = getOperation();

  LLVMTypeConverter converter(&getContext());
  converter.addConversion([](Shape::RankedShapeType, SmallVectorImpl<Type> &) {
    return success();
  });

  OwningRewritePatternList patterns;
  populateAffineToStdConversionPatterns(patterns, &getContext());
  populateLoopToStdConversionPatterns(patterns, &getContext());
  populateExpandTanhPattern(patterns, &getContext());
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns, &getContext());
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns);

  // clang-format off
  patterns.insert<
    ConvertFunc,
    ConvertHALInterfaceBindingSubspanToView,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupIDOp, kIndexWorkGroupId>,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupCountOp, kIndexWorkGroupCount>,
    ConvertWorkgroupInfoOpPattern<
      IREE::HAL::InterfaceWorkgroupSizeOp, kIndexWorkGroupSize>
  >(&getContext(), converter);
  // clang-format on

  LLVMConversionTarget target(getContext());
  // IREE::HAL::InterfaceOp will be removed after successful conversion of the
  // rest of the IR.
  target.addLegalOp<ModuleOp, ModuleTerminatorOp, IREE::HAL::InterfaceOp,
                    IREE::HAL::InterfaceBindingOp, IREE::HAL::InterfaceEndOp>();
  target.addIllegalOp<FuncOp>();
  target.addIllegalDialect<ShapeDialect, StandardOpsDialect, IREEDialect,
                           IREE::HAL::HALDialect, IREE::Flow::FlowDialect>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Once we're done with conversion, remove InterfaceOp.
  module.walk([](IREE::HAL::InterfaceOp op) { op.erase(); });
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVM2Pass() {
  return std::make_unique<ConvertToLLVMPass>();
}

static PassRegistration<ConvertToLLVMPass> pass(
    "iree-codegen-convert-to-llvm-2",
    "Perform final conversion from Linalg/HAL/Shape/Vector/Standard to "
    "LLVMIR dialect",
    [] { return std::make_unique<ConvertToLLVMPass>(); });

}  // namespace iree_compiler
}  // namespace mlir
