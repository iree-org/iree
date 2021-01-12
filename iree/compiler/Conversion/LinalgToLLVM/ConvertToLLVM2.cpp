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

// Return the index type: i32.
static Type getIndexTy(MLIRContext *context) {
  return IntegerType::get(context, 32);
}

// func foo(%packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>,
//          %push_constant: !llvm.ptr<i32>,
//          workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//          workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//          workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
static SmallVector<Type, 5> getABITypes(MLIRContext *context) {
  auto indexTy = getIndexTy(context);
  return SmallVector<Type, 5>{
      // %packed_buffer_args: !llvm.ptr<!llvm.ptr<i8>>
      LLVM::LLVMPointerType::get(
          LLVM::LLVMPointerType::get(IntegerType::get(context, 8))),
      // %push_constant: !llvm.ptr<i32>
      LLVM::LLVMPointerType::get(indexTy),
      // %workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3)),
      // %workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3)),
      // %workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>
      LLVM::LLVMPointerType::get(LLVM::LLVMArrayType::get(indexTy, 3))};
}

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
//                  workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
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
    //          workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
    //          workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    SmallVector<Type, 5> llvmInputTypes = getABITypes(context);
    signatureConverter.addInputs(llvmInputTypes);

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
//                  workgroup_id[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_count[3]: !llvm.ptr<!llvm.array<i32, 3>>,
//                  workgroup_size[3]: !llvm.ptr<!llvm.array<i32, 3>>)
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
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto llvmTypeConverter = getTypeConverter();
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
        loc, llvmTypeConverter->convertType(bufferIndex.getType()),
        bufferIndex);
    Value llvmBufferBasePtrAddr = rewriter.create<LLVM::GEPOp>(
        loc, llvmFuncOp.getArgument(kIndexPackedBuffer).getType(),
        llvmFuncOp.getArgument(kIndexPackedBuffer), llvmBufferIndex);
    Value llvmBufferBasePtr =
        rewriter.create<LLVM::LoadOp>(loc, llvmBufferBasePtrAddr);

    // Base memref is memref<?xi8>.
    MemRefType baseMemRefType =
        MemRefType::get(/*shape*/ {-1}, rewriter.getIntegerType(8),
                        /*layoutMap*/ {}, memrefType.getMemorySpace());
    // Just create a descriptor and set the allocatedPtr and alignedPtr.
    // The size is deemed unimportant at this point.
    Value oneIndex = rewriter.create<ConstantIndexOp>(loc, 1);
    Value llvmOneIndex = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmTypeConverter->convertType(oneIndex.getType()), oneIndex);
    Value llvmByteOffset = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmTypeConverter->convertType(oneIndex.getType()),
        interfaceBindingSubspanOp.byte_offset());
    // If no size is specified, just use size 1 as it does not matter.
    Value llvmByteLength =
        interfaceBindingSubspanOp.byte_length()
            ? rewriter.create<LLVM::DialectCastOp>(
                  loc, llvmTypeConverter->convertType(oneIndex.getType()),
                  interfaceBindingSubspanOp.byte_length())
            : llvmOneIndex;
    auto llvmBaseDesc = MemRefDescriptor::pack(
        rewriter, loc, *llvmTypeConverter, baseMemRefType,
        ValueRange{/*allocatedPointer=*/llvmBufferBasePtr,
                   /*alignedPointer=*/llvmBufferBasePtr,
                   /*offset=*/llvmByteOffset,
                   /*size[1]=*/llvmByteLength, /*stride[1]=*/llvmOneIndex});
    rewriter.replaceOp(op, llvmBaseDesc);
    return success();
  }
};

class ConvertHALInterfaceLoadConstant : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceLoadConstant(MLIRContext *context,
                                           LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceLoadConstantOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();

    assert(llvmFuncOp.getNumArguments() > 0);

    auto llvmTypeConverter = getTypeConverter();
    Location loc = op->getLoc();
    auto interfaceLoadConstantOp = cast<IREE::HAL::InterfaceLoadConstantOp>(op);
    Value pushConstantOffset = rewriter.create<ConstantIndexOp>(
        loc, interfaceLoadConstantOp.offset().getSExtValue());
    Value llvmPushConstantOffset = rewriter.create<LLVM::DialectCastOp>(
        loc, llvmTypeConverter->convertType(pushConstantOffset.getType()),
        pushConstantOffset);
    Value llvmPushConstantOffsetAddr = rewriter.create<LLVM::GEPOp>(
        loc, llvmFuncOp.getArgument(kIndexPushConstant).getType(),
        llvmFuncOp.getArgument(kIndexPushConstant), llvmPushConstantOffset);
    Value llvmPushConstantValue =
        rewriter.create<LLVM::LoadOp>(loc, llvmPushConstantOffsetAddr);
    Value indexValue = rewriter.create<LLVM::ZExtOp>(
        loc, llvmTypeConverter->convertType(rewriter.getIndexType()),
        llvmPushConstantValue);
    rewriter.replaceOp(op, indexValue);
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
    auto xyzArrayPtr = newFuncOp.getArgument(ArgIndex);
    auto xyzArrayValue = rewriter.createOrFold<LLVM::LoadOp>(loc, xyzArrayPtr);
    auto dimValue = rewriter.createOrFold<LLVM::ExtractValueOp>(
        loc, getIndexTy(op->getContext()), xyzArrayValue,
        rewriter.getI32ArrayAttr(ArrayRef<int>(
            op->getAttrOfType<IntegerAttr>("dimension").getInt())));

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
  {
    OwningRewritePatternList vectorToLoopsPatterns;
    populateVectorToSCFConversionPatterns(
        vectorToLoopsPatterns, &getContext(),
        VectorTransferToSCFOptions().setUnroll(true));
    applyPatternsAndFoldGreedily(getOperation(),
                                 std::move(vectorToLoopsPatterns));
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
    ConvertHALInterfaceLoadConstant,
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
