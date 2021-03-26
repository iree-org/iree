// Copyright 2021 Google LLC
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
#include "iree/compiler/Conversion/LinalgToNVVM/Passes.h"
#include "iree/compiler/Dialect/IREE/IR/IREEOps.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

class ConvertFunc : public ConvertToLLVMPattern {
 public:
  explicit ConvertFunc(MLIRContext *context, LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::FuncOp::getOperationName(), context,
                             converter, 100) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto funcOp = cast<FuncOp>(op);
    FunctionType fnType = funcOp.getType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    SmallVector<Type, 8> llvmInputTypes;
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp input) {
      auto memrefType = input.getType().cast<MemRefType>();
      Type elType = memrefType.getElementType();
      auto llvmType =
          LLVM::LLVMPointerType::get(elType, memrefType.getMemorySpaceAsInt());
      llvmInputTypes.push_back(llvmType);
    });
    signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp->getAttrs()) {
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

class ConvertIREEBindingOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEBindingOp(MLIRContext *context,
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

    Location loc = op->getLoc();
    auto ireeBindingOp = cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
    IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(operands);
    MemRefType memrefType =
        ireeBindingOp.getResult().getType().dyn_cast<MemRefType>();

    // Fetch the interface binding op and extract the buffer index from void**.
    auto symbol = SymbolTable::lookupNearestSymbolFrom(
        op, op->getAttrOfType<SymbolRefAttr>("binding"));
    auto interfaceBindingOp = cast<IREE::HAL::InterfaceBindingOp>(symbol);
    Value llvmBufferBasePtr =
        llvmFuncOp.getArgument(interfaceBindingOp.binding());
    if (memrefType.hasStaticShape()) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      // TODO: pull those paramters from HAL constants.
      assert(0 && "TODO: implement dynamic shape");
    }

    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding NVVM ops.
template <typename InterfaceOpTy, typename XOp, typename YOp, typename ZOp>
struct HALInterfaceWorkgroupOpsConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type i32Type = rewriter.getI32Type();
    Value newOp;
    int32_t index = static_cast<int32_t>(op.dimension().getSExtValue());
    switch (index) {
      case 0:
        newOp = rewriter.create<XOp>(loc, i32Type);
        break;
      case 1:
        newOp = rewriter.create<YOp>(loc, i32Type);
        break;
      case 2:
        newOp = rewriter.create<ZOp>(loc, i32Type);
        break;
      default:
        return failure();
    }

    newOp =
        rewriter.create<LLVM::SExtOp>(loc, rewriter.getIntegerType(64), newOp);
    rewriter.replaceOp(op, {newOp});
    return success();
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding NVVM equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToNVVMPass
    : public PassWrapper<ConvertToNVVMPass, OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, NVVM::NVVMDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(64);
    LLVMTypeConverter converter(m.getContext(), options);
    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    {
      OwningRewritePatternList patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      (void)applyPatternsAndFoldGreedily(m, std::move(patterns));
    }
    {
      OwningRewritePatternList llvmPatterns(&getContext());
      llvmPatterns.insert<ConvertFunc, ConvertIREEBindingOp>(m.getContext(),
                                                             converter);
      llvmPatterns
          .insert<HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupIDOp, NVVM::BlockIdXOp,
                      NVVM::BlockIdYOp, NVVM::BlockIdZOp>,
                  HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupCountOp, NVVM::GridDimXOp,
                      NVVM::GridDimYOp, NVVM::GridDimZOp>,
                  HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupSizeOp, NVVM::BlockDimXOp,
                      NVVM::BlockDimYOp, NVVM::BlockDimZOp>>(m.getContext());
      populateStdToLLVMConversionPatterns(converter, llvmPatterns);
      populateGpuToNVVMConversionPatterns(converter, llvmPatterns);
      LLVMConversionTarget target(getContext());
      populateStdToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToNVVMConversionLegality(target);
      target.addDynamicallyLegalOp<FuncOp>([&](FuncOp funcOp) {
        if (isEntryPoint(funcOp)) return false;
        return true;
      });
      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
        signalPassFailure();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass() {
  return std::make_unique<ConvertToNVVMPass>();
}

static PassRegistration<ConvertToNVVMPass> pass(
    "iree-codegen-convert-to-nvvm",
    "Perform final conversion from builtin/GPU/HAL/standard dialect to LLVM "
    "and NVVM dialects");

}  // namespace iree_compiler
}  // namespace mlir
