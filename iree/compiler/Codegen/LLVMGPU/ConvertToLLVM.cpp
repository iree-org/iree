// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

/// Scalarize math ops. It is needed to lower vector operation that don't have
/// vector support in CUDA and ROCM device library.
template <typename MathOpTy>
struct ScalarizeMathOp : public OpRewritePattern<MathOpTy> {
  using OpRewritePattern<MathOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOpTy mathOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = mathOp.getType().template dyn_cast<VectorType>();
    if (!vecType) return failure();
    Location loc = mathOp.getLoc();
    Value newVector = rewriter.create<arith::ConstantOp>(
        loc, vecType, rewriter.getZeroAttr(vecType));

    for (int64_t element : llvm::seq(int64_t(0), vecType.getNumElements())) {
      llvm::SmallVector<int64_t> indices;
      int64_t projectIndex = element;
      for (int64_t dim : llvm::seq(int64_t(0), vecType.getRank())) {
        int64_t index = projectIndex % vecType.getDimSize(dim);
        projectIndex = projectIndex / vecType.getDimSize(dim);
        indices.push_back(index);
      }
      SmallVector<Value> newOperands;
      for (Value operand : mathOp->getOperands()) {
        newOperands.push_back(
            rewriter.create<vector::ExtractOp>(loc, operand, indices));
      }
      Value scalarOp = rewriter.create<MathOpTy>(loc, newOperands);
      newVector =
          rewriter.create<vector::InsertOp>(loc, scalarOp, newVector, indices);
    }
    rewriter.replaceOp(mathOp, newVector);
    return success();
  }
};

struct ConvertSharedMemAllocOp : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    if (allocOp.getType().getMemorySpaceAsInt() != 3) return failure();
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (llvm::any_of(
            shape, [](int64_t dim) { return dim == ShapedType::kDynamicSize; }))
      return failure();
    // In CUDA workgroup memory is represented by a global variable.
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<FuncOp>();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp.getLoc(), "__shared_memory__",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false, /*alignment=*/IntegerAttr());
    symbolTable.insert(global);

    rewriter.setInsertionPointToStart(&(*funcOp.getBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.type(),
                                                     global.getName());
    return success();
  }
};

/// Pass to test in dialect transformation used to legalize the IR before
/// convertToNVVM/ConvertToROCDL.
class TestLLVMGPULegalizeOpPass
    : public TestLLVMGPUScalarizeMathOpBase<TestLLVMGPULegalizeOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    populateScalarizeMathOps(patterns);
    populateConvertSharedMemoryAllocOps(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

/// Convention with the HAL side to pass kernel arguments.
/// The bindings are ordered based on binding set and binding index then
/// compressed and mapped to dense set of arguments.
/// This function looks at the symbols and return the mapping between
/// InterfaceBindingOp and kernel argument index.
/// For instance if the kernel has (set, bindings) A(0, 1), B(1, 5), C(0, 6) it
/// will return the mapping [A, 0], [C, 1], [B, 2]
static llvm::SmallDenseMap<Operation *, size_t> getKernelArgMapping(
    Operation *func) {
  llvm::SmallDenseMap<Operation *, size_t> mapBindingArgIndex;
  llvm::SmallVector<IREE::HAL::InterfaceBindingOp> bindingUsed;
  Operation *symbolTableOp = SymbolTable::getNearestSymbolTable(func);
  SymbolTable::walkSymbolTables(symbolTableOp, true, [&](Operation *op, bool) {
    if (auto interface = dyn_cast<IREE::HAL::InterfaceOp>(op)) {
      interface.walk([&](Operation *symbolOp) {
        if (auto binding = dyn_cast<IREE::HAL::InterfaceBindingOp>(symbolOp)) {
          bindingUsed.push_back(binding);
        }
      });
    }
  });

  std::sort(bindingUsed.begin(), bindingUsed.end(),
            [](IREE::HAL::InterfaceBindingOp bindingA,
               IREE::HAL::InterfaceBindingOp bindingB) {
              uint64_t bindingAIndex = bindingA.binding().getZExtValue();
              uint64_t bindingASet = bindingA.set().getZExtValue();
              uint64_t bindingBIndex = bindingB.binding().getZExtValue();
              uint64_t bindingBSet = bindingB.set().getZExtValue();
              if (bindingASet == bindingBSet)
                return bindingAIndex < bindingBIndex;
              return bindingASet < bindingBSet;
            });
  for (auto binding : llvm::enumerate(bindingUsed)) {
    mapBindingArgIndex[binding.value()] = binding.index();
  }
  return mapBindingArgIndex;
}

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
    llvm::SmallDenseMap<Operation *, size_t> argMapping =
        getKernelArgMapping(funcOp);
    // There may be dead symbols, we pick i32 pointer as default argument type.
    SmallVector<Type, 8> llvmInputTypes(
        argMapping.size(), LLVM::LLVMPointerType::get(rewriter.getI32Type()));
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp input) {
      auto memrefType = input.getType().cast<MemRefType>();
      Type elType = memrefType.getElementType();
      auto llvmType =
          LLVM::LLVMPointerType::get(elType, memrefType.getMemorySpaceAsInt());
      IREE::HAL::InterfaceBindingOp binding = input.queryBindingOp();
      llvmInputTypes[argMapping[binding]] = llvmType;
    });
    // As a convention with HAL, push constants are appended as kernel arguments
    // after all the binding inputs.
    uint64_t numConstants = 0;
    funcOp.walk([&](IREE::HAL::InterfaceLoadConstantOp constant) {
      numConstants =
          std::max(constant.offset().getZExtValue() + 1, numConstants);
    });
    llvmInputTypes.resize(argMapping.size() + numConstants,
                          rewriter.getI32Type());
    signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : funcOp->getAttrs()) {
      if (attr.first == SymbolTable::getSymbolAttrName() ||
          attr.first == mlir::function_like_impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dso_local=*/false, funcAttrs);

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

    llvm::SmallDenseMap<Operation *, size_t> argMapping =
        getKernelArgMapping(llvmFuncOp);
    Location loc = op->getLoc();
    auto ireeBindingOp = cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
    IREE::HAL::InterfaceBindingSubspanOpAdaptor adaptor(
        operands, op->getAttrDictionary());
    MemRefType memrefType =
        ireeBindingOp.getResult().getType().dyn_cast<MemRefType>();
    IREE::HAL::InterfaceBindingOp binding = ireeBindingOp.queryBindingOp();
    mlir::BlockArgument llvmBufferArg =
        llvmFuncOp.getArgument(argMapping[binding]);
    // As a convention with HAL all the kernel argument pointers are 16Bytes
    // aliigned.
    llvmFuncOp.setArgAttr(llvmBufferArg.getArgNumber(),
                          LLVM::LLVMDialect::getAlignAttrName(),
                          rewriter.getI32IntegerAttr(16));
    // Add the byte offset.
    Value llvmBufferBasei8Ptr = rewriter.create<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(rewriter.getIntegerType(8),
                                   llvmBufferArg.getType()
                                       .cast<LLVM::LLVMPointerType>()
                                       .getAddressSpace()),
        llvmBufferArg);
    llvmBufferBasei8Ptr = rewriter.create<LLVM::GEPOp>(
        loc, llvmBufferBasei8Ptr.getType(), llvmBufferBasei8Ptr,
        adaptor.byte_offset());
    auto llvmPtrType = LLVM::LLVMPointerType::get(
        memrefType.getElementType(), memrefType.getMemorySpaceAsInt());
    Value llvmBufferBasePtr =
        rewriter.create<LLVM::BitcastOp>(loc, llvmPtrType, llvmBufferBasei8Ptr);
    if (memrefType.hasStaticShape()) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      ValueRange dynamicDims = adaptor.dynamic_dims();
      assert(memrefType.getNumDynamicDims() == dynamicDims.size());
      int64_t rank = memrefType.getRank();

      // Build MemRef descriptor for this interface binding.
      auto desc = MemRefDescriptor::undef(
          rewriter, loc, typeConverter->convertType(memrefType));
      desc.setAllocatedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setAlignedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setConstantOffset(rewriter, loc, 0);

      // Update memref descriptor shape. Dynamic dimensions can be mixed with
      // static dimensions, like [128, ?, 128].
      int dynamicDimIndex = 0;
      for (int i = 0; i < rank; ++i) {
        if (memrefType.isDynamicDim(i)) {
          desc.setSize(rewriter, loc, i, dynamicDims[dynamicDimIndex++]);
        } else {
          desc.setConstantSize(rewriter, loc, i, memrefType.getDimSize(i));
        }
      }

      // Compute and update strides. Assume that MemRefs are row-major, that is,
      // following index linearization:
      //   x[i, j, k] = i * x.dim[1] * x.dim[2] + j * x.dim[2] + k
      desc.setConstantStride(rewriter, loc, rank - 1, 1);
      for (int i = rank - 2; i >= 0; --i) {
        auto stride = desc.stride(rewriter, loc, i + 1);
        auto dim = desc.size(rewriter, loc, i + 1);
        Value strideVal = rewriter.create<LLVM::MulOp>(loc, stride, dim);
        desc.setStride(rewriter, loc, i, strideVal);
      }
      rewriter.replaceOp(op, {desc});
    }

    return success();
  }
};

class ConvertIREEConstantOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertIREEConstantOp(MLIRContext *context,
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

    llvm::SmallDenseMap<Operation *, size_t> argMapping =
        getKernelArgMapping(llvmFuncOp);
    auto ireeConstantOp = cast<IREE::HAL::InterfaceLoadConstantOp>(op);
    mlir::BlockArgument llvmBufferArg = llvmFuncOp.getArgument(
        argMapping.size() + ireeConstantOp.offset().getZExtValue());
    assert(llvmBufferArg.getType().isInteger(32));
    Type dstType = getTypeConverter()->convertType(ireeConstantOp.getType());
    rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, llvmBufferArg);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding GPU ops.
template <typename InterfaceOpTy, typename NewOpTy>
struct HALInterfaceWorkgroupOpsConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.dimension().getSExtValue());
    std::array<const char *, 3> dimAttr{"x", "y", "z"};
    rewriter.replaceOpWithNewOp<NewOpTy>(op, op.getType(), dimAttr[index]);
    return success();
  }
};

}  // anonymous namespace

void populateLLVMConversionPatterns(MLIRContext *context,
                                    OwningRewritePatternList &patterns,
                                    LLVMTypeConverter &converter) {
  patterns.insert<ConvertFunc, ConvertIREEBindingOp, ConvertIREEConstantOp>(
      context, converter);
}

void populateScalarizeMathOps(RewritePatternSet &patterns) {
  patterns.add<ScalarizeMathOp<math::SqrtOp>, ScalarizeMathOp<math::AbsOp>,
               ScalarizeMathOp<math::AtanOp>, ScalarizeMathOp<math::Atan2Op>,
               ScalarizeMathOp<math::CeilOp>, ScalarizeMathOp<math::CosOp>,
               ScalarizeMathOp<math::ExpOp>, ScalarizeMathOp<math::ExpM1Op>,
               ScalarizeMathOp<math::FloorOp>, ScalarizeMathOp<math::LogOp>,
               ScalarizeMathOp<math::Log1pOp>, ScalarizeMathOp<math::Log10Op>,
               ScalarizeMathOp<math::Log2Op>, ScalarizeMathOp<math::PowFOp>,
               ScalarizeMathOp<math::RsqrtOp>, ScalarizeMathOp<math::SinOp>,
               ScalarizeMathOp<math::SqrtOp>, ScalarizeMathOp<math::TanhOp>>(
      patterns.getContext());
}

void populateConvertSharedMemoryAllocOps(RewritePatternSet &patterns) {
  patterns.add<ConvertSharedMemAllocOp>(patterns.getContext());
}

void populateLowerHALInterfaceOp(RewritePatternSet &patterns) {
  patterns.insert<HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupIDOp, gpu::BlockIdOp>,
                  HALInterfaceWorkgroupOpsConverter<
                      IREE::HAL::InterfaceWorkgroupCountOp, gpu::GridDimOp>>(
      patterns.getContext());
}

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass() {
  return std::make_unique<TestLLVMGPULegalizeOpPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
