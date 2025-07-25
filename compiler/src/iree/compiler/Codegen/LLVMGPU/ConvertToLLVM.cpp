// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_TESTLLVMGPUSCALARIZEMATHOPPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

void ConvertToDynamicSharedMemory(ModuleOp moduleOp) {
  SymbolTableCollection symbolTableCollection;
  // Collect all the addressOfOps to static shared memory globals.
  SmallVector<LLVM::AddressOfOp> addressOfOps;
  moduleOp.walk([&](LLVM::AddressOfOp addressOfOp) {
    // Check that the global associated with this addressOfOp has shared memory
    // space.
    if (addressOfOp.getGlobal(symbolTableCollection).getAddrSpace() == 3)
      addressOfOps.push_back(addressOfOp);
  });
  if (addressOfOps.size() == 0)
    return;
  OpBuilder builder(moduleOp);
  builder.setInsertionPoint(&moduleOp.front());
  auto type =
      LLVM::LLVMArrayType::get(IntegerType::get(builder.getContext(), 8), 0);
  LLVM::GlobalOp global = builder.create<LLVM::GlobalOp>(
      moduleOp.getLoc(), type, /*isConstant=*/false, LLVM::Linkage::External,
      "__dynamic_shared_memory__", Attribute(),
      /*alignment=*/16, /*addr_space=*/3);
  uint32_t numberOfBytes = 0;
  // Replace the addressOfOps with correctly offseted pointers to dynamic
  // shared memory.
  llvm::SmallDenseMap<LLVM::GlobalOp, uint32_t> globalMemoryOffsetMap;
  for (auto addressOfOp : addressOfOps) {
    uint32_t offset = 0;
    auto globalOp = addressOfOp.getGlobal(symbolTableCollection);
    if (globalMemoryOffsetMap.count(globalOp)) {
      offset = globalMemoryOffsetMap[globalOp];
    } else {
      offset = numberOfBytes;
      if (std::optional<uint64_t> alignment = globalOp.getAlignment()) {
        offset = llvm::alignTo(offset, *alignment);
      }
      globalMemoryOffsetMap[globalOp] = offset;
      auto thisarray = globalOp.getType();
      DataLayout dataLayout = DataLayout::closest(addressOfOp);
      numberOfBytes = offset + dataLayout.getTypeSizeInBits(thisarray) / 8;
    }
    auto loc = addressOfOp.getLoc();
    builder.setInsertionPoint(addressOfOp);
    LLVM::AddressOfOp globalPtr =
        builder.create<LLVM::AddressOfOp>(loc, global);
    Value zero = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getI64IntegerAttr(0));
    Value offsetValue = builder.create<LLVM::ConstantOp>(
        loc, IntegerType::get(builder.getContext(), 64),
        builder.getI64IntegerAttr(offset));
    Value shiftedPtr = builder.create<LLVM::GEPOp>(
        loc, globalPtr.getType(), global.getGlobalType(), globalPtr,
        ValueRange({zero, offsetValue}));
    addressOfOp.replaceAllUsesWith(shiftedPtr);
    addressOfOp.erase();
  }
  // Add the amount of shared memory required as an attribute.
  auto variantOp = moduleOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
  if (variantOp != nullptr) {
    for (auto exportOp : variantOp.getExportOps()) {
      exportOp->setAttr(exportOp.getWorkgroupLocalMemoryAttrName(),
                        builder.getIndexAttr(numberOfBytes));
    }
  }
}

void setSharedMemoryAlignment(ModuleOp moduleOp, uint64_t newAlignment) {
  for (auto global : moduleOp.getOps<LLVM::GlobalOp>()) {
    if (global.getAddrSpace() == 3) {
      uint64_t baseAlignment = 0;
      if (std::optional<uint64_t> alignment = global.getAlignment()) {
        baseAlignment = alignment.value();
      }
      global.setAlignment(std::max<uint64_t>(baseAlignment, newAlignment));
    }
  }
}

namespace {

/// Scalarize math ops. It is needed to lower vector operation that don't have
/// vector support in CUDA and ROCM device library.
template <typename MathOpTy>
struct ScalarizeMathOp : public OpRewritePattern<MathOpTy> {
  using OpRewritePattern<MathOpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(MathOpTy mathOp,
                                PatternRewriter &rewriter) const override {
    auto vecType = llvm::dyn_cast<VectorType>(mathOp.getType());
    if (!vecType)
      return failure();
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
    if (!hasSharedMemoryAddressSpace(allocOp.getType()))
      return failure();
    ArrayRef<int64_t> shape = allocOp.getType().getShape();
    if (ShapedType::isDynamicShape(shape)) {
      return failure();
    }

    uint64_t alignement;
    if (std::optional<uint64_t> alignementInfo = allocOp.getAlignment()) {
      alignement = alignementInfo.value();
    } else {
      // If no alignment specified align at least to the size of an element.
      Type elType = allocOp.getType().getElementType();
      if (auto shapeType = llvm::dyn_cast<ShapedType>(elType))
        alignement =
            shapeType.getNumElements() * shapeType.getElementTypeBitWidth() / 8;
      else if (elType.isIndex()) {
        auto mod = allocOp->getParentOfType<ModuleOp>();
        LowerToLLVMOptions options(mod.getContext(), DataLayout(mod));
        alignement = options.getIndexBitwidth() / 8;
      } else
        alignement = elType.getIntOrFloatBitWidth() / 8;
    }
    // In CUDA workgroup memory is represented by a global variable.
    MemRefType allocType = allocOp.getType();
    auto funcOp = allocOp->getParentOfType<mlir::FunctionOpInterface>();
    auto moduleOp = funcOp->getParentOfType<ModuleOp>();
    SymbolTable symbolTable(moduleOp);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&moduleOp.front());
    auto global = rewriter.create<memref::GlobalOp>(
        funcOp.getLoc(), "__shared_memory__",
        /*sym_visibility=*/rewriter.getStringAttr("private"),
        /*type=*/allocType,
        /*initial_value=*/ElementsAttr(),
        /*constant=*/false,
        /*alignment=*/rewriter.getI64IntegerAttr(alignement));
    symbolTable.insert(global);

    rewriter.setInsertionPointToStart(&(*funcOp.getFunctionBody().begin()));
    rewriter.replaceOpWithNewOp<memref::GetGlobalOp>(allocOp, global.getType(),
                                                     global.getName());
    return success();
  }
};

/// Pass to test in dialect transformation used to legalize the IR before
/// convertToNVVM/ConvertToROCDL.
class TestLLVMGPULegalizeOpPass final
    : public impl::TestLLVMGPUScalarizeMathOpPassBase<
          TestLLVMGPULegalizeOpPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateConvertSharedMemoryAllocOps(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

namespace {
/// A package for the results of `analyzeSubspanOps` to avoid
/// arbitrary tuples. The default values are the results for an unused
/// binding, which is read-only, unused, and in address space 0.
struct BindingProperties {
  bool readonly = true;
  bool unused = true;
  unsigned addressSpace = 0;
};
} // namespace
/// Analyze subspan binding ops to recover properties of the binding, such as
/// if it is read-only and the address space it lives in.
static FailureOr<SmallVector<BindingProperties>>
analyzeSubspans(llvm::SetVector<IREE::HAL::InterfaceBindingSubspanOp> &subspans,
                int64_t numBindings, const LLVMTypeConverter *typeConverter) {
  SmallVector<BindingProperties> result(numBindings, BindingProperties{});
  for (auto subspan : subspans) {
    int64_t binding = subspan.getBinding().getSExtValue();
    result[binding].unused = false;
    result[binding].readonly &= IREE::HAL::bitEnumContainsAny(
        subspan.getDescriptorFlags().value_or(IREE::HAL::DescriptorFlags::None),
        IREE::HAL::DescriptorFlags::ReadOnly);
    unsigned bindingAddrSpace = 0;
    auto bindingType = dyn_cast<BaseMemRefType>(subspan.getType());
    if (bindingType) {
      bindingAddrSpace = *typeConverter->getMemRefAddressSpace(bindingType);
    }
    if (result[binding].addressSpace != 0 &&
        result[binding].addressSpace != bindingAddrSpace) {
      return subspan.emitOpError("address space for this op (" +
                                 Twine(bindingAddrSpace) +
                                 ") doesn't match previously found space (" +
                                 Twine(result[binding].addressSpace) + ")");
    }
    result[binding].addressSpace = bindingAddrSpace;
  }
  return result;
}

class ConvertFunc : public ConvertOpToLLVMPattern<func::FuncOp> {
public:
  explicit ConvertFunc(LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, 100) {}
  LogicalResult
  matchAndRewrite(func::FuncOp funcOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FunctionType fnType = funcOp.getFunctionType();
    (void)fnType;
    if (!funcOp.isPublic())
      return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    // Note: we assume that the pipeline layout is the same for all bindings
    // in this function.
    IREE::HAL::PipelineLayoutAttr layout;
    llvm::SetVector<IREE::HAL::InterfaceBindingSubspanOp> subspans;
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      if (!layout) {
        layout = subspanOp.getLayout();
      }
      subspans.insert(subspanOp);
    });

    funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp constOp) {
      if (!layout) {
        layout = constOp.getLayout();
      }
      return WalkResult::interrupt();
    });

    int64_t numBindings = 0;
    int64_t numConstants = 0;
    if (layout) {
      numConstants = layout.getConstants();
      numBindings = layout.getBindings().size();
    }

    FailureOr<SmallVector<BindingProperties>> maybeBindingsInfo =
        analyzeSubspans(subspans, numBindings, getTypeConverter());
    if (failed(maybeBindingsInfo))
      return failure();
    auto bindingsInfo = std::move(*maybeBindingsInfo);

    SmallVector<Type, 8> llvmInputTypes;
    llvmInputTypes.reserve(numBindings + numConstants);
    for (const auto &info : bindingsInfo) {
      llvmInputTypes.push_back(
          LLVM::LLVMPointerType::get(rewriter.getContext(), info.addressSpace));
    }
    // All the push constants are i32 and go at the end of the argument list.
    llvmInputTypes.resize(numBindings + numConstants, rewriter.getI32Type());

    if (!llvmInputTypes.empty())
      signatureConverter.addInputs(llvmInputTypes);

    // Construct newFunc with all attributes except return type & symbol name.
    SmallVector<NamedAttribute> funcAttrs;
    for (auto attr : funcOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == funcOp.getFunctionTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        LLVM::LLVMVoidType::get(rewriter.getContext()), llvmInputTypes);
    auto newFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C,
        /*comdat=*/nullptr, funcAttrs);

    // Copy all of funcOp's operations into newFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getFunctionBody(),
                                newFuncOp.getFunctionBody(), newFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getFunctionBody(),
                                           *typeConverter,
                                           &signatureConverter))) {
      return failure();
    }

    // Set argument attributes.
    Attribute unit = rewriter.getUnitAttr();
    for (auto [idx, info] : llvm::enumerate(bindingsInfo)) {
      // As a convention with HAL all the kernel argument pointers are 16Bytes
      // aligned.
      newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getAlignAttrName(),
                           rewriter.getI32IntegerAttr(16));
      // It is safe to set the noalias attribute as it is guaranteed that the
      // ranges within bindings won't alias.
      newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getNoAliasAttrName(), unit);
      newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getNonNullAttrName(), unit);
      newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getNoUndefAttrName(), unit);
      if (info.unused) {
        // While LLVM can work this out from the lack of use, we might as well
        // be explicit here just to be safe.
        newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getReadnoneAttrName(),
                             unit);
      } else if (info.readonly) {
        // Setting the readonly attribute here will generate non-coherent cache
        // loads.
        newFuncOp.setArgAttr(idx, LLVM::LLVMDialect::getReadonlyAttrName(),
                             unit);
      }
    }
    for (int64_t i = 0; i < numConstants; ++i) {
      // Push constants are never `undef`, annotate that here, just as with
      // bindings.
      newFuncOp.setArgAttr(numBindings + i,
                           LLVM::LLVMDialect::getNoUndefAttrName(), unit);
    }

    rewriter.eraseOp(funcOp);
    return success();
  }
};

struct ConvertIREEBindingSubspanOp final
    : public ConvertOpToLLVMPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp)
      return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    Location loc = op->getLoc();
    auto subspanOp = cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
    MemRefType memrefType =
        llvm::dyn_cast<MemRefType>(subspanOp.getResult().getType());
    mlir::BlockArgument llvmBufferArg =
        llvmFuncOp.getArgument(subspanOp.getBinding().getZExtValue());
    // Add the byte offset.
    Value llvmBufferBasePtr = llvmBufferArg;

    auto [strides, offset] = memrefType.getStridesAndOffset();
    if (memrefType.hasStaticShape() &&
        !llvm::any_of(strides, ShapedType::isDynamic) &&
        ShapedType::isStatic(offset)) {
      auto desc = MemRefDescriptor::fromStaticShape(
          rewriter, loc, *getTypeConverter(), memrefType, llvmBufferBasePtr);
      rewriter.replaceOp(op, {desc});
    } else {
      ValueRange dynamicDims = adaptor.getDynamicDims();
      assert(memrefType.getNumDynamicDims() == dynamicDims.size());
      int64_t rank = memrefType.getRank();

      // Build MemRef descriptor for this interface binding.
      auto desc = MemRefDescriptor::poison(
          rewriter, loc, typeConverter->convertType(memrefType));
      desc.setAllocatedPtr(rewriter, loc, llvmBufferBasePtr);
      desc.setAlignedPtr(rewriter, loc, llvmBufferBasePtr);

      auto llvmIndexType =
          typeConverter->convertType(IndexType::get(rewriter.getContext()));
      auto baseOffsetValue = adaptor.getByteOffset();
      if (ShapedType::isDynamic(offset)) {
        int32_t elementBitWidth =
            IREE::Util::getTypeBitWidth(memrefType.getElementType());
        Value elementBitWidthVal = rewriter.create<LLVM::ConstantOp>(
            loc, llvmIndexType, elementBitWidth);
        Value eight = rewriter.create<LLVM::ConstantOp>(loc, llvmIndexType, 8);
        Value bitOffset =
            rewriter.create<LLVM::MulOp>(loc, baseOffsetValue, eight);
        Value elementOffsetVal =
            rewriter.create<LLVM::UDivOp>(loc, bitOffset, elementBitWidthVal);
        desc.setOffset(rewriter, loc, elementOffsetVal);
      } else {
        desc.setConstantOffset(rewriter, loc, offset);
      }

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
      if (!strides.empty()) {
        assert(strides.back() == 1 &&
               "unexpected non-unit stride for innermost dimension");
        desc.setConstantStride(rewriter, loc, rank - 1, 1);
        OpFoldResult currentStride = rewriter.getIndexAttr(1);
        for (int i = rank - 1; i > 0; --i) {
          if (ShapedType::isDynamic(strides[i - 1])) {
            auto dim = desc.size(rewriter, loc, i);
            Value currentStrideVal;
            if (std::optional<int64_t> currentStrideInt =
                    getConstantIntValue(currentStride)) {
              currentStrideVal = rewriter.create<LLVM::ConstantOp>(
                  loc, llvmIndexType, currentStrideInt.value());
            } else {
              currentStrideVal = cast<Value>(currentStride);
            }
            currentStride =
                rewriter.create<LLVM::MulOp>(loc, currentStrideVal, dim)
                    .getResult();
            desc.setStride(rewriter, loc, i - 1, cast<Value>(currentStride));
          } else {
            currentStride = rewriter.getIndexAttr(strides[i - 1]);
            desc.setConstantStride(rewriter, loc, i - 1, strides[i - 1]);
          }
        }
      }
      rewriter.replaceOp(op, {desc});
    }

    return success();
  }
};

struct ConvertIREEConstantOp final
    : public ConvertOpToLLVMPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp)
      return failure();
    assert(llvmFuncOp.getNumArguments() > 0);

    auto ireeConstantOp = cast<IREE::HAL::InterfaceConstantLoadOp>(op);
    size_t numBindings = ireeConstantOp.getLayout().getBindings().size();
    mlir::BlockArgument llvmBufferArg = llvmFuncOp.getArgument(
        numBindings + ireeConstantOp.getOrdinal().getZExtValue());
    assert(llvmBufferArg.getType().isInteger(32));

    // If the constant has non-trivial assumptions placed on it about
    // its min and max values or divisibility, use that information to
    // annotate the corresponding arguments. The hasOneUse() check prevents us
    // from applying assumptions that don't hold at all usage sites.
    if (op.getResult().hasOneUse()) {
      OpOperand *operand = op.getResult().getUses().begin().getOperand();
      auto assumeOp = dyn_cast<IREE::Util::AssumeIntOp>(operand->getOwner());
      if (assumeOp) {
        unsigned opIdx = operand->getOperandNumber();
        auto [min, max] = assumeOp.getUnionedUnsignedRange(opIdx);

        if (min.has_value() && max.has_value()) {
          assert(*min <= std::numeric_limits<uint32_t>::max() &&
                 "Push-constant's maximum value can't be outside 32 bits, but "
                 "this is assumed");
          // Note: LLVM's range(iN lb, ub) is [lb, ub), while MLIR's is [lb,
          // ub], so we add 1 to the upper bound.
          llvmFuncOp.setArgAttr(llvmBufferArg.getArgNumber(),
                                LLVM::LLVMDialect::getRangeAttrName(),
                                rewriter.getAttr<LLVM::ConstantRangeAttr>(
                                    APInt(32, *min), APInt(32, *max) + 1));
        }

        auto divisibility = assumeOp.getUnionedUnsignedDivisor(opIdx);

        auto makeI32Const = [&](uint32_t val) -> Value {
          return rewriter.create<LLVM::ConstantOp>(
              assumeOp.getLoc(), rewriter.getI32Type(),
              rewriter.getI32IntegerAttr(val));
        };
        if (divisibility.has_value() && *divisibility > 1) {
          Location loc = assumeOp.getLoc();
          assert(*divisibility <= std::numeric_limits<uint32_t>::max() &&
                 "push constant shouldn't be statically divisible by a value "
                 "it can't hold");
          Value knownDivisibleBy = makeI32Const(*divisibility);
          // This'll almost always become an and
          Value lowPart = rewriter.create<LLVM::URemOp>(loc, llvmBufferArg,
                                                        knownDivisibleBy);
          Value zero = makeI32Const(0);
          Value isEvenlyDivided = rewriter.create<LLVM::ICmpOp>(
              loc, LLVM::ICmpPredicate::eq, lowPart, zero);
          rewriter.create<LLVM::AssumeOp>(loc, isEvenlyDivided);
        }
      }
    }

    Type dstType = getTypeConverter()->convertType(ireeConstantOp.getType());
    // llvm.zext requires that the result type has a larger bitwidth.
    if (dstType == llvmBufferArg.getType()) {
      rewriter.replaceOp(op, llvmBufferArg);
    } else {
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, dstType, llvmBufferArg);
    }
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count/size into
/// corresponding GPU ops.
template <typename InterfaceOpTy, typename NewOpTy>
struct HALInterfaceWorkgroupOpsConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.getDimension().getSExtValue());
    std::array<gpu::Dimension, 3> dimAttr{gpu::Dimension::x, gpu::Dimension::y,
                                          gpu::Dimension::z};
    NewOpTy newOp =
        rewriter.replaceOpWithNewOp<NewOpTy>(op, op.getType(), dimAttr[index]);
    if (IntegerAttr bound = op.getUpperBoundAttr())
      newOp.setUpperBoundAttr(bound);
    return success();
  }
};

class ConvertNullPointerOp
    : public ConvertOpToLLVMPattern<IREE::Codegen::NullPointerOp> {
public:
  using ConvertOpToLLVMPattern<
      IREE::Codegen::NullPointerOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Codegen::NullPointerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<LLVM::ZeroOp>(
        op, LLVM::LLVMPointerType::get(getContext()));
    return success();
  }
};

struct ConvertIREEUtilAssumeIntOp final
    : public ConvertOpToLLVMPattern<IREE::Util::AssumeIntOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IREE::Util::AssumeIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Bail until nested under an LLVMFuncOp.
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp)
      return failure();

    Location loc = op.getLoc();
    auto updateConds = [&](std::optional<Value> &conds, Value cond) {
      if (!conds)
        conds = cond;
      else
        conds = rewriter.create<LLVM::AndOp>(loc, *conds, cond);
    };
    // Materialize the assumptions that aren't atteched directly to arguments
    // in order to account for the fact that i64 inputs get passed in as a pair
    // of i32 constants.
    for (auto [idx, mlirVal, llvmVal] :
         llvm::enumerate(op.getOperands(), adaptor.getOperands())) {
      if (mlirVal.getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>())
        continue;
      std::optional<Value> conds;
      Type type = llvmVal.getType();
      auto [min, max] = op.getUnionedUnsignedRange(idx);
      // This should be a range() bundle but LLVM doesn't understand those yet.
      if (min.has_value() && *min > 0) {
        Value minConst = createIndexAttrConstant(rewriter, loc, type, *min);
        Value minCond = rewriter.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::uge, llvmVal, minConst);
        updateConds(conds, minCond);
      }
      if (max.has_value()) {
        Value maxConst = createIndexAttrConstant(rewriter, loc, type, *max);
        Value maxCond = rewriter.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::ule, llvmVal, maxConst);
        updateConds(conds, maxCond);
      }
      std::optional<uint64_t> divisor = op.getUnionedUnsignedDivisor(idx);
      if (divisor && *divisor > 1) {
        Value divisorConst =
            createIndexAttrConstant(rewriter, loc, type, *divisor);
        Value remainder =
            rewriter.create<LLVM::URemOp>(loc, llvmVal, divisorConst);
        Value zero = createIndexAttrConstant(rewriter, loc, type, 0);
        Value divisorCond = rewriter.create<LLVM::ICmpOp>(
            loc, LLVM::ICmpPredicate::eq, remainder, zero);
        updateConds(conds, divisorCond);
      }

      if (conds.has_value()) {
        rewriter.create<LLVM::AssumeOp>(loc, *conds);
      }
    }
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};
} // namespace

void populateLLVMConversionPatterns(MLIRContext *context,
                                    RewritePatternSet &patterns,
                                    LLVMTypeConverter &converter) {
  patterns.add<ConvertFunc, ConvertIREEBindingSubspanOp, ConvertIREEConstantOp,
               ConvertNullPointerOp, ConvertIREEUtilAssumeIntOp>(converter);
  converter.addConversion([context](IREE::Codegen::NullPointerType type) {
    return LLVM::LLVMPointerType::get(context);
  });
}

void populateConvertSharedMemoryAllocOps(RewritePatternSet &patterns) {
  patterns.add<ConvertSharedMemAllocOp>(patterns.getContext());
}

void populateLowerHALInterfaceOp(RewritePatternSet &patterns) {
  patterns.add<HALInterfaceWorkgroupOpsConverter<
                   IREE::HAL::InterfaceWorkgroupIDOp, gpu::BlockIdOp>,
               HALInterfaceWorkgroupOpsConverter<
                   IREE::HAL::InterfaceWorkgroupSizeOp, gpu::BlockDimOp>,
               HALInterfaceWorkgroupOpsConverter<
                   IREE::HAL::InterfaceWorkgroupCountOp, gpu::GridDimOp>>(
      patterns.getContext());
}

static IntegerAttr wrapNumericMemorySpace(MLIRContext *ctx, unsigned space) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), space);
}

void populateGpuMemorySpaceAttributeConversions(
    TypeConverter &typeConverter, const MemorySpaceMapping &mapping) {
  typeConverter.addTypeAttributeConversion(
      [mapping](BaseMemRefType type, gpu::AddressSpaceAttr memorySpaceAttr) {
        gpu::AddressSpace memorySpace = memorySpaceAttr.getValue();
        unsigned addressSpace = mapping(memorySpace);
        return wrapNumericMemorySpace(memorySpaceAttr.getContext(),
                                      addressSpace);
      });
}

} // namespace mlir::iree_compiler
