// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/DispatchABI.h"
#include "iree/compiler/Codegen/LLVMCPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "iree/compiler/Codegen/LLVMCPU/Utils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/schemas/instruments/dispatch.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

template <typename OpT>
struct ConvertOpToLLVMWithABIPattern : public ConvertOpToLLVMPattern<OpT> {
  ConvertOpToLLVMWithABIPattern(HALDispatchABI &abi,
                                LLVMTypeConverter &typeConverter,
                                PatternBenefit benefit = 1)
      : ConvertOpToLLVMPattern<OpT>(typeConverter, benefit), abi(abi) {}
  HALDispatchABI &abi;
};

/// Converts Standard MLIR FuncOps to LLVMFuncOps matching the IREE HAL ABI.
/// This is an IREE-specific conversion that assumes the input function is
/// `() -> ()` and that hal.interface.* ops are used to access all state.
///
/// Source function:
///
/// ```
/// func.func @foo() {
///   %0 = hal.interface.binding.subspan ...
/// }
/// ```
///
/// into:
///
/// ```
/// llvm.func foo(%state: !llvm.ptr<!...>,
///               %workgroup_id : !llvm.ptr<!llvm.array<i32, 3>>) {
///   %0 = <GEP/loads to access binding in %state>
/// }
/// ```
///
/// See `iree/hal/local/executable_library.h` for more information.
///
/// NOTE: we bump the benefit of the pattern to 100 to pick this pattern instead
/// of a competing pattern inserted by `populateFuncToLLVMConversionPatterns`.
struct ConvertHALEntryPointFuncOp
    : public ConvertOpToLLVMWithABIPattern<func::FuncOp> {
  ConvertHALEntryPointFuncOp(HALDispatchABI &abi,
                             LLVMTypeConverter &typeConverter)
      : ConvertOpToLLVMWithABIPattern(abi, typeConverter,
                                      /*benefit=*/100) {}
  LogicalResult
  matchAndRewrite(func::FuncOp stdFuncOp, func::FuncOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!stdFuncOp.isPublic())
      return failure();
    FunctionType fnType = stdFuncOp.getFunctionType();
    if (fnType.getNumInputs() != 0 || fnType.getNumResults() != 0) {
      stdFuncOp->emitWarning()
          << "public functions on executables must be () -> ()";
      return failure();
    }

    // Convert the function signature to take the HAL ABI LLVM pointers.
    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    MLIRContext *context = rewriter.getContext();
    auto abiInputTypes =
        HALDispatchABI::getInputTypes(context, getTypeConverter());
    signatureConverter.addInputs(abiInputTypes);

    // Copy all attributes onto the LLVM function except the ones handled by
    // MLIR implicitly.
    SmallVector<NamedAttribute> funcAttrs;
    for (auto attr : stdFuncOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == stdFuncOp.getFunctionTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    // Clone the function as an LLVMFuncOp and convert all interior types.
    auto int32Type = IntegerType::get(rewriter.getContext(), 32);
    auto llvmFuncType = LLVM::LLVMFunctionType::get(int32Type, abiInputTypes);
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        stdFuncOp.getLoc(), stdFuncOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dsoLocal=*/false, /*cconv=*/LLVM::CConv::C,
        /*comdat=*/nullptr, funcAttrs);
    rewriter.inlineRegionBefore(stdFuncOp.getFunctionBody(),
                                llvmFuncOp.getFunctionBody(), llvmFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&llvmFuncOp.getFunctionBody(),
                                           *typeConverter,
                                           &signatureConverter))) {
      return failure();
    }

    // Tag all arguments so LLVM can reason about our exports it otherwise
    // cannot analyze. We do this early on so that MLIR-based LLVM transforms
    // can use the attributes.
    // (%arg0: environment, %arg1: dispatch_state, %arg2: workgroup_state)
    for (unsigned i = 0; i <= 2; ++i) {
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getNoAliasAttrName(),
                            rewriter.getUnitAttr());
      llvmFuncOp.setArgAttr(i, LLVM::LLVMDialect::getAlignAttrName(),
                            rewriter.getI64IntegerAttr(16));
    }

    // Add default zero return value.
    // TODO(ataei): do something meaningful with the return value; non-zero will
    // have the runtime bail out with an error.
    for (auto returnOp : llvm::make_early_inc_range(
             llvmFuncOp.getOps<mlir::func::ReturnOp>())) {
      rewriter.setInsertionPoint(returnOp);
      auto returnValue = rewriter.createOrFold<mlir::arith::ConstantIntOp>(
          returnOp.getLoc(), 0, 32);
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp, returnValue);
    }

    // Populate debug info for the subprogram signature. This is required in
    // order to get any debug information (including just line tables) from MLIR
    // into LLVM IR.
    auto scopeAttr = HALDispatchABI::buildScopeAttr(
        llvmFuncOp->getParentOfType<mlir::ModuleOp>(), llvmFuncOp.getName(),
        getTypeConverter());
    llvmFuncOp->setLoc(FusedLoc::get(llvmFuncOp.getContext(),
                                     {llvmFuncOp->getLoc()}, scopeAttr));

    rewriter.eraseOp(stdFuncOp);
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
/// Because ordinals are not yet available we emit a placeholder global that
/// later gets updated with the value after linking.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALExecutableConstantLoadOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::ExecutableConstantLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::ExecutableConstantLoadOp loadOp,
                  IREE::HAL::ExecutableConstantLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType =
        typeConverter->convertType(loadOp->getResult(0).getType());
    rewriter.replaceOp(loadOp,
                       abi.loadExecutableConstant(loadOp, loadOp.getKey(),
                                                  resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.id to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALInterfaceWorkgroupIDOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupIDOp idOp,
                  IREE::HAL::InterfaceWorkgroupIDOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)idOp.getDimension().getZExtValue();
    auto resultType = typeConverter->convertType(idOp->getResult(0).getType());
    rewriter.replaceOp(idOp,
                       abi.loadWorkgroupID(idOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALInterfaceWorkgroupSizeOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceWorkgroupSizeOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupSizeOp sizeOp,
                  IREE::HAL::InterfaceWorkgroupSizeOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)sizeOp.getDimension().getZExtValue();
    auto resultType =
        typeConverter->convertType(sizeOp->getResult(0).getType());
    rewriter.replaceOp(
        sizeOp, abi.loadWorkgroupSize(sizeOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALInterfaceWorkgroupCountOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceWorkgroupCountOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceWorkgroupCountOp countOp,
                  IREE::HAL::InterfaceWorkgroupCountOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int32_t dim = (int32_t)countOp.getDimension().getZExtValue();
    auto resultType =
        typeConverter->convertType(countOp->getResult(0).getType());
    rewriter.replaceOp(
        countOp, abi.loadWorkgroupCount(countOp, dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALInterfaceConstantLoadOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceConstantLoadOp loadOp,
                  IREE::HAL::InterfaceConstantLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    int64_t index = loadOp.getIndex().getZExtValue();
    auto resultType =
        typeConverter->convertType(loadOp->getResult(0).getType());
    rewriter.replaceOp(
        loadOp, abi.loadPushConstant(loadOp, index, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
struct ConvertHALInterfaceBindingSubspanOp
    : public ConvertOpToLLVMWithABIPattern<
          IREE::HAL::InterfaceBindingSubspanOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InterfaceBindingSubspanOp subspanOp,
                  IREE::HAL::InterfaceBindingSubspanOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    MemRefType memRefType =
        llvm::dyn_cast<MemRefType>(subspanOp->getResult(0).getType());
    if (!memRefType) {
      return rewriter.notifyMatchFailure(
          subspanOp,
          "failed to convert interface.binding.subspan result to memref type");
    }
    auto memRefDesc = abi.loadBinding(
        subspanOp, operands.getBindingAttr().getInt(), operands.getByteOffset(),
        memRefType, operands.getDynamicDims(), rewriter);
    rewriter.replaceOp(subspanOp, {memRefDesc});
    return success();
  }
};

struct InstrumentationEntry {
  // !llvm.ptr<i8> pointing at the base of the ringbuffer.
  Value basePtr;
  // !llvm.ptr<i8> pointing at the start of the entry (basePtr + offset).
  Value entryPtr;
  // i64 offset within the ringbuffer of the entry.
  Value offset;
};

// entrySize must be 16-byte aligned
static InstrumentationEntry
acquireInstrumentationEntry(Location loc, Value buffer, Value bufferPtr,
                            Value entrySize, OpBuilder &builder) {
  auto i64Type = builder.getI64Type();
  auto bufferType = llvm::cast<MemRefType>(buffer.getType());
  int64_t totalBufferSize =
      (bufferType.getNumElements() * bufferType.getElementTypeBitWidth()) / 8;
  int64_t headOffset = totalBufferSize - 8;
  int64_t ringSize = totalBufferSize - IREE_INSTRUMENT_DISPATCH_PADDING;
  assert(llvm::isPowerOf2_64(ringSize) &&
         "ringbuffer storage size must be a power-of-two");

  Value basePtr = MemRefDescriptor(bufferPtr).alignedPtr(builder, loc);

  Value offsetIndex =
      builder.create<LLVM::ConstantOp>(loc, i64Type, headOffset);
  auto i8Type = builder.getI8Type();
  Value offsetPtr = builder.create<LLVM::GEPOp>(loc, basePtr.getType(), i8Type,
                                                basePtr, offsetIndex,
                                                /*inbounds=*/true);
  Value rawOffset = builder.create<LLVM::AtomicRMWOp>(
      loc, LLVM::AtomicBinOp::add, offsetPtr, entrySize,
      LLVM::AtomicOrdering::monotonic);
  Value offsetMask =
      builder.create<LLVM::ConstantOp>(loc, i64Type, ringSize - 1);
  Value wrappedOffset = builder.create<LLVM::AndOp>(loc, rawOffset, offsetMask);

  Value entryPtr = builder.create<LLVM::GEPOp>(loc, basePtr.getType(), i8Type,
                                               basePtr, wrappedOffset);

  return {basePtr, entryPtr, wrappedOffset};
}

static InstrumentationEntry appendInstrumentationEntry(
    Location loc, Value buffer, Value bufferPtr, LLVM::LLVMStructType entryType,
    ArrayRef<Value> entryValues, DataLayout &dataLayout, OpBuilder &builder) {
  auto i64Type = builder.getI64Type();

  Value entrySize = builder.create<LLVM::ConstantOp>(
      loc, i64Type, dataLayout.getTypeSize(entryType));
  auto entry =
      acquireInstrumentationEntry(loc, buffer, bufferPtr, entrySize, builder);

  Value entryStruct = builder.create<LLVM::UndefOp>(loc, entryType);
  for (auto entryValue : llvm::enumerate(entryValues)) {
    entryStruct = builder.create<LLVM::InsertValueOp>(
        loc, entryStruct, entryValue.value(), entryValue.index());
  }

  builder.create<LLVM::StoreOp>(
      loc, entryStruct,
      builder.create<LLVM::BitcastOp>(
          loc, LLVM::LLVMPointerType::get(builder.getContext()),
          entry.entryPtr),
      /*alignment=*/16);

  return entry;
}

static int64_t getMemoryAccessByteSize(Type type) {
  if (auto vectorType = llvm::dyn_cast<VectorType>(type)) {
    return (vectorType.getNumElements() * vectorType.getElementTypeBitWidth()) /
           8;
  } else {
    return type.getIntOrFloatBitWidth() / 8;
  }
}

struct ConvertHALInstrumentWorkgroupOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentWorkgroupOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentWorkgroupOp instrumentOp,
                  IREE::HAL::InstrumentWorkgroupOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i32Type = rewriter.getI32Type();
    auto i64Type = rewriter.getI64Type();

    auto entryType = LLVM::LLVMStructType::getLiteral(
        getContext(), {
                          i32Type, // header
                          i32Type, // workgroup_id_x
                          i32Type, // workgroup_id_y
                          i32Type, // workgroup_id_z
                          i32Type, // workgroup_count_x
                          i32Type, // workgroup_count_y
                          i32Type, // workgroup_count_z
                          i32Type, // processor_id
                      });

    // 8 bit tag = 00 | 24 bit dispatch id
    // NOTE: we could pre-shift this to avoid needing to do it in each group.
    // We just need to do the shift - the bottom two bits will be the 00 tag.
    Value rawDispatchId = instrumentOp.getDispatchId();
    Value header = rewriter.create<LLVM::ShlOp>(
        loc, i32Type, rawDispatchId,
        rewriter.create<LLVM::ConstantOp>(loc, i32Type, 8)); // | 8bit tag

    auto entry = appendInstrumentationEntry(
        loc, instrumentOp.getBuffer(), operands.getBuffer(), entryType,
        {
            header,
            abi.loadWorkgroupID(instrumentOp, 0, i32Type, rewriter),
            abi.loadWorkgroupID(instrumentOp, 1, i32Type, rewriter),
            abi.loadWorkgroupID(instrumentOp, 2, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 0, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 1, i32Type, rewriter),
            abi.loadWorkgroupCount(instrumentOp, 2, i32Type, rewriter),
            abi.loadProcessorID(instrumentOp, rewriter),
        },
        dataLayout, rewriter);

    // Prepare the 40-bit key used by all accesses - we do this once so that we
    // can ensure it's hoisted.
    // Consumers expect 40 bits of offset << 24 bits.
    Value workgroupKey = rewriter.create<LLVM::ShlOp>(
        loc,
        rewriter.create<LLVM::AndOp>(
            loc, entry.offset,
            rewriter.create<LLVM::ConstantOp>(loc, i64Type, 0xFFFFFFFFFFll)),
        rewriter.create<LLVM::ConstantOp>(loc, i64Type, 24));

    rewriter.replaceOp(instrumentOp, workgroupKey);
    return success();
  }
};

static std::optional<uint64_t> mapValueType(Type type) {
  return TypeSwitch<Type, std::optional<uint64_t>>(type)
      .Case<IntegerType>([&](Type type) -> std::optional<uint64_t> {
        if (type.isUnsignedInteger()) {
          switch (type.getIntOrFloatBitWidth()) {
          case 8:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_8;
          case 16:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_16;
          case 32:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_32;
          case 64:
            return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_UINT_64;
          default:
            return std::nullopt;
          }
        }
        switch (type.getIntOrFloatBitWidth()) {
        case 8:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_8;
        case 16:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_16;
        case 32:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_32;
        case 64:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64;
        default:
          return std::nullopt;
        }
      })
      .Case<FloatType>([&](Type type) -> std::optional<uint64_t> {
        if (type.isBF16()) {
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_BFLOAT_16;
        }
        switch (type.getIntOrFloatBitWidth()) {
        case 16:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_16;
        case 32:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_32;
        case 64:
          return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_FLOAT_64;
        default:
          return std::nullopt;
        }
      })
      .Case<IndexType>([&](Type type) -> std::optional<uint64_t> {
        return IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_SINT_64;
      })
      .Default([&](Type) -> std::optional<uint64_t> { return std::nullopt; });
}

struct ConvertHALInstrumentValueOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentValueOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentValueOp instrumentOp,
                  IREE::HAL::InstrumentValueOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();

    // Only convert ops we can handle, otherwise warn and discard.
    std::optional<uint64_t> valueType;
    if (llvm::isa<LLVM::LLVMPointerType>(operands.getOperand().getType())) {
      valueType = IREE_INSTRUMENT_DISPATCH_VALUE_TYPE_POINTER;
    } else {
      valueType = mapValueType(instrumentOp.getType());
    }
    if (!valueType) {
      mlir::emitWarning(loc,
                        "skipping hal.instrument.value on unsupported type: ")
          << instrumentOp.getType();
      rewriter.replaceOp(instrumentOp, {operands.getOperand()});
      return success();
    }

    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // value
                                                       });

    // 8 bit tag
    // 8 bit type
    // 8 bit ordinal
    // 40 bit workgroup offset
    Value header = rewriter.create<LLVM::OrOp>(
        loc, operands.getWorkgroupKey(),
        rewriter.create<LLVM::ConstantOp>(
            loc, i64Type,
            (instrumentOp.getOrdinal().getZExtValue() << 16) |
                (valueType.value() << 8) |
                IREE_INSTRUMENT_DISPATCH_TYPE_VALUE));

    // Bitcast to an integer and widen to 64 bits.
    Value bits = rewriter.create<LLVM::ZExtOp>(
        loc, i64Type,
        rewriter.create<LLVM::BitcastOp>(
            loc,
            rewriter.getIntegerType(
                instrumentOp.getType().getIntOrFloatBitWidth()),
            operands.getOperand()));

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   bits,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getOperand());
    return success();
  }
};

struct ConvertHALInstrumentMemoryLoadOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentMemoryLoadOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentMemoryLoadOp instrumentOp,
                  IREE::HAL::InstrumentMemoryLoadOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // address
                                                       });

    // 8 bit tag = 100 (read), 101 (write)
    // 16 bit length
    // 40 bit workgroup offset
    int64_t loadSize = getMemoryAccessByteSize(instrumentOp.getType());
    assert(loadSize <= UINT16_MAX && "16-bit length maximum");
    Value header = rewriter.create<LLVM::OrOp>(
        loc, operands.getWorkgroupKey(),
        rewriter.create<LLVM::ConstantOp>(
            loc, i64Type,
            (loadSize << 8) | IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_LOAD));

    Value loadPtr = getStridedElementPtr(
        loc, llvm::cast<MemRefType>(instrumentOp.getBase().getType()),
        operands.getBase(), operands.getIndices(), rewriter);
    Value addressI64 = rewriter.create<LLVM::PtrToIntOp>(loc, i64Type, loadPtr);

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   addressI64,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getLoadValue());
    return success();
  }
};

struct ConvertHALInstrumentMemoryStoreOp
    : public ConvertOpToLLVMWithABIPattern<IREE::HAL::InstrumentMemoryStoreOp> {
  using ConvertOpToLLVMWithABIPattern::ConvertOpToLLVMWithABIPattern;
  LogicalResult
  matchAndRewrite(IREE::HAL::InstrumentMemoryStoreOp instrumentOp,
                  IREE::HAL::InstrumentMemoryStoreOpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = instrumentOp.getLoc();
    auto dataLayout =
        getTypeConverter()->getDataLayoutAnalysis()->getAbove(instrumentOp);
    auto i64Type = rewriter.getI64Type();

    auto entryType =
        LLVM::LLVMStructType::getLiteral(getContext(), {
                                                           i64Type, // header
                                                           i64Type, // address
                                                       });

    // 8 bit tag = 10 (read), 11 (write)
    // 16 bit length
    // 40 bit workgroup offset
    int64_t storeSize = getMemoryAccessByteSize(instrumentOp.getType());
    assert(storeSize <= UINT16_MAX && "16-bit length maximum");
    Value header = rewriter.create<LLVM::OrOp>(
        loc, operands.getWorkgroupKey(),
        rewriter.create<LLVM::ConstantOp>(
            loc, i64Type,
            (storeSize << 8) | IREE_INSTRUMENT_DISPATCH_TYPE_MEMORY_STORE));

    Value storePtr = getStridedElementPtr(
        loc, llvm::cast<MemRefType>(instrumentOp.getBase().getType()),
        operands.getBase(), operands.getIndices(), rewriter);
    Value addressI64 =
        rewriter.create<LLVM::PtrToIntOp>(loc, i64Type, storePtr);

    appendInstrumentationEntry(loc, instrumentOp.getBuffer(),
                               operands.getBuffer(), entryType,
                               {
                                   header,
                                   addressI64,
                               },
                               dataLayout, rewriter);

    rewriter.replaceOp(instrumentOp, operands.getStoreValue());
    return success();
  }
};

/// Helper method to get information about extra operands that need to be
/// appended to a function defn/call operation.
static SmallVector<StringRef> getExtraFields(Operation *forOp) {
  SmallVector<StringRef> extraFields;
  if (auto extraFieldsAttr =
          forOp->getAttrOfType<ArrayAttr>("hal.import.fields")) {
    extraFields =
        llvm::map_to_vector(extraFieldsAttr.getValue(), [](Attribute attr) {
          return llvm::cast<StringAttr>(attr).getValue();
        });
  }
  return extraFields;
}

/// Return calling convention to use for the operation.
static IREE::HAL::CallingConvention getCallingConvention(Operation *forOp) {
  auto cConv = IREE::HAL::CallingConvention::Default;
  if (auto cConvAttr = forOp->getAttrOfType<IREE::HAL::CallingConventionAttr>(
          "hal.import.cconv")) {
    cConv = cConvAttr.getValue();
  }
  return cConv;
}

/// Lower func ops with specified ABI. Currently this pattern is triggered
/// only for operations with the `hal.import.bitcode` attribute set.
///
/// Note: this is an LLVM::CallOp -> LLVM::CallOp rewrite that is introduced
/// after all conversions are done. Importantly, this is not a conversion
/// pattern.
struct RewriteFuncOpABI : public OpRewritePattern<LLVM::LLVMFuncOp> {
  RewriteFuncOpABI(HALDispatchABI &abi, LLVMTypeConverter &typeConverter)
      : OpRewritePattern(&typeConverter.getContext()), abi(abi),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp funcOp,
                                PatternRewriter &rewriter) const override {
    if (!funcOp.isExternal()) {
      return rewriter.notifyMatchFailure(funcOp, "skipping non-external calls");
    }
    if (!funcOp->hasAttr("hal.import.bitcode")) {
      return rewriter.notifyMatchFailure(
          funcOp, "callee is not imported using bitcode linkage; skipping");
    }
    IREE::HAL::CallingConvention cConv = getCallingConvention(funcOp);

    SmallVector<StringRef> extraFields = getExtraFields(funcOp);
    auto funcType = funcOp.getFunctionType();
    FailureOr<LLVM::LLVMFunctionType> expectedType =
        abi.getABIFunctionType(funcOp, cConv, funcType.getReturnTypes(),
                               funcType.getParams(), extraFields);
    if (failed(expectedType)) {
      return rewriter.notifyMatchFailure(
          funcOp,
          "unable to get function type to match the calling convention");
    }
    if (abi.hasCompatibleFunctionSignature(
            rewriter.getContext(), expectedType.value(),
            funcType.getReturnTypes(), funcType.getParams())) {
      return failure();
    }
    auto attrs = getPrunedAttributeList(
        funcOp, llvm::to_vector(LLVM::LLVMFuncOp::getAttributeNames()));
    SmallVector<DictionaryAttr> argAttrs;
    if (auto currArgAttrs = funcOp.getArgAttrsAttr()) {
      argAttrs = llvm::map_to_vector(currArgAttrs, [](Attribute attr) {
        return llvm::cast<DictionaryAttr>(attr);
      });
    }
    rewriter.create<LLVM::LLVMFuncOp>(
        funcOp.getLoc(), funcOp.getName(), expectedType.value(),
        funcOp.getLinkage(), funcOp.getDsoLocal(), funcOp.getCConv(),
        /*comdat=*/nullptr, attrs, argAttrs, funcOp.getFunctionEntryCount());
    rewriter.eraseOp(funcOp);
    return success();
  }

private:
  HALDispatchABI &abi;
  LLVMTypeConverter &typeConverter;
};

/// Lower call ops with specified ABI. The ABI to use is looked up from the
/// callee. Currently this pattern is triggered only for operations where the
/// callee has the `hal.import.bitcode` attribute set.
///
/// Note: this is an LLVM::CallOp -> LLVM::CallOp rewrite that is introduced
/// after all conversions are done. Importantly, this is not a conversion
/// pattern.
struct RewriteCallOpABI : public OpRewritePattern<LLVM::CallOp> {
  RewriteCallOpABI(HALDispatchABI &abi, LLVMTypeConverter &typeConverter)
      : OpRewritePattern(&typeConverter.getContext()), abi(abi),
        typeConverter(typeConverter) {}

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto symbol = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    auto flatSymbol = llvm::dyn_cast_if_present<FlatSymbolRefAttr>(symbol);
    if (!flatSymbol)
      return failure();

    // Ensure the target function is extern.
    // To support conversion inserting calls in local patterns that can't add
    // global function symbols we assume any missing callee is extern.
    auto calleeOp =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(callOp, symbol);
    if (!calleeOp || !calleeOp->hasAttr("hal.import.bitcode") ||
        !calleeOp.isExternal()) {
      return rewriter.notifyMatchFailure(
          callOp, "callee is not imported using bitcode linakge; skipping");
    }

    IREE::HAL::CallingConvention cConv = getCallingConvention(calleeOp);
    SmallVector<StringRef> extraFields = getExtraFields(calleeOp);

    FailureOr<SmallVector<Value>> results = abi.materializeABI(
        callOp, calleeOp.getSymName(), cConv, callOp->getResultTypes(),
        callOp->getOperands(), extraFields, rewriter);
    if (failed(results)) {
      return failure();
    }
    rewriter.replaceOp(callOp, *results);
    return success();
  }

private:
  HALDispatchABI &abi;
  LLVMTypeConverter &typeConverter;
};

/// Rewrites calls to extern functions to dynamic library import calls.
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
///
/// Note: this is an LLVM::CallOp -> LLVM::CallOp rewrite that is introduced
/// after all conversions are done. Importantly, this is not a conversion
/// pattern.
struct RewriteExternCallOpToDynamicImportCallOp
    : public OpRewritePattern<LLVM::CallOp> {
  RewriteExternCallOpToDynamicImportCallOp(HALDispatchABI &abi,
                                           LLVMTypeConverter &typeConverter)
      : OpRewritePattern(&typeConverter.getContext()), abi(abi),
        typeConverter(typeConverter) {}
  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    // Ignore indirect calls (they're probably already converted imports).
    auto symbol = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    auto flatSymbol = llvm::dyn_cast_if_present<FlatSymbolRefAttr>(symbol);
    if (!flatSymbol)
      return failure();

    // Ensure the target function is extern.
    // To support conversion inserting calls in local patterns that can't add
    // global function symbols we assume any missing callee is extern.
    auto calleeOp =
        SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(callOp, symbol);
    if (calleeOp && !calleeOp.isExternal()) {
      return rewriter.notifyMatchFailure(
          callOp,
          "callee is not external; treating as a normal call and skipping "
          "import logic");
    }

    // If the function is marked as statically linked we don't touch it. That'll
    // let it fall through to the linker stage where it can be picked up either
    // from the runtime build (in the case of us producing static libraries) or
    // the user-specified object files (when producing dynamic libraries).
    if (calleeOp->hasAttr("hal.import.static") ||
        calleeOp->hasAttr("hal.import.bitcode")) {
      return rewriter.notifyMatchFailure(callOp,
                                         "external function is marked static "
                                         "and does not need an import wrapper");
    }

    // The call may need some additional internal fields appended.
    SmallVector<StringRef> extraFields;
    if (auto extraFieldsAttr =
            calleeOp->getAttrOfType<ArrayAttr>("hal.import.fields")) {
      for (auto extraFieldAttr : extraFieldsAttr) {
        extraFields.push_back(
            llvm::cast<StringAttr>(extraFieldAttr).getValue());
      }
    }

    // Allow multiple imports to alias by having their name explicitly
    // specified.
    StringRef importName = flatSymbol.getValue();
    if (auto importNameAttr =
            calleeOp->getAttrOfType<StringAttr>("hal.import.name")) {
      importName = importNameAttr.getValue();
    }

    // TODO(benvanik): way to determine if weak (maybe via linkage?).
    bool weak = false;

    // Rewrite the call to a dynamic import call.
    SmallVector<Value> results = abi.wrapAndCallImport(
        callOp, importName, weak, callOp->getResultTypes(),
        callOp->getOperands(), extraFields, rewriter);

    rewriter.replaceOp(callOp, results);
    return success();
  }
  HALDispatchABI &abi;
  LLVMTypeConverter &typeConverter;
};

/// The 32-bit RISC-V backend is very sensitive to how extended multiplication
/// is lowered. This pattern lowers `arith.mulsi_extended` before going to the
/// LLVM dialect, in a way compatible with that backend, so that we break down
/// any 64-bit constants that would otherwise prevent the code from being
/// vectorized.
class ExpandMulSIExtended : public OpRewritePattern<arith::MulSIExtendedOp> {
public:
  using OpRewritePattern<arith::MulSIExtendedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulSIExtendedOp op,
                                PatternRewriter &rewriter) const override {
    Type resultType = op.getLhs().getType();
    if (getElementTypeOrSelf(resultType).getIntOrFloatBitWidth() != 32) {
      return failure();
    }

    Location loc = op.getLoc();

    Type wideType = rewriter.getIntegerType(64);
    // Shift amount necessary to extract the high bits from widened result.
    TypedAttr shiftValAttr = rewriter.getI64IntegerAttr(32);
    if (auto vecTy = llvm::dyn_cast<VectorType>(resultType)) {
      wideType = VectorType::get(vecTy.getShape(), wideType);
      shiftValAttr =
          SplatElementsAttr::get(cast<ShapedType>(wideType), shiftValAttr);
    }
    Value shiftVal = rewriter.create<arith::ConstantOp>(loc, shiftValAttr);

    Value lhsExt = rewriter.create<arith::ExtSIOp>(loc, wideType, op.getLhs());
    Value rhsExt = rewriter.create<arith::ExtSIOp>(loc, wideType, op.getRhs());
    Value mulExt =
        rewriter.create<arith::MulIOp>(loc, wideType, lhsExt, rhsExt);
    Value low = rewriter.create<arith::MulIOp>(loc, resultType, op.getLhs(),
                                               op.getRhs());

    // Produce two 32-bit results.
    Value highExt = rewriter.create<arith::ShRUIOp>(loc, mulExt, shiftVal);
    Value high = rewriter.create<arith::TruncIOp>(loc, resultType, highExt);

    rewriter.replaceOp(op, {low, high});
    return success();
  }
};

/// Adds a link dependency on the ArmSME ABI routines if the LLVMFuncOp has
/// either the `armNewZA` or `armLocallyStreaming` attribute set.
struct LinkArmSMERoutinesIfNeeded : public OpRewritePattern<LLVM::LLVMFuncOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp llvmFuncOp,
                                PatternRewriter &rewriter) const override {
    if (!llvmFuncOp.getArmNewZa() && !llvmFuncOp.getArmLocallyStreaming())
      return failure();

    auto variantOp =
        llvmFuncOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();

    if (variantOp.getObjectsAttr())
      return failure();

    static auto sme_compiler_rt_lib =
        std::getenv("IREE_ARM_SME_COMPILER_RT_BUILTINS_STATIC_LIB");

    if (!sme_compiler_rt_lib)
      return llvmFuncOp.emitError(
          "IREE_ARM_SME_COMPILER_RT_BUILTINS_STATIC_LIB must be set!");

    Attribute objectAttr = rewriter.getAttr<IREE::HAL::ExecutableObjectAttr>(
        rewriter.getStringAttr(sme_compiler_rt_lib), DenseIntElementsAttr{});
    variantOp.setObjectsAttr(rewriter.getArrayAttr(objectAttr));

    return success();
  }
};

class ConvertToLLVMPass : public ConvertToLLVMBase<ConvertToLLVMPass> {
public:
  ConvertToLLVMPass(bool reassociateFpReductions) {
    targetReassociateFpReductions.setValue(reassociateFpReductions);
  }
  ConvertToLLVMPass(const ConvertToLLVMPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, arm_neon::ArmNeonDialect>();
  }

  void runOnOperation() override;

private:
  Option<std::string> targetTriple{
      *this, "target-triple", llvm::cl::desc("Code generation target triple."),
      llvm::cl::init("")};
  Option<std::string> targetDataLayout{
      *this, "target-data-layout",
      llvm::cl::desc("Code generation target data layout."),
      llvm::cl::init("")};
  Option<bool> targetReassociateFpReductions{
      *this, "target-reassociate-fp-reductions",
      llvm::cl::desc("Code generation target reassociate FP reductions."),
      llvm::cl::init("false")};
};

} // namespace

static std::string getStringAttrFromTargetAttr(ModuleOp module,
                                               StringRef attrName) {
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(module);
  auto stringAttr = getConfigStringAttr(targetAttr, attrName);
  return stringAttr ? stringAttr.value().str() : std::string("");
}

void ConvertToLLVMPass::runOnOperation() {
  auto module = getOperation();
  std::string dataLayoutStr = targetDataLayout.getValue();
  if (targetDataLayout.empty()) {
    dataLayoutStr = getStringAttrFromTargetAttr(module, "data_layout");
  }
  std::string targetTripleStr = targetTriple.getValue();
  if (targetTripleStr.empty()) {
    targetTripleStr = getStringAttrFromTargetAttr(module, "target_triple");
  }

  // Add required attributes to the module so that the lowering knows how to
  // handle structs and data layouts.
  module->setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
                  StringAttr::get(module->getContext(), targetTripleStr));
  module->setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
                  StringAttr::get(module->getContext(), dataLayoutStr));

  // Run Vector -> Vector transformations ahead of conversion to LLVM.
  {
    RewritePatternSet patterns(&getContext());
    vector::populateVectorToVectorCanonicalizationPatterns(patterns);
    vector::populateVectorBroadcastLoweringPatterns(patterns);
    // TODO: doubtful that the "default" does what one want here, it is likely
    // better to use outerproduct.
    vector::populateVectorContractLoweringPatterns(
        patterns, vector::VectorTransformsOptions());
    vector::populateVectorMaskMaterializationPatterns(
        patterns, /*force32BitVectorIndices=*/false);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    // TODO: doubtful that the "default" does what one want here, it is likely
    // better to use shuffle.
    vector::populateVectorTransposeLoweringPatterns(
        patterns, vector::VectorTransformsOptions());
    populateConvertArmNeon2dToIntrPatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
  {
    RewritePatternSet vectorToLoopsPatterns(&getContext());
    populateVectorToSCFConversionPatterns(
        vectorToLoopsPatterns, VectorTransferToSCFOptions().enableFullUnroll());
    if (failed(applyPatternsAndFoldGreedily(
            getOperation(), std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }

  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(&getContext(),
                             dataLayoutAnalysis.getAtOrAbove(module));
  options.dataLayout = llvm::DataLayout(dataLayoutStr);
  options.overrideIndexBitwidth(options.dataLayout.getPointerSizeInBits());
  LLVMTypeConverter typeConverter(&getContext(), options, &dataLayoutAnalysis);

  RewritePatternSet patterns(&getContext());

  // Use the default 64-bit lowering for TOSA's ApplyScale operator:
  //   This lowering widens integer types to 64-bit an performs the non-fused
  //   operations, specifically multiply, add, and shift. Bit-widening
  //   is used to guarantee higher-order bits are not truncated during the
  //   multiply or add.
  //
  // TODO(bjacob): Use a lowering that uses specific ARM/X86 intrinsics.
  bool use32BitImpl = false;
  auto targetAttr = IREE::HAL::ExecutableTargetAttr::lookup(module);
  if (isRISCV(targetAttr)) {
    // Use the 32-bit lowering for RISC-V if 'zve32*' is specified and there is
    // no 64-bit integer vector support.
    // TODO(#9440) Simplify logic when 'cpu_features' is simplified.
    use32BitImpl =
        (hasZve32xFeature(targetAttr) || hasZve32fFeature(targetAttr)) &&
        !hasVFeature(targetAttr) && !hasZve64xFeature(targetAttr);
  }
  tosa::populateTosaRescaleToArithConversionPatterns(&patterns, use32BitImpl);

  // Make sure we expand any `arith.mulsi_extended` before going to the LLVM
  // dialect.
  if (use32BitImpl) {
    patterns.add<ExpandMulSIExtended>(patterns.getContext(), /*benefit=*/1024);
  }

  LLVMConversionTarget target(getContext());
  bool hasAArch64SME = isAArch64(targetAttr) && hasSMEFeature(targetAttr);
  if (hasAArch64SME) {
    // Enable ArmSME to LLVM lowerings.
    configureArmSMEToLLVMConversionLegality(target);
    populateArmSMEToLLVMConversionPatterns(typeConverter, patterns);
  }

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateExpandTanhPattern(patterns);

  populateComplexToLLVMConversionPatterns(typeConverter, patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  memref::populateExpandStridedMetadataPatterns(patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateExpandBFloat16Patterns(patterns);
  populateVectorToSCFConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(
      typeConverter, patterns, targetReassociateFpReductions.getValue());
  populateReconcileUnrealizedCastsPatterns(patterns);

  HALDispatchABI abi(&typeConverter);
  // clang-format off
  patterns.insert<
    ConvertHALEntryPointFuncOp,
    ConvertHALExecutableConstantLoadOp,
    ConvertHALInterfaceWorkgroupIDOp,
    ConvertHALInterfaceWorkgroupSizeOp,
    ConvertHALInterfaceWorkgroupCountOp,
    ConvertHALInterfaceConstantLoadOp,
    ConvertHALInterfaceBindingSubspanOp,
    ConvertHALInstrumentWorkgroupOp,
    ConvertHALInstrumentValueOp,
    ConvertHALInstrumentMemoryLoadOp,
    ConvertHALInstrumentMemoryStoreOp
  >(abi, typeConverter);
  // clang-format on

  target.addLegalOp<ModuleOp>();
  target.addIllegalDialect<func::FuncDialect, mlir::arith::ArithDialect,
                           IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                           math::MathDialect, tosa::TosaDialect>();
  target.addIllegalOp<UnrealizedConversionCastOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
  }

  // Rewrite any extern calls emitted to dynamic library imports.
  {
    RewritePatternSet patterns(&getContext());
    patterns.insert<RewriteExternCallOpToDynamicImportCallOp, RewriteCallOpABI,
                    RewriteFuncOpABI>(abi, typeConverter);
    if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }

  // Post conversion patterns.
  {
    RewritePatternSet postPatterns(&getContext());
    // TODO(ravishankarm): Move this to a separate pass.
    llvm::Triple triple(targetTripleStr);
    if (triple.isWasm()) {
      populateUnfusedFMAOpsPassPatterns(&getContext(), postPatterns);
    }
    if (hasAArch64SME) {
      // TODO(macdue): Find a better place for this.
      postPatterns.insert<LinkArmSMERoutinesIfNeeded>(&getContext());
    }
    if (failed(applyPatternsAndFoldGreedily(module, std::move(postPatterns)))) {
      return signalPassFailure();
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertToLLVMPass(bool reassociateFpReductions) {
  return std::make_unique<ConvertToLLVMPass>(reassociateFpReductions);
}

} // namespace mlir::iree_compiler
