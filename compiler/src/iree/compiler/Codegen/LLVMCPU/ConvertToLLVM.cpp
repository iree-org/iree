// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

// NOTE: HALDispatchABI and the associated conversion patterns should live under
// iree/compiler/Dialect/HAL/Target/LLVM/ instead of here as they have nothing
// to do with linalg. If we need to use the patterns in this conversion we can
// expose a populate*Patterns() function to access them without needing them
// defined here.

// Utility for accessing the IREE HAL dispatch function ABI values.
// All accesses should route through this vs directly manipulating LLVM function
// arguments so that we can adjust the ABI over time and support multiple
// versions in the same compiled output.
class HALDispatchABI {
 public:
  // Matches the field order in iree_hal_processor_v0_t.
  enum class ProcessorField {
    data = 0,
  };

  // Matches IREE_HAL_PROCESSOR_DATA_CAPACITY_V0.
  static constexpr int ProcessorDataCapacity = 8;

  // Returns a Type representing iree_hal_processor_v0_t.
  static LLVM::LLVMStructType getProcessorType(
      MLIRContext *context, LLVMTypeConverter *typeConverter) {
    llvm::sys::ScopedLock lock(sMutex);
    auto structType =
        LLVM::LLVMStructType::getIdentified(context, "iree_hal_processor_v0_t");
    if (structType.isInitialized()) return structType;

    auto uint64Type = IntegerType::get(context, 64);
    SmallVector<Type> fieldTypes;

    // uint64_t data[IREE_HAL_PROCESSOR_DATA_CAPACITY_V0];
    fieldTypes.push_back(
        LLVM::LLVMArrayType::get(uint64Type, ProcessorDataCapacity));

    LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;

    return structType;
  }

  // Matches the field order in iree_hal_executable_environment_v0_t.
  enum class EnvironmentField {
    constants,
    import_thunk,
    import_funcs,
    import_contexts,
    processor,
  };

  // Returns a Type representing iree_hal_executable_environment_v0_t.
  static LLVM::LLVMStructType getEnvironmentType(
      MLIRContext *context, LLVMTypeConverter *typeConverter,
      LLVM::LLVMStructType processorType) {
    llvm::sys::ScopedLock lock(sMutex);
    auto structType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_environment_v0_t");
    if (structType.isInitialized()) return structType;

    auto int8Type = IntegerType::get(context, 8);
    auto uint32Type = IntegerType::get(context, 32);
    auto int8PtrType = LLVM::LLVMPointerType::get(int8Type);
    auto uint32PtrType = LLVM::LLVMPointerType::get(uint32Type);
    SmallVector<Type, 4> fieldTypes;

    // const uint32_t* constants;
    fieldTypes.push_back(uint32PtrType);

    // iree_hal_executable_import_thunk_v0_t import_thunk;
    // const iree_hal_executable_import_v0_t* import_funcs;
    // const void** import_contexts;
    auto importType = LLVM::LLVMFunctionType::get(
        uint32Type, {int8PtrType, int8PtrType, int8PtrType});
    auto importPtrType = LLVM::LLVMPointerType::get(importType);
    auto importThunkType = LLVM::LLVMFunctionType::get(
        uint32Type, {importPtrType, int8PtrType, int8PtrType, int8PtrType});
    fieldTypes.push_back(LLVM::LLVMPointerType::get(importThunkType));
    fieldTypes.push_back(LLVM::LLVMPointerType::get(importPtrType));
    fieldTypes.push_back(LLVM::LLVMPointerType::get(int8PtrType));

    // iree_hal_processor_v0_t processor;
    fieldTypes.push_back(processorType);

    LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;

    return structType;
  }

  // Matches the field order in iree_hal_executable_dispatch_state_v0_t.
  enum class DispatchStateField {
    /*uint32_t*/ workgroup_size_x,
    /*uint32_t*/ workgroup_size_y,
    /*uint16_t*/ workgroup_size_z,
    /*uint16_t*/ push_constant_count,
    /*uint32_t*/ workgroup_count_x,
    /*uint32_t*/ workgroup_count_y,
    /*uint16_t*/ workgroup_count_z,
    /*uint8_t*/ max_concurrency,
    /*uint8_t*/ binding_count,
    /*intptr_t*/ push_constants,
    /*intptr_t*/ binding_ptrs,
    /*intptr_t*/ binding_lengths,
  };
  friend DispatchStateField operator+(DispatchStateField lhs, int32_t rhs) {
    return static_cast<DispatchStateField>(static_cast<int32_t>(lhs) + rhs);
  }

  // Returns a Type representing iree_hal_executable_dispatch_state_v0_t.
  static LLVM::LLVMStructType getDispatchStateType(
      MLIRContext *context, LLVMTypeConverter *typeConverter) {
    llvm::sys::ScopedLock lock(sMutex);
    auto structType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_dispatch_state_v0_t");
    if (structType.isInitialized()) return structType;

    auto indexType = typeConverter->convertType(IndexType::get(context));
    auto int8Type = IntegerType::get(context, 8);
    auto uint8Type = IntegerType::get(context, 8);
    auto uint16Type = IntegerType::get(context, 16);
    auto uint32Type = IntegerType::get(context, 32);
    auto int8PtrType = LLVM::LLVMPointerType::get(int8Type);
    auto uint32PtrType = LLVM::LLVMPointerType::get(uint32Type);
    SmallVector<Type, 4> fieldTypes;

    // uint32_t workgroup_size_x;
    // uint32_t workgroup_size_y;
    // uint16_t workgroup_size_z;
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint16Type);

    // uint16_t push_constant_count;
    fieldTypes.push_back(uint16Type);

    // uint32_t workgroup_count_x;
    // uint32_t workgroup_count_y;
    // uint16_t workgroup_count_z;
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint16Type);

    // uint8_t max_concurrency;
    fieldTypes.push_back(uint8Type);

    // uint8_t binding_count;
    fieldTypes.push_back(uint8Type);

    // const uint32_t * push_constants;
    fieldTypes.push_back(uint32PtrType);
    // void *const * binding_ptrs;
    // const size_t * binding_lengths;
    fieldTypes.push_back(LLVM::LLVMPointerType::get(int8PtrType));
    fieldTypes.push_back(LLVM::LLVMPointerType::get(indexType));

    LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;

    return structType;
  }

  enum class WorkgroupStateField {
    /*uint32_t*/ workgroup_id_x = 0,
    /*uint32_t*/ workgroup_id_y,
    /*uint16_t*/ workgroup_id_z,
    /*uint16_t*/ reserved,
    /*uint32_t*/ processor_id,
    /*intptr_t*/ local_memory,
    /*uint32_t*/ local_memory_size,
  };
  friend WorkgroupStateField operator+(WorkgroupStateField lhs, int32_t rhs) {
    return static_cast<WorkgroupStateField>(static_cast<int32_t>(lhs) + rhs);
  }

  // Returns a Type representing iree_hal_executable_workgroup_state_v0_t.
  static LLVM::LLVMStructType getWorkgroupStateType(
      MLIRContext *context, LLVMTypeConverter *typeConverter) {
    llvm::sys::ScopedLock lock(sMutex);
    auto structType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_workgroup_state_v0_t");
    if (structType.isInitialized()) return structType;

    auto int8Type = IntegerType::get(context, 8);
    auto uint16Type = IntegerType::get(context, 16);
    auto uint32Type = IntegerType::get(context, 32);
    auto int8PtrType = LLVM::LLVMPointerType::get(int8Type);
    SmallVector<Type, 4> fieldTypes;

    // uint32_t workgroup_id_x;
    // uint32_t workgroup_id_y;
    // uint16_t workgroup_id_z;
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint32Type);
    fieldTypes.push_back(uint16Type);

    // uint16_t reserved;
    fieldTypes.push_back(uint16Type);

    // uint32_t processor_id;
    fieldTypes.push_back(uint32Type);

    // void* local_memory;
    // uint32_t local_memory_size;
    fieldTypes.push_back(LLVM::LLVMPointerType::get(int8PtrType));
    fieldTypes.push_back(uint32Type);

    LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
    assert(succeeded(bodySet) &&
           "could not set the body of an identified struct");
    (void)bodySet;

    return structType;
  }

  // Returns the types of the LLVM function inputs for the ABI.
  // This matches the signature of `iree_hal_executable_dispatch_v0_t` in
  // `iree/hal/local/executable_library.h`.
  static SmallVector<Type, 5> getInputTypes(MLIRContext *context,
                                            LLVMTypeConverter *typeConverter) {
    auto environmentType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_environment_v0_t");
    assert(environmentType &&
           "environment type must be defined by ConvertToLLVM");
    auto dispatchStateType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_dispatch_state_v0_t");
    assert(dispatchStateType &&
           "dispatch state type must be defined by ConvertToLLVM");
    auto workgroupStateType = LLVM::LLVMStructType::getIdentified(
        context, "iree_hal_executable_workgroup_state_v0_t");
    assert(workgroupStateType &&
           "workgroup state type must be defined by ConvertToLLVM");
    return SmallVector<Type, 5>{
        // const iree_hal_executable_environment_v0_t* IREE_RESTRICT
        //   environment
        LLVM::LLVMPointerType::get(environmentType),
        // const iree_hal_executable_dispatch_state_v0_t* IREE_RESTRICT
        //   dispatch_state
        LLVM::LLVMPointerType::get(dispatchStateType),
        // const iree_hal_executable_workgroup_state_v0_t* IREE_RESTRICT
        //   workgroup_state
        LLVM::LLVMPointerType::get(workgroupStateType),
    };
  }

  explicit HALDispatchABI(LLVM::LLVMFuncOp &funcOp,
                          LLVMTypeConverter *typeConverter)
      : funcOp(funcOp),
        typeConverter(typeConverter),
        processorType(getProcessorType(funcOp.getContext(), typeConverter)),
        environmentType(getEnvironmentType(funcOp.getContext(), typeConverter,
                                           processorType)),
        dispatchStateType(
            getDispatchStateType(funcOp.getContext(), typeConverter)),
        workgroupStateType(
            getWorkgroupStateType(funcOp.getContext(), typeConverter)) {}

  LLVM::LLVMFuncOp getFuncOp() { return funcOp; }

  // Loads the workgroup_id[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupID(Location loc, int32_t dim, Type resultType,
                        OpBuilder &builder) {
    auto dimValue =
        loadFieldValue(loc, WorkgroupStateField::workgroup_id_x + dim, builder);
    return castValueToType(loc, dimValue, resultType, builder);
  }

  // Loads the workgroup_count[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupCount(Location loc, int32_t dim, Type resultType,
                           OpBuilder &builder) {
    auto dimValue = loadFieldValue(
        loc, DispatchStateField::workgroup_count_x + dim, builder);
    return castValueToType(loc, dimValue, resultType, builder);
  }

  // Loads the workgroup_size[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupSize(Location loc, int32_t dim, Type resultType,
                          OpBuilder &builder) {
    auto dimValue = loadFieldValue(
        loc, DispatchStateField::workgroup_size_x + dim, builder);
    return castValueToType(loc, dimValue, resultType, builder);
  }

  // Returns the estimated maximum concurrency as an index-converted type.
  Value loadMaxConcurrency(Location loc, OpBuilder &builder) {
    auto value =
        loadFieldValue(loc, DispatchStateField::max_concurrency, builder);
    return castValueToType(loc, value,
                           typeConverter->convertType(builder.getIndexType()),
                           builder);
  }

  // Returns the total number of bytes available in workgroup local memory.
  // This may be larger than the requested size.
  Value loadWorkgroupLocalMemorySize(Location loc, OpBuilder &builder) {
    auto value =
        loadFieldValue(loc, WorkgroupStateField::local_memory_size, builder);
    return castValueToType(loc, value,
                           typeConverter->convertType(builder.getIndexType()),
                           builder);
  }

  // Loads the base pointer of the workgroup local memory.
  // Note that this may be NULL if no workgroup local memory was requested.
  Value loadWorkgroupLocalMemoryPtr(Location loc, OpBuilder &builder) {
    return loadFieldValue(loc, WorkgroupStateField::local_memory, builder);
  }

  // Returns the total push constant count as an index-converted type.
  Value loadPushConstantCount(Location loc, OpBuilder &builder) {
    auto value =
        loadFieldValue(loc, DispatchStateField::push_constant_count, builder);
    return castValueToType(loc, value,
                           typeConverter->convertType(builder.getIndexType()),
                           builder);
  }

  // Loads a push constant at |offset| and casts it to |resultType|.
  Value loadPushConstant(Location loc, int64_t offset, Type resultType,
                         OpBuilder &builder) {
    auto constantsPtrValue =
        loadFieldValue(loc, DispatchStateField::push_constants, builder);
    auto offsetValue = getIndexValue(loc, offset, builder);
    Value constantPtrValue = builder.create<LLVM::GEPOp>(
        loc, constantsPtrValue.getType(), constantsPtrValue, offsetValue);
    Value constantValue = builder.create<LLVM::LoadOp>(loc, constantPtrValue);
    return castValueToType(loc, constantValue, resultType, builder);
  }

  // Returns the total binding count as an index-converted type.
  Value loadBindingCount(Location loc, OpBuilder &builder) {
    auto value =
        loadFieldValue(loc, DispatchStateField::binding_count, builder);
    return castValueToType(loc, value,
                           typeConverter->convertType(builder.getIndexType()),
                           builder);
  }

  // Loads the base pointer of the binding |ordinal| as an `i8**`.
  // Equivalent to:
  //   int8_t** base_ptr = &state->binding_ptrs[ordinal];
  Value loadBindingPtr(Location loc, int64_t ordinal, OpBuilder &builder) {
    auto ptrsPtrValue =
        loadFieldValue(loc, DispatchStateField::binding_ptrs, builder);
    auto ordinalValue = getIndexValue(loc, ordinal, builder);
    auto elementPtrValue = builder.createOrFold<LLVM::GEPOp>(
        loc, ptrsPtrValue.getType(), ptrsPtrValue, ordinalValue);
    return builder.createOrFold<LLVM::LoadOp>(loc, elementPtrValue);
  }

  // Loads the byte length of the binding |ordinal| as an index-converted type.
  Value loadBindingLength(Location loc, int64_t ordinal, OpBuilder &builder) {
    auto lengthsPtrValue =
        loadFieldValue(loc, DispatchStateField::binding_lengths, builder);
    auto ordinalValue = getIndexValue(loc, ordinal, builder);
    auto elementPtrValue = builder.createOrFold<LLVM::GEPOp>(
        loc, lengthsPtrValue.getType(), lengthsPtrValue, ordinalValue);
    return builder.createOrFold<LLVM::LoadOp>(loc, elementPtrValue);
  }

  // Loads a binding as a constructed MemRefDescriptor.
  // |baseOffset| can optionally adjust the base byte offset of the buffer.
  MemRefDescriptor loadBinding(Location loc, int64_t ordinal,
                               Value baseOffsetValue, MemRefType memRefType,
                               ValueRange dynamicDims, OpBuilder &builder) {
    // Load the base buffer pointer in the appropriate type (f32*, etc).
    Value basePtrValue = loadBindingPtr(loc, ordinal, builder);

    // Adjust by baseOffset (if needed).
    if (baseOffsetValue) {
      basePtrValue = builder.createOrFold<LLVM::GEPOp>(
          loc, basePtrValue.getType(), basePtrValue, baseOffsetValue);
    }

    // NOTE: if we wanted to check the range was in bounds here would be the
    // place to do it.

    // Cast to the desired memref element type.
    auto elementType = typeConverter->convertType(memRefType.getElementType());
    Value typedPtrValue = builder.createOrFold<LLVM::BitcastOp>(
        loc,
        LLVM::LLVMPointerType::get(elementType,
                                   memRefType.getMemorySpaceAsInt()),
        basePtrValue);

    // Construct the MemRefDescriptor type based on the information we have.
    // NOTE: we could use the binding length to clamp this/check that the
    // requested range is valid.
    if (memRefType.hasStaticShape()) {
      return MemRefDescriptor::fromStaticShape(builder, loc, *typeConverter,
                                               memRefType, typedPtrValue);
    } else {
      assert(memRefType.getNumDynamicDims() == dynamicDims.size());
      int64_t rank = memRefType.getRank();

      // Build MemRef descriptor for this interface binding.
      auto desc = MemRefDescriptor::undef(
          builder, loc, typeConverter->convertType(memRefType));
      desc.setAllocatedPtr(builder, loc, typedPtrValue);
      desc.setAlignedPtr(builder, loc, typedPtrValue);
      desc.setConstantOffset(builder, loc, 0);

      // Update memref descriptor shape. Dynamic dimensions can be mixed with
      // static dimensions, like [128, ?, 128].
      int dynamicDimIndex = 0;
      for (int i = 0; i < rank; ++i) {
        if (memRefType.isDynamicDim(i)) {
          desc.setSize(builder, loc, i, dynamicDims[dynamicDimIndex++]);
        } else {
          desc.setConstantSize(builder, loc, i, memRefType.getDimSize(i));
        }
      }

      // Compute and update strides. Assume that MemRefs are row-major, that is,
      // following index linearization:
      //   x[i, j, k] = i * x.dim[1] * x.dim[2] + j * x.dim[2] + k
      desc.setConstantStride(builder, loc, rank - 1, 1);
      for (int i = rank - 2; i >= 0; --i) {
        auto stride = desc.stride(builder, loc, i + 1);
        auto dim = desc.size(builder, loc, i + 1);
        Value strideVal = builder.create<LLVM::MulOp>(loc, stride, dim);
        desc.setStride(builder, loc, i, strideVal);
      }

      return desc;
    }
  }

  // Loads the processor ID the code is (most likely) being run on.
  // Equivalent to:
  //   uint32_t processor_id = state->processor_id;
  Value loadProcessorID(Location loc, OpBuilder &builder) {
    return loadFieldValue(loc, WorkgroupStateField::processor_id, builder);
  }

  // Loads a processor information data field at the given index.
  // May be 0 if the field is not available.
  Value loadProcessorData(Location loc, int64_t index, OpBuilder &builder) {
    // Load the value; it should always be in bounds.
    Value dataArrayValue = loadFieldValue(loc, ProcessorField::data, builder);
    SmallVector<int64_t, 1> position = {index};
    Value dataValue =
        builder.create<LLVM::ExtractValueOp>(loc, dataArrayValue, position);
    return dataValue;
  }

  // Loads an executable constant with |key| and casts it to |resultType|.
  // A placeholder global will be added for the ordinal.
  Value loadExecutableConstant(Location loc, StringRef key, Type resultType,
                               OpBuilder &builder) {
    // Create top-level global placeholder.
    // The magic attribute is used by future assignment passes.
    std::string globalName = ("__constant_ordinal_" + key).str();
    auto moduleOp =
        builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>();
    LLVM::GlobalOp globalOp;
    if (!(globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName))) {
      auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
      globalOp = moduleBuilder.create<LLVM::GlobalOp>(loc, builder.getI32Type(),
                                                      /*isConstant=*/false,
                                                      LLVM::Linkage::Internal,
                                                      globalName, Attribute{});
      globalOp->setAttr(IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName(),
                        builder.getStringAttr(key));
    }

    // Load the placeholder global ordinal.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
    Value ordinalValue = builder.create<LLVM::LoadOp>(loc, globalPtr);

    // Load constant from the executable constants struct.
    auto constantsPtrValue =
        loadFieldValue(loc, EnvironmentField::constants, builder);
    Value constantPtrValue = builder.create<LLVM::GEPOp>(
        loc, constantsPtrValue.getType(), constantsPtrValue, ordinalValue);
    Value constantValue = builder.create<LLVM::LoadOp>(loc, constantPtrValue);
    return castValueToType(loc, constantValue, resultType, builder);
  }

  // Loads the ordinal of the import with the given |importName|.
  // A placeholder global will be inserted that will be updated with the
  // assigned ordinal after linking.
  Value loadImportOrdinal(Location loc, StringRef importName, bool weak,
                          OpBuilder &builder) {
    // Create top-level global placeholder.
    // The magic attribute is used by future assignment passes.
    std::string globalName = ("__import_ordinal_" + importName).str();
    auto moduleOp =
        builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>();
    LLVM::GlobalOp globalOp;
    if (!(globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName))) {
      auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
      globalOp = moduleBuilder.create<LLVM::GlobalOp>(loc, builder.getI32Type(),
                                                      /*isConstant=*/false,
                                                      LLVM::Linkage::Internal,
                                                      globalName, Attribute{});
      globalOp->setAttr("hal.executable.import.key",
                        builder.getStringAttr(importName));
      if (weak) {
        globalOp->setAttr("hal.executable.import.weak", builder.getUnitAttr());
      }
    }

    // Load the placeholder global ordinal.
    Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
    return builder.create<LLVM::LoadOp>(loc, globalPtr);
  }

  // Loads the import function pointer of the import |ordinal|.
  // Equivalent to:
  //   iree_hal_executable_import_v0_t fn_ptr = state->import_funcs[ordinal];
  //   void* context = state->import_contexts[ordinal];
  std::pair<Value, Value> loadImportFunc(Location loc, Value importOrdinal,
                                         OpBuilder &builder) {
    auto funcPtrsValue =
        loadFieldValue(loc, EnvironmentField::import_funcs, builder);
    auto funcPtrValue = builder.createOrFold<LLVM::GEPOp>(
        loc, funcPtrsValue.getType(), funcPtrsValue, importOrdinal);
    auto contextPtrsValue =
        loadFieldValue(loc, EnvironmentField::import_contexts, builder);
    auto contextPtrValue = builder.createOrFold<LLVM::GEPOp>(
        loc, contextPtrsValue.getType(), contextPtrsValue, importOrdinal);
    return std::make_pair(
        builder.createOrFold<LLVM::LoadOp>(loc, funcPtrValue),
        builder.createOrFold<LLVM::LoadOp>(loc, contextPtrValue));
  }

  // Returns an i1 indicating whether the optional import with |importName| is
  // defined. Equivalent to:
  //   state->import_funcs[ordinal] != NULL
  Value isImportFuncAvailable(Location loc, StringRef importName,
                              OpBuilder &builder) {
    auto importOrdinal =
        loadImportOrdinal(loc, importName, /*weak=*/true, builder);
    auto importFunc = loadImportFunc(loc, importOrdinal, builder);
    Value nullPtrValue =
        builder.create<LLVM::NullOp>(loc, importFunc.first.getType());
    return builder.create<LLVM::ICmpOp>(loc, builder.getI1Type(),
                                        LLVM::ICmpPredicate::ne,
                                        importFunc.first, nullPtrValue);
  }

  // Emits a call to the import with the given |importName|.
  // The provided |params| struct containing the function-specific arguments
  // is passed without modification.
  // Returns 0 on success and non-zero otherwise.
  Value callImport(Location loc, StringRef importName, bool weak, Value params,
                   OpBuilder &builder) {
    auto importOrdinal = loadImportOrdinal(loc, importName, weak, builder);
    auto thunkPtrValue =
        loadFieldValue(loc, EnvironmentField::import_thunk, builder);
    auto importFunc = loadImportFunc(loc, importOrdinal, builder);

    // TODO(benvanik): if weak is set then we should bail if the import is not
    // found. Since we've loaded the import func here we can just compare for
    // null as in isImportFuncAvailable but we'll need to make the control flow.
    assert(!weak && "calls to weak imports not yet implemented");

    Value nullPtrValue = builder.create<LLVM::NullOp>(
        loc, LLVM::LLVMPointerType::get(builder.getI8Type()));
    auto callOp =
        builder.create<LLVM::CallOp>(loc, TypeRange{builder.getI32Type()},
                                     ValueRange{
                                         /*thunk_func_ptr=*/thunkPtrValue,
                                         /*import_func_ptr=*/importFunc.first,
                                         /*context=*/importFunc.second,
                                         /*params=*/params,
                                         /*reserved=*/nullPtrValue,
                                     });
    return callOp.getResult();
  }

  // Emits a call to a dynamically linked import using the given |importName|
  // as a template.
  // The provided |resultTypes| and |args| are packed in a struct and transit
  // through memory so that we can expose a single void* argument.
  // Returns 0 on success and non-zero otherwise.
  SmallVector<Value> wrapAndCallImport(Location loc, StringRef importName,
                                       bool weak, TypeRange resultTypes,
                                       ValueRange args, OpBuilder &builder) {
    // Struct types are ordered [results..., args...].
    SmallVector<Type> types(resultTypes);
    types.reserve(resultTypes.size() + args.size());
    for (Value arg : args) {
      types.push_back(typeConverter->convertType(arg.getType()));
    }

    // Pack parameter structure.
    Type structType;
    Value paramsPtr, voidPtr;
    auto voidPtrTy = LLVM::LLVMPointerType::get(builder.getI8Type());
    if (!types.empty()) {
      // TODO(benvanik): set specific layout to match runtime.
      structType =
          LLVM::LLVMStructType::getLiteral(builder.getContext(), types);
      auto ptrStructType = LLVM::LLVMPointerType::get(structType);
      Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                                   builder.getIndexAttr(1));
      paramsPtr = builder.create<LLVM::AllocaOp>(loc, ptrStructType, one,
                                                 /*alignment=*/0);
      Value structVal = builder.create<LLVM::UndefOp>(loc, structType);
      for (int64_t i = 0, e = args.size(); i < e; ++i) {
        structVal = builder.create<LLVM::InsertValueOp>(loc, structVal, args[i],
                                                        i + resultTypes.size());
      }
      // Store into the alloca'ed descriptor.
      builder.create<LLVM::StoreOp>(loc, structVal, paramsPtr);
      voidPtr = builder.create<LLVM::BitcastOp>(loc, voidPtrTy, paramsPtr);
    } else {
      voidPtr = builder.create<LLVM::UndefOp>(loc, voidPtrTy);
    }

    // Calls return 0 (success) or non-zero (failure).
    auto callResult = callImport(loc, importName, weak, voidPtr, builder);
    Block *trueDest =
        builder.getInsertionBlock()->splitBlock(++builder.getInsertionPoint());
    Block *falseDest = builder.createBlock(trueDest);

    // Check the call results and branch to exit if it failed.
    // Note that we weight the true branch (call successful) higher.
    builder.setInsertionPointAfterValue(callResult);
    Value zeroI32 = builder.create<LLVM::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0));
    Value cmpZero = builder.create<LLVM::ICmpOp>(
        loc, builder.getI1Type(), LLVM::ICmpPredicate::eq, callResult, zeroI32);
    builder.create<LLVM::CondBrOp>(loc, cmpZero, trueDest, ValueRange{},
                                   falseDest, ValueRange{callResult},
                                   std::make_pair(1u, 0u));

    // Failure return block.
    // Return the call result to the runtime.
    builder.setInsertionPointToStart(falseDest);
    builder.create<LLVM::ReturnOp>(
        loc, falseDest->addArgument(builder.getI32Type(), loc));

    // Successful continuation block.
    // Marshal results out of the params struct.
    builder.setInsertionPointToStart(trueDest);
    SmallVector<Value> results;
    if (!resultTypes.empty()) {
      results.reserve(resultTypes.size());
      Value structVal =
          builder.create<LLVM::LoadOp>(loc, structType, paramsPtr);
      for (int64_t i = 0, e = resultTypes.size(); i < e; ++i) {
        results.push_back(
            builder.create<LLVM::ExtractValueOp>(loc, structVal, i));
      }
    }
    return results;
  }

 private:
  Value getIndexValue(Location loc, int64_t value, OpBuilder &builder) {
    return builder.createOrFold<LLVM::ConstantOp>(
        loc, typeConverter->convertType(builder.getIndexType()),
        builder.getI64IntegerAttr(value));
  }

  Value castValueToType(Location loc, Value value, Type resultType,
                        OpBuilder &builder) {
    // NOTE: we should handle more cases here (and proper sign extension).
    if (value.getType() == resultType) return value;
    return builder.createOrFold<LLVM::ZExtOp>(loc, resultType, value);
  }

  Value loadFieldValue(Location loc, EnvironmentField field,
                       OpBuilder &builder) {
    auto environmentPtrValue = funcOp.getArgument(0);
    Value environmentValue =
        builder.create<LLVM::LoadOp>(loc, environmentPtrValue);
    SmallVector<int64_t, 1> position = {int64_t(field)};
    return builder.createOrFold<LLVM::ExtractValueOp>(loc, environmentValue,
                                                      position);
  }

  Value loadFieldValue(Location loc, ProcessorField field, OpBuilder &builder) {
    Value processorValue =
        loadFieldValue(loc, EnvironmentField::processor, builder);
    SmallVector<int64_t, 1> position = {int64_t(field)};
    return builder.createOrFold<LLVM::ExtractValueOp>(loc, processorValue,
                                                      position);
  }

  Value loadFieldValue(Location loc, DispatchStateField field,
                       OpBuilder &builder) {
    Value statePtrValue = funcOp.getArgument(1);
    Value stateValue = builder.createOrFold<LLVM::LoadOp>(loc, statePtrValue);
    SmallVector<int64_t, 1> position = {int64_t(field)};
    return builder.createOrFold<LLVM::ExtractValueOp>(loc, stateValue,
                                                      position);
  }

  Value loadFieldValue(Location loc, WorkgroupStateField field,
                       OpBuilder &builder) {
    Value statePtrValue = funcOp.getArgument(2);
    Value stateValue = builder.createOrFold<LLVM::LoadOp>(loc, statePtrValue);
    SmallVector<int64_t, 1> position = {int64_t(field)};
    return builder.createOrFold<LLVM::ExtractValueOp>(loc, stateValue,
                                                      position);
  }

  LLVM::LLVMFuncOp funcOp;
  LLVMTypeConverter *typeConverter;
  LLVM::LLVMStructType processorType;
  LLVM::LLVMStructType environmentType;
  LLVM::LLVMStructType dispatchStateType;
  LLVM::LLVMStructType workgroupStateType;

  // Used to lock around mutations of shared LLVM type information, e.g.
  // mlir::LLVM::LLVMStructType::getIdentified.
  static llvm::sys::Mutex sMutex;
};

llvm::sys::Mutex HALDispatchABI::sMutex;

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
class ConvertHALEntryPointFuncOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALEntryPointFuncOp(MLIRContext *context,
                                      LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(mlir::func::FuncOp::getOperationName(), context,
                             converter, 100) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto stdFuncOp = cast<func::FuncOp>(op);
    if (!stdFuncOp.isPublic()) return failure();
    FunctionType fnType = stdFuncOp.getFunctionType();
    if (fnType.getNumInputs() != 0 || fnType.getNumResults() != 0) {
      op->emitWarning() << "public functions on executables must be () -> ()";
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
    SmallVector<NamedAttribute, 4> funcAttrs;
    for (auto attr : stdFuncOp->getAttrs()) {
      if (attr.getName() == SymbolTable::getSymbolAttrName() ||
          attr.getName() == mlir::function_interface_impl::getTypeAttrName()) {
        continue;
      }
      funcAttrs.push_back(attr);
    }

    // Clone the function as an LLVMFuncOp and convert all interior types.
    auto llvmFuncType = LLVM::LLVMFunctionType::get(
        IntegerType::get(rewriter.getContext(), 32), abiInputTypes);
    auto llvmFuncOp = rewriter.create<LLVM::LLVMFuncOp>(
        stdFuncOp.getLoc(), stdFuncOp.getName(), llvmFuncType,
        LLVM::Linkage::External, /*dso_local=*/false, /*cconv*/ LLVM::CConv::C,
        funcAttrs);
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

    rewriter.eraseOp(stdFuncOp);
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
/// Because ordinals are not yet available we emit a placeholder global that
/// later gets updated with the value after linking.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALExecutableConstantLoadOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALExecutableConstantLoadOp(MLIRContext *context,
                                              LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::ExecutableConstantLoadOp::getOperationName(), context,
            converter) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    auto loadOp = cast<IREE::HAL::ExecutableConstantLoadOp>(op);
    auto resultType = typeConverter->convertType(op->getResult(0).getType());
    rewriter.replaceOp(op,
                       abi.loadExecutableConstant(op->getLoc(), loadOp.getKey(),
                                                  resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.id to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALInterfaceWorkgroupIDOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceWorkgroupIDOp(MLIRContext *context,
                                            LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceWorkgroupIDOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    int32_t dim = (int32_t)cast<IREE::HAL::InterfaceWorkgroupIDOp>(op)
                      .getDimension()
                      .getZExtValue();
    auto resultType = typeConverter->convertType(op->getResult(0).getType());
    rewriter.replaceOp(
        op, abi.loadWorkgroupID(op->getLoc(), dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.size to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALInterfaceWorkgroupSizeOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceWorkgroupSizeOp(MLIRContext *context,
                                              LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceWorkgroupSizeOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    int32_t dim = (int32_t)cast<IREE::HAL::InterfaceWorkgroupSizeOp>(op)
                      .getDimension()
                      .getZExtValue();
    auto resultType = typeConverter->convertType(op->getResult(0).getType());
    rewriter.replaceOp(
        op, abi.loadWorkgroupSize(op->getLoc(), dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.workgroup.count to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALInterfaceWorkgroupCountOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceWorkgroupCountOp(MLIRContext *context,
                                               LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceWorkgroupCountOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    int32_t dim = (int32_t)cast<IREE::HAL::InterfaceWorkgroupCountOp>(op)
                      .getDimension()
                      .getZExtValue();
    auto resultType = typeConverter->convertType(op->getResult(0).getType());
    rewriter.replaceOp(
        op, abi.loadWorkgroupCount(op->getLoc(), dim, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.constant.load to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALInterfaceConstantLoadOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceConstantLoadOp(MLIRContext *context,
                                             LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceConstantLoadOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    auto loadConstantOp = cast<IREE::HAL::InterfaceConstantLoadOp>(op);
    int64_t index = loadConstantOp.getIndex().getZExtValue();
    auto resultType = typeConverter->convertType(op->getResult(0).getType());
    rewriter.replaceOp(
        op, abi.loadPushConstant(op->getLoc(), index, resultType, rewriter));
    return success();
  }
};

/// Rewrites hal.interface.binding.subspan to ops loading from the ABI structs.
///
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
class ConvertHALInterfaceBindingSubspanOp : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceBindingSubspanOp(MLIRContext *context,
                                               LLVMTypeConverter &converter)
      : ConvertToLLVMPattern(
            IREE::HAL::InterfaceBindingSubspanOp::getOperationName(), context,
            converter) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto llvmFuncOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, getTypeConverter());
    IREE::HAL::InterfaceBindingSubspanOpAdaptor newOperands(
        operands, op->getAttrDictionary());
    MemRefType memRefType = op->getResult(0).getType().dyn_cast<MemRefType>();
    if (!memRefType) {
      return rewriter.notifyMatchFailure(
          op,
          "failed to convert interface.binding.subspan result to memref type");
    }
    auto memRefDesc =
        abi.loadBinding(op->getLoc(), newOperands.getBindingAttr().getInt(),
                        newOperands.getByteOffset(), memRefType,
                        newOperands.getDynamicDims(), rewriter);
    rewriter.replaceOp(op, {memRefDesc});
    return success();
  }
};

/// Rewrites calls to extern functions to dynamic library import calls.
/// The parent LLVMFuncOp must be compatible with HALDispatchABI.
///
/// Note: this is an LLVM::CallOp -> LLVM::CallOp rewrite that is introduced
/// after all conversions are done. Importantly, this is not a conversion
/// pattern.
class RewriteExternCallOpToDynamicImportCallOp
    : public OpRewritePattern<LLVM::CallOp> {
 public:
  explicit RewriteExternCallOpToDynamicImportCallOp(
      MLIRContext *context, LLVMTypeConverter &converter)
      : OpRewritePattern<LLVM::CallOp>(context), typeConverter(converter) {}

  LogicalResult matchAndRewrite(LLVM::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    auto llvmFuncOp = callOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFuncOp) return failure();
    HALDispatchABI abi(llvmFuncOp, &typeConverter);

    // Ignore indirect calls (they're probably already converted imports).
    auto symbol = callOp.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    auto flatSymbol = symbol.dyn_cast_or_null<FlatSymbolRefAttr>();
    if (!flatSymbol) return failure();

    // Ensure the target function is extern.
    // To support conversion inserting calls in local patterns that can't add
    // global function symbols we assume any missing callee is extern.
    auto calleeOp = SymbolTable::lookupNearestSymbolFrom<LLVM::LLVMFuncOp>(
        llvmFuncOp, symbol);
    if (calleeOp && !calleeOp.isExternal()) return failure();

    // TODO(benvanik): way to determine if weak (maybe via linkage?).
    bool weak = false;

    // Rewrite the call to a dynamic import call.
    SmallVector<Value> results = abi.wrapAndCallImport(
        callOp->getLoc(), flatSymbol.getValue(), weak, callOp->getResultTypes(),
        callOp->getOperands(), rewriter);

    rewriter.replaceOp(callOp, results);
    return success();
  }

 private:
  LLVMTypeConverter &typeConverter;
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

}  // namespace

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
    vector::populateVectorContractLoweringPatterns(patterns);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    vector::populateVectorShapeCastLoweringPatterns(patterns);
    vector::populateVectorTransposeLoweringPatterns(patterns);
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
  LLVMTypeConverter converter(&getContext(), options, &dataLayoutAnalysis);

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
    // Use the 32-bit lowering for RISC-V if 'zve32x' is specified and there is
    // no 64-bit integer vector support.
    // TODO(#9440) Simplify logic when 'cpu_features' is simplified.
    use32BitImpl = hasZve32xFeature(targetAttr) && !hasVFeature(targetAttr) &&
                   !hasZve64xFeature(targetAttr);
  }
  tosa::populateTosaRescaleToArithConversionPatterns(&patterns, use32BitImpl);

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateExpandTanhPattern(patterns);

  populateComplexToLLVMConversionPatterns(converter, patterns);
  populateMathToLLVMConversionPatterns(converter, patterns);
  memref::populateExpandStridedMetadataPatterns(patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(
      converter, patterns, targetReassociateFpReductions.getValue());
  populateLinalgToLLVMConversionPatterns(converter, patterns);
  populateReconcileUnrealizedCastsPatterns(patterns);

  // clang-format off
  patterns.insert<
    ConvertHALEntryPointFuncOp,
    ConvertHALExecutableConstantLoadOp,
    ConvertHALInterfaceWorkgroupIDOp,
    ConvertHALInterfaceWorkgroupSizeOp,
    ConvertHALInterfaceWorkgroupCountOp,
    ConvertHALInterfaceConstantLoadOp,
    ConvertHALInterfaceBindingSubspanOp
  >(&getContext(), converter);
  // clang-format on

  LLVMConversionTarget target(getContext());
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
    patterns.insert<RewriteExternCallOpToDynamicImportCallOp>(&getContext(),
                                                              converter);
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
      if (failed(
              applyPatternsAndFoldGreedily(module, std::move(postPatterns)))) {
        return signalPassFailure();
      }
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass(
    bool reassociateFpReductions) {
  return std::make_unique<ConvertToLLVMPass>(reassociateFpReductions);
}

}  // namespace iree_compiler
}  // namespace mlir
