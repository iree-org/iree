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
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ArmNeon2dToIntr/ArmNeon2dToIntr.h"
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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
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
    imports,
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
    // const iree_hal_executable_import_v0_t* imports;
    auto importType = LLVM::LLVMFunctionType::get(uint32Type, int8PtrType);
    auto importPtrType = LLVM::LLVMPointerType::get(importType);
    auto importThunkType =
        LLVM::LLVMFunctionType::get(uint32Type, {importPtrType, int8PtrType});
    fieldTypes.push_back(LLVM::LLVMPointerType::get(importThunkType));
    fieldTypes.push_back(LLVM::LLVMPointerType::get(importPtrType));

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
    Type elementType =
        dataArrayValue.getType().cast<LLVM::LLVMArrayType>().getElementType();
    Value dataValue = builder.create<LLVM::ExtractValueOp>(
        loc, elementType, dataArrayValue, builder.getI64ArrayAttr(index));
    return dataValue;
  }

  // Loads an executable constant at |index| and casts it to |resultType|.
  Value loadExecutableConstant(Location loc, int64_t index, Type resultType,
                               OpBuilder &builder) {
    auto constantsPtrValue =
        loadFieldValue(loc, EnvironmentField::constants, builder);
    auto indexValue = getIndexValue(loc, index, builder);
    Value constantPtrValue = builder.create<LLVM::GEPOp>(
        loc, constantsPtrValue.getType(), constantsPtrValue, indexValue);
    Value constantValue = builder.create<LLVM::LoadOp>(loc, constantPtrValue);
    return castValueToType(loc, constantValue, resultType, builder);
  }

  // Loads the import function pointer of the import |ordinal|.
  // Equivalent to:
  //   iree_hal_executable_import_v0_t func_ptr = state->imports[ordinal];
  Value loadImportFuncPtr(Location loc, int64_t ordinal, OpBuilder &builder) {
    auto importsPtrValue =
        loadFieldValue(loc, EnvironmentField::imports, builder);
    auto ordinalValue = getIndexValue(loc, ordinal, builder);
    auto elementPtrValue = builder.createOrFold<LLVM::GEPOp>(
        loc, importsPtrValue.getType(), importsPtrValue, ordinalValue);
    return builder.createOrFold<LLVM::LoadOp>(loc, elementPtrValue);
  }

  // Returns an i1 indicating whether the optional import with |ordinal| is
  // defined. Equivalent to:
  //   state->imports[ordinal] != NULL
  Value isImportFuncAvailable(Location loc, int64_t ordinal,
                              OpBuilder &builder) {
    auto importPtrValue = loadImportFuncPtr(loc, ordinal, builder);
    auto nullPtrValue =
        builder.create<LLVM::NullOp>(loc, importPtrValue.getType()).getResult();
    return builder.create<LLVM::ICmpOp>(loc, builder.getI1Type(),
                                        LLVM::ICmpPredicate::ne, importPtrValue,
                                        nullPtrValue);
  }

  // Emits a call to the import with the given |importOrdinal|.
  // The provided |params| struct containing the function-specific arguments
  // is passed without modification.
  // Returns 0 on success and non-zero otherwise.
  Value callImport(Location loc, unsigned importOrdinal, Value params,
                   OpBuilder &builder) {
    auto thunkPtrValue =
        loadFieldValue(loc, EnvironmentField::import_thunk, builder);
    auto importPtrValue = loadImportFuncPtr(loc, importOrdinal, builder);
    auto callOp =
        builder.create<LLVM::CallOp>(loc, TypeRange{builder.getI32Type()},
                                     ValueRange{
                                         /*thunk_func_ptr=*/thunkPtrValue,
                                         /*import_func_ptr=*/importPtrValue,
                                         /*import_params=*/params,
                                     });
    return callOp.getResult(0);
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
    Type fieldType = environmentType.getBody()[(int)field];
    return builder.createOrFold<LLVM::ExtractValueOp>(
        loc, fieldType, environmentValue, builder.getI64ArrayAttr((int)field));
  }

  Value loadFieldValue(Location loc, ProcessorField field, OpBuilder &builder) {
    Value processorValue =
        loadFieldValue(loc, EnvironmentField::processor, builder);
    Type fieldType = processorType.getBody()[(int)field];
    return builder.createOrFold<LLVM::ExtractValueOp>(
        loc, fieldType, processorValue, builder.getI64ArrayAttr((int)field));
  }

  Value loadFieldValue(Location loc, DispatchStateField field,
                       OpBuilder &builder) {
    Value statePtrValue = funcOp.getArgument(1);
    Value stateValue = builder.createOrFold<LLVM::LoadOp>(loc, statePtrValue);
    Type fieldType = dispatchStateType.getBody()[(int)field];
    return builder.createOrFold<LLVM::ExtractValueOp>(
        loc, fieldType, stateValue, builder.getI64ArrayAttr((int)field));
  }

  Value loadFieldValue(Location loc, WorkgroupStateField field,
                       OpBuilder &builder) {
    Value statePtrValue = funcOp.getArgument(2);
    Value stateValue = builder.createOrFold<LLVM::LoadOp>(loc, statePtrValue);
    Type fieldType = dispatchStateType.getBody()[(int)field];
    return builder.createOrFold<LLVM::ExtractValueOp>(
        loc, fieldType, stateValue, builder.getI64ArrayAttr((int)field));
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
        LLVM::Linkage::Internal, /*dso_local=*/false, /*cconv*/ LLVM::CConv::C,
        funcAttrs);
    rewriter.inlineRegionBefore(stdFuncOp.getBody(), llvmFuncOp.getBody(),
                                llvmFuncOp.end());
    if (failed(rewriter.convertRegionTypes(
            &llvmFuncOp.getBody(), *typeConverter, &signatureConverter))) {
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
                      .dimension()
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
                      .dimension()
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
                      .dimension()
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
class ConvertHALInterfaceLoadConstant : public ConvertToLLVMPattern {
 public:
  explicit ConvertHALInterfaceLoadConstant(MLIRContext *context,
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
    int64_t index = loadConstantOp.index().getZExtValue();
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
        abi.loadBinding(op->getLoc(), newOperands.bindingAttr().getInt(),
                        newOperands.byte_offset(), memRefType,
                        newOperands.dynamic_dims(), rewriter);
    rewriter.replaceOp(op, {memRefDesc});
    return success();
  }
};

class ConvertToLLVMPass : public ConvertToLLVMBase<ConvertToLLVMPass> {
 public:
  ConvertToLLVMPass() = default;
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
};

}  // namespace

static std::string getStringAttrFromTargetAttr(ModuleOp module,
                                               StringRef attrName) {
  if (auto variantOp =
          module->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.target();
    if (auto config = targetAttr.getConfiguration()) {
      if (auto attr = config.getAs<StringAttr>(attrName)) {
        return attr.getValue().str();
      }
    }
  }
  return "";
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
  auto variantOp = getExecutableVariantOp(module);
  if (succeeded(variantOp) && isRISCV(*variantOp)) {
    // Use the 32-bit lowering for RISC-V if 'zve32x' is specified and there is
    // no 64-bit integer vector support.
    // TODO(#9440) Simplify logic when 'cpu_features' is simplified.
    use32BitImpl = hasZve32xFeature(*variantOp) && !hasVFeature(*variantOp) &&
                   !hasZve64xFeature(*variantOp);
  }
  tosa::populateTosaRescaleToArithConversionPatterns(&patterns, use32BitImpl);

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateExpandTanhPattern(patterns);

  populateMathToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(converter, patterns);
  populateVectorToSCFConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(converter, patterns);
  populateVectorToLLVMConversionPatterns(converter, patterns);
  populateLinalgToLLVMConversionPatterns(converter, patterns);
  populateReconcileUnrealizedCastsPatterns(patterns);

  // clang-format off
  patterns.insert<
    ConvertHALEntryPointFuncOp,
    ConvertHALInterfaceWorkgroupIDOp,
    ConvertHALInterfaceWorkgroupSizeOp,
    ConvertHALInterfaceWorkgroupCountOp,
    ConvertHALInterfaceLoadConstant,
    ConvertHALInterfaceBindingSubspanOp
  >(&getContext(), converter);
  // clang-format on

  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();
  target.addIllegalDialect<func::FuncDialect, mlir::arith::ArithmeticDialect,
                           IREE::Util::UtilDialect, IREE::HAL::HALDialect,
                           math::MathDialect, tosa::TosaDialect>();
  target.addIllegalOp<UnrealizedConversionCastOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
    return;
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

std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
