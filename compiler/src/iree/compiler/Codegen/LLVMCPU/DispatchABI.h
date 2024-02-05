// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMCPU_DISPATCHABI_H_
#define IREE_COMPILER_CODEGEN_LLVMCPU_DISPATCHABI_H_

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

// NOTE: HALDispatchABI and the associated conversion patterns should live under
// iree/compiler/Dialect/HAL/Target/LLVMCPU/ instead of here as they have
// nothing to do with linalg. If we need to use the patterns in this conversion
// we can expose a populate*Patterns() function to access them without needing
// them defined here.

//------------------------------------------------------------------------------
// ExecutableLibraryDI
//------------------------------------------------------------------------------

// Debug information adapter for the executable_library.h types.
// Manually synchronized with the runtime header as needed.
class ExecutableLibraryDI {
public:
  // Initializes a cached DI provider using |typeConverter| to determine
  // variable-width types such as index/size_t.
  explicit ExecutableLibraryDI(const LLVMTypeConverter *typeConverter);

  // Returns `void*`.
  LLVM::DIDerivedTypeAttr getVoidPtr() { return voidPtr; }
  // Returns `int8_t`.
  LLVM::DIDerivedTypeAttr getInt8T() { return int8T; }
  // Returns `uint8_t`.
  LLVM::DIDerivedTypeAttr getUint8T() { return uint8T; }
  // Returns `int16_t`.
  LLVM::DIDerivedTypeAttr getInt16T() { return int16T; }
  // Returns `uint16_t`.
  LLVM::DIDerivedTypeAttr getUint16T() { return uint16T; }
  // Returns `int32_t`.
  LLVM::DIDerivedTypeAttr getInt32T() { return int32T; }
  // Returns `uint32_t`.
  LLVM::DIDerivedTypeAttr getUint32T() { return uint32T; }
  // Returns `int64_t`.
  LLVM::DIDerivedTypeAttr getInt64T() { return int64T; }
  // Returns `uint64_t`.
  LLVM::DIDerivedTypeAttr getUint64T() { return uint64T; }
  // Returns `intptr_t`.
  LLVM::DIDerivedTypeAttr getIntptrT() { return intptrT; }
  // Returns `size_t`.
  LLVM::DIDerivedTypeAttr getSizeT() { return sizeT; }

  // Returns `const |typeAttr|`.
  LLVM::DIDerivedTypeAttr getConstOf(LLVM::DITypeAttr typeAttr);

  // Returns `|typeAttr|*`.
  LLVM::DIDerivedTypeAttr getPtrOf(LLVM::DITypeAttr typeAttr);

  // Returns `|typeAttr|[count]`.
  LLVM::DICompositeTypeAttr getArrayOf(LLVM::DITypeAttr typeAttr,
                                       int64_t count);

  // Returns `using |name| = |typeAttr|`.
  LLVM::DIDerivedTypeAttr getTypedefOf(StringRef name,
                                       LLVM::DITypeAttr typeAttr);

  // Returns a member |name| of |typeAttr| at bit offset |offsetInBits|.
  // Upon return |offsetInBits| is updated to point after the member.
  LLVM::DIDerivedTypeAttr getMemberOf(StringRef name, LLVM::DITypeAttr typeAttr,
                                      unsigned *offsetInBits);

  // Returns the DI type for the given LLVM type.
  LLVM::DITypeAttr getBasicType(Type type);

  // Returns `iree_hal_processor_v0_t`.
  LLVM::DICompositeTypeAttr getProcessorV0T();
  // Returns `iree_hal_executable_environment_v0_t`.
  LLVM::DIDerivedTypeAttr getEnvironmentV0T();
  // Returns `iree_hal_executable_dispatch_state_v0_t`.
  LLVM::DIDerivedTypeAttr getDispatchStateV0T();
  // Returns `iree_hal_executable_workgroup_state_v0_t`.
  LLVM::DIDerivedTypeAttr getWorkgroupStateV0T();

private:
  Builder builder;
  LLVM::DIFileAttr fileAttr;
  unsigned ptrBitwidth;

  // Cached as they are commonly constructed with any usage of the DI info.
  LLVM::DIDerivedTypeAttr voidPtr;
  LLVM::DIDerivedTypeAttr int8T;
  LLVM::DIDerivedTypeAttr uint8T;
  LLVM::DIDerivedTypeAttr int16T;
  LLVM::DIDerivedTypeAttr uint16T;
  LLVM::DIDerivedTypeAttr int32T;
  LLVM::DIDerivedTypeAttr uint32T;
  LLVM::DIDerivedTypeAttr int64T;
  LLVM::DIDerivedTypeAttr uint64T;
  LLVM::DIDerivedTypeAttr intptrT;
  LLVM::DIDerivedTypeAttr sizeT;
};

//------------------------------------------------------------------------------
// HALDispatchABI
//------------------------------------------------------------------------------

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
  static LLVM::LLVMStructType
  getProcessorType(MLIRContext *context,
                   const LLVMTypeConverter *typeConverter);

  // Matches the field order in iree_hal_executable_environment_v0_t.
  enum class EnvironmentField {
    constants,
    import_thunk,
    import_funcs,
    import_contexts,
    processor,
  };

  // Returns a Type representing iree_hal_executable_environment_v0_t.
  static LLVM::LLVMStructType
  getEnvironmentType(MLIRContext *context,
                     const LLVMTypeConverter *typeConverter,
                     LLVM::LLVMStructType processorType);

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
  static LLVM::LLVMStructType
  getDispatchStateType(MLIRContext *context,
                       const LLVMTypeConverter *typeConverter);

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
  static LLVM::LLVMStructType
  getWorkgroupStateType(MLIRContext *context,
                        const LLVMTypeConverter *typeConverter);

  // Returns the types of the LLVM function inputs for the ABI.
  // This matches the signature of `iree_hal_executable_dispatch_v0_t` in
  // `iree/hal/local/executable_library.h`.
  static SmallVector<Type, 5>
  getInputTypes(MLIRContext *context, const LLVMTypeConverter *typeConverter);

  // Builds a DISubprogram for |llvmFuncOp| function in |moduleOp|.
  // This is required in order to get any debug information (including line
  // tables) from MLIR into LLVM IR. It does not need to match the exact
  // definition but the closer we can make it to the real thing the more useful
  // downstream tools will be.
  static LLVM::DISubprogramAttr
  buildScopeAttr(mlir::ModuleOp moduleOp, LLVM::LLVMFuncOp llvmFuncOp,
                 const LLVMTypeConverter *typeConverter);

  explicit HALDispatchABI(LLVMTypeConverter *typeConverter)
      : context(&typeConverter->getContext()), typeConverter(typeConverter),
        processorType(getProcessorType(context, typeConverter)),
        environmentType(
            getEnvironmentType(context, typeConverter, processorType)),
        dispatchStateType(getDispatchStateType(context, typeConverter)),
        workgroupStateType(getWorkgroupStateType(context, typeConverter)),
        di(typeConverter) {}

  // Loads the workgroup_id[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupID(Operation *forOp, int32_t dim, Type resultType,
                        OpBuilder &builder);

  // Loads the workgroup_count[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupCount(Operation *forOp, int32_t dim, Type resultType,
                           OpBuilder &builder);

  // Loads the workgroup_size[dim] value (XYZ) and casts it to |resultType|.
  Value loadWorkgroupSize(Operation *forOp, int32_t dim, Type resultType,
                          OpBuilder &builder);

  // Returns the estimated maximum concurrency as an index-converted type.
  Value loadMaxConcurrency(Operation *forOp, OpBuilder &builder);

  // Returns the total number of bytes available in workgroup local memory.
  // This may be larger than the requested size.
  Value loadWorkgroupLocalMemorySize(Operation *forOp, OpBuilder &builder);

  // Loads the base pointer of the workgroup local memory.
  // Note that this may be NULL if no workgroup local memory was requested.
  Value loadWorkgroupLocalMemoryPtr(Operation *forOp, OpBuilder &builder);

  // Returns the total push constant count as an index-converted type.
  Value loadPushConstantCount(Operation *forOp, OpBuilder &builder);

  // Loads a push constant at |offset| and casts it to |resultType|.
  Value loadPushConstant(Operation *forOp, int64_t offset, Type resultType,
                         OpBuilder &builder);

  // Returns the total binding count as an index-converted type.
  Value loadBindingCount(Operation *forOp, OpBuilder &builder);

  // Loads the base pointer of the binding |ordinal| as an `i8**`.
  // Equivalent to:
  //   int8_t** base_ptr = &state->binding_ptrs[ordinal];
  Value loadBindingPtr(Operation *forOp, int64_t ordinal, OpBuilder &builder);

  // Loads the byte length of the binding |ordinal| as an index-converted type.
  Value loadBindingLength(Operation *forOp, int64_t ordinal,
                          OpBuilder &builder);

  // Loads a binding as a constructed MemRefDescriptor.
  // |baseOffset| can optionally adjust the base byte offset of the buffer.
  MemRefDescriptor loadBinding(Operation *forOp, int64_t ordinal,
                               Value baseOffsetValue, MemRefType memRefType,
                               ValueRange dynamicDims, OpBuilder &builder);

  // Loads the processor ID the code is (most likely) being run on.
  // Equivalent to:
  //   uint32_t processor_id = state->processor_id;
  Type getProcessorIDType();
  Value loadProcessorID(Operation *forOp, OpBuilder &builder);

  // Loads a pointer to the processor information data fields.
  Type getProcessorDataType();
  Value loadProcessorData(Operation *forOp, OpBuilder &builder);

  // Loads a processor information data field at the given index.
  // May be 0 if the field is not available.
  Value loadProcessorData(Operation *forOp, int64_t index, OpBuilder &builder);

  // Loads an executable constant with |key| and casts it to |resultType|.
  // A placeholder global will be added for the ordinal.
  Value loadExecutableConstant(Operation *forOp, StringRef key, Type resultType,
                               OpBuilder &builder);

  // Loads the ordinal of the import with the given |importName|.
  // A placeholder global will be inserted that will be updated with the
  // assigned ordinal after linking.
  Value loadImportOrdinal(Operation *forOp, StringRef importName, bool weak,
                          OpBuilder &builder);

  // Loads the import function pointer of the import |ordinal|.
  // Equivalent to:
  //   iree_hal_executable_import_v0_t fn_ptr = state->import_funcs[ordinal];
  //   void* context = state->import_contexts[ordinal];
  std::pair<Value, Value> loadImportFunc(Operation *forOp, Value importOrdinal,
                                         OpBuilder &builder);

  // Returns an i1 indicating whether the optional import with |importName| is
  // defined. Equivalent to:
  //   state->import_funcs[ordinal] != NULL
  Value isImportFuncAvailable(Operation *forOp, StringRef importName,
                              OpBuilder &builder);

  // Emits a call to the import with the given |importName|.
  // The provided |params| struct containing the function-specific arguments
  // is passed without modification.
  // Returns 0 on success and non-zero otherwise.
  Value callImport(Operation *forOp, StringRef importName, bool weak,
                   Value params, OpBuilder &builder);

  // Emits a call to a dynamically linked import using the given |importName|
  // as a template.
  //
  // The provided |resultTypes| and |args| are packed in a struct and transit
  // through memory so that we can expose a single void* argument. Optionally
  // |extraFields| can be specified with an ordered list of field names to be
  // appended to the end of the struct.
  //
  // Returns 0 on success and non-zero otherwise.
  SmallVector<Value> wrapAndCallImport(Operation *forOp, StringRef importName,
                                       bool weak, TypeRange resultTypes,
                                       ValueRange args,
                                       ArrayRef<StringRef> extraFields,
                                       OpBuilder &builder);

  //===--------------------------------------------------------------------==//
  // External/bitcode function ABI handling methods.
  //===--------------------------------------------------------------------==//
  // Methods required for handling ABI for functions whose definitions are
  // external.

  /// Check if the `funcType` is equivalent to a function with return being
  /// of type `resultTypes` and operands of type `paramTypes`.
  static bool hasCompatibleFunctionSignature(MLIRContext *context,
                                             LLVM::LLVMFunctionType funcType,
                                             TypeRange resultTypes,
                                             TypeRange paramTypes);

  /// Given a calling convention `cConv`, and callee with return of
  /// `resultTypes` and operands with type `argTypes`, along with extra fields
  /// to append to argument list specified in `extraFields`; return the function
  /// type of use for the function that implements the specified calling
  /// convention.
  FailureOr<LLVM::LLVMFunctionType>
  getABIFunctionType(Operation *forOp, IREE::HAL::CallingConvention cConv,
                     TypeRange resultTypes, TypeRange argTypes,
                     ArrayRef<StringRef> extraFields);

  /// Given a calling convention `cConv`, and callee with return of
  /// `resultTypes` and operands with type `argTypes`, along with extra fields
  /// to append to argument list specified in `extraFields`; modify the `callOp`
  /// to implement the specified ABI. The calleee signature is expected to have
  /// been/to be modified separately, i.e. it isnt done within this method.
  FailureOr<SmallVector<Value>>
  materializeABI(Operation *callOp, StringRef symbolName,
                 IREE::HAL::CallingConvention cConv, TypeRange resultTypes,
                 ValueRange args, ArrayRef<StringRef> extraFields,
                 RewriterBase &builder);

private:
  Value getIndexValue(Location loc, int64_t value, OpBuilder &builder);

  Value castValueToType(Location loc, Value value, Type resultType,
                        OpBuilder &builder);

  Value loadFieldValue(Operation *forOp, EnvironmentField field,
                       OpBuilder &builder);
  Value loadFieldValue(Operation *forOp, ProcessorField field,
                       OpBuilder &builder);
  Value loadFieldValue(Operation *forOp, DispatchStateField field,
                       OpBuilder &builder);
  Type getFieldType(WorkgroupStateField field);
  Value loadFieldValue(Operation *forOp, WorkgroupStateField field,
                       OpBuilder &builder);

  Type getExtraFieldType(StringRef extraField);
  Value getExtraField(Operation *forOp, StringRef extraField,
                      OpBuilder &builder);

  // Update the processor data based on the `cpu_features` present in
  // `executable.target` attribute.
  Value updateProcessorDataFromTargetAttr(Operation *forOp,
                                          Value processorDataPtrValue,
                                          OpBuilder &builder);

  // Return LLVM Struct type that represents a container for arguments
  // and return types. The struct type are ordered [results..., args...]
  std::optional<Type> getParameterStructType(TypeRange resultTypes,
                                             ValueRange args,
                                             TypeRange extraFieldsTypes);
  // For a given call operation, generate the struct that is the container
  // for passing the arguments.
  //
  // The provided |resultTypes| and |args| are packed in a struct and transit
  // through memory so that we can expose a single void* argument. Optionally
  // |extraFields| can be specified with an ordered list of field names to be
  // appended to the end of the struct.
  std::tuple<Type, Value> packIntoParameterStruct(Operation *forOp,
                                                  TypeRange resultTypes,
                                                  ValueRange args,
                                                  ValueRange extraFields,
                                                  OpBuilder &builder);

  mlir::MLIRContext *context;
  const LLVMTypeConverter *typeConverter;
  LLVM::LLVMStructType processorType;
  LLVM::LLVMStructType environmentType;
  LLVM::LLVMStructType dispatchStateType;
  LLVM::LLVMStructType workgroupStateType;
  LLVM::DISubprogramAttr scopeAttr;
  ExecutableLibraryDI di;

  // Used to lock around mutations of shared LLVM type information, e.g.
  // mlir::LLVM::LLVMStructType::getIdentified.
  static llvm::sys::Mutex sMutex;
};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMCPU_DISPATCHABI_H_
