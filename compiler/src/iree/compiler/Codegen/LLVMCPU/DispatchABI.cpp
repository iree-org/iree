// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/DispatchABI.h"

#include "iree/schemas/cpu_data.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

static llvm::cl::opt<bool> clVerboseDebugInfo(
    "iree-codegen-llvm-verbose-debug-info",
    llvm::cl::desc("Emit verbose debug information in LLVM IR."),
    llvm::cl::init(false));

namespace mlir {
namespace iree_compiler {

// List of all defined llvm feature-name to bit pattern used to represent it.
// This is derived based on the schema in `runtime/src/iree/schemas/`.
// TODO(ravishankarm): This link to the runtime schemas needs to be broken.
// Instead we should use a reflection callback to resolve arch guarded features
// directly in the compiler.

// Struct to capture tuple of llvm feature-name to bit pattern used to represent
// it.
struct iree_llvm_name_and_bit_pattern_t {
  const char *llvm_name;
  unsigned long long bit_pattern;
};

#define IREE_CPU_FEATURE_BIT_NAME(arch, field_index, bit_name) \
  IREE_CPU_DATA##field_index##_##arch##_##bit_name

#define IREE_CPU_FEATURE_NAME_AND_BIT_PATTERN(arch, field_index, bit_name, \
                                              llvm_name)                   \
  {llvm_name, IREE_CPU_FEATURE_BIT_NAME(arch, field_index, bit_name)},

static const struct iree_llvm_name_and_bit_pattern_t
    iree_llvm_name_and_bit_pattern_list[] = {

#define IREE_CPU_FEATURE_BIT(arch, field_index, bit_pos, bit_name, llvm_name) \
  IREE_CPU_FEATURE_NAME_AND_BIT_PATTERN(arch, field_index, bit_name, llvm_name)
#include "iree/schemas/cpu_feature_bits.inl"
#undef IREE_CPU_FEATURE_BIT

};

#undef IREE_CPU_FEATURE_NAME_AND_BIT_PATTERN
#undef IREE_CPU_FEATURE_BIT_NAME

//------------------------------------------------------------------------------
// ExecutableLibraryDI
//------------------------------------------------------------------------------

// NOTE: the debug information used here is only present as a compiler developer
// aid. It may get out of sync with published versions of the executable ABI and
// may not be very clean. For example, we size push constant and binding arrays
// not based on the actual layout of the pipeline layout but with some
// reasonable limit that allows for use in a debugger. If we wanted to improve
// this we could customize per-scope the structures and look up the
// IREE::HAL::PipelineLayoutAttr for each entry point to discover the real
// limits.
//
// NOTE: MLIR and subsequent LLVM optimizations will often remove a lot of this
// debug information (or at least make it less useful). This can happen even in
// LLVM modes of -O0 as MLIR has no such configurability at this time.
//
// It'd be nice to have an automatic sync of the debug information and structs
// in this file such that we'd get matching source file/line information with
// the runtime headers and be able to delete most of this hand-authored code.
// The current debug information and types were constructed by compiling
// executable_library_demo.c to LLVM IR and then importing it into MLIR to see
// what the attributes look like. We could automate this and embed the
// attributes in the compiler binary but due to the differences in 32/64-bit
// pointer widths some manual massaging may still be required (or we just embed
// both).
//
// $ clang -emit-llvm -Iruntime/src/ \
//     runtime/src/iree/hal/local/executable_library_demo.c -g -S \
//     --target=x86_64-pc-windows-elf
// $ mlir-translate --import-llvm executable_library_demo.ll

// Returns the size, in bits, of |typeAttr|.
static unsigned getDITypeSizeInBits(LLVM::DITypeAttr typeAttr) {
  if (auto basicTypeAttr = typeAttr.dyn_cast<LLVM::DIBasicTypeAttr>()) {
    return basicTypeAttr.getSizeInBits();
  } else if (auto derivedTypeAttr =
                 typeAttr.dyn_cast<LLVM::DIDerivedTypeAttr>()) {
    if (unsigned derivedSize = derivedTypeAttr.getSizeInBits()) {
      return derivedSize;
    } else {
      return getDITypeSizeInBits(derivedTypeAttr.getBaseType());
    }
  } else {
    return 0;
  }
}

ExecutableLibraryDI::ExecutableLibraryDI(LLVMTypeConverter *typeConverter)
    : typeConverter(typeConverter), builder(&typeConverter->getContext()) {
  auto *context = builder.getContext();
  fileAttr = LLVM::DIFileAttr::get(
      context, "runtime/src/iree/hal/local/executable_library.h", ".");
  ptrBitwidth = typeConverter->getPointerBitwidth();

  voidPtr = getPtrOf(LLVM::DIBasicTypeAttr::get(
      context, llvm::dwarf::DW_TAG_base_type, "void",
      /*sizeInBits=*/0, llvm::dwarf::DW_ATE_address));
  int8T = getTypedefOf("int8_t",
                       LLVM::DIBasicTypeAttr::get(
                           context, llvm::dwarf::DW_TAG_base_type, "char",
                           /*sizeInBits=*/8, llvm::dwarf::DW_ATE_signed_char));
  uint8T = getTypedefOf(
      "uint8_t", LLVM::DIBasicTypeAttr::get(
                     context, llvm::dwarf::DW_TAG_base_type, "unsigned char",
                     /*sizeInBits=*/8, llvm::dwarf::DW_ATE_unsigned_char));
  int16T = getTypedefOf("int16_t",
                        LLVM::DIBasicTypeAttr::get(
                            context, llvm::dwarf::DW_TAG_base_type, "short",
                            /*sizeInBits=*/16, llvm::dwarf::DW_ATE_signed));
  uint16T = getTypedefOf(
      "uint16_t", LLVM::DIBasicTypeAttr::get(
                      context, llvm::dwarf::DW_TAG_base_type, "unsigned short",
                      /*sizeInBits=*/16, llvm::dwarf::DW_ATE_unsigned));
  int32T = getTypedefOf("int32_t",
                        LLVM::DIBasicTypeAttr::get(
                            context, llvm::dwarf::DW_TAG_base_type, "int",
                            /*sizeInBits=*/32, llvm::dwarf::DW_ATE_signed));
  uint32T = getTypedefOf(
      "uint32_t", LLVM::DIBasicTypeAttr::get(
                      context, llvm::dwarf::DW_TAG_base_type, "unsigned int",
                      /*sizeInBits=*/32, llvm::dwarf::DW_ATE_unsigned));
  int64T = getTypedefOf(
      "int64_t", LLVM::DIBasicTypeAttr::get(
                     context, llvm::dwarf::DW_TAG_base_type, "long long int",
                     /*sizeInBits=*/64, llvm::dwarf::DW_ATE_signed));
  uint64T = getTypedefOf(
      "uint64_t",
      LLVM::DIBasicTypeAttr::get(
          context, llvm::dwarf::DW_TAG_base_type, "long long unsigned int",
          /*sizeInBits=*/64, llvm::dwarf::DW_ATE_unsigned));
  intptrT =
      getTypedefOf("intptr_t", ptrBitwidth == 32 ? getInt32T() : getInt64T());
  sizeT =
      getTypedefOf("size_t", ptrBitwidth == 32 ? getUint32T() : getUint64T());
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getConstOf(
    LLVM::DITypeAttr typeAttr) {
  return LLVM::DIDerivedTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_const_type,
      /*name=*/nullptr, typeAttr, /*sizeInBits=*/0, /*alignInBits=*/0,
      /*offsetInBits=*/0);
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getPtrOf(
    LLVM::DITypeAttr typeAttr) {
  return LLVM::DIDerivedTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_pointer_type,
      /*name=*/nullptr, typeAttr, /*sizeInBits=*/ptrBitwidth,
      /*alignInBits=*/0,
      /*offsetInBits=*/0);
}

LLVM::DICompositeTypeAttr ExecutableLibraryDI::getArrayOf(
    LLVM::DITypeAttr typeAttr, int64_t count) {
  return LLVM::DICompositeTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_array_type,
      /*name=*/builder.getStringAttr(""), fileAttr,
      /*line=*/227, fileAttr,
      /*baseType=*/typeAttr, LLVM::DIFlags::Zero,
      /*sizeInBits=*/getDITypeSizeInBits(typeAttr) * count,
      /*alignInBits=*/0,
      {
          LLVM::DISubrangeAttr::get(
              builder.getContext(), builder.getI64IntegerAttr(count),
              /*lowerBound=*/nullptr, /*upperBound=*/nullptr,
              /*stride=*/nullptr),
      });
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getTypedefOf(
    StringRef name, LLVM::DITypeAttr typeAttr) {
  return LLVM::DIDerivedTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_typedef,
      builder.getStringAttr(name), typeAttr, /*sizeInBits=*/0,
      /*alignInBits=*/0, /*offsetInBits=*/0);
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getMemberOf(
    StringRef name, LLVM::DITypeAttr typeAttr, unsigned *offsetInBits) {
  unsigned memberOffsetInBits = *offsetInBits;
  unsigned memberSizeInBits = getDITypeSizeInBits(typeAttr);
  *offsetInBits += memberSizeInBits;
  return LLVM::DIDerivedTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_member,
      builder.getStringAttr(name), typeAttr,
      /*sizeInBits=*/memberSizeInBits, /*alignInBits=*/0,
      /*offsetInBits=*/memberOffsetInBits);
}

LLVM::DITypeAttr ExecutableLibraryDI::getBasicType(Type type) {
  return TypeSwitch<Type, LLVM::DITypeAttr>(type)
      .Case([&](IndexType) { return getIntptrT(); })
      .Case([&](IntegerType integerType) -> LLVM::DITypeAttr {
        unsigned bitWidth = integerType.getIntOrFloatBitWidth();
        switch (bitWidth) {
          case 8:
            return integerType.isUnsigned() ? getUint8T() : getInt8T();
          case 16:
            return integerType.isUnsigned() ? getUint16T() : getInt16T();
          case 32:
            return integerType.isUnsigned() ? getUint32T() : getInt32T();
          case 64:
            return integerType.isUnsigned() ? getUint64T() : getInt64T();
          default:
            return LLVM::DIBasicTypeAttr::get(
                builder.getContext(), llvm::dwarf::DW_TAG_base_type,
                StringRef("int") + std::to_string(bitWidth),
                /*sizeInBits=*/bitWidth,
                integerType.isUnsigned() ? llvm::dwarf::DW_ATE_unsigned
                                         : llvm::dwarf::DW_ATE_signed);
        }
      })
      .Case([&](FloatType floatType) -> LLVM::DITypeAttr {
        unsigned bitWidth = floatType.getIntOrFloatBitWidth();
        return LLVM::DIBasicTypeAttr::get(
            builder.getContext(), llvm::dwarf::DW_TAG_base_type,
            StringRef("float") + std::to_string(bitWidth),
            /*sizeInBits=*/bitWidth, llvm::dwarf::DW_ATE_float);
      })
      .Default([](Type) {
        assert(false && "unhandled basic type");
        return nullptr;
      });
}

LLVM::DICompositeTypeAttr ExecutableLibraryDI::getProcessorV0T() {
  unsigned offsetInBits = 0;
  return LLVM::DICompositeTypeAttr::get(
      builder.getContext(), llvm::dwarf::DW_TAG_structure_type,
      builder.getStringAttr("iree_hal_processor_v0_t"), fileAttr,
      /*line=*/227, fileAttr,
      /*baseType=*/nullptr, LLVM::DIFlags::Zero, /*sizeInBits=*/512,
      /*alignInBits=*/0,
      {
          getMemberOf("data", getArrayOf(getUint64T(), 8), &offsetInBits),
      });
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getEnvironmentV0T() {
  unsigned offsetInBits = 0;
  return getTypedefOf(
      "iree_hal_executable_environment_v0_t",
      LLVM::DICompositeTypeAttr::get(
          builder.getContext(), llvm::dwarf::DW_TAG_structure_type,
          builder.getStringAttr("iree_hal_executable_environment_v0_t"),
          fileAttr,
          /*line=*/246, fileAttr,
          /*baseType=*/nullptr, LLVM::DIFlags::Zero, /*sizeInBits=*/768,
          /*alignInBits=*/0,
          {
              getMemberOf("constants",
                          getPtrOf(getConstOf(getArrayOf(getUint32T(), 64))),
                          &offsetInBits),
              getMemberOf("import_thunk", getVoidPtr(), &offsetInBits),
              getMemberOf("import_funcs", getPtrOf(getConstOf(getVoidPtr())),
                          &offsetInBits),
              getMemberOf("import_contexts",
                          getPtrOf(getPtrOf(getConstOf(getVoidPtr()))),
                          &offsetInBits),
              getMemberOf("processor", getProcessorV0T(), &offsetInBits),
          }));
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getDispatchStateV0T() {
  unsigned offsetInBits = 0;
  return getTypedefOf(
      "iree_hal_executable_dispatch_state_v0_t",
      LLVM::DICompositeTypeAttr::get(
          builder.getContext(), llvm::dwarf::DW_TAG_structure_type,
          builder.getStringAttr("iree_hal_executable_dispatch_state_v0_t"),
          fileAttr, /*line=*/275, fileAttr,
          /*baseType=*/nullptr, LLVM::DIFlags::Zero, /*sizeInBits=*/384,
          /*alignInBits=*/0,
          {
              getMemberOf("workgroup_size_x", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_size_y", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_size_z", getUint16T(), &offsetInBits),
              getMemberOf("push_constant_count", getUint16T(), &offsetInBits),
              getMemberOf("workgroup_count_x", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_count_y", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_count_z", getUint16T(), &offsetInBits),
              getMemberOf("max_concurrency", getUint8T(), &offsetInBits),
              getMemberOf("binding_count", getUint8T(), &offsetInBits),
              getMemberOf("push_constants",
                          getPtrOf(getConstOf(getArrayOf(getUint32T(), 64))),
                          &offsetInBits),
              getMemberOf(
                  "binding_ptrs",
                  getPtrOf(getConstOf(getArrayOf(getPtrOf(getUint8T()), 64))),
                  &offsetInBits),
              getMemberOf("binding_lengths",
                          getPtrOf(getConstOf(getArrayOf(getSizeT(), 64))),
                          &offsetInBits),
          }));
}

LLVM::DIDerivedTypeAttr ExecutableLibraryDI::getWorkgroupStateV0T() {
  unsigned offsetInBits = 0;
  return getTypedefOf(
      "iree_hal_executable_workgroup_state_v0_t",
      LLVM::DICompositeTypeAttr::get(
          builder.getContext(), llvm::dwarf::DW_TAG_structure_type,
          builder.getStringAttr("iree_hal_executable_workgroup_state_v0_t"),
          fileAttr, /*line=*/321, fileAttr,
          /*baseType=*/nullptr, LLVM::DIFlags::Zero, /*sizeInBits=*/256,
          /*alignInBits=*/0,
          {
              getMemberOf("workgroup_id_x", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_id_y", getUint32T(), &offsetInBits),
              getMemberOf("workgroup_id_z", getUint16T(), &offsetInBits),
              getMemberOf("reserved", getUint16T(), &offsetInBits),
              getMemberOf("processor_id", getUint32T(), &offsetInBits),
              getMemberOf("local_memory", getVoidPtr(), &offsetInBits),
              getMemberOf("local_memory_size", getUint32T(), &offsetInBits),
          }));
}

//------------------------------------------------------------------------------
// HALDispatchABI
//------------------------------------------------------------------------------

// static
llvm::sys::Mutex HALDispatchABI::sMutex;

// static
LLVM::LLVMStructType HALDispatchABI::getProcessorType(
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

// static
LLVM::LLVMStructType HALDispatchABI::getEnvironmentType(
    MLIRContext *context, LLVMTypeConverter *typeConverter,
    LLVM::LLVMStructType processorType) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_environment_v0_t");
  if (structType.isInitialized()) return structType;

  auto uint32Type = IntegerType::get(context, 32);
  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
  SmallVector<Type, 4> fieldTypes;

  // const uint32_t* constants;
  fieldTypes.push_back(opaquePtrType);

  // iree_hal_executable_import_thunk_v0_t import_thunk;
  // const iree_hal_executable_import_v0_t* import_funcs;
  // const void** import_contexts;
  auto importThunkType = LLVM::LLVMFunctionType::get(
      uint32Type, {opaquePtrType, opaquePtrType, opaquePtrType, opaquePtrType});
  fieldTypes.push_back(LLVM::LLVMPointerType::get(importThunkType));
  fieldTypes.push_back(LLVM::LLVMPointerType::get(opaquePtrType));
  fieldTypes.push_back(LLVM::LLVMPointerType::get(opaquePtrType));

  // iree_hal_processor_v0_t processor;
  fieldTypes.push_back(processorType);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
LLVM::LLVMStructType HALDispatchABI::getDispatchStateType(
    MLIRContext *context, LLVMTypeConverter *typeConverter) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_dispatch_state_v0_t");
  if (structType.isInitialized()) return structType;

  auto uint8Type = IntegerType::get(context, 8);
  auto uint16Type = IntegerType::get(context, 16);
  auto uint32Type = IntegerType::get(context, 32);
  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
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
  // void *const * binding_ptrs;
  // const size_t * binding_lengths;
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(opaquePtrType);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
LLVM::LLVMStructType HALDispatchABI::getWorkgroupStateType(
    MLIRContext *context, LLVMTypeConverter *typeConverter) {
  llvm::sys::ScopedLock lock(sMutex);
  auto structType = LLVM::LLVMStructType::getIdentified(
      context, "iree_hal_executable_workgroup_state_v0_t");
  if (structType.isInitialized()) return structType;

  auto uint16Type = IntegerType::get(context, 16);
  auto uint32Type = IntegerType::get(context, 32);
  auto opaquePtrType = LLVM::LLVMPointerType::get(context);
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
  fieldTypes.push_back(opaquePtrType);
  fieldTypes.push_back(uint32Type);

  LogicalResult bodySet = structType.setBody(fieldTypes, /*isPacked=*/false);
  assert(succeeded(bodySet) &&
         "could not set the body of an identified struct");
  (void)bodySet;

  return structType;
}

// static
SmallVector<Type, 5> HALDispatchABI::getInputTypes(
    MLIRContext *context, LLVMTypeConverter *typeConverter) {
  return SmallVector<Type, 5>{
      // const iree_hal_executable_environment_v0_t* IREE_RESTRICT
      //   environment
      LLVM::LLVMPointerType::get(context),
      // const iree_hal_executable_dispatch_state_v0_t* IREE_RESTRICT
      //   dispatch_state
      LLVM::LLVMPointerType::get(context),
      // const iree_hal_executable_workgroup_state_v0_t* IREE_RESTRICT
      //   workgroup_state
      LLVM::LLVMPointerType::get(context),
  };
}

// static
LLVM::DISubprogramAttr HALDispatchABI::buildScopeAttr(
    mlir::ModuleOp moduleOp, StringRef funcName,
    LLVMTypeConverter *typeConverter) {
  auto *context = &typeConverter->getContext();
  Builder builder(context);

  std::string inputFilePath("-");
  if (auto fileLoc = moduleOp.getLoc().dyn_cast<mlir::FileLineColLoc>()) {
    inputFilePath = fileLoc.getFilename().getValue();
  }

  auto fileAttr =
      LLVM::DIFileAttr::get(context, llvm::sys::path::filename(inputFilePath),
                            llvm::sys::path::parent_path(inputFilePath));
  auto compileUnitAttr = LLVM::DICompileUnitAttr::get(
      context, llvm::dwarf::DW_LANG_C17, fileAttr,
      builder.getStringAttr("IREE"), /*isOptimized=*/true,
      LLVM::DIEmissionKind::Full);

  auto int32TypeAttr =
      LLVM::DIBasicTypeAttr::get(context, llvm::dwarf::DW_TAG_base_type, "int",
                                 /*sizeInBits=*/32, llvm::dwarf::DW_ATE_signed);
  ExecutableLibraryDI di(typeConverter);
  auto subroutineTypeAttr = LLVM::DISubroutineTypeAttr::get(
      context, llvm::dwarf::DW_CC_normal,
      {
          int32TypeAttr,
          di.getPtrOf(di.getConstOf(di.getEnvironmentV0T())),
          di.getPtrOf(di.getConstOf(di.getDispatchStateV0T())),
          di.getPtrOf(di.getConstOf(di.getWorkgroupStateV0T())),
      });

  auto funcNameAttr = builder.getStringAttr(funcName);
  return LLVM::DISubprogramAttr::get(
      context, compileUnitAttr, fileAttr, funcNameAttr, funcNameAttr, fileAttr,
      /*line=*/1,
      /*scopeline=*/1,
      LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
      subroutineTypeAttr);
}

// Returns the most local DISubprogramAttr starting from |forOp|.
static LLVM::DISubprogramAttr getLocalScopeAttr(Operation *forOp) {
  auto funcOp = forOp->getParentOfType<LLVM::LLVMFuncOp>();
  assert(funcOp && "usage requires an enclosing LLVMFuncOp");
  auto scopeLocAttr =
      funcOp.getLoc()
          ->findInstanceOf<mlir::FusedLocWith<LLVM::DISubprogramAttr>>();
  assert(scopeLocAttr &&
         "must have attached a DISubprogramAttr to the parent function");
  return scopeLocAttr.getMetadata();
}

// Returns the argument at |argIndex| in the parent function of |forOp|.
static Value getLocalArgument(Operation *forOp, unsigned argIndex) {
  auto funcOp = forOp->getParentOfType<LLVM::LLVMFuncOp>();
  assert(funcOp && "usage requires an enclosing LLVMFuncOp");
  return funcOp.getArgument(argIndex);
}

// Returns "x" "y" or "z" based on |dim|.
static StringRef getDimName(int32_t dim) {
  assert(dim >= 0 && dim <= 2 && "must be x, y, z");
  static const char *dims[3] = {"x", "y", "z"};
  return StringRef(dims[dim]);
}

// Debug intrinsics require valid location information to pass LLVM's verifier.
// Since nothing checks these cases in MLIR before converting we avoid creating
// the ops if MLIR or LLVM is likely to reject them.
static bool isLocationValidForDI(Location loc) {
  // Unknown locations are passed as null and DI doesn't like that.
  if (loc.isa<UnknownLoc>()) return false;
  // MLIR currently can't handle name-only locations. We do this check to ensure
  // there's at least one real location MLIR can pass along.
  if (auto callLoc = loc.dyn_cast<CallSiteLoc>()) {
    return isLocationValidForDI(callLoc.getCaller()) &&
           isLocationValidForDI(callLoc.getCallee());
  } else if (auto fileLoc = loc.dyn_cast<FileLineColLoc>()) {
    return true;
  } else if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
    return llvm::all_of(fusedLoc.getLocations(), isLocationValidForDI);
  } else if (auto namedLoc = loc.dyn_cast<NameLoc>()) {
    return isLocationValidForDI(namedLoc.getChildLoc());
  } else if (auto opaqueLoc = loc.dyn_cast<OpaqueLoc>()) {
    return isLocationValidForDI(opaqueLoc.getFallbackLocation());
  }
  return false;
}

static Value buildArgDI(Operation *forOp, int argNum, Value value, Twine name,
                        LLVM::DITypeAttr type, OpBuilder &builder) {
  if (!clVerboseDebugInfo) return value;
  auto loc = forOp->getLoc();
  if (!isLocationValidForDI(loc)) return value;
  auto scopeAttr = getLocalScopeAttr(forOp);
  builder.create<LLVM::DbgValueOp>(
      loc, value,
      LLVM::DILocalVariableAttr::get(scopeAttr, builder.getStringAttr(name),
                                     scopeAttr.getFile(),
                                     /*line=*/1, /*arg=*/argNum + 1,
                                     /*alignInBits=*/0, type));
  return value;
}

static Value buildValueDI(Operation *forOp, Value value, Twine name,
                          LLVM::DITypeAttr type, OpBuilder &builder) {
  if (!clVerboseDebugInfo) return value;
  auto loc = forOp->getLoc();
  if (!isLocationValidForDI(loc)) return value;
  auto scopeAttr = getLocalScopeAttr(forOp);
  builder.create<LLVM::DbgValueOp>(
      loc, value,
      LLVM::DILocalVariableAttr::get(scopeAttr, builder.getStringAttr(name),
                                     scopeAttr.getFile(),
                                     /*line=*/1, /*arg=*/0,
                                     /*alignInBits=*/0, type));
  return value;
}

Value HALDispatchABI::loadWorkgroupID(Operation *forOp, int32_t dim,
                                      Type resultType, OpBuilder &builder) {
  auto dimValue =
      loadFieldValue(forOp, WorkgroupStateField::workgroup_id_x + dim, builder);
  auto resultValue =
      castValueToType(forOp->getLoc(), dimValue, resultType, builder);
  return buildValueDI(forOp, resultValue,
                      StringRef("workgroup_id_") + getDimName(dim),
                      di.getBasicType(resultType), builder);
}

Value HALDispatchABI::loadWorkgroupCount(Operation *forOp, int32_t dim,
                                         Type resultType, OpBuilder &builder) {
  auto dimValue = loadFieldValue(
      forOp, DispatchStateField::workgroup_count_x + dim, builder);
  auto resultValue =
      castValueToType(forOp->getLoc(), dimValue, resultType, builder);
  return buildValueDI(forOp, resultValue,
                      StringRef("workgroup_count_") + getDimName(dim),
                      di.getBasicType(resultType), builder);
}

Value HALDispatchABI::loadWorkgroupSize(Operation *forOp, int32_t dim,
                                        Type resultType, OpBuilder &builder) {
  auto dimValue = loadFieldValue(
      forOp, DispatchStateField::workgroup_size_x + dim, builder);
  auto resultValue =
      castValueToType(forOp->getLoc(), dimValue, resultType, builder);
  return buildValueDI(forOp, resultValue,
                      StringRef("workgroup_size_") + getDimName(dim),
                      di.getBasicType(resultType), builder);
}

Value HALDispatchABI::loadMaxConcurrency(Operation *forOp, OpBuilder &builder) {
  auto maxValue =
      loadFieldValue(forOp, DispatchStateField::max_concurrency, builder);
  auto resultValue = castValueToType(
      forOp->getLoc(), maxValue,
      typeConverter->convertType(builder.getIndexType()), builder);
  return buildValueDI(forOp, resultValue, "max_concurrency", di.getIntptrT(),
                      builder);
}

Value HALDispatchABI::loadWorkgroupLocalMemorySize(Operation *forOp,
                                                   OpBuilder &builder) {
  auto sizeValue =
      loadFieldValue(forOp, WorkgroupStateField::local_memory_size, builder);
  auto resultValue = castValueToType(
      forOp->getLoc(), sizeValue,
      typeConverter->convertType(builder.getIndexType()), builder);
  return buildValueDI(forOp, resultValue, "local_memory_size", di.getSizeT(),
                      builder);
}

Value HALDispatchABI::loadWorkgroupLocalMemoryPtr(Operation *forOp,
                                                  OpBuilder &builder) {
  auto resultValue =
      loadFieldValue(forOp, WorkgroupStateField::local_memory, builder);
  return buildValueDI(forOp, resultValue, "local_memory", di.getVoidPtr(),
                      builder);
}

Value HALDispatchABI::loadPushConstantCount(Operation *forOp,
                                            OpBuilder &builder) {
  auto countValue =
      loadFieldValue(forOp, DispatchStateField::push_constant_count, builder);
  auto resultValue = castValueToType(
      forOp->getLoc(), countValue,
      typeConverter->convertType(builder.getIndexType()), builder);
  return buildValueDI(forOp, resultValue, "push_constant_count", di.getSizeT(),
                      builder);
}

Value HALDispatchABI::loadPushConstant(Operation *forOp, int64_t offset,
                                       Type resultType, OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto constantsPtrValue =
      loadFieldValue(forOp, DispatchStateField::push_constants, builder);
  auto pushConstantType = IntegerType::get(context, 32);
  Value constantPtrValue = builder.create<LLVM::GEPOp>(
      loc, constantsPtrValue.getType(), pushConstantType, constantsPtrValue,
      LLVM::GEPArg(int32_t(offset)));
  Value constantValue =
      builder.create<LLVM::LoadOp>(loc, pushConstantType, constantPtrValue);
  auto resultValue = castValueToType(loc, constantValue, resultType, builder);
  return buildValueDI(
      forOp, resultValue,
      StringRef("push_constant[") + std::to_string(offset) + "]",
      di.getBasicType(resultType), builder);
}

Value HALDispatchABI::loadBindingCount(Operation *forOp, OpBuilder &builder) {
  auto countValue =
      loadFieldValue(forOp, DispatchStateField::binding_count, builder);
  auto resultValue = castValueToType(
      forOp->getLoc(), countValue,
      typeConverter->convertType(builder.getIndexType()), builder);
  return buildValueDI(forOp, resultValue, "binding_count", di.getSizeT(),
                      builder);
}

Value HALDispatchABI::loadBindingPtr(Operation *forOp, int64_t ordinal,
                                     OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto ptrsPtrValue =
      loadFieldValue(forOp, DispatchStateField::binding_ptrs, builder);
  auto elementPtrValue = builder.create<LLVM::GEPOp>(
      loc, ptrsPtrValue.getType(),
      mlir::LLVM::LLVMPointerType::get(builder.getContext()), ptrsPtrValue,
      LLVM::GEPArg(int32_t(ordinal)));
  auto elementValue = builder.create<LLVM::LoadOp>(
      loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
      elementPtrValue);
  return buildValueDI(
      forOp, elementValue,
      StringRef("binding_ptrs[") + std::to_string(ordinal) + "]",
      di.getPtrOf(di.getUint8T()), builder);
}

Value HALDispatchABI::loadBindingLength(Operation *forOp, int64_t ordinal,
                                        OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto lengthsPtrValue =
      loadFieldValue(forOp, DispatchStateField::binding_lengths, builder);
  auto indexType = typeConverter->convertType(IndexType::get(context));
  auto elementPtrValue = builder.create<LLVM::GEPOp>(
      loc, lengthsPtrValue.getType(), indexType, lengthsPtrValue,
      LLVM::GEPArg(int32_t(ordinal)));
  auto elementValue =
      builder.create<LLVM::LoadOp>(loc, indexType, elementPtrValue);
  return buildValueDI(
      forOp, elementValue,
      StringRef("binding_lengths[") + std::to_string(ordinal) + "]",
      di.getSizeT(), builder);
}

MemRefDescriptor HALDispatchABI::loadBinding(Operation *forOp, int64_t ordinal,
                                             Value baseOffsetValue,
                                             MemRefType memRefType,
                                             ValueRange dynamicDims,
                                             OpBuilder &builder) {
  auto loc = forOp->getLoc();

  // Load the base buffer pointer in the appropriate type (f32*, etc).
  Value basePtrValue = loadBindingPtr(forOp, ordinal, builder);

  // NOTE: if we wanted to check the range was in bounds here would be the
  // place to do it.

  // Construct the MemRefDescriptor type based on the information we have.
  // NOTE: we could use the binding length to clamp this/check that the
  // requested range is valid.
  auto [strides, offset] = getStridesAndOffset(memRefType);
  if (memRefType.hasStaticShape() &&
      !llvm::any_of(strides, ShapedType::isDynamic) &&
      !ShapedType::isDynamic(offset)) {
    return MemRefDescriptor::fromStaticShape(builder, loc, *typeConverter,
                                             memRefType, basePtrValue);
  } else {
    assert(memRefType.getNumDynamicDims() == dynamicDims.size());
    int64_t rank = memRefType.getRank();

    // Build MemRef descriptor for this interface binding.
    auto desc = MemRefDescriptor::undef(builder, loc,
                                        typeConverter->convertType(memRefType));
    desc.setAllocatedPtr(builder, loc, basePtrValue);
    desc.setAlignedPtr(builder, loc, basePtrValue);
    auto llvmIndexType = typeConverter->convertType(builder.getIndexType());
    if (ShapedType::isDynamic(offset)) {
      // The offset in the subspan is byteoffset. It is converted to element
      // offset here. It is assumed that the byte offset is a multiple of
      // the element type byte width.
      int32_t elementWidth =
          IREE::Util::getRoundedElementByteWidth(memRefType.getElementType());
      Value elementWidthVal =
          builder.create<LLVM::ConstantOp>(loc, llvmIndexType, elementWidth);
      Value elementOffsetVal =
          builder.create<LLVM::UDivOp>(loc, baseOffsetValue, elementWidthVal);
      desc.setOffset(builder, loc, elementOffsetVal);
    } else {
      desc.setConstantOffset(builder, loc, offset);
    }

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
    if (!strides.empty()) {
      assert(strides.back() == 1 &&
             "unexpected non-unit stride for innermost dimension");
      desc.setConstantStride(builder, loc, rank - 1, 1);
      OpFoldResult currentStride = builder.getIndexAttr(1);
      for (int i = rank - 1; i > 0; --i) {
        if (strides[i - 1] == ShapedType::kDynamic) {
          auto dim = desc.size(builder, loc, i);
          Value currentStrideVal;
          if (std::optional<int64_t> currentStrideInt =
                  getConstantIntValue(currentStride)) {
            currentStrideVal = builder.create<LLVM::ConstantOp>(
                loc, llvmIndexType, currentStrideInt.value());
          } else {
            currentStrideVal = currentStride.get<Value>();
          }
          currentStride =
              builder.create<LLVM::MulOp>(loc, currentStrideVal, dim)
                  .getResult();
          desc.setStride(builder, loc, i - 1, currentStride.get<Value>());
        } else {
          currentStride = builder.getIndexAttr(strides[i - 1]);
          desc.setConstantStride(builder, loc, i - 1, strides[i - 1]);
        }
      }
    }

    return desc;
  }
}

Type HALDispatchABI::getProcessorIDType() {
  return getFieldType(WorkgroupStateField::processor_id);
}

Value HALDispatchABI::loadProcessorID(Operation *forOp, OpBuilder &builder) {
  auto resultValue =
      loadFieldValue(forOp, WorkgroupStateField::processor_id, builder);
  return buildValueDI(forOp, resultValue, "processor_id",
                      di.getBasicType(resultValue.getType()), builder);
}

Value HALDispatchABI::updateProcessorDataFromTargetAttr(
    Operation *forOp, Value processorDataPtrValue, OpBuilder &builder) {
  // Get the target attr.
  IREE::HAL::ExecutableTargetAttr targetAttr =
      IREE::HAL::ExecutableTargetAttr::lookup(forOp);
  if (!targetAttr) {
    return processorDataPtrValue;
  }

  // Lookup CPU features.
  std::optional<NamedAttribute> cpuFeatures =
      targetAttr.getConfiguration().getNamed("cpu_features");
  if (!cpuFeatures) {
    return processorDataPtrValue;
  }

  SmallVector<uint64_t> specifiedFeatureBitPatterns;
  {
    llvm::StringMap<uint64_t> featureToBitPattern;
    for (auto [llvmName, bitPattern] : iree_llvm_name_and_bit_pattern_list) {
      featureToBitPattern[llvmName] = bitPattern;
    }
    SmallVector<StringRef> cpuFeatureStrings;
    cpuFeatures->getValue().cast<StringAttr>().getValue().split(
        cpuFeatureStrings, ',', /*MakeSplit=*/-1, /*KeepEmpty=*/false);
    for (auto featureString : cpuFeatureStrings) {
      if (featureToBitPattern.count(featureString.drop_front())) {
        specifiedFeatureBitPatterns.push_back(
            featureToBitPattern.lookup(featureString.drop_front()));
      }
    }
  }
  if (specifiedFeatureBitPatterns.empty()) {
    return processorDataPtrValue;
  }

  // Create a new stack allocation for the bit pattern.
  Location loc = forOp->getLoc();
  MLIRContext *context = forOp->getContext();
  auto ptrType = LLVM::LLVMPointerType::get(context);
  auto i64Ty = builder.getI64Type();
  Value arraySize = builder.create<LLVM::ConstantOp>(
      loc, i64Ty, builder.getI64IntegerAttr(ProcessorDataCapacity));
  Value alloca = builder.create<LLVM::AllocaOp>(loc, ptrType, i64Ty, arraySize,
                                                /*alignment=*/sizeof(uint64_t));
  // Load the 0-th value.
  Value srcData0 =
      builder.create<LLVM::LoadOp>(loc, i64Ty, processorDataPtrValue);
  // Set the specified CPU arch data.
  for (auto bitPattern : specifiedFeatureBitPatterns) {
    Value bitPatternVal = builder.create<LLVM::ConstantOp>(
        loc, i64Ty, builder.getI64IntegerAttr(bitPattern));
    srcData0 = builder.create<LLVM::OrOp>(loc, srcData0, bitPatternVal);
  }
  builder.create<LLVM::StoreOp>(loc, srcData0, alloca);
  // Copy over the rest.
  for (int64_t i = 1, e = ProcessorDataCapacity; i < e; ++i) {
    Value loadPtr = builder.create<LLVM::GEPOp>(
        loc, processorDataPtrValue.getType(), i64Ty, processorDataPtrValue,
        LLVM::GEPArg(int32_t(i)), /*inbounds =*/true);
    Value loadVal = builder.create<LLVM::LoadOp>(loc, i64Ty, loadPtr);
    Value storePtr = builder.create<LLVM::GEPOp>(
        loc, alloca.getType(), i64Ty, alloca, LLVM::GEPArg(int32_t(i)),
        /*inbounds =*/true);
    builder.create<LLVM::StoreOp>(loc, loadVal, storePtr);
  }
  return alloca;
}

Type HALDispatchABI::getProcessorDataType() {
  return LLVM::LLVMPointerType::get(processorType.getContext());
}

Value HALDispatchABI::loadProcessorData(Operation *forOp, OpBuilder &builder) {
  // To get a pointer to the processor data we need to track pointers all the
  // way from the environment argument. This is redundant with loadFieldValue
  // but that returns values instead.
  auto loc = forOp->getLoc();
  auto environmentPtrValue =
      buildArgDI(forOp, /*argNum=*/0, getLocalArgument(forOp, 0), "environment",
                 di.getPtrOf(di.getConstOf(di.getEnvironmentV0T())), builder);
  Value processorPtrValue = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(context),
      LLVM::LLVMPointerType::get(environmentType), environmentPtrValue,
      LLVM::GEPArg(int32_t(EnvironmentField::processor)),
      /*inbounds=*/true);
  Value processorDataPtrValue = builder.create<LLVM::GEPOp>(
      loc, LLVM::LLVMPointerType::get(context),
      LLVM::LLVMPointerType::get(processorType), processorPtrValue,
      LLVM::GEPArg(int32_t(ProcessorField::data)),
      /*inbounds=*/true);
  Value updatedProcessorData =
      updateProcessorDataFromTargetAttr(forOp, processorDataPtrValue, builder);
  return buildValueDI(forOp, updatedProcessorData, "processor_data",
                      di.getPtrOf(di.getConstOf(di.getArrayOf(
                          di.getUint64T(), ProcessorDataCapacity))),
                      builder);
}

Value HALDispatchABI::loadProcessorData(Operation *forOp, int64_t index,
                                        OpBuilder &builder) {
  // Load the value; it should always be in bounds.
  Value dataArrayValue = loadFieldValue(forOp, ProcessorField::data, builder);
  SmallVector<int64_t, 1> position = {index};
  Value dataValue = builder.create<LLVM::ExtractValueOp>(
      forOp->getLoc(), dataArrayValue, position);
  return buildValueDI(
      forOp, dataValue,
      StringRef("processor_data[") + std::to_string(index) + "]",
      di.getBasicType(dataValue.getType()), builder);
}

Value HALDispatchABI::loadExecutableConstant(Operation *forOp, StringRef key,
                                             Type resultType,
                                             OpBuilder &builder) {
  auto loc = forOp->getLoc();

  // Create top-level global placeholder.
  // The magic attribute is used by future assignment passes.
  std::string globalName = ("__constant_ordinal_" + key).str();
  auto moduleOp =
      builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>();
  LLVM::GlobalOp globalOp;
  if (!(globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName))) {
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    globalOp = moduleBuilder.create<LLVM::GlobalOp>(
        loc, builder.getI32Type(),
        /*isConstant=*/false, LLVM::Linkage::Internal, globalName, Attribute{});
    globalOp->setAttr(IREE::HAL::ExecutableConstantBlockOp::getKeyAttrName(),
                      builder.getStringAttr(key));
  }

  // Load the placeholder global ordinal.
  Value globalPtr = builder.create<LLVM::AddressOfOp>(loc, globalOp);
  Value ordinalValue = builder.create<LLVM::LoadOp>(loc, globalPtr);

  // Load constant from the executable constants struct.
  auto constantsPtrValue =
      loadFieldValue(forOp, EnvironmentField::constants, builder);
  Value constantPtrValue =
      builder.create<LLVM::GEPOp>(loc, constantsPtrValue.getType(), resultType,
                                  constantsPtrValue, ordinalValue);
  Value constantValue =
      builder.create<LLVM::LoadOp>(loc, resultType, constantPtrValue);
  auto resultValue = castValueToType(loc, constantValue, resultType, builder);
  return buildValueDI(forOp, resultValue,
                      StringRef("executable_constant['") + key + "']",
                      di.getBasicType(resultValue.getType()), builder);
}

Value HALDispatchABI::loadImportOrdinal(Operation *forOp, StringRef importName,
                                        bool weak, OpBuilder &builder) {
  auto loc = forOp->getLoc();

  // Create top-level global placeholder.
  // The magic attribute is used by future assignment passes.
  std::string globalName = ("__import_ordinal_" + importName).str();
  auto moduleOp =
      builder.getInsertionPoint()->getParentOfType<mlir::ModuleOp>();
  LLVM::GlobalOp globalOp;
  if (!(globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName))) {
    auto moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    globalOp = moduleBuilder.create<LLVM::GlobalOp>(
        loc, builder.getI32Type(),
        /*isConstant=*/false, LLVM::Linkage::Internal, globalName, Attribute{});
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

std::pair<Value, Value> HALDispatchABI::loadImportFunc(Operation *forOp,
                                                       Value importOrdinal,
                                                       OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto funcPtrsValue =
      loadFieldValue(forOp, EnvironmentField::import_funcs, builder);
  auto opaquePtrType = LLVM::LLVMPointerType::get(builder.getContext());
  auto funcPtrValue =
      builder.create<LLVM::GEPOp>(loc, funcPtrsValue.getType(), opaquePtrType,
                                  funcPtrsValue, importOrdinal);
  auto contextPtrsValue =
      loadFieldValue(forOp, EnvironmentField::import_contexts, builder);
  auto contextPtrValue = builder.create<LLVM::GEPOp>(
      loc, contextPtrsValue.getType(), opaquePtrType, contextPtrsValue,
      importOrdinal);
  return std::make_pair(
      builder.create<LLVM::LoadOp>(loc, opaquePtrType, funcPtrValue),
      builder.create<LLVM::LoadOp>(loc, opaquePtrType, contextPtrValue));
}

Value HALDispatchABI::isImportFuncAvailable(Operation *forOp,
                                            StringRef importName,
                                            OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto importOrdinal =
      loadImportOrdinal(forOp, importName, /*weak=*/true, builder);
  auto importFunc = loadImportFunc(forOp, importOrdinal, builder);
  Value nullPtrValue =
      builder.create<LLVM::NullOp>(loc, importFunc.first.getType());
  return builder.create<LLVM::ICmpOp>(loc, builder.getI1Type(),
                                      LLVM::ICmpPredicate::ne, importFunc.first,
                                      nullPtrValue);
}

Value HALDispatchABI::callImport(Operation *forOp, StringRef importName,
                                 bool weak, Value params, OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto importOrdinal = loadImportOrdinal(forOp, importName, weak, builder);
  auto thunkPtrValue =
      loadFieldValue(forOp, EnvironmentField::import_thunk, builder);
  auto importFunc = loadImportFunc(forOp, importOrdinal, builder);

  // TODO(benvanik): if weak is set then we should bail if the import is not
  // found. Since we've loaded the import func here we can just compare for
  // null as in isImportFuncAvailable but we'll need to make the control flow.
  assert(!weak && "calls to weak imports not yet implemented");

  Value nullPtrValue = builder.create<LLVM::NullOp>(
      loc, LLVM::LLVMPointerType::get(builder.getContext()));
  auto callOp =
      builder.create<LLVM::CallOp>(loc, TypeRange{builder.getI32Type()},
                                   ValueRange{
                                       /*thunk_func_ptr=*/thunkPtrValue,
                                       /*import_func_ptr=*/importFunc.first,
                                       /*params=*/params,
                                       /*context=*/importFunc.second,
                                       /*reserved=*/nullPtrValue,
                                   });
  return callOp.getResult();
}

// static
std::optional<Type> HALDispatchABI::getParameterStructType(
    TypeRange resultTypes, ValueRange args, TypeRange extraFieldsTypes) {
  // Struct types are ordered [results..., args...].
  SmallVector<Type> types(resultTypes);
  types.reserve(resultTypes.size() + args.size());
  for (Value arg : args) {
    types.push_back(typeConverter->convertType(arg.getType()));
  }
  types.append(extraFieldsTypes.begin(), extraFieldsTypes.end());

  if (types.empty()) {
    return std::nullopt;
  }
  return LLVM::LLVMStructType::getLiteral(context, types);
}

// static
std::tuple<Type, Value> HALDispatchABI::packIntoParameterStruct(
    Operation *forOp, TypeRange resultTypes, ValueRange args,
    ValueRange extraFields, OpBuilder &builder) {
  Location loc = forOp->getLoc();
  MLIRContext *context = builder.getContext();

  // Query any extra fields that were requested and append them to the struct.
  auto extraFieldsTypes = llvm::to_vector(
      llvm::map_range(extraFields, [](Value v) { return v.getType(); }));

  std::optional<Type> structType =
      getParameterStructType(resultTypes, args, extraFieldsTypes);

  if (!structType) {
    Type voidPtrType = LLVM::LLVMPointerType::get(context);
    return {voidPtrType,
            builder.create<LLVM::UndefOp>(loc, voidPtrType).getResult()};
  }

  auto ptrStructType = LLVM::LLVMPointerType::get(context);
  Value one = builder.create<LLVM::ConstantOp>(loc, builder.getI64Type(),
                                               builder.getIndexAttr(1));
  Value paramsPtr =
      builder.create<LLVM::AllocaOp>(loc, ptrStructType, *structType, one,
                                     /*alignment=*/0);
  Value structVal = builder.create<LLVM::UndefOp>(loc, *structType);
  for (int64_t i = 0, e = args.size(); i < e; ++i) {
    structVal = builder.create<LLVM::InsertValueOp>(loc, structVal, args[i],
                                                    i + resultTypes.size());
  }
  for (int64_t i = 0, e = extraFields.size(); i < e; ++i) {
    structVal = builder.create<LLVM::InsertValueOp>(
        loc, structVal, extraFields[i], i + resultTypes.size() + args.size());
  }
  // Store into the alloca'ed descriptor.
  builder.create<LLVM::StoreOp>(loc, structVal, paramsPtr);
  return {*structType, paramsPtr};
}

// static
FailureOr<LLVM::LLVMFunctionType> HALDispatchABI::getABIFunctionType(
    Operation *forOp, IREE::HAL::CallingConvention cConv, TypeRange resultTypes,
    TypeRange argTypes, ArrayRef<StringRef> extraFields) {
  MLIRContext *context = forOp->getContext();
  SmallVector<Type> extraFieldsTypes = llvm::to_vector(llvm::map_range(
      extraFields, [&](StringRef name) { return getExtraFieldType(name); }));

  // Check for extra fields already added.
  if (argTypes.size() >= extraFieldsTypes.size()) {
    if (llvm::all_of(llvm::zip(argTypes.take_back(extraFieldsTypes.size()),
                               extraFieldsTypes),
                     [](auto it) {
                       auto lhsType = std::get<0>(it);
                       auto rhsType = std::get<1>(it);
                       return (lhsType.template isa<LLVM::LLVMPointerType>() &&
                               rhsType.template isa<LLVM::LLVMPointerType>()) ||
                              std::get<0>(it) == std::get<1>(it);
                     })) {
      // Extra fields already added. Drop them.
      extraFieldsTypes.clear();
    }
  }

  switch (cConv) {
    case IREE::HAL::CallingConvention::Default: {
      if (resultTypes.size() > 1) {
        return forOp->emitOpError(
            "Cannot have multiple return values for function");
      }
      Type resultType = resultTypes.size() == 1
                            ? resultTypes[0]
                            : LLVM::LLVMVoidType::get(context);
      SmallVector<Type> allArgTypes = argTypes;
      allArgTypes.append(extraFieldsTypes.begin(), extraFieldsTypes.end());
      return LLVM::LLVMFunctionType::get(resultType, allArgTypes);
    }
    case IREE::HAL::CallingConvention::ParameterStruct:
      return LLVM::LLVMFunctionType::get(LLVM::LLVMVoidType::get(context),
                                         LLVM::LLVMPointerType::get(context));
  }
}

// static
bool HALDispatchABI::hasCompatibleFunctionSignature(
    MLIRContext *context, LLVM::LLVMFunctionType funcType,
    TypeRange resultTypes, TypeRange paramTypes) {
  TypeRange funcParamTypes = funcType.getParams();
  if (funcParamTypes.size() != paramTypes.size()) {
    return false;
  }
  if (!llvm::all_of(llvm::zip(funcParamTypes, paramTypes), [](auto it) {
        auto lhsType = std::get<0>(it);
        auto rhsType = std::get<1>(it);
        return (lhsType.template isa<LLVM::LLVMPointerType>() &&
                rhsType.template isa<LLVM::LLVMPointerType>()) ||
               std::get<0>(it) == std::get<1>(it);
      })) {
    return false;
  }
  if (resultTypes.size() > 1) {
    return false;
  }
  Type funcResultType = funcType.getReturnType();
  if (resultTypes.empty() &&
      funcResultType != LLVM::LLVMVoidType::get(context)) {
    return false;
  }
  if (resultTypes.size() == 1 && resultTypes[0] != funcResultType) {
    return false;
  }
  return true;
}

FailureOr<SmallVector<Value>> HALDispatchABI::materializeABI(
    Operation *forOp, StringRef symbolName, IREE::HAL::CallingConvention cConv,
    TypeRange resultTypes, ValueRange args, ArrayRef<StringRef> extraFields,
    RewriterBase &rewriter) {
  auto argTypes = llvm::to_vector(
      llvm::map_range(args, [](Value v) { return v.getType(); }));
  FailureOr<LLVM::LLVMFunctionType> abiFunctionType =
      getABIFunctionType(forOp, cConv, resultTypes, argTypes, extraFields);
  if (failed(abiFunctionType)) {
    return forOp->emitOpError(
        "failed to get function type for calling convention");
  }
  if (hasCompatibleFunctionSignature(rewriter.getContext(),
                                     abiFunctionType.value(), resultTypes,
                                     argTypes)) {
    return rewriter.notifyMatchFailure(
        forOp, "no change in function signature. skipping");
  }

  // Combined args list.
  SmallVector<Value> allArgsList = llvm::to_vector(args);
  SmallVector<Value> extraFieldVals =
      llvm::to_vector(llvm::map_range(extraFields, [&](StringRef fieldName) {
        return getExtraField(forOp, fieldName, rewriter);
      }));
  allArgsList.append(extraFieldVals);

  Location loc = forOp->getLoc();
  if (cConv == IREE::HAL::CallingConvention::Default) {
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, abiFunctionType->getReturnTypes(), allArgsList, forOp->getAttrs());
    return llvm::to_vector(llvm::map_range(
        callOp.getResults(), [](OpResult v) -> Value { return v; }));
  }

  if (cConv == IREE::HAL::CallingConvention::ParameterStruct) {
    auto [structType, paramsStructPtr] = packIntoParameterStruct(
        forOp, resultTypes, args, extraFieldVals, rewriter);
    rewriter.create<LLVM::CallOp>(loc, TypeRange{}, paramsStructPtr,
                                  forOp->getAttrs());
    SmallVector<Value> results;
    if (!resultTypes.empty()) {
      results.reserve(resultTypes.size());
      Value structVal =
          rewriter.create<LLVM::LoadOp>(loc, structType, paramsStructPtr);
      for (int64_t i = 0, e = resultTypes.size(); i < e; ++i) {
        results.push_back(
            rewriter.create<LLVM::ExtractValueOp>(loc, structVal, i));
      }
    }
    return results;
  }
  return forOp->emitOpError("unhandled calling convention");
}

SmallVector<Value> HALDispatchABI::wrapAndCallImport(
    Operation *forOp, StringRef importName, bool weak, TypeRange resultTypes,
    ValueRange args, ArrayRef<StringRef> extraFields, OpBuilder &builder) {
  auto loc = forOp->getLoc();

  SmallVector<Value> extraFieldVals =
      llvm::to_vector(llvm::map_range(extraFields, [&](StringRef fieldName) {
        return getExtraField(forOp, fieldName, builder);
      }));

  auto [structType, paramsPtr] = packIntoParameterStruct(
      forOp, resultTypes, args, extraFieldVals, builder);

  // Calls return 0 (success) or non-zero (failure).
  auto callResult = callImport(forOp, importName, weak, paramsPtr, builder);
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
    Value structVal = builder.create<LLVM::LoadOp>(loc, structType, paramsPtr);
    for (int64_t i = 0, e = resultTypes.size(); i < e; ++i) {
      results.push_back(
          builder.create<LLVM::ExtractValueOp>(loc, structVal, i));
    }
  }
  return results;
}

Value HALDispatchABI::getIndexValue(Location loc, int64_t value,
                                    OpBuilder &builder) {
  return builder.create<LLVM::ConstantOp>(
      loc, typeConverter->convertType(builder.getIndexType()),
      builder.getI64IntegerAttr(value));
}

Value HALDispatchABI::castValueToType(Location loc, Value value,
                                      Type resultType, OpBuilder &builder) {
  // NOTE: we should handle more cases here (and proper sign extension).
  if (value.getType() == resultType) return value;
  return builder.createOrFold<LLVM::ZExtOp>(loc, resultType, value);
}

Value HALDispatchABI::loadFieldValue(Operation *forOp, EnvironmentField field,
                                     OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto environmentPtrValue =
      buildArgDI(forOp, /*argNum=*/0, getLocalArgument(forOp, 0), "environment",
                 di.getPtrOf(di.getConstOf(di.getEnvironmentV0T())), builder);
  Value environmentValue =
      builder.create<LLVM::LoadOp>(loc, environmentType, environmentPtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return builder.create<LLVM::ExtractValueOp>(loc, environmentValue, position);
}

Value HALDispatchABI::loadFieldValue(Operation *forOp, ProcessorField field,
                                     OpBuilder &builder) {
  auto loc = forOp->getLoc();
  Value processorValue =
      loadFieldValue(forOp, EnvironmentField::processor, builder);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return builder.create<LLVM::ExtractValueOp>(loc, processorValue, position);
}

Value HALDispatchABI::loadFieldValue(Operation *forOp, DispatchStateField field,
                                     OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto statePtrValue = buildArgDI(
      forOp, /*argNum=*/1, getLocalArgument(forOp, 1), "dispatch_state",
      di.getPtrOf(di.getConstOf(di.getDispatchStateV0T())), builder);
  Value stateValue =
      builder.create<LLVM::LoadOp>(loc, dispatchStateType, statePtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return builder.create<LLVM::ExtractValueOp>(loc, stateValue, position);
}

Type HALDispatchABI::getFieldType(WorkgroupStateField field) {
  return workgroupStateType.getBody()[int64_t(field)];
}

Value HALDispatchABI::loadFieldValue(Operation *forOp,
                                     WorkgroupStateField field,
                                     OpBuilder &builder) {
  auto loc = forOp->getLoc();
  auto statePtrValue = buildArgDI(
      forOp, /*argNum=*/2, getLocalArgument(forOp, 2), "workgroup_state",
      di.getPtrOf(di.getConstOf(di.getWorkgroupStateV0T())), builder);
  Value stateValue =
      builder.create<LLVM::LoadOp>(loc, workgroupStateType, statePtrValue);
  SmallVector<int64_t, 1> position = {int64_t(field)};
  return builder.create<LLVM::ExtractValueOp>(loc, stateValue, position);
}

Type HALDispatchABI::getExtraFieldType(StringRef extraField) {
  if (extraField == "processor_id") {
    return getProcessorIDType();
  }
  if (extraField == "processor_data") {
    return getProcessorDataType();
  }
  assert(false && "unhandled extra filed");
  return {};
}

Value HALDispatchABI::getExtraField(Operation *forOp, StringRef extraField,
                                    OpBuilder &builder) {
  if (extraField == "processor_id") {
    return loadProcessorID(forOp, builder);
  } else if (extraField == "processor_data") {
    return loadProcessorData(forOp, builder);
  } else {
    assert(false && "unhandled extra field");
    return {};
  }
}

}  // namespace iree_compiler
}  // namespace mlir
