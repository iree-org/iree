// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ConstEval/Runtime.h"

#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace iree_compiler {
namespace ConstEval {

namespace {

Type mapElementType(Location loc, iree_hal_element_type_t halElementType) {
  Builder builder(loc.getContext());
  if (iree_hal_element_numerical_type_is_boolean(halElementType)) {
    return builder.getIntegerType(1);
  } else if (iree_hal_element_numerical_type_is_integer(halElementType)) {
    return builder.getIntegerType(iree_hal_element_bit_count(halElementType));
  } else if (halElementType == IREE_HAL_ELEMENT_TYPE_FLOAT_32) {
    return builder.getF32Type();
  } else if (halElementType == IREE_HAL_ELEMENT_TYPE_FLOAT_64) {
    return builder.getF64Type();
  } else if (halElementType == IREE_HAL_ELEMENT_TYPE_FLOAT_16) {
    return builder.getF16Type();
  } else if (halElementType == IREE_HAL_ELEMENT_TYPE_BFLOAT_16) {
    return builder.getBF16Type();
  }

  emitError(loc) << "unrecognized evaluated buffer view element type: "
                 << halElementType;
  return {};
}

static Attribute createAttributeFromRawData(Location loc,
                                            RankedTensorType tensorType,
                                            MutableArrayRef<char> rawBuffer) {
  Type elementType = tensorType.getElementType();
  // For numeric types that are byte-width aligned, we just use the raw buffer
  // loading support of DenseElementsAttr.
  if (elementType.isIntOrFloat() &&
      elementType.getIntOrFloatBitWidth() % 8 == 0) {
    bool detectedSplat = false;
    if (DenseElementsAttr::isValidRawBuffer(tensorType, rawBuffer,
                                            detectedSplat)) {
      return DenseElementsAttr::getFromRawBuffer(tensorType, rawBuffer);
    } else {
      emitError(loc) << "mapped memory region was not valid for constructing "
                        "tensor of type "
                     << tensorType << " (length=" << rawBuffer.size() << ")";
      return {};
    }
  }

  // For i1, IREE (currently) returns these as 8bit integer values and MLIR
  // has a loader that accepts bool arrays (the raw buffer loader also
  // supports them but bit-packed, which is not convenient for us). So, if
  // sizeof(bool) == 1, we just bit-cast. Otherwise, we go through a temporary.
  if (elementType.isInteger(1)) {
    if (sizeof(bool) == 1) {
      ArrayRef<bool> boolArray(reinterpret_cast<bool*>(rawBuffer.data()),
                               rawBuffer.size());
      return DenseElementsAttr::get(tensorType, boolArray);
    } else {
      // Note: cannot use std::vector because it specializes bool in a way
      // that is not compatible with ArrayRef.
      llvm::SmallVector<bool> boolVector(rawBuffer.begin(), rawBuffer.end());
      ArrayRef<bool> boolArray(boolVector.data(), boolVector.size());
      return DenseElementsAttr::get(tensorType, boolArray);
    }
  }

  emitError(loc) << "unhandled case when converting raw buffer of "
                 << tensorType << " to Attribute";
  return {};
}

}  // namespace

CompiledBinary::CompiledBinary() {}

CompiledBinary::~CompiledBinary() {}

void CompiledBinary::deinitialize() {
  hal_module.reset();
  main_module.reset();
  context.reset();
  device.reset();
}

LogicalResult CompiledBinary::invokeNullary(Location loc, StringRef name,
                                            ResultsCallback callback) {
  iree_vm_function_t function;
  if (auto status = iree_vm_module_lookup_function_by_name(
          main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_string_view_t{name.data(), name.size()}, &function)) {
    iree_status_ignore(status);
    return emitError(loc) << "internal error evaling constant: func '" << name
                          << "' not found";
  }

  iree::vm::ref<iree_vm_list_t> inputs;
  IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 0,
                                    iree_allocator_system(), &inputs));
  iree::vm::ref<iree_vm_list_t> outputs;
  IREE_CHECK_OK(iree_vm_list_create(/*element_type=*/nullptr, 1,
                                    iree_allocator_system(), &outputs));

  if (auto status =
          iree_vm_invoke(context.get(), function, IREE_VM_INVOCATION_FLAG_NONE,
                         /*policy=*/nullptr, inputs.get(), outputs.get(),
                         iree_allocator_system())) {
    std::string message;
    message.resize(512);
    iree_host_size_t buffer_length;
    if (!iree_status_format(status, message.size(), &message[0],
                            &buffer_length)) {
      message.resize(buffer_length + 1);
      iree_status_format(status, message.size(), &message[0], &buffer_length);
    }
    message.resize(buffer_length);
    iree_status_ignore(status);
    return emitError(loc) << "internal error evaling constant: " << message;
  }

  if (failed(callback(outputs.get()))) {
    return failure();
  }
  return success();
}

Attribute CompiledBinary::invokeNullaryAsAttribute(Location loc,
                                                   StringRef name) {
  Attribute result;
  if (failed(invokeNullary(
          loc, name, [&](iree_vm_list_t* outputs) -> LogicalResult {
            if (iree_vm_list_size(outputs) != 1) {
              return emitError(loc) << "expected 1 result for func " << name
                                    << " got " << iree_vm_list_size(outputs);
            }
            iree_vm_variant_t variant = iree_vm_variant_empty();
            IREE_CHECK_OK(
                iree_vm_list_get_variant_assign(outputs, 0, &variant));
            result = convertVariantToAttribute(loc, variant);
            return success(result != nullptr);
          }))) {
    return nullptr;
  }

  return result;
}

bool CompiledBinary::isSupportedResultType(Type type) {
  // TODO(laurenzo): Not currently supported. VMVX would need to support these
  // and today it doesn't. We could use alternative backends (LLVM CPU/etc) if
  // we wanted to handle f64, but f16 and bf16 often need special hardware.
  if (type.isa<Float16Type>() || type.isa<BFloat16Type>() ||
      type.isa<Float64Type>()) {
    return false;
  }

  // Support scalar int and float type of byte aligned widths.
  if (type.isIntOrFloat() && type.getIntOrFloatBitWidth() % 8 == 0) {
    return true;
  }

  // Special support for i1.
  if (type.isa<IntegerType>() && type.getIntOrFloatBitWidth() == 1) {
    return true;
  }

  // Support tensors.
  if (auto tt = type.dyn_cast<RankedTensorType>()) {
    return isSupportedResultType(tt.getElementType());
  }

  return false;
}

Attribute CompiledBinary::convertVariantToAttribute(
    Location loc, iree_vm_variant_t& variant) {
  auto context = loc.getContext();
  Builder builder(context);
  if (iree_vm_variant_is_value(variant)) {
    switch (variant.type.value_type) {
      case IREE_VM_VALUE_TYPE_I8:
        return builder.getI8IntegerAttr(variant.i8);
      case IREE_VM_VALUE_TYPE_I16:
        return builder.getI16IntegerAttr(variant.i16);
      case IREE_VM_VALUE_TYPE_I32:
        return builder.getI32IntegerAttr(variant.i32);
      case IREE_VM_VALUE_TYPE_I64:
        return builder.getI64IntegerAttr(variant.i64);
      case IREE_VM_VALUE_TYPE_F32:
        return builder.getF32FloatAttr(variant.f32);
      case IREE_VM_VALUE_TYPE_F64:
        return builder.getF64FloatAttr(variant.f64);
      default:
        emitError(loc) << "unrecognized evaluated value type: "
                       << static_cast<int>(variant.type.value_type);
        return {};
    }
  }

  if (iree_vm_variant_is_ref(variant)) {
    if (iree_hal_buffer_view_isa(variant.ref)) {
      iree_hal_buffer_view_t* bufferView =
          iree_hal_buffer_view_deref(variant.ref);

      // Get the shape.
      int rank = iree_hal_buffer_view_shape_rank(bufferView);
      SmallVector<int64_t> shape(rank);
      for (int i = 0; i < rank; ++i) {
        shape[i] = iree_hal_buffer_view_shape_dim(bufferView, i);
      }

      // Map the element type.
      iree_hal_element_type_t halElementType =
          iree_hal_buffer_view_element_type(bufferView);
      Type elementType = mapElementType(loc, halElementType);
      if (!elementType) return {};

      auto tensorType = RankedTensorType::get(shape, elementType);

      auto length = iree_hal_buffer_view_byte_length(bufferView);
      iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(bufferView);

      // Map the memory and construct.
      // TODO(benvanik): fallback to alloc + iree_hal_device_transfer_range if
      // mapping is not available. Today with the CPU backends it's always
      // possible but would not work with accelerators.
      iree_hal_buffer_mapping_t mapping;
      IREE_CHECK_OK(iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          /*byte_offset=*/0, length, &mapping));
      MutableArrayRef<char> rawBufferArray(
          reinterpret_cast<char*>(mapping.contents.data),
          mapping.contents.data_length);
      auto convertedAttr =
          createAttributeFromRawData(loc, tensorType, rawBufferArray);
      iree_hal_buffer_unmap_range(&mapping);
      return convertedAttr;
    } else {
      iree_string_view_t typeName =
          iree_vm_ref_type_name(variant.type.ref_type);
      emitError(loc) << "unrecognized evaluated ref type: "
                     << StringRef(typeName.data, typeName.size);
      return {};
    }
  }

  emitError(loc) << "unrecognized evaluated variant type";
  return {};
}

void CompiledBinary::initialize(void* data, size_t length) {
  Runtime& runtime = Runtime::getInstance();

  // Create driver and device.
  iree_hal_driver_t* driver = nullptr;
  IREE_CHECK_OK(iree_hal_driver_registry_try_create(
      runtime.registry, iree_make_cstring_view("local-task"),
      iree_allocator_system(), &driver));
  IREE_CHECK_OK(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &device));
  iree_hal_driver_release(driver);

  // Create hal module.
  IREE_CHECK_OK(iree_hal_module_create(runtime.instance.get(), device.get(),
                                       IREE_HAL_MODULE_FLAG_NONE,
                                       iree_allocator_system(), &hal_module));

  // Bytecode module.
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      runtime.instance.get(), iree_make_const_byte_span(data, length),
      iree_allocator_null(), iree_allocator_system(), &main_module));

  // Context.
  std::array<iree_vm_module_t*, 2> modules = {
      hal_module.get(),
      main_module.get(),
  };
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      runtime.instance.get(), IREE_VM_CONTEXT_FLAG_NONE, modules.size(),
      modules.data(), iree_allocator_system(), &context));
}

InMemoryCompiledBinary::~InMemoryCompiledBinary() { deinitialize(); }

LogicalResult InMemoryCompiledBinary::translateFromModule(
    mlir::ModuleOp moduleOp) {
  llvm::raw_string_ostream os(binary);
  iree_compiler::IREE::VM::TargetOptions vmOptions;
  iree_compiler::IREE::VM::BytecodeTargetOptions bytecodeOptions;
  if (failed(iree_compiler::IREE::VM::translateModuleToBytecode(
          moduleOp, vmOptions, bytecodeOptions, os))) {
    return failure();
  }
  os.flush();
  initialize(&binary[0], binary.length());
  return success();
}

Runtime::Runtime() {
  IREE_CHECK_OK(
      iree_hal_driver_registry_allocate(iree_allocator_system(), &registry));
  IREE_CHECK_OK(iree_hal_local_task_driver_module_register(registry));
  IREE_CHECK_OK(iree_vm_instance_create(iree_allocator_system(), &instance));
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance.get()));
}

Runtime::~Runtime() {
  instance.reset();
  iree_hal_driver_registry_free(registry);
}

Runtime& Runtime::getInstance() {
  static Runtime instance;
  return instance;
}

}  // namespace ConstEval
}  // namespace iree_compiler
}  // namespace mlir
