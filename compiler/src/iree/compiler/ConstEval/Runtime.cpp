// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/ConstEval/Runtime.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/BytecodeModuleTarget.h"
#include "iree/hal/drivers/local_task/registration/driver_module.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "iree-const-eval"
using llvm::dbgs;

namespace mlir::iree_compiler::ConstEval {

namespace {

LogicalResult handleRuntimeError(Location loc, iree_status_t status) {
  if (iree_status_is_ok(status))
    return success();
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
  return emitError(loc) << "runtime error in consteval: " << message;
}

LogicalResult convertToElementType(Location loc, Type baseType,
                                   iree_hal_element_type_t *outElementType) {
  Builder builder(loc.getContext());
  if (auto t = llvm::dyn_cast<IntegerType>(baseType)) {
    switch (t.getWidth()) {
    case 32:
      *outElementType = IREE_HAL_ELEMENT_TYPE_INT_32;
      return success();
    case 64:
      *outElementType = IREE_HAL_ELEMENT_TYPE_INT_64;
      return success();
    case 8:
      *outElementType = IREE_HAL_ELEMENT_TYPE_INT_8;
      return success();
    case 16:
      *outElementType = IREE_HAL_ELEMENT_TYPE_INT_16;
      return success();
    case 4:
      *outElementType = IREE_HAL_ELEMENT_TYPE_INT_4;
      return success();
    case 1:
      *outElementType = IREE_HAL_ELEMENT_TYPE_BOOL_8;
      return success();
    }
  } else if (baseType == builder.getF32Type()) {
    *outElementType = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
    return success();
  } else if (baseType == builder.getF64Type()) {
    *outElementType = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
    return success();
  } else if (baseType == builder.getF16Type()) {
    *outElementType = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
    return success();
  } else if (baseType == builder.getBF16Type()) {
    *outElementType = IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
    return success();
  }

  return emitError(loc)
         << "internal error: unhandled element type in consteval: " << baseType;
}

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

static TypedAttr createAttributeFromRawData(Location loc,
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
  // supports them but bit-packed, which is not convenient for us).
  if (elementType.isInteger(1)) {
    // Note: cannot use std::vector because it specializes bool in a way
    // that is not compatible with ArrayRef.
    llvm::SmallVector<bool> boolVector(rawBuffer.begin(), rawBuffer.end());
    ArrayRef<bool> boolArray(boolVector.data(), boolVector.size());
    return DenseElementsAttr::get(tensorType, boolArray);
  }

  emitError(loc) << "unhandled case when converting raw buffer of "
                 << tensorType << " to Attribute";
  return {};
}

} // namespace

CompiledBinary::CompiledBinary() = default;

CompiledBinary::~CompiledBinary() = default;

void CompiledBinary::deinitialize() {
  hal_module.reset();
  main_module.reset();
  context.reset();
  device.reset();
}

FunctionCall::FunctionCall(CompiledBinary &binary, iree_host_size_t argCapacity,
                           iree_host_size_t resultCapacity)
    : binary(binary) {
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    argCapacity, iree_allocator_system(),
                                    &inputs));
  IREE_CHECK_OK(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                    resultCapacity, iree_allocator_system(),
                                    &outputs));
}

FailureOr<iree::vm::ref<iree_hal_buffer_t>>
FunctionCall::importSerializableAttr(
    Location loc, IREE::Util::SerializableAttrInterface serializableAttr) {

  // Allocate buffer.
  int64_t storageSize = serializableAttr.getStorageSize();
  if (storageSize < 0) {
    return emitError(loc) << "unsupported serializable attribute: "
                          << serializableAttr;
  }

  iree::vm::ref<iree_hal_buffer_t> buffer;
  iree_hal_buffer_params_t params;
  std::memset(&params, 0, sizeof(params));
  params.type =
      IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (failed(handleRuntimeError(
          loc, iree_hal_allocator_allocate_buffer(binary.getAllocator(), params,
                                                  storageSize, &buffer))))
    return failure();

  iree_hal_buffer_mapping_t mapping;
  if (failed(handleRuntimeError(
          loc, iree_hal_buffer_map_range(
                   buffer.get(), IREE_HAL_MAPPING_MODE_SCOPED,
                   IREE_HAL_MEMORY_ACCESS_WRITE, /*byte_offset=*/0,
                   /*byte_length=*/storageSize, &mapping))))
    return failure();

  // Copy.
  LogicalResult copyResult = serializableAttr.serializeToBuffer(
      loc, llvm::endianness::native,
      ArrayRef<char>(reinterpret_cast<char *>(mapping.contents.data),
                     storageSize));

  if (failed(handleRuntimeError(loc, iree_hal_buffer_unmap_range(&mapping))) ||
      failed(copyResult)) {
    return failure();
  }

  return buffer;
}

LogicalResult FunctionCall::addBufferArgumentAttr(
    Location loc, IREE::Util::SerializableAttrInterface serializableAttr) {
  auto buffer = importSerializableAttr(loc, serializableAttr);
  if (failed(buffer))
    return failure();
  return handleRuntimeError(
      loc, iree_vm_list_push_ref_move(inputs.get(), std::move(*buffer)));
}

LogicalResult FunctionCall::addBufferViewArgumentAttr(
    Location loc, ShapedType shapedType,
    IREE::Util::SerializableAttrInterface serializableAttr) {
  // Metadata taken from the elements attr type.
  auto rank = static_cast<size_t>(shapedType.getRank());
  iree_hal_dim_t *shape =
      static_cast<iree_hal_dim_t *>(alloca(rank * sizeof(iree_hal_dim_t)));
  for (size_t i = 0; i < rank; ++i) {
    shape[i] = shapedType.getDimSize(i);
  }
  iree_hal_element_type_t elementType = IREE_HAL_ELEMENT_TYPE_NONE;
  if (failed(
          convertToElementType(loc, shapedType.getElementType(), &elementType)))
    return failure();

  // Import buffer contents.
  auto buffer = importSerializableAttr(loc, serializableAttr);
  if (failed(buffer))
    return failure();

  // Construct buffer view.
  iree::vm::ref<iree_hal_buffer_view_t> bufferView;
  if (failed(handleRuntimeError(
          loc,
          iree_hal_buffer_view_create(buffer->get(), rank, shape, elementType,
                                      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
                                      iree_allocator_system(), &bufferView))))
    return failure();

  return handleRuntimeError(
      loc, iree_vm_list_push_ref_move(inputs.get(), std::move(bufferView)));
}

static ShapedType getAttrShapedType(Attribute attr) {
  if (auto typedAttr = llvm::dyn_cast<TypedAttr>(attr)) {
    if (auto shapedType = llvm::dyn_cast<ShapedType>(typedAttr.getType())) {
      return shapedType;
    }
  }
  return {};
}

LogicalResult FunctionCall::addArgument(Location loc, Attribute attr) {
  if (auto elementsAttr = llvm::dyn_cast<ElementsAttr>(attr)) {
    return addBufferViewArgumentAttr(
        loc, elementsAttr.getShapedType(),
        cast<IREE::Util::SerializableAttrInterface>(attr));
  } else if (auto serializableAttr =
                 llvm::dyn_cast<IREE::Util::SerializableAttrInterface>(attr)) {
    if (auto shapedType = getAttrShapedType(attr)) {
      return addBufferViewArgumentAttr(loc, shapedType, serializableAttr);
    } else {
      return addBufferArgumentAttr(loc, serializableAttr);
    }
  } else if (auto integerAttr = llvm::dyn_cast<IntegerAttr>(attr)) {
    iree_vm_value_t value;
    APInt apValue = integerAttr.getValue();
    switch (apValue.getBitWidth()) {
    case 8:
      value =
          iree_vm_value_make_i8(static_cast<uint8_t>(apValue.getZExtValue()));
      break;
    case 16:
      value =
          iree_vm_value_make_i16(static_cast<uint16_t>(apValue.getZExtValue()));
      break;
    case 32:
      value =
          iree_vm_value_make_i32(static_cast<uint32_t>(apValue.getZExtValue()));
      break;
    case 64:
      value =
          iree_vm_value_make_i64(static_cast<uint64_t>(apValue.getZExtValue()));
      break;
    default:
      return emitError(loc) << "internal error: unsupported consteval jit "
                               "function integer input type ("
                            << attr << ")";
    }
    return handleRuntimeError(loc,
                              iree_vm_list_push_value(inputs.get(), &value));
  } else if (auto floatAttr = llvm::dyn_cast<FloatAttr>(attr)) {
    iree_vm_value_t value;
    APFloat apValue = floatAttr.getValue();
    // Note that there are many floating point semantics that LLVM knows
    // about, but we restrict to only those that the VM natively supports
    // here.
    switch (APFloat::SemanticsToEnum(apValue.getSemantics())) {
    case APFloat::S_IEEEsingle:
      value = iree_vm_value_make_f32(apValue.convertToFloat());
      break;
    case APFloat::S_IEEEdouble:
      value = iree_vm_value_make_f64(apValue.convertToDouble());
      break;
    default:
      return emitError(loc) << "internal error: unsupported consteval jit "
                               "function float input type ("
                            << attr << ")";
    }
    return handleRuntimeError(loc,
                              iree_vm_list_push_value(inputs.get(), &value));
  }

  return emitError(loc)
         << "internal error: unsupported consteval jit function input (" << attr
         << ")";
}

LogicalResult FunctionCall::invoke(Location loc, StringRef name) {
  // Lookup function.
  iree_vm_function_t function;
  if (auto status = iree_vm_module_lookup_function_by_name(
          binary.main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
          iree_string_view_t{name.data(),
                             static_cast<iree_host_size_t>(name.size())},
          &function)) {
    iree_status_ignore(status);
    return emitError(loc) << "internal error evaling constant: func '" << name
                          << "' not found";
  }

  return handleRuntimeError(loc, iree_vm_invoke(binary.context.get(), function,
                                                IREE_VM_INVOCATION_FLAG_NONE,
                                                /*policy=*/nullptr,
                                                inputs.get(), outputs.get(),
                                                iree_allocator_system()));
}

LogicalResult FunctionCall::getResultAsAttr(Location loc, size_t index,
                                            Type mlirType, TypedAttr &outAttr) {
  iree_vm_variant_t variant = iree_vm_variant_empty();
  if (failed(handleRuntimeError(loc, iree_vm_list_get_variant_assign(
                                         outputs.get(), index, &variant))))
    return failure();

  outAttr = binary.convertVariantToAttribute(loc, variant, mlirType);
  if (!outAttr)
    return failure();

  return success();
}

TypedAttr CompiledBinary::convertVariantToAttribute(Location loc,
                                                    iree_vm_variant_t &variant,
                                                    Type mlirType) {
  auto context = loc.getContext();
  Builder builder(context);
  if (iree_vm_variant_is_value(variant)) {
    switch (iree_vm_type_def_as_value(variant.type)) {
    case IREE_VM_VALUE_TYPE_I32:
      return builder.getIntegerAttr(mlirType, variant.i32);
    case IREE_VM_VALUE_TYPE_I64:
      return builder.getIntegerAttr(mlirType, variant.i64);
    case IREE_VM_VALUE_TYPE_F32:
      return builder.getF32FloatAttr(variant.f32);
    case IREE_VM_VALUE_TYPE_F64:
      return builder.getF64FloatAttr(variant.f64);
    default:
      emitError(loc) << "unrecognized evaluated value type: "
                     << static_cast<int>(
                            iree_vm_type_def_as_value(variant.type));
      return {};
    }
  }

  if (iree_vm_variant_is_ref(variant)) {
    if (iree_hal_buffer_view_isa(variant.ref)) {
      iree_hal_buffer_view_t *bufferView =
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
      if (!elementType)
        return {};

      auto tensorType = RankedTensorType::get(shape, elementType);

      auto length = iree_hal_buffer_view_byte_length(bufferView);
      iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(bufferView);

      // Map the memory and construct.
      // TODO(benvanik): fallback to alloc + iree_hal_device_transfer_range if
      // mapping is not available. Today with the CPU backends it's always
      // possible but would not work with accelerators.
      iree_hal_buffer_mapping_t mapping;
      IREE_CHECK_OK(iree_hal_buffer_map_range(
          buffer, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_READ,
          /*byte_offset=*/0, length, &mapping));
      MutableArrayRef<char> rawBufferArray(
          reinterpret_cast<char *>(mapping.contents.data),
          mapping.contents.data_length);
      auto convertedAttr =
          createAttributeFromRawData(loc, tensorType, rawBufferArray);
      iree_hal_buffer_unmap_range(&mapping);
      return convertedAttr;
    } else {
      iree_string_view_t typeName =
          iree_vm_ref_type_name(iree_vm_type_def_as_ref(variant.type));
      emitError(loc) << "unrecognized evaluated ref type: "
                     << StringRef(typeName.data, typeName.size);
      return {};
    }
  }

  emitError(loc) << "unrecognized evaluated variant type";
  return {};
}

void CompiledBinary::initialize(void *data, size_t length) {
  Runtime &runtime = Runtime::getInstance();

  // Create driver and device.
  iree_hal_driver_t *driver = nullptr;
  IREE_CHECK_OK(iree_hal_driver_registry_try_create(
      runtime.registry, iree_make_cstring_view("local-task"),
      iree_allocator_system(), &driver));
  IREE_CHECK_OK(iree_hal_driver_create_default_device(
      driver, iree_allocator_system(), &device));
  iree_hal_driver_release(driver);

  // Create hal module.
  iree_hal_device_t *device_ptr = device.get();
  IREE_CHECK_OK(iree_hal_module_create(
      runtime.instance.get(), /*device_count=*/1, &device_ptr,
      IREE_HAL_MODULE_FLAG_NONE, iree_allocator_system(), &hal_module));

  // Bytecode module.
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      runtime.instance.get(), iree_make_const_byte_span(data, length),
      iree_allocator_null(), iree_allocator_system(), &main_module));

  // Context.
  std::array<iree_vm_module_t *, 2> modules = {
      hal_module.get(),
      main_module.get(),
  };
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      runtime.instance.get(), IREE_VM_CONTEXT_FLAG_NONE, modules.size(),
      modules.data(), iree_allocator_system(), &context));
}

InMemoryCompiledBinary::~InMemoryCompiledBinary() { deinitialize(); }

LogicalResult
InMemoryCompiledBinary::translateFromModule(mlir::ModuleOp moduleOp) {
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
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        iree_allocator_system(), &instance));
  IREE_CHECK_OK(iree_hal_module_register_all_types(instance.get()));
}

Runtime::~Runtime() {
  instance.reset();
  iree_hal_driver_registry_free(registry);
}

Runtime &Runtime::getInstance() {
  static Runtime instance;
  return instance;
}

} // namespace mlir::iree_compiler::ConstEval
