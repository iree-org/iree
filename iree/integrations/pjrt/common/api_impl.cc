// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/api_impl.h"

#include <iostream>
#include <optional>

#include "iree/base/tracing.h"
#include "iree/hal/api.h"

using iree::vm::retain_ref;

namespace iree::pjrt {

// Chopped down utilities from various TPU support libraries. Basically all for
// populating Trimmed device shapes. Since that is supposed to go away at
// some point, just copy-pasta here.
namespace ApiConverter {
// Helper functions for copying data to possibly-inlined C arrays.

// 'Src' and 'Dst' are allowed to be different types to make this usable with
// memory-identical types, e.g. int64_t and int64_t. This should not be used
// with types that require a static_cast.
template <typename Src, typename Dst, typename DstList>
static void CreateVectorBase(const absl::Span<Src> src, DstList* dst) {
  dst->size = src.size();
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new Dst[dst->size];
    std::copy(src.begin(), src.end(), dst->heap);
  } else {
    std::copy(src.begin(), src.end(), dst->inlined);
  }
}

void CreateVector(const absl::Span<const int64_t> src, Int64List* dst) {
  return CreateVectorBase<const int64_t, int64_t, Int64List>(src, dst);
}

void CreateVector(const absl::Span<const bool> src, BoolList* dst) {
  return CreateVectorBase<const bool, bool, BoolList>(src, dst);
}

static void CreateVector(const absl::Span<const bool> src, IntList* dst) {
  CreateVectorBase<const bool, int, IntList>(src, dst);
}

static void CreateVector(const absl::Span<const xla::DimLevelType> src,
                         IntList* dst) {
  CreateVectorBase<const xla::DimLevelType, int, IntList>(src, dst);
}

void ToC(const xla::Tile& tile, XLA_Tile* c_tile) {
  CreateVector(tile.dimensions(), &c_tile->dimensions);
}

static void CreateVector(const absl::Span<const xla::Tile> src, TileList* dst) {
  dst->size = src.size();
  XLA_Tile* c_tiles;
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new XLA_Tile[dst->size];
    c_tiles = dst->heap;
  } else {
    c_tiles = dst->inlined;
  }
  for (int i = 0; i < dst->size; ++i) {
    ToC(src[i], &c_tiles[i]);
  }
}

void ToC(const xla::Layout& layout, XLA_Layout* c_layout) {
  CreateVector(layout.minor_to_major(), &c_layout->minor_to_major);
  CreateVector(layout.dim_level_types(), &c_layout->dim_level_types);
  CreateVector(layout.dim_unique(), &c_layout->dim_unique);
  CreateVector(layout.dim_ordered(), &c_layout->dim_ordered);
  c_layout->index_primitive_type = layout.index_primitive_type();
  c_layout->pointer_primitive_type = layout.pointer_primitive_type();
  c_layout->memory_space = layout.memory_space();
  CreateVector(layout.tiles(), &c_layout->tiles);
}

}  // namespace ApiConverter

namespace {

iree_status_t MapElementTypeToXlaElementType(
    iree_hal_element_type_t element_type, xla::PrimitiveType* xla_primitive) {
  // TODO: Cascade on bit-field sub-types to avoid large linear scan.
  switch (element_type) {
    // TODO: How do I interpret signless?
    case IREE_HAL_ELEMENT_TYPE_BOOL_8:
      *xla_primitive = xla::PrimitiveType::PRED;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      *xla_primitive = xla::PrimitiveType::U8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      *xla_primitive = xla::PrimitiveType::U16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      *xla_primitive = xla::PrimitiveType::U64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *xla_primitive = xla::PrimitiveType::S8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      *xla_primitive = xla::PrimitiveType::S16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      *xla_primitive = xla::PrimitiveType::S32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      *xla_primitive = xla::PrimitiveType::S64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *xla_primitive = xla::PrimitiveType::U8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *xla_primitive = xla::PrimitiveType::U16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *xla_primitive = xla::PrimitiveType::U64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *xla_primitive = xla::PrimitiveType::F16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *xla_primitive = xla::PrimitiveType::F64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *xla_primitive = xla::PrimitiveType::BF16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      *xla_primitive = xla::PrimitiveType::C64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      *xla_primitive = xla::PrimitiveType::C128;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown element type 0x%x",
                              (int)element_type);
  }
}

iree_status_t MapBufferTypeToElementType(
    PJRT_Buffer_Type buffer_type, iree_hal_element_type_t* element_type) {
  switch (buffer_type) {
    case PJRT_Buffer_Type_INVALID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    case PJRT_Buffer_Type_PRED:
      *element_type = IREE_HAL_ELEMENT_TYPE_BOOL_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_S8:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_S16:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_S32:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_S64:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_U8:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_U16:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_U32:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_U64:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_F16:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_F32:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_F64:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_BF16:
      *element_type = IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_C64:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_C128:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown buffer type %d",
                              (int)buffer_type);
  }
}

}  // namespace

//===----------------------------------------------------------------------===//
// Error
//===----------------------------------------------------------------------===//

void ErrorInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args* args) {
    if (!args->error) return;
    delete ErrorInstance::FromError(args->error);
  };
  api->PJRT_Error_Message = +[](PJRT_Error_Message_Args* args) {
    auto* error = ErrorInstance::FromError(args->error);
    if (!error) {
      args->message = "OK";
      args->message_size = 2;
      return;
    }

    const std::string& message = error->message();
    args->message = message.data();
    args->message_size = message.size();
  };
  api->PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args* args) -> PJRT_Error* {
    auto* error = ErrorInstance::FromError(args->error);
    iree_status_code_t status_code = iree_status_code(error->status());
    switch (status_code) {
      case IREE_STATUS_CANCELLED:
        args->code = PJRT_Error_Code_CANCELLED;
        break;
      case IREE_STATUS_UNKNOWN:
        args->code = PJRT_Error_Code_UNKNOWN;
        break;
      case IREE_STATUS_INVALID_ARGUMENT:
        args->code = PJRT_Error_Code_INVALID_ARGUMENT;
        break;
      case IREE_STATUS_DEADLINE_EXCEEDED:
        args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
        break;
      case IREE_STATUS_NOT_FOUND:
        args->code = PJRT_Error_Code_NOT_FOUND;
        break;
      case IREE_STATUS_ALREADY_EXISTS:
        args->code = PJRT_Error_Code_ALREADY_EXISTS;
        break;
      case IREE_STATUS_PERMISSION_DENIED:
        args->code = PJRT_Error_Code_PERMISSION_DENIED;
        break;
      case IREE_STATUS_RESOURCE_EXHAUSTED:
        args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
        break;
      case IREE_STATUS_FAILED_PRECONDITION:
        args->code = PJRT_Error_Code_FAILED_PRECONDITION;
        break;
      case IREE_STATUS_ABORTED:
        args->code = PJRT_Error_Code_ABORTED;
        break;
      case IREE_STATUS_OUT_OF_RANGE:
        args->code = PJRT_Error_Code_OUT_OF_RANGE;
        break;
      case IREE_STATUS_UNIMPLEMENTED:
        args->code = PJRT_Error_Code_UNIMPLEMENTED;
        break;
      case IREE_STATUS_INTERNAL:
        args->code = PJRT_Error_Code_INTERNAL;
        break;
      case IREE_STATUS_UNAVAILABLE:
        args->code = PJRT_Error_Code_UNAVAILABLE;
        break;
      case IREE_STATUS_DATA_LOSS:
        args->code = PJRT_Error_Code_DATA_LOSS;
        break;
      case IREE_STATUS_UNAUTHENTICATED:
        args->code = PJRT_Error_Code_UNAUTHENTICATED;
        break;
      case IREE_STATUS_DEFERRED:
        args->code = PJRT_Error_Code_UNKNOWN;  // No mapping
        break;
      default:
        // Should not happen.
        args->code = PJRT_Error_Code_UNKNOWN;
    }
    return nullptr;
  };
}

const std::string& ErrorInstance::message() const {
  if (cached_message_.empty()) {
    std::string buffer;
    iree_host_size_t actual_len;
    buffer.resize(1024);  // TODO: Actually reallocate to full size on trunc.
    if (!iree_status_format(status_, buffer.size(), buffer.data(),
                            &actual_len)) {
      buffer.resize(actual_len);
      if (!iree_status_format(status_, buffer.size(), buffer.data(),
                              &actual_len)) {
        actual_len = 0;
      }
    }
    buffer.resize(actual_len);
    cached_message_ = std::move(buffer);
  }
  return cached_message_;
}

//===----------------------------------------------------------------------===//
// BufferInstance
//===----------------------------------------------------------------------===//

BufferInstance::~BufferInstance() = default;

iree_status_t BufferInstance::GetXlaShape(xla::Shape** out_shape) {
  if (cached_shape_) {
    *out_shape = &(*cached_shape_);
    return iree_ok_status();
  }

  iree_hal_element_type_t hal_element_type =
      iree_hal_buffer_view_element_type(buffer_view());
  xla::PrimitiveType xla_element_type;
  IREE_RETURN_IF_ERROR(
      MapElementTypeToXlaElementType(hal_element_type, &xla_element_type));

  size_t rank = iree_hal_buffer_view_shape_rank(buffer_view());
  const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(buffer_view());
  std::array<int64_t, 9> xla_dims;
  if (rank > xla_dims.size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rank > 9 not supported");
  }
  for (size_t i = 0; i < rank; ++i) {
    xla_dims[i] = dims[i];
  }

  cached_shape_ = xla::ShapeUtil::MakeShape(
      xla_element_type,
      absl::MakeSpan(xla_dims.begin(), xla_dims.begin() + rank));
  *out_shape = &(*cached_shape_);
  return iree_ok_status();
}

BufferInstance::BufferInstance(
    DeviceInstance& device, iree::vm::ref<iree_hal_buffer_view_t> buffer_view)
    : device_(device), buffer_view_(std::move(buffer_view)) {
  IREE_CHECK_OK(device.CreateFence(&ready_fence_));
  IREE_CHECK_OK(device.CreateFence(&done_fence_));
}

void BufferInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Buffer_Destroy =
      +[](PJRT_Buffer_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_Destroy");
    iree_status_t status =
        BufferInstance::Unwrap(args->buffer)->AsyncDeallocate();
    delete BufferInstance::Unwrap(args->buffer);
    return MakeError(status);
  };
  api->PJRT_Buffer_OnDeviceTrimmedShape =
      +[](PJRT_Buffer_OnDeviceTrimmedShape_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_OnDeviceTrimmedShape");
    auto impl = [&]() -> iree_status_t {
      // TODO: This function is terrible and not exposed properly to C.
      // It is slated to be deleted...
      // See Google bug b/238999986
      BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
      xla::Shape* shape;
      IREE_RETURN_IF_ERROR(buffer->GetXlaShape(&shape));

      args->element_type = shape->element_type();
      ApiConverter::CreateVector(shape->dimensions(), &args->dimensions);
      ApiConverter::CreateVector(shape->dynamic_dimensions(),
                                 &args->dynamic_dimensions);

      if (shape->has_layout()) {
        args->has_layout = true;
        ApiConverter::ToC(shape->layout(), &args->layout);
      } else {
        args->has_layout = false;
      }
      return iree_ok_status();
    };
    return MakeError(impl());
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_ToHostBuffer");
    BufferInstance* buffer = BufferInstance::Unwrap(args->src);
    if (!args->dst) {
      // Size query.
      return MakeError(buffer->GetHostSizeInBytes(&args->dst_size));
    } else {
      // Initiate transfer.
      return MakeError(
          buffer->CopyToHost(args->dst, args->dst_size,
                             reinterpret_cast<EventInstance**>(&args->event)));
    }
  };
  api->PJRT_Buffer_OnDeviceSizeInBytes =
      +[](PJRT_Buffer_OnDeviceSizeInBytes_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_OnDeviceSizeInBytes");
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_OnDeviceSizeInBytes"));
  };
  api->PJRT_Buffer_Delete = +[](PJRT_Buffer_Delete_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_Delete");
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_Delete"));
  };
  api->PJRT_Buffer_IsDeleted =
      +[](PJRT_Buffer_IsDeleted_Args* args) -> PJRT_Error* {
    args->is_deleted = BufferInstance::Unwrap(args->buffer)->is_deleted();
    return nullptr;
  };
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_CopyToDevice");
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_CopyToDevice"));
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args* args) -> PJRT_Error* {
    args->is_on_cpu = BufferInstance::Unwrap(args->buffer)->is_on_cpu();
    return nullptr;
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args* args) -> PJRT_Error* {
    args->device = BufferInstance::Unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_ReadyEvent =
      +[](PJRT_Buffer_ReadyEvent_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Buffer_ReadyEvent");
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_ReadyEvent"));
  };
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_UnsafePointer"));
  };
}

iree_status_t BufferInstance::GetHostSizeInBytes(iree_host_size_t* host_size) {
  *host_size = iree_hal_buffer_view_byte_length(buffer_view());
  return iree_ok_status();
}

iree_status_t BufferInstance::AsyncDeallocate() {
  IREE_TRACE_SCOPE();
  return iree_hal_device_queue_dealloca(
      device().device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_fence_semaphore_list(done_fence()),
      /*signal_semaphore_list=*/iree_hal_semaphore_list_empty(),
      iree_hal_buffer_view_buffer(buffer_view_.get()));
  return iree_ok_status();
}

iree_status_t BufferInstance::CopyToHost(void* dst, iree_host_size_t dst_size,
                                         EventInstance** out_done_event) {
  // Set up an event for external triggering. While a little wonky, we
  // trigger it in the host buffer release callback, which happens once the
  // transfer is done. I don't love this option but it seems to match what
  // I'm looking for.
  EventInstance* capture_done_event =
      new EventInstance(EventInstance::Type::EXTERNAL);
  *out_done_event = capture_done_event;

  // Import the destination (host) buffer as an iree_hal_buffer_t so that we
  // can issue copy commands.
  iree::vm::ref<iree_hal_buffer_t> dst_buffer;
  iree_hal_buffer_params_t dst_buffer_params = {
      /*usage=*/IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET,
      // TODO: We should be able to use WRITE access here since the buffer
      // is never actually mapped to read back out (just accessed through the
      // void* later). However, that seems to cause the memory to never be
      // committed and the interaction aborted.
      /*access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL |
          IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
  };
  iree_hal_external_buffer_t dst_external_buffer;
  memset(&dst_external_buffer, 0, sizeof(dst_external_buffer));
  dst_external_buffer.type = IREE_HAL_EXTERNAL_BUFFER_TYPE_HOST_ALLOCATION;
  dst_external_buffer.flags = IREE_HAL_EXTERNAL_BUFFER_FLAG_NONE;
  dst_external_buffer.size = dst_size;
  dst_external_buffer.handle.host_allocation.ptr = dst;
  auto release_callback = +[](void* user_data, iree_hal_buffer_t* buffer) {
    IREE_TRACE_SCOPE0("PJRT_CopyToHost_ReleaseCallback");
    auto* local_done_event = static_cast<EventInstance*>(user_data);
    local_done_event->ExternalSignalReady(iree_ok_status());
  };
  IREE_RETURN_IF_ERROR(iree_hal_allocator_import_buffer(
      device_.device_allocator(), dst_buffer_params, &dst_external_buffer,
      /*release_callback=*/{release_callback, capture_done_event},
      &dst_buffer));

  // Create the transfer command buffer.
  iree::vm::ref<iree_hal_command_buffer_t> transfer_cb;
  iree_hal_transfer_command_t transfer_command;
  memset(&transfer_command, 0, sizeof(transfer_command));
  transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY;
  transfer_command.copy.source_buffer =
      iree_hal_buffer_view_buffer(buffer_view());
  transfer_command.copy.source_offset = 0;
  transfer_command.copy.target_buffer = dst_buffer.get();
  transfer_command.copy.target_offset = 0;
  transfer_command.copy.length = dst_size;
  IREE_RETURN_IF_ERROR(iree_hal_create_transfer_command_buffer(
      device_.device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*transfer_count=*/1, &transfer_command, &transfer_cb));
  dst_buffer.reset();

  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device_.device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/iree_hal_fence_semaphore_list(ready_fence_.get()),
      /*signal_semaphore_list=*/iree_hal_semaphore_list_empty(),
      /*command_buffer_count=*/1, &transfer_cb));

  return iree_ok_status();
}

iree_status_t BufferInstance::AdvanceReadyFence(iree_hal_semaphore_t* semaphore,
                                                uint64_t timepoint) {
  return iree_hal_fence_insert(ready_fence_.get(), semaphore, timepoint);
}

iree_status_t BufferInstance::AdvanceDoneFence(iree_hal_semaphore_t* semaphore,
                                               uint64_t timepoint) {
  return iree_hal_fence_insert(done_fence_.get(), semaphore, timepoint);
}

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

DeviceInstance::~DeviceInstance() = default;

void DeviceInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Device_Id = +[](PJRT_Device_Id_Args* args) -> PJRT_Error* {
    args->id = DeviceInstance::Unwrap(args->device)->client_id();
    return nullptr;
  };
  api->PJRT_Device_ProcessIndex =
      +[](PJRT_Device_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index = DeviceInstance::Unwrap(args->device)->process_index();
    return nullptr;
  };
  api->PJRT_Device_IsAddressable =
      +[](PJRT_Device_IsAddressable_Args* args) -> PJRT_Error* {
    args->is_addressable =
        DeviceInstance::Unwrap(args->device)->is_addressable();
    return nullptr;
  };

  api->PJRT_Device_Attributes =
      +[](PJRT_Device_Attributes_Args* args) -> PJRT_Error* {
    // TODO: Implement something.
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_Device_Kind = +[](PJRT_Device_Kind_Args* args) -> PJRT_Error* {
    auto sv = DeviceInstance::Unwrap(args->device)->kind_string();
    args->device_kind = sv.data();
    args->device_kind_size = sv.size();
    return nullptr;
  };
  api->PJRT_Device_LocalHardwareId =
      +[](PJRT_Device_LocalHardwareId_Args* args) -> PJRT_Error* {
    args->local_hardware_id =
        DeviceInstance::Unwrap(args->device)->local_hardware_id();
    return nullptr;
  };
  api->PJRT_Device_DebugString =
      +[](PJRT_Device_DebugString_Args* args) -> PJRT_Error* {
    auto sv = DeviceInstance::Unwrap(args->device)->debug_string();
    args->debug_string = sv.data();
    args->debug_string_size = sv.size();
    return nullptr;
  };
  api->PJRT_Device_ToString =
      +[](PJRT_Device_ToString_Args* args) -> PJRT_Error* {
    auto sv = DeviceInstance::Unwrap(args->device)->user_string();
    args->to_string = sv.data();
    args->to_string_size = sv.size();
    return nullptr;
  };
}

iree_status_t DeviceInstance::CreateFence(iree_hal_fence_t** out_fence) {
  return iree_hal_fence_create(/*capacity=*/2, client_.host_allocator(),
                               out_fence);
}

iree_status_t DeviceInstance::OpenDevice() {
  if (device_) return iree_ok_status();
  IREE_RETURN_IF_ERROR(iree_hal_driver_create_device_by_id(
      driver_, /*device_id=*/info_->device_id,
      /*param_count=*/0, /*params=*/nullptr, client_.host_allocator(),
      &device_));
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device(), 0ull, &main_timeline_));
  IREE_RETURN_IF_ERROR(
      iree_hal_semaphore_create(device(), 0ull, &transfer_timeline_));

  // Initialize debug strings.
  user_string_ = std::string(info_->path.data, info_->path.size);
  debug_string_ = std::string(info_->name.data, info_->name.size);
  kind_string_ = std::string(info_->name.data, info_->name.size);

  return iree_ok_status();
}

iree_status_t DeviceInstance::HostBufferToDevice(
    const void* data, PJRT_Buffer_Type type, const int64_t* dims,
    size_t num_dims, const int64_t* byte_strides, size_t num_byte_strides,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  IREE_RETURN_IF_ERROR(OpenDevice());

  // Map element type.
  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(MapBufferTypeToElementType(type, &element_type));
  // TODO: Do something sensible with sub-byte aligned types.
  if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
      IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  }
  iree_device_size_t element_type_byte_size =
      iree_hal_element_dense_byte_count(element_type);

  // Handle strided layouts.
  bool dense_row_major_layout = true;
  if (byte_strides && num_dims > 0) {
    int64_t stride = element_type_byte_size;
    for (int64_t i = num_dims - 1; i >= 0; --i) {
      if (byte_strides[i] != stride) {
        dense_row_major_layout = false;
        break;
      }
      stride *= dims[i];
    }
  }
  if (!dense_row_major_layout) {
    // TODO: Compile a transpose program and invoke that to load the
    // array.
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "only dense, row-major layouts currently supported");
  }

  // Compute dense size.
  std::array<iree_hal_dim_t, 9> shape;
  if (num_dims > shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)shape.size(), (int)num_dims);
  }

  iree_device_size_t byte_length = element_type_byte_size;
  for (size_t i = 0; i < num_dims; ++i) {
    shape[i] = dims[i];
    byte_length *= dims[i];
  }

  iree::vm::ref<iree_hal_buffer_t> buffer;
  // There are multiple ways to implement zero-copy/staged transfers and each
  // implementation will have different performance cliffs associated with
  // directly operating on imported host buffers. In many actual
  // host/device situations, such unified memory is a productivity (not a
  // performance) feature and best avoided. As such, we always need to be
  // able to decide to do a staged transfer and implement that here. Using
  // an imported buffer on the device is left as an optimization for
  // implementations on which we believe it will be beneficial.
  bool require_snapshot_now = host_buffer_semantics ==
                              PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  bool caller_data_done = false;
  iree::vm::ref<iree_hal_buffer_t> host_staging_buffer;
  IREE_RETURN_IF_ERROR(AcquireHostStagingBuffer(
      iree_make_const_byte_span(data, byte_length), require_snapshot_now,
      &caller_data_done, &host_staging_buffer));
  if (!caller_data_done) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "Deferred snapshot of host data not yet implemented");
  }

  // Allocate on stream. We serialize across 3 timepoints:
  //   0. Last transfer complete
  //   1. Allocation
  //   2. This transfer complete
  // There are various ways to be smarter about this but without more
  // information from the caller, this is ok. If we wanted to favor smaller
  // allocation scopes, it may be desirable to join with the main execution
  // timeline, but that would obviously serialize more.
  uint64_t wait_transfer_start = last_transfer_timepoint_;
  uint64_t signal_alloca_complete = ++last_transfer_timepoint_;
  uint64_t signal_copy_complete = ++last_transfer_timepoint_;
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage =
      IREE_HAL_BUFFER_USAGE_DEFAULT | IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_alloca(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &wait_transfer_start},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, byte_length, &buffer));

  // Queue up the transfer command.
  iree::vm::ref<iree_hal_command_buffer_t> transfer_cb;
  iree_hal_transfer_command_t transfer_command;
  memset(&transfer_command, 0, sizeof(transfer_command));
  transfer_command.type = IREE_HAL_TRANSFER_COMMAND_TYPE_COPY;
  transfer_command.copy.source_buffer = host_staging_buffer.get(),
  transfer_command.copy.source_offset = 0;
  transfer_command.copy.target_buffer = buffer.get();
  transfer_command.copy.target_offset = 0;
  transfer_command.copy.length = byte_length;
  IREE_RETURN_IF_ERROR(iree_hal_create_transfer_command_buffer(
      device(), IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
      IREE_HAL_QUEUE_AFFINITY_ANY,
      /*transfer_count=*/1, &transfer_command, &transfer_cb));
  IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
      device(), IREE_HAL_QUEUE_AFFINITY_ANY,
      /*wait_semaphore_list=*/
      {1, &transfer_timeline_, &signal_alloca_complete},
      /*signal_semaphore_list=*/
      {1, &transfer_timeline_, &signal_copy_complete},
      /*command_buffer_count=*/1, &transfer_cb));

  // Wrap in a buffer view and return.
  iree::vm::ref<iree_hal_buffer_view_t> result_buffer_view;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      buffer.get(), num_dims, &shape[0], element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, client_.host_allocator(),
      &result_buffer_view));

  *out_buffer = new BufferInstance(*this, std::move(result_buffer_view));
  (*out_buffer)
      ->AdvanceReadyFence(transfer_timeline_.get(), signal_copy_complete);
  (*out_buffer)
      ->AdvanceDoneFence(transfer_timeline_.get(), signal_copy_complete);

  // We snapshotted the caller data when acquiring the host staging buffer,
  // so we won't be touching it again.
  *out_done_with_host_buffer_event =
      new EventInstance(EventInstance::Type::SIGNALLED);

  return iree_ok_status();
}

iree_status_t DeviceInstance::AcquireHostStagingBuffer(
    iree_const_byte_span_t initial_contents, bool snapshot_initial_contents_now,
    bool* initial_contents_snapshotted, iree_hal_buffer_t** out_buffer) {
  IREE_TRACE_SCOPE();
  // There are multiple ways to do this that have different cost/benefits.
  // Here we do the simplest thing and snapshot into a new host allocation.
  // This could be replaced with either some form of staging ring buffer
  // or importing from a raw pointer (on implementations where the cost of
  // unified addressing is zero).
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
  params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      device_allocator(), params, initial_contents.data_length,
      initial_contents, out_buffer));
  // We did a synchronous snapshot (memcpy).
  *initial_contents_snapshotted = true;
  return iree_ok_status();
}

iree_status_t DeviceInstance::GetHalDevice(iree_hal_device_t** out_device) {
  IREE_RETURN_IF_ERROR(OpenDevice());
  *out_device = device_.get();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)) {
  host_allocator_ = iree_allocator_system();
  cached_platform_version_ = "git";  // TODO: Plumb through version info.
}

ClientInstance::~ClientInstance() {
  for (auto* device : devices_) {
    delete device;
  }
  if (device_infos_) {
    iree_allocator_free(host_allocator_, device_infos_);
  }
  // Explicitly releasing vs using a ref so as to better control shut-down
  // ordering (bad shutdown ordering of the driver is a frequent cause of
  // bugs).
  iree_hal_driver_release(driver_);
}

void ClientInstance::BindApi(PJRT_Api* api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Client_Destroy");
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_name = client->cached_platform_name().data();
    args->platform_name_size = client->cached_platform_name().size();
    return nullptr;
  };
  api->PJRT_Client_ProcessIndex =
      +[](PJRT_Client_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index = 0;
    return nullptr;
  };
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    args->devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return MakeError(
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "because device id %d is invalid (%d devices known)",
                           (int)id_as_size, (int)devices.size()));
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_Compile =
      +[](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Client_Compile");
    // TODO: It is not great that we only get a client here vs a list of
    // devices to consider (or something). The issue is that systems often
    // have unrelated devices that will not actually be scheduled and those
    // will very naturally have different tuning flags. We therefore have to
    // guess... which is an accident waiting to happen.
    // Looks like what I need is buried in the compile options... need to
    // work on that.
    auto* client = ClientInstance::Unwrap(args->client);
    LoadedExecutableInstance* executable;
    auto* error = client->Compile(args->program, &executable);
    if (error) return error;
    args->executable = *executable;
    return nullptr;
  };
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args* args) -> PJRT_Error* {
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer =
      +[](PJRT_Client_BufferFromHostBuffer_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Client_BufferFromHostBuffer");
    auto status =
        DeviceInstance::Unwrap(args->device)
            ->HostBufferToDevice(
                args->data, args->type, args->dims, args->num_dims,
                args->byte_strides, args->num_byte_strides,
                args->host_buffer_semantics,
                reinterpret_cast<EventInstance**>(&args->done_with_host_buffer),
                reinterpret_cast<BufferInstance**>(&args->buffer));
    return MakeError(status);
  };
}

PJRT_Error* ClientInstance::Initialize() {
  // TODO: Remove calls to iree_status_fprint once JAX properly reports
  // initialization errors: https://github.com/google/jax/issues/13763
  auto status = CreateDriver(&driver_);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = InitializeVM();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = PopulateDevices();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  // More initialization.
  return nullptr;
}

iree_status_t ClientInstance::InitializeVM() {
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(host_allocator_, &vm_instance_));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(vm_instance_.get()));
  return iree_ok_status();
}

iree_status_t ClientInstance::PopulateDevices() {
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      driver_, host_allocator_, &device_info_count_, &device_infos_));
  devices_.resize(device_info_count_);
  for (iree_host_size_t i = 0; i < device_info_count_; ++i) {
    // Note that we assume one driver per client here.
    // But device is modeled with a driver in case if it ever becomes
    // more heterogenous.
    devices_[i] = new DeviceInstance(i, *this, driver_, &device_infos_[i]);
  }

  // For now, just make all devices addressable.
  addressable_devices_.reserve(devices_.size());
  for (auto* device : devices_) {
    addressable_devices_.push_back(device);
  }
  return iree_ok_status();
}

PJRT_Error* ClientInstance::Compile(PJRT_Program* program,
                                    LoadedExecutableInstance** out_executable) {
  std::unique_ptr<ArtifactDumper::Transaction> artifact_tx;
  if (platform().artifact_dumper().enabled()) {
    artifact_tx = platform().artifact_dumper().CreateTransaction();
  }

  iree_status_t status;
  std::string_view format(program->format, program->format_size);
  std::string_view code(program->code, program->code_size);
  if (artifact_tx) {
    artifact_tx->WriteArtifact(/*label=*/"program", /*extension=*/"mlirbc",
                               /*index=*/-1, code);
  }

  if (format != "mlir") {
    // See: https://github.com/google/jax/issues/13722
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "because IREE only supports MLIR input but got something else"));
  }

  std::unique_ptr<CompilerJob> job = platform().compiler().StartJob();
  if (artifact_tx) {
    job->EnableCrashDumps(artifact_tx.get());
  }
  auto MakeCompilerError = [&]() {
    std::string message = job->GetErrorMessage();
    return MakeError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, ": %s",
                                      message.c_str()));
  };

  // Set flags.
  // TODO: This should be done as part of session setup from a named pool.
  // TODO: The HAL backends and other flags should come from the assigned
  // devices.
  if (!job->SetFlag("--iree-input-type=mhlo")) {
    return MakeCompilerError();
  }
  if (!job->SetFlag("--iree-execution-model=async-external")) {
    return MakeCompilerError();
  }
  if (!SetDefaultCompilerFlags(job.get())) {
    return MakeCompilerError();
  }
  if (artifact_tx) {
    artifact_tx->WriteArtifact(
        /*label=*/"flags", /*extension=*/"txt", /*index=*/-1, job->GetFlags());
  }

  // Parse the source.
  if (!job->ParseSourceBuffer(code.data(), code.size())) {
    return MakeCompilerError();
  }

  // Perform main compilation.
  std::unique_ptr<CompilerOutput> output = job->CompileStandardPipeline();
  if (!output) {
    return MakeCompilerError();
  }
  if (artifact_tx) {
    artifact_tx->WriteArtifact(
        /*label=*/"program", /*extension=*/"vmfb", /*index=*/-1,
        std::string_view(static_cast<const char*>(output->GetData()),
                         output->GetDataSize()));
  }

  auto executable = std::make_unique<LoadedExecutableInstance>(
      *this, new ExecutableImage(std::move(output)), addressable_devices_);
  status = executable->LoadAll();
  if (!iree_status_is_ok(status)) {
    return MakeError(status);
  }

  *out_executable = executable.release();

  if (artifact_tx) {
    artifact_tx->Cancel();
  }
  return nullptr;
}

iree_status_t ClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  // HAL module.
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      vm_instance(), hal_device, IREE_HAL_MODULE_FLAG_NONE, host_allocator(),
      &modules.back()));

  // Main module.
  modules.push_back(main_module);
  return iree_ok_status();
}

std::tuple<uint64_t, uint64_t> ClientInstance::AdvanceTimeline() {
  uint64_t current = execution_timeline_;
  uint64_t next = current + 1;
  execution_timeline_ = next;
  return std::make_tuple(current, next);
}

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

EventInstance::EventInstance(Type type) : type_(type) {
  switch (type) {
    case Type::SIGNALLED:
      is_ready_ = true;
      break;
    case Type::EXTERNAL:
      is_ready_ = false;
      break;
  }
}

EventInstance::~EventInstance() { iree_status_ignore(status_); }

void EventInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Event_Destroy");
    delete EventInstance::Unwrap(args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Event_IsReady");
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Event_Error");
    return (PJRT_Error*)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_Await = +[](PJRT_Event_Await_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Event_Await");
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Event_Await"));
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Event_OnReady");
    return MakeError(EventInstance::Unwrap(args->event)
                         ->OnReady(args->callback, args->user_arg));
  };
}

ErrorInstance* EventInstance::error() {
  std::lock_guard<std::mutex> guard(lock_);
  if (!iree_status_is_ok(status_)) return new ErrorInstance(status_);
  return nullptr;
}
bool EventInstance::is_ready() {
  std::lock_guard<std::mutex> guard(lock_);
  return is_ready_;
}

iree_status_t EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                     void* user_arg) {
  iree_status_t local_status;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (!is_ready_) {
      pending_callbacks_.push_back({callback, user_arg});
      return iree_ok_status();
    }
    local_status = status_;
  }

  // Already signalled. Callback out of lock scope.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  callback(
      iree_status_is_ok(local_status)
          ? nullptr
          : (PJRT_Error*)new ErrorInstance(iree_status_clone(local_status)),
      user_arg);
  return iree_ok_status();
}

void EventInstance::ExternalSignalReady(iree_status_t status) {
  IREE_TRACE_SCOPE();
  assert(type_ == Type::EXTERNAL && "expected EXTERNAL Event type");
  iree_status_t local_status;
  std::vector<std::pair<PJRT_Event_OnReadyCallback, void*>> local_callbacks;
  {
    std::lock_guard<std::mutex> guard(lock_);
    if (is_ready_) {
      return;
    }
    local_callbacks.swap(pending_callbacks_);
    is_ready_ = true;
    status_ = status;
    local_status = status_;
  }

  // Trigger callbacks outside of the lock.
  // Note that the callback may destroy the event - so must only operate on
  // locals.
  for (auto& cb : local_callbacks) {
    IREE_TRACE_SCOPE0("PJRT_User_Callback_Invoke");
    cb.first(
        iree_status_is_ok(local_status)
            ? nullptr
            : (PJRT_Error*)new ErrorInstance(iree_status_clone(local_status)),
        cb.second);
  }
}

//===----------------------------------------------------------------------===//
// LoadedExecutableInstance
//===----------------------------------------------------------------------===//

void ExecutableImage::BindApi(PJRT_Api* api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Executable_Destroy");
    ExecutableImage::Unwrap(args->executable)->DecRef();
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0(PJRT_Executable_Name);
    const char* dummy_name = "iree_vmfb";
    args->executable_name = dummy_name;
    args->executable_name_size = strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args)
      -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Executable_SizeOfGeneratedCodeInBytes");
    args->size_in_bytes =
        ExecutableImage::Unwrap(args->executable)->binary->GetDataSize();
    return nullptr;
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_Executable_NumOutputs");
    auto* exec = ExecutableImage::Unwrap(args->executable);
    assert(exec->metadata_initialized);
    args->num_outputs = exec->result_count;
    return nullptr;
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_Serialize"));
  };
  api->PJRT_Executable_DeserializeAndLoad =
      +[](PJRT_Executable_DeserializeAndLoad_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_DeserializeAndLoad"));
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_Serialize"));
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_OptimizedProgram"));
  };
}

void LoadedExecutableInstance::BindApi(PJRT_Api* api) {
  api->PJRT_LoadedExecutable_Destroy =
      +[](PJRT_LoadedExecutable_Destroy_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_LoadedExecutable_Destroy");
    delete LoadedExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_LoadedExecutable_AddressableDevices =
      +[](PJRT_LoadedExecutable_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = LoadedExecutableInstance::Unwrap(args->executable)
                        ->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_LoadedExecutable_Delete =
      +[](PJRT_LoadedExecutable_Delete_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_LoadedExecutable_Delete"));
  };
  api->PJRT_LoadedExecutable_IsDeleted =
      +[](PJRT_LoadedExecutable_IsDeleted_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_LoadedExecutable_IsDeleted"));
  };
  api->PJRT_LoadedExecutable_Execute =
      +[](PJRT_LoadedExecutable_Execute_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_LoadedExecutable_Execute");
    return MakeError(
        LoadedExecutableInstance::Unwrap(args->executable)->BatchExecute(args));
  };
  api->PJRT_LoadedExecutable_GetCostAnalysis =
      +[](PJRT_LoadedExecutable_GetCostAnalysis_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_LoadedExecutable_GetCostAnalysis"));
  };
  api->PJRT_LoadedExecutable_GetExecutable =
      +[](PJRT_LoadedExecutable_GetExecutable_Args* args) -> PJRT_Error* {
    IREE_TRACE_SCOPE0("PJRT_LoadedExecutable_GetExecutable");
    auto* loaded_exe =
        LoadedExecutableInstance::Unwrap(args->loaded_executable);
    ExecutableImage* image = loaded_exe->image_;
    if (!image->metadata_initialized) {
      auto status = loaded_exe->GetArgResultCount(&image->arg_count,
                                                  &image->result_count);
      if (!iree_status_is_ok(status)) {
        return MakeError(status);
      }
      image->metadata_initialized = true;
    }

    image->AddRef();
    args->executable = *image;
    return nullptr;
  };
}

iree_status_t LoadedExecutableInstance::LoadAll() {
  IREE_TRACE_SCOPE();
  if (!resident_executables_.empty()) return iree_ok_status();

  std::vector<ResidentExecutable> new_list;
  for (DeviceInstance* device_instance : addressable_devices_) {
    iree_hal_device_t* hal_device;
    IREE_RETURN_IF_ERROR(device_instance->GetHalDevice(&hal_device));
    new_list.push_back({});
    ResidentExecutable& loaded = new_list.back();
    loaded.device_instance = device_instance;

    // Only de-reference through the image_ shared_ptr once to get the
    // binary CompilerOutput (mmap).
    auto* binary = image_->binary.get();
    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        client_.vm_instance(),
        iree_make_const_byte_span(binary->GetData(), binary->GetDataSize()),
        /*archive_allocator=*/iree_allocator_null(), client_.host_allocator(),
        &loaded.main_module));

    // Lookup main function.
    const char kNameMain[] = "main";
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        loaded.main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{kNameMain, sizeof(kNameMain) - 1},
        &loaded.main_function));

    // Record number of args/results.
    iree_vm_function_signature_t sig =
        iree_vm_function_signature(&loaded.main_function);
    IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
        &sig, &loaded.arg_count, &loaded.result_count));

    // Defer to the client to populate the stack of modules.
    std::vector<iree::vm::ref<iree_vm_module_t>> modules;
    IREE_RETURN_IF_ERROR(
        client_.PopulateVMModules(modules, hal_device, loaded.main_module));
    std::vector<iree_vm_module_t*> module_ptrs;
    module_ptrs.resize(modules.size());
    for (size_t i = 0; i < modules.size(); ++i) {
      module_ptrs[i] = modules[i].get();
    }

    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        client_.vm_instance(), IREE_VM_CONTEXT_FLAG_NONE, module_ptrs.size(),
        module_ptrs.data(), iree_allocator_system(), &loaded.vm_context));
  }

  new_list.swap(resident_executables_);
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::GetDefaultResidentExecutable(
    ResidentExecutable** out_loaded) {
  IREE_RETURN_IF_ERROR(LoadAll());
  if (resident_executables_.empty()) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no executables could be loaded");
  }
  *out_loaded = &resident_executables_.front();
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::GetArgResultCount(
    iree_host_size_t* out_arg_count, iree_host_size_t* out_result_count) {
  ResidentExecutable* loaded;
  IREE_RETURN_IF_ERROR(GetDefaultResidentExecutable(&loaded));
  *out_arg_count = loaded->arg_count;
  *out_result_count = loaded->result_count;
  return iree_ok_status();
}

iree_status_t LoadedExecutableInstance::BatchExecute(
    PJRT_LoadedExecutable_Execute_Args* args) {
  // Early exit for unsupported features and illegal input.
  if (args->execute_device) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "executing with a specific device not supported");
  }
  if (args->num_devices != addressable_devices_.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "incorrect number of devices to execute on (%d vs %d)",
        (int)args->num_devices, (int)addressable_devices_.size());
  }

  // Make sure loaded.
  IREE_RETURN_IF_ERROR(LoadAll());

  // Timeline setup. There are two timelines that we synchronize to:
  // the main execution timeline, which preserves as-called ordering to
  // execution, and the transfer timeline of each device.
  auto [wait_timepoint, signal_timepoint] = client_.AdvanceTimeline();

  // Initialize invocations.
  auto allocator = client_.host_allocator();
  auto& resident_executables_ecs = resident_executables_;
  struct Invocation {
    ResidentExecutable* res_exe;
    iree::vm::ref<iree_vm_list_t> inputs;
    iree::vm::ref<iree_vm_list_t> outputs;
    iree::vm::ref<iree_hal_fence_t> wait_fence;
    iree::vm::ref<iree_hal_fence_t> signal_fence;
  };
  std::vector<Invocation> invs;
  invs.resize(args->num_devices);
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    inv.res_exe = &resident_executables_ecs[dev_index];

    // Wait fence initial value.
    // We allocate it to be able to hold two semaphores (main timeline and
    // transfer timeline) and initialize it with the global invocation order
    // of the main timeline. As we process inputs, we will also insert their
    // transfer ready semaphore value so that execution can only begin once
    // all dependencies are ready. This at most represents two unique
    // semaphores.
    IREE_RETURN_IF_ERROR(
        inv.res_exe->device_instance->CreateFence(&inv.wait_fence));
    IREE_RETURN_IF_ERROR(iree_hal_fence_insert(
        inv.wait_fence.get(), inv.res_exe->device_instance->main_timeline(),
        wait_timepoint));

    // Signal fence. This signals the next tick on the main execution
    // timeline.
    IREE_RETURN_IF_ERROR(iree_hal_fence_create_at(
        inv.res_exe->device_instance->main_timeline(), signal_timepoint,
        client_.host_allocator(), &inv.signal_fence));

    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, args->num_args, allocator, &inv.inputs));
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, inv.res_exe->result_count, allocator,
        &inv.outputs));

    // Populate inputs.
    for (size_t i = 0; i < args->num_args; ++i) {
      auto* buffer = BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
      iree_vm_ref_t bv_ref =
          iree_hal_buffer_view_retain_ref(buffer->buffer_view());
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(inv.inputs.get(), &bv_ref));

      // Extend the execute wait to include the input's ready signal.
      IREE_RETURN_IF_ERROR(
          iree_hal_fence_extend(inv.wait_fence.get(), buffer->ready_fence()));

      // And extend the buffer's done fence to close over this execution.
      buffer->AdvanceDoneFence(inv.res_exe->device_instance->main_timeline(),
                               signal_timepoint);
    }

    // Add (wait, signal) fences as required by the async-external execution
    // model.
    iree_vm_list_push_ref_retain(inv.inputs.get(), inv.wait_fence);
    iree_vm_list_push_ref_retain(inv.inputs.get(), inv.signal_fence);
  }

  // Issue invocations.
  // TODO: Switch to using the async API. I've tried to structure this
  // so that we can move to that. Obviously important before we have more
  // than one device.
  iree_status_t status = iree_ok_status();
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    status = iree_vm_invoke(
        inv.res_exe->vm_context.get(), inv.res_exe->main_function,
        IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/nullptr, inv.inputs.get(), inv.outputs.get(), allocator);
    if (!iree_status_is_ok(status)) break;
  }

  // Process results.
  // Early exit before committing things to the client if anything failed.
  if (!iree_status_is_ok(status)) return status;
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    for (size_t i = 0; i < inv.res_exe->result_count; ++i) {
      iree::vm::ref<iree_hal_buffer_view_t> ret_buffer_view =
          retain_ref((iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
              inv.outputs.get(), i, iree_hal_buffer_view_get_descriptor()));
      // This should not be possible so just hard-assert.
      IREE_ASSERT_ARGUMENT(ret_buffer_view);
      auto result_buffer = std::make_unique<BufferInstance>(
          *inv.res_exe->device_instance, std::move(ret_buffer_view));
      IREE_RETURN_IF_ERROR(result_buffer->AdvanceReadyFence(
          inv.res_exe->device_instance->main_timeline(), signal_timepoint));
      IREE_RETURN_IF_ERROR(result_buffer->AdvanceDoneFence(
          inv.res_exe->device_instance->main_timeline(), signal_timepoint));
      args->output_lists[dev_index][i] = *(result_buffer.release());
    }

    if (args->device_complete_events) {
      // TODO: Plumb through signal fence. This doesn't seem to be used in
      // the simple cases I've seen so far.
      args->device_complete_events[dev_index] =
          *(new EventInstance(EventInstance::Type::SIGNALLED));
    }
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->priv = nullptr;

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceInstance::BindApi(api);
  ErrorInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableImage::BindApi(api);
  LoadedExecutableInstance::BindApi(api);
}

}  // namespace iree::pjrt
