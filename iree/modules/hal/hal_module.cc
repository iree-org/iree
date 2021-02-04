// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/modules/hal/hal_module.h"

#include <inttypes.h>

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/types/span.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"
#include "iree/vm/native_module_cc.h"

//===----------------------------------------------------------------------===//
// Type registration
//===----------------------------------------------------------------------===//

static iree_vm_ref_type_descriptor_t iree_hal_allocator_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_buffer_view_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_command_buffer_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_descriptor_set_layout_descriptor =
    {0};
static iree_vm_ref_type_descriptor_t iree_hal_device_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_event_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_cache_descriptor = {0};
static iree_vm_ref_type_descriptor_t iree_hal_executable_layout_descriptor = {
    0};
static iree_vm_ref_type_descriptor_t iree_hal_semaphore_descriptor = {0};

#define IREE_VM_REGISTER_HAL_C_TYPE(type, name, destroy_fn, descriptor)   \
  descriptor.type_name = iree_make_cstring_view(name);                    \
  descriptor.offsetof_counter = offsetof(iree_hal_resource_t, ref_count); \
  descriptor.destroy = (iree_vm_ref_destroy_t)destroy_fn;                 \
  IREE_RETURN_IF_ERROR(iree_vm_ref_register_type(&descriptor));

IREE_API_EXPORT iree_status_t IREE_API_CALL iree_hal_module_register_types() {
  static bool has_registered = false;
  if (has_registered) return iree_ok_status();

  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_allocator_t, "hal.allocator",
                              iree_hal_allocator_destroy,
                              iree_hal_allocator_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_t, "hal.buffer",
                              iree_hal_buffer_destroy,
                              iree_hal_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_buffer_view_t, "hal.buffer_view",
                              iree_hal_buffer_view_destroy,
                              iree_hal_buffer_view_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_command_buffer_t, "hal.command_buffer",
                              iree_hal_command_buffer_destroy,
                              iree_hal_command_buffer_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_t, "hal.descriptor_set",
                              iree_hal_descriptor_set_destroy,
                              iree_hal_descriptor_set_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_descriptor_set_layout_t,
                              "hal.descriptor_set_layout",
                              iree_hal_descriptor_set_layout_destroy,
                              iree_hal_descriptor_set_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_device_t, "hal.device",
                              iree_hal_device_destroy,
                              iree_hal_device_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_event_t, "hal.event",
                              iree_hal_event_destroy,
                              iree_hal_event_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_t, "hal.executable",
                              iree_hal_executable_destroy,
                              iree_hal_executable_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(
      iree_hal_executable_cache_t, "hal.executable_cache",
      iree_hal_executable_cache_destroy, iree_hal_executable_cache_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_executable_layout_t,
                              "hal.executable_layout",
                              iree_hal_executable_layout_destroy,
                              iree_hal_executable_layout_descriptor);
  IREE_VM_REGISTER_HAL_C_TYPE(iree_hal_semaphore_t, "hal.semaphore",
                              iree_hal_semaphore_destroy,
                              iree_hal_semaphore_descriptor);

  has_registered = true;
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Type wrappers
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_allocator, iree_hal_allocator_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer, iree_hal_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_buffer_view, iree_hal_buffer_view_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_command_buffer,
                             iree_hal_command_buffer_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set,
                             iree_hal_descriptor_set_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_descriptor_set_layout,
                             iree_hal_descriptor_set_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_device, iree_hal_device_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_event, iree_hal_event_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable, iree_hal_executable_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_cache,
                             iree_hal_executable_cache_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_executable_layout,
                             iree_hal_executable_layout_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_hal_semaphore, iree_hal_semaphore_t);

namespace iree {
namespace hal {
namespace {

//===----------------------------------------------------------------------===//
// Module type definitions
//===----------------------------------------------------------------------===//

class HALModuleState final {
 public:
  HALModuleState(iree_allocator_t allocator, iree_hal_device_t* shared_device)
      : allocator_(allocator), shared_device_(shared_device) {
    iree_hal_device_retain(shared_device_);
  }

  ~HALModuleState() {
    for (auto& ref : deferred_releases_) {
      iree_vm_ref_release(&ref);
    }
    deferred_releases_.clear();
    iree_hal_executable_cache_release(executable_cache_);
    iree_hal_device_release(shared_device_);
  }

  Status Initialize() {
    IREE_TRACE_SCOPE0("HALModuleState::Initialize");

    IREE_RETURN_IF_ERROR(iree_hal_executable_cache_create(
        shared_device_, iree_string_view_empty(), &executable_cache_));

    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // Experimental APIs
  //===--------------------------------------------------------------------===//
  // NOTE: Ex* APIs are experimental and likely to be removed soon. Modules
  // using these APIs are not forward compatible.

  StatusOr<vm::ref<iree_hal_device_t>> ExSharedDevice() {
    return vm::retain_ref(shared_device_);
  }

  template <typename T>
  void ExDeferRelease(const vm::ref<T>& value) {
    deferred_releases_.push_back({0});
    iree_vm_ref_retain((iree_vm_ref_t*)&value, &deferred_releases_.back());
  }

  Status ExSubmitAndWait(
      const vm::ref<iree_hal_device_t>& device,
      const vm::ref<iree_hal_command_buffer_t>& command_buffer) {
    IREE_TRACE_SCOPE0("HALModuleState::ExSubmitAndWait");

    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device.get(), 0ull, &semaphore));

    iree_hal_submission_batch_t batch;
    memset(&batch, 0, sizeof(batch));
    batch.command_buffer_count = 1;
    iree_hal_command_buffer_t* command_buffer_ptrs[] = {command_buffer.get()};
    batch.command_buffers = command_buffer_ptrs;
    batch.signal_semaphores.count = 1;
    iree_hal_semaphore_t* semaphore_ptrs[] = {semaphore.get()};
    batch.signal_semaphores.semaphores = semaphore_ptrs;
    uint64_t signal_value = 1ull;
    batch.signal_semaphores.payload_values = &signal_value;
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_submit(
        device.get(), IREE_HAL_COMMAND_CATEGORY_ANY, 0, 1, &batch));

    IREE_RETURN_IF_ERROR(iree_hal_semaphore_wait_with_deadline(
        semaphore.get(), 1ull, IREE_TIME_INFINITE_FUTURE));

    {
      IREE_TRACE_SCOPE0("HALModuleState::DeferredReleases");
      for (auto& ref : deferred_releases_) {
        iree_vm_ref_release(&ref);
      }
      deferred_releases_.clear();
    }

    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_allocator_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorAllocate(
      const vm::ref<iree_hal_allocator_t>& allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      int32_t allocation_size) {
    IREE_TRACE_SCOPE0("HALModuleState::AllocatorAllocate");
    vm::ref<iree_hal_buffer_t> buffer;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        allocator.get(), memory_types, buffer_usage, allocation_size, &buffer));
    return std::move(buffer);
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> AllocatorWrapByteBuffer(
      const vm::ref<iree_hal_allocator_t>& allocator,
      iree_hal_memory_type_t memory_types, iree_hal_buffer_usage_t buffer_usage,
      const vm::ref<iree_vm_ro_byte_buffer_t>& source, int32_t offset,
      int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::AllocatorWrapByteBuffer");

    // TODO(benvanik): wrap when supported.

    buffer_usage |= IREE_HAL_BUFFER_USAGE_MAPPING;

    size_t buffer_length = source->data.data_length;
    if (length == -1) {
      length = static_cast<size_t>(buffer_length);
    }
    if (length < 0 || offset < 0 || offset > buffer_length ||
        offset + length > buffer_length) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "byte range out of bounds (requested %d-%d of available %" PRIu64 ")",
          offset, (offset + length - 1), buffer_length);
    }

    vm::ref<iree_hal_buffer_t> buffer;
    IREE_RETURN_IF_ERROR(
        iree_hal_allocator_allocate_buffer(allocator.get(), memory_types,
                                           buffer_usage, length, &buffer),
        "failed to allocate buffer");

    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_write_data(buffer.get(), 0, source->data.data + offset,
                                   length),
        "writing constant data");

    return buffer;
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_buffer_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_allocator_t>> BufferAllocator(
      const vm::ref<iree_hal_buffer_t>& buffer) {
    return vm::retain_ref(iree_hal_buffer_allocator(buffer.get()));
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> BufferSubspan(
      const vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
      int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferSubspan");
    vm::ref<iree_hal_buffer_t> target_buffer;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_subspan(source_buffer.get(), source_offset, length,
                                &target_buffer),
        "subspan of an existing buffer (source_offset=%u, length=%u)",
        source_offset, length);
    return target_buffer;
  }

  Status BufferFill(const vm::ref<iree_hal_buffer_t>& target_buffer,
                    int32_t target_offset, int32_t length, int32_t pattern) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferFill");
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_fill(target_buffer.get(), target_offset, length,
                             &pattern, sizeof(pattern)),
        "fill range failed (target_offset=%u, length=%u)", target_offset,
        length);
    return OkStatus();
  }

  Status BufferReadData(const vm::ref<iree_hal_buffer_t>& source_buffer,
                        int32_t source_offset,
                        const vm::ref<iree_vm_rw_byte_buffer_t>& target_buffer,
                        int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferReadData");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "BufferReadData");
  }

  Status BufferWriteData(const vm::ref<iree_hal_buffer_t>& target_buffer,
                         int32_t target_offset,
                         const vm::ref<iree_vm_ro_byte_buffer_t>& source_buffer,
                         int32_t source_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferWriteData");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "BufferWriteData");
  }

  Status BufferCopyData(const vm::ref<iree_hal_buffer_t>& source_buffer,
                        int32_t source_offset,
                        const vm::ref<iree_hal_buffer_t>& target_buffer,
                        int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferCopyData");
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "BufferCopyData");
  }

  StatusOr<int32_t> BufferLoad(const vm::ref<iree_hal_buffer_t>& source_buffer,
                               int32_t source_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferLoad");

    uint32_t target_buffer = 0;
    if (length > sizeof(target_buffer)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "length %d exceeds max", length);
    }

    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_read_data(source_buffer.get(), source_offset,
                                  &target_buffer, length),
        "read failed");
    return target_buffer;
  }

  Status BufferStore(int32_t value,
                     const vm::ref<iree_hal_buffer_t>& target_buffer,
                     int32_t target_offset, int32_t length) {
    IREE_TRACE_SCOPE0("HALModuleState::BufferStore");

    if (target_offset + length >
        iree_hal_buffer_byte_length(target_buffer.get())) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "out of bounds store");
    } else if (length > sizeof(value)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "length %d exceeds max", length);
    }

    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_write_data(target_buffer.get(), target_offset, &value,
                                   length),
        "write failed");
    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_buffer_view_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewCreate(
      const vm::ref<iree_hal_buffer_t>& buffer, absl::Span<const int32_t> shape,
      iree_hal_element_type_t element_type) {
    vm::ref<iree_hal_buffer_view_t> buffer_view;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_create(buffer.get(), shape.data(), shape.size(),
                                    element_type, &buffer_view),
        "failed to create buffer view");
    return std::move(buffer_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_view_t>> BufferViewSubview(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> indices, absl::Span<const int32_t> lengths) {
    vm::ref<iree_hal_buffer_view_t> new_buffer_view;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_subview(
                             buffer_view.get(), indices.data(), indices.size(),
                             lengths.data(), lengths.size(), &new_buffer_view),
                         "failed to create subview");
    return std::move(new_buffer_view);
  }

  StatusOr<vm::ref<iree_hal_buffer_t>> BufferViewBuffer(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return vm::retain_ref(iree_hal_buffer_view_buffer(buffer_view.get()));
  }

  StatusOr<int32_t> BufferViewByteLength(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return iree_hal_buffer_view_byte_length(buffer_view.get());
  }

  StatusOr<int32_t> BufferViewComputeOffset(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> indices) {
    iree_device_size_t offset = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_offset(
        buffer_view.get(), indices.data(), indices.size(), &offset));
    return offset;
  }

  StatusOr<std::tuple<int32_t, int32_t>> BufferViewComputeRange(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view,
      absl::Span<const int32_t> start_indices,
      absl::Span<const int32_t> lengths) {
    iree_device_size_t start_offset = 0;
    iree_device_size_t subspan_length = 0;
    IREE_RETURN_IF_ERROR(iree_hal_buffer_view_compute_range(
        buffer_view.get(), start_indices.data(), start_indices.size(),
        lengths.data(), lengths.size(), &start_offset, &subspan_length));
    return std::make_tuple<int32_t, int32_t>(
        static_cast<int32_t>(start_offset),
        static_cast<int32_t>(subspan_length));
  }

  StatusOr<int32_t> BufferViewRank(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return static_cast<int32_t>(
        iree_hal_buffer_view_shape_rank(buffer_view.get()));
  }

  StatusOr<int32_t> BufferViewDim(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view, int32_t index) {
    return static_cast<int32_t>(
        iree_hal_buffer_view_shape_dim(buffer_view.get(), index));
  }

  template <size_t N>
  StatusOr<std::array<int32_t, N>> BufferViewDimsN(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    std::array<int32_t, N> value;
    iree_host_size_t rank = 0;
    IREE_RETURN_IF_ERROR(
        iree_hal_buffer_view_shape(buffer_view.get(), N, value.data(), &rank));
    return value;
  }

  StatusOr<std::array<int32_t, 1>> BufferViewDims1(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return BufferViewDimsN<1>(buffer_view);
  }

  StatusOr<std::array<int32_t, 2>> BufferViewDims2(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return BufferViewDimsN<2>(buffer_view);
  }

  StatusOr<std::array<int32_t, 3>> BufferViewDims3(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return BufferViewDimsN<3>(buffer_view);
  }

  StatusOr<std::array<int32_t, 4>> BufferViewDims4(
      const vm::ref<iree_hal_buffer_view_t>& buffer_view) {
    return BufferViewDimsN<4>(buffer_view);
  }

  Status BufferViewTrace(
      absl::Span<const vm::ref<iree_hal_buffer_view_t>> buffer_views,
      absl::string_view trace_info) {
    fprintf(stderr, "=== %s ===\n", std::string(trace_info).c_str());
    for (auto& view : buffer_views) {
      std::string result_str(4096, '\0');
      iree_status_t status;
      auto max_element_count = std::numeric_limits<iree_host_size_t>::max();
      do {
        iree_host_size_t actual_length = 0;
        status = iree_hal_buffer_view_format(view.get(), max_element_count,
                                             result_str.size() + 1,
                                             &result_str[0], &actual_length);
        result_str.resize(actual_length);
      } while (iree_status_is_out_of_range(status));
      IREE_RETURN_IF_ERROR(status);
      fprintf(stderr, "%s\n", result_str.c_str());
    }
    fprintf(stderr, "\n");
    return OkStatus();
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_command_buffer_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_command_buffer_t>> CommandBufferCreate(
      const vm::ref<iree_hal_device_t>& device,
      iree_hal_command_buffer_mode_t modes,
      iree_hal_command_category_t command_categories) {
    vm::ref<iree_hal_command_buffer_t> command_buffer;
    IREE_RETURN_IF_ERROR(
        iree_hal_command_buffer_create(device.get(), modes, command_categories,
                                       &command_buffer),
        "failed to create command buffer");
    return command_buffer;
  }

  Status CommandBufferBegin(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer) {
    return iree_hal_command_buffer_begin(command_buffer.get());
  }

  Status CommandBufferEnd(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer) {
    return iree_hal_command_buffer_end(command_buffer.get());
  }

  Status CommandBufferExecutionBarrier(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      iree_hal_execution_stage_t source_stage_mask,
      iree_hal_execution_stage_t target_stage_mask,
      absl::Span<const int32_t> memory_barriers,
      absl::Span<const int32_t> buffer_barriers) {
    // TODO(benvanik): decode barriers.
    iree_hal_memory_barrier_t global_barrier;
    global_barrier.source_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE;
    global_barrier.target_scope = IREE_HAL_ACCESS_SCOPE_DISPATCH_READ;
    return iree_hal_command_buffer_execution_barrier(
        command_buffer.get(), source_stage_mask, target_stage_mask, 1,
        &global_barrier, 0, nullptr);
  }

  Status CommandBufferFillBuffer(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
      int32_t length, uint32_t pattern) {
    ExDeferRelease(target_buffer);
    return iree_hal_command_buffer_fill_buffer(
        command_buffer.get(), target_buffer.get(), target_offset, length,
        &pattern, sizeof(pattern));
  }

  Status CommandBufferCopyBuffer(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_buffer_t>& source_buffer, int32_t source_offset,
      const vm::ref<iree_hal_buffer_t>& target_buffer, int32_t target_offset,
      int32_t length) {
    ExDeferRelease(source_buffer);
    ExDeferRelease(target_buffer);
    return iree_hal_command_buffer_copy_buffer(
        command_buffer.get(), source_buffer.get(), source_offset,
        target_buffer.get(), target_offset, length);
  }

  Status CommandBufferPushConstants(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_executable_layout_t>& executable_layout,
      uint32_t offset, absl::Span<const uint32_t> values) {
    ExDeferRelease(executable_layout);
    return iree_hal_command_buffer_push_constants(
        command_buffer.get(), executable_layout.get(),
        offset * sizeof(uint32_t), values.data(),
        values.size() * sizeof(uint32_t));
  }

  Status CommandBufferPushDescriptorSet(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_executable_layout_t>& executable_layout,
      uint32_t set, absl::Span<const uint32_t> binding_ordinals,
      absl::Span<const vm::ref<iree_hal_buffer_t>> binding_buffers,
      absl::Span<const int32_t> binding_offsets,
      absl::Span<const int32_t> binding_lengths) {
    ExDeferRelease(executable_layout);
    absl::InlinedVector<iree_hal_descriptor_set_binding_t, 16> binding_structs(
        binding_ordinals.size());
    for (int i = 0; i < binding_ordinals.size(); ++i) {
      binding_structs[i] = {
          binding_ordinals[i], binding_buffers[i].get(),
          static_cast<iree_device_size_t>(binding_offsets[i]),
          static_cast<iree_device_size_t>(binding_lengths[i])};
      ExDeferRelease(binding_buffers[i]);
    }
    return iree_hal_command_buffer_push_descriptor_set(
        command_buffer.get(), executable_layout.get(), set,
        binding_structs.size(), binding_structs.data());
  }

  Status CommandBufferBindDescriptorSet(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_executable_layout_t>& executable_layout,
      uint32_t set, const vm::ref<iree_hal_descriptor_set_t>& descriptor_set,
      absl::Span<const int32_t> dynamic_offsets) {
    ExDeferRelease(executable_layout);
    ExDeferRelease(descriptor_set);
    absl::InlinedVector<iree_device_size_t, 4> dynamic_offset_values(
        dynamic_offsets.size());
    for (int i = 0; i < dynamic_offsets.size(); ++i) {
      dynamic_offset_values[i] =
          static_cast<iree_device_size_t>(dynamic_offsets[i]);
    }
    return iree_hal_command_buffer_bind_descriptor_set(
        command_buffer.get(), executable_layout.get(), set,
        descriptor_set.get(), dynamic_offset_values.size(),
        dynamic_offset_values.data());
  }

  Status CommandBufferDispatch(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
      uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
    ExDeferRelease(executable);
    return iree_hal_command_buffer_dispatch(
        command_buffer.get(), executable.get(), entry_point, workgroup_x,
        workgroup_y, workgroup_z);
  }

  Status CommandBufferDispatchIndirect(
      const vm::ref<iree_hal_command_buffer_t>& command_buffer,
      const vm::ref<iree_hal_executable_t>& executable, int32_t entry_point,
      const vm::ref<iree_hal_buffer_t>& workgroups_buffer,
      int32_t workgroups_offset) {
    ExDeferRelease(executable);
    ExDeferRelease(workgroups_buffer);
    return iree_hal_command_buffer_dispatch_indirect(
        command_buffer.get(), executable.get(), entry_point,
        workgroups_buffer.get(), workgroups_offset);
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_descriptor_set_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_descriptor_set_t>> DescriptorSetCreate(
      const vm::ref<iree_hal_device_t>& device,
      const vm::ref<iree_hal_descriptor_set_layout_t>& set_layout,
      absl::Span<const uint32_t> binding_ordinals,
      absl::Span<const vm::ref<iree_hal_buffer_t>> binding_buffers,
      absl::Span<const uint32_t> binding_offsets,
      absl::Span<const uint32_t> binding_lengths) {
    absl::InlinedVector<iree_hal_descriptor_set_binding_t, 4> binding_structs(
        binding_ordinals.size());
    for (int i = 0; i < binding_ordinals.size(); ++i) {
      binding_structs[i] = {
          binding_ordinals[i],                                   // binding
          binding_buffers[i].get(),                              // buffer
          static_cast<iree_device_size_t>(binding_offsets[i]),   // offset
          static_cast<iree_device_size_t>(binding_lengths[i])};  // length
    }
    vm::ref<iree_hal_descriptor_set_t> descriptor_set;
    IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_create(
        device.get(), set_layout.get(), binding_structs.size(),
        binding_structs.data(), &descriptor_set));
    return std::move(descriptor_set);
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_descriptor_set_layout_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_descriptor_set_layout_t>> DescriptorSetLayoutCreate(
      const vm::ref<iree_hal_device_t>& device,
      iree_hal_descriptor_set_layout_usage_type_t usage_type,
      absl::Span<const std::tuple<uint32_t, iree_hal_descriptor_type_t,
                                  iree_hal_memory_access_t>>
          bindings) {
    // TODO(benvanik): custom marshaling for the structs.
    absl::InlinedVector<iree_hal_descriptor_set_layout_binding_t, 4>
        binding_structs(bindings.size());
    for (int i = 0; i < bindings.size(); ++i) {
      binding_structs[i] = {std::get<0>(bindings[i]), std::get<1>(bindings[i]),
                            std::get<2>(bindings[i])};
    }
    vm::ref<iree_hal_descriptor_set_layout_t> descriptor_set_layout;
    IREE_RETURN_IF_ERROR(iree_hal_descriptor_set_layout_create(
        device.get(), usage_type, binding_structs.size(),
        binding_structs.data(), &descriptor_set_layout));
    return std::move(descriptor_set_layout);
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_device_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_allocator_t>> DeviceAllocator(
      const vm::ref<iree_hal_device_t>& device) {
    return vm::retain_ref(iree_hal_device_allocator(device.get()));
  }

  StatusOr<int32_t> DeviceMatchID(const vm::ref<iree_hal_device_t>& device,
                                  absl::string_view pattern) {
    iree_string_view_t device_id = iree_hal_device_id(device.get());
    return iree_string_view_match_pattern(
               device_id, iree_string_view_t{pattern.data(), pattern.size()})
               ? 1
               : 0;
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_event_t
  //===--------------------------------------------------------------------===//

  //===--------------------------------------------------------------------===//
  // iree_hal_executable_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_executable_t>> ExecutableCreate(
      const vm::ref<iree_hal_device_t>& device,
      iree_hal_executable_format_t executable_format,
      const vm::ref<iree_vm_ro_byte_buffer_t>& executable_data,
      absl::Span<const vm::ref<iree_hal_executable_layout_t>>
          executable_layouts) {
    iree_hal_executable_spec_t spec;
    iree_hal_executable_spec_initialize(&spec);

    spec.executable_format = executable_format;
    spec.executable_data = executable_data->data;

    spec.executable_layout_count = executable_layouts.size();
    iree_hal_executable_layout_t** executable_layouts_ptr =
        (iree_hal_executable_layout_t**)iree_alloca(
            sizeof(executable_layouts_ptr[0]) * executable_layouts.size());
    for (size_t i = 0; i < executable_layouts.size(); ++i) {
      executable_layouts_ptr[i] = executable_layouts[i].get();
    }
    spec.executable_layouts = executable_layouts_ptr;

    vm::ref<iree_hal_executable_t> executable;
    IREE_RETURN_IF_ERROR(iree_hal_executable_cache_prepare_executable(
        executable_cache_, &spec, &executable));
    return std::move(executable);
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_executable_layout_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_executable_layout_t>> ExecutableLayoutCreate(
      const vm::ref<iree_hal_device_t>& device,
      absl::Span<const vm::ref<iree_hal_descriptor_set_layout_t>> set_layouts,
      int32_t push_constants) {
    iree_hal_descriptor_set_layout_t** set_layouts_ptr =
        (iree_hal_descriptor_set_layout_t**)iree_alloca(
            sizeof(set_layouts_ptr[0]) * set_layouts.size());
    for (size_t i = 0; i < set_layouts.size(); ++i) {
      set_layouts_ptr[i] = set_layouts[i].get();
    }

    vm::ref<iree_hal_executable_layout_t> executable_layout;
    IREE_RETURN_IF_ERROR(iree_hal_executable_layout_create(
        device.get(), set_layouts.size(), set_layouts_ptr, push_constants,
        &executable_layout));
    return std::move(executable_layout);
  }

  //===--------------------------------------------------------------------===//
  // iree_hal_semaphore_t
  //===--------------------------------------------------------------------===//

  StatusOr<vm::ref<iree_hal_semaphore_t>> SemaphoreCreate(
      const vm::ref<iree_hal_device_t>& device, uint32_t initial_value) {
    vm::ref<iree_hal_semaphore_t> semaphore;
    IREE_RETURN_IF_ERROR(
        iree_hal_semaphore_create(device.get(), initial_value, &semaphore));
    return std::move(semaphore);
  }

  StatusOr<std::tuple<int32_t, uint32_t>> SemaphoreQuery(
      const vm::ref<iree_hal_semaphore_t>& semaphore) {
    uint64_t value = 0;
    iree_status_t query_status =
        iree_hal_semaphore_query(semaphore.get(), &value);
    return std::make_tuple<int32_t, uint32_t>(iree_status_code(query_status),
                                              static_cast<uint32_t>(value));
  }

  Status SemaphoreSignal(const vm::ref<iree_hal_semaphore_t>& semaphore,
                         uint32_t new_value) {
    return iree_hal_semaphore_signal(semaphore.get(), new_value);
  }

  Status SemaphoreFail(const vm::ref<iree_hal_semaphore_t>& semaphore,
                       int32_t status_code) {
    iree_status_t status = iree_make_status(
        static_cast<iree_status_code_t>(status_code & IREE_STATUS_CODE_MASK));
    iree_hal_semaphore_fail(semaphore.get(), status);
    return OkStatus();
  }

  StatusOr<int32_t> SemaphoreAwait(
      const vm::ref<iree_hal_semaphore_t>& semaphore, uint32_t new_value) {
    // TODO(benvanik): coroutine magic.
    iree_status_t status = iree_hal_semaphore_wait_with_deadline(
        semaphore.get(), new_value, IREE_TIME_INFINITE_FUTURE);
    if (iree_status_is_ok(status)) {
      return 0;
    } else if (iree_status_is_deadline_exceeded(status)) {
      // Propagate deadline exceeded back to the VM.
      return static_cast<int32_t>(iree_status_consume_code(status));
    }
    return Status(std::move(status));
  }

 private:
  iree_allocator_t allocator_;
  iree_hal_device_t* shared_device_ = NULL;
  iree_hal_executable_cache_t* executable_cache_ = NULL;

  std::vector<iree_vm_ref_t> deferred_releases_;
};

//===----------------------------------------------------------------------===//
// VM module interface implementation
//===----------------------------------------------------------------------===//

static const vm::NativeFunction<HALModuleState> kHALModuleFunctions[] = {
    vm::MakeNativeFunction("ex.shared_device", &HALModuleState::ExSharedDevice),
    vm::MakeNativeFunction("ex.submit_and_wait",
                           &HALModuleState::ExSubmitAndWait),

    vm::MakeNativeFunction("allocator.allocate",
                           &HALModuleState::AllocatorAllocate),
    vm::MakeNativeFunction("allocator.wrap.byte_buffer",
                           &HALModuleState::AllocatorWrapByteBuffer),

    vm::MakeNativeFunction("buffer.allocator",
                           &HALModuleState::BufferAllocator),
    vm::MakeNativeFunction("buffer.subspan", &HALModuleState::BufferSubspan),
    vm::MakeNativeFunction("buffer.fill", &HALModuleState::BufferFill),
    vm::MakeNativeFunction("buffer.read_data", &HALModuleState::BufferReadData),
    vm::MakeNativeFunction("buffer.write_data",
                           &HALModuleState::BufferWriteData),
    vm::MakeNativeFunction("buffer.copy_data", &HALModuleState::BufferCopyData),
    vm::MakeNativeFunction("buffer.load", &HALModuleState::BufferLoad),
    vm::MakeNativeFunction("buffer.store", &HALModuleState::BufferStore),

    vm::MakeNativeFunction("buffer_view.create",
                           &HALModuleState::BufferViewCreate),
    vm::MakeNativeFunction("buffer_view.subview",
                           &HALModuleState::BufferViewSubview),
    vm::MakeNativeFunction("buffer_view.buffer",
                           &HALModuleState::BufferViewBuffer),
    vm::MakeNativeFunction("buffer_view.byte_length",
                           &HALModuleState::BufferViewByteLength),
    vm::MakeNativeFunction("buffer_view.compute_offset",
                           &HALModuleState::BufferViewComputeOffset),
    vm::MakeNativeFunction("buffer_view.compute_range",
                           &HALModuleState::BufferViewComputeRange),
    vm::MakeNativeFunction("buffer_view.rank", &HALModuleState::BufferViewRank),
    vm::MakeNativeFunction("buffer_view.dim", &HALModuleState::BufferViewDim),
    vm::MakeNativeFunction("buffer_view.dims.1",
                           &HALModuleState::BufferViewDims1),
    vm::MakeNativeFunction("buffer_view.dims.2",
                           &HALModuleState::BufferViewDims2),
    vm::MakeNativeFunction("buffer_view.dims.3",
                           &HALModuleState::BufferViewDims3),
    vm::MakeNativeFunction("buffer_view.dims.4",
                           &HALModuleState::BufferViewDims4),
    vm::MakeNativeFunction("buffer_view.trace",
                           &HALModuleState::BufferViewTrace),

    vm::MakeNativeFunction("command_buffer.create",
                           &HALModuleState::CommandBufferCreate),
    vm::MakeNativeFunction("command_buffer.begin",
                           &HALModuleState::CommandBufferBegin),
    vm::MakeNativeFunction("command_buffer.end",
                           &HALModuleState::CommandBufferEnd),
    vm::MakeNativeFunction("command_buffer.execution_barrier",
                           &HALModuleState::CommandBufferExecutionBarrier),
    vm::MakeNativeFunction("command_buffer.fill_buffer",
                           &HALModuleState::CommandBufferFillBuffer),
    vm::MakeNativeFunction("command_buffer.copy_buffer",
                           &HALModuleState::CommandBufferCopyBuffer),
    vm::MakeNativeFunction("command_buffer.push_constants",
                           &HALModuleState::CommandBufferPushConstants),
    vm::MakeNativeFunction("command_buffer.push_descriptor_set",
                           &HALModuleState::CommandBufferPushDescriptorSet),
    vm::MakeNativeFunction("command_buffer.bind_descriptor_set",
                           &HALModuleState::CommandBufferBindDescriptorSet),
    vm::MakeNativeFunction("command_buffer.dispatch",
                           &HALModuleState::CommandBufferDispatch),
    vm::MakeNativeFunction("command_buffer.dispatch.indirect",
                           &HALModuleState::CommandBufferDispatchIndirect),

    vm::MakeNativeFunction("descriptor_set.create",
                           &HALModuleState::DescriptorSetCreate),
    vm::MakeNativeFunction("descriptor_set_layout.create",
                           &HALModuleState::DescriptorSetLayoutCreate),

    vm::MakeNativeFunction("device.allocator",
                           &HALModuleState::DeviceAllocator),
    vm::MakeNativeFunction("device.match.id", &HALModuleState::DeviceMatchID),

    vm::MakeNativeFunction("executable.create",
                           &HALModuleState::ExecutableCreate),

    vm::MakeNativeFunction("executable_layout.create",
                           &HALModuleState::ExecutableLayoutCreate),

    vm::MakeNativeFunction("semaphore.create",
                           &HALModuleState::SemaphoreCreate),
    vm::MakeNativeFunction("semaphore.query", &HALModuleState::SemaphoreQuery),
    vm::MakeNativeFunction("semaphore.signal",
                           &HALModuleState::SemaphoreSignal),
    vm::MakeNativeFunction("semaphore.fail", &HALModuleState::SemaphoreFail),
    vm::MakeNativeFunction("semaphore.await", &HALModuleState::SemaphoreAwait),
};

class HALModule final : public vm::NativeModule<HALModuleState> {
 public:
  HALModule(iree_allocator_t allocator, iree_hal_device_t* shared_device)
      : vm::NativeModule<HALModuleState>(
            "hal", allocator, absl::MakeConstSpan(kHALModuleFunctions)),
        shared_device_(shared_device) {
    iree_hal_device_retain(shared_device_);
  }

  ~HALModule() { iree_hal_device_release(shared_device_); }

  Status Initialize() {
    IREE_TRACE_SCOPE0("HALModule::Initialize");
    return OkStatus();
  }

  StatusOr<std::unique_ptr<HALModuleState>> CreateState(
      iree_allocator_t allocator) override {
    IREE_TRACE_SCOPE0("HALModule::CreateState");
    auto state = std::make_unique<HALModuleState>(allocator, shared_device_);
    IREE_RETURN_IF_ERROR(state->Initialize());
    return state;
  }

 private:
  iree_hal_device_t* shared_device_ = NULL;
};

IREE_API_EXPORT iree_status_t IREE_API_CALL
iree_hal_module_create(iree_hal_device_t* device, iree_allocator_t allocator,
                       iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(device);
  IREE_ASSERT_ARGUMENT(out_module);
  *out_module = nullptr;
  auto module = std::make_unique<HALModule>(allocator, device);
  IREE_RETURN_IF_ERROR(module->Initialize());
  *out_module = module.release()->interface();
  return iree_ok_status();
}

}  // namespace
}  // namespace hal
}  // namespace iree
